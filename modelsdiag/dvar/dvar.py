"""
DiffusionVAR model implementation.

Modified from VAR: https://github.com/FoundationVision/VAR

This module implements the DiffusionVAR model that combines autoregressive generation
with diffusion-based masking for improved image synthesis quality.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import dist

from models.helpers import gumbel_softmax_with_rng, filter_top_k_top_p_
from models.vqvae import VQVAE
from models.quant import VectorQuantizer2
from models.transformer import Backbone
from models.schedule import MaskingSchedule


class DiffusionVAR(nn.Module):
    """diffusion-based vector autoregressive model for image generation.
    
    This model combines autoregressive generation with diffusion processes for high-quality
    image synthesis. It supports both standard VAR and diffusion-based training/inference
    with progressive training across multiple patch scales.
    """
    def __init__(
        self, algo: str, 
        vae_local: VQVAE,
        num_classes=1000, 
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
        flash_if_available=True, fused_if_available=True, 
        diffusion_args=None,
        **kwargs,
    ):
        """
        initialize diffusionvar model.
        
        :param algo: algorithm type ('var' or 'diffusion-var')
        :param vae_local: vqvae model for image tokenization
        :param num_classes: number of class categories
        :param depth: number of transformer layers
        :param embed_dim: embedding dimension
        :param num_heads: number of attention heads
        :param mlp_ratio: mlp expansion ratio
        :param drop_rate: dropout rate
        :param attn_drop_rate: attention dropout rate
        :param drop_path_rate: stochastic depth rate
        :param norm_eps: layer normalization epsilon
        :param shared_aln: whether to use shared adaptive layer norm
        :param cond_drop_rate: classifier-free guidance dropout rate
        :param attn_l2_norm: whether to use l2 normalization in attention
        :param patch_nums: tuple of patch sizes for progressive training
        :param flash_if_available: whether to use flash attention if available
        :param fused_if_available: whether to use fused operations if available
        :param diffusion_args: dictionary of diffusion-specific arguments
        :param kwargs: additional keyword arguments
        """
        super().__init__()
        self.algo = algo
        
        # hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (depth, embed_dim, 
                                                     embed_dim, num_heads)
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(
            device=(dist.get_device() if dist.initialized() 
                   else next(vae_local.parameters()).device))
        
        # input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)

        # class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.init_std = init_std
        self.num_classes = num_classes
        self.uniform_prob = torch.full(
            (1, num_classes), 
            fill_value=1.0 / num_classes, 
            dtype=torch.float32, 
            device=(dist.get_device() if dist.initialized() 
                   else torch.device('cpu')))
        
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        nn.init.trunc_normal_(self.word_embed.weight.data, mean=0, std=init_std)
        
        # absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # set attention values
        d: torch.Tensor = torch.cat([
            torch.full((pn*pn,), i) 
            for i, pn in enumerate(self.patch_nums)
        ]).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        
        # algorithm-specific components
        self.noise_schedule = diffusion_args.noise_schedule
        self.cont_time = diffusion_args.cont_time
        self.t_emb_dim = diffusion_args.t_emb_dim
        self.pstep = diffusion_args.pstep
        
        self.noise = MaskingSchedule(schedule_fn_type=self.noise_schedule)
        self.mask_index = self.V 

        # 5. algorithm-specific components
        attn_bias_for_masking = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        

        # this allows us to mask VAE embeddings without extending the VQVAE vocabulary
        self.mask_embedding = nn.Parameter(torch.empty(1, 1, self.Cvae))
        nn.init.trunc_normal_(self.mask_embedding.data, mean=0, std=init_std)
        self.cond_embed = CondEmbedder(self.C, self.t_emb_dim)

        # inference
        if self.pstep == 'log':
            self.gen_steps = [
                (math.log(self.patch_nums[i]**2 + 1)) / 
                math.log(self.patch_nums[-1]**2 + 1) 
                for i in range(len(self.patch_nums))]
        elif self.pstep == 'lin':
            self.gen_steps = [
                self.patch_nums[i] / self.patch_nums[-1] 
                for i in range(len(self.patch_nums))]
        else:
            self.gen_steps = [1] * len(self.patch_nums)
        
        # create backbone with extended vocabulary
        self.backbone = Backbone(
            vocab_size=self.V,  # include mask token
            embed_dim=embed_dim,
            num_heads=num_heads, 
            depth=depth,
            mlp_ratio=mlp_ratio, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
            zero_init=True,
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor,
                gt_BL: torch.Tensor, p: torch.Tensor, 
                t: Optional[torch.Tensor]) -> torch.Tensor:
        """
        forward pass for diffusion-based training.
        
        this method implements the diffusion training forward pass with masking and
        timestep conditioning. it applies noise to ground truth tokens and predicts
        the original tokens given the noisy input.
        
        :param label_B: class labels (B,)
        :param x_BLCv_wo_first_l: input embeddings without first level (B, L, Cvae)
        :param gt_BL: ground truth token indices (B, L)
        :param p: probability of masking (B, L)
        :param t: diffusion timesteps (B,) - sampled if None
        :return: logits for token prediction (B, L, V+1) including mask token
        """        
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        
        with torch.amp.autocast('cuda', enabled=False):
            label_B = torch.where(
                torch.rand(B, device=self.device) < self.cond_drop_rate, 
                self.num_classes, label_B)
            cond_BD = self.class_emb(label_B)  # for sos
            tcond_BD = self.cond_embed(t, cond_BD)
            sos = (cond_BD.unsqueeze(1).expand(B, self.first_l, -1) + 
                  self.pos_start.expand(B, self.first_l, -1))
            
            noise_BLCvae = self.q_xt(gt_BL[:, :ed], sos, x_BLCv_wo_first_l.float(), p)
            noise_embeddings = self.word_embed(noise_BLCvae.float())

            lvl_embed_seq = self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
            pos_embed_seq = self.pos_1LC[:, :ed]

            if self.prog_si == 0:
                x_BLC = sos
            else:
                input_embeddings = self.word_embed(x_BLCv_wo_first_l.float())
                x_BLC = torch.cat((sos, noise_embeddings), dim=1)
            x_BLC += lvl_embed_seq + pos_embed_seq

        # hack to get main dtype for mixed precision
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)

        # attn_bias = self.attn_bias_slice(ed)
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed].to(dtype=main_type)

        # forward
        x_BLC = self.backbone(x_BLC=x_BLC, cond_BD=tcond_BD, attn_bias=attn_bias)
        logits_BlV1 = self.backbone.get_logits(x_BLC.float(), tcond_BD)
        
        # apply parameterization hack for first scale
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                logits_BlV1[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                logits_BlV1[0, 0, 0] += s
        return logits_BlV1  # logits BL(V+1) including mask token

    @torch.no_grad()
    def infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, steps: int = 50, temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        generate images using diffusion-based inference with classifier-free guidance.
        
        this method performs diffusion-based image generation by iteratively denoising
        masked tokens across multiple patch scales. it supports cfg for improved quality
        and various sampling strategies.
        
        :param B: batch size for generation
        :param label_B: class labels for conditional generation; if None, randomly sampled
        :param g_seed: random seed for reproducible generation
        :param cfg: classifier-free guidance ratio (higher = more class-conditional)
        :param top_k: top-k sampling parameter (0 = disabled)
        :param top_p: top-p (nucleus) sampling parameter (0.0 = disabled)
        :param more_smooth: whether to use gumbel softmax smoothing (for visualization only)
        :param steps: number of diffusion steps for inference
        :return: generated images (B, 3, H, W) in [0, 1] range
        """
        if g_seed is None: 
            rng = None
        else: 
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(
                self.uniform_prob, num_samples=B, replacement=True, 
                generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,), 
                fill_value=(self.num_classes if label_B < 0 else label_B), 
                device=self.device)
        
        sos = cond_BD = self.class_emb(torch.cat((
            label_B, torch.full_like(label_B, fill_value=self.num_classes)), 
            dim=0)) 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (sos.unsqueeze(1).expand(2*B, self.first_l, -1) + 
                         self.pos_start.expand(2*B, self.first_l, -1) + 
                         lvl_pos[:, :self.first_l])

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            patch_length = pn * pn
            bg = cur_L
            ed = cur_L + patch_length
            cur_L += pn*pn

            # prepare current level for diffusion
            current_tokens = torch.full(
                (B, patch_length), self.mask_index, 
                device=self.device, dtype=torch.long)
            x = next_token_map  # 2B, L, C

            # variable number of steps for different patches
            scale_steps = int(max(1, steps*self.gen_steps[si]))
            if self.cont_time: num_scale_steps = scale_steps + 1
            else: num_scale_steps = scale_steps
            for step in range(num_scale_steps):  # ensure s is not negative
                t = 1.0 - step / scale_steps 
                s = t - 1.0 / scale_steps  # previous step 
                t_batch = torch.full((B,), t, device=self.device) 
                s_batch = torch.full((B,), s, device=self.device)
                tcond_BD = self.cond_embed(t_batch.repeat(2), cond_BD)

                x_out = self.backbone(x_BLC=x, cond_BD=tcond_BD)
                logits_2BlV1 = self.backbone.get_logits(x_out, tcond_BD)
                logits_2BlV1 = logits_2BlV1[:, -patch_length:, :]
                # apply CFG
                ratio = si / self.num_stages_minus_1
                y = cfg * ratio
                logits_BlV1 = ((1+y) * logits_2BlV1[:B] - 
                              y * logits_2BlV1[B:])

                # apply temperature scaling
                if temperature != 1.0:
                    logits_BlV1 = logits_BlV1 / temperature
                    logits_BlV1 = F.softmax(logits_BlV1, dim=-1)

                # sampling: create proper probability distribution
                alpha_t = self.noise.alpha(t_batch)
                alpha_s = self.noise.alpha(s_batch)
                if step == (num_scale_steps - 1): 
                    unmask_prob = torch.ones_like(alpha_t, dtype=torch.float32) # should be 1 but due to floating point precision, it may not
                else: 
                    unmask_prob = torch.zeros_like(alpha_t, dtype=torch.float32)
                    # if self.cont_time:
                    #     unmask_rate = self.noise.dgamma_times_alpha(t_batch) #[:, None, None]  # (B, 1, 1)
                    #     dt = (t_batch - s_batch) #[:, None, None]  # (B, 1, 1)
                    #     unmask_prob = 1 - torch.exp(unmask_rate * dt)
                    #     if unmask_prob.min() < 0 or unmask_prob.max() > 1:
                    #         print(f"unmask_prob: {unmask_prob.max().item()}")
                    #         unmask_prob = torch.clamp(unmask_prob, 0, 1)
                    # else:
                    #     unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)             
                unmask_prob = unmask_prob[:, None, None]
                
                # apply filtering
                vocab_probs = filter_top_k_top_p_(logits_BlV1, top_k, top_p)
                probs_vocab = unmask_prob * vocab_probs  
                probs_mask = (1 - unmask_prob).expand(-1, patch_length, -1) 
                probs_BlV1 = torch.cat([probs_vocab, probs_mask], dim=-1) 

                probs_flat = probs_BlV1.view(-1, probs_BlV1.shape[-1]) 
                sampled_tokens = torch.multinomial(
                    probs_flat, num_samples=1, generator=rng).view(B, patch_length)  

                # SUBS carry-over constraint: only update currently masked positions
                is_masked = (current_tokens == self.mask_index)
                current_tokens = torch.where(is_masked, sampled_tokens, current_tokens)

                # update embeddings for newly unmasked positions
                newly_unmasked = is_masked & (sampled_tokens != self.mask_index).bool()
                if newly_unmasked.any():
                    t_vae_embeddings = self.vae_quant_proxy[0].embedding(sampled_tokens[newly_unmasked])
                    t_word_embeddings = self.word_embed(t_vae_embeddings)
                    # update the input embeddings
                    x[:B][newly_unmasked] = t_word_embeddings 
                    x[B:] = x[:B]

                if (current_tokens == self.mask_index).sum() == 0:
                    break

            # process final tokens and prepare next_token_map for next stage
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(current_tokens)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV1.mul(1 + ratio), tau=gum_t, hard=False, 
                    dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw)
            
            # prepare next_token_map for next stage
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = (self.word_embed(next_token_map) + 
                                lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2])
                next_token_map = next_token_map.repeat(2, 1, 1)

        result = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]
        return result

    def loss(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor,
        gt_BL: torch.Tensor, num_steps: int = None, 
        ce_loss: bool = True, return_preds: bool = True, 
        loss_w_max: float = None) -> torch.Tensor:
        """
        compute diffusion loss for training.
        
        :param label_B: class labels (B,)
        :param x_BLCv_wo_first_l: input embeddings without first level (B, L, Cvae)
        :param gt_BL: ground truth token indices (B, L)
        :param num_steps: number of diffusion steps for inference
        :param ce_loss: whether to use cross entropy loss
        :param return_preds: whether to return predictions
        :param loss_w_max: maximum loss weight for scaling
        :return: dictionary containing diffusion losses and predictions
        """
        B = label_B.shape[0]
        t = self.sample_t(B, num_steps=num_steps)

        alpha_t = self.noise.alpha(t)  # probs token remains unmasked at time t
        p = 1.0 - alpha_t
        logits_BLV1 = self.forward(label_B, x_BLCv_wo_first_l, gt_BL, p, t)

        ce_flat = self.ce_loss(logits_BLV1.reshape(-1, self.V), gt_BL.reshape(-1))
        ce = ce_flat.reshape(B, -1)

        alpha_t1 = self.noise.alpha(torch.tensor(0.0, device=self.device))
        # loss_recon = - (torch.prod(torch.tensor(gt_BL.shape[1:], device=self.device))
        #              * (1.0 - alpha_t1)
        #              * torch.log(torch.tensor(self.V, device=self.device)))
        # latent_loss = torch.tensor(0.0)

        mask = self.last_masking_info
        masked_cross_ent = mask * ce
        sclaed_mce = (mask.shape[0]*mask.shape[1] / mask.sum()) * masked_cross_ent

        # if not self.cont_time:
        #     # loss for finite depth T, i.e. discrete time
        #     s = t - (1.0 / num_steps)
        #     gt = self.noise(t)
        #     gs = self.noise(s)
        #     weights = torch.expm1(gt - gs)[:, None] * self.noise.alpha(s)[:, None]
        #     if loss_w_max is not None:
        #         weights = weights.clamp(min=-loss_w_max, max=0)
        #     diff_loss = num_steps * weights * masked_neg_cross_ent
        # else:
        #     # cont-time loss
        #     weights = self.noise.dgamma_times_alpha(t)[:, None]
        #     if loss_w_max is not None:
        #         weights = weights.clamp(min=-loss_w_max, max=0)
        #     diff_loss = weights * masked_neg_cross_ent

        if return_preds:
            preds = logits_BLV1.argmax(dim=-1)
        else:
            preds = None

        losses = {
            # 'diff_loss': diff_loss,
            # 'loss_recon': loss_recon,
            # 'latent_loss': latent_loss,
            'scale': (mask.shape[0]*mask.shape[1] / mask.sum()),
            'ce': ce,
            'preds': preds,
            'masked': mask
        }
        return losses

    def q_xt(self, gt_BL: torch.Tensor, sos: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """uniform masking across all scales."""
        B, L = gt_BL.shape
        mask_prob = p.unsqueeze(1)  # (B, 1)
        mask_decisions = torch.rand(B, L-self.first_l, device=self.device) < mask_prob
        gt_BLC = self.vae_quant_proxy[0].embedding(gt_BL)
        noised_embeddings = torch.where(
            mask_decisions.unsqueeze(-1), 
            x_BLCv_wo_first_l, 
            gt_BLC[:, self.first_l:, :]
        )
        # store for accuracy calculation
        self.last_masking_info = torch.cat((
            torch.ones(B, self.first_l, device=self.device), 
            mask_decisions),
            dim=1)
        return noised_embeddings
    
    def sample_t(self, batch_size: int, min_t: float = 1e-3, 
                num_steps: int = None) -> torch.Tensor:
        """
        sample diffusion timesteps for training.
        
        :param batch_size: number of timesteps to sample
        :param min_t: minimum timestep value
        :param max_t: maximum timestep value
        :param num_steps: number of discrete steps (for discrete time)
        :return: sampled timesteps (batch_size,)
        """
        if num_steps is not None:
            min_t = 1.0 / num_steps + 1e-6
        t = torch.rand(batch_size, device=self.device) * (1.0 - min_t) + min_t
        if not self.cont_time and num_steps is not None and num_steps > 0: # discretize time steps
            t = (torch.floor(t * num_steps) + 1) / num_steps
        return t
    
    def attn_bias_slice(self, ed, bg=0, seq='both'):
        """
        slice attention bias matrix for specific sequence ranges.
        used for progressive training.
        
        :param ed: end index for sequence
        :param bg: begin index for sequence (default: 0)
        :param seq: sequence type ('both', 'xt')
        :return: sliced attention bias matrix
        """
        if ed == self.L and bg == 0 and seq == 'both':
            return self.attn_bias_for_masking

        if seq=='xt':
            tl = self.attn_bias_for_masking[:, :, :ed, :ed]
            tr = self.attn_bias_for_masking[:, :, :ed, self.L+bg:self.L+ed]
            bl = self.attn_bias_for_masking[:, :, self.L+bg:self.L+ed, :ed]
            br = self.attn_bias_for_masking[:, :, self.L+bg:self.L+ed, self.L+bg:self.L + ed]
        else:
            tl = self.attn_bias_for_masking[:, :, bg:ed, bg:ed]
            tr = self.attn_bias_for_masking[:, :, bg:ed, self.L+bg:self.L + ed]
            bl = self.attn_bias_for_masking[:, :, self.L+bg:self.L + ed, bg:ed]
            br = self.attn_bias_for_masking[:, :, self.L+bg:self.L + ed, self.L+bg:self.L + ed]

        top_attn_bias = torch.cat([tl, tr], dim=3)
        bottom_attn_bias = torch.cat([bl, br], dim=3)
        attn_bias = torch.cat([top_attn_bias, bottom_attn_bias], dim=2)
        return attn_bias

    def diff_mask_bias(self) -> torch.Tensor:
        """create diagonal attention bias for diffusion masking."""
        scale_sizes = [pn * pn for pn in self.patch_nums]
        T = sum(scale_sizes)
        total_length = 2*T  # x0 + xt 
        # initialize the full mask matrix
        mask = torch.zeros((total_length, total_length), dtype=torch.bool)
        # bottom left (xt → x0): block diagonal (next-scale attention)
        for i in range(len(scale_sizes)):
            xt_start_idx = T + sum(scale_sizes[:i])
            xt_end_idx = T + sum(scale_sizes[:i+1])
            # xt[i] can see x0[j] where j <= i (teacher forcing)
            x0_start_idx = sum(scale_sizes[:i])
            x0_end_idx = sum(scale_sizes[:i+1])
            mask[xt_start_idx:xt_end_idx, x0_start_idx:x0_end_idx] = True
        # bottom right (xt → xt): block diagonal (within-scale attention)
        for i in range(len(scale_sizes)):
            start_idx = T + sum(scale_sizes[:i])
            end_idx = T + sum(scale_sizes[:i+1])
            mask[start_idx:end_idx, start_idx:end_idx] = True
        attn_bias = torch.where(mask, 0.0, -torch.inf).unsqueeze(0).unsqueeze(0)
        return attn_bias

class CondEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, cond_dim, t_emb_dim=256):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.cond_dim = cond_dim
        self.total_dim = self.cond_dim + self.t_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, 4*self.cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(4*self.cond_dim, cond_dim, bias=True),
        )

        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)
    
    def forward(self, t, cond_BD):
        t_freq = get_timestep_embedding(t, self.t_emb_dim)
        t_emb = torch.cat((t_freq, cond_BD), dim=-1) 
        t_emb = self.mlp(t_emb)
        return t_emb

def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Build sinusoidal embeddings for timesteps.
    
    Creates sinusoidal position embeddings for diffusion timesteps, matching
    the implementation used in MD4 and other diffusion models.
    
    :param timesteps: Timestep values (B,)
    :param embedding_dim: Embedding dimension
    :param max_period: Maximum period for sinusoidal functions
    :return: Timestep embeddings (B, embedding_dim)
    """
    assert embedding_dim > 2
    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb