"""
Vector Autoregressive (VAR) model for image generation.

References:
- Scalable Image Generation via Next-Scale Prediction
  Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang

This file is modified from:
- VAR: https://github.com/FoundationVision/VAR

The VAR model implements autoregressive generation using transformer architecture
with VQVAE tokenization for high-quality image synthesis.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import dist

from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_, filter_top_k_top_p_
from models.vqvae import VQVAE
from models.quant import VectorQuantizer2
from models.transformer import Backbone


class VAR(nn.Module):
    """
    Vector Autoregressive model for image generation.
    
    This model implements autoregressive generation without diffusion,
    using a transformer-based architecture with progressive training across
    multiple patch scales.
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
        diffusion_args=None, **kwargs,
    ):
        """
        Initialize VAR model.
        
        :param algo: Algorithm type ('var')
        :param vae_local: VQVAE model for image tokenization
        :param num_classes: Number of class categories
        :param depth: Number of transformer layers
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param mlp_ratio: MLP expansion ratio
        :param drop_rate: Dropout rate
        :param attn_drop_rate: Attention dropout rate
        :param drop_path_rate: Stochastic depth rate
        :param norm_eps: Layer normalization epsilon
        :param shared_aln: Whether to use shared adaptive layer norm
        :param cond_drop_rate: Classifier-free guidance dropout rate
        :param attn_l2_norm: Whether to use L2 normalization in attention
        :param patch_nums: Tuple of patch sizes for progressive training
        :param flash_if_available: Whether to use Flash Attention if available
        :param fused_if_available: Whether to use fused operations if available
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.algo = algo
        
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device() if dist.initialized() else next(vae_local.parameters()).device)
    
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device() if dist.initialized() else torch.device('cpu'))
        
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        nn.init.trunc_normal_(self.word_embed.weight.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. set attention values
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        
        # 5. algorithm-specific components
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. create backbone
        self.backbone = Backbone(
            vocab_size=self.V, 
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
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, gt_BL: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for standard autoregressive training.
        
        This method implements the standard VAR forward pass with teacher forcing
        and progressive training support.
        
        :param label_B: Class labels (B,)
        :param x_BLCv_wo_first_l: Teacher forcing input embeddings (B, L-first_l, Cvae)
        :param gt_BL: Ground truth tokens (ignored for VAR)
        :param t: Timesteps (ignored for VAR)
        :return: Logits for next token prediction (B, L, V)
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast('cuda', enabled=False):
            label_B = torch.where(torch.rand(B, device=self.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: 
                x_BLC = sos
            else: 
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        x_BLC = self.backbone(x_BLC, cond_BD=cond_BD, attn_bias=attn_bias)
        logits_BlV = self.backbone.get_logits(x_BLC.float(), cond_BD)
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                logits_BlV[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                logits_BlV[0, 0, 0] += s
        return logits_BlV    # logits BLV, V is vocab_size


    @torch.no_grad()
    def infer_cfg_sr(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, max_pn = 16,
        more_smooth=False, **kwargs
    ) -> torch.Tensor:
        """
        Generate images using standard autoregressive inference with classifier-free guidance.
        
        This method performs standard autoregressive image generation by iteratively
        predicting the next token at each patch scale. It supports CFG for improved
        quality and various sampling strategies.
        
        :param B: Batch size for generation
        :param label_B: Class labels for conditional generation; if None, randomly sampled
        :param g_seed: Random seed for reproducible generation
        :param cfg: Classifier-free guidance ratio (higher = more class-conditional)
        :param top_k: Top-k sampling parameter (0 = disabled)
        :param top_p: Top-p (nucleus) sampling parameter (0.0 = disabled)
        :param more_smooth: Whether to use Gumbel softmax smoothing (for visualization only)
        :param kwargs: Additional arguments 
        :return: Generated images (B, 3, H, W) in [0, 1] range
        """

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.backbone.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            x = next_token_map

            x = self.backbone(x, cond_BD=cond_BD, attn_bias=None)
            logits_BlV = self.backbone.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.backbone.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def loss(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor,
        gt_BL: torch.Tensor, num_steps: int = None, 
        ce_loss: bool = True, return_preds: bool = True, 
        loss_w_max: float = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for autoregressive training.
        
        :param label_B: class labels (B,)
        :param x_BLCv_wo_first_l: input embeddings without first level (B, L, Cvae)
        :param gt_BL: ground truth token indices (B, L)
        :param num_steps: number of diffusion steps (ignored for VAR)
        :param ce_loss: whether to use cross entropy loss (ignored for VAR)
        :param return_preds: whether to return predictions
        :param loss_w_max: maximum loss weight for scaling (ignored for VAR)
        :return: dictionary containing 'ce' loss and 'preds' predictions
        """
        # Forward pass to get logits
        logits_BlV = self.forward(label_B, x_BLCv_wo_first_l, gt_BL)
        
        # Compute cross-entropy loss (reshape like DiffusionVAR)
        B = label_B.shape[0]
        ce_flat = self.loss_fn(logits_BlV.reshape(-1, self.V), gt_BL.reshape(-1))
        ce_BL = ce_flat.reshape(B, -1)
        
        if return_preds:
            preds_BL = torch.argmax(logits_BlV, dim=-1)
        else:
            preds_BL = None
            
        losses = {
            'ce': ce_BL,
            'preds': preds_BL,
        }
        return losses

        