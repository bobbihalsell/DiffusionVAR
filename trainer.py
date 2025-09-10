"""
Trainer module for VAR and DiffusionVAR models.

References:
- VAR: https://github.com/FoundationVision/VAR

This module provides the training step implementation, validation logic,
and metric calculation for both VAR and DiffusionVAR models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, Union, List
import time

from utils import AmpOptimizer, MetricLogger, TensorboardLogger
from models import VQVAE
import dist

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class Trainer:    
    """
    unified trainer class for both VAR and DiffusionVAR models.
    
    this trainer handles training, evaluation, and checkpointing for both autoregressive
    and diffusion-based vector autoregressive models. It supports progressive training,
    mixed precision training, and comprehensive logging.
    """
    def __init__(self, algo: str, device: torch.device, patch_nums: tuple, resos: tuple,
                 vae_local: VQVAE, var_wo_ddp: torch.nn.Module, var: torch.nn.Module,
                 var_opt: AmpOptimizer, label_smooth: float = 0.0,
                 diffusion_args: dict = None):
        """
        initialize the trainer with models and training configuration.
        
        :param algo: algorithm type ('var' or 'diffusion-var')
        :param device: training device (cuda/cpu)
        :param patch_nums: tuple of patch sizes for progressive training
        :param resos: tuple of resolutions for each patch size
        :param vae_local: vqvae model for image tokenization
        :param var_wo_ddp: VAR/DiffusionVAR model without DDP wrapper
        :param var: VAR/DiffusionVAR model with DDP wrapper
        :param var_opt: mixed precision optimizer
        :param label_smooth: label smoothing factor
        :param diffusion_args: dictionary of diffusion-specific arguments
        """
        self.algo = algo
        self.device = device
        self.patch_nums = patch_nums
        self.resos = resos
        self.vae_local = vae_local
        self.var_wo_ddp = var_wo_ddp
        self.var = var
        self.var_opt = var_opt
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        self.label_smooth = label_smooth
        self.diffusion_args = diffusion_args
        self.train_steps = getattr(diffusion_args, 'train_steps', 1000)
        self.cont_time = getattr(diffusion_args, 'cont_time', False)
        self.loss_w_max = getattr(diffusion_args, 'loss_w_max', 5.1)

        assert self.device == self.var_wo_ddp.device
        
        # quantizer reference
        self.quantize_local = vae_local.quantize
        
        # loss functions
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        # progressive training boundaries
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        # progressive training state
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True


    def get_config(self):
        """
        return current configuration dictionary for checkpointing.
        
        :return: configuration dictionary containing training state and parameters
        """
        return {
            'prog_it': getattr(self, 'prog_it', 0),
            'last_prog_si': getattr(self, 'last_prog_si', -1),
            'first_prog': getattr(self, 'first_prog', True),
            'algo': getattr(self, 'algo', 'diffusion-var'),
            'diffusion_args': getattr(self, 'diffusion_args', None),
        }

    def eval(self):
        """
        set models to evaluation mode.
        """
        self.var_wo_ddp.eval()
        self.vae_local.eval()
    
    def train(self, training: bool = True):
        """
        set models to training mode.
        """
        self.var_wo_ddp.train(training)
        self.vae_local.train(training)

    def stepping_loss(self, losses: dict,
        prog_si: int = -1, prog_wp: float = 0.0,
    ) -> torch.Tensor:
        """
        compute training loss with progressive training and diffusion weighting.
        
        For diffusion models, applies masking-aware loss weighting and diffusion-specific
        loss scaling. For progressive training, applies warmup weighting to new stages.
        
        :param losses: model output logits (B, L, V)
        :param prog_si: progressive training stage index (-1 for no progressive training)
        :param prog_wp: progressive training warmup progress [0, 1]
        :return: scalar loss tensor
        """ 
        if self.algo == 'diffusion-var':
            loss_BL = losses['diff_loss']
        else:
            loss_BL = losses['ce']

        B, L = loss_BL.shape

        if prog_si >= 0:    # in progressive training
            bg, ed = self.begin_ends[prog_si]
            assert loss_BL.shape[1] == ed
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
        else:               # not in progressive training
            lw = self.loss_weight

        loss_BL = loss_BL.mul(lw)

        if self.algo == 'diffusion-var':
            diff_loss = loss_BL.sum(dim=-1).mean()
            recon_loss = losses['loss_recon']
            latent_loss = losses['latent_loss']
            loss = diff_loss + recon_loss + latent_loss
        else:
            loss = loss_BL.sum(dim=-1).mean()

        return loss

    def scale_losses(self, losses: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        compute losses at different scales for evaluation.
        
        calculates both overall mean loss and per-scale losses for progressive training
        evaluation. For diffusion models, applies diffusion weighting.
        
        :param loss_BL: model output logits (B, L)
        :param t: diffusion timesteps (B,) - required for diffusion models
        :return: tuple of (mean_loss, scale_losses) where scale_losses has shape (num_scales,)
        """
        loss_BL = losses['ce']
        B, L = loss_BL.shape
        
        # Lmean = loss_BL.mean()
        scale_L = torch.zeros(len(self.begin_ends), device=self.device)

        if self.algo == 'diffusion-var':
            masked = losses['masked']
            # Average loss per masked token (across entire batch)
            Lmean = (loss_BL * masked).sum() / masked.sum()
        else:
            # Average loss per token (across entire batch)  
            Lmean = loss_BL.mean()

        # In progressive training, only compute up to current stage + 1
        if hasattr(self.var_wo_ddp, 'prog_si') and self.var_wo_ddp.prog_si >= 0:  
            num_scales = min(self.var_wo_ddp.prog_si + 1, len(self.begin_ends))
        else:
            num_scales = len(self.begin_ends)

        for i in range(num_scales):
            bg, ed = self.begin_ends[i]
            curr_loss_BL = loss_BL[:, bg:ed]
            if self.algo == 'diffusion-var':
                curr_masked = masked[:, bg:ed]
                curr_loss = (curr_loss_BL * curr_masked).sum() / curr_masked.sum()
            else:
                curr_loss = curr_loss_BL.mean()

            scale_L[i] = curr_loss
        return Lmean, scale_L

    def accuracy(self, losses: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        calculate accuracy for the model.
        
        computes both overall accuracy and per-scale accuracies. For diffusion models,
        only considers masked positions for accuracy calculation.
        
        :param losses: dict containing 'preds' and 'masked' keys
        :param targets: ground truth tokens (B, L)
        :return: tuple of (mean_accuracy, scale_accuracies) where accuracies are in percentage (0-100)
        """
        B, L = losses['preds'].shape[0], losses['preds'].shape[1]
        if hasattr(self.var_wo_ddp, 'prog_si') and self.var_wo_ddp.prog_si >= 0:
            num_scales_to_compute = min(self.var_wo_ddp.prog_si + 1, len(self.begin_ends))
        else:
            num_scales_to_compute = len(self.begin_ends)
        acc_scales = torch.zeros(len(self.begin_ends), device=self.device)
        
        if self.algo == 'diffusion-var':
            masking_info = losses['masked']
                
            masked_correct_BL = (losses['preds'] == targets) & masking_info
            num_masked = masking_info.sum().item()
            
            if num_masked > 0: acc_mean = (masked_correct_BL.sum().item() / num_masked) * 100
            else: acc_mean = 0.0

            for i in range(num_scales_to_compute):
                bg, ed = self.begin_ends[i]
                if masking_info[:, bg:ed].sum().item() > 0:
                    acc_scales[i] = (masked_correct_BL[:, bg:ed].sum().item() / 
                                    masking_info[:, bg:ed].sum().item()) * 100
                else:
                    acc_scales[i] = 0.0
        else:
            acc_mean = (losses['preds'] == targets).float().mean().item() * 100
            for i in range(num_scales_to_compute):
                bg, ed = self.begin_ends[i]
                acc_scales[i] = (losses['preds'][:, bg:ed] == targets[:, bg:ed]).float().mean().item() * 100
            
        return acc_mean, acc_scales

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader) -> Tuple[float, torch.Tensor, float, torch.Tensor, int, float]:
        """
        evaluate one epoch on validation data.
        
        Performs full evaluation pass through validation dataset, computing losses and
        accuracies at all scales. Handles distributed evaluation with proper aggregation.
        
        :param ld_val: validation dataloader
        :return: tuple of (mean_loss, scale_losses, mean_accuracy, scale_accuracies, total_samples, eval_time)
        """
        tot = 0
        Lmean, acc_mean = 0, 0
        num_scales = len(self.begin_ends)
        Lscale = torch.zeros(num_scales, device=self.device)
        acc_scale = torch.zeros(num_scales, device=self.device)

        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        
        for inp_B3HW, label_B in ld_val:
            B = label_B.shape[0]
            inp_B3HW = inp_B3HW.to(self.device, non_blocking=True)
            label_B = label_B.to(self.device, non_blocking=True)

            gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: torch.Tensor = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            losses = self.var_wo_ddp.loss(
                label_B=label_B, 
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                gt_BL=gt_BL,
                num_steps=self.train_steps,
                loss_w_max=self.loss_w_max,
            )

            Lmean_batch, Lscale_batch = self.scale_losses(losses)
            Lmean += Lmean_batch * B
            Lscale += Lscale_batch * B

            batch_acc_mean, batch_acc_scale = self.accuracy(losses, gt_BL)
            acc_mean += batch_acc_mean * B
            acc_scale += batch_acc_scale * B

            tot += B
            
        self.var_wo_ddp.train(training)
        stats = torch.cat([
            torch.tensor([Lmean], device=self.device),
            Lscale,
            torch.tensor([acc_mean], device=self.device), 
            acc_scale,
            torch.tensor([tot], device=self.device)
        ])
        
        if dist.initialized():
            dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot  
        Lmean = stats[0].item()
        Lscale = stats[1:1+num_scales]
        acc_mean = stats[1+num_scales].item()
        acc_scale = stats[2+num_scales:2+2*num_scales]
        return Lmean, Lscale, acc_mean, acc_scale, tot, time.time()-stt

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, 
        tb_lg: TensorboardLogger, inp_B3HW: torch.Tensor, label_B: torch.Tensor, 
        prog_si: int, prog_wp_it: float, wandb_run=None, 
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float], Optional[dict]]:
        """
        single training step with forward/backward pass and logging.
        
        performs one training iteration including progressive training setup, forward pass,
        loss computation, backward pass, and comprehensive logging to multiple backends.
        
        :param it: current iteration within epoch
        :param g_it: global iteration across all epochs
        :param stepping: whether to perform optimizer step (vs gradient accumulation)
        :param metric_lg: metric logger for console output
        :param tb_lg: tensorboard logger
        :param inp_B3HW: input images (B, 3, H, W) in [0, 1] range
        :param label_B: class labels (B,) with values 0 to num_classes-1
        :param prog_si: progressive training stage index
        :param prog_wp_it: progressive training warmup iterations
        :param wandb_run: wandb run object for experiment tracking
        :return: tuple of (grad_norm, scale_log2, wandb_metrics)
        """
        wandb_log_data = None
        
        B = label_B.shape[0]
        # progressive training setup
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1   # max prog, as if no prog
        
        # forward pass
        self.var.require_backward_grad_sync = stepping
        inp_B3HW = inp_B3HW.to(self.device, non_blocking=True)
        label_B = label_B.to(self.device, non_blocking=True)
        
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
 
        with self.var_opt.amp_ctx:
            losses = self.var_wo_ddp.loss(
                label_B=label_B, 
                x_BLCv_wo_first_l=x_BLCv_wo_first_l, 
                gt_BL=gt_BL,        
                num_steps=self.train_steps,
                loss_w_max=self.loss_w_max,
            )
            loss = self.stepping_loss(losses, prog_si=prog_si, prog_wp=prog_wp)  
        # backward pass
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # logging
        if it == 0 or it in metric_lg.log_iters:
            Lmean, Lscale = self.scale_losses(losses)
            acc_mean, acc_scale = self.accuracy(losses, gt_BL)
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            metric_lg.update(Lm=Lmean, Lt=Lscale[-1], Accm=acc_mean, Acct=acc_scale[-1], tnm=grad_norm_val)

            # tensorboard logging
            if tb_lg is not None:
                tb_lg.update(step=g_it, DIFFUSION_iter_loss=Lmean)
                tb_lg.update(step=g_it, DIFFUSION_iter_acc=acc_mean)
                if Lscale[-1] != -1:
                    tb_lg.update(step=g_it, DIFFUSION_iter_loss_tail=Lscale[-1])
                    tb_lg.update(step=g_it, DIFFUSION_iter_acc_tail=acc_scale[-1])
                tb_lg.update(step=g_it, DIFFUSION_iter_grad_norm=grad_norm)
                tb_lg.update(step=g_it, DIFFUSION_iter_scale_log2=scale_log2)
            
            # wandb logging
            if wandb_run is not None and it // 4 in metric_lg.log_iters: # log less frequently
                wandb_log_data = {
                    "diffusion/iter_loss": Lmean,
                    "diffusion/iter_acc": acc_mean,
                    "diffusion/iter_grad_norm": grad_norm,
                    "diffusion/iter_scale_log2": scale_log2,
                    "diffusion/prog_si": prog_si,
                    "diffusion/prog_wp": prog_wp,
                    "diffusion/tloss": loss,
                }

                for i, (loss_val, acc_val) in enumerate(zip(Lscale, acc_scale)):
                    if loss_val != -1:  # Only log valid scales
                        wandb_log_data[f"train_scale_loss/scale_{i}"] = loss_val
                        wandb_log_data[f"train_scale_acc/scale_{i}"] = acc_val
                
                if Lscale[-1] != -1:
                    wandb_log_data.update({
                        "diffusion/iter_loss_tail": Lscale[-1],
                        "diffusion/iter_acc_tail": acc_scale[-1],
                    })
                    
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2, wandb_log_data

    def state_dict(self):
        """
        get state dict for checkpointing.
        :return: dictionary containing model states and configuration
        """
        state = {}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        """
        load state dict from checkpoint.
        :param state: state dictionary to load
        :param strict: whether to strictly enforce state dict matching
        :param skip_vae: whether to skip loading VAE state
        """
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        self.algo = config.get('algo', 'diffusion-var')
        self.diffusion_args = config.get('diffusion_args', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
