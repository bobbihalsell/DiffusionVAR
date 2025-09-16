"""
This module provides organized implementations of VAR and DiffusionVAR models
with VQVAE tokenization, transformer backbones, and diffusion scheduling.
"""

from typing import Tuple, Type, Union
import torch.nn as nn

from models.quant import VectorQuantizer2
from models.vqvae import VQVAE
from models.schedule import MaskingSchedule
from models.transformer import Backbone

from models.dvar import DiffusionVAR
from models.dvar.var import VAR


__all__ = ['DiffusionVAR', 'VAR', 'VQVAE', 'VectorQuantizer2', 'MaskingSchedule', 'Backbone', 'build_vae_var', 'get_model_class']


def get_model_class(algo: str) -> Type[nn.Module]:
    """Get model class based on algorithm type."""
    if algo == 'diffusion-var':
        from .dvar import DiffusionVAR
        return DiffusionVAR
    elif algo == 'var':
        from .dvar.var import VAR
        return VAR
    else:
        raise ValueError(f"Invalid algorithm: {algo}")


def build_vae_var(
    # shared args
    device, 
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, 
    Cvae=32, 
    ch=160, 
    share_quant_resi=4,
    # VAR args
    algo='diffusion-var', 
    cond_drop_rate=0.1,
    num_classes=10, 
    depth=16, 
    shared_aln=False, 
    attn_l2_norm=True,
    flash_if_available=True, 
    fused_if_available=True,
    init_adaln=0.5, 
    init_adaln_gamma=1e-5, 
    init_head=0.02, 
    init_std=-1,    # init_std < 0: automated
    drop_rate=0.,
    attn_drop_rate=0.,
    diffusion_args=None,
    ) -> Tuple[VQVAE, nn.Module]:

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # Disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, 
                nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, 
                      z_channels=Cvae, 
                      ch=ch, 
                      test_mode=True, 
                      share_quant_resi=share_quant_resi, 
                      v_patch_nums=patch_nums, 
                      ).to(device)

    # Build VAR/DiffusionVAR
    ModelClass = get_model_class(algo)
    var_wo_ddp = ModelClass(
        algo=algo,
        vae_local=vae_local,
        num_classes=num_classes, 
        depth=depth, 
        embed_dim=width, 
        num_heads=heads, 
        drop_rate=drop_rate, 
        attn_drop_rate=attn_drop_rate, 
        drop_path_rate=dpr,
        norm_eps=1e-6, 
        shared_aln=shared_aln, 
        cond_drop_rate=cond_drop_rate,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, 
        fused_if_available=fused_if_available,
        diffusion_args=diffusion_args,
        ).to(device)
        
    var_wo_ddp.backbone.init_weights(
        init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, 
        init_head=init_head, init_std=init_std
    )
    
    return vae_local, var_wo_ddp