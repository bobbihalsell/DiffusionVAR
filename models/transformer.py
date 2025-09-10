import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class Backbone(nn.Module):
    """
    Extracted backbone from VAR model - the transformer architecture for autoregressive generation.
    This contains the core transformer blocks, embeddings, and attention mechanisms.
    """
    def __init__(
        self, 
        vocab_size: int,
        embed_dim: int = 1024, 
        num_heads: int = 16, 
        depth: int = 16,
        mlp_ratio: float = 4., 
        drop_rate: float = 0., 
        attn_drop_rate: float = 0., 
        drop_path_rate: float = 0.,
        norm_eps: float = 1e-6, 
        shared_aln: bool = False, 
        attn_l2_norm: bool = False,
        flash_if_available: bool = True, 
        fused_if_available: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        
        # validate inputs
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # core hyperparameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.zero_init = zero_init
        # backbone transformer blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(embed_dim, 6*embed_dim)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                block_idx=block_idx, 
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                cond_dim=embed_dim, 
                embed_dim=embed_dim, 
                shared_aln=shared_aln,
                norm_layer=norm_layer, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx], 
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, 
                fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        # check for fused add-norm functionality
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # output head
        self.head_nm = AdaLNBeforeHead(embed_dim, embed_dim, norm_layer=norm_layer)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Print configuration
        print(
            f'\n[Backbone] ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [Drop ratios] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g}\n',
            end='\n\n', flush=True
        )
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        """Extract logits from transformer output"""
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def forward(self, x_BLC, cond_BD, attn_bias):
        """
        Forward pass through the transformer backbone
        :param x_BLC: Input tensor (B, L, C) where L is sequence length
        :param cond_BD: Conditioning tensor (B, D) for AdaLN
        :param attn_bias: Attention bias mask (1, 1, L, L)
        :return: Output tensor after all transformer blocks (B, L, C)
        """
        # global AdaLN conditioning
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        # Forward through transformer blocks
        for i, block in enumerate(self.blocks):
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        return x_BLC
    

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.embed_dim / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}, zero_init={self.zero_init}')
        # general initialization
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        # head initialization
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        # AdaLN head initialization
        if isinstance(self.head_nm, AdaLNBeforeHead):
            if self.zero_init: nn.init.zeros_(self.head_nm.ada_lin[-1].weight.data)
            else: self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # AdaLN initialization
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                if self.zero_init: 
                    nn.init.zeros_(sab.ada_lin[-1].weight.data)
                else:
                    sab.ada_lin[-1].weight.data[2*sab.C:].mul_(init_adaln)
                    sab.ada_lin[-1].weight.data[:2*sab.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                if self.zero_init:
                    nn.init.zeros_(sab.ada_gss.data)    
                else:
                    sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                    sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
