import math
import torch
from torch import nn
import torch.nn.functional as F
from .local_xattn import SpatialLocalXAttnBlock

class FusionEncoder(nn.Module):
    def __init__(
        self,
        strategy: str = 'xattn',
        learned_emb_mask: bool = True,
        inp_dim:    int  = 1024,
        proj_dim:   int  = 1024,
        n_rgb_tokens: int = 577,
        n_ir_tokens:  int = 257,
        nheads:     int  = 1,
        dropout:    float= 0.1,
        window_radius: int  = 1,
        nblocks:    int  = 1
    ):
        super().__init__()
        assert strategy in ["local_xattn", 'concat'], f"Unknown fusion strategy: {strategy}"
        self.strategy = strategy
        self.learned_emb_mask = learned_emb_mask
        self.inp_dim  = inp_dim
        self.proj_dim = proj_dim
        self.window_radius = window_radius


        print("Using dropout:", dropout)

        if strategy == "local_xattn":
            window_radii = [i for i in range(1, nblocks+1)]
            self.local_xattn = nn.ModuleList()
            for i in range(nblocks):
                block = SpatialLocalXAttnBlock(
                    embed_dim=inp_dim,
                    num_heads=nheads,
                    num_q_tokens=n_rgb_tokens,
                    num_k_tokens=n_ir_tokens,
                    window_radius=window_radii[i] if self.window_radius > 0 else -1,
                    attn_dropout=dropout,
                    resid_dropout=dropout,
                    ffn_dropout=dropout
                )
                self.local_xattn.append(block)
        elif strategy == 'concat':
            pass  # no extra modules needed
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")



    def local_xattn_forward(self, rgb, m1):
        out = rgb
        for block in self.local_xattn:
            out = block(out, m1, m1)

        return out

    def forward(self, img_feats, m1_feats):
        rgb = img_feats
        m1  = m1_feats
        
        if self.strategy == "local_xattn":
            out = self.local_xattn_forward(rgb, m1)
            return out

        elif self.strategy == 'concat':
            return torch.cat([rgb, m1], dim=1)  # (B, 2*N)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")



