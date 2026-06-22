import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- your explicit FFN (no Sequential, no "proj") ----
class FeedForward(nn.Module):
    """
    FFN with explicit layers:
      - GELU MLP: Linear -> GELU -> Dropout -> Linear
      - Optional SwiGLU: Linear(2*inner) -> split -> SiLU(gate)*val -> Dropout -> Linear
    """
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0, gated: bool = False):
        super().__init__()
        self.gated = gated
        inner = int(mult * dim)
        self.drop = nn.Dropout(dropout)

        if gated:
            self.fc_in  = nn.Linear(dim, 2 * inner)  # gate/value
            self.fc_out = nn.Linear(inner, dim)
        else:
            self.fc_in  = nn.Linear(dim, inner)
            self.fc_out = nn.Linear(inner, dim)

        nn.init.xavier_uniform_(self.fc_in.weight);  nn.init.zeros_(self.fc_in.bias)
        nn.init.xavier_uniform_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            u, v = torch.chunk(self.fc_in(x), 2, dim=-1)
            x = F.silu(u) * v
            x = self.drop(x)
            return self.fc_out(x)
        else:
            x = self.fc_in(x)
            x = F.gelu(x)
            x = self.drop(x)
            return self.fc_out(x)


class SpatialLocalCrossAttention(nn.Module):
    """
    Pure cross-attention (no internal residual):
      - Q attends locally over K grid (CLS↔all).
      - Q/K/V may have different square spatial sizes.
      - 2D RoPE on Q and K spatial tokens.
      - Uses a float attention mask: 0 for allowed, NEG_LARGE for disallowed.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_q_tokens: int,
        num_k_tokens: int,
        window_radius: int = 1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.dim   = embed_dim
        self.heads = num_heads
        self.hdim  = embed_dim // num_heads
        self.r     = window_radius

        self.nq = num_q_tokens
        self.nk = num_k_tokens
        self.Hq = self.Wq = int((self.nq - 1) ** 0.5)
        self.Hk = self.Wk = int((self.nk - 1) ** 0.5)
        assert self.Hq * self.Wq == (self.nq - 1), "num_q_tokens must be 1 + Hq*Wq"
        assert self.Hk * self.Wk == (self.nk - 1), "num_k_tokens must be 1 + Hk*Wk"

        # linear layers (no "proj" names)
        self.to_q   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_k   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_v   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=True)

        for m in [self.to_q, self.to_k, self.to_v, self.to_out]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        # precompute boolean "allow" mask (True = allowed)
        self.register_buffer("allow_mask", self._build_allow_mask(), persistent=False)
        self.attn_dropout = attn_dropout

    def _build_allow_mask(self) -> torch.Tensor:
        Nq, Nk = self.nq, self.nk
        allow = torch.zeros((Nq, Nk), dtype=torch.bool)
        allow[0, :] = True          # Q-CLS attends to all K
        allow[:, 0] = True          # all Q can attend to K-CLS

        for qi in range(1, Nq):
            hq, wq = divmod(qi - 1, self.Wq)
            # center-aligned mapping
            hk_c = int((hq + 0.5) * self.Hk / self.Hq); hk_c = max(0, min(self.Hk - 1, hk_c))
            wk_c = int((wq + 0.5) * self.Wk / self.Wq); wk_c = max(0, min(self.Wk - 1, wk_c))
            h0, h1 = max(0, hk_c - self.r), min(self.Hk, hk_c + self.r + 1)
            w0, w1 = max(0, wk_c - self.r), min(self.Wk, wk_c + self.r + 1)
            for hk in range(h0, h1):
                for wk in range(w0, w1):
                    kj = 1 + hk * self.Wk + wk
                    allow[qi, kj] = True
        return allow


    def _float_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        neg_large = torch.finfo(dtype).min
        allow = self.allow_mask.to(device=device)
        mask = torch.zeros_like(allow, dtype=dtype, device=device)
        mask = torch.where(allow, mask, torch.full_like(mask, neg_large))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,Nq,Nk)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        Bq, Nq, Dq = q_in.shape
        Bk, Nk, Dk = k_in.shape
        Bv, Nv, Dv = v_in.shape
        assert Dq == Dk == Dv == self.dim
        assert Nq == self.nq and Nk == Nv == self.nk
        assert Bk == Bv and (Bk == 1 or Bk == Bq)  # allow broadcasting K/V

        q = self.to_q(q_in).view(Bq, Nq, self.heads, self.hdim)
        k = self.to_k(k_in).view(Bk, Nk, self.heads, self.hdim)
        v = self.to_v(v_in).view(Bk, Nv, self.heads, self.hdim)

        q = q.transpose(1, 2)  # (Bq, H, Nq, d)
        k = k.transpose(1, 2)  # (Bk, H, Nk, d)
        v = v.transpose(1, 2)  # (Bk, H, Nk, d)

        attn_mask = self._float_mask(q.dtype, q.device)  # (1,1,Nq,Nk)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask if self.r > 0 else None,                         # float mask added to scores
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (Bq, H, Nq, d)

        out = out.transpose(1, 2).reshape(Bq, Nq, self.dim)  # (Bq, Nq, D)
        return self.to_out(out) 

class SpatialLocalXAttnBlock(nn.Module):
    """
    Standard Pre-Norm cross-attention block with exactly two residuals:
      1) q = q + Dropout( CrossAttn( LN(q), LN(k), LN(v) ) )
      2) q = q + Dropout( FFN( LN(q) ) )
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_q_tokens: int,
        num_k_tokens: int,
        window_radius: int = 1,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_mult: float = 4.0,
        ffn_dropout: float = 0.0,
        ffn_gated: bool = True,
        norm_kv: bool = True,
    ):
        super().__init__()
        self.norm_q_attn = nn.LayerNorm(embed_dim)
        self.norm_q_ffn  = nn.LayerNorm(embed_dim)
        self.norm_kv     = nn.LayerNorm(embed_dim) if norm_kv else nn.Identity()

        self.attn = SpatialLocalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_q_tokens=num_q_tokens,
            num_k_tokens=num_k_tokens,
            window_radius=window_radius,
            attn_dropout=attn_dropout,
        )

        self.drop_resid_attn = nn.Dropout(resid_dropout)
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=ffn_dropout, gated=ffn_gated)
        self.drop_resid_ffn = nn.Dropout(resid_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # ---- residual around cross-attention ----
        q_norm = self.norm_q_attn(q)
        k_norm = self.norm_kv(k)
        v_norm = self.norm_kv(v)
        attn_out = self.attn(q_norm, k_norm, v_norm)
        q = q + self.drop_resid_attn(attn_out)

        # ---- residual around FFN ----
        y = self.norm_q_ffn(q)
        ffn_out = self.ffn(y)
        q = q + self.drop_resid_ffn(ffn_out)
        return q