from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import FeedForward, RMSNorm


def compute_cos_sin_emb(
    dim: int, t: torch.Tensor = None, seq_len: int = None, base: float = 10000.0
):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    if t is None:
        t = torch.arange(seq_len)[:, None]
    freqs = t * freqs
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
):
    dim = xq.size(-1)
    half_dim = dim // 2

    xq1 = xq[..., :half_dim]
    xq2 = xq[..., half_dim:]
    xq_ = torch.cat([-xq2, xq1], dim=-1)
    xq_out = xq * cos_emb + xq_ * sin_emb

    xk1 = xk[..., :half_dim]
    xk2 = xk[..., half_dim:]
    xk_ = torch.cat([-xk2, xk1], dim=-1)
    xk_out = xk * cos_emb + xk_ * sin_emb
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEMHSA(nn.Module):
    def __init__(self, num_heads: int = 8, dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq, xk = apply_rotary_emb(xq, xk, cos_emb, sin_emb)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores + ~mask.unsqueeze(1) * -1e9
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        if mask is not None:
            scores = scores * mask.unsqueeze(1)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class RoPESelfAttnLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.attention = RoPEMHSA(num_heads, dim)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(self.attention_norm(x), cos_emb, sin_emb, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
