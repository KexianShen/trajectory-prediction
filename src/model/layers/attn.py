import torch
import torch.nn.functional as F
from torch import nn


def compute_cos_sin_emb(
    dim: int = 128,
    t: torch.Tensor = None,
    seq_len: int = None,
    base: float = 10000.0,
):
    if t is None:
        t = torch.arange(seq_len)[:, None]
    t = t.float()
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=t.device).float() / dim))
    freqs = t * freqs
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def compute_2d_cos_sin_emb(
    t: torch.Tensor = None,
    dim: int = 128,
    base: float = 10000.0,
):
    freqs = 1.0 / (base ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
    freqs = freqs.to(t.device)
    freqs_x, freqs_y = t[..., :1] * freqs, t[..., 1:2] * freqs
    freqs_x, freqs_y = freqs_x.repeat(1, 1, 2), freqs_y.repeat(1, 1, 2)
    cos_emb = torch.cat([freqs_x.cos(), freqs_y.cos()], dim=-1)
    sin_emb = torch.cat([freqs_x.sin(), freqs_y.sin()], dim=-1)
    return cos_emb, sin_emb


def rotate_half(x: torch.Tensor):
    half_dim = x.size(-1) // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_emb_q: torch.Tensor,
    sin_emb_q: torch.Tensor,
    cos_emb_k: torch.Tensor = None,
    sin_emb_k: torch.Tensor = None,
):
    xq_out = xq * cos_emb_q + rotate_half(xq) * sin_emb_q
    if cos_emb_k is None and sin_emb_k is None:
        cos_emb_k, sin_emb_k = cos_emb_q, sin_emb_q
    xk_out = xk * cos_emb_k + rotate_half(xk) * sin_emb_k
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x).type_as(x)
        return output * self.w


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int = 3, dim: int = 128, num_freqs: int = 64):
        super().__init__()
        self.input_dim = input_dim

        self.freqs = nn.Embedding(input_dim, num_freqs)
        self.w = nn.ModuleList(
            nn.Sequential(
                nn.Linear(num_freqs * 2 + 1, dim),
                RMSNorm(dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            for _ in range(input_dim)
        )
        self.wo = nn.Sequential(RMSNorm(dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        x_ = x[..., None] * self.freqs.weight * 2 * torch.pi
        x_ = torch.cat([x_.cos(), x_.sin(), x[..., None]], dim=-1)
        x_ = torch.stack([self.w[i](x_[..., i, :]) for i in range(self.input_dim)])
        return self.wo(x_.sum(dim=0))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MHSA(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        dropout: float = 0.1,
        headwise_attn_output_gate: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate

        if self.headwise_attn_output_gate:
            self.wq = nn.Linear(dim, num_heads * self.head_dim + num_heads, bias=False)
        else:
            self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor = None,
        cos_emb: torch.Tensor = None,
        sin_emb: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.headwise_attn_output_gate:
            xq, gate_score = xq.split(
                [self.num_heads * self.head_dim, self.num_heads], dim=-1
            )
            gate_score = gate_score.view(batch_size, seq_len, self.num_heads, 1)

        if pos_emb is not None:
            xq, xk = xq + pos_emb, xk + pos_emb
        elif cos_emb is not None and sin_emb is not None:
            xq, xk = apply_rotary_emb(xq, xk, cos_emb, sin_emb)

        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        attn_mask = None if mask is None else mask.view(batch_size, 1, 1, -1)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        if self.headwise_attn_output_gate:
            output = output * torch.sigmoid(gate_score)
        output = output.view(batch_size, seq_len, -1)
        return self.wo(output)


class MHCA(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        dropout: float = 0.1,
        headwise_attn_output_gate: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate

        if self.headwise_attn_output_gate:
            self.wq = nn.Linear(dim, num_heads * self.head_dim + num_heads, bias=False)
        else:
            self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        pos_emb_q: torch.Tensor = None,
        pos_emb_k: torch.Tensor = None,
        cos_emb_q: torch.Tensor = None,
        sin_emb_q: torch.Tensor = None,
        cos_emb_k: torch.Tensor = None,
        sin_emb_k: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        batch_size, query_len, _ = xq.shape
        xq, xk, xv = self.wq(xq), self.wk(xk), self.wv(xv)

        if self.headwise_attn_output_gate:
            xq, gate_score = xq.split(
                [self.num_heads * self.head_dim, self.num_heads], dim=-1
            )
            gate_score = gate_score.view(batch_size, query_len, self.num_heads, 1)

        if pos_emb_q is not None and pos_emb_k is not None:
            xq, xk = xq + pos_emb_q, xk + pos_emb_k
        elif (
            cos_emb_q is not None
            and sin_emb_q is not None
            and cos_emb_k is not None
            and sin_emb_k is not None
        ):
            xq, xk = apply_rotary_emb(
                xq, xk, cos_emb_q, sin_emb_q, cos_emb_k, sin_emb_k
            )

        xq = xq.view(batch_size, query_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        attn_mask = None if mask is None else mask.view(batch_size, 1, 1, -1)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        if self.headwise_attn_output_gate:
            output = output * torch.sigmoid(gate_score)
        output = output.view(batch_size, query_len, -1)
        return self.wo(output)


class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-6,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        headwise_attn_output_gate: bool = True,
    ):
        super().__init__()

        self.mhsa = MHSA(
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.mhsa_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.mhsa_dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForward(dim=dim, hidden_dim=int(ffn_multiplier * dim))
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.out_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        pos_emb: torch.Tensor = None,
        cos_emb: torch.Tensor = None,
        sin_emb: torch.Tensor = None,
    ):
        x_ = self.mhsa(self.mhsa_norm(x), pos_emb, cos_emb, sin_emb, mask)
        x = x + self.mhsa_dropout(x_)
        x_ = self.ffn(self.ffn_norm(x))
        x = x + self.ffn_dropout(x_)
        return self.out_norm(x)


class CrossAttnLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-6,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        headwise_attn_output_gate: bool = True,
    ):
        super().__init__()

        self.mhca = MHCA(
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.mhca_norm_q = RMSNorm(dim=dim, eps=norm_eps)
        self.mhca_norm_k = RMSNorm(dim=dim, eps=norm_eps)
        self.mhca_norm_v = RMSNorm(dim=dim, eps=norm_eps)
        self.mhca_dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForward(dim=dim, hidden_dim=int(ffn_multiplier * dim))
        self.ffn_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: torch.Tensor = None,
        pos_emb_q: torch.Tensor = None,
        pos_emb_k: torch.Tensor = None,
        cos_emb_q: torch.Tensor = None,
        sin_emb_q: torch.Tensor = None,
        cos_emb_k: torch.Tensor = None,
        sin_emb_k: torch.Tensor = None,
    ):
        xq_ = self.mhca(
            self.mhca_norm_q(xq),
            self.mhca_norm_k(xk),
            self.mhca_norm_v(xv),
            pos_emb_q,
            pos_emb_k,
            cos_emb_q,
            sin_emb_q,
            cos_emb_k,
            sin_emb_k,
            mask,
        )
        xq = xq + self.mhca_dropout(xq_)
        xq_ = self.ffn(self.ffn_norm(xq))
        xq = xq + self.ffn_dropout(xq_)
        return xq
