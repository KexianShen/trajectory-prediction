from typing import Optional

import torch
from torch import nn

from .attn import RMSNorm, SelfAttnLayer, compute_cos_sin_emb


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(
            input_dim,
            emb_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, seq_len, input_dim = x.shape
        x = x.reshape(batch_size * num_tokens, seq_len, input_dim).transpose(1, 2)
        x = self.proj(x).transpose(1, 2)
        return x.reshape(batch_size, num_tokens, seq_len, -1)


class SequencePooling(nn.Module):
    def __init__(self, dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.score = nn.Linear(dim, 1, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm(x)
        score = self.score(x_norm).squeeze(-1)

        if mask is None:
            weights = torch.softmax(score, dim=-1)
            return self.out(torch.sum(weights.unsqueeze(-1) * self.value(x), dim=1))

        score = score.masked_fill(~mask, -1e4)
        weights = torch.softmax(score, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(weights.unsqueeze(-1) * self.value(x), dim=1)
        pooled = pooled * mask.any(dim=-1, keepdim=True)
        return self.out(pooled)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        seq_len: int = 110,
        headwise_attn_output_gate: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.proj_layer = ConvTokenizer(input_dim=input_dim, emb_dim=dim)
        self.layers = nn.ModuleList(
            SelfAttnLayer(
                num_heads=num_heads,
                dim=dim,
                norm_eps=norm_eps,
                dropout=dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(dim, eps=norm_eps)
        cos_emb, sin_emb = compute_cos_sin_emb(dim=dim, seq_len=seq_len)
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.cos_emb.size(0):
            raise ValueError(
                f"input sequence length {seq_len} exceeds configured RoPE length {self.cos_emb.size(0)}"
            )
        cos_emb = self.cos_emb[:seq_len]
        sin_emb = self.sin_emb[:seq_len]

        for layer in self.layers:
            x = layer(x, mask=mask, cos_emb=cos_emb, sin_emb=sin_emb)
        return self.norm(x)
