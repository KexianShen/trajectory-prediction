import torch
from torch import nn

from .attn import RMSNorm
from .rope_attn import RoPESelfAttnLayer, precompute_freqs_cis


class ConvTokenizer(nn.Module):
    def __init__(self, input_dim=2, emb_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv1d(
            input_dim, emb_dim, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor):
        B, A, T, D = x.shape
        x = (
            self.proj(x.view(B * A, T, D).permute(0, 2, 1))
            .permute(0, 2, 1)
            .view(B, A, T, -1)
        )
        return x


class SpaNet(nn.Module):
    def __init__(
        self,
        lane_dim: int = 2,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
        max_t_len: int = 20,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.proj_layer = ConvTokenizer(lane_dim, dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(RoPESelfAttnLayer(num_heads, dim, norm_eps))
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.freqs_cis = precompute_freqs_cis(dim // num_heads, max_t_len)

    def forward(self, map_feat: torch.Tensor, agent_feat: torch.Tensor = None):
        if agent_feat is not None:
            x = torch.cat((agent_feat, map_feat), dim=1)
        else:
            x = map_feat

        self.freqs_cis = self.freqs_cis.to(x.device)
        mask = None

        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask)
        x = self.norm(x)
        return x
