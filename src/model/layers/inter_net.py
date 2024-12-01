import torch
from torch import nn

from .attn import CrossAttnLayer, RMSNorm, SelfAttnLayer


class InterNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.self_atten_layers = torch.nn.ModuleList()
        self.cross_atten_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.self_atten_layers.append(SelfAttnLayer(num_heads, dim, norm_eps))
            self.cross_atten_layers.append(CrossAttnLayer(num_heads, dim, norm_eps))
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        agent_feat: torch.Tensor,
        lane_feat: torch.Tensor,
    ):
        mask = None

        for layer_id in range(self.num_layers):
            agent_feat = self.self_atten_layers[layer_id](agent_feat, mask)
            agent_feat = self.cross_atten_layers[layer_id](
                agent_feat, lane_feat, lane_feat, mask
            )
        agent_feat = self.norm(agent_feat)
        return agent_feat
