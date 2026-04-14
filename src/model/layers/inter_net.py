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
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        headwise_attn_output_gate: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.self_atten_layers = torch.nn.ModuleList()
        self.cross_atten_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.self_atten_layers.append(
                SelfAttnLayer(
                    num_heads=num_heads,
                    dim=dim,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                )
            )
            self.cross_atten_layers.append(
                CrossAttnLayer(
                    num_heads=num_heads,
                    dim=dim,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                )
            )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        agent_feat: torch.Tensor,
        lane_feat: torch.Tensor,
        agent_mask: torch.Tensor = None,
        lane_mask: torch.Tensor = None,
    ):
        for layer_id in range(self.num_layers):
            agent_feat = self.self_atten_layers[layer_id](
                agent_feat,
                mask=agent_mask,
            )
            agent_feat = self.cross_atten_layers[layer_id](
                agent_feat,
                lane_feat,
                lane_feat,
                mask=lane_mask,
            )
        agent_feat = self.norm(agent_feat)
        return agent_feat
