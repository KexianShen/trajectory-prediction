import torch
from torch import nn

from .attn import CrossAttnLayer, RMSNorm, SelfAttnLayer, compute_2d_cos_sin_emb


class InterNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        headwise_attn_output_gate: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.self_atten_layers = torch.nn.ModuleList()
        self.lane_self_atten_layers = torch.nn.ModuleList()
        self.cross_atten_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.self_atten_layers.append(
                SelfAttnLayer(
                    num_heads=num_heads,
                    dim=dim,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                )
            )
            self.lane_self_atten_layers.append(
                SelfAttnLayer(
                    num_heads=num_heads,
                    dim=dim,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                )
            )
            self.cross_atten_layers.append(
                CrossAttnLayer(
                    num_heads=num_heads,
                    dim=dim,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                )
            )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        agent_feat: torch.Tensor,
        lane_feat: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
        agent_pos: torch.Tensor | None = None,
        lane_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos_emb_agent, sin_emb_agent = (None, None)
        cos_emb_lane, sin_emb_lane = (None, None)
        if agent_pos is not None and lane_pos is not None:
            cos_emb_agent, sin_emb_agent = compute_2d_cos_sin_emb(
                agent_pos, dim=agent_feat.size(-1)
            )
            cos_emb_lane, sin_emb_lane = compute_2d_cos_sin_emb(
                lane_pos, dim=lane_feat.size(-1)
            )

        for layer_id in range(self.num_layers):
            lane_feat = self.lane_self_atten_layers[layer_id](
                lane_feat,
                mask=lane_mask,
            )
            agent_feat = self.self_atten_layers[layer_id](
                agent_feat,
                mask=agent_mask,
            )
            agent_feat = self.cross_atten_layers[layer_id](
                agent_feat,
                lane_feat,
                lane_feat,
                mask=lane_mask,
                cos_emb_q=cos_emb_agent,
                sin_emb_q=sin_emb_agent,
                cos_emb_k=cos_emb_lane,
                sin_emb_k=sin_emb_lane,
            )
        agent_feat = self.norm(agent_feat)
        return agent_feat
