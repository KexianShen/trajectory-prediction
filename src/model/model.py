from typing import Dict

import torch
import torch.nn as nn

from .layers import AgentEncoder, InterNet, LaneEncoder, SpaNet, TempoNet
from .layers.attn import SelfAttnLayer
from .layers.sequence import SequencePooling


class Model(nn.Module):
    def __init__(
        self,
        agent_dim: int = 4,
        lane_dim: int = 2,
        num_agent_type: int = 4,
        num_lane_type: int = 1,
        dim: int = 128,
        encoder_depth: int = 4,
        decoder_depth: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        headwise_attn_output_gate: bool = True,
        num_modals: int = 6,
        future_steps: int = 60,
    ) -> None:
        super().__init__()
        self.num_modals = num_modals
        self.future_steps = future_steps

        self.agent_encoder = AgentEncoder(
            dim=dim,
            num_agent_type=num_agent_type,
        )
        self.lane_encoder = LaneEncoder(
            dim=dim,
            num_lane_type=num_lane_type,
        )
        self.tempo_net = TempoNet(
            agent_dim=agent_dim,
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            norm_eps=norm_eps,
            seq_len=50,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.spa_net = SpaNet(
            lane_dim=lane_dim,
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            norm_eps=norm_eps,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.inter_net = InterNet(
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            norm_eps=norm_eps,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.modal_emb = nn.Embedding(num_modals, dim)
        self.agent_pool = SequencePooling(dim=dim, norm_eps=norm_eps)
        self.lane_pool = SequencePooling(dim=dim, norm_eps=norm_eps)
        self.mode_layers = nn.ModuleList(
            SelfAttnLayer(
                num_heads=num_heads,
                dim=dim,
                norm_eps=norm_eps,
                dropout=dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
            )
            for _ in range(decoder_depth)
        )

        self.traj_decoder = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, future_steps * 2),
        )
        self.prob_decoder = nn.Linear(dim, 1)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.modal_emb.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_agents(self, data: Dict) -> tuple[torch.Tensor, torch.Tensor]:
        agent_feat = torch.cat(
            [
                data["x"],
                data["x_angles"][..., :50][..., None],
                data["x_velocity"][..., :50][..., None],
            ],
            dim=-1,
        )
        agent_feat = self.tempo_net.proj_layer(agent_feat)
        agent_feat = self.agent_encoder(
            agent_emb=agent_feat,
            centers=data["x_centers"],
            headings=data["x_angles"][..., 49],
            agent_type=data["x_attr"][..., 2],
        )
        B, A, _, D = agent_feat.shape
        agent_temporal_mask = (~data["x_padding_mask"][..., :50]).reshape(B * A, -1)
        agent_feat = agent_feat.reshape(B * A, -1, D)
        agent_feat = self.tempo_net(agent_feat, mask=agent_temporal_mask)
        agent_feat = self.agent_pool(agent_feat, mask=agent_temporal_mask).reshape(
            B, A, D
        )
        agent_mask = ~data["x_key_padding_mask"]
        return agent_feat, agent_mask

    def encode_lanes(self, data: Dict) -> tuple[torch.Tensor, torch.Tensor]:
        lane_feat = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = self.spa_net.proj_layer(lane_feat)
        lane_feat = self.lane_encoder(
            lane_emb=lane_feat,
            centers=data["lane_centers"],
            headings=data["lane_angles"],
            lane_attr=data["lane_attr"],
        )
        B, L, P, D = lane_feat.shape
        lane_temporal_mask = (~data["lane_padding_mask"]).reshape(B * L, P)
        lane_feat = lane_feat.reshape(B * L, P, D)
        lane_feat = self.spa_net(lane_feat, mask=lane_temporal_mask)
        lane_feat = self.lane_pool(lane_feat, mask=lane_temporal_mask).reshape(B, L, D)
        lane_mask = ~data["lane_key_padding_mask"]
        return lane_feat, lane_mask

    def fuse_scene(
        self,
        agent_feat: torch.Tensor,
        lane_feat: torch.Tensor,
        agent_mask: torch.Tensor,
        lane_mask: torch.Tensor,
        agent_centers: torch.Tensor,
        lane_centers: torch.Tensor,
    ) -> torch.Tensor:
        return self.inter_net(
            agent_feat,
            lane_feat,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
            agent_pos=agent_centers,
            lane_pos=lane_centers,
        )

    def decode_predictions(self, agent_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_agents, dim = agent_feat.shape
        modal_feat = (
            agent_feat.unsqueeze(2) + self.modal_emb.weight.unsqueeze(0).unsqueeze(0)
        ).reshape(batch_size * num_agents, self.num_modals, dim)
        for mode_layer in self.mode_layers:
            modal_feat = mode_layer(modal_feat)
        modal_feat = modal_feat.reshape(batch_size, num_agents, self.num_modals, dim)

        y_hat = self.traj_decoder(modal_feat).reshape(
            batch_size,
            num_agents,
            self.num_modals,
            self.future_steps,
            2,
        )
        pi = self.prob_decoder(modal_feat).reshape(
            batch_size,
            num_agents,
            self.num_modals,
        )
        return {"y_hat": y_hat, "pi": pi}

    def forward(self, data: Dict) -> dict[str, torch.Tensor]:
        agent_feat, agent_mask = self.encode_agents(data)
        lane_feat, lane_mask = self.encode_lanes(data)
        fused_agent_feat = self.fuse_scene(
            agent_feat,
            lane_feat,
            agent_mask,
            lane_mask,
            data["x_centers"],
            data["lane_centers"],
        )
        return self.decode_predictions(fused_agent_feat)
