import torch
import torch.nn as nn

from .layers import AgentEncoder, InterNet, LaneEncoder, SpaNet, TempoNet


class Model(nn.Module):
    def __init__(
        self,
        agent_dim=4,
        lane_dim=2,
        num_agent_type=4,
        num_lane_type=1,
        dim=128,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=8,
        dropout=0.1,
        norm_eps=1e-5,
        ffn_multiplier=4.0,
        headwise_attn_output_gate=True,
        num_modals=6,
        future_steps=60,
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
            ffn_multiplier=ffn_multiplier,
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
            ffn_multiplier=ffn_multiplier,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.inter_net = InterNet(
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=dim,
            dropout=dropout,
            norm_eps=norm_eps,
            ffn_multiplier=ffn_multiplier,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.modal_emb = nn.Embedding(num_modals, dim)
        self.agent_squeeze = nn.Conv1d(50, 1, 3, 1, 1, bias=True)
        self.lane_squeeze = nn.Conv1d(20, 1, 3, 1, 1, bias=True)

        self.traj_decoder = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, future_steps * 2),
        )
        self.prob_decoder = nn.Linear(dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.modal_emb.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_agents(self, data):
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
        agent_feat = self.agent_squeeze(agent_feat).reshape(B, A, 1, D)
        modal_emb = self.modal_emb.weight.unsqueeze(0).unsqueeze(0)
        agent_feat = agent_feat + modal_emb
        agent_mask = (~data["x_key_padding_mask"]).repeat_interleave(
            self.num_modals, dim=1
        )
        return agent_feat.reshape(B, A * self.num_modals, D), agent_mask

    def encode_lanes(self, data):
        lane_feat = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = self.spa_net.proj_layer(lane_feat)
        lane_feat = self.lane_encoder(
            lane_emb=lane_feat,
            centers=data["lane_centers"],
            headings=data["lane_angles"],
        )
        B, L, P, D = lane_feat.shape
        lane_temporal_mask = (~data["lane_padding_mask"]).reshape(B * L, P)
        lane_feat = lane_feat.reshape(B * L, P, D)
        lane_feat = self.spa_net(lane_feat, mask=lane_temporal_mask)
        lane_feat = self.lane_squeeze(lane_feat).reshape(B, L, D)
        lane_mask = ~data["lane_key_padding_mask"]
        return lane_feat, lane_mask

    def fuse_scene(self, agent_feat, lane_feat, agent_mask, lane_mask):
        return self.inter_net(
            agent_feat,
            lane_feat,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )

    def decode_predictions(self, agent_feat):
        batch_size, total_agent_modes, _ = agent_feat.shape
        y_hat = self.traj_decoder(agent_feat).reshape(
            batch_size,
            total_agent_modes // self.num_modals,
            self.num_modals,
            self.future_steps,
            2,
        )
        pi = self.prob_decoder(agent_feat).reshape(
            batch_size,
            total_agent_modes // self.num_modals,
            self.num_modals,
        )
        return {"y_hat": y_hat, "pi": pi}

    def forward(self, data):
        agent_feat, agent_mask = self.encode_agents(data)
        lane_feat, lane_mask = self.encode_lanes(data)
        fused_agent_feat = self.fuse_scene(agent_feat, lane_feat, agent_mask, lane_mask)
        return self.decode_predictions(fused_agent_feat)
