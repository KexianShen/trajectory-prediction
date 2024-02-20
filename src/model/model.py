import torch
import torch.nn as nn

from .layers.inter_net import InterNet, SceneNet
from .layers.spa_net import SpaNet
from .layers.tempo_net import TempoNet


class Model(nn.Module):
    def __init__(
        self,
        agent_dim=4,
        lane_dim=2,
        dim=128,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=8,
        num_modals=6,
        future_steps=60,
    ) -> None:
        super().__init__()
        self.num_modals = num_modals
        self.future_steps = future_steps

        self.tempo_net = TempoNet(
            agent_dim=agent_dim,
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=dim,
            max_t_len=50,
        )
        self.spa_net = SpaNet(
            lane_dim=lane_dim, num_layers=encoder_depth, num_heads=num_heads, dim=dim
        )
        self.scene_net = SceneNet(
            num_layers=encoder_depth, num_heads=num_heads, dim=dim
        )
        self.inter_net = InterNet(
            num_layers=encoder_depth, num_heads=num_heads, dim=dim
        )
        self.agent_pos_embed = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.lane_pos_embed = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.agent_type_embed = nn.Parameter(torch.Tensor(4, dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, dim))
        self.modal_query = nn.Embedding(num_modals, dim)
        self.agent_squeeze = nn.Conv1d(50, 1, 3, 1, 1, bias=True)
        self.lane_squeeze = nn.Conv1d(20, 1, 3, 1, 1, bias=True)

        self.traj_decoder = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, future_steps * 2),
        )
        self.prob_decoder = nn.Linear(dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_pretrain(self, mtm_ckpt_path, mrm_ckpt_path):
        mtm_ckpt = torch.load(mtm_ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.tempo_net.") :]: v
            for k, v in mtm_ckpt.items()
            if k.startswith("net.tempo_net.") and not k.endswith("freqs_cis")
        }
        self.tempo_net.load_state_dict(state_dict=state_dict, strict=False)
        self.agent_type_embed.data.copy_(mtm_ckpt["net.agent_type_embed"])
        state_dict = {
            k[len("net.pos_embed.") :]: v
            for k, v in mtm_ckpt.items()
            if k.startswith("net.pos_embed.")
        }
        self.agent_pos_embed.load_state_dict(state_dict=state_dict, strict=False)

        mrm_ckpt = torch.load(mrm_ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.spa_net.") :]: v
            for k, v in mrm_ckpt.items()
            if k.startswith("net.spa_net.")
        }
        self.spa_net.load_state_dict(state_dict=state_dict, strict=False)
        self.lane_type_embed.data.copy_(mrm_ckpt["net.lane_type_embed"])
        state_dict = {
            k[len("net.pos_embed.") :]: v
            for k, v in mrm_ckpt.items()
            if k.startswith("net.pos_embed.")
        }
        self.lane_pos_embed.load_state_dict(state_dict=state_dict, strict=False)
        return self

    def forward(self, data):
        agent_feat = torch.cat(
            [
                data["x"],
                data["x_angles"][..., :50][..., None],
                data["x_velocity"][..., :50][..., None],
            ],
            dim=-1,
        )
        agent_feat = self.tempo_net.proj_layer(agent_feat)
        agent_pos_feat = torch.cat(
            [
                data["x_centers"],
                torch.cos(data["x_angles"][..., 49][..., None]),
                torch.sin(data["x_angles"][..., 49][..., None]),
            ],
            dim=-1,
        )
        agent_pos_embed = self.agent_pos_embed(agent_pos_feat)
        agent_feat += agent_pos_embed.unsqueeze(2)

        agent_type_embed = self.agent_type_embed[data["x_attr"][..., 2].long()]
        agent_feat += agent_type_embed.unsqueeze(2)
        B, A, T, D = agent_feat.shape
        agent_feat = agent_feat.view(B * A, T, D)
        agent_feat = self.tempo_net(agent_feat)
        agent_feat = self.agent_squeeze(agent_feat).view(B, A, D) + agent_pos_embed

        lane_feat = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = self.spa_net.proj_layer(lane_feat)
        lane_pos_feat = torch.cat(
            [
                data["lane_centers"],
                torch.cos(data["lane_angles"][..., None]),
                torch.sin(data["lane_angles"][..., None]),
            ],
            dim=-1,
        )
        lane_pos_embed = self.lane_pos_embed(lane_pos_feat)
        lane_feat += lane_pos_embed.unsqueeze(2)
        lane_feat += self.lane_type_embed
        B, L, P, D = lane_feat.shape
        lane_feat = lane_feat.view(B * L, P, D)
        lane_feat = self.spa_net(lane_feat)
        lane_feat = self.lane_squeeze(lane_feat).view(B, L, D) + lane_pos_embed

        agent_emb = agent_feat.repeat(1, 1, self.num_modals).view(
            B, A * self.num_modals, D
        )
        modal_query = self.modal_query.weight.unsqueeze(0).repeat(B, A, 1)
        scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        scene_emb = self.scene_net(scene_feat)
        agent_emb = self.inter_net(agent_emb, modal_query, scene_emb)
        y_hat = self.traj_decoder(agent_emb).view(
            B, A, self.num_modals, self.future_steps, 2
        )
        pi = self.prob_decoder(agent_emb).view(B, A, self.num_modals)
        return {"y_hat": y_hat, "pi": pi}
