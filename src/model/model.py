import torch
import torch.nn as nn

from .layers import InterNet, SpaNet, TempoNet


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
            seq_len=50,
        )
        self.spa_net = SpaNet(
            lane_dim=lane_dim, num_layers=encoder_depth, num_heads=num_heads, dim=dim
        )
        self.inter_net = InterNet(
            num_layers=encoder_depth, num_heads=num_heads, dim=dim
        )
        self.agent_pos_emb = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.lane_pos_emb = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.agent_type_emb = nn.Parameter(torch.Tensor(4, dim))
        self.lane_type_emb = nn.Parameter(torch.Tensor(1, 1, dim))
        self.modal_emb = nn.Parameter(torch.Tensor(num_modals, dim))
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

    # def load_from_pretrain(self, mtm_ckpt_path, mrm_ckpt_path):
    #     mtm_ckpt = torch.load(mtm_ckpt_path, map_location="cpu", weights_only=True)[
    #         "state_dict"
    #     ]
    #     state_dict = {
    #         k[len("net.tempo_net.") :]: v
    #         for k, v in mtm_ckpt.items()
    #         if k.startswith("net.tempo_net.")
    #     }
    #     self.tempo_net.load_state_dict(state_dict=state_dict, strict=True)
    #     state_dict = {
    #         k[len("net.pos_emb.") :]: v
    #         for k, v in mtm_ckpt.items()
    #         if k.startswith("net.pos_emb.")
    #     }
    #     self.agent_pos_emb.load_state_dict(state_dict=state_dict, strict=True)
    #     self.agent_type_emb.data.copy_(mtm_ckpt["net.agent_type_emb"])

    #     mrm_ckpt = torch.load(mrm_ckpt_path, map_location="cpu", weights_only=True)[
    #         "state_dict"
    #     ]
    #     state_dict = {
    #         k[len("net.spa_net.") :]: v
    #         for k, v in mrm_ckpt.items()
    #         if k.startswith("net.spa_net.")
    #     }
    #     self.spa_net.load_state_dict(state_dict=state_dict, strict=True)
    #     self.lane_type_emb.data.copy_(mrm_ckpt["net.lane_type_emb"])
    #     state_dict = {
    #         k[len("net.pos_emb.") :]: v
    #         for k, v in mrm_ckpt.items()
    #         if k.startswith("net.pos_emb.")
    #     }
    #     self.lane_pos_emb.load_state_dict(state_dict=state_dict, strict=True)
    #     return self

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
        agent_pos_emb = self.agent_pos_emb(agent_pos_feat)
        agent_feat += agent_pos_emb.unsqueeze(2)

        agent_type_emb = self.agent_type_emb[data["x_attr"][..., 2].long()]
        agent_feat += agent_type_emb.unsqueeze(2)
        B, A, _, D = agent_feat.shape
        agent_feat = agent_feat.view(B * A, -1, D)
        agent_feat = self.tempo_net(agent_feat)
        agent_feat = self.agent_squeeze(agent_feat).view(B, A, 1, D)
        modal_emb = self.modal_emb.unsqueeze(0).unsqueeze(0)
        agent_feat = agent_feat + modal_emb
        agent_feat = agent_feat.view(B, A * self.num_modals, D)

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
        lane_pos_emb = self.lane_pos_emb(lane_pos_feat)
        lane_feat += lane_pos_emb.unsqueeze(2)
        lane_feat += self.lane_type_emb
        B, L, P, D = lane_feat.shape
        lane_feat = lane_feat.view(B * L, P, D)
        lane_feat = self.spa_net(lane_feat)
        lane_feat = self.lane_squeeze(lane_feat).view(B, L, D)

        agent_feat = self.inter_net(agent_feat, lane_feat)
        y_hat = self.traj_decoder(agent_feat).view(
            B, A, self.num_modals, self.future_steps, 2
        )
        pi = self.prob_decoder(agent_feat).view(B, A, self.num_modals)
        return {"y_hat": y_hat, "pi": pi}
