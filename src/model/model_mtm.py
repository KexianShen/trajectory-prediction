import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.tempo_net import TempoNet


class ModelMTM(nn.Module):
    def __init__(
        self,
        agent_dim=4,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
    ) -> None:
        super(ModelMTM, self).__init__()
        self.tempo_net = TempoNet(
            agent_dim=agent_dim, num_layers=encoder_depth, num_heads=num_heads
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.agent_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.mtm_mask_token = nn.Parameter(torch.Tensor(1, embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, agent_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.agent_type_embed, std=0.02)
        nn.init.normal_(self.mtm_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    @staticmethod
    def agent_random_masking(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mask_token: torch.Tensor,
        mask_ratio=0.4,
        eligible_threshold=5,
    ):
        """
        Args:
            tokens: (batch_size, num_agent, num_time, embed_dim)
            mask: (batch_size, seq_len)
            mask_ratio: float
            eligible_threshold: int
        Returns:
            masked_tokens: (batch_size, seq_len, embed_dim)
        """
        masked_tokens = tokens.clone()
        num_B_A = mask.sum(-1)
        mask_B_A = num_B_A > eligible_threshold
        idx_B_A = mask_B_A.nonzero()
        mask_BA_T = mask[mask_B_A]
        num_mask_BA = (num_B_A[mask_B_A] * mask_ratio).int()
        idx_mask = []
        for i in range(idx_B_A.size(0)):
            idx_T = torch.nonzero(mask_BA_T[i] == True)
            idx = torch.cat(
                [
                    idx_B_A[i].repeat(num_mask_BA[i], 1),
                    idx_T[torch.randperm(idx_T.size(0))[: num_mask_BA[i]]],
                ],
                dim=-1,
            )
            idx_mask.append(idx)
        idx_mask = torch.cat(idx_mask, dim=0)
        masked_tokens[idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]] = mask_token
        return masked_tokens, idx_mask

    def forward(self, data):
        agent_padding_mask = ~data["x_padding_mask"]
        agent_feat = torch.cat(
            [
                torch.cat([data["x"], data["y"]], dim=2),
                data["x_angles"][..., None],
                data["x_velocity"][..., None],
            ],
            dim=-1,
        )
        original_agent_feat = agent_feat.clone()
        agent_feat = self.tempo_net.proj_layer(agent_feat)
        pos_feat = torch.cat(
            [
                data["x_centers"],
                torch.cos(data["x_angles"][..., 49][..., None]),
                torch.sin(data["x_angles"][..., 49][..., None]),
            ],
            dim=-1,
        )
        pos_embed = self.pos_embed(pos_feat)
        agent_feat += pos_embed.unsqueeze(2)

        agent_type_embed = self.agent_type_embed[data["x_attr"][..., 2].long()]
        agent_feat += agent_type_embed.unsqueeze(2)

        masked_tokens, idx_mask = self.agent_random_masking(
            agent_feat, agent_padding_mask, self.mtm_mask_token
        )
        B, A, T, D = masked_tokens.shape
        masked_tokens = masked_tokens.view(B * A, T, D)
        agent_feat = self.tempo_net(masked_tokens)
        y_hat = self.decoder(agent_feat).view(B, A, T, -1)[
            idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]
        ]

        loss = F.mse_loss(
            y_hat, original_agent_feat[idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]]
        )
        return {
            "loss": loss,
            "y_hat": y_hat,
            "x": original_agent_feat,
            "idx_mask": idx_mask,
        }
