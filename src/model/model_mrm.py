import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.spa_net import SpaNet


class ModelMRM(nn.Module):
    def __init__(
        self,
        lane_dim=2,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
    ) -> None:
        super(ModelMRM, self).__init__()
        self.spa_net = SpaNet(
            lane_dim=lane_dim, num_layers=encoder_depth, num_heads=num_heads
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.mrm_mask_token = nn.Parameter(torch.Tensor(1, embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, lane_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.mrm_mask_token, std=0.02)

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
    def lane_random_masking(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mask_token: torch.Tensor,
        mask_ratio=0.5,
        eligible_threshold=5,
    ):
        """
        Args:
            tokens: (BLtch_size, num_lane, num_point, embed_dim)
            mask: (BLtch_size, seq_len)
            mask_ratio: float
            eligible_threshold: int
        Returns:
            masked_tokens: (BLtch_size, seq_len, embed_dim)
        """
        masked_tokens = tokens.clone()
        num_B_L = mask.sum(-1)
        mask_B_L = num_B_L > eligible_threshold
        idx_B_L = mask_B_L.nonzero()
        mask_BL_P = mask[mask_B_L]
        num_mask_BL = (num_B_L[mask_B_L] * mask_ratio).int()
        idx_mask = []
        for i in range(idx_B_L.size(0)):
            idx_T = torch.nonzero(mask_BL_P[i] == True)
            idx = torch.cat(
                [
                    idx_B_L[i].repeat(num_mask_BL[i], 1),
                    idx_T[torch.randperm(idx_T.size(0))[: num_mask_BL[i]]],
                ],
                dim=-1,
            )
            idx_mask.append(idx)
        idx_mask = torch.cat(idx_mask, dim=0)
        masked_tokens[idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]] = mask_token
        return masked_tokens, idx_mask

    def forward(self, data):
        lane_padding_mask = ~data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)

        lane_feat = torch.cat([lane_normalized], dim=-1)
        original_lane_feat = lane_feat.clone()
        lane_feat = self.spa_net.proj_layer(lane_feat)
        pos_feat = torch.cat(
            [
                data["lane_centers"],
                torch.cos(data["lane_angles"][..., None]),
                torch.sin(data["lane_angles"][..., None]),
            ],
            dim=-1,
        )
        pos_embed = self.pos_embed(pos_feat)
        lane_feat += pos_embed.unsqueeze(2)
        lane_feat += self.lane_type_embed

        masked_tokens, idx_mask = self.lane_random_masking(
            lane_feat, lane_padding_mask, self.mrm_mask_token
        )
        B, L, P, D = masked_tokens.shape
        masked_tokens = masked_tokens.view(B * L, P, D)
        lane_feat = self.spa_net(masked_tokens)
        y_hat = self.decoder(lane_feat).view(B, L, P, -1)[
            idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]
        ]

        loss = F.mse_loss(
            y_hat, original_lane_feat[idx_mask[:, 0], idx_mask[:, 1], idx_mask[:, 2]]
        )
        return {
            "loss": loss,
            "y_hat": y_hat,
            "x": original_lane_feat,
            "idx_mask": idx_mask,
        }
