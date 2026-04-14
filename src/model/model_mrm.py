import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LaneEncoder
from .layers.spa_net import SpaNet
from .modeling_utils import random_token_masking


class ModelMRM(nn.Module):
    def __init__(
        self,
        lane_dim=2,
        embed_dim=128,
        num_lane_type=1,
        encoder_depth=4,
        num_heads=8,
        dropout=0.1,
        norm_eps=1e-5,
        ffn_multiplier=4.0,
        headwise_attn_output_gate=True,
    ) -> None:
        super(ModelMRM, self).__init__()
        self.lane_encoder = LaneEncoder(
            dim=embed_dim,
            num_lane_type=num_lane_type,
        )
        self.spa_net = SpaNet(
            lane_dim=lane_dim,
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=embed_dim,
            dropout=dropout,
            norm_eps=norm_eps,
            ffn_multiplier=ffn_multiplier,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.mrm_mask_token = nn.Parameter(torch.Tensor(1, embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, lane_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
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

    def encode_lanes(self, data):
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized], dim=-1)
        original_lane_feat = lane_feat.clone()
        lane_feat = self.spa_net.proj_layer(lane_feat)
        lane_feat = self.lane_encoder(
            lane_emb=lane_feat,
            centers=data["lane_centers"],
            headings=data["lane_angles"],
        )
        return lane_feat, original_lane_feat

    def forward(self, data):
        lane_padding_mask = ~data["lane_padding_mask"]
        lane_feat, original_lane_feat = self.encode_lanes(data)
        masked_tokens, idx_mask = random_token_masking(
            lane_feat,
            lane_padding_mask,
            self.mrm_mask_token,
            mask_ratio=0.5,
            eligible_threshold=5,
        )
        B, L, P, D = masked_tokens.shape
        masked_tokens = masked_tokens.reshape(B * L, P, D)
        lane_feat = self.spa_net(
            masked_tokens,
            mask=lane_padding_mask.reshape(B * L, P),
        )
        if idx_mask.numel() == 0:
            loss = lane_feat.sum() * 0.0
            return {
                "loss": loss,
                "y_hat": torch.empty(
                    (0, original_lane_feat.size(-1)),
                    device=original_lane_feat.device,
                    dtype=original_lane_feat.dtype,
                ),
                "x": original_lane_feat,
                "idx_mask": idx_mask,
            }
        y_hat = self.decoder(lane_feat).reshape(B, L, P, -1)[
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
