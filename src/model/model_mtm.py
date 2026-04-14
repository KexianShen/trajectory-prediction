import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AgentEncoder
from .layers.tempo_net import TempoNet
from .modeling_utils import random_token_masking


class ModelMTM(nn.Module):
    def __init__(
        self,
        agent_dim=4,
        embed_dim=128,
        num_agent_type=4,
        encoder_depth=4,
        num_heads=8,
        dropout=0.1,
        norm_eps=1e-5,
        ffn_multiplier=4.0,
        headwise_attn_output_gate=True,
    ) -> None:
        super(ModelMTM, self).__init__()
        self.agent_encoder = AgentEncoder(
            dim=embed_dim,
            num_agent_type=num_agent_type,
        )
        self.tempo_net = TempoNet(
            agent_dim=agent_dim,
            num_layers=encoder_depth,
            num_heads=num_heads,
            dim=embed_dim,
            dropout=dropout,
            norm_eps=norm_eps,
            ffn_multiplier=ffn_multiplier,
            seq_len=110,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
        self.mtm_mask_token = nn.Parameter(torch.Tensor(1, embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, agent_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
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

    def encode_agents(self, data):
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
        agent_feat = self.agent_encoder(
            agent_emb=agent_feat,
            centers=data["x_centers"],
            headings=data["x_angles"][..., 49],
            agent_type=data["x_attr"][..., 2],
        )
        return agent_feat, original_agent_feat

    def forward(self, data):
        agent_padding_mask = ~data["x_padding_mask"]
        agent_feat, original_agent_feat = self.encode_agents(data)
        masked_tokens, idx_mask = random_token_masking(
            agent_feat,
            agent_padding_mask,
            self.mtm_mask_token,
            mask_ratio=0.6,
            eligible_threshold=5,
        )
        B, A, T, D = masked_tokens.shape
        masked_tokens = masked_tokens.reshape(B * A, T, D)
        agent_feat = self.tempo_net(
            masked_tokens,
            mask=agent_padding_mask.reshape(B * A, T),
        )
        if idx_mask.numel() == 0:
            loss = agent_feat.sum() * 0.0
            return {
                "loss": loss,
                "y_hat": torch.empty(
                    (0, original_agent_feat.size(-1)),
                    device=original_agent_feat.device,
                    dtype=original_agent_feat.dtype,
                ),
                "x": original_agent_feat,
                "idx_mask": idx_mask,
            }
        y_hat = self.decoder(agent_feat).reshape(B, A, T, -1)[
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
