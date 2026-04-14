import torch

from .sequence import SequenceEncoder


class SpaNet(SequenceEncoder):
    def __init__(
        self,
        lane_dim: int = 2,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        seq_len: int = 20,
        headwise_attn_output_gate: bool = True,
    ) -> None:
        super().__init__(
            input_dim=lane_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            norm_eps=norm_eps,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            seq_len=seq_len,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )

    def forward(
        self,
        map_feat: torch.Tensor,
        agent_feat: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if agent_feat is not None:
            x = torch.cat((agent_feat, map_feat), dim=1)
        else:
            x = map_feat
        return super().forward(x, mask=mask)
