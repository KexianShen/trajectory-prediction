from .sequence import SequenceEncoder


class TempoNet(SequenceEncoder):
    def __init__(
        self,
        agent_dim: int = 4,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        seq_len: int = 110,
        headwise_attn_output_gate: bool = True,
    ) -> None:
        super().__init__(
            input_dim=agent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            norm_eps=norm_eps,
            dropout=dropout,
            seq_len=seq_len,
            headwise_attn_output_gate=headwise_attn_output_gate,
        )
