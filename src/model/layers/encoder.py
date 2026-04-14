import torch
from torch import nn

from .attn import FourierEmbedding


class AgentEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_agent_type: int = 4,
    ) -> None:
        super().__init__()
        self.pos_emb = FourierEmbedding(input_dim=4, dim=dim)
        self.type_emb = nn.Embedding(num_agent_type, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_emb.weight, std=0.02)

    def forward(
        self,
        agent_emb: torch.Tensor,
        centers: torch.Tensor,
        headings: torch.Tensor,
        agent_type: torch.Tensor,
    ) -> torch.Tensor:
        pos_feat = torch.cat(
            [centers, headings.cos().unsqueeze(-1), headings.sin().unsqueeze(-1)],
            dim=-1,
        )
        agent_emb = agent_emb + self.pos_emb(pos_feat).unsqueeze(2)
        agent_emb = agent_emb + self.type_emb(agent_type.long()).unsqueeze(2)
        return agent_emb


class LaneEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_lane_type: int = 1,
    ) -> None:
        super().__init__()
        self.pos_emb = FourierEmbedding(input_dim=4, dim=dim)
        self.type_emb = nn.Embedding(num_lane_type, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_emb.weight, std=0.02)

    def forward(
        self,
        lane_emb: torch.Tensor,
        centers: torch.Tensor,
        headings: torch.Tensor,
    ) -> torch.Tensor:
        pos_feat = torch.cat(
            [centers, headings.cos().unsqueeze(-1), headings.sin().unsqueeze(-1)],
            dim=-1,
        )
        lane_type = torch.zeros(
            centers.shape[:2], device=centers.device, dtype=torch.long
        )
        lane_emb = lane_emb + self.pos_emb(pos_feat).unsqueeze(2)
        lane_emb = lane_emb + self.type_emb(lane_type).unsqueeze(2)
        return lane_emb
