from typing import Tuple, Type

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    activation: Type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    )


def empty_index_mask(tokens: torch.Tensor) -> torch.Tensor:
    return torch.empty((0, 3), dtype=torch.long, device=tokens.device)


def random_token_masking(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token: torch.Tensor,
    mask_ratio: float,
    eligible_threshold: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_tokens = tokens.clone()
    valid_count = valid_mask.sum(-1)
    eligible = valid_count > eligible_threshold
    if not eligible.any():
        return masked_tokens, empty_index_mask(tokens)

    eligible_index = eligible.nonzero(as_tuple=False)
    eligible_mask = valid_mask[eligible]
    num_to_mask = (valid_count[eligible] * mask_ratio).int()
    masked_index = []

    for row in range(eligible_index.size(0)):
        current_num_to_mask = num_to_mask[row]
        if current_num_to_mask == 0:
            continue
        token_index = torch.nonzero(eligible_mask[row], as_tuple=False)
        token_index = token_index[
            torch.randperm(token_index.size(0), device=token_index.device)[
                :current_num_to_mask
            ]
        ]
        masked_index.append(
            torch.cat(
                [eligible_index[row].repeat(current_num_to_mask, 1), token_index],
                dim=-1,
            )
        )

    if not masked_index:
        return masked_tokens, empty_index_mask(tokens)

    masked_index = torch.cat(masked_index, dim=0)
    masked_tokens[masked_index[:, 0], masked_index[:, 1], masked_index[:, 2]] = (
        mask_token.type_as(masked_tokens)
    )
    return masked_tokens, masked_index
