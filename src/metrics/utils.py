import torch


def sort_predictions(
    predictions: torch.Tensor, probability: torch.Tensor, k: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, k', t, 2].
    """
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = probability[torch.arange(probability.size(0))[:, None], indices]
    sorted_predictions = predictions.gather(
        1,
        indices[..., None, None].expand(
            -1, -1, predictions.size(-2), predictions.size(-1)
        ),
    )
    return sorted_predictions[:, :k], sorted_prob[:, :k]


def sort_multi_predictions(
    predictions: torch.Tensor, probability: torch.Tensor, k: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, a, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, a, k].
    Returns:
        torch.Tensor: The sorted predictions [b, a, k', t, 2].
    """
    B, A, _, _, _ = predictions.shape
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = torch.gather(probability, -1, indices)
    sorted_predictions = predictions.gather(
        2,
        indices[..., None, None].expand(
            -1, -1, -1, predictions.size(-2), predictions.size(-1)
        ),
    )
    return sorted_predictions[:, :, :k], sorted_prob[:, :, :k]
