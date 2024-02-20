from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from .utils import sort_multi_predictions


class AvgMinADE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(AvgMinADE, self).__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        scored_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            y_hat = sort_multi_predictions(outputs["y_hat"], outputs["pi"], k=self.k)[
                0
            ].transpose(1, 2)
            B, K, A, T, _ = y_hat.shape
            valid_mask = scored_mask.unsqueeze(1).float()  # [B, 1, N]
            num_valid_agents = valid_mask.sum(-1)
            avg_ade = (
                torch.norm(y_hat[..., :2] - target.unsqueeze(1)[..., :2], dim=-1).mean(
                    dim=-1
                )
                * valid_mask
            ).sum(-1) / num_valid_agents
            avg_min_ade = torch.min(avg_ade, dim=-1)[0]

            self.sum += avg_min_ade.sum()
            self.count += B

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
