from collections.abc import Callable
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from torchmetrics import Metric

from .utils import sort_multi_predictions


class ActorMR(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k: int = 6,
        miss_threshold: float = 2.0,
        dist_sync_on_step: bool = False,
        process_group: ProcessGroup | None = None,
        dist_sync_fn: Callable[..., object] | None = None,
    ) -> None:
        super(ActorMR, self).__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.miss_threshold = miss_threshold

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        scored_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            y_hat = sort_multi_predictions(outputs["y_hat"], outputs["pi"], k=self.k)[
                0
            ].transpose(1, 2)
            B, K, A, T, _ = y_hat.shape
            valid_mask = scored_mask.unsqueeze(1).float()  # [B, 1, N]
            num_scored_agents = valid_mask.long().sum(-1)
            fde = torch.norm(
                y_hat[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], dim=-1
            )
            avg_fde = (fde * valid_mask).sum(-1) / num_scored_agents

            best_world = torch.argmin(avg_fde, dim=-1)
            best_world_fde = fde[torch.arange(B), best_world]
            missed_predictions = best_world_fde > self.miss_threshold
            missed_predictions[~scored_mask] = False

            self.sum += missed_predictions.sum()
            self.count += num_scored_agents.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
