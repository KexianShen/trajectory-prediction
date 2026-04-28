from collections.abc import Callable
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from torchmetrics import Metric

from .utils import sort_predictions


class minADE(Metric):
    """Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k: int = 6,
        dist_sync_on_step: bool = False,
        process_group: ProcessGroup | None = None,
        dist_sync_fn: Callable[..., object] | None = None,
    ) -> None:
        super(minADE, self).__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred, _ = sort_predictions(outputs["y_hat"], outputs["pi"], k=self.k)
            ade = torch.norm(
                pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
            ).mean(-1)
            min_ade = ade.min(-1)[0]
            self.sum += min_ade.sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
