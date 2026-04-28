import math
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        lr: float,
        warmup_epochs: int,
        epochs: int,
        last_epoch: int = -1,
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self) -> float:
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]


def build_weight_decay_groups(
    module: nn.Module,
    weight_decay: float,
    extra_no_decay_keywords: tuple[str, ...] = (),
) -> list[dict[str, float | list[nn.Parameter]]]:
    decay = []
    no_decay = []

    def no_weight_decay(name: str) -> bool:
        name = name.lower()
        if name.endswith("bias"):
            return True
        no_decay_keywords = (
            "norm",
            "emb",
            "pos",
            "gate",
            "mask",
            "alpha",
        ) + extra_no_decay_keywords
        return any(keyword in name for keyword in no_decay_keywords)

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if no_weight_decay(name):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]
