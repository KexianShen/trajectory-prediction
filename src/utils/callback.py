import lightning as L
from lightning import Callback


class FindUnusedParams(Callback):
    def __init__(self) -> None:
        super(FindUnusedParams, self).__init__()

    def on_after_backward(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        print("Finding unused parameters ...")
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                print(name)
        print("Finding unused parameters done")
