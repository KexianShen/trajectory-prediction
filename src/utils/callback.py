from pytorch_lightning.callbacks import Callback


class FindUnusedParams(Callback):
    def __init__(self):
        super(FindUnusedParams, self).__init__()

    def on_after_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print("Finding unused parameters ...")
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                print(name)
        print("Finding unused parameters done")
