import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.optim import WarmupCosLR, build_weight_decay_groups

from .model_mtm import ModelMTM


class Trainer(pl.LightningModule):
    def __init__(
        self,
        net: DictConfig,
        optim: DictConfig,
    ) -> None:
        super(Trainer, self).__init__()
        self.save_hyperparameters()

        self.net: ModelMTM = hydra.utils.instantiate(self.hparams.net)

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        last_position = data["x_positions"][:, :, -1, :].view(1, -1, 1, 2)
        origin = data["origin"].view(1, 1, 1, 2).double()
        theta = data["theta"].double()

        global_pos = out["x"]
        idx_mask = out["idx_mask"]

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).view(1, 1, 2, 2)
        with torch.no_grad():
            global_pos = (
                (
                    torch.matmul(
                        global_pos[..., :2].double() + last_position, rotate_mat
                    )
                    + origin
                )
                .cpu()
                .numpy()
            )
            predict_pos = (
                (
                    torch.matmul(
                        out["y_hat"][..., :2].double()
                        + last_position[0, idx_mask[:, 1], 0, :],
                        rotate_mat,
                    )
                    + origin
                )
                .cpu()
                .numpy()
            )
        return global_pos, predict_pos

    def training_step(self, data, batch_idx):
        out = self(data)

        self.log(
            f"train/loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=out["x"].size(0),
        )

        return out["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)

        self.log(
            "val/loss",
            out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=out["x"].size(0),
        )

    def configure_optimizers(self):
        optim_groups = build_weight_decay_groups(
            self,
            weight_decay=self.hparams.optim.weight_decay,
        )

        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=self.hparams.optim.lr,
            betas=(self.hparams.optim.beta[0], self.hparams.optim.beta[1]),
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.hparams.optim.lr,
            min_lr=self.hparams.optim.min_lr,
            warmup_epochs=self.hparams.optim.warmup_epochs,
            epochs=self.hparams.optim.epochs,
        )
        return [optimizer], [scheduler]
