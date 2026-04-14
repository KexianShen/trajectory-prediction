from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from src.metrics import ActorMR, AvgMinADE, AvgMinFDE
from src.utils.optim import WarmupCosLR, build_weight_decay_groups
from src.utils.submission_av2_multiagent import SubmissionAv2MultiAgent

from .model import Model


class Trainer(pl.LightningModule):
    def __init__(
        self,
        net: DictConfig,
        optim: DictConfig,
        mtm_checkpoint: str = None,
        mrm_checkpoint: str = None,
        fine_tuning_ratio: float = 1e-3,
    ) -> None:
        super(Trainer, self).__init__()
        self.fine_tuning_ratio = fine_tuning_ratio
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2MultiAgent()

        self.net: Model = hydra.utils.instantiate(self.hparams.net)

        # if mtm_checkpoint is not None and mrm_checkpoint is not None:
        #     self.net.load_from_pretrain(mtm_checkpoint, mrm_checkpoint)

        metrics = MetricCollection([AvgMinADE(), AvgMinFDE(), ActorMR()])
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        last_position = data["x_positions"][:, :, -1, :].view(1, -1, 1, 2)
        origin = data["origin"].view(1, 1, 1, 2).double()
        theta = data["theta"].double()
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
            predict_pos = (
                (
                    torch.matmul(
                        out["y_hat"][..., :2].double() + last_position.unsqueeze(2),
                        rotate_mat.unsqueeze(2),
                    )
                    + origin.unsqueeze(2)
                )
                .cpu()
                .numpy()
            )
        return predict_pos

    def cal_loss(self, outputs, data):
        y_hat, pi = outputs["y_hat"], outputs["pi"]
        x_scored, y, y_padding_mask = (
            data["x_scored"],
            data["y"],
            data["x_padding_mask"][..., 50:],
        )

        # only consider scored agents
        valid_mask = ~y_padding_mask
        valid_mask[~x_scored] = False
        valid_mask = valid_mask.unsqueeze(2).float()

        l2_norm = (
            torch.norm(y_hat[..., :2] - y.unsqueeze(2), dim=-1) * valid_mask
        ).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)

        reg_mask = ~y_padding_mask
        y_hat_best = y_hat[
            torch.arange(y_hat.shape[0]).unsqueeze(1),
            torch.arange(y_hat.shape[1]).unsqueeze(0),
            best_mode,
            :,
            :,
        ]

        reg_loss = F.smooth_l1_loss(y_hat_best[reg_mask], y[reg_mask])
        cls_loss = F.cross_entropy(
            pi.view(-1, pi.size(-1)), best_mode.view(-1).detach()
        )

        loss = reg_loss + cls_loss
        out = {
            "loss": loss,
            "reg_loss": reg_loss.item(),
            "cls_loss": cls_loss.item(),
        }

        return out

    def training_step(self, data, batch_idx):
        out = self(data)
        res = self.cal_loss(out, data)

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=out["y_hat"].shape[0],
                )

        return res["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        res = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"], data["x_scored"])

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"val/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=out["y_hat"].shape[0],
                )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2MultiAgent(save_dir=save_dir)

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

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
