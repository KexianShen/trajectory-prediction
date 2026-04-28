import time
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torchmetrics import MetricCollection

from src.metrics import ActorMR, AvgMinADE, AvgMinFDE
from src.utils.optim import WarmupCosLR, build_weight_decay_groups
from src.utils.submission_av2_multiagent import SubmissionAv2MultiAgent

from .model import Model


class Trainer(LightningModule):
    def __init__(self, net: DictConfig, optim: DictConfig) -> None:
        super(Trainer, self).__init__()
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2MultiAgent()

        self.net: Model = hydra.utils.instantiate(self.hparams.net)

        metrics = MetricCollection([AvgMinADE(), AvgMinFDE(), ActorMR()])
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, data: Dict) -> dict[str, torch.Tensor]:
        return self.net(data)

    def predict(self, data: Dict) -> np.ndarray:
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

    def cal_loss(
        self, outputs: dict[str, torch.Tensor], data: Dict
    ) -> dict[str, torch.Tensor | float]:
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

        with torch.no_grad():
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

    def training_step(self, data: Dict, batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, data: Dict, batch_idx: int) -> None:
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

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        val_AvgMinADE = self.trainer.callback_metrics.get("val/AvgMinADE")
        if val_AvgMinADE is not None:
            self.logger.log_metrics(
                {"hp_metric": val_AvgMinADE}, step=self.current_epoch
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        file_stem = self._build_submission_stem()
        self.submission_handler = SubmissionAv2MultiAgent(
            save_dir=save_dir,
            file_stem=file_stem,
        )

    def test_step(self, data: Dict, batch_idx: int) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        rank = self.global_rank if self.trainer.world_size > 1 else 0
        self.submission_handler.save_partial_file(rank)
        self._barrier()
        if self.trainer.is_global_zero:
            self.submission_handler.merge_partial_files(
                world_size=self.trainer.world_size,
            )
        self._barrier()

    def _build_submission_stem(self) -> str:
        file_stem = f"multi_agent_{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}"
        if dist.is_available() and dist.is_initialized():
            shared = [file_stem if self.trainer.is_global_zero else None]
            dist.broadcast_object_list(shared, src=0)
            file_stem = shared[0]
        return file_stem

    def _barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def configure_optimizers(self) -> tuple[list[Optimizer], list[WarmupCosLR]]:
        optim_groups = build_weight_decay_groups(
            self, weight_decay=self.hparams.optim.weight_decay
        )
        optimizer = AdamW(
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
