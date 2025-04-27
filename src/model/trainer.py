from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import ActorMR, AvgMinADE, AvgMinFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2_multiagent import SubmissionAv2MultiAgent

from .layers.attn import MHCA, MHSA, FeedForward, RMSNorm
from .layers.rope_attn import RoPEMHSA
from .model import Model


class Trainer(pl.LightningModule):
    def __init__(
        self,
        agent_dim=4,
        lane_dim=2,
        dim=128,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=8,
        num_modals=6,
        mtm_checkpoint: str = None,
        mrm_checkpoint: str = None,
        lr: float = 1e-3,
        fine_tuning_ratio: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.fine_tuning_ratio = fine_tuning_ratio
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2MultiAgent()

        self.net = Model(
            agent_dim=agent_dim,
            lane_dim=lane_dim,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            num_modals=num_modals,
        )

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
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            MHCA,
            MHSA,
            FeedForward,
            RoPEMHSA,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            RMSNorm,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "lr": self.lr,
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
