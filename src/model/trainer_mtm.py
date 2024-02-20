import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.optim import WarmupCosLR

from .model_mtm import ModelMTM


class Trainer(pl.LightningModule):
    def __init__(
        self,
        agent_dim=4,
        dim=128,
        encoder_depth=4,
        num_heads=8,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.net = ModelMTM(
            agent_dim=agent_dim,
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
        )

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
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
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
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
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
