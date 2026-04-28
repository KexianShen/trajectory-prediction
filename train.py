import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src.utils.callback import FindUnusedParams


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if conf.fast_dev_run:
        callbacks.append(FindUnusedParams())

    trainer = Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        precision=conf.precision,
        strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else "auto",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
        fast_dev_run=conf.fast_dev_run,
    )

    model = instantiate(conf.model.target, _recursive_=False)
    datamodule = instantiate(conf.datamodule)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint, weights_only=False)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # maximize Tensor Core usage
    main()
