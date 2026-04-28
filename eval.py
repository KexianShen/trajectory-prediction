import os
from importlib import import_module

import hydra
from hydra.utils import instantiate, to_absolute_path
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf: DictConfig) -> None:
    seed_everything(conf.seed)

    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    model_path = conf.model.target._target_
    module = import_module(model_path[: model_path.rfind(".")])
    Model: LightningModule = getattr(module, model_path[model_path.rfind(".") + 1 :])
    model = Model.load_from_checkpoint(checkpoint, weights_only=False)

    trainer = Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: LightningDataModule = instantiate(conf.datamodule, test=conf.test)

    if not conf.test:
        trainer.validate(model, datamodule)
    else:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
