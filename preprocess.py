from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig

from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.utils.ray_utils import ActorHandle, ProgressBar


def glob_files(data_root: Path, mode: str) -> list[Path]:
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


@ray.remote
def preprocess_batch(
    extractor: Av2ExtractorMultiAgent, file_list: list[Path], pb: ActorHandle
) -> None:
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    if not ray.is_initialized():
        ray.init(num_cpus=conf.preprocess_num_cpus)

    batch = conf.preprocess_batch_size
    data_root = Path(conf.data_root)

    for mode in ["train", "val", "test"]:
        save_dir = data_root / conf.model.name / mode
        extractor = Av2ExtractorMultiAgent(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
        pb_actor = pb.actor

        tasks = []
        for i in range(0, len(scenario_files), batch):
            tasks.append(
                preprocess_batch.remote(
                    extractor, scenario_files[i : i + batch], pb_actor
                )
            )

        pb.print_until_done()
        ray.get(tasks)

    ray.shutdown()


if __name__ == "__main__":
    main()
