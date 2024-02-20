from pathlib import Path
from typing import List

import hydra
import ray

from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.utils.ray_utils import ActorHandle, ProgressBar

ray.init(num_cpus=16)


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


@ray.remote
def preprocess_batch(
    extractor: Av2ExtractorMultiAgent, file_list: List[Path], pb: ActorHandle
):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    batch = 50
    data_root = Path(conf.data_root)

    for mode in ["train", "val", "test"]:
        save_dir = data_root / conf.model.name / mode
        extractor = Av2ExtractorMultiAgent(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
        pb_actor = pb.actor

        for i in range(0, len(scenario_files), batch):
            preprocess_batch.remote(extractor, scenario_files[i : i + batch], pb_actor)

        pb.print_until_done()


if __name__ == "__main__":
    main()
