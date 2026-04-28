from importlib import import_module
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from lightning import LightningModule
from omegaconf import DictConfig

from src.datamodule.av2_dataset import collate_fn
from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent

SceneData = dict[str, object]


def glob_files(data_root: Path, split: str) -> list[Path]:
    return sorted((data_root / split).rglob("*.parquet"))


def select_scenario_files(
    data_root: Path,
    split: str,
    num_scenes: int,
) -> list[Path]:
    scenario_files = glob_files(data_root, split)
    if not scenario_files:
        raise FileNotFoundError(f"No parquet files found under {data_root / split}")
    return scenario_files[:num_scenes]


def load_model(conf: DictConfig) -> LightningModule:
    checkpoint = Path(to_absolute_path(conf.checkpoint))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist")

    model_path = conf.model.target._target_
    module = import_module(model_path[: model_path.rfind(".")])
    model_cls = getattr(module, model_path[model_path.rfind(".") + 1 :])
    model = model_cls.load_from_checkpoint(str(checkpoint), weights_only=False)
    model.eval()
    model.freeze()
    return model


def move_batch_to_device(
    batch: SceneData,
    device: torch.device,
) -> SceneData:
    moved: SceneData = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def local_points_to_world(
    points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> np.ndarray:
    if points.numel() == 0:
        return np.empty((0, 2), dtype=np.float64)

    angle = float(theta.item())
    rotate_mat = torch.tensor(
        [
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ],
        dtype=torch.double,
        device=points.device,
    )
    world = torch.matmul(points.double(), rotate_mat) + origin.double()
    return world.cpu().numpy()


def decode_world_predictions(
    outputs: dict[str, torch.Tensor],
    batch: SceneData,
) -> tuple[np.ndarray, np.ndarray]:
    local_pred = outputs["y_hat"][..., :2].double()
    prob = torch.softmax(outputs["pi"], dim=-1).detach().cpu().numpy()[0]

    if local_pred.shape[0] != 1:
        raise ValueError("visualize.py expects batch size 1 during rendering")

    last_position = batch["x_positions"][:, :, -1, :].double().view(1, -1, 1, 2)
    origin = batch["origin"].double().view(1, 1, 1, 2)
    theta = batch["theta"].double()
    rotate_mat = torch.stack(
        [
            torch.cos(theta),
            torch.sin(theta),
            -torch.sin(theta),
            torch.cos(theta),
        ],
        dim=1,
    ).view(1, 1, 2, 2)

    world_pred = torch.matmul(
        local_pred + last_position.unsqueeze(2), rotate_mat.unsqueeze(2)
    ) + origin.unsqueeze(2)
    return world_pred.detach().cpu().numpy()[0], prob


def compute_plot_bounds(
    collected: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid = [points for points in collected if points.size]
    if not valid:
        return (-20.0, 20.0), (-20.0, 20.0)

    stacked = np.concatenate(valid, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    margin = 10.0
    return (x_min - margin, x_max + margin), (y_min - margin, y_max + margin)


def plot_scenario(
    sample: SceneData,
    world_pred: np.ndarray,
    prob: np.ndarray,
    output_path: Path,
    num_modes: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    origin = sample["origin"][0]
    theta = sample["theta"]
    av_index = int(sample["av_index"].item())
    plot_points: list[np.ndarray] = []

    lane_positions = sample["lane_positions"]
    lane_padding = sample["lane_padding_mask"]
    for lane_idx in range(lane_positions.shape[0]):
        valid_mask = ~lane_padding[lane_idx]
        if not valid_mask.any():
            continue
        lane_world = local_points_to_world(
            lane_positions[lane_idx][valid_mask], origin, theta
        )
        plot_points.append(lane_world)
        ax.plot(
            lane_world[:, 0], lane_world[:, 1], color="#c7c7c7", linewidth=1.0, zorder=1
        )

    prediction_labels_used = False
    future_labels_used = False
    history_labels_used = False

    for actor_idx, track_id in enumerate(sample["track_id"]):
        past_mask = ~sample["x_padding_mask"][actor_idx, :50]
        if not past_mask.any():
            continue

        past_world = local_points_to_world(
            sample["x_positions"][actor_idx][past_mask], origin, theta
        )
        plot_points.append(past_world)

        is_scored = bool(sample["x_scored"][actor_idx].item())
        is_av = actor_idx == av_index

        history_color = "#1f77b4" if is_scored else "#444444"
        history_width = 2.2 if is_av else 1.8 if is_scored else 1.0
        history_alpha = 1.0 if is_scored or is_av else 0.35
        history_label = None
        if (is_scored or is_av) and not history_labels_used:
            history_label = "history"
            history_labels_used = True
        ax.plot(
            past_world[:, 0],
            past_world[:, 1],
            color=history_color,
            linewidth=history_width,
            alpha=history_alpha,
            zorder=3,
            label=history_label,
        )
        ax.scatter(
            past_world[-1, 0], past_world[-1, 1], color=history_color, s=18, zorder=4
        )

        if sample["y"] is not None:
            future_mask = ~sample["x_padding_mask"][actor_idx, 50:]
            if future_mask.any():
                future_local = (
                    sample["y"][actor_idx][future_mask]
                    + sample["x_positions"][actor_idx, 49]
                )
                future_world = local_points_to_world(future_local, origin, theta)
                plot_points.append(future_world)
                future_label = None
                if is_scored and not future_labels_used:
                    future_label = "ground truth"
                    future_labels_used = True
                ax.plot(
                    future_world[:, 0],
                    future_world[:, 1],
                    color="#2ca02c",
                    linewidth=1.8 if is_scored else 1.0,
                    alpha=0.95 if is_scored else 0.3,
                    zorder=3,
                    label=future_label,
                )

        if not is_scored:
            continue

        modal_order = np.argsort(prob[actor_idx])[::-1][:num_modes]
        for rank, mode_idx in enumerate(modal_order):
            pred_world = world_pred[actor_idx, mode_idx]
            plot_points.append(pred_world)
            pred_label = None
            if rank == 0 and not prediction_labels_used:
                pred_label = "prediction"
                prediction_labels_used = True
            ax.plot(
                pred_world[:, 0],
                pred_world[:, 1],
                color="#d62728",
                linewidth=2.0 if rank == 0 else 1.2,
                alpha=max(0.2, float(prob[actor_idx, mode_idx])),
                zorder=5,
                label=pred_label,
            )
            ax.scatter(
                pred_world[-1, 0],
                pred_world[-1, 1],
                color="#d62728",
                s=10,
                alpha=0.8,
                zorder=6,
            )

        ax.text(
            past_world[-1, 0],
            past_world[-1, 1],
            track_id if not is_av else f"{track_id} (AV)",
            fontsize=7,
            color="#111111",
            zorder=7,
        )

    x_bounds, y_bounds = compute_plot_bounds(plot_points)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(sample["scenario_id"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    device = torch.device("cuda")
    model = load_model(conf).to(device)
    data_root = Path(conf.data_root)
    split = "val"
    num_scenes = 16
    num_modes = 3
    output_dir = Path(HydraConfig.get().runtime.output_dir) / "visualizations"

    scenario_files = select_scenario_files(
        data_root=data_root,
        split=split,
        num_scenes=num_scenes,
    )
    extractor = Av2ExtractorMultiAgent(mode=split)

    for scenario_file in scenario_files:
        scenario_id = scenario_file.stem.split("_")[-1]
        sample = extractor.get_data(scenario_file)
        batch = collate_fn([sample])
        batch_on_device = move_batch_to_device(batch, device)

        with torch.no_grad():
            outputs = model(batch_on_device)
            world_pred, prob = decode_world_predictions(outputs, batch_on_device)

        output_path = output_dir / split / f"{scenario_id}.png"
        plot_scenario(
            sample=sample,
            world_pred=world_pred,
            prob=prob,
            output_path=output_path,
            num_modes=num_modes,
            dpi=200,
        )
        print(f"saved visualization: {output_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
