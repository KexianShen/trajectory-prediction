import time
from pathlib import Path

import torch
from torch import Tensor

from .av2_multiagent_submission_protocol.submission import ChallengeSubmission


class SubmissionAv2MultiAgent:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.submission_file = Path(save_dir) / f"multi_agent_{stamp}.parquet"
        self.challenge_submission = ChallengeSubmission(predictions={})

    @torch.no_grad()
    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
    ) -> None:
        """
        trajectory: (B, A, K, 60, 2) or (B, K, A, 60, 2)
        probability: (B, A, K) or (B, K)
        normalized_probability: if the input probability is normalized,
        """
        scenario_ids_list = data["scenario_id"]
        track_ids_list = data["track_id"]
        scored_agents_list = data["x_scored"].cpu()
        num_actors = data["x_positions"].size(1)

        if trajectory.dim() != 5:
            raise ValueError(f"unexpected trajectory shape: {trajectory.shape}")
        if probability.dim() == 3:
            if trajectory.size(1) != num_actors:
                trajectory = trajectory.transpose(1, 2)
            B, A, K, _, _ = trajectory.shape
        elif probability.dim() == 2:
            B, K, A, _, _ = trajectory.shape
            trajectory = trajectory.transpose(1, 2)
        else:
            raise ValueError(f"unexpected probability shape: {probability.shape}")

        last_position = data["x_positions"][:, :, -1, :].view(B, A, 1, 1, 2)
        origin = data["origin"].view(B, 1, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).view(B, 1, 1, 2, 2)

        global_trajectory = (
            (
                torch.matmul(trajectory[..., :2].double() + last_position, rotate_mat)
                + origin
            )
            .cpu()
            .numpy()
        )

        if not normalized_probability:
            probability = torch.softmax(probability.double(), dim=-1)

        if probability.dim() == 3:
            scored_agents = (
                data["x_scored"].to(probability.device).unsqueeze(-1).double()
            )
            has_scored_agents = scored_agents.sum(dim=1, keepdim=True) > 0
            fallback_agents = torch.ones_like(scored_agents)
            weights = torch.where(has_scored_agents, scored_agents, fallback_agents)
            probability = (probability.double() * weights).sum(dim=1) / weights.sum(
                dim=1
            )
        probability = probability.cpu().numpy()

        for i, (scene_id, track_ids, scored_agents) in enumerate(
            zip(scenario_ids_list, track_ids_list, scored_agents_list)
        ):
            scored_track_id = [
                track_ids[j] for j in range(len(scored_agents)) if scored_agents[j]
            ]
            scored_trajectory = global_trajectory[i, scored_agents]
            scenario_predictions = {
                track_id: trajectory
                for track_id, trajectory in zip(scored_track_id, scored_trajectory)
            }

            self.challenge_submission.predictions[scene_id] = (
                probability[i],
                scenario_predictions,
            )

    def generate_submission_file(self):
        print(
            "generating submission file for argoverse 2.0 motion forecasting challenge"
        )
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.submission_file}")
