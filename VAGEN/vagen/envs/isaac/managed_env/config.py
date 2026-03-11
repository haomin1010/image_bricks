from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


DEFAULT_GROUND_TRUTH_ROOT = str(
    Path(__file__).resolve().parents[5]
    / "IsaacLab"
    / "scripts"
    / "data_gen"
    / "convex_json_batch"
)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class IsaacManagedEnvConfig:
    """Configuration for the Isaac-managed environment."""

    num_total_envs: int = 64
    n_cameras: int = 3
    image_size: tuple[int, int] = (224, 224)
    max_steps: int = 200
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True
    format_reward: float = 0.1
    success_reward: float = 1.0
    correct_placement_reward: float | None = None
    floating_placement_penalty: float = -10.0
    non_candidate_penalty: float = -5.0
    max_attempts_factor: float = 1.5
    dataset_root: str = "/mnt/data/image_bricks/assets/snapshots"
    ground_truth_root: str = DEFAULT_GROUND_TRUTH_ROOT
    collapse_mock_after_attempt: int = -1

    def __post_init__(self) -> None:
        self.num_total_envs = int(self.num_total_envs)
        self.n_cameras = int(self.n_cameras)
        self.image_size = tuple(int(value) for value in self.image_size)
        self.max_steps = int(self.max_steps)
        self.use_example_in_sys_prompt = _coerce_bool(self.use_example_in_sys_prompt)
        self.format_reward = float(self.format_reward)
        self.success_reward = float(self.success_reward)
        if self.correct_placement_reward in ("", None):
            self.correct_placement_reward = None
        elif self.correct_placement_reward is not None:
            self.correct_placement_reward = float(self.correct_placement_reward)
        self.floating_placement_penalty = float(self.floating_placement_penalty)
        self.non_candidate_penalty = float(self.non_candidate_penalty)
        self.max_attempts_factor = float(self.max_attempts_factor)
        if not self.ground_truth_root:
            self.ground_truth_root = DEFAULT_GROUND_TRUTH_ROOT
        self.collapse_mock_after_attempt = int(self.collapse_mock_after_attempt)


CONFIG_FIELDS = {field.name for field in fields(IsaacManagedEnvConfig)}

__all__ = [
    "CONFIG_FIELDS",
    "DEFAULT_GROUND_TRUTH_ROOT",
    "IsaacManagedEnvConfig",
]
