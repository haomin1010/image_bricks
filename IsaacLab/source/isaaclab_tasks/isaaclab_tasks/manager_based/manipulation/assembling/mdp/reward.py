"""Placeholder reward definitions for assembling tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def placeholder_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """No-op reward placeholder."""
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)


__all__ = ["placeholder_reward"]
