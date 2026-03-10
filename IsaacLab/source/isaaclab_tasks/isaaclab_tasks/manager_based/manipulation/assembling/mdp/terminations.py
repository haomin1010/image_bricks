"""Placeholder termination definitions for assembling tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def placeholder_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """No-op termination placeholder, always returns False for all envs."""
    return torch.zeros((env.num_envs,), device=torch.device(env.device), dtype=torch.bool)


__all__ = ["placeholder_termination"]
