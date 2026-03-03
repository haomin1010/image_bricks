# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_index_exceeds_max_cubes(
    env: ManagerBasedRLEnv,
    max_cubes: int,
):
    """Terminate env when external task index points outside configured cube range."""
    device = torch.device(env.device)
    exceeded = torch.zeros((env.num_envs,), device=device, dtype=torch.bool)
    if int(max_cubes) <= 0:
        return exceeded

    env_unwrapped = getattr(env, "unwrapped", env)
    upper_bound = int(max_cubes)

    for attr_name in ("_vagen_new_task_index", "_vagen_magic_suction_active_cube_idx"):
        raw_indices = getattr(env_unwrapped, attr_name, None)
        if not isinstance(raw_indices, torch.Tensor) or raw_indices.numel() < env.num_envs:
            continue
        indices = raw_indices.reshape(-1)[: env.num_envs].to(device=device, dtype=torch.long)
        exceeded |= indices >= upper_bound

    return exceeded
