# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg


@configclass
class TeleportStackEnvCfg(AssemblingEnvCfg):
    """Teleport stack environment config based on AssemblingEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
