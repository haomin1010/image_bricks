# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.assembling import mdp
from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import (
    ASSEMBLING_MAX_CUBES,
    DEFAULT_CUBE_SIZE,
    AssemblingEnvCfg,
    EventsCfg,
)

TELEPORT_STACK_TASK_ID = "multipicture_teleport_stack_from_begin"

HIDDEN_CUBE_SOURCE_PICK_POS_X = -5.0
HIDDEN_CUBE_SOURCE_PICK_POS_Y = -5.0
DEFAULT_TELEPORT_CUBE_COLLISION_ENABLED = False


@configclass
class TeleportStackEventsCfg(EventsCfg):
    """Cube events for teleport stack runtime."""

    setup_orthographic_cameras = EventTerm(
        func=mdp.setup_orthographic_cameras_event,
        mode="startup",
    )

    setup_cubes_startup = EventTerm(
        func=mdp.place_cubes_event,
        mode="startup",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
            "source_pick_pos_x": HIDDEN_CUBE_SOURCE_PICK_POS_X,
            "source_pick_pos_y": HIDDEN_CUBE_SOURCE_PICK_POS_Y,
        },
    )
    setup_cubes_reset = EventTerm(
        func=mdp.place_cubes_event,
        mode="reset",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
            "source_pick_pos_x": HIDDEN_CUBE_SOURCE_PICK_POS_X,
            "source_pick_pos_y": HIDDEN_CUBE_SOURCE_PICK_POS_Y,
        },
    )


@configclass
class TeleportStackEnvCfg(AssemblingEnvCfg):
    """Teleport stack environment config based on AssemblingEnvCfg."""

    runtime_builder = staticmethod(mdp.build_teleport_runtime)
    events: TeleportStackEventsCfg = TeleportStackEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep cube spawning logic aligned with FrankaStackEnvCfg.
        self.cube_properties, self.cube_mass_props, self.cube_scale = mdp.build_default_cube_spawn_settings(mass=0.02)
        mdp.configure_stack_scene_cameras(scene_cfg=self.scene, enable_cameras=True, cube_size=DEFAULT_CUBE_SIZE)
        mdp.apply_policy_observation_overrides(
            getattr(self.observations, "policy", None),
            max_cubes=ASSEMBLING_MAX_CUBES,
        )
        self._configure_cubes_for_server()

    def _configure_cubes_for_server(self):
        mdp.configure_server_cube_layout(
            scene_cfg=self.scene,
            cube_properties=self.cube_properties,
            cube_mass_props=self.cube_mass_props,
            cube_scale=getattr(self, "cube_scale", (1.0, 1.0, 1.0)),
            max_cubes=max(1, int(os.getenv("VAGEN_MAX_CUBES", str(ASSEMBLING_MAX_CUBES)))),
            cube_size=float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
            cube_name_prefix="cube_",
            source_pick_pos_x=HIDDEN_CUBE_SOURCE_PICK_POS_X,
            source_pick_pos_y=HIDDEN_CUBE_SOURCE_PICK_POS_Y,
            collision_enabled=DEFAULT_TELEPORT_CUBE_COLLISION_ENABLED,
        )
