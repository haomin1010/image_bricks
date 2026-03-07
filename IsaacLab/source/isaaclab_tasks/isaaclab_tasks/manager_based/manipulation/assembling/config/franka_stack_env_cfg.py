# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.assembling import mdp
from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import (
    ASSEMBLING_MAX_CUBES,
    AssemblingEnvCfg,
    DEFAULT_CUBE_SIZE,
    EventsCfg,
    ObjectTableSceneCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.assembling.mdp.terminations import task_index_exceeds_max_cubes
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

FRANKA_STACK_TASK_ID = "multipicture_franka_stack_from_begin"

FRANKA_ARM_ONLY_CFG = mdp.build_franka_osc_cfg()


@configclass
class FrankaStackSceneCfg(ObjectTableSceneCfg):
    """Assembling scene with Franka robot and ee frame."""

    robot: ArticulationCfg = FRANKA_ARM_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = mdp.build_default_ee_frame_cfg()


@configclass
class FrankaStackActionsCfg:
    """Franka OSC arm + parallel gripper actions."""

    arm_action: OperationalSpaceControllerActionCfg = mdp.build_franka_osc_action_cfg()
    gripper_action: BinaryJointPositionActionCfg = mdp.build_franka_gripper_action_cfg()


@configclass
class FrankaStackTerminationsCfg(TerminationsCfg):
    """Termination terms for assembling runtime."""

    max_cube_exceeded = DoneTerm(
        func=task_index_exceeds_max_cubes,
        params={"max_cubes": int(ASSEMBLING_MAX_CUBES)},
    )


@configclass
class FrankaStackEventsCfg(EventsCfg):
    """Event terms for assembling runtime."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": list(mdp.DEFAULT_FRANKA_RESET_POSE)},
    )

    setup_cubes_startup = EventTerm(
        func=mdp.place_cubes_event,
        mode="startup",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
        },
    )
    setup_cubes_reset = EventTerm(
        func=mdp.place_cubes_event,
        mode="reset",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
        },
    )


@configclass
class FrankaStackEnvCfg(AssemblingEnvCfg):
    """Default Franka stack environment."""

    scene: FrankaStackSceneCfg = FrankaStackSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    actions: FrankaStackActionsCfg = FrankaStackActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: FrankaStackTerminationsCfg = FrankaStackTerminationsCfg()
    events: FrankaStackEventsCfg = FrankaStackEventsCfg()
    gripper_joint_names = ["panda_finger_.*"]
    gripper_open_val = 0.04
    gripper_threshold = 0.005

    def __post_init__(self):
        super().__post_init__()

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
            max_cubes=max(1, int(os.getenv("VAGEN_MAX_CUBES", "2"))),
            cube_size=float(os.getenv("VAGEN_CUBE_SIZE", str(DEFAULT_CUBE_SIZE))),
            cube_name_prefix="cube_",
            source_pick_pos_x=0.5,
            source_pick_pos_y=-0.33,
        )
