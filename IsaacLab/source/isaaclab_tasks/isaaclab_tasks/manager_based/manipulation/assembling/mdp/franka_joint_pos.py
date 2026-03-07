# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, OperationalSpaceControllerActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

ASSEMBLING_EE_BODY_NAME = "panda_hand"
FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
DEFAULT_FRANKA_RESET_POSE = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.0400, 0.0400)


def resolve_local_franka_usd() -> str | None:
    rel_path = Path("Robots/FrankaRobotics/FrankaPanda/franka.usd")
    for parent in Path(__file__).resolve().parents:
        assets_root = parent / "assets" / "Isaac"
        candidate = assets_root / rel_path
        if candidate.exists():
            return str(candidate)
    return None


def build_franka_osc_cfg() -> ArticulationCfg:
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.5,
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,
        "panda_joint5": 0.0,
        "panda_joint6": 1.5,
        "panda_joint7": 0.7,
        "panda_finger_joint.*": 0.04,
    }
    robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
    robot_cfg.actuators["panda_shoulder"].damping = 0.0
    robot_cfg.actuators["panda_forearm"].stiffness = 0.0
    robot_cfg.actuators["panda_forearm"].damping = 0.0
    # Soften parallel-gripper closure to avoid aggressive snap during grasp.
    robot_cfg.actuators["panda_hand"].stiffness = 600.0
    robot_cfg.actuators["panda_hand"].damping = 40.0
    robot_cfg.actuators["panda_hand"].effort_limit_sim = 80.0
    robot_cfg.spawn.rigid_props.disable_gravity = True

    local_franka_usd = resolve_local_franka_usd()
    if local_franka_usd is not None:
        robot_cfg.spawn.usd_path = local_franka_usd
    return robot_cfg


def build_default_ee_frame_cfg() -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


def build_franka_osc_action_cfg() -> OperationalSpaceControllerActionCfg:
    return OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name=ASSEMBLING_EE_BODY_NAME,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
            motion_damping_ratio_task=(1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        nullspace_joint_pos_target="center",
        position_scale=1.0,
        orientation_scale=1.0,
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.1034],
        ),
    )


def build_franka_gripper_action_cfg() -> BinaryJointPositionActionCfg:
    return BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.01},
    )
