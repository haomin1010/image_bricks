# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp

if TYPE_CHECKING:
    from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _resolve_required_repo_asset(rel_path: str) -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / rel_path
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Required local asset not found: {rel_path}")


FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_ARM_ONLY_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_ARM_ONLY_CFG.spawn.usd_path = _resolve_required_repo_asset(
    "assets/Isaac/Robots/FrankaRobotics/FrankaPanda_copy/configuration/franka_Gripper_UR10_Short_Suction.usd"
)
FRANKA_ARM_ONLY_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
}
FRANKA_ARM_ONLY_CFG.actuators = {
    actuator_name: actuator_cfg
    for actuator_name, actuator_cfg in FRANKA_ARM_ONLY_CFG.actuators.items()
    if actuator_name in {"panda_shoulder", "panda_forearm"}
}
# Apply stiffer PD control suitable for differential IK task-space control
FRANKA_ARM_ONLY_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_ARM_ONLY_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_ARM_ONLY_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_ARM_ONLY_CFG.actuators["panda_forearm"].damping = 80.0
ASSEMBLING_EE_BODY_NAME = "panda_link7"
# Local transform from UR10 short gripper rigid root (/Root) to suction_cup frame.
UR10_SHORT_SUCTION_CUP_LOCAL_POS = [0.1585, 0.0, 0.0]
UR10_SHORT_SUCTION_CUP_LOCAL_ROT = [0.0, 0.70710678, 0.0, 0.70710678]
FRANKA_UR10_TCP_OFFSET_POS_DEFAULT = list(UR10_SHORT_SUCTION_CUP_LOCAL_POS)
FRANKA_UR10_TCP_OFFSET_ROT_DEFAULT = list(UR10_SHORT_SUCTION_CUP_LOCAL_ROT)
DEFAULT_UR10_SUCTION_TARGET_PRIM_PATH = "{ENV_REGEX_NS}/Robot/UR10ShortSuction"
ASSEMBLING_EE_TARGET_PRIM_PATH = DEFAULT_UR10_SUCTION_TARGET_PRIM_PATH
ASSEMBLING_MAX_CUBES = int(getattr(mdp, "DEFAULT_MAX_CUBES", 8))
DEFAULT_GRID_ORIGIN: tuple[float, float, float] = (0.5, 0.0, 0.001)
DEFAULT_GRID_SIZE = 8
DEFAULT_GRID_LINE_THICKNESS = 0.001
DEFAULT_GRID_CELL_SIZE = 0.055 + DEFAULT_GRID_LINE_THICKNESS
# Downward-pointing reset pose: J=[0,-0.785,0,-2.356,0,1.571,0.785]
# Arm starts with wrist pointing DOWN so the IK doesn't need to flip 180°.
DEFAULT_ARM_RESET_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class AssemblingCfgOverride:
    """Centralized runtime override config for assembling env cfg."""

    def __init__(
        self,
        *,
        enable_cameras: bool,
        cube_size: float,
        cube_properties: RigidBodyPropertiesCfg,
        cube_scale: tuple[float, float, float],
        ik_lambda_val: float,
        ik_step_gain: float,
        ik_max_joint_delta: float,
        ik_nullspace_gain: float,
        ee_body_name: str,
        max_cubes: int,
        magic_suction_close_command_threshold: float,
        decimation: int,
        episode_length_s: float,
        sim_dt: float,
        sim_render_interval: int,
        physx_bounce_threshold_velocity: float,
        physx_gpu_found_lost_aggregate_pairs_capacity: int,
        physx_gpu_total_aggregate_pairs_capacity: int,
        physx_friction_correlation_distance: float,
        physx_enable_external_forces_every_iteration: bool,
    ):
        self.enable_cameras = bool(enable_cameras)
        self.cube_size = float(cube_size)
        self.cube_properties = copy.deepcopy(cube_properties)
        self.cube_scale = tuple(float(v) for v in cube_scale)
        self.ik_lambda_val = float(ik_lambda_val)
        self.ik_step_gain = float(ik_step_gain)
        self.ik_max_joint_delta = float(ik_max_joint_delta)
        self.ik_nullspace_gain = float(ik_nullspace_gain)
        self.ee_body_name = str(ee_body_name)
        self.max_cubes = int(max_cubes)
        self.magic_suction_close_command_threshold = float(magic_suction_close_command_threshold)
        self.decimation = int(decimation)
        self.episode_length_s = float(episode_length_s)
        self.sim_dt = float(sim_dt)
        self.sim_render_interval = int(sim_render_interval)
        self.physx_bounce_threshold_velocity = float(physx_bounce_threshold_velocity)
        self.physx_gpu_found_lost_aggregate_pairs_capacity = int(physx_gpu_found_lost_aggregate_pairs_capacity)
        self.physx_gpu_total_aggregate_pairs_capacity = int(physx_gpu_total_aggregate_pairs_capacity)
        self.physx_friction_correlation_distance = float(physx_friction_correlation_distance)
        self.physx_enable_external_forces_every_iteration = bool(physx_enable_external_forces_every_iteration)

    @classmethod
    def from_env(
        cls,
        *,
        cube_size: float | None = None,
        enable_cameras: bool | None = None,
    ) -> "AssemblingCfgOverride":
        ee_body_name = os.getenv("VAGEN_IK_EE_BODY_NAME", ASSEMBLING_EE_BODY_NAME).strip() or ASSEMBLING_EE_BODY_NAME
        max_cubes = int(os.getenv("VAGEN_MAX_CUBES", str(ASSEMBLING_MAX_CUBES)))

        return cls(
            enable_cameras=(os.getenv("VAGEN_ENABLE_CAMERAS", "1") != "0") if enable_cameras is None else enable_cameras,
            cube_size=float(os.getenv("VAGEN_CUBE_SIZE", "0.045")) if cube_size is None else float(cube_size),
            cube_properties=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            cube_scale=(1.0, 1.0, 1.0),
            ik_lambda_val=float(os.getenv("VAGEN_IK_LAMBDA_VAL", "0.10")),
            ik_step_gain=float(os.getenv("VAGEN_IK_STEP_GAIN", "0.30")),
            ik_max_joint_delta=float(os.getenv("VAGEN_IK_MAX_JOINT_DELTA", "0.02")),
            ik_nullspace_gain=float(os.getenv("VAGEN_IK_NULLSPACE_GAIN", "0.0")),
            ee_body_name=ee_body_name,
            max_cubes=max_cubes,
            magic_suction_close_command_threshold=float(os.getenv("VAGEN_MAGIC_SUCTION_CLOSE_CMD_THRESHOLD", "0.0")),
            decimation=5,
            episode_length_s=600.0,
            sim_dt=0.01,
            sim_render_interval=5,
            physx_bounce_threshold_velocity=0.01,
            physx_gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            physx_gpu_total_aggregate_pairs_capacity=16 * 1024,
            physx_friction_correlation_distance=0.00625,
            physx_enable_external_forces_every_iteration=_env_flag(
                "VAGEN_PHYSX_ENABLE_EXTERNAL_FORCES_EVERY_ITERATION",
                True,
            ),
        )

    def apply(self, env_cfg: "AssemblingEnvCfg", *, arm_joint_names: list[str]) -> None:
        setattr(env_cfg, "cube_properties", copy.deepcopy(self.cube_properties))
        setattr(env_cfg, "cube_scale", tuple(self.cube_scale))

        mdp.configure_stack_scene_cameras(
            scene_cfg=env_cfg.scene,
            enable_cameras=self.enable_cameras,
            cube_size=self.cube_size,
        )

        arm_action = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
        if isinstance(arm_action, mdp.PinocchioPoseActionCfg):
            arm_action.joint_names = list(arm_joint_names)
            arm_action.ee_body_name = self.ee_body_name
            arm_action.damping = self.ik_lambda_val
            arm_action.step_gain = self.ik_step_gain
            arm_action.max_joint_delta = self.ik_max_joint_delta
            arm_action.nullspace_gain = self.ik_nullspace_gain

        observations = getattr(env_cfg, "observations", None)
        policy_obs = getattr(observations, "policy", None)
        if policy_obs is not None:
            cube_positions = getattr(policy_obs, "cube_positions", None)
            if cube_positions is not None and hasattr(cube_positions, "params"):
                cube_positions.params["max_cubes"] = self.max_cubes
            cube_orientations = getattr(policy_obs, "cube_orientations", None)
            if cube_orientations is not None and hasattr(cube_orientations, "params"):
                cube_orientations.params["max_cubes"] = self.max_cubes
            ee_pos = getattr(policy_obs, "ee_pos", None)
            if ee_pos is not None and hasattr(ee_pos, "params"):
                ee_pos.params["ee_body_name"] = self.ee_body_name
            ee_quat = getattr(policy_obs, "ee_quat", None)
            if ee_quat is not None and hasattr(ee_quat, "params"):
                ee_quat.params["ee_body_name"] = self.ee_body_name
        privileged_obs = getattr(observations, "privileged", None)
        privileged_state = getattr(privileged_obs, "state", None) if privileged_obs is not None else None
        if privileged_state is not None and hasattr(privileged_state, "params"):
            privileged_state.params["ee_body_name"] = self.ee_body_name
            privileged_state.params["max_cubes"] = self.max_cubes

        events = getattr(env_cfg, "events", None)
        magic_suction_controller = getattr(events, "magic_suction_controller", None) if events is not None else None
        if magic_suction_controller is not None and hasattr(magic_suction_controller, "params"):
            magic_suction_controller.params["max_cubes"] = self.max_cubes
            magic_suction_controller.params["cube_size"] = self.cube_size
            magic_suction_controller.params["ee_body_name"] = self.ee_body_name
        teleport_pending_cubes = getattr(events, "teleport_pending_cubes", None) if events is not None else None
        if teleport_pending_cubes is not None and hasattr(teleport_pending_cubes, "params"):
            teleport_pending_cubes.params["max_cubes"] = self.max_cubes
            teleport_pending_cubes.params["cube_size"] = self.cube_size

        gripper_action = getattr(getattr(env_cfg, "actions", None), "gripper_action", None)
        if gripper_action is not None and hasattr(gripper_action, "close_command_threshold"):
            gripper_action.close_command_threshold = self.magic_suction_close_command_threshold

        if hasattr(env_cfg, "decimation"):
            env_cfg.decimation = self.decimation
        if hasattr(env_cfg, "episode_length_s"):
            env_cfg.episode_length_s = self.episode_length_s

        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is None:
            return
        if hasattr(sim_cfg, "dt"):
            sim_cfg.dt = self.sim_dt
        if hasattr(sim_cfg, "render_interval"):
            sim_cfg.render_interval = self.sim_render_interval

        physx_cfg = getattr(sim_cfg, "physx", None)
        if physx_cfg is None:
            return
        if hasattr(physx_cfg, "bounce_threshold_velocity"):
            physx_cfg.bounce_threshold_velocity = self.physx_bounce_threshold_velocity
        if hasattr(physx_cfg, "gpu_found_lost_aggregate_pairs_capacity"):
            physx_cfg.gpu_found_lost_aggregate_pairs_capacity = self.physx_gpu_found_lost_aggregate_pairs_capacity
        if hasattr(physx_cfg, "gpu_total_aggregate_pairs_capacity"):
            physx_cfg.gpu_total_aggregate_pairs_capacity = self.physx_gpu_total_aggregate_pairs_capacity
        if hasattr(physx_cfg, "friction_correlation_distance"):
            physx_cfg.friction_correlation_distance = self.physx_friction_correlation_distance
        if hasattr(physx_cfg, "enable_external_forces_every_iteration"):
            physx_cfg.enable_external_forces_every_iteration = self.physx_enable_external_forces_every_iteration
