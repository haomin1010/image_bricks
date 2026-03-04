# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp

if TYPE_CHECKING:
    from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


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
FRANKA_ARM_ONLY_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}
# Apply stiffer PD control suitable for differential IK task-space control
FRANKA_ARM_ONLY_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_ARM_ONLY_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_ARM_ONLY_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_ARM_ONLY_CFG.actuators["panda_forearm"].damping = 80.0


def _resolve_local_franka_usd() -> str | None:
    rel_path = Path("Robots/FrankaRobotics/FrankaPanda/franka.usd")
    for parent in Path(__file__).resolve().parents:
        assets_root = parent / "assets" / "Isaac"
        candidate = assets_root / rel_path
        if candidate.exists():
            return str(candidate)
    return None


_LOCAL_FRANKA_USD = _resolve_local_franka_usd()
if _LOCAL_FRANKA_USD is not None:
    FRANKA_ARM_ONLY_CFG.spawn.usd_path = _LOCAL_FRANKA_USD

ASSEMBLING_EE_BODY_NAME = "panda_hand"
DEFAULT_PANDA_HAND_TARGET_PRIM_PATH = "{ENV_REGEX_NS}/Robot/panda_hand"
ASSEMBLING_EE_TARGET_PRIM_PATH = DEFAULT_PANDA_HAND_TARGET_PRIM_PATH
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
        cube_mass_props: MassPropertiesCfg | None,
        cube_scale: tuple[float, float, float],
        ik_lambda_val: float,
        ik_step_gain: float,
        ik_max_joint_delta: float,
        ik_nullspace_gain: float,
        ee_body_name: str,
        obs_ee_body_name: str,
        max_cubes: int,
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
        self.cube_mass_props = copy.deepcopy(cube_mass_props)
        self.cube_scale = tuple(float(v) for v in cube_scale)
        self.ik_lambda_val = float(ik_lambda_val)
        self.ik_step_gain = float(ik_step_gain)
        self.ik_max_joint_delta = float(ik_max_joint_delta)
        self.ik_nullspace_gain = float(ik_nullspace_gain)
        self.ee_body_name = str(ee_body_name)
        self.obs_ee_body_name = str(obs_ee_body_name)
        self.max_cubes = int(max_cubes)
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
        obs_ee_body_name = os.getenv("VAGEN_OBS_EE_BODY_NAME", "panda_hand").strip() or ee_body_name
        max_cubes = int(os.getenv("VAGEN_MAX_CUBES", str(ASSEMBLING_MAX_CUBES)))
        cube_mass_kg = float(os.getenv("VAGEN_CUBE_MASS_KG", "0.12"))
        cube_mass_props = MassPropertiesCfg(mass=cube_mass_kg) if cube_mass_kg > 0.0 else None

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
            cube_mass_props=cube_mass_props,
            cube_scale=(1.0, 1.0, 1.0),
            ik_lambda_val=float(os.getenv("VAGEN_IK_LAMBDA_VAL", "0.10")),
            ik_step_gain=float(os.getenv("VAGEN_IK_STEP_GAIN", "0.30")),
            ik_max_joint_delta=float(os.getenv("VAGEN_IK_MAX_JOINT_DELTA", "0.02")),
            ik_nullspace_gain=float(os.getenv("VAGEN_IK_NULLSPACE_GAIN", "0.0")),
            ee_body_name=ee_body_name,
            obs_ee_body_name=obs_ee_body_name,
            max_cubes=max_cubes,
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
        setattr(env_cfg, "cube_mass_props", copy.deepcopy(self.cube_mass_props))
        setattr(env_cfg, "cube_scale", tuple(self.cube_scale))

        mdp.configure_stack_scene_cameras(
            scene_cfg=env_cfg.scene,
            enable_cameras=self.enable_cameras,
            cube_size=self.cube_size,
        )

        arm_action = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
        if isinstance(arm_action, DifferentialInverseKinematicsActionCfg):
            arm_action.joint_names = list(arm_joint_names)
            arm_action.body_name = self.ee_body_name
            if arm_action.controller is not None:
                arm_action.controller.use_relative_mode = False
                if arm_action.controller.ik_params is None:
                    arm_action.controller.ik_params = {}
                arm_action.controller.ik_params["lambda_val"] = self.ik_lambda_val

        observations = getattr(env_cfg, "observations", None)
        policy_obs = getattr(observations, "policy", None)
        if policy_obs is not None:
            cube_pos = getattr(policy_obs, "cube_pos", None)
            if cube_pos is not None and hasattr(cube_pos, "params"):
                cube_pos.params["max_cubes"] = self.max_cubes
            cube_quat = getattr(policy_obs, "cube_quat", None)
            if cube_quat is not None and hasattr(cube_quat, "params"):
                cube_quat.params["max_cubes"] = self.max_cubes
            ee_pos = getattr(policy_obs, "ee_pos", None)
            if ee_pos is not None and hasattr(ee_pos, "params"):
                # ee_pos is ee_frame-based in assembling; drop stale body-based override keys.
                ee_pos.params.pop("ee_body_name", None)
            ee_quat = getattr(policy_obs, "ee_quat", None)
            if ee_quat is not None and hasattr(ee_quat, "params"):
                # ee_quat is ee_frame-based in assembling; drop stale body-based override keys.
                ee_quat.params.pop("ee_body_name", None)
            grasped = getattr(policy_obs, "grasped", None)
            if grasped is not None and hasattr(grasped, "params"):
                # object_grasped is also ee_frame-based.
                grasped.params.pop("ee_body_name", None)

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
