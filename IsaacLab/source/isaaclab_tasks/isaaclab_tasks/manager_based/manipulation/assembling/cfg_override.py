# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from . import mdp

if TYPE_CHECKING:
    from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg


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
        magic_suction_close_command_threshold: float,
        decimation: int,
        episode_length_s: float,
        sim_dt: float,
        sim_render_interval: int,
        physx_bounce_threshold_velocity: float,
        physx_gpu_found_lost_aggregate_pairs_capacity: int,
        physx_gpu_total_aggregate_pairs_capacity: int,
        physx_friction_correlation_distance: float,
    ):
        self.enable_cameras = bool(enable_cameras)
        self.cube_size = float(cube_size)
        self.cube_properties = copy.deepcopy(cube_properties)
        self.cube_scale = tuple(float(v) for v in cube_scale)
        self.ik_lambda_val = float(ik_lambda_val)
        self.ik_step_gain = float(ik_step_gain)
        self.ik_max_joint_delta = float(ik_max_joint_delta)
        self.ik_nullspace_gain = float(ik_nullspace_gain)
        self.magic_suction_close_command_threshold = float(magic_suction_close_command_threshold)
        self.decimation = int(decimation)
        self.episode_length_s = float(episode_length_s)
        self.sim_dt = float(sim_dt)
        self.sim_render_interval = int(sim_render_interval)
        self.physx_bounce_threshold_velocity = float(physx_bounce_threshold_velocity)
        self.physx_gpu_found_lost_aggregate_pairs_capacity = int(physx_gpu_found_lost_aggregate_pairs_capacity)
        self.physx_gpu_total_aggregate_pairs_capacity = int(physx_gpu_total_aggregate_pairs_capacity)
        self.physx_friction_correlation_distance = float(physx_friction_correlation_distance)

    @classmethod
    def from_env(
        cls,
        *,
        cube_size: float | None = None,
        enable_cameras: bool | None = None,
    ) -> "AssemblingCfgOverride":
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
            ik_step_gain=float(os.getenv("VAGEN_IK_STEP_GAIN", "0.70")),
            ik_max_joint_delta=float(os.getenv("VAGEN_IK_MAX_JOINT_DELTA", "0.08")),
            ik_nullspace_gain=float(os.getenv("VAGEN_IK_NULLSPACE_GAIN", "0.02")),
            magic_suction_close_command_threshold=float(os.getenv("VAGEN_MAGIC_SUCTION_CLOSE_CMD_THRESHOLD", "0.0")),
            decimation=5,
            episode_length_s=600.0,
            sim_dt=0.01,
            sim_render_interval=5,
            physx_bounce_threshold_velocity=0.01,
            physx_gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            physx_gpu_total_aggregate_pairs_capacity=16 * 1024,
            physx_friction_correlation_distance=0.00625,
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
            ee_body_name = os.getenv("VAGEN_IK_EE_BODY_NAME", "").strip()
            if ee_body_name:
                arm_action.ee_body_name = ee_body_name
            arm_action.damping = self.ik_lambda_val
            arm_action.step_gain = self.ik_step_gain
            arm_action.max_joint_delta = self.ik_max_joint_delta
            arm_action.nullspace_gain = self.ik_nullspace_gain

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
