# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_MAX_CUBES = int(os.getenv("VAGEN_MAX_CUBES", "16"))

def _obs_device_and_dtype(env: ManagerBasedRLEnv) -> tuple[torch.device, torch.dtype]:
    env_origins = getattr(env.scene, "env_origins", None)
    if isinstance(env_origins, torch.Tensor):
        return env_origins.device, env_origins.dtype
    return torch.device(env.device), torch.float32


def _obs_static_cache(env: ManagerBasedRLEnv) -> dict:
    env_unwrapped = getattr(env, "unwrapped", env)
    cache = getattr(env_unwrapped, "_assembling_obs_static_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env_unwrapped, "_assembling_obs_static_cache", cache)
    return cache


def _discover_cube_names(scene, cube_name_prefix: str, max_cubes: int) -> list[str]:
    """Resolve cube entity names strictly as {prefix}1..{prefix}N."""
    if max_cubes <= 0:
        return []

    expected_names = [f"{cube_name_prefix}{cube_idx}" for cube_idx in range(1, max_cubes + 1)]
    keys_fn = getattr(scene, "keys", None)
    available_names = set(keys_fn()) if callable(keys_fn) else set()
    missing_names = [name for name in expected_names if name not in available_names and not hasattr(scene, name)]
    if missing_names:
        keys_sample = [str(k) for k in list(available_names)[:16]]
        raise RuntimeError(
            "Strict cube discovery failed: missing cube entities. "
            f"expected={expected_names} missing={missing_names} scene_keys_sample={keys_sample}"
        )
    return expected_names


def _iter_cube_names(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
) -> list[str]:
    if max_cubes <= 0:
        return []

    cache_key = ("cube_names", cube_name_prefix, int(max_cubes))
    cache = _obs_static_cache(env)
    cached_names = cache.get(cache_key, None)
    cube_names = _discover_cube_names(env.scene, cube_name_prefix=cube_name_prefix, max_cubes=max_cubes)
    cache[cache_key] = tuple(cube_names)
    return cube_names


def _resolve_collection_object_indices(env: ManagerBasedRLEnv, default_index: int) -> torch.Tensor:
    device, _ = _obs_device_and_dtype(env)
    indices = torch.full((env.num_envs,), int(default_index), device=device, dtype=torch.long)
    focused_objects = getattr(env, "rigid_objects_in_focus", None)
    if focused_objects is None:
        return indices

    if torch.is_tensor(focused_objects):
        if focused_objects.ndim == 2 and focused_objects.shape[0] == env.num_envs and focused_objects.shape[1] > default_index:
            return focused_objects[:, default_index].to(device=device, dtype=torch.long)
        if focused_objects.ndim == 1 and focused_objects.shape[0] == env.num_envs and default_index == 0:
            return focused_objects.to(device=device, dtype=torch.long)
        return indices

    if not isinstance(focused_objects, (list, tuple)):
        return indices

    resolved: list[int] = []
    for env_id in range(env.num_envs):
        env_focus = focused_objects[env_id] if env_id < len(focused_objects) else None
        idx = int(default_index)
        if torch.is_tensor(env_focus) and env_focus.numel() > default_index:
            idx = int(env_focus[default_index].item())
        elif isinstance(env_focus, (list, tuple)) and len(env_focus) > default_index:
            idx = int(env_focus[default_index])
        elif isinstance(env_focus, (int, float)) and default_index == 0:
            idx = int(env_focus)
        resolved.append(idx)
    return torch.tensor(resolved, device=device, dtype=torch.long)


def all_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Flattened positions of all configured cubes in the world frame."""
    device, dtype = _obs_device_and_dtype(env)
    cube_pos_w = torch.full((env.num_envs, max_cubes, 3), float(fill_value), device=device, dtype=dtype)

    cube_names = _iter_cube_names(env, max_cubes=max_cubes, cube_name_prefix=cube_name_prefix)
    for cube_slot, cube_name in enumerate(cube_names):
        cube_asset = env.scene[cube_name]
        if isinstance(cube_asset, RigidObject):
            pos_w = cube_asset.data.root_pos_w[:, :3]
        elif isinstance(cube_asset, RigidObjectCollection):
            env_ids = torch.arange(env.num_envs, device=cube_asset.data.object_pos_w.device, dtype=torch.long)
            object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
            pos_w = cube_asset.data.object_pos_w[env_ids, object_indices, :3]
        else:
            cube_data = cube_asset.data
            if hasattr(cube_data, "root_pos_w"):
                pos_w = cube_data.root_pos_w[:, :3]
            elif hasattr(cube_data, "object_pos_w"):
                env_ids = torch.arange(env.num_envs, device=cube_data.object_pos_w.device, dtype=torch.long)
                object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
                pos_w = cube_data.object_pos_w[env_ids, object_indices, :3]
            else:
                raise TypeError(f"Unsupported cube asset type for '{cube_name}': {type(cube_asset)}")
        cube_pos_w[:, cube_slot, :] = pos_w.to(device=device, dtype=dtype)

    return cube_pos_w.reshape(env.num_envs, max_cubes * 3)


def env_origin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment origin in world frame."""
    return env.scene.env_origins[:, 0:3]


def all_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Flattened orientations of all configured cubes in the world frame."""
    device, dtype = _obs_device_and_dtype(env)
    cube_quat_w = torch.full((env.num_envs, max_cubes, 4), float(fill_value), device=device, dtype=dtype)

    cube_names = _iter_cube_names(env, max_cubes=max_cubes, cube_name_prefix=cube_name_prefix)
    for cube_slot, cube_name in enumerate(cube_names):
        cube_asset = env.scene[cube_name]
        if isinstance(cube_asset, RigidObject):
            quat_w = cube_asset.data.root_quat_w[:, :4]
        elif isinstance(cube_asset, RigidObjectCollection):
            env_ids = torch.arange(env.num_envs, device=cube_asset.data.object_quat_w.device, dtype=torch.long)
            object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
            quat_w = cube_asset.data.object_quat_w[env_ids, object_indices, :4]
        else:
            cube_data = cube_asset.data
            if hasattr(cube_data, "root_quat_w"):
                quat_w = cube_data.root_quat_w[:, :4]
            elif hasattr(cube_data, "object_quat_w"):
                env_ids = torch.arange(env.num_envs, device=cube_data.object_quat_w.device, dtype=torch.long)
                object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
                quat_w = cube_data.object_quat_w[env_ids, object_indices, :4]
            else:
                raise TypeError(f"Unsupported cube asset type for '{cube_name}': {type(cube_asset)}")
        cube_quat_w[:, cube_slot, :] = quat_w.to(device=device, dtype=dtype)

    return cube_quat_w.reshape(env.num_envs, max_cubes * 4)


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Obtain gripper position for both suction and parallel gripper setups."""
    robot: Articulation = env.scene[robot_cfg.name]

    surface_grippers = getattr(env.scene, "surface_grippers", None)
    if surface_grippers is not None and len(surface_grippers) > 0:
        gripper_states: list[torch.Tensor] = []
        for _, surface_gripper in surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))
        if len(gripper_states) == 1:
            return gripper_states[0]
        return torch.cat(gripper_states, dim=1)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observation gripper_pos only supports parallel gripper for now."
        finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
        finger_joint_2 = -1.0 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
        return torch.cat((finger_joint_1, finger_joint_2), dim=1)

    raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def gripper_closed_flag(
    env: ManagerBasedRLEnv,
    close_pos_threshold: float = 0.005,
) -> torch.Tensor:
    """Binary flag (1=closed) from mean finger opening."""
    finger_pos = torch.abs(gripper_pos(env))
    mean_opening = torch.mean(finger_pos, dim=1, keepdim=True)
    return (mean_opening <= float(close_pos_threshold)).to(dtype=finger_pos.dtype)


def ee_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in env frame from ee_frame target pose."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]


def ee_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector quaternion (wxyz) from ee_frame target pose."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w[:, :3]
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    surface_grippers = getattr(env.scene, "surface_grippers", None)
    if surface_grippers is not None and len(surface_grippers) > 0:
        surface_gripper = surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = suction_cup_status == 1
        return torch.logical_and(suction_cup_is_closed, pose_diff < float(diff_threshold))

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now."
        open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        joint_1_closed = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val) > env.cfg.gripper_threshold
        joint_2_closed = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - open_val) > env.cfg.gripper_threshold
        return torch.logical_and(torch.logical_and(pose_diff < float(diff_threshold), joint_1_closed), joint_2_closed)

    raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")
