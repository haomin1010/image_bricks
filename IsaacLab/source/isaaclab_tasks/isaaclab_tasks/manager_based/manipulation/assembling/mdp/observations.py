# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_MAX_CUBES = int(os.getenv("VAGEN_MAX_CUBES", "8"))
DEFAULT_EE_BODY_NAME = os.getenv("VAGEN_IK_EE_BODY_NAME", "panda_link7") or "panda_link7"
USE_EE_FRAME_TCP = os.getenv("VAGEN_USE_EE_FRAME_TCP", "1").strip().lower() not in {"0", "false", "off", "no"}


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


def _try_ee_frame_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not USE_EE_FRAME_TCP:
        return None
    try:
        ee_frame: FrameTransformer = env.scene["ee_frame"]
    except Exception:
        return None
    try:
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    except Exception:
        return None
    return ee_pos_w, ee_quat_w


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
    if isinstance(cached_names, tuple):
        return list(cached_names)

    pattern = re.compile(rf"^{re.escape(cube_name_prefix)}(\d+)$")
    indexed_names: list[tuple[int, str]] = []
    for attr_name in dir(env.scene):
        match = pattern.match(attr_name)
        if match is None:
            continue
        cube_idx = int(match.group(1))
        if 1 <= cube_idx <= max_cubes and hasattr(env.scene, attr_name):
            indexed_names.append((cube_idx, attr_name))

    indexed_names.sort(key=lambda item: item[0])
    cube_names = [name for _, name in indexed_names]
    if not cube_names:
        # Fallback for scenes that expose cubes only through dynamic attributes.
        for cube_idx in range(1, max_cubes + 1):
            cube_name = f"{cube_name_prefix}{cube_idx}"
            if hasattr(env.scene, cube_name):
                cube_names.append(cube_name)

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


def magic_suction_command(env: ManagerBasedRLEnv, default_open_cmd: float = 1.0) -> torch.Tensor:
    """Latest per-env magic suction command written by action terms."""
    device, dtype = _obs_device_and_dtype(env)
    cmd = getattr(env.unwrapped, "_vagen_magic_suction_cmd", None)
    if isinstance(cmd, torch.Tensor) and cmd.numel() >= env.num_envs:
        return cmd[: env.num_envs].to(device=device, dtype=dtype).reshape(env.num_envs, 1)
    return torch.full((env.num_envs, 1), float(default_open_cmd), device=device, dtype=dtype)


def magic_suction_closed_flag(env: ManagerBasedRLEnv, close_command_threshold: float = 0.0) -> torch.Tensor:
    """Binary flag (1=close command active) derived from magic suction command."""
    cmd = magic_suction_command(env)
    return (cmd < float(close_command_threshold)).to(dtype=cmd.dtype)


def privileged_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = DEFAULT_EE_BODY_NAME,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    close_command_threshold: float = 0.0,
) -> torch.Tensor:
    """Privileged state vector with robot, object, and suction state."""
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    env_origin_obs = env_origin(env=env).to(dtype=joint_pos.dtype)
    ee_pos_obs = ee_pos(env=env, robot_cfg=robot_cfg, ee_body_name=ee_body_name)
    ee_quat_obs = ee_quat(env=env, robot_cfg=robot_cfg, ee_body_name=ee_body_name)

    cube_pos = all_cube_positions_in_world_frame(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
    ).to(dtype=joint_pos.dtype)
    cube_quat = all_cube_orientations_in_world_frame(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
    ).to(dtype=joint_pos.dtype)
    suction_cmd = magic_suction_command(env).to(dtype=joint_pos.dtype)
    suction_closed = magic_suction_closed_flag(
        env=env, close_command_threshold=close_command_threshold
    ).to(dtype=joint_pos.dtype)

    return torch.cat(
        (
            env_origin_obs,
            root_pos,
            root_quat,
            joint_pos,
            joint_vel,
            ee_pos_obs,
            ee_quat_obs,
            cube_pos,
            cube_quat,
            suction_cmd,
            suction_closed,
        ),
        dim=1,
    )


def ee_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = DEFAULT_EE_BODY_NAME,
) -> torch.Tensor:
    """End-effector position in world frame from ee_frame target pose or articulation body state."""
    ee_frame_pose = _try_ee_frame_pose(env)
    if ee_frame_pose is not None:
        ee_pos_w, _ = ee_frame_pose
        return ee_pos_w

    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, body_names = robot.find_bodies(ee_body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError(f"Expected one body for '{ee_body_name}', got {len(body_ids)}: {body_names}.")
    body_idx = int(body_ids[0])
    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    return ee_pos_w


def ee_quat(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = DEFAULT_EE_BODY_NAME,
) -> torch.Tensor:
    """End-effector quaternion (wxyz) in world frame from ee_frame target pose or articulation body state."""
    ee_frame_pose = _try_ee_frame_pose(env)
    if ee_frame_pose is not None:
        _, ee_quat_w = ee_frame_pose
        return ee_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, body_names = robot.find_bodies(ee_body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError(f"Expected one body for '{ee_body_name}', got {len(body_ids)}: {body_names}.")
    body_idx = int(body_ids[0])
    ee_quat_w = robot.data.body_quat_w[:, body_idx]
    return ee_quat_w
