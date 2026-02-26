# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .cameras import STACK_CAMERA_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_MAX_CUBES = int(os.getenv("VAGEN_MAX_CUBES", "8"))
DEFAULT_CAMERA_NAMES = STACK_CAMERA_NAMES


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


def _obs_step_cache(env: ManagerBasedRLEnv) -> dict:
    env_unwrapped = getattr(env, "unwrapped", env)
    step_idx = int(getattr(env_unwrapped, "common_step_counter", -1))
    cache = getattr(env_unwrapped, "_assembling_obs_step_cache", None)
    if not isinstance(cache, dict) or cache.get("_step_idx") != step_idx:
        cache = {"_step_idx": step_idx}
        setattr(env_unwrapped, "_assembling_obs_step_cache", cache)
    return cache


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


def _read_cube_pose_world(
    env: ManagerBasedRLEnv,
    cube_name: str,
    cube_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cube_asset = env.scene[cube_name]
    if isinstance(cube_asset, RigidObject):
        return cube_asset.data.root_pos_w[:, :3], cube_asset.data.root_quat_w[:, :4]

    if isinstance(cube_asset, RigidObjectCollection):
        env_ids = torch.arange(env.num_envs, device=cube_asset.data.object_pos_w.device, dtype=torch.long)
        object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
        cube_pos = cube_asset.data.object_pos_w[env_ids, object_indices, :3]
        cube_quat = cube_asset.data.object_quat_w[env_ids, object_indices, :4]
        return cube_pos, cube_quat

    cube_data = cube_asset.data
    if hasattr(cube_data, "root_pos_w") and hasattr(cube_data, "root_quat_w"):
        return cube_data.root_pos_w[:, :3], cube_data.root_quat_w[:, :4]
    if hasattr(cube_data, "object_pos_w") and hasattr(cube_data, "object_quat_w"):
        env_ids = torch.arange(env.num_envs, device=cube_data.object_pos_w.device, dtype=torch.long)
        object_indices = _resolve_collection_object_indices(env, default_index=cube_slot)
        cube_pos = cube_data.object_pos_w[env_ids, object_indices, :3]
        cube_quat = cube_data.object_quat_w[env_ids, object_indices, :4]
        return cube_pos, cube_quat
    raise TypeError(f"Unsupported cube asset type for '{cube_name}': {type(cube_asset)}")


def _collect_cube_pose_batch(
    env: ManagerBasedRLEnv,
    max_cubes: int,
    cube_name_prefix: str,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step_cache = _obs_step_cache(env)
    cache_key = ("cube_pose_batch", int(max_cubes), cube_name_prefix, float(fill_value))
    cached = step_cache.get(cache_key, None)
    if isinstance(cached, tuple) and len(cached) == 3:
        return cached

    device, dtype = _obs_device_and_dtype(env)
    cube_pos_w = torch.full((env.num_envs, max_cubes, 3), float(fill_value), device=device, dtype=dtype)
    cube_quat_w = torch.full((env.num_envs, max_cubes, 4), float(fill_value), device=device, dtype=dtype)
    cube_mask = torch.zeros((env.num_envs, max_cubes), device=device, dtype=dtype)

    cube_names = _iter_cube_names(env, max_cubes=max_cubes, cube_name_prefix=cube_name_prefix)
    for cube_slot, cube_name in enumerate(cube_names):
        pos_w, quat_w = _read_cube_pose_world(env, cube_name=cube_name, cube_slot=cube_slot)
        cube_pos_w[:, cube_slot, :] = pos_w.to(device=device, dtype=dtype)
        cube_quat_w[:, cube_slot, :] = quat_w.to(device=device, dtype=dtype)
        cube_mask[:, cube_slot] = 1.0

    step_cache[cache_key] = (cube_pos_w, cube_quat_w, cube_mask)
    return cube_pos_w, cube_quat_w, cube_mask


def cube_availability_mask(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
) -> torch.Tensor:
    """Binary mask for which cube slots are available in the current scene."""
    _, _, mask = _collect_cube_pose_batch(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
        fill_value=-1.0,
    )
    return mask


def all_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    fill_value: float = -1.0,
    subtract_env_origins: bool = False,
) -> torch.Tensor:
    """Flattened positions of all configured cubes in the scene."""
    cube_pos_w, _, _ = _collect_cube_pose_batch(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
        fill_value=fill_value,
    )
    cube_pos = cube_pos_w - env.scene.env_origins.unsqueeze(1) if subtract_env_origins else cube_pos_w
    return cube_pos.reshape(env.num_envs, max_cubes * 3)


def all_cube_positions_in_env_frame(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Flattened positions of all configured cubes in each env local frame."""
    return all_cube_positions_in_world_frame(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
        fill_value=fill_value,
        subtract_env_origins=True,
    )


def all_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Flattened orientations of all configured cubes in the world frame."""
    _, cube_quat, _ = _collect_cube_pose_batch(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
        fill_value=fill_value,
    )
    return cube_quat.reshape(env.num_envs, max_cubes * 4)


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


def camera_availability_mask(
    env: ManagerBasedRLEnv,
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
) -> torch.Tensor:
    """Binary mask for available cameras in configured order."""
    step_cache = _obs_step_cache(env)
    cache_key = ("camera_mask", tuple(camera_names))
    cached = step_cache.get(cache_key, None)
    if isinstance(cached, torch.Tensor):
        return cached

    device, dtype = _obs_device_and_dtype(env)
    mask = torch.zeros((env.num_envs, len(camera_names)), device=device, dtype=dtype)
    for cam_idx, camera_name in enumerate(camera_names):
        if hasattr(env.scene, camera_name):
            sensor = getattr(env.scene, camera_name, None)
            if sensor is not None:
                mask[:, cam_idx] = 1.0
    step_cache[cache_key] = mask
    return mask


def _camera_resolution_xy(sensor) -> tuple[float | None, float | None]:
    sensor_cfg = getattr(sensor, "cfg", None)
    width = getattr(sensor_cfg, "width", None)
    height = getattr(sensor_cfg, "height", None)
    if width is not None and height is not None:
        return float(width), float(height)

    sensor_data = getattr(sensor, "data", None)
    output_dict = getattr(sensor_data, "output", None)
    if isinstance(output_dict, dict):
        rgb = output_dict.get("rgb", None)
        if isinstance(rgb, torch.Tensor) and rgb.ndim >= 3:
            return float(rgb.shape[2]), float(rgb.shape[1])
    return None, None


def camera_parameters(
    env: ManagerBasedRLEnv,
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
    fill_value: float = -1.0,
    include_resolution: bool = True,
) -> torch.Tensor:
    """Flattened camera intrinsics/extrinsics for all requested cameras.

    Per camera layout is:
    ``[K(9), pos_local(3), quat_world(4), resolution_xy(2, optional), valid(1)]``.
    """
    step_cache = _obs_step_cache(env)
    cache_key = ("camera_params", tuple(camera_names), float(fill_value), bool(include_resolution))
    cached = step_cache.get(cache_key, None)
    if isinstance(cached, torch.Tensor):
        return cached

    device, dtype = _obs_device_and_dtype(env)
    per_camera_dim = 9 + 3 + 4 + (2 if include_resolution else 0) + 1
    params = torch.full(
        (env.num_envs, len(camera_names), per_camera_dim),
        float(fill_value),
        device=device,
        dtype=dtype,
    )
    params[:, :, per_camera_dim - 1] = 0.0

    for cam_idx, camera_name in enumerate(camera_names):
        if not hasattr(env.scene, camera_name):
            continue
        sensor = env.scene[camera_name]
        if sensor is None:
            continue
        sensor_data = getattr(sensor, "data", None)
        if sensor_data is None:
            continue

        offset = 0
        intrinsic = getattr(sensor_data, "intrinsic_matrices", None)
        if isinstance(intrinsic, torch.Tensor) and intrinsic.shape[0] == env.num_envs:
            params[:, cam_idx, offset : offset + 9] = intrinsic.reshape(env.num_envs, 9).to(device=device, dtype=dtype)
        offset += 9

        cam_pos_w = getattr(sensor_data, "pos_w", None)
        cam_quat_w = getattr(sensor_data, "quat_w_world", None)
        if cam_quat_w is None:
            cam_quat_w = getattr(sensor_data, "quat_w", None)
        if isinstance(cam_pos_w, torch.Tensor) and isinstance(cam_quat_w, torch.Tensor):
            params[:, cam_idx, offset : offset + 3] = (cam_pos_w - env.scene.env_origins).to(device=device, dtype=dtype)
            params[:, cam_idx, offset + 3 : offset + 7] = cam_quat_w.to(device=device, dtype=dtype)
        offset += 7

        if include_resolution:
            cam_width, cam_height = _camera_resolution_xy(sensor)
            if cam_width is not None and cam_height is not None:
                params[:, cam_idx, offset] = float(cam_width)
                params[:, cam_idx, offset + 1] = float(cam_height)
            offset += 2

        params[:, cam_idx, offset] = 1.0

    flattened = params.reshape(env.num_envs, len(camera_names) * per_camera_dim)
    step_cache[cache_key] = flattened
    return flattened


def privileged_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_cubes: int = DEFAULT_MAX_CUBES,
    cube_name_prefix: str = "cube_",
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
    close_command_threshold: float = 0.0,
) -> torch.Tensor:
    """Privileged state vector with robot, object, suction, and camera parameters."""
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_quat = robot.data.root_quat_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    ee_pos = ee_frame_pos(env, ee_frame_cfg=ee_frame_cfg)
    ee_quat = ee_frame_quat(env, ee_frame_cfg=ee_frame_cfg)

    cube_pos = all_cube_positions_in_env_frame(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
    ).to(dtype=joint_pos.dtype)
    cube_quat = all_cube_orientations_in_world_frame(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
    ).to(dtype=joint_pos.dtype)
    cube_mask = cube_availability_mask(
        env=env,
        max_cubes=max_cubes,
        cube_name_prefix=cube_name_prefix,
    ).to(dtype=joint_pos.dtype)
    suction_cmd = magic_suction_command(env).to(dtype=joint_pos.dtype)
    suction_closed = magic_suction_closed_flag(
        env=env, close_command_threshold=close_command_threshold
    ).to(dtype=joint_pos.dtype)

    cam_params = camera_parameters(env=env, camera_names=camera_names).to(dtype=joint_pos.dtype)
    cam_mask = camera_availability_mask(env=env, camera_names=camera_names).to(dtype=joint_pos.dtype)

    return torch.cat(
        (
            root_pos,
            root_quat,
            joint_pos,
            joint_vel,
            ee_pos,
            ee_quat,
            cube_pos,
            cube_quat,
            cube_mask,
            suction_cmd,
            suction_closed,
            cam_params,
            cam_mask,
        ),
        dim=1,
    )


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_pos_w, cube_2.data.root_pos_w, cube_3.data.root_pos_w), dim=1)


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)

    return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
    """The orientation of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_quat_w, cube_2.data.root_quat_w, cube_3.data.root_quat_w), dim=1)


def instance_randomize_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = cube_1.data.root_pos_w
    cube_1_quat_w = cube_1.data.root_quat_w

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            # Fallback for suction tools that do not expose finger joints.
            action_manager = getattr(env, "action_manager", None)
            terms = getattr(action_manager, "_terms", None)
            if isinstance(terms, dict) and "gripper_action" in terms:
                cmd = getattr(terms["gripper_action"], "command", None)
                if isinstance(cmd, torch.Tensor):
                    return cmd.clone()
            return torch.zeros((env.num_envs, 1), device=env.device, dtype=robot.data.joint_pos.dtype)


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
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def cube_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the cubes in the robot base frame."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_cube_1_world = cube_1.data.root_pos_w
    pos_cube_2_world = cube_2.data.root_pos_w
    pos_cube_3_world = cube_3.data.root_pos_w

    quat_cube_1_world = cube_1.data.root_quat_w
    quat_cube_2_world = cube_2.data.root_quat_w
    quat_cube_3_world = cube_3.data.root_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_1_world, quat_cube_1_world
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_2_world, quat_cube_2_world
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_3_world, quat_cube_3_world
    )

    pos_cubes_base = torch.cat((pos_cube_1_base, pos_cube_2_base, pos_cube_3_base), dim=1)
    quat_cubes_base = torch.cat((quat_cube_1_base, quat_cube_2_base, quat_cube_3_base), dim=1)

    if return_key == "pos":
        return pos_cubes_base
    elif return_key == "quat":
        return quat_cubes_base
    else:
        return torch.cat((pos_cubes_base, quat_cubes_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations,
    and add abs gripper pos and quat in robot base frame
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper pos,
        gripper quat,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    cube_1_pos_w = cube_1.data.root_pos_w
    cube_1_quat_w = cube_1.data.root_quat_w

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_1_pos_w, cube_1_quat_w
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_2_pos_w, cube_2_quat_w
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_3_pos_w, cube_3_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_cube_1_base,
            quat_cube_1_base,
            pos_cube_2_base,
            quat_cube_2_base,
            pos_cube_3_base,
            quat_cube_3_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    else:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
