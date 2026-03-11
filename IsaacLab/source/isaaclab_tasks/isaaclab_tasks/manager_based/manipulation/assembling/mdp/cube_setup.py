# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math

import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

DEFAULT_CUBE_SIZE = 0.0203 * 2.0


def build_default_cube_spawn_settings(
    *,
    mass: float = 0.02,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[RigidBodyPropertiesCfg, MassPropertiesCfg, tuple[float, float, float]]:
    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    cube_mass_props = MassPropertiesCfg(mass=float(mass))
    cube_scale = (float(scale[0]), float(scale[1]), float(scale[2]))
    return cube_properties, cube_mass_props, cube_scale


def build_aligned_cube_poses(
    *,
    max_cubes: int,
    cube_size: float,
    source_pick_pos_x: float,
    source_pick_pos_y: float,
) -> list[list[float]]:
    z = cube_size / 2.0
    hidden_center_x = float(source_pick_pos_x)
    hidden_center_y = float(source_pick_pos_y)
    spacing = float(cube_size) + 0.01

    cols = int(math.ceil(math.sqrt(max_cubes)))
    rows = int(math.ceil(max_cubes / cols))
    x0 = hidden_center_x - (cols - 1) * spacing / 2.0
    y0 = hidden_center_y - (rows - 1) * spacing / 2.0

    aligned_poses: list[list[float]] = []
    for cube_idx in range(max_cubes):
        row = cube_idx // cols
        col = cube_idx % cols
        x = x0 + col * spacing
        y = y0 + row * spacing
        aligned_poses.append([x, y, z, 1.0, 0.0, 0.0, 0.0])
    return aligned_poses


def apply_cube_layout(
    scene_cfg,
    *,
    template_cube_cfg,
    cube_names: list[str],
    aligned_poses: list[list[float]],
    cube_usd_path: str,
) -> None:
    for idx, name in enumerate(cube_names):
        pos = aligned_poses[idx]
        cube_cfg = getattr(scene_cfg, name, None)
        if cube_cfg is None:
            cube_cfg = copy.deepcopy(template_cube_cfg)
            cube_cfg.prim_path = f"{{ENV_REGEX_NS}}/Cube_{idx + 1}"
            setattr(scene_cfg, name, cube_cfg)

        cube_cfg.spawn.usd_path = cube_usd_path
        if hasattr(cube_cfg.spawn, "semantic_tags"):
            cube_cfg.spawn.semantic_tags = [("class", name)]
        cube_cfg.init_state.pos = (pos[0], pos[1], pos[2])
        cube_cfg.init_state.rot = (pos[3], pos[4], pos[5], pos[6])


def resolve_event_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def place_cubes_event(
    env,
    env_ids,
    *,
    cube_names: list[str] | None = None,
    cube_name_prefix: str = "cube_",
    max_cubes: int = 0,
    cube_size: float = DEFAULT_CUBE_SIZE,
    source_pick_pos_x: float = 0.5,
    source_pick_pos_y: float = -0.35,
) -> None:
    env_ids_t = resolve_event_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    if cube_names:
        resolved_cube_names = list(cube_names)
    elif int(max_cubes) > 0:
        resolved_cube_names = [f"{cube_name_prefix}{i + 1}" for i in range(int(max_cubes))]
    else:
        raise ValueError("cube_names must be provided or max_cubes must be > 0")

    aligned_poses = build_aligned_cube_poses(
        max_cubes=len(resolved_cube_names),
        cube_size=float(cube_size),
        source_pick_pos_x=float(source_pick_pos_x),
        source_pick_pos_y=float(source_pick_pos_y),
    )

    env_origins = env.scene.env_origins[env_ids_t]
    device = env_origins.device
    dtype = env_origins.dtype
    env_ids_i32 = env_ids_t.to(dtype=torch.int32)

    zero_vel = torch.zeros((env_ids_t.numel(), 6), device=device, dtype=dtype)
    for idx, cube_name in enumerate(resolved_cube_names):
        try:
            cube_asset = env.scene[cube_name]
        except KeyError:
            continue
        pose = aligned_poses[idx]
        pos_local = torch.tensor(pose[0:3], device=device, dtype=dtype)
        quat_w = torch.tensor(pose[3:7], device=device, dtype=dtype).unsqueeze(0).repeat(env_ids_t.numel(), 1)
        pos_w = env_origins + pos_local.unsqueeze(0)
        root_pose = torch.cat([pos_w, quat_w], dim=-1)
        cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_i32)
        cube_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_i32)


def apply_policy_observation_overrides(policy_obs, *, max_cubes: int) -> None:
    if policy_obs is None:
        return

    cube_pos = getattr(policy_obs, "cube_pos", None)
    if cube_pos is not None and hasattr(cube_pos, "params"):
        cube_pos.params["max_cubes"] = int(max_cubes)

    cube_quat = getattr(policy_obs, "cube_quat", None)
    if cube_quat is not None and hasattr(cube_quat, "params"):
        cube_quat.params["max_cubes"] = int(max_cubes)

    ee_pos = getattr(policy_obs, "ee_pos", None)
    if ee_pos is not None and hasattr(ee_pos, "params"):
        ee_pos.params.pop("ee_body_name", None)

    ee_quat = getattr(policy_obs, "ee_quat", None)
    if ee_quat is not None and hasattr(ee_quat, "params"):
        ee_quat.params.pop("ee_body_name", None)

    grasped = getattr(policy_obs, "grasped", None)
    if grasped is not None and hasattr(grasped, "params"):
        grasped.params.pop("ee_body_name", None)


def configure_server_cube_layout(
    scene_cfg,
    *,
    cube_properties: RigidBodyPropertiesCfg,
    cube_mass_props: MassPropertiesCfg,
    cube_scale: tuple[float, float, float],
    max_cubes: int,
    cube_size: float = DEFAULT_CUBE_SIZE,
    cube_name_prefix: str = "cube_",
    source_pick_pos_x: float = 0.5,
    source_pick_pos_y: float = -0.33,
    collision_enabled: bool | None = None,
) -> list[str]:
    _, grid_size, _, _ = scene_cfg.get_grid_spec()
    blue_usd = scene_cfg.resolve_asset_path(
        local_rel="Props/Blocks/blue_block.usd",
        nucleus_rel="Props/Blocks/blue_block.usd",
    )
    template_cube_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -10.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=blue_usd,
            scale=tuple(cube_scale),
            rigid_props=cube_properties,
            mass_props=cube_mass_props,
            collision_props=(
                None if collision_enabled is None else CollisionPropertiesCfg(collision_enabled=bool(collision_enabled))
            ),
        ),
    )
    setattr(scene_cfg, f"{cube_name_prefix}1", template_cube_cfg)

    resolved_max_cubes = max(1, int(max_cubes))
    cube_names = [f"{cube_name_prefix}{i + 1}" for i in range(resolved_max_cubes)]

    aligned_poses = build_aligned_cube_poses(
        max_cubes=resolved_max_cubes,
        cube_size=float(cube_size),
        source_pick_pos_x=float(source_pick_pos_x),
        source_pick_pos_y=float(source_pick_pos_y),
    )
    apply_cube_layout(
        scene_cfg,
        template_cube_cfg=template_cube_cfg,
        cube_names=cube_names,
        aligned_poses=aligned_poses,
        cube_usd_path=blue_usd,
    )
    return cube_names
