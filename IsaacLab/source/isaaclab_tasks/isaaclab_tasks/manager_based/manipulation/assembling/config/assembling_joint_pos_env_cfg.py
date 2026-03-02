# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import os

import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import (
    ASSEMBLING_MAX_CUBES,
    AssemblingEnvCfg,
    EventsCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.assembling.mdp.terminations import task_index_exceeds_max_cubes

ASSEMBLING_TASK_ID = "multipicture_assembling_from_begin"


def _resolve_max_cubes(max_cubes: int | None = None) -> int:
    return max(3, int(os.getenv("VAGEN_MAX_CUBES", "8"))) if max_cubes is None else max(3, int(max_cubes))


def _build_cube_names(max_cubes: int) -> list[str]:
    return [f"cube_{i + 1}" for i in range(int(max_cubes))]


def _build_aligned_cube_poses(
    *,
    max_cubes: int,
    cube_size: float,
    source_pick_pos_x: float,
    source_pick_pos_y: float,
) -> list[list[float]]:
    visible_z = cube_size / 2.0
    aligned_poses: list[list[float]] = [
        [source_pick_pos_x, source_pick_pos_y, visible_z, 1.0, 0.0, 0.0, 0.0]
    ]
    if max_cubes <= 1:
        return aligned_poses

    hidden_center_x = float(os.getenv("VAGEN_CUBE_PARKING_CENTER_X", "-0.60"))
    hidden_center_y = float(os.getenv("VAGEN_CUBE_PARKING_CENTER_Y", "0.00"))
    hidden_z = float(os.getenv("VAGEN_CUBE_PARKING_Z", "-1.00"))
    default_spacing = max(float(cube_size) + 0.01, 0.055)
    spacing = float(os.getenv("VAGEN_CUBE_PARKING_SPACING", str(default_spacing)))

    num_hidden = max_cubes - 1
    cols = max(1, int(math.ceil(math.sqrt(num_hidden))))
    rows = int(math.ceil(num_hidden / cols))
    x0 = hidden_center_x - (cols - 1) * spacing / 2.0
    y0 = hidden_center_y - (rows - 1) * spacing / 2.0

    for hidden_idx in range(num_hidden):
        # Keep cube_1/cube_2/cube_3 visible on table, aligned in one row (same y).
        if hidden_idx < 2:
            x = source_pick_pos_x + float(hidden_idx + 1) * spacing
            y = source_pick_pos_y
            z = visible_z
        else:
            row = hidden_idx // cols
            col = hidden_idx % cols
            x = x0 + col * spacing
            y = y0 + row * spacing
            z = hidden_z
        aligned_poses.append([x, y, z, 1.0, 0.0, 0.0, 0.0])
    return aligned_poses


def _resolve_cube_template(scene_cfg, cube_names: list[str]):
    for name in cube_names:
        cube_cfg = getattr(scene_cfg, name, None)
        if cube_cfg is not None:
            return cube_cfg
    cube_1_cfg = getattr(scene_cfg, "cube_1", None)
    if cube_1_cfg is not None:
        return cube_1_cfg
    raise AttributeError("Scene does not define a base cube config (expected at least 'cube_1').")


def _apply_cube_layout(
    scene_cfg,
    *,
    cube_names: list[str],
    aligned_poses: list[list[float]],
    cube_usd_path: str,
) -> None:
    hidden_pos = (0.0, 0.0, -10.0)
    for name in cube_names:
        cube_cfg = getattr(scene_cfg, name, None)
        if cube_cfg is not None:
            cube_cfg.init_state.pos = hidden_pos

    template_cube_cfg = _resolve_cube_template(scene_cfg, cube_names)

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


def _configure_cubes(
    scene_cfg,
    *,
    cube_size: float,
    grid_origin,
    cell_size: float,
    grid_size_for_source: int,
    max_cubes: int | None = None,
) -> list[str]:
    resolved_max_cubes = _resolve_max_cubes(max_cubes)
    cube_names = _build_cube_names(resolved_max_cubes)
    half_width = grid_size_for_source * cell_size / 2.0
    # Move source pile beside the grid at optimal reach distance (X=0.5)
    source_pick_pos_x = grid_origin[0]
    source_pick_pos_y = grid_origin[1] - half_width - cell_size / 2.0

    blue_usd = scene_cfg.resolve_asset_path(
        local_rel="Props/Blocks/blue_block.usd",
        nucleus_rel="Props/Blocks/blue_block.usd",
    )
    aligned_poses = _build_aligned_cube_poses(
        max_cubes=resolved_max_cubes,
        cube_size=cube_size,
        source_pick_pos_x=source_pick_pos_x,
        source_pick_pos_y=source_pick_pos_y,
    )
    _apply_cube_layout(
        scene_cfg,
        cube_names=cube_names,
        aligned_poses=aligned_poses,
        cube_usd_path=blue_usd,
    )
    return cube_names


def _resolve_event_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def _place_cubes_event(
    env,
    env_ids,
    *,
    cube_names: list[str] | None = None,
    cube_name_prefix: str = "cube_",
    max_cubes: int = 0,
    cube_size: float = 0.045,
    source_grid_size: int | None = None,
) -> None:
    env_ids_t = _resolve_event_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return

    if cube_names:
        resolved_cube_names = list(cube_names)
    else:
        resolved_max_cubes = _resolve_max_cubes(max_cubes if max_cubes > 0 else None)
        resolved_cube_names = [f"{cube_name_prefix}{i + 1}" for i in range(resolved_max_cubes)]
    if not resolved_cube_names:
        return

    scene_cfg = getattr(env.cfg, "scene", None)
    if scene_cfg is not None and hasattr(scene_cfg, "get_grid_spec"):
        grid_origin, grid_size, cell_size, _ = scene_cfg.get_grid_spec()
    else:
        grid_origin = [0.5, 0.0, 0.001]
        cell_size = 0.056
    grid_size = len(resolved_cube_names)

    source_size = int(source_grid_size) if source_grid_size is not None else int(grid_size)
    half_width = source_size * cell_size / 2.0
    # Move source pile beside the grid at optimal reach distance (X=0.5)
    source_pick_pos_x = grid_origin[0]
    source_pick_pos_y = grid_origin[1] - half_width - cell_size / 2.0
    aligned_poses = _build_aligned_cube_poses(
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
        if not hasattr(env.scene, cube_name):
            continue
        cube_asset = env.scene[cube_name]
        pose = aligned_poses[idx]
        pos_local = torch.tensor(pose[0:3], device=device, dtype=dtype)
        quat_w = torch.tensor(pose[3:7], device=device, dtype=dtype).unsqueeze(0).repeat(env_ids_t.numel(), 1)
        pos_w = env_origins + pos_local.unsqueeze(0)
        root_pose = torch.cat([pos_w, quat_w], dim=-1)
        cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_i32)
        cube_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_i32)


@configclass
class AssemblingTerminationsCfg(TerminationsCfg):
    """Termination terms for assembling runtime."""

    max_cube_exceeded = DoneTerm(
        func=task_index_exceeds_max_cubes,
        params={"max_cubes": int(ASSEMBLING_MAX_CUBES)},
    )


@configclass
class EventCfgAssembling(EventsCfg):
    """Event terms for assembling runtime."""

    setup_cubes_startup = EventTerm(
        func=_place_cubes_event,
        mode="startup",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", "0.045")),
        },
    )
    setup_cubes_reset = EventTerm(
        func=_place_cubes_event,
        mode="reset",
        params={
            "cube_name_prefix": "cube_",
            "max_cubes": int(ASSEMBLING_MAX_CUBES),
            "cube_size": float(os.getenv("VAGEN_CUBE_SIZE", "0.045")),
        },
    )


@configclass
class AssemblingCubeStackEnvCfg(AssemblingEnvCfg):
    """Default assembling stack environment."""

    observations: ObservationsCfg = ObservationsCfg()
    terminations: AssemblingTerminationsCfg = AssemblingTerminationsCfg()
    events: EventCfgAssembling = EventCfgAssembling()

    def __post_init__(self):
        super().__post_init__()

        self._configure_cubes_for_server()

    def _configure_cubes_for_server(self):
        max_cubes = max(3, int(os.getenv("VAGEN_MAX_CUBES", "8")))
        grid_origin = [0.5, 0.0, 0.001]
        line_thickness = 0.001
        cell_size = 0.055 + line_thickness
        if not hasattr(self.scene, "cube_1") or getattr(self.scene, "cube_1") is None:
            blue_usd = self.scene.resolve_asset_path(
                local_rel="Props/Blocks/blue_block.usd",
                nucleus_rel="Props/Blocks/blue_block.usd",
            )
            self.scene.cube_1 = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -10.0], rot=[1.0, 0.0, 0.0, 0.0]),
                spawn=UsdFileCfg(
                    usd_path=blue_usd,
                    scale=getattr(self, "cube_scale", (1.0, 1.0, 1.0)),
                    rigid_props=getattr(self, "cube_properties", None),
                ),
            )
        _configure_cubes(
            self.scene,
            cube_size=0.045,
            grid_origin=grid_origin,
            cell_size=cell_size,
            grid_size_for_source=max_cubes,
            max_cubes=max_cubes,
        )
