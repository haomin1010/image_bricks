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


def _build_aligned_cube_poses(
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
    print(f"[DBG CUBE_LAYOUT] aligned cube poses: {aligned_poses}")
    return aligned_poses


def _apply_cube_layout(
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
) -> None:
    env_ids_t = _resolve_event_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    if cube_names:
        resolved_cube_names = list(cube_names)
    elif int(max_cubes) > 0:
        resolved_cube_names = [f"{cube_name_prefix}{i + 1}" for i in range(int(max_cubes))]
    else:
        raise ValueError("cube_names must be provided or max_cubes must be > 0")

    scene_cfg = env.cfg.scene
    grid_origin, grid_size, cell_size, _ = scene_cfg.get_grid_spec()
    half_width = (int(grid_size) + 1) * float(cell_size) / 2.0
    source_pick_pos_x = 0.5
    source_pick_pos_y = -0.35
    aligned_poses = _build_aligned_cube_poses(
        max_cubes=len(resolved_cube_names),
        cube_size=float(cube_size),
        source_pick_pos_x=source_pick_pos_x,
        source_pick_pos_y=source_pick_pos_y,
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
        max_cubes = max(1, int(os.getenv("VAGEN_MAX_CUBES", "2")))
        grid_origin, grid_size, cell_size, _ = self.scene.get_grid_spec()
        blue_usd = self.scene.resolve_asset_path(
            local_rel="Props/Blocks/blue_block.usd",
            nucleus_rel="Props/Blocks/blue_block.usd",
        )
        template_cube_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -10.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=blue_usd,
                scale=getattr(self, "cube_scale", (1.0, 1.0, 1.0)),
                rigid_props=getattr(self, "cube_properties", None),
                mass_props=getattr(self, "cube_mass_props", None),
            ),
        )
        self.scene.cube_1 = template_cube_cfg
        print(
            "[DBG CUBE_LAYOUT] "
            f"grid_size={int(grid_size)} "
            f"resolved_max_cubes={int(max_cubes)}"
        )
        cube_names = [f"cube_{i + 1}" for i in range(int(max_cubes))]
        half_width = (int(grid_size) + 1) * float(cell_size) / 2.0
        source_pick_pos_x = 0.5
        source_pick_pos_y = -0.35

        aligned_poses = _build_aligned_cube_poses(
            max_cubes=int(max_cubes),
            cube_size=0.045,
            source_pick_pos_x=source_pick_pos_x,
            source_pick_pos_y=source_pick_pos_y,
        )
        _apply_cube_layout(
            self.scene,
            template_cube_cfg=template_cube_cfg,
            cube_names=cube_names,
            aligned_poses=aligned_poses,
            cube_usd_path=blue_usd,
        )
