# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _scene_has_entity(scene, name: str) -> bool:
    if hasattr(scene, name):
        return True
    try:
        scene[name]
        return True
    except Exception:
        return False


def _discover_cube_names(scene, cube_name_prefix: str, max_cubes: int) -> list[str]:
    pattern = re.compile(rf"^{re.escape(cube_name_prefix)}(\d+)$", re.IGNORECASE)
    fallback_pattern = re.compile(r"^cube_(\d+)$", re.IGNORECASE)
    names_and_indices: list[tuple[int, str]] = []

    candidates: list[str] = []
    keys_fn = getattr(scene, "keys", None)
    if callable(keys_fn):
        try:
            candidates.extend(list(keys_fn()))
        except Exception:
            pass
    candidates.extend(list(dir(scene)))

    for raw_name in candidates:
        if not isinstance(raw_name, str):
            continue
        short_name = raw_name.rsplit("/", 1)[-1]
        match = pattern.match(short_name) or fallback_pattern.match(short_name)
        if match is None:
            continue

        cube_idx = int(match.group(1))
        if cube_idx < 1:
            continue
        if max_cubes > 0 and cube_idx > max_cubes:
            continue

        candidate_names = (
            raw_name,
            short_name,
            f"{cube_name_prefix}{cube_idx}",
            f"Cube_{cube_idx}",
            f"cube_{cube_idx}",
        )
        resolved_name = next((name for name in candidate_names if _scene_has_entity(scene, name)), None)
        if resolved_name is not None:
            names_and_indices.append((cube_idx, resolved_name))

    names_and_indices = sorted(set(names_and_indices), key=lambda item: item[0])
    return [name for _, name in names_and_indices]


class TeleportPendingCubesEvent(ManagerTermBase):
    """Event term that owns lifecycle of teleport callback controller."""

    cfg: EventTermCfg

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._controller: VagenTeleportPendingCubesController | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | None,
        cube_names: list[str] | None = None,
        cube_name_prefix: str = "cube_",
        max_cubes: int = 0,
        cube_size: float = 0.045,
    ):
        del env_ids
        if self._controller is not None:
            return

        resolved_cube_names = list(cube_names) if cube_names else _discover_cube_names(
            scene=env.scene,
            cube_name_prefix=cube_name_prefix,
            max_cubes=int(max_cubes),
        )
        if not resolved_cube_names:
            print("[WARN]: TeleportPendingCubesEvent did not find cube assets; callback not registered.")
            return

        self._controller = VagenTeleportPendingCubesController(
            env=env,
            cube_names=resolved_cube_names,
            cube_size=float(cube_size),
        )
        env.unwrapped._vagen_teleport_pending_cubes_controller = self._controller

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._controller is None:
            return
        self._controller.reset(env_ids=env_ids)

    def __del__(self):
        if self._controller is None:
            return
        try:
            self._controller.close()
        except Exception:
            pass


class VagenTeleportPendingCubesController:
    """Teleport runtime controller driven by per-physics-step callback."""

    def __init__(
        self,
        env,
        cube_names: list[str],
        cube_size: float,
    ):
        self.env = env
        self.scene = env.unwrapped.scene
        self.device = env.unwrapped.device
        self.cube_names = list(cube_names)
        self.cube_size = float(cube_size)
        self._warned_missing_state = False

        self._cb_name = f"vagen_teleport_pending_cubes_{id(self)}"
        self._cb_registered = False
        try:
            env.unwrapped.sim.add_physics_callback(self._cb_name, self.physics_step_cb)
            self._cb_registered = True
            print(f"[INFO]: Teleport callback registered: name={self._cb_name}")
        except Exception as exc:
            print(f"[WARN]: Failed to register teleport physics callback: {exc}")

    def close(self):
        if not self._cb_registered:
            return
        remove_cb = getattr(self.env.unwrapped.sim, "remove_physics_callback", None)
        if remove_cb is None:
            return
        try:
            remove_cb(self._cb_name)
            self._cb_registered = False
        except Exception:
            pass

    def physics_step_cb(self, dt: float):
        del dt
        self.apply()

    def apply(self):
        pending_flags = getattr(self.env.unwrapped, "_vagen_new_task_available", None)
        pending_indices = getattr(self.env.unwrapped, "_vagen_new_task_index", None)
        source_pick_xy = getattr(self.env.unwrapped, "_vagen_source_pick_pos_xy", None)
        if not isinstance(pending_flags, torch.Tensor) or not isinstance(pending_indices, torch.Tensor):
            if not self._warned_missing_state:
                print(
                    "[WARN]: TeleportPendingCubesEvent missing shared state "
                    "(_vagen_new_task_available/_vagen_new_task_index); skipping."
                )
                self._warned_missing_state = True
            return
        if not isinstance(source_pick_xy, torch.Tensor) or source_pick_xy.numel() < 2:
            return

        env_ids_t = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        valid_pending_mask = pending_flags[env_ids_t].to(dtype=torch.bool)
        if not valid_pending_mask.any():
            return

        pending_env_ids = env_ids_t[valid_pending_mask]
        for env_id in pending_env_ids.tolist():
            cube_idx = int(pending_indices[env_id].item())
            if cube_idx < 0 or cube_idx >= len(self.cube_names):
                pending_flags[env_id] = False
                pending_indices[env_id] = -1
                continue

            cube_name = self.cube_names[cube_idx]
            if not hasattr(self.scene, cube_name):
                pending_flags[env_id] = False
                pending_indices[env_id] = -1
                continue

            cube_asset = self.scene[cube_name]
            env_origin = self.scene.env_origins[env_id]
            pick_offset = torch.tensor(
                [float(source_pick_xy[0]), float(source_pick_xy[1]), self.cube_size / 2.0],
                device=env_origin.device,
                dtype=torch.float32,
            )
            target_pos_w = env_origin + pick_offset
            target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_origin.device, dtype=torch.float32)
            root_pose = torch.cat([target_pos_w, target_quat_w], dim=-1).unsqueeze(0)
            env_ids_i32 = torch.tensor([env_id], device=env_origin.device, dtype=torch.int32)
            cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_i32)
            cube_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env_origin.device), env_ids=env_ids_i32)

            pending_flags[env_id] = False
            pending_indices[env_id] = -1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pending_flags = getattr(self.env.unwrapped, "_vagen_new_task_available", None)
        pending_indices = getattr(self.env.unwrapped, "_vagen_new_task_index", None)
        if not isinstance(pending_flags, torch.Tensor) or not isinstance(pending_indices, torch.Tensor):
            return

        if env_ids is None or isinstance(env_ids, slice):
            pending_flags[:] = False
            pending_indices[:] = -1
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids_t.numel() == 0:
            return
        pending_flags[env_ids_t] = False
        pending_indices[env_ids_t] = -1
