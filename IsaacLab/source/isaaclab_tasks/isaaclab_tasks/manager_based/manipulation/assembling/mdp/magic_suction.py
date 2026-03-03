from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from .observations import ee_pos as obs_ee_pos
from .observations import ee_quat as obs_ee_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MagicSuctionBinaryAction(ActionTerm):
    """Binary gripper action term for magic suction.

    Input action is one scalar per-env:
    - close when action < ``close_command_threshold`` (default: 0.0)
    - open otherwise
    """

    cfg: MagicSuctionBinaryActionCfg

    def __init__(self, cfg: MagicSuctionBinaryActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.ones(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.ones_like(self._raw_actions)

        # Shared command buffer consumed by VagenMagicSuctionController every physics step.
        self._env.unwrapped._vagen_magic_suction_cmd = self._processed_actions[:, 0].clone()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != 1:
            raise ValueError(f"MagicSuctionBinaryAction expects last dim=1, got shape={tuple(actions.shape)}")
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, min=-1.0, max=1.0)

    def apply_actions(self):
        self._env.unwrapped._vagen_magic_suction_cmd = self._processed_actions[:, 0].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 1.0
        self._processed_actions[env_ids] = 1.0
        cmd = getattr(self._env.unwrapped, "_vagen_magic_suction_cmd", None)
        if isinstance(cmd, torch.Tensor):
            cmd[env_ids] = 1.0
        else:
            self._env.unwrapped._vagen_magic_suction_cmd = self._processed_actions[:, 0].clone()


@configclass
class MagicSuctionBinaryActionCfg(ActionTermCfg):
    """Configuration for :class:`MagicSuctionBinaryAction`."""

    class_type: type[ActionTerm] = MagicSuctionBinaryAction
    asset_name: str = MISSING
    close_command_threshold: float = 0.0


def _scene_has_entity(scene, name: str) -> bool:
    if hasattr(scene, name):
        return True
    try:
        scene[name]
        return True
    except Exception:
        return False


def _discover_cube_names(scene, cube_name_prefix: str, max_cubes: int) -> list[str]:
    """Discover cube entities from scene keys/attributes with case-insensitive matching."""
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


class VagenMagicSuctionController:
    """Magic suction implemented as a physics callback controller."""

    def __init__(
        self,
        env,
        cube_names: list[str],
        cube_size: float,
        attach_distance: float = 0.05,
        close_command_threshold: float = 0.0,
        ee_body_name: str = "panda_link7",
    ):
        self.env = env
        self.scene = env.unwrapped.scene
        self.device = env.unwrapped.device
        self.cube_names = list(cube_names)
        self.cube_size = float(cube_size)
        self.attach_distance = float(attach_distance)
        self.close_command_threshold = float(close_command_threshold)
        self.num_envs = int(env.unwrapped.num_envs)
        self._attached_cube_idx = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.env.unwrapped._vagen_magic_suction_attached_cube_idx = self._attached_cube_idx

        self._robot_cfg = SceneEntityCfg("robot")
        self._ee_body_name = str(ee_body_name)
        self._cb_name = f"vagen_magic_suction_{id(self)}"
        self._cb_registered = False

        # Default to open command if no action term has written to the buffer yet.
        cmd = getattr(self.env.unwrapped, "_vagen_magic_suction_cmd", None)
        if not isinstance(cmd, torch.Tensor) or cmd.numel() < self.num_envs:
            cmd = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        else:
            cmd = cmd.to(device=self.device, dtype=torch.float32).reshape(-1)[: self.num_envs]
        self.env.unwrapped._vagen_magic_suction_cmd = cmd

        try:
            env.unwrapped.sim.add_physics_callback(self._cb_name, self.physics_step_cb)
            self._cb_registered = True
            print(
                f"[INFO]: Magic suction callback registered: name={self._cb_name} "
                f"attach_distance={self.attach_distance} "
                f"close_command_threshold={self.close_command_threshold}"
            )
        except Exception as exc:
            print(f"[WARN]: Failed to register magic suction physics callback: {exc}")

    def set_latest_obs(self, obs: dict | None):
        # Kept for compatibility with existing execution manager calls.
        del obs

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or isinstance(env_ids, slice):
            self._attached_cube_idx[:] = -1
            cmd = getattr(self.env.unwrapped, "_vagen_magic_suction_cmd", None)
            if isinstance(cmd, torch.Tensor):
                cmd[:] = 1.0
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        self._attached_cube_idx[env_ids_t] = -1
        cmd = getattr(self.env.unwrapped, "_vagen_magic_suction_cmd", None)
        if isinstance(cmd, torch.Tensor):
            cmd[env_ids_t] = 1.0

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
        try:
            ee_pos_w = obs_ee_pos(env=self.env, robot_cfg=self._robot_cfg, ee_body_name=self._ee_body_name)
            ee_quat_w = obs_ee_quat(env=self.env, robot_cfg=self._robot_cfg, ee_body_name=self._ee_body_name)
        except Exception:
            return

        cmd = getattr(self.env.unwrapped, "_vagen_magic_suction_cmd", None)
        if isinstance(cmd, torch.Tensor) and cmd.numel() >= self.num_envs:
            cmd = cmd.to(device=self.device, dtype=torch.float32).view(-1)[: self.num_envs]
            close_mask = cmd < self.close_command_threshold
        else:
            close_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        open_mask = ~close_mask

        releasing_mask = open_mask & (self._attached_cube_idx >= 0)
        if releasing_mask.any():
            self._attached_cube_idx[releasing_mask] = -1

        grabbing_mask = close_mask & (self._attached_cube_idx == -1)
        grabbing_env_ids = torch.where(grabbing_mask)[0]
        if grabbing_env_ids.numel() > 0:
            target_cube_idx = getattr(self.env.unwrapped, "_vagen_magic_suction_active_cube_idx", None)
            if isinstance(target_cube_idx, torch.Tensor) and target_cube_idx.numel() >= self.num_envs:
                target_cube_idx = target_cube_idx.to(device=self.device, dtype=torch.long).view(-1)[: self.num_envs]
                for cube_idx, cube_name in enumerate(self.cube_names):
                    checking_mask = target_cube_idx[grabbing_env_ids] == cube_idx
                    if not checking_mask.any():
                        continue
                    target_ids = grabbing_env_ids[checking_mask]
                    cube_asset = self.scene[cube_name]
                    cube_pos_w = cube_asset.data.root_pos_w[target_ids]
                    dists = torch.norm(ee_pos_w[target_ids] - cube_pos_w, dim=-1)
                    can_attach = dists < self.attach_distance
                    if can_attach.any():
                        self._attached_cube_idx[target_ids[can_attach]] = cube_idx
            else:
                best_idx = torch.full(
                    (grabbing_env_ids.numel(),),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
                best_dist = torch.full((grabbing_env_ids.numel(),), float("inf"), device=self.device, dtype=torch.float32)
                for cube_idx, cube_name in enumerate(self.cube_names):
                    cube_asset = self.scene[cube_name]
                    cube_pos_w = cube_asset.data.root_pos_w[grabbing_env_ids]
                    dists = torch.norm(ee_pos_w[grabbing_env_ids] - cube_pos_w, dim=-1)
                    is_better = dists < best_dist
                    best_dist[is_better] = dists[is_better]
                    best_idx[is_better] = cube_idx
                can_attach = best_dist < self.attach_distance
                if can_attach.any():
                    self._attached_cube_idx[grabbing_env_ids[can_attach]] = best_idx[can_attach]

        for cube_idx, cube_name in enumerate(self.cube_names):
            attached_mask = self._attached_cube_idx == cube_idx
            env_ids = torch.where(attached_mask)[0]
            if env_ids.numel() == 0:
                continue

            cube_asset = self.scene[cube_name]
            target_cube_pos_w = ee_pos_w[env_ids].clone()
            target_cube_pos_w[:, 2] -= self.cube_size / 4.0
            quat = cube_asset.data.root_quat_w[env_ids] if ee_quat_w is None else ee_quat_w[env_ids]
            root_poses = torch.cat([target_cube_pos_w, quat], dim=-1)
            env_ids_i32 = env_ids.to(dtype=torch.int32)
            cube_asset.write_root_pose_to_sim(root_poses, env_ids=env_ids_i32)
            cube_asset.write_root_velocity_to_sim(
                torch.zeros((env_ids.numel(), 6), device=self.device),
                env_ids=env_ids_i32,
            )


class MagicSuctionControllerEvent(ManagerTermBase):
    """Event wrapper that owns the lifecycle of :class:`VagenMagicSuctionController`."""

    cfg: EventTermCfg

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._controller: VagenMagicSuctionController | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | None,
        cube_names: list[str] | None = None,
        cube_name_prefix: str = "cube_",
        max_cubes: int = 0,
        cube_size: float = 0.045,
        attach_distance: float = 0.05,
        close_command_threshold: float = 0.0,
        ee_body_name: str = "panda_link7",
    ):
        del env_ids
        if self._controller is not None:
            return

        resolved_names = list(cube_names) if cube_names else _discover_cube_names(
            scene=env.scene,
            cube_name_prefix=cube_name_prefix,
            max_cubes=int(max_cubes),
        )
        if not resolved_names:
            print("[WARN]: MagicSuctionControllerEvent did not find cube assets; callback not registered.")
            return

        self._controller = VagenMagicSuctionController(
            env=env,
            cube_names=resolved_names,
            cube_size=float(cube_size),
            attach_distance=float(attach_distance),
            close_command_threshold=float(close_command_threshold),
            ee_body_name=str(ee_body_name),
        )
        env.unwrapped._vagen_magic_suction_controller = self._controller
        env.unwrapped._vagen_magic_suction_cube_names = tuple(resolved_names)

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
