# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

DEFAULT_CUBE_SIZE = 0.0203 * 2.0


class TeleportRuntime:
    """Server runtime that teleports cubes to goals instead of using robot actions."""

    IDLE = -2

    def __init__(
        self,
        *,
        env,
        cube_names: list[str],
        cube_size: float,
        max_tasks: int = 8,
        grid_origin: tuple[float, float, float] = (0.5, 0.0, 0.001),
        cell_size: float = 0.056,
    ):
        self.env = env
        self.cube_names = list(cube_names)
        self.cube_size = float(cube_size)
        self.max_tasks = int(max_tasks)
        self.num_envs = int(env.unwrapped.num_envs)
        self.device = env.unwrapped.device
        self.grid_origin = torch.tensor(grid_origin, device=self.device, dtype=torch.float32)
        self.cell_size = float(cell_size)

        self.state = torch.full((self.num_envs,), self.IDLE, dtype=torch.long, device=self.device)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_tasks_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.new_task_available = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.new_task_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pending_step_events: dict[int, dict[str, bool | int]] = {}
        self._collision_enabled_cubes: set[tuple[int, str]] = set()
        self._task_limit_per_env = torch.full(
            (self.num_envs,),
            max(1, min(self.max_tasks, max(1, len(self.cube_names)))),
            dtype=torch.long,
            device=self.device,
        )
        self._active_cube_count_per_env = torch.full(
            (self.num_envs,),
            max(1, len(self.cube_names)),
            dtype=torch.long,
            device=self.device,
        )

        self._idle_action_template = self._build_idle_action_template()

    def _build_idle_action_template(self) -> torch.Tensor | None:
        action_space = getattr(self.env, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape is None:
            return None
        if len(shape) == 1 and int(shape[0]) > 0:
            return torch.zeros((self.num_envs, int(shape[0])), device=self.device, dtype=torch.float32)
        return None

    def bind_shared_state(self) -> None:
        self.env.unwrapped._vagen_new_task_available = self.new_task_available
        self.env.unwrapped._vagen_new_task_index = self.new_task_index

    def reset_tracking(self, env_id: int) -> None:
        self._pending_step_events.pop(int(env_id), None)

    def on_reset_env(self, env_id: int) -> None:
        env_id_i = int(env_id)
        self.state[env_id_i] = self.IDLE
        self.task_index[env_id_i] = 0
        self.num_tasks_per_env[env_id_i] = 0
        self.new_task_available[env_id_i] = False
        self.new_task_index[env_id_i] = -1
        self.reset_tracking(env_id_i)

    def set_env_task_limit(self, env_id: int, requested_num_tasks: int) -> None:
        env_id_i = int(env_id)
        requested = max(1, int(requested_num_tasks))
        available_entities = max(1, len(self.cube_names))
        effective = min(requested, self.max_tasks, available_entities)
        self._task_limit_per_env[env_id_i] = int(effective)
        self._active_cube_count_per_env[env_id_i] = int(effective)
        self._hide_inactive_cubes(env_id=env_id_i, active_cube_count=int(effective))
        if effective < requested:
            print(
                "[WARN][TeleportRuntime] task limit clipped "
                f"(env={env_id_i}, requested={requested}, available_entities={available_entities}, "
                f"runtime_max={self.max_tasks}, effective={effective})"
            )

    def _hide_inactive_cubes(self, *, env_id: int, active_cube_count: int) -> None:
        if active_cube_count >= len(self.cube_names):
            return
        scene = self.env.unwrapped.scene
        env_id_i32 = torch.tensor([int(env_id)], device=self.device, dtype=torch.int32)
        env_origin = scene.env_origins[int(env_id)].to(device=self.device, dtype=torch.float32)
        base_pos = env_origin + torch.tensor([-5.0, -5.0, self.cube_size / 2.0], device=self.device, dtype=torch.float32)
        quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        zero_vel = torch.zeros((1, 6), device=self.device, dtype=torch.float32)
        for idx in range(int(active_cube_count), len(self.cube_names)):
            cube_name = self.cube_names[idx]
            try:
                cube_asset = scene[cube_name]
            except KeyError:
                continue
            offset = torch.tensor([0.06 * float(idx - active_cube_count), 0.0, 0.0], device=self.device, dtype=torch.float32)
            pos_w = (base_pos + offset).reshape(1, 3)
            root_pose = torch.cat([pos_w, quat_w], dim=-1)
            cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_id_i32)
            cube_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_id_i32)

    def reset_state_for_all_envs(self) -> None:
        for env_id in range(self.num_envs):
            self.on_reset_env(env_id)

    def get_step_snapshot(self, env_id: int) -> dict[str, int | bool]:
        env_id_i = int(env_id)
        return {
            "task_index": int(self.task_index[env_id_i].item()),
            "state": int(self.state[env_id_i].item()),
            "num_tasks": int(self.num_tasks_per_env[env_id_i].item()),
            "new_task_available": bool(self.new_task_available[env_id_i].item()),
            "new_task_index": int(self.new_task_index[env_id_i].item()),
        }

    def _parse_goal_to_world(self, env_id: int, goal: dict) -> torch.Tensor:
        if not isinstance(goal, dict):
            raise ValueError(f"Invalid goal type: {type(goal)}")
        if not all(k in goal for k in ("x", "y", "z")):
            raise KeyError("Goal must contain keys 'x','y','z'")

        g_x, g_y, g_z = float(goal["x"]), float(goal["y"]), float(goal["z"])
        env_origin = self.env.unwrapped.scene.env_origins[int(env_id)]
        target_x = self.grid_origin[0].item() + (g_x - 3.5) * self.cell_size
        target_y = self.grid_origin[1].item() + (g_y - 3.5) * self.cell_size
        target_z = (g_z + 0.5) * self.cube_size + 0.002
        return env_origin + torch.tensor([target_x, target_y, target_z], device=env_origin.device)

    def _teleport_cube_to_world(self, env_id: int, cube_slot: int, target_pos_w: torch.Tensor) -> bool:
        if cube_slot < 0 or cube_slot >= len(self.cube_names):
            return False
        cube_name = self.cube_names[cube_slot]
        scene = self.env.unwrapped.scene
        try:
            cube_asset = scene[cube_name]
        except KeyError:
            return False

        # Teleport mode starts with cube collisions disabled by default.
        # Enable collision lazily when a cube is first used in a teleport step.
        self._enable_cube_collision_if_needed(env_id=env_id, cube_name=cube_name, cube_asset=cube_asset)

        env_origins = scene.env_origins
        device = env_origins.device
        dtype = env_origins.dtype
        env_id_i32 = torch.tensor([int(env_id)], device=device, dtype=torch.int32)

        pos_w = target_pos_w.to(device=device, dtype=dtype).reshape(1, 3)
        quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        root_pose = torch.cat([pos_w, quat_w], dim=-1)
        zero_vel = torch.zeros((1, 6), device=device, dtype=dtype)

        cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_id_i32)
        cube_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_id_i32)
        return True

    def _enable_cube_collision_if_needed(self, *, env_id: int, cube_name: str, cube_asset) -> None:
        collision_key = (int(env_id), str(cube_name))
        if collision_key in self._collision_enabled_cubes:
            return

        try:
            import isaaclab.sim as sim_utils

            collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            prim_paths = []
            root_physx_view = getattr(cube_asset, "root_physx_view", None)
            if root_physx_view is not None and hasattr(root_physx_view, "prim_paths"):
                all_prim_paths = [str(path) for path in root_physx_view.prim_paths]
                env_index = int(env_id)
                if 0 <= env_index < len(all_prim_paths):
                    prim_paths = [all_prim_paths[env_index]]
                else:
                    prim_paths = all_prim_paths
            if not prim_paths:
                asset_cfg = getattr(cube_asset, "cfg", None)
                prim_path = getattr(asset_cfg, "prim_path", "")
                if prim_path:
                    prim_paths = [str(prim_path)]

            for prim_path in prim_paths:
                sim_utils.modify_collision_properties(prim_path, collision_cfg)

            self._collision_enabled_cubes.add(collision_key)
        except Exception as exc:
            print(
                f"[WARN][TeleportRuntime] Failed to enable collision for cube '{cube_name}' "
                f"(env_id={int(env_id)}): {exc}"
            )

    def _queue_step_event(self, env_id: int, *, done: bool, success: bool, timeout: bool) -> None:
        env_id_i = int(env_id)
        snapshot = self.get_step_snapshot(env_id_i)
        self._pending_step_events[env_id_i] = {
            "done": bool(done),
            "success": bool(success),
            "timeout": bool(timeout),
            "new_task_available": bool(snapshot["new_task_available"]),
            "new_task_index": int(snapshot["new_task_index"]),
            "task_index": int(snapshot["task_index"]),
            "state": int(snapshot["state"]),
            "num_tasks": int(snapshot["num_tasks"]),
        }

    def handle_step_goal(self, env_id: int, goal) -> dict:
        env_id_i = int(env_id)
        is_submit = isinstance(goal, dict) and goal.get("type") == "submit"
        current_task_idx = int(self.task_index[env_id_i].item())

        if is_submit:
            num_tasks = int(self.num_tasks_per_env[env_id_i].item())
            is_success = num_tasks > 0 and current_task_idx >= num_tasks
            self.new_task_available[env_id_i] = False
            self.new_task_index[env_id_i] = -1
            self._queue_step_event(env_id_i, done=True, success=bool(is_success), timeout=False)
            return {"immediate_done": False, "done_payload": None}

        env_task_limit = int(self._task_limit_per_env[env_id_i].item())
        if current_task_idx >= env_task_limit:
            self.new_task_available[env_id_i] = False
            self.new_task_index[env_id_i] = -1
            self._queue_step_event(env_id_i, done=True, success=False, timeout=True)
            return {"immediate_done": False, "done_payload": None}

        target_pos_w = self._parse_goal_to_world(env_id_i, goal)
        cube_slot = min(current_task_idx, max(0, len(self.cube_names) - 1))
        teleported = self._teleport_cube_to_world(env_id_i, cube_slot, target_pos_w)
        if not teleported:
            self.new_task_available[env_id_i] = False
            self.new_task_index[env_id_i] = -1
            self._queue_step_event(env_id_i, done=True, success=False, timeout=True)
            return {"immediate_done": False, "done_payload": None}

        self.num_tasks_per_env[env_id_i] = max(int(self.num_tasks_per_env[env_id_i].item()), current_task_idx + 1)
        self.task_index[env_id_i] = current_task_idx + 1
        self.new_task_available[env_id_i] = False
        self.new_task_index[env_id_i] = -1
        self._queue_step_event(env_id_i, done=False, success=False, timeout=False)
        return {"immediate_done": False, "done_payload": None}

    def collect_completed_step_events(self) -> list[dict[str, int | bool]]:
        events: list[dict[str, int | bool]] = []
        for env_id in list(self._pending_step_events.keys()):
            payload = self._pending_step_events.pop(env_id)
            events.append({"env_id": int(env_id), **payload})
        return events

    def _build_idle_actions(self, obs: dict) -> torch.Tensor:
        policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else None
        if isinstance(policy_obs, dict):
            last_actions = policy_obs.get("actions")
            if isinstance(last_actions, torch.Tensor):
                return torch.zeros_like(last_actions)
            self.env.unwrapped._vagen_policy_obs = policy_obs
        if self._idle_action_template is not None:
            return self._idle_action_template.clone()
        return torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

    def step(self, obs: dict):
        actions = self._build_idle_actions(obs)
        obs, _, _, _, _ = self.env.step(actions)
        return obs


def build_teleport_runtime(
    *,
    env,
    cube_names,
    cube_size=DEFAULT_CUBE_SIZE,
    max_tasks=8,
    grid_origin=(0.5, 0.0, 0.001),
    cell_size=0.056,
    grid_size=8,
):
    del grid_size
    runtime = TeleportRuntime(
        env=env,
        cube_names=list(cube_names),
        cube_size=float(cube_size),
        max_tasks=int(max_tasks),
        grid_origin=tuple(grid_origin),
        cell_size=float(cell_size),
    )
    runtime.bind_shared_state()
    return runtime
