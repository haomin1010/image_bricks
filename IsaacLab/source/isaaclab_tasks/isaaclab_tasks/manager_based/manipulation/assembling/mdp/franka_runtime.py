# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.utils.math as math_utils
import torch

DEFAULT_CUBE_SIZE = 0.0203 * 2.0


class FrankaRuntimeStateMachine:
    """Six-state direct-target state machine used by VAGEN server."""

    def __init__(
        self,
        num_envs,
        device,
        scene,
        cube_names,
        max_tasks=8,
        cube_z_size=DEFAULT_CUBE_SIZE,
        grid_origin=[0.5, 0.0, 0.001],
        cell_size=0.056,
        grid_size=8,
    ):
        del cube_z_size, grid_size
        self.num_envs = int(num_envs)
        self.device = device
        self.scene = scene
        self.cube_names = list(cube_names)
        self.max_tasks = int(max_tasks)
        self.grid_origin = torch.tensor(grid_origin, device=device, dtype=torch.float32)
        self.cell_size = float(cell_size)

        # Fields consumed by server runtime.
        self.IDLE = -2
        self.IDLE_TO_TARGET = 0
        self.TARGET_TO_IDLE = 1
        self.IDLE_TO_CUBE = 2
        self.PLACE = 3
        self.PLACE_TO_CUBE_TOP = 4
        self.CUBE_TO_IDLE = 5

        self.state = torch.full((self.num_envs,), self.IDLE, dtype=torch.long, device=device)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.place_close_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.place_pre_grasp_pos_w = torch.zeros((self.num_envs, 3), device=device, dtype=torch.float32)
        self.place_pre_grasp_quat_w = torch.zeros((self.num_envs, 4), device=device, dtype=torch.float32)
        self.target_positions = torch.zeros((self.num_envs, self.max_tasks, 3), device=device)
        self.num_tasks_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.new_task_available = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Fixed terminal rotation for all motion states (wxyz).
        self._fixed_target_quat_wxyz = (0.0, 1.0, 0.0, 0.0)
        self.place_pre_grasp_quat_w[:, 0] = float(self._fixed_target_quat_wxyz[0])
        self.place_pre_grasp_quat_w[:, 1] = float(self._fixed_target_quat_wxyz[1])
        self.place_pre_grasp_quat_w[:, 2] = float(self._fixed_target_quat_wxyz[2])
        self.place_pre_grasp_quat_w[:, 3] = float(self._fixed_target_quat_wxyz[3])

        # Direct-target reach/timeout thresholds.
        self._reach_pos_tol_m = 0.02
        self._reach_rot_tol_rad = 0.20
        self._state_timeout_steps = 120
        self._grasp_extra_hold_steps = 40
        self._cube_stage_z_lift_m = 0.10
        self._goal_stage_z_offset_m = 0.05
        self._gripper_open_cmd = 1.0
        self._gripper_close_cmd = -1.0

    def set_env_targets(self, env_ids, targets):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        targets = torch.as_tensor(targets, device=self.device, dtype=self.target_positions.dtype).reshape(-1, 3)
        n = min(env_ids.numel(), targets.shape[0])
        env_ids = env_ids[:n]
        targets = targets[:n]
        task_slots = torch.clamp(self.task_index[env_ids].to(torch.long), 0, self.max_tasks - 1)
        self.target_positions[env_ids, task_slots] = targets
        self.num_tasks_per_env[env_ids] = torch.maximum(self.num_tasks_per_env[env_ids], task_slots + 1)
        self.state[env_ids] = self.IDLE_TO_CUBE
        self.state_timer[env_ids] = 0
        self.place_close_latch[env_ids] = False
        self.place_pre_grasp_pos_w[env_ids] = 0.0
        self.place_pre_grasp_quat_w[env_ids, 0] = float(self._fixed_target_quat_wxyz[0])
        self.place_pre_grasp_quat_w[env_ids, 1] = float(self._fixed_target_quat_wxyz[1])
        self.place_pre_grasp_quat_w[env_ids, 2] = float(self._fixed_target_quat_wxyz[2])
        self.place_pre_grasp_quat_w[env_ids, 3] = float(self._fixed_target_quat_wxyz[3])
        self.new_task_available[env_ids] = True
        self.new_task_index[env_ids] = task_slots
        return None

    def reset_envs(self, env_ids):
        self.state[env_ids] = self.IDLE
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.place_close_latch[env_ids] = False
        self.place_pre_grasp_pos_w[env_ids] = 0.0
        self.place_pre_grasp_quat_w[env_ids, 0] = float(self._fixed_target_quat_wxyz[0])
        self.place_pre_grasp_quat_w[env_ids, 1] = float(self._fixed_target_quat_wxyz[1])
        self.place_pre_grasp_quat_w[env_ids, 2] = float(self._fixed_target_quat_wxyz[2])
        self.place_pre_grasp_quat_w[env_ids, 3] = float(self._fixed_target_quat_wxyz[3])
        self.new_task_available[env_ids] = False
        self.new_task_index[env_ids] = -1

    @staticmethod
    def _align_quaternion_hemisphere(target_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
        """Keep target quaternions in the same hemisphere to avoid sign-flip jitter."""
        same_hemisphere = torch.sum(target_quat * reference_quat, dim=-1, keepdim=True) >= 0.0
        return torch.where(same_hemisphere, target_quat, -target_quat)

    def _fixed_target_quat(self, dtype: torch.dtype) -> torch.Tensor:
        target_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        target_quat_w[:, 0] = float(self._fixed_target_quat_wxyz[0])
        target_quat_w[:, 1] = float(self._fixed_target_quat_wxyz[1])
        target_quat_w[:, 2] = float(self._fixed_target_quat_wxyz[2])
        target_quat_w[:, 3] = float(self._fixed_target_quat_wxyz[3])
        return target_quat_w

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0
        self.place_close_latch[env_idx] = False
        self.place_pre_grasp_pos_w[env_idx] = 0.0
        self.place_pre_grasp_quat_w[env_idx, 0] = float(self._fixed_target_quat_wxyz[0])
        self.place_pre_grasp_quat_w[env_idx, 1] = float(self._fixed_target_quat_wxyz[1])
        self.place_pre_grasp_quat_w[env_idx, 2] = float(self._fixed_target_quat_wxyz[2])
        self.place_pre_grasp_quat_w[env_idx, 3] = float(self._fixed_target_quat_wxyz[3])

    def compute_ee_pose_targets(self, obs):
        policy_obs = obs.get("policy", obs)
        ee_pos_env = policy_obs.get("ee_pos")
        ee_quat_w = policy_obs.get("ee_quat")
        cube_pos_w = policy_obs.get("cube_pos")
        root_pos_w = policy_obs.get("root_pos")
        root_quat_w = policy_obs.get("root_quat")
        env_origin = policy_obs.get("env_origin")

        # Task-space action expects absolute pose in robot root frame.
        ee_pos_w = ee_pos_env + env_origin

        if cube_pos_w.ndim != 2 or cube_pos_w.shape[1] < 3 or (cube_pos_w.shape[1] % 3 != 0):
            raise ValueError(f"Invalid cube_pos shape: {tuple(cube_pos_w.shape)}")
        cube_pos_w_reshaped = cube_pos_w.view(self.num_envs, -1, 3)
        num_obs_cubes = int(cube_pos_w_reshaped.shape[1])

        env_ids_all = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        task_slots = torch.clamp(self.task_index.to(torch.long), 0, self.max_tasks - 1)
        goal_pos_w = self.target_positions[env_ids_all, task_slots]
        goal_pos_w_lifted = goal_pos_w.clone()
        goal_pos_w_lifted[:, 2] += float(self._goal_stage_z_offset_m)
        cube_slots = torch.clamp(self.task_index.to(torch.long), min=0, max=max(0, num_obs_cubes - 1))
        cube_pos_w_curr = cube_pos_w_reshaped[env_ids_all, cube_slots]
        cube_pos_w_lifted = cube_pos_w_curr.clone()
        cube_pos_w_lifted[:, 2] += float(self._cube_stage_z_lift_m)

        idle_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_pos_w.dtype)
        idle_pos_w[:, 0] = 0.5
        idle_pos_w[:, 1] = 0.0
        idle_pos_w[:, 2] = 0.30

        # Direct terminal target for each state (no interpolation).
        target_pos_w = idle_pos_w.clone()
        idle_to_target_mask = self.state == self.IDLE_TO_TARGET
        target_to_idle_mask = self.state == self.TARGET_TO_IDLE
        idle_to_cube_mask = self.state == self.IDLE_TO_CUBE
        place_mask = self.state == self.PLACE
        place_to_cube_top_mask = self.state == self.PLACE_TO_CUBE_TOP
        cube_to_idle_mask = self.state == self.CUBE_TO_IDLE

        if torch.any(idle_to_target_mask):
            target_pos_w[idle_to_target_mask] = goal_pos_w_lifted[idle_to_target_mask]
        if torch.any(target_to_idle_mask):
            target_pos_w[target_to_idle_mask] = idle_pos_w[target_to_idle_mask]
        if torch.any(idle_to_cube_mask):
            target_pos_w[idle_to_cube_mask] = cube_pos_w_lifted[idle_to_cube_mask]
        if torch.any(place_mask):
            target_pos_w[place_mask] = cube_pos_w_curr[place_mask]
        if torch.any(place_to_cube_top_mask):
            # Ascend back to the pose right before descending for grasp.
            target_pos_w[place_to_cube_top_mask] = self.place_pre_grasp_pos_w[place_to_cube_top_mask]
        if torch.any(cube_to_idle_mask):
            target_pos_w[cube_to_idle_mask] = idle_pos_w[cube_to_idle_mask]

        target_quat_w = self._fixed_target_quat(dtype=root_quat_w.dtype)
        if torch.any(place_to_cube_top_mask):
            target_quat_w[place_to_cube_top_mask] = self.place_pre_grasp_quat_w[
                place_to_cube_top_mask
            ].to(dtype=target_quat_w.dtype)
        target_quat_w = self._align_quaternion_hemisphere(target_quat_w, ee_quat_w)

        pos_err = torch.linalg.vector_norm(target_pos_w - ee_pos_w, dim=-1)
        rot_err = math_utils.quat_error_magnitude(target_quat_w, ee_quat_w)
        reached_mask = torch.logical_and(pos_err <= self._reach_pos_tol_m, rot_err <= self._reach_rot_tol_rad)
        timeout_mask = self.state_timer >= self._state_timeout_steps
        advance_mask = torch.logical_or(reached_mask, timeout_mask)

        gripper_cmd_all = torch.full(
            (self.num_envs,),
            float(self._gripper_open_cmd),
            device=self.device,
            dtype=ee_pos_w.dtype,
        )
        # Close only after PLACE has reached the bottom target, then keep closed while carrying.
        place_reached_mask = torch.logical_and(place_mask, reached_mask)
        if torch.any(place_reached_mask):
            self.place_close_latch[place_reached_mask] = True
        self.place_close_latch[~place_mask] = False
        place_closed_mask = torch.logical_and(place_mask, self.place_close_latch)
        close_gripper_mask = torch.logical_or(place_closed_mask, place_to_cube_top_mask)
        close_gripper_mask = torch.logical_or(close_gripper_mask, cube_to_idle_mask)
        close_gripper_mask = torch.logical_or(close_gripper_mask, idle_to_target_mask)
        if torch.any(close_gripper_mask):
            gripper_cmd_all[close_gripper_mask] = float(self._gripper_close_cmd)

        # During PLACE (grasp) stage, enforce an additional dwell before lifting.
        place_hold_done_mask = self.state_timer >= self._grasp_extra_hold_steps
        place_advance_mask = torch.logical_or(timeout_mask, torch.logical_and(place_closed_mask, place_hold_done_mask))

        # Six-state sequence:
        # 1) idle->cube(top)  2) place(down to cube center)  3) lift(up to cube top)
        # 4) cube->idle  5) idle->target  6) target->idle
        to_place = torch.logical_and(idle_to_cube_mask, advance_mask)
        to_place_to_cube_top = torch.logical_and(place_mask, place_advance_mask)
        to_cube_to_idle = torch.logical_and(place_to_cube_top_mask, advance_mask)
        to_idle_to_target = torch.logical_and(cube_to_idle_mask, advance_mask)
        to_target_to_idle = torch.logical_and(idle_to_target_mask, advance_mask)
        to_done = torch.logical_and(target_to_idle_mask, advance_mask)

        if torch.any(to_place):
            self.state[to_place] = self.PLACE
            self.state_timer[to_place] = 0
            self.place_pre_grasp_pos_w[to_place] = ee_pos_w[to_place]
            self.place_pre_grasp_quat_w[to_place] = ee_quat_w[to_place]
        if torch.any(to_place_to_cube_top):
            self.state[to_place_to_cube_top] = self.PLACE_TO_CUBE_TOP
            self.state_timer[to_place_to_cube_top] = 0
        if torch.any(to_cube_to_idle):
            self.state[to_cube_to_idle] = self.CUBE_TO_IDLE
            self.state_timer[to_cube_to_idle] = 0
        if torch.any(to_idle_to_target):
            self.state[to_idle_to_target] = self.IDLE_TO_TARGET
            self.state_timer[to_idle_to_target] = 0
        if torch.any(to_target_to_idle):
            self.state[to_target_to_idle] = self.TARGET_TO_IDLE
            self.state_timer[to_target_to_idle] = 0
        if torch.any(to_done):
            self.state[to_done] = self.IDLE
            self.state_timer[to_done] = 0
            self.place_pre_grasp_pos_w[to_done] = 0.0
            self.place_pre_grasp_quat_w[to_done, 0] = float(self._fixed_target_quat_wxyz[0])
            self.place_pre_grasp_quat_w[to_done, 1] = float(self._fixed_target_quat_wxyz[1])
            self.place_pre_grasp_quat_w[to_done, 2] = float(self._fixed_target_quat_wxyz[2])
            self.place_pre_grasp_quat_w[to_done, 3] = float(self._fixed_target_quat_wxyz[3])
            self.task_index[to_done] += 1
            self.new_task_available[to_done] = False
            self.new_task_index[to_done] = -1

        active_mask = self.state != self.IDLE
        self.state_timer[active_mask] += 1
        self.state_timer[~active_mask] = 0

        target_pos_all, target_quat_all = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        return target_pos_all, target_quat_all, gripper_cmd_all


def build_franka_state_machine(
    *,
    num_envs,
    device,
    scene,
    cube_names,
    max_tasks=8,
    cube_z_size=DEFAULT_CUBE_SIZE,
    grid_origin=(0.5, 0.0, 0.001),
    cell_size=0.056,
    grid_size=8,
):
    """Build server-facing state-machine runtime from task config side."""
    return FrankaRuntimeStateMachine(
        num_envs=num_envs,
        device=device,
        scene=scene,
        cube_names=cube_names,
        max_tasks=max_tasks,
        cube_z_size=cube_z_size,
        grid_origin=grid_origin,
        cell_size=cell_size,
        grid_size=grid_size,
    )


class FrankaRuntime:
    """Server runtime wrapper that encapsulates state-machine + action logic."""

    def __init__(self, *, env, state_machine: FrankaRuntimeStateMachine, cube_size: float):
        self.env = env
        self.sm = state_machine
        self.cube_size = float(cube_size)
        self.num_envs = int(env.unwrapped.num_envs)
        self._step_initial_task_idx: dict[int, dict[str, int | bool | None]] = {}
        self._gripper_center_marker = None
        self._gripper_marker_enabled = os.getenv("VAGEN_VIS_GRIPPER_CENTER", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if self._gripper_marker_enabled:
            self._init_gripper_center_marker()

    def _init_gripper_center_marker(self) -> None:
        """Initialize gripper-center marker visualization in world frame."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import SPHERE_MARKER_CFG

            marker_cfg = SPHERE_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/VAGEN/gripper_center"
            marker_cfg.markers["sphere"].radius = 0.008
            marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0)
            )
            self._gripper_center_marker = VisualizationMarkers(marker_cfg)
            self._gripper_center_marker.set_visibility(True)
            print("[INFO]: Enabled gripper-center marker visualization at /Visuals/VAGEN/gripper_center")
        except Exception as exc:
            self._gripper_center_marker = None
            print(f"[WARN]: Failed to initialize gripper-center marker visualization: {exc}")

    def _update_gripper_center_marker(self, policy_obs: dict | None) -> None:
        marker = self._gripper_center_marker
        if marker is None or not isinstance(policy_obs, dict):
            return
        ee_pos_env = policy_obs.get("ee_pos")
        env_origin = policy_obs.get("env_origin")
        if ee_pos_env is None or env_origin is None:
            return
        if ee_pos_env.ndim != 2 or ee_pos_env.shape[-1] != 3:
            return
        if env_origin.ndim != 2 or env_origin.shape[-1] != 3:
            return
        # ee_pos is env-local for this task. Convert to world before visualization.
        marker.visualize(translations=ee_pos_env + env_origin)

    def bind_shared_state(self) -> None:
        # Shared state consumed by runtime terms.
        self.env.unwrapped._vagen_new_task_available = self.sm.new_task_available
        self.env.unwrapped._vagen_new_task_index = self.sm.new_task_index

    def reset_tracking(self, env_id: int) -> None:
        self._step_initial_task_idx.pop(int(env_id), None)

    def on_reset_env(self, env_id: int) -> None:
        self.sm.reset_envs([env_id])
        self.reset_tracking(env_id)

    def _parse_goal_to_world(self, env_id: int, goal: dict) -> torch.Tensor:
        if not isinstance(goal, dict):
            raise ValueError(f"Invalid goal type: {type(goal)}")
        if not all(k in goal for k in ("x", "y", "z")):
            raise KeyError("Goal must contain keys 'x','y','z'")

        g_x, g_y, g_z = float(goal["x"]), float(goal["y"]), float(goal["z"])
        grid_origin = self.sm.grid_origin
        cell_size = self.sm.cell_size
        env_origin = self.env.unwrapped.scene.env_origins[env_id]
        target_x = grid_origin[0].item() + (g_x - 4.5) * cell_size
        target_y = grid_origin[1].item() + (g_y - 4.5) * cell_size
        target_z = (g_z + 0.5) * self.cube_size + 0.002
        return env_origin + torch.tensor([target_x, target_y, target_z], device=env_origin.device)

    def handle_step_goal(self, env_id: int, goal) -> dict:
        is_submit = isinstance(goal, dict) and goal.get("type") == "submit"
        current_task_idx = int(self.sm.task_index[env_id].item())
        if is_submit:
            num_tasks = int(self.sm.num_tasks_per_env[env_id].item())
            is_success = current_task_idx >= num_tasks
            self._step_initial_task_idx[int(env_id)] = {
                "init_idx": int(current_task_idx),
                "was_submit": True,
                "submit_success": bool(is_success),
            }
            return {"immediate_done": False, "done_payload": None}

        max_tasks = int(getattr(self.sm, "max_tasks", self.sm.target_positions.shape[1]))
        if current_task_idx >= max_tasks:
            self.reset_tracking(env_id)
            self.sm.num_tasks_per_env[env_id] = max_tasks
            self.sm.state[env_id] = self.sm.IDLE
            return {
                "immediate_done": True,
                "done_payload": {
                    "done": True,
                    "success": False,
                    "timeout": True,
                    "new_task_available": False,
                    "new_task_index": -1,
                },
            }

        self._step_initial_task_idx[int(env_id)] = {
            "init_idx": int(current_task_idx),
            "was_submit": False,
            "submit_success": None,
        }
        target_pos_w = self._parse_goal_to_world(env_id, goal)
        self.sm.set_env_targets([env_id], target_pos_w)
        return {"immediate_done": False, "done_payload": None}

    def get_step_snapshot(self, env_id: int) -> dict[str, int | bool]:
        return {
            "task_index": int(self.sm.task_index[env_id].item()),
            "state": int(self.sm.state[env_id].item()),
            "num_tasks": int(self.sm.num_tasks_per_env[env_id].item()),
            "new_task_available": bool(self.sm.new_task_available[env_id].item()),
            "new_task_index": int(self.sm.new_task_index[env_id].item()),
        }

    def collect_completed_step_events(self) -> list[dict[str, int | bool]]:
        events: list[dict[str, int | bool]] = []
        for env_id in list(self._step_initial_task_idx.keys()):
            snapshot = self.get_step_snapshot(int(env_id))
            task_idx_now = int(snapshot["task_index"])
            sm_state_now = int(snapshot["state"])

            init_val = self._step_initial_task_idx[env_id]
            init_idx = int(init_val.get("init_idx", 0))
            was_submit = bool(init_val.get("was_submit", False))
            submit_success = init_val.get("submit_success", None)

            if (task_idx_now > init_idx) or sm_state_now == -1:
                done_flag = True if was_submit else False
                success_flag = done_flag if submit_success is None else bool(submit_success)
                events.append(
                    {
                        "env_id": int(env_id),
                        "done": bool(done_flag),
                        "success": bool(success_flag),
                        "timeout": False,
                        "new_task_available": bool(snapshot["new_task_available"]),
                        "new_task_index": int(snapshot["new_task_index"]),
                        "task_index": int(task_idx_now),
                        "state": int(sm_state_now),
                        "num_tasks": int(snapshot["num_tasks"]),
                    }
                )
                del self._step_initial_task_idx[env_id]
        return events

    def reset_state_for_all_envs(self) -> None:
        for env_id in range(self.num_envs):
            self.sm.reset_state(env_id)

    def compute_joint_actions(self, obs: dict) -> torch.Tensor:
        policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else None
        if isinstance(policy_obs, dict):
            self.env.unwrapped._vagen_policy_obs = policy_obs
            self._update_gripper_center_marker(policy_obs)
        ee_pos_des, ee_quat_des, gripper_cmd = self.sm.compute_ee_pose_targets(obs)
        quat_norm = torch.linalg.vector_norm(ee_quat_des, dim=-1, keepdim=True).clamp_min(1e-8)
        ee_quat_des = ee_quat_des / quat_norm
        gripper_cmd = gripper_cmd.to(dtype=ee_pos_des.dtype)
        return torch.cat([ee_pos_des, ee_quat_des, gripper_cmd.unsqueeze(-1)], dim=-1)

    def step(self, obs: dict):
        actions = self.compute_joint_actions(obs)
        obs, _, _, _, _ = self.env.step(actions)
        return obs


def build_franka_runtime(
    *,
    env,
    cube_names,
    cube_size=DEFAULT_CUBE_SIZE,
    max_tasks=8,
    grid_origin=(0.5, 0.0, 0.001),
    cell_size=0.056,
    grid_size=8,
):
    """Build action-runtime wrapper from task-config side for server usage."""
    sm = build_franka_state_machine(
        num_envs=int(env.unwrapped.num_envs),
        device=env.unwrapped.device,
        scene=env.unwrapped.scene,
        cube_names=cube_names,
        max_tasks=max_tasks,
        cube_z_size=cube_size,
        grid_origin=grid_origin,
        cell_size=cell_size,
        grid_size=grid_size,
    )
    runtime = FrankaRuntime(env=env, state_machine=sm, cube_size=cube_size)
    runtime.bind_shared_state()
    return runtime
