from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils


class StackingStateMachine:
    """Six-state direct-target state machine (no smoothing/interpolation)."""

    def __init__(
        self,
        num_envs,
        device,
        scene,
        cube_names,
        max_tasks=8,
        cube_z_size=0.0203 * 2.0,
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

        # Fields consumed by server.py.
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
