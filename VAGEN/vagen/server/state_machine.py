from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils


class StackingStateMachine:
    """Four-state direct-target state machine (no smoothing/interpolation)."""

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
        self.CUBE_TO_IDLE = 3

        self.state = torch.full((self.num_envs,), self.IDLE, dtype=torch.long, device=device)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.target_positions = torch.zeros((self.num_envs, self.max_tasks, 3), device=device)
        self.num_tasks_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.new_task_available = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Fixed terminal rotation for all four states (wxyz).
        self._fixed_target_quat_wxyz = (0.0, 1.0, 0.0, 0.0)
        fixed_quat = torch.tensor([self._fixed_target_quat_wxyz], device=device, dtype=torch.float32)
        fixed_roll, fixed_pitch, _ = math_utils.euler_xyz_from_quat(fixed_quat)
        self._fixed_roll = float(fixed_roll[0].item())
        self._fixed_pitch = float(fixed_pitch[0].item())

        # Direct-target reach/timeout thresholds.
        self._reach_pos_tol_m = 0.02
        self._reach_rot_tol_rad = 0.20
        self._state_timeout_steps = 120
        self._cube_yaw_offset_rad = 0.5 * float(torch.pi)

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
        self.new_task_available[env_ids] = True
        self.new_task_index[env_ids] = task_slots
        return None

    def reset_envs(self, env_ids):
        self.state[env_ids] = self.IDLE
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
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

    def _vertical_quat_with_yaw(self, yaw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        roll = torch.full_like(yaw, self._fixed_roll, dtype=dtype)
        pitch = torch.full_like(yaw, self._fixed_pitch, dtype=dtype)
        return math_utils.normalize(math_utils.quat_from_euler_xyz(roll, pitch, yaw.to(dtype=dtype)))

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0

    def compute_ee_pose_targets(self, obs):
        policy_obs = obs.get("policy", obs)
        ee_pos_env = policy_obs.get("ee_pos")
        ee_quat_w = policy_obs.get("ee_quat")
        cube_pos_w = policy_obs.get("cube_pos")
        cube_quat_w = policy_obs.get("cube_quat")
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
        cube_slots = torch.clamp(self.task_index.to(torch.long), min=0, max=max(0, num_obs_cubes - 1))
        cube_pos_w_curr = cube_pos_w_reshaped[env_ids_all, cube_slots]

        idle_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_pos_w.dtype)
        idle_pos_w[:, 0] = 0.5
        idle_pos_w[:, 1] = 0.0
        idle_pos_w[:, 2] = 0.30

        # Direct terminal target for each state (no interpolation).
        target_pos_w = idle_pos_w.clone()
        idle_to_target_mask = self.state == self.IDLE_TO_TARGET
        target_to_idle_mask = self.state == self.TARGET_TO_IDLE
        idle_to_cube_mask = self.state == self.IDLE_TO_CUBE
        cube_to_idle_mask = self.state == self.CUBE_TO_IDLE

        if torch.any(idle_to_target_mask):
            target_pos_w[idle_to_target_mask] = goal_pos_w[idle_to_target_mask]
        if torch.any(target_to_idle_mask):
            target_pos_w[target_to_idle_mask] = idle_pos_w[target_to_idle_mask]
        if torch.any(idle_to_cube_mask):
            target_pos_w[idle_to_cube_mask] = cube_pos_w_curr[idle_to_cube_mask]
        if torch.any(cube_to_idle_mask):
            target_pos_w[cube_to_idle_mask] = idle_pos_w[cube_to_idle_mask]

        target_quat_w = self._fixed_target_quat(dtype=root_quat_w.dtype)
        target_quat_w = self._align_quaternion_hemisphere(target_quat_w, ee_quat_w)

        # Cube-related states use cube yaw while keeping roll/pitch vertical.
        cube_related_mask = torch.logical_or(idle_to_cube_mask, cube_to_idle_mask)
        if torch.any(cube_related_mask) and cube_quat_w is not None:
            if cube_quat_w.ndim != 2 or cube_quat_w.shape[1] < 4 or (cube_quat_w.shape[1] % 4 != 0):
                raise ValueError(f"Invalid cube_quat shape: {tuple(cube_quat_w.shape)}")
            cube_quat_w_reshaped = cube_quat_w.view(self.num_envs, -1, 4)
            num_quat_cubes = int(cube_quat_w_reshaped.shape[1])
            cube_quat_slots = torch.clamp(self.task_index.to(torch.long), min=0, max=max(0, num_quat_cubes - 1))
            cube_quat_w_curr = cube_quat_w_reshaped[env_ids_all, cube_quat_slots]

            # Missing cube slots are padded as [-1, -1, -1, -1]; fall back to current EE yaw.
            missing_cube_quat_mask = torch.all(torch.abs(cube_quat_w_curr + 1.0) < 1.0e-4, dim=-1)
            cube_quat_norm = torch.linalg.vector_norm(cube_quat_w_curr, dim=-1, keepdim=True).clamp_min(1.0e-8)
            cube_quat_w_curr = cube_quat_w_curr / cube_quat_norm

            _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w_curr)
            _, _, ee_yaw = math_utils.euler_xyz_from_quat(math_utils.normalize(ee_quat_w))
            cube_yaw = torch.where(missing_cube_quat_mask, ee_yaw, cube_yaw)
            cube_yaw = cube_yaw + self._cube_yaw_offset_rad
            cube_target_quat_w = self._vertical_quat_with_yaw(cube_yaw, dtype=root_quat_w.dtype)
            target_quat_w[cube_related_mask] = cube_target_quat_w[cube_related_mask]

        # Gripper is kept open in this direct 4-state mode.
        gripper_cmd_all = torch.ones((self.num_envs,), device=self.device)

        pos_err = torch.linalg.vector_norm(target_pos_w - ee_pos_w, dim=-1)
        rot_err = math_utils.quat_error_magnitude(target_quat_w, ee_quat_w)
        reached_mask = torch.logical_and(pos_err <= self._reach_pos_tol_m, rot_err <= self._reach_rot_tol_rad)
        timeout_mask = self.state_timer >= self._state_timeout_steps
        advance_mask = torch.logical_or(reached_mask, timeout_mask)

        # Four-state sequence:
        # 1) idle->cube  2) cube->idle  3) idle->target  4) target->idle
        to_cube_to_idle = torch.logical_and(idle_to_cube_mask, advance_mask)
        to_idle_to_target = torch.logical_and(cube_to_idle_mask, advance_mask)
        to_target_to_idle = torch.logical_and(idle_to_target_mask, advance_mask)
        to_done = torch.logical_and(target_to_idle_mask, advance_mask)

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
            self.task_index[to_done] += 1
            self.new_task_available[to_done] = False
            self.new_task_index[to_done] = -1

        active_mask = self.state != self.IDLE
        self.state_timer[active_mask] += 1
        self.state_timer[~active_mask] = 0

        # Keep vertical orientation while IDLE, but track current EE yaw.
        idle_mask = self.state == self.IDLE
        if torch.any(idle_mask):
            _, _, ee_yaw = math_utils.euler_xyz_from_quat(math_utils.normalize(ee_quat_w))
            idle_quat_w = self._vertical_quat_with_yaw(ee_yaw, dtype=root_quat_w.dtype)
            target_quat_w[idle_mask] = idle_quat_w[idle_mask]

        target_pos_all, target_quat_all = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        return target_pos_all, target_quat_all, gripper_cmd_all
