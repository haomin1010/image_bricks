from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

# Isaac Lab and Gym imports will be deferred inside the Actor to ensure
# they are only loaded in the process that has the simulation_app.

class StackingStateMachine:
    def __init__(
        self,
        num_envs,
        device,
        scene,
        cube_names,
        max_tasks=8,
        cube_z_size=0.025,
        grid_origin=[0.5, 0.0, 0.001],
        cell_size=0.056,
        grid_size=8,
    ):
        self.num_envs = num_envs
        self.device = device
        self.scene = scene
        self.cube_names = cube_names
        self.max_tasks = max_tasks
        self.cube_z_size = cube_z_size
        self.grid_origin = torch.tensor(grid_origin, device=device)
        self.cell_size = cell_size
        
        # Per-environment target data
        self.target_positions = torch.zeros((num_envs, max_tasks, 3), device=device)
        self.num_tasks_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        self.state = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.wait_timer = torch.zeros(num_envs, device=device)
        self.task_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(num_envs, device=device)
        self.lock_yaw = torch.zeros(num_envs, device=device)
        self.lock_pos = torch.zeros((num_envs, 3), device=device) # 新增：锁定目标位置
        self.rotation_target = torch.zeros(num_envs, device=device)
        self.grasp_yaw = torch.zeros(num_envs, device=device)
        
        # IDLE position: initial end-effector position
        self.idle_pos = torch.zeros((num_envs, 3), device=device)
        self.idle_pos_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # IDLE state constant
        self.IDLE = -2

        # Server-driven flags
        self.has_pending_targets = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.new_task_available = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # Magic-suction attachment state (updated by IsaacLab execution layer).
        self.attached_cube_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # 记录方块抓取位置：基于网格动态计算（默认放在网格右侧，垂直居中）
        half_width = (grid_size * cell_size) / 2.0
        src_x = grid_origin[0] + half_width + cell_size / 2.0
        src_y = grid_origin[1] - half_width
        self.source_pick_pos = torch.tensor([src_x, src_y], device=device)

        
        # PID state
        self.prev_error = torch.zeros((num_envs, 3), device=device)
        self.error_sum = torch.zeros((num_envs, 3), device=device)
        
        # States
        self.APPROACH_CUBE = 0
        self.DESCEND_CUBE = 1
        self.GRASP = 2
        self.LIFT = 3
        self.APPROACH_TARGET = 4
        self.DESCEND_TARGET = 5
        self.RELEASE = 6
        self.RETRACT = 7
        self.ROTATE_TARGET = 8
        self.ROTATE_BACK = 9

    def set_env_targets(self, env_ids, targets):
        """Update targets for specific environments. targets: list of (x,y,z) tuples"""
        for i, env_id in enumerate(env_ids):
            # Accept either torch tensors or numeric tuples; append new targets
            env_id = int(env_id)
            num_t = min(len(targets), self.max_tasks)
            start_idx = int(self.num_tasks_per_env[env_id].item())
            write_count = min(num_t, self.max_tasks - start_idx)
            for t_idx in range(write_count):
                val = targets[t_idx]
                if isinstance(val, torch.Tensor):
                    self.target_positions[env_id, start_idx + t_idx] = val.to(self.device)
                else:
                    self.target_positions[env_id, start_idx + t_idx] = torch.tensor(val, device=self.device)
            # update count
            self.num_tasks_per_env[env_id] = start_idx + write_count
            # mark new task available for teleport logic
            self.has_pending_targets[env_id] = True
            self.new_task_available[env_id] = True
            self.new_task_index[env_id] = start_idx
            if self.state[env_id] == self.IDLE:
                self.task_index[env_id] = start_idx
                self.state[env_id] = self.APPROACH_CUBE

    def reset_envs(self, env_ids):
        self.state[env_ids] = self.IDLE
        self.wait_timer[env_ids] = 0
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.prev_error[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.attached_cube_idx[env_ids] = -1 # 重置吸附状态
        # clear server-driven flags
        self.has_pending_targets[env_ids] = False
        self.new_task_available[env_ids] = False
        self.new_task_index[env_ids] = -1
        # Re-capture idle position from the first post-reset observation.
        self.idle_pos_initialized[env_ids] = False

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0
        self.error_sum[env_idx] *= 0
        self.prev_error[env_idx] *= 0

    def compute_ee_pose_targets(self, obs):
        """Compute absolute EE pose targets in world frame.

        This is the authoritative output from server-side task logic: pose coordinates
        (position + quaternion), not velocities.
        """
        if not isinstance(obs, dict):
            raise TypeError(f"Invalid observation container: {type(obs)}")

        policy_obs = obs.get("policy", obs)
        if not isinstance(policy_obs, dict):
            raise TypeError(f"Invalid policy observation container: {type(policy_obs)}")

        ee_pos = policy_obs.get("ee_pos", None)
        ee_quat = policy_obs.get("ee_quat", None)
        if ee_pos is None or ee_quat is None:
            raise KeyError(
                "Policy observations must contain EE pose terms. "
                "Expected keys: 'ee_pos' and 'ee_quat'. "
                f"Available keys: {list(policy_obs.keys())}"
            )

        # Record per-env idle position from the first valid observation after reset.
        new_idle_envs = ~self.idle_pos_initialized
        if torch.any(new_idle_envs):
            self.idle_pos[new_idle_envs] = ee_pos[new_idle_envs]
            self.idle_pos_initialized[new_idle_envs] = True

        target_pos_all = torch.zeros((self.num_envs, 3), device=self.device)
        target_quat_all = torch.zeros((self.num_envs, 4), device=self.device)
        gripper_cmd_all = torch.ones((self.num_envs,), device=self.device)

        cube_z_size = self.cube_z_size
        safe_z_local = 0.45

        for i in range(self.num_envs):
            env_origin = self.scene.env_origins[i]
            safe_z = env_origin[2] + safe_z_local
            source_pick_local = torch.tensor(
                [self.source_pick_pos[0], self.source_pick_pos[1], cube_z_size / 2.0],
                device=self.device,
            )
            source_pick_fallback = source_pick_local + env_origin

            if self.num_tasks_per_env[i] == 0 or self.task_index[i] >= self.num_tasks_per_env[i]:
                target_pos = self.idle_pos[i].clone()
                gripper_cmd = 1.0
            else:
                cube_name = self.cube_names[self.task_index[i].item()]
                cube_asset = self.scene[cube_name]
                cube_pos_world = cube_asset.data.root_pos_w[i]
                cube_quat_w = cube_asset.data.root_quat_w[i]
                _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w.unsqueeze(0))
                # Use live cube pose for grasp/descend targets. Fall back to nominal
                # source point if the cube pose is temporarily invalid.
                live_source_pos = cube_pos_world.clone()
                live_valid = bool(torch.isfinite(live_source_pos).all().item())
                if live_valid:
                    live_z = float(live_source_pos[2].item())
                    if live_z < -0.2 or live_z > 1.5:
                        live_valid = False
                if not live_valid:
                    live_source_pos = source_pick_fallback.clone()

                target_world_pos = self.target_positions[i, self.task_index[i]].clone()

                if self.state[i] == self.APPROACH_CUBE:
                    tool_target_pos = live_source_pos.clone()
                    tool_target_pos[2] = safe_z
                    target_pos = tool_target_pos
                    gripper_cmd = 1.0

                    _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat[i].unsqueeze(0))
                    yaw_diff = torch.abs(ee_yaw - cube_yaw)
                    yaw_diff = torch.min(yaw_diff, 2 * 3.14159 - yaw_diff)
                    if torch.norm(ee_pos[i] - target_pos) < 0.01 and yaw_diff < 0.03:
                        self.state[i] = self.DESCEND_CUBE
                        self.state_timer[i] = 0
                        self.lock_yaw[i] = cube_yaw.squeeze(0)

                elif self.state[i] == self.DESCEND_CUBE:
                    tool_target_pos = live_source_pos.clone()
                    tool_target_pos[2] = live_source_pos[2] + (cube_z_size / 2.0) + 0.01
                    target_pos = tool_target_pos
                    gripper_cmd = 1.0
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.01:
                        self.state[i] = self.GRASP
                        self.state_timer[i] = 0

                elif self.state[i] == self.GRASP:
                    tool_target_pos = live_source_pos.clone()
                    tool_target_pos[2] = live_source_pos[2] + (cube_z_size / 2.0) - 0.002
                    target_pos = tool_target_pos
                    gripper_cmd = -1.0
                    if int(self.attached_cube_idx[i].item()) == int(self.task_index[i].item()):
                        self.state[i] = self.LIFT
                        self.state_timer[i] = 0

                elif self.state[i] == self.LIFT:
                    target_pos = ee_pos[i].clone()
                    target_pos[2] = safe_z
                    gripper_cmd = -1.0
                    if ee_pos[i, 2] > safe_z - 0.02:
                        self.state[i] = self.APPROACH_TARGET
                        self.state_timer[i] = 0

                elif self.state[i] == self.APPROACH_TARGET:
                    tool_target_pos = target_world_pos.clone()
                    tool_target_pos[2] = safe_z
                    target_pos = tool_target_pos
                    gripper_cmd = -1.0
                    if torch.norm(ee_pos[i, :2] - target_pos[:2]) < 0.01:
                        self.state[i] = self.DESCEND_TARGET
                        self.state_timer[i] = 0

                elif self.state[i] == self.DESCEND_TARGET:
                    tool_target_pos = target_world_pos.clone()
                    tool_target_pos[2] = target_world_pos[2] + (cube_z_size / 2.0) + 0.005
                    target_pos = tool_target_pos
                    gripper_cmd = -1.0
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.01:
                        self.state[i] = self.RELEASE
                        self.wait_timer[i] = 2
                        self.state_timer[i] = 0

                elif self.state[i] == self.RELEASE:
                    target_pos = ee_pos[i].clone()
                    gripper_cmd = 1.0
                    self.wait_timer[i] -= 1
                    if self.wait_timer[i] <= 0:
                        self.state[i] = self.RETRACT
                        self.wait_timer[i] = 4
                        self.state_timer[i] = 0

                elif self.state[i] == self.RETRACT:
                    target_pos = ee_pos[i].clone()
                    target_pos[2] = safe_z
                    gripper_cmd = 1.0
                    if ee_pos[i, 2] > safe_z - 0.02 or self.state_timer[i] > 2400:
                        self.task_index[i] += 1
                        if self.task_index[i] >= self.num_tasks_per_env[i]:
                            self.state[i] = -1
                            self.wait_timer[i] = 8
                        else:
                            self.state[i] = self.APPROACH_CUBE
                            self.wait_timer[i] = 2
                        self.state_timer[i] = 0
                else:
                    target_pos = self.idle_pos[i].clone()
                    gripper_cmd = 1.0

            state_i = int(self.state[i].item())
            if state_i >= 0 and state_i not in [self.GRASP, self.RELEASE] and self.state_timer[i] > 150:
                print(f"[Env {i}] !!! STATE {state_i} TIMEOUT !!! Forcing transition. (server.py)")
                if state_i == self.APPROACH_CUBE:
                    self.state[i] = self.DESCEND_CUBE
                elif state_i == self.DESCEND_CUBE:
                    self.state[i] = self.GRASP
                elif state_i == self.LIFT:
                    self.state[i] = self.APPROACH_TARGET
                elif state_i == self.APPROACH_TARGET:
                    self.state[i] = self.DESCEND_TARGET
                elif state_i == self.DESCEND_TARGET:
                    self.state[i] = self.RELEASE
                self.state_timer[i] = 0

            target_yaw = torch.tensor([0.0], device=self.device)
            if self.task_index[i] < self.num_tasks_per_env[i]:
                if self.state[i] == self.APPROACH_CUBE:
                    cube_name = self.cube_names[self.task_index[i].item()]
                    cube_quat_w = self.scene[cube_name].data.root_quat_w[i]
                    _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w.unsqueeze(0))
                    target_yaw = cube_yaw.squeeze(0)
                elif self.state[i] == -1:
                    target_yaw = torch.tensor([0.0], device=self.device)
                else:
                    target_yaw = self.lock_yaw[i]

            target_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([1.5707], device=self.device),
                target_yaw,
            ).to(self.device).squeeze(0)
            quat_dot = torch.sum(target_quat * ee_quat[i])
            if quat_dot < 0:
                target_quat = -target_quat

            target_pos_all[i] = target_pos
            target_quat_all[i] = target_quat
            gripper_cmd_all[i] = gripper_cmd
            self.state_timer[i] += 1

        return target_pos_all, target_quat_all, gripper_cmd_all
