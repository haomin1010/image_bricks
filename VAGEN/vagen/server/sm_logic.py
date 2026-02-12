import torch
import isaaclab.utils.math as math_utils

class StackingStateMachine:
    def __init__(self, num_envs, device, scene, cube_names, max_tasks=8, cube_z_size=0.025, grid_origin=[0.5, 0.0, 0.001], cell_size=0.056):
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
        
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait_timer = torch.zeros(num_envs, device=device)
        self.task_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(num_envs, device=device)
        self.lock_yaw = torch.zeros(num_envs, device=device)
        self.lock_pos = torch.zeros((num_envs, 3), device=device) 
        self.rotation_target = torch.zeros(num_envs, device=device)
        self.grasp_yaw = torch.zeros(num_envs, device=device)
        
        # --- Magic Suction ---
        self.attached_cube_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # Pick position
        self.source_pick_pos = torch.tensor([0.3, -0.2], device=device)
        
        # PID state
        self.prev_error = torch.zeros((num_envs, 3), device=device)
        self.error_sum = torch.zeros((num_envs, 3), device=device)
        self.soft_start_timer = torch.zeros(num_envs, dtype=torch.long, device=device) 
        
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

    def apply_magic_suction(self, obs):
        """Magic Suction implementation."""
        ee_pos = obs['policy']['eef_pos']     
        ee_quat = obs['policy']['eef_quat']   
        env_origins = self.scene.env_origins
        
        # 1. Attach logic
        grabbing_mask = (self.state == self.GRASP) & (self.attached_cube_idx == -1)
        grabbing_env_ids = torch.where(grabbing_mask)[0]
        
        if grabbing_env_ids.numel() > 0:
            for cube_idx in range(len(self.cube_names)):
                checking_mask = (self.task_index[grabbing_env_ids] == cube_idx)
                if not checking_mask.any():
                    continue
                
                target_ids = grabbing_env_ids[checking_mask]
                cube_asset = self.scene[self.cube_names[cube_idx]]
                
                cube_pos_w = cube_asset.data.root_pos_w[target_ids]
                cube_pos_local = cube_pos_w - env_origins[target_ids]
                
                dists = torch.norm(ee_pos[target_ids] - cube_pos_local, dim=-1)
                can_attach = dists < 0.05 
                if can_attach.any():
                    self.attached_cube_idx[target_ids[can_attach]] = cube_idx
        
        # 2. Release logic
        releasing_mask = (self.state == self.RELEASE)
        if releasing_mask.any():
            self.attached_cube_idx[releasing_mask] = -1
        
        # 3. Synchronize pose
        for cube_idx in range(len(self.cube_names)):
            attached_mask = (self.attached_cube_idx == cube_idx)
            env_ids = torch.where(attached_mask)[0]
            
            if env_ids.numel() > 0:
                cube_asset = self.scene[self.cube_names[cube_idx]]
                target_cube_pos_local = ee_pos[env_ids].clone()
                target_cube_pos_local[:, 2] -= self.cube_z_size / 4.0
                target_cube_pos_w = target_cube_pos_local + env_origins[env_ids]
                
                root_poses = torch.cat([target_cube_pos_w, ee_quat[env_ids]], dim=-1)
                target_ids_int32 = env_ids.to(dtype=torch.int32)
                cube_asset.write_root_pose_to_sim(root_poses, env_ids=target_ids_int32)
                cube_asset.write_root_velocity_to_sim(torch.zeros((env_ids.numel(), 6), device=self.device), env_ids=target_ids_int32)

    def set_env_targets(self, env_ids, targets):
        for i, env_id in enumerate(env_ids):
            num_t = min(len(targets), self.max_tasks)
            self.num_tasks_per_env[env_id] = num_t
            for t_idx in range(num_t):
                self.target_positions[env_id, t_idx] = torch.tensor(targets[t_idx], device=self.device)

    def reset_envs(self, env_ids):
        self.state[env_ids] = 0
        self.wait_timer[env_ids] = 0
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.prev_error[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.soft_start_timer[env_ids] = 100 
        self.attached_cube_idx[env_ids] = -1 

    def compute_action(self, obs):
        policy_obs = obs['policy']
        ee_pos = policy_obs['eef_pos']
        ee_quat = policy_obs['eef_quat']
        
        num_actions = 7
        if "surface_gripper" not in self.scene.keys():
            num_actions = 6 
            
        actions = torch.zeros((self.num_envs, num_actions), device=self.device)
        
        grid_origin = self.grid_origin
        cell_size = self.cell_size
        cube_z_size = self.cube_z_size
        safe_z = 0.25 
        
        for i in range(self.num_envs):
            # Dynamic Source Position: Target the current cube's actual location
            if self.task_index[i] < len(self.cube_names):
                cube_name = self.cube_names[self.task_index[i].item()]
                cube_pos_w = self.scene[cube_name].data.root_pos_w[i]
                env_origin = self.scene.env_origins[i]
                
                current_source_pos = cube_pos_w - env_origin
                current_source_pos[2] = cube_z_size / 2.0
            else:
                # Default fallback if index out of bounds (e.g. finished)
                current_source_pos = torch.tensor([0.3, -0.2, cube_z_size/2.0], device=self.device)

            if self.num_tasks_per_env[i] == 0 or self.task_index[i] >= self.num_tasks_per_env[i]:
                target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)
                self.state[i] = -1 
                self.wait_timer[i] -= 1
            else:
                target_world_pos = self.target_positions[i, self.task_index[i]].clone()
                
                # State Machine Logic
                if self.state[i] == self.APPROACH_CUBE:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = safe_z
                    _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat[i].unsqueeze(0))
                    cube_name = self.cube_names[self.task_index[i].item()]
                    cube_quat_w = self.scene[cube_name].data.root_quat_w[i]
                    _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w.unsqueeze(0))
                    yaw_diff = torch.abs(ee_yaw - cube_yaw)
                    yaw_diff = torch.min(yaw_diff, 2 * 3.14159 - yaw_diff)
                    
                    if torch.norm(ee_pos[i] - target_pos) < 0.015 and yaw_diff < 0.04 and self.state_timer[i] > 20: 
                        self.state[i] = self.DESCEND_CUBE
                        self.state_timer[i] = 0
                        self.lock_yaw[i] = cube_yaw.squeeze(0)
                        
                elif self.state[i] == self.DESCEND_CUBE:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = cube_z_size - 0.002
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.015: 
                        self.state[i] = self.GRASP
                        self.wait_timer[i] = 40
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.GRASP:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = cube_z_size - 0.002
                    self.wait_timer[i] -= 1
                    if self.wait_timer[i] <= 0:
                        self.state[i] = self.LIFT
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.LIFT:
                    target_pos = ee_pos[i].clone()
                    target_pos[2] = safe_z
                    if ee_pos[i, 2] > safe_z - 0.02:
                        self.state[i] = self.APPROACH_TARGET
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.APPROACH_TARGET:
                    target_pos = target_world_pos.clone()
                    target_pos[2] = safe_z
                    if torch.norm(ee_pos[i, :2] - target_pos[:2]) < 0.05:
                        self.state[i] = self.DESCEND_TARGET
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.DESCEND_TARGET:
                    target_pos = target_world_pos.clone()
                    target_pos[2] = target_world_pos[2] + (cube_z_size / 2.0) - 0.002
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.015:
                        self.state[i] = self.RELEASE
                        self.wait_timer[i] = 20
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.RELEASE:
                    target_pos = ee_pos[i].clone()
                    self.wait_timer[i] -= 1
                    if self.wait_timer[i] <= 0:
                        self.state[i] = self.RETRACT
                        self.wait_timer[i] = 30 
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.RETRACT:
                    target_pos = ee_pos[i].clone()
                    target_pos[2] = safe_z
                    if ee_pos[i, 2] > safe_z - 0.02 or self.state_timer[i] > 2400:
                        self.task_index[i] += 1
                        # If we reached the number of requested tasks, go to IDLE (-1)
                        if self.task_index[i] >= self.num_tasks_per_env[i]:
                            self.state[i] = -1
                            self.wait_timer[i] = 60 
                        else:
                            # Otherwise (if pre-planned), continue
                            self.state[i] = self.APPROACH_CUBE
                            self.wait_timer[i] = 20 
                        self.state_timer[i] = 0
                else:
                    # IDLE State (-1) or Finished
                    # Hold a safe "home" position to avoid blocking cameras or colliding
                    target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)

            # Action calculation (IK/PID) - simplified for server
            diff_pos = target_pos - ee_pos[i]
            
            # PID
            kp, ki, kd = 2.0, 0.01, 0.05
            self.error_sum[i] += diff_pos
            d_error = diff_pos - self.prev_error[i]
            vel = kp * diff_pos + ki * self.error_sum[i] + kd * d_error
            self.prev_error[i] = diff_pos
            
            vel = torch.clamp(vel, -0.5, 0.5)
            
            # Orientation
            if self.state[i] in [self.APPROACH_CUBE, self.DESCEND_CUBE, self.GRASP, self.LIFT]:
                target_yaw = self.lock_yaw[i]
            else:
                target_yaw = 0.0
                
            target_quat = math_utils.quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device),
                torch.tensor(3.14159, device=self.device),
                torch.tensor(target_yaw, device=self.device)
            )
            
            quat_error = math_utils.quat_mul(target_quat, math_utils.quat_inv(ee_quat[i]))
            axis, angle = math_utils.axis_angle_from_quat(quat_error)
            # Handle wrapping
            angle = torch.where(angle > 3.14159, angle - 2 * 3.14159, angle)
            ang_vel = axis * angle * 2.0
            ang_vel = torch.clamp(ang_vel, -1.0, 1.0)
            
            actions[i, :3] = vel
            actions[i, 3:6] = ang_vel
            # Grip action removed (Magic Suction handles it internally)
            
            self.state_timer[i] += 1
            
        return actions
