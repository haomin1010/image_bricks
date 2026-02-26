from asyncio.log import logger
from pathlib import Path
import os
import torch
import ray
import asyncio
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# Additional imports for StackingStateMachine
import gymnasium as gym
import torch.distributions
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import copy
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf, Sdf, UsdPhysics, UsdGeom, PhysxSchema

# Isaac Lab and Gym imports will be deferred inside the Actor to ensure 
# they are only loaded in the process that has the simulation_app.

class StackingStateMachine:
    def __init__(self, num_envs, device, scene, cube_names, max_tasks=8, cube_z_size=0.025, grid_origin=[0.5, 0.0, 0.001], cell_size=0.056, grid_size=8):
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

        # IDLE state constant
        self.IDLE = -2

        # Server-driven flags
        self.has_pending_targets = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.new_task_available = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # --- Magic Suction 状态 ---
        self.attached_cube_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        # GPU-native suction: create/destroy fixed joints per env.
        self._stage = None
        self._active_joint_paths = [None] * num_envs
        
        # 记录方块抓取位置：基于网格动态计算（默认放在网格右侧，垂直居中）
        half_width = (grid_size * cell_size) / 2.0
        src_x = grid_origin[0] + half_width + cell_size / 2.0
        src_y = grid_origin[1] - half_width
        self.source_pick_pos = torch.tensor([src_x, src_y], device=device)
        
        # PID state
        self.prev_error = torch.zeros((num_envs, 3), device=device)
        self.error_sum = torch.zeros((num_envs, 3), device=device)

        # Internal buffers for suction synchronization with physics steps
        self._desired_local_pos = torch.zeros((num_envs, 3), device=device)
        self._desired_quat = torch.zeros((num_envs, 4), device=device)
        self.last_sim_step = -1
        self._suction_attach_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._suction_detach_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Latest observation cache for physics callback suction
        self._last_obs = None
        self._cb_error_logged = False
        
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

    def set_stage(self, stage):
        self._stage = stage

    def set_last_obs(self, obs):
        self._last_obs = obs
        self._cb_error_logged = False

    def physics_suction_cb(self, dt):
        if not isinstance(self._last_obs, dict):
            return
        if "policy" not in self._last_obs:
            return
        self.apply_magic_suction(self._last_obs)


    def apply_magic_suction(self, obs):
        env_origins = self.scene.env_origins
        policy_obs = obs.get("policy", {})
        ee_pos = policy_obs.get("eef_pos")
        ee_quat = policy_obs.get("eef_quat")
        if ee_pos is None:
            return

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

        # 2) Detach condition: release only
        releasing_mask = (self.state == self.RELEASE)
        if releasing_mask.any():
            self.attached_cube_idx[releasing_mask] = -1

        # 3) Sync attached cube pose
        for cube_idx in range(len(self.cube_names)):
            attached_mask = (self.attached_cube_idx == cube_idx)
            env_ids = torch.where(attached_mask)[0]
            if env_ids.numel() == 0:
                continue
            cube_asset = self.scene[self.cube_names[cube_idx]]
            target_cube_pos_local = ee_pos[env_ids].clone()
            target_cube_pos_local[:, 2] -= self.cube_z_size / 4.0
            target_cube_pos_w = target_cube_pos_local + env_origins[env_ids]
            if ee_quat is None:
                quat = cube_asset.data.root_quat_w[env_ids]
            else:
                quat = ee_quat[env_ids]
            root_poses = torch.cat([target_cube_pos_w, quat], dim=-1)
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            cube_asset.write_root_pose_to_sim(root_poses, env_ids=env_ids_int32)
            cube_asset.write_root_velocity_to_sim(
                torch.zeros((env_ids.numel(), 6), device=self.device),
                env_ids=env_ids_int32,
            )
        return

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
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
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

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0
        self.error_sum[env_idx] *= 0
        self.prev_error[env_idx] *= 0

    def compute_action(self, obs):
        policy_obs = obs['policy']
        ee_pos = policy_obs['eef_pos']
        ee_quat = policy_obs['eef_quat']
        env_origins = self.scene.env_origins
        
        # Get cube positions and orientations
        cube_assets = [self.scene[name] for name in self.cube_names]
        
        # Dynamically infer action dimension from observation to avoid 6/7D mismatches.
        num_actions = int(policy_obs["actions"].shape[-1])
            
        actions = torch.zeros((self.num_envs, num_actions), device=self.device)
        
        # Grid parameters from self
        grid_origin = self.grid_origin
        cell_size = self.cell_size
        cube_z_size = self.cube_z_size
        safe_z = 0.45 # Hover height (increased)
        
        for i in range(self.num_envs):
            # 1. current_source_pos 应该使用类实例中定义的 self.source_pick_pos
            # 2. 之前硬编码导致计算了两次偏移
            current_source_pos = torch.tensor([
                self.source_pick_pos[0], 
                self.source_pick_pos[1], 
                cube_z_size / 2.0
            ], device=self.device)

            if self.num_tasks_per_env[i] == 0 or self.task_index[i] >= self.num_tasks_per_env[i]:
                #logger.debug(f"[Env {i}] No more tasks or invalid task index. Entering IDLE state.")
                # 所有任务完成或无任务：停留在安全硬编码位置，并允许计时器递减以触发重置
                target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)
                gripper_cmd = 1.0 # 停止吸气
                #self.state[i] = -1 # Finished state
                #self.wait_timer[i] -= 1
            else:
                # 获取任务信息
                target_world_pos = self.target_positions[i, self.task_index[i]].clone()
                # Convert stored world-space target into env-local coordinates
                try:
                    target_local = target_world_pos - env_origins[i]
                except Exception:
                    target_local = target_world_pos.clone()
                
                # --- State Machine Logic (Single Chain) ---
                if self.state[i] == self.APPROACH_CUBE:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = safe_z
                    gripper_cmd = 1.0 
                    
                    # 检查位置和水平旋转误差
                    # 我们在下方会计算 target_quat，但这里可以先做一个简单的 Yaw 检查
                    # 获取当前 EE 的 Yaw
                    _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat[i].unsqueeze(0))
                    # 获取方块的 Yaw
                    cube_name = self.cube_names[self.task_index[i].item()]
                    cube_quat_w = self.scene[cube_name].data.root_quat_w[i]
                    _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w.unsqueeze(0))
                    
                    yaw_diff = torch.abs(ee_yaw - cube_yaw)
                    # 处理角度回绕
                    yaw_diff = torch.min(yaw_diff, 2 * 3.14159 - yaw_diff)
                    
                    # 严谨化：只有当位置到位、旋转对齐，且在该状态稳定停留一小段时间后才下降
                    if torch.norm(ee_pos[i] - target_pos) < 0.01 and yaw_diff < 0.03: 
                        self.state[i] = self.DESCEND_CUBE
                        self.state_timer[i] = 0
                        # 锁定此时的 Yaw，防止后续搬运过程中因为方块旋转导致机械臂乱扭
                        self.lock_yaw[i] = cube_yaw.squeeze(0)
                        
                elif self.state[i] == self.DESCEND_CUBE:
                    target_pos = current_source_pos.clone()
                    # Descend target: stop a bit higher above the cube/table for safety
                    target_pos[2] = cube_z_size + 0.01
                    gripper_cmd = 1.0 
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.01: 
                        self.state[i] = self.GRASP
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.GRASP:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = cube_z_size - 0.002
                    gripper_cmd = -1.0 
                    # 只在成功吸附后切换到 LIFT
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
                    target_pos = target_local.clone()
                    target_pos[2] = safe_z
                    gripper_cmd = -1.0
                    if torch.norm(ee_pos[i, :2] - target_pos[:2]) < 0.01:
                        self.state[i] = self.DESCEND_TARGET
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.DESCEND_TARGET:
                    target_pos = target_local.clone()
                    # Descend target at placement: stop slightly above contact to reduce collisions
                    target_pos[2] = target_local[2] + (cube_z_size / 2.0) + 0.005
                    gripper_cmd = -1.0 
                    # 扩大到达判定范围，防止因为物理碰撞造成的细微抖动导致无法切换状态
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
                        self.wait_timer[i] = 4 # 虽然 RETRACT 判位置，但 wait_timer 留着备用
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.RETRACT:
                    target_pos = ee_pos[i].clone()
                    target_pos[2] = safe_z
                    gripper_cmd = 1.0 
                    if ee_pos[i, 2] > safe_z - 0.02 or self.state_timer[i] > 2400:
                        self.task_index[i] += 1
                        # 检查是否完成所有任务
                        if self.task_index[i] >= self.num_tasks_per_env[i]:
                            self.state[i] = -1
                            self.wait_timer[i] = 8 # 任务完成后短暂停留再 reset
                        else:
                            self.state[i] = self.APPROACH_CUBE
                            self.wait_timer[i] = 2 
                        self.state_timer[i] = 0
                else:
                    target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)
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
                elif state_i == self.RETRACT:
                    pass
                self.state_timer[i] = 0

            # 调整控制增益与速度限制，使动作更轻柔并保持精度
            kp = 0.45

            if self.state[i] in [self.DESCEND_CUBE, self.DESCEND_TARGET]:
                current_max_vel = 0.05 # 5cm/s
            elif self.state[i] in [self.GRASP, self.RELEASE]:
                current_max_vel = 0.003 # 0.3cm/s for fine adjustments
            elif self.state[i] in [self.LIFT, self.RETRACT]:
                current_max_vel = 0.08 # 8cm/s
            else:
                current_max_vel = 0.1 # 10cm/s (Approach stage)
            
            diff_pos = target_pos - ee_pos[i]
            pos_action = diff_pos * kp
            
            # 使用最简化的限速逻辑
            pos_norm = torch.norm(pos_action)
            if pos_norm > current_max_vel:
                pos_action = pos_action * (current_max_vel / pos_norm)
            
            # 计算目标旋转：
            # 1. 基础姿态为垂直向下 (Pitch = 90deg)
            # 2. 增加动态 Yaw 对齐：抓取阶段对齐源方块，之后锁定该 Yaw。
            target_yaw = torch.tensor([0.0], device=self.device)
            if self.task_index[i] < self.num_tasks_per_env[i]:
                # 只有在 APPROACH_CUBE 状态（还未开始下降）时，动态跟踪方块 Yaw
                if self.state[i] == self.APPROACH_CUBE:
                    cube_name = self.cube_names[self.task_index[i].item()]
                    cube_quat_w = self.scene[cube_name].data.root_quat_w[i]
                    _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_w.unsqueeze(0))
                    target_yaw = cube_yaw.squeeze(0)
                elif self.state[i] == -1: # 完成状态使用默认
                    target_yaw = torch.tensor([0.0], device=self.device)
                else:
                    # 一旦进入 DESCEND_CUBE 或更高阶状态，严格使用锁定的 Yaw，绝对禁止实时更新
                    target_yaw = self.lock_yaw[i]

            # UR10 Orientation: 核心垂直向下姿态 ([Pitch=90deg]) + 动态 Yaw
            target_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([1.5707], device=self.device),
                target_yaw
            ).to(self.device).squeeze(0)

            current_rot_max_vel = 0.1 # 限制旋转速度，防止旋转过猛造成的“甩动"
            
            actions[i, :3] = pos_action
            
            if torch.sum(target_quat * ee_quat[i]) < 0:
                target_quat = -target_quat
            
            q_error = math_utils.quat_mul(target_quat, math_utils.quat_inv(ee_quat[i]))
            rot_action = math_utils.axis_angle_from_quat(q_error) * 1.2 

            # 对旋转也进行限速
            rot_norm = torch.norm(rot_action)
            if rot_norm > current_rot_max_vel:
                rot_action = rot_action * (current_rot_max_vel / rot_norm)
            
            actions[i, 3:6] = rot_action
            if num_actions > 6:
                actions[i, 6] = gripper_cmd

            self.state_timer[i] += 1
        return actions

def get_stack_cube_env_cfg(task_name, device, num_envs, enable_cameras):
    # create environment
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs) 
    if hasattr(env_cfg, "device"): 
        env_cfg.device = device

    if hasattr(env_cfg.scene, "surface_gripper"):
        print("[INFO]: Switching to GPU Magic Suction. Disabling CPU SurfaceGripper plugin.")
        env_cfg.scene.surface_gripper = None
    for action_key in ["gripper_action", "suction_gripper", "gripper"]:
        if hasattr(env_cfg.actions, action_key):
            setattr(env_cfg.actions, action_key, None)
    if hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "gripper_pos"):
            env_cfg.observations.policy.gripper_pos = None
    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms = None

    # Grid Parameters (match batch_gen.py canonical values)
    grid_origin = [0.5, 0.0, 0.001]
    grid_size = 8
    line_thickness = 0.001
    # Increase cell size to be slightly larger than the cube (0.04m)
    # for easier placement and better visual spacing.
    cell_size = 0.055 + line_thickness
    half_width = (grid_size) * cell_size / 2
    

    # Spawning Logic: We spawn a fixed number of cubes to handle various tasks
    max_cubes = 8
    cube_size = 0.045 # Exact cube size as requested by user
    repo_root = Path(__file__).resolve().parents[3]
    isaac_assets = repo_root / "assets" / "Isaac"
    blue_usd_candidate = isaac_assets / "Props" / "Blocks" / "blue_block.usd"
    if blue_usd_candidate.exists():
        blue_usd = str(blue_usd_candidate)
        print(f"[INFO]: Using local asset at {blue_usd}")
    else:
        blue_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
        print(f"[WARN]: Local asset not found, falling back to {blue_usd}")
    robot_usd_candidate = isaac_assets / "Robots" / "UniversalRobots" / "ur10" / "ur10.usd"
    if robot_usd_candidate.exists() and hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
        if hasattr(env_cfg.scene.robot.spawn, "usd_path"):
            env_cfg.scene.robot.spawn.usd_path = str(robot_usd_candidate)
            print(f"[INFO]: Using local robot asset at {robot_usd_candidate}")
    table_usd_candidate = isaac_assets / "Props" / "Mounts" / "SeattleLabTable" / "table_instanceable.usd"
    if table_usd_candidate.exists() and hasattr(env_cfg.scene, "table"):
        if hasattr(env_cfg.scene.table.spawn, "usd_path"):
            env_cfg.scene.table.spawn.usd_path = str(table_usd_candidate)
            print(f"[INFO]: Using local table asset at {table_usd_candidate}")
    ground_usd_candidate = isaac_assets / "Environments" / "Grid" / "default_environment.usd"
    if ground_usd_candidate.exists() and hasattr(env_cfg.scene, "plane"):
        if hasattr(env_cfg.scene.plane.spawn, "usd_path"):
            env_cfg.scene.plane.spawn.usd_path = str(ground_usd_candidate)
            print(f"[INFO]: Using local ground asset at {ground_usd_candidate}")
    cube_names = [f"cube_{i+1}" for i in range(max_cubes)]
    
    # 核心回归：取料位置恢复到 [0.3, -0.2]
    # Compute source pick position relative to grid: place to the right of the grid
    # and vertically centered on the grid origin row for consistent pickup location.
    source_pick_pos_x = grid_origin[0] + half_width + cell_size / 2.0
    source_pick_pos_y = grid_origin[1] - half_width
    
    aligned_poses = []
    for i in range(max_cubes):
        if i == 0:
            # 第一个方块在桌面上
            aligned_poses.append([source_pick_pos_x, source_pick_pos_y, cube_size / 2.0, 1.0, 0.0, 0.0, 0.0])
        else:
            # 其余方块在地下水平排开（X轴方向负向延伸），远离桌面工作区
            aligned_poses.append([-0.5 - (i * 0.1), 0.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    # First, hide ALL potential default cubes mapping from the original scene (cube_1, 2, 3)
    # This prevents red/green cubes from staying in the scene.
    for default_name in ["cube_1", "cube_2", "cube_3"]:
        if hasattr(env_cfg.scene, default_name):
            getattr(env_cfg.scene, default_name).init_state.pos = (0.0, 0.0, -10.0)

    # Base cube to copy from (env_cfg has cube_1, cube_2, cube_3 by default)
    # We'll re-configure them and add missing ones
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import SceneEntityCfg
    
    # 配置机械臂操作的蓝色方块
    for i, name in enumerate(cube_names):
        pos = aligned_poses[i]
        if i < 3: # Modify existing cube_1, cube_2, cube_3
            cube_cfg = getattr(env_cfg.scene, name)
            cube_cfg.spawn.usd_path = blue_usd
            cube_cfg.init_state.pos = (pos[0], pos[1], pos[2])
            cube_cfg.init_state.rot = (pos[3], pos[4], pos[5], pos[6])
        else: # Add new cubes
            new_cube = copy.deepcopy(env_cfg.scene.cube_1)
            new_cube.prim_path = f"{{ENV_REGEX_NS}}/Cube_{i+1}"
            new_cube.spawn.semantic_tags = [("class", name)]
            new_cube.init_state.pos = (pos[0], pos[1], pos[2])
            new_cube.init_state.rot = (pos[3], pos[4], pos[5], pos[6])
            setattr(env_cfg.scene, name, new_cube)

    # Override the randomization event
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions.params["poses"] = aligned_poses
        # Ensure all cubes are in the asset_cfgs list
        env_cfg.events.randomize_cube_positions.params["asset_cfgs"] = [SceneEntityCfg(name) for name in cube_names]
    
    # Disable default terminations to let our state machine handle successes and resets
    if hasattr(env_cfg, "terminations"):
        for term_name in list(env_cfg.terminations.__dict__.keys()):
            if "cube" in term_name or "success" in term_name:
                setattr(env_cfg.terminations, term_name, None)
    # Increase time limit to allow stacking many cubes (default is often 30s)
    env_cfg.episode_length_s = 600.0 # 10 minutes

    # --- 核心性能优化：移除所有默认相机，只保留我们需要的相机 ---
    # 这样可以极大地节省 GPU Descriptor Sets
    default_cams = ["table_cam", "table_high_cam", "robot_cam", "cam_default"]
    for d_cam in default_cams:
        if hasattr(env_cfg.scene, d_cam):
            setattr(env_cfg.scene, d_cam, None)

    # Add dataset-aligned cameras (Only if enable_cameras is requested)
    if enable_cameras:
        from isaaclab.sensors import CameraCfg, TiledCameraCfg
        use_tiled = os.getenv("VAGEN_USE_TILED", "1") != "0"
        cam_cfg_type = TiledCameraCfg if use_tiled else CameraCfg
        print(f"[DEBUG]: Camera config type: {'TiledCameraCfg' if use_tiled else 'CameraCfg'}")
        # Use dataset-aligned camera parameters (copy from batch_gen.py)
        TABLE_HEIGHT = 1.03
        # Table prim in StackEnvCfg uses an init_state.pos of (0.0, 0.0, 1.03).
        # Camera positions should be expressed relative to the Table origin,
        # so subtract the table init-state position when computing camera world
        # offsets for environments that place the table at z=1.03.
        TABLE_INIT_POS = (0.5, 0.0, -1.03)
        CAMERA_HEIGHT = 0.7
        CUBE_SIZE = cube_size

        camera_configs = {
            "camera": {
                "pos": (0.0, 0.0, TABLE_HEIGHT + CAMERA_HEIGHT + 0.5),
                "rot": (0.707, 0.0, 0.707, 0.0),
            },
            "camera_front": {
                "pos": (CAMERA_HEIGHT, 0.0, TABLE_HEIGHT + CUBE_SIZE * 2),
                "rot": (0.0, 0.0, 0.0, 1.0),
            },
            "camera_side": {
                "pos": (0.0, CAMERA_HEIGHT, TABLE_HEIGHT + CUBE_SIZE * 2),
                "rot": (0.707, 0.0, 0.0, -0.707),
            },
            "camera_iso": {
                "pos": (-CAMERA_HEIGHT / np.sqrt(2), CAMERA_HEIGHT / np.sqrt(2), TABLE_HEIGHT + CAMERA_HEIGHT),
                "rot": (0.85355, 0.14645, 0.35355, -0.35355),
            },
            "camera_iso2": {
                "pos": (CAMERA_HEIGHT / np.sqrt(2), -CAMERA_HEIGHT / np.sqrt(2), TABLE_HEIGHT + CAMERA_HEIGHT),
                "rot": (0.36, -0.33, 0.14, 0.85),
            },
        }

        # For each desired view spawn a TiledCameraCfg that matches one prim per env
        # (prim_path is per-camera name, not a regex alternation). This way each
        # TiledCamera has exactly one prim per env and tiled rendering will be used
        # internally to accelerate rendering while avoiding leaf-level regex checks.
        cam_names = list(camera_configs.keys())
        for cam_name in cam_names:
            cam_cfg = camera_configs[cam_name]
            pos = cam_cfg["pos"]
            # convert to table-relative coordinates by subtracting the table init pos
            pos = (pos[0] + TABLE_INIT_POS[0], pos[1] + TABLE_INIT_POS[1], pos[2] + TABLE_INIT_POS[2])
            rot = cam_cfg["rot"]
            # Debug: log camera config before attaching to env
            print(
                "[DEBUG]: Adding tiled camera",
                {
                    "name": cam_name,
                    "prim_path": f"{{ENV_REGEX_NS}}/{cam_name}",
                    "pos": pos,
                    "rot": rot,
                    "width": 224,
                    "height": 224,
                    "data_types": ["rgb"],
                    "clipping_range": (0.01, 1000.0),
                },
            )
            setattr(
                env_cfg.scene,
                cam_name,
                cam_cfg_type(
                    prim_path=f"{{ENV_REGEX_NS}}/{cam_name}",
                    update_period=0.0,
                    height=224,
                    width=224,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.01, 1000.0),
                    ),
                    offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=(rot[0], rot[1], rot[2], rot[3]), convention="world"),
                ),
            )
    else:
        print("[INFO]: Cameras are disabled via CLI. Skipping TiledCamera setup.")

    # Add Grid Lines (Physical Assets)
    origin = grid_origin
    for i in range(grid_size + 1):
        suffix = f"_{i}"
        # Horizontal lines
        y_pos = origin[1] - half_width + i * cell_size
        setattr(env_cfg.scene, f"grid_h{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_h{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(grid_size * cell_size + line_thickness, line_thickness, 0.0002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                semantic_tags=[("class", "grid")],
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], y_pos, origin[2]))
        ))
        # Vertical lines
        x_pos = origin[0] - half_width + i * cell_size
        setattr(env_cfg.scene, f"grid_v{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_v{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(line_thickness, grid_size * cell_size + line_thickness, 0.0002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                semantic_tags=[("class", "grid")],
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x_pos, origin[1], origin[2]))
        ))

    # Add colored axis markers and a yellow origin marker for easy visual alignment
    # Place axes along the grid edges (bottom and left) and origin at bottom-left corner
    setattr(env_cfg.scene, "grid_x_axis", AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/grid_x_axis",
        spawn=sim_utils.CuboidCfg(
            size=(grid_size * cell_size + line_thickness, line_thickness * 3.0, 0.0004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            semantic_tags=[("class", "grid_axis")],
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], origin[1] - half_width, origin[2]))
    ))

    setattr(env_cfg.scene, "grid_y_axis", AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/grid_y_axis",
        spawn=sim_utils.CuboidCfg(
            size=(line_thickness * 3.0, grid_size * cell_size + line_thickness, 0.0004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            semantic_tags=[("class", "grid_axis")],
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0] - half_width, origin[1], origin[2]))
    ))

    setattr(env_cfg.scene, "grid_origin_marker", AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/grid_origin_marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.0006),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            semantic_tags=[("class", "grid_origin")],
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0] - half_width, origin[1] - half_width, origin[2] + 0.0003))
    ))

    return env_cfg
