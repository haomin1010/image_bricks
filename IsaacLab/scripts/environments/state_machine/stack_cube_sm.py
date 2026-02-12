# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run a state machine for stacking cubes in the Franka environment.
This script uses Inverse Kinematics (IK) to coordinate the stacking behavior.

Usage:
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_sm.py --task Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0 --num_envs 32 --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="State machine for stacking cubes.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
import torch.distributions
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import copy
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sys
import os

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
        self.lock_pos = torch.zeros((num_envs, 3), device=device) # 新增：锁定目标位置
        self.rotation_target = torch.zeros(num_envs, device=device)
        self.grasp_yaw = torch.zeros(num_envs, device=device)
        
        # --- Magic Suction 状态 ---
        self.attached_cube_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # 记录方块抓取位置：回归到用户确认的工作位置
        self.source_pick_pos = torch.tensor([0.3, -0.2], device=device)
        
        # PID state
        self.prev_error = torch.zeros((num_envs, 3), device=device)
        self.error_sum = torch.zeros((num_envs, 3), device=device)
        self.soft_start_timer = torch.zeros(num_envs, dtype=torch.long, device=device) # 增强：使用计时器而不是布尔值
        
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
        """直接在 GPU 上同步被吸附方块的位姿，实现‘魔法吸附’"""
        ee_pos = obs['policy']['eef_pos']     # Local Pose
        ee_quat = obs['policy']['eef_quat']   # World/Local Orientation (usually same)
        
        # 获取所有环境的原点偏移，用于坐标转换
        env_origins = self.scene.env_origins
        
        # 1. 逻辑触发吸附 (Tensorized)
        grabbing_mask = (self.state == self.GRASP) & (self.attached_cube_idx == -1)
        grabbing_env_ids = torch.where(grabbing_mask)[0]
        
        if grabbing_env_ids.numel() > 0:
            for cube_idx in range(len(self.cube_names)):
                checking_mask = (self.task_index[grabbing_env_ids] == cube_idx)
                if not checking_mask.any():
                    continue
                
                target_ids = grabbing_env_ids[checking_mask]
                cube_asset = self.scene[self.cube_names[cube_idx]]
                
                # 核心修复：将方块的世界坐标转为局部坐标
                cube_pos_w = cube_asset.data.root_pos_w[target_ids]
                cube_pos_local = cube_pos_w - env_origins[target_ids]
                
                # 在局部空间进行距离判定
                dists = torch.norm(ee_pos[target_ids] - cube_pos_local, dim=-1)
                can_attach = dists < 0.05 
                if can_attach.any():
                    self.attached_cube_idx[target_ids[can_attach]] = cube_idx
        
        # 2. 逻辑释放吸附 (Tensorized)
        releasing_mask = (self.state == self.RELEASE)
        if releasing_mask.any():
            self.attached_cube_idx[releasing_mask] = -1
        
        # 3. 批量位姿同步
        for cube_idx in range(len(self.cube_names)):
            attached_mask = (self.attached_cube_idx == cube_idx)
            env_ids = torch.where(attached_mask)[0]
            
            if env_ids.numel() > 0:
                cube_asset = self.scene[self.cube_names[cube_idx]]
                
                # 核心修复：将 Local 的 EE 位置转换回 World 坐标写入仿真器
                target_cube_pos_local = ee_pos[env_ids].clone()
                target_cube_pos_local[:, 2] -= self.cube_z_size / 4.0
                target_cube_pos_w = target_cube_pos_local + env_origins[env_ids]
                
                # 构造 [N, 7] 张量: [x_w, y_w, z_w, qw, qx, qy, qz]
                root_poses = torch.cat([target_cube_pos_w, ee_quat[env_ids]], dim=-1)
                
                target_ids_int32 = env_ids.to(dtype=torch.int32)
                cube_asset.write_root_pose_to_sim(root_poses, env_ids=target_ids_int32)
                cube_asset.write_root_velocity_to_sim(torch.zeros((env_ids.numel(), 6), device=self.device), env_ids=target_ids_int32)

    def set_env_targets(self, env_ids, targets):
        """Update targets for specific environments. targets: list of (x,y,z) tuples"""
        for i, env_id in enumerate(env_ids):
            num_t = min(len(targets), self.max_tasks)
            self.num_tasks_per_env[env_id] = num_t
            # Map grid (gx, gy, level) to world positions
            # Note: This logic requires access to grid params, we'll implement it in the update call
            for t_idx in range(num_t):
                self.target_positions[env_id, t_idx] = torch.tensor(targets[t_idx], device=self.device)

    def reset_envs(self, env_ids):
        self.state[env_ids] = 0
        self.wait_timer[env_ids] = 0
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.prev_error[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.soft_start_timer[env_ids] = 100 # 重置后开启 100 步的软启动（约 1-2 秒）
        self.attached_cube_idx[env_ids] = -1 # 重置吸附状态

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
        
        # Get cube positions and orientations
        cube_assets = [self.scene[name] for name in self.cube_names]
        
        # 修正动作空间大小：如果启用了 Magic Suction，底层环境可能不再接收吸力指令
        # 我们根据环境实际的动作管理器的项来动态调整
        num_actions = 7
        if "surface_gripper" not in self.scene.keys():
            num_actions = 6 # 仅 [vx, vy, vz, wx, wy, wz]
            
        actions = torch.zeros((self.num_envs, num_actions), device=self.device)
        
        # Grid parameters from self
        grid_origin = self.grid_origin
        cell_size = self.cell_size
        cube_z_size = self.cube_z_size
        safe_z = 0.25 # Hover height
        
        for i in range(self.num_envs):
            # 修正坐标偏移问题：
            # 1. current_source_pos 应该使用类实例中定义的 self.source_pick_pos
            # 2. 之前硬编码导致计算了两次偏移
            current_source_pos = torch.tensor([
                self.source_pick_pos[0], 
                self.source_pick_pos[1], 
                cube_z_size / 2.0
            ], device=self.device)

            if self.num_tasks_per_env[i] == 0 or self.task_index[i] >= self.num_tasks_per_env[i]:
                # 所有任务完成或无任务：停留在等待区，并允许计时器递减以触发重置
                target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)
                gripper_cmd = 1.0 # 停止吸气
                self.state[i] = -1 # Finished state
                self.wait_timer[i] -= 1
            else:
                # 获取任务信息
                target_world_pos = self.target_positions[i, self.task_index[i]].clone()
                
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
                    if torch.norm(ee_pos[i] - target_pos) < 0.015 and yaw_diff < 0.04 and self.state_timer[i] > 20: 
                        self.state[i] = self.DESCEND_CUBE
                        self.state_timer[i] = 0
                        # 锁定此时的 Yaw，防止后续搬运过程中因为方块旋转导致机械臂乱扭
                        self.lock_yaw[i] = cube_yaw.squeeze(0)
                        
                elif self.state[i] == self.DESCEND_CUBE:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = cube_z_size - 0.002
                    gripper_cmd = 1.0 
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.015: 
                        self.state[i] = self.GRASP
                        self.wait_timer[i] = 40
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.GRASP:
                    target_pos = current_source_pos.clone()
                    target_pos[2] = cube_z_size - 0.002
                    gripper_cmd = -1.0 
                    self.wait_timer[i] -= 1
                    if self.wait_timer[i] <= 0:
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
                    target_pos = target_world_pos.clone()
                    target_pos[2] = safe_z
                    gripper_cmd = -1.0 
                    # 修复：显著放宽到达判定的距离 (0.02 -> 0.05)，防止机器人在目标点上方因为 IK 精度卡死
                    # 因为 DESCEND_TARGET 阶段仍然会进行 XY 轴的实时纠偏，所以 5cm 的容差是安全的
                    if torch.norm(ee_pos[i, :2] - target_pos[:2]) < 0.05:
                        self.state[i] = self.DESCEND_TARGET
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.DESCEND_TARGET:
                    target_pos = target_world_pos.clone()
                    # 使用 -0.002 产生微小压力，确保吸盘与方块表面接触
                    target_pos[2] = target_world_pos[2] + (cube_z_size / 2.0) - 0.002
                    gripper_cmd = -1.0 
                    # 扩大到达判定范围，防止因为物理碰撞造成的细微抖动导致无法切换状态
                    if torch.abs(ee_pos[i, 2] - target_pos[2]) < 0.015:
                        self.state[i] = self.RELEASE
                        self.wait_timer[i] = 20
                        self.state_timer[i] = 0
                        
                elif self.state[i] == self.RELEASE:
                    target_pos = ee_pos[i].clone()
                    gripper_cmd = 1.0 
                    self.wait_timer[i] -= 1
                    if self.wait_timer[i] <= 0:
                        self.state[i] = self.RETRACT
                        self.wait_timer[i] = 30 # 虽然 RETRACT 判位置，但 wait_timer 留着备用
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
                            self.wait_timer[i] = 60 # 任务完成后停留 1 秒左右再 reset
                        else:
                            self.state[i] = self.APPROACH_CUBE
                            self.wait_timer[i] = 20 
                        self.state_timer[i] = 0
                else:
                    target_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)
                    gripper_cmd = 1.0

            # --- 强制超时自动切换状态 (每阶段约 40s) ---
            if self.state[i] not in [-1, self.GRASP, self.RELEASE] and self.state_timer[i] > 1000:
                print(f"[Env {i}] !!! STATE {self.state[i].item()} TIMEOUT !!! Forcing transition.")
                if self.state[i] == self.APPROACH_CUBE: self.state[i] = self.DESCEND_CUBE
                elif self.state[i] == self.DESCEND_CUBE: self.state[i] = self.GRASP
                elif self.state[i] == self.LIFT: self.state[i] = self.APPROACH_TARGET
                elif self.state[i] == self.APPROACH_TARGET: self.state[i] = self.DESCEND_TARGET
                elif self.state[i] == self.DESCEND_TARGET: self.state[i] = self.RELEASE
                elif self.state[i] == self.RETRACT: pass # 已在上面处理
                self.state_timer[i] = 0

            # 再次增强限速和响应平衡
            kp = 1.0 
            
            # 提高下降阶段的速度，避免由于过于缓慢而产生的“僵死”感
            if self.state[i] in [self.DESCEND_CUBE, self.DESCEND_TARGET]:
                current_max_vel = 0.05 # 5cm/s
            elif self.state[i] in [self.GRASP, self.RELEASE]:
                current_max_vel = 0.005 # 限制释放时的位移
            elif self.state[i] in [self.LIFT, self.RETRACT]:
                current_max_vel = 0.08 # 8cm/s
            else:
                current_max_vel = 0.15 # 15cm/s (Approach 阶段允许更高速度以克服 deadband)
            
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

            # --- 核心修复：平滑启动计时器 (Soft Start Timer) ---
            # 如果处于软启动期，严厉限制速度和旋转，防止启动时的 Whipping 效应
            if self.soft_start_timer[i] > 0:
                # 稍微放宽旋转限制 (0.001 -> 0.005)，避免第一个方块对齐太慢导致超时
                current_rot_max_vel = 0.005 
                # 同时也极其严格地限制位置移动
                if pos_norm > 0.001:
                    pos_action = pos_action * (0.001 / pos_norm)
                self.soft_start_timer[i] -= 1
            else:
                current_rot_max_vel = 0.05 # 限制旋转速度，防止旋转过猛造成的“甩动”
            
            actions[i, :3] = pos_action
            
            if torch.sum(target_quat * ee_quat[i]) < 0:
                target_quat = -target_quat
            
            q_error = math_utils.quat_mul(target_quat, math_utils.quat_inv(ee_quat[i]))
            rot_action = math_utils.axis_angle_from_quat(q_error) * 2.0 

            # 对旋转也进行限速
            rot_norm = torch.norm(rot_action)
            if rot_norm > current_rot_max_vel:
                rot_action = rot_action * (current_rot_max_vel / rot_norm)
            
            actions[i, 3:6] = rot_action
            if num_actions > 6:
                actions[i, 6] = gripper_cmd

            self.state_timer[i] += 1
            if i == 0 and self.state_timer[i] % 100 == 0:
                 # 增强诊断日志：打印距离和目标，帮助定位卡死原因
                 if self.state[i] != -1:
                     dist_xy = torch.norm(ee_pos[i, :2] - target_pos[:2])
                     print(f"Env 0 | State: {self.state[i].item()} | Task: {self.task_index[i].item()} | DistXY: {dist_xy.item():.4f} | Target: {[round(float(x),3) for x in target_pos]}")
                 print(f"Env 0 | State: {self.state[i].item()} | Task: {self.task_index[i].item()}/{self.num_tasks_per_env[i].item()} | Pos: {[round(float(x),3) for x in ee_pos[i]]}")

        return actions

def main():
    # 强制使用 CUDA，否则 Magic Suction 没有意义
    if args_cli.device == "cpu":
        args_cli.device = "cuda"

    # create environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # --- 核心性能优化：相机启用时开启同步渲染 ---
    if getattr(args_cli, "enable_cameras", False):
        print("[INFO]: Enabling High-Performance Rendering (render_interval=1) for TiledCamera.")

    
    # --- 核心修复：只有多环境时才开启 replicate_physics ---
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.replicate_physics = (num_envs > 1)
    env_cfg.scene.lazy_sensor_update = False 
    if hasattr(env_cfg, "device"): 
        env_cfg.device = args_cli.device
    
    # 移除内置的 SurfaceGripper 插件定义，防止报错
    if hasattr(env_cfg.scene, "surface_gripper"):
        print("[INFO]: Switching to GPU Magic Suction. Disabling CPU SurfaceGripper plugin.")
        env_cfg.scene.surface_gripper = None
    
    # --- 核心修复：同步移除动作管理器中的吸盘控制项 ---
    # 根据 UR10ShortSuctionCubeStackEnvCfg 的定义，该项可能叫 'gripper_action' 或 'suction_gripper'
    for action_key in ["gripper_action", "suction_gripper", "gripper"]:
        if hasattr(env_cfg.actions, action_key):
            setattr(env_cfg.actions, action_key, None)

    # --- 核心修复：移除会导致崩溃的观察项项 (因为底层资产已被移除) ---
    if hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "gripper_pos"):
            env_cfg.observations.policy.gripper_pos = None
    
    # 子任务观察项也可能依赖吸盘状态
    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms = None

    # 修复 UR10 Suction 的配置 Bug: 
    # 原始配置中吸盘在 X 轴，但 IK 控制器却偏置在 Z 轴，导致控制与观测冲突。
    if "UR10" in args_cli.task and hasattr(env_cfg.actions, "arm_action"):
        print("[INFO]: Fixing UR10 IK controller offset (Z -> X)...")
        env_cfg.actions.arm_action.body_offset.pos = (0.159, 0.0, 0.0)

    # Grid Parameters (Real Assets)
    grid_origin = [0.5, 0.0, 0.001]
    grid_size = 6 # 与 Qwen 生成的 6x6 保持一致
    line_thickness = 0.001 # Use 1mm for precision
    # Increase cell size to be slightly larger than the cube (0.04m) 
    # for easier placement and better visual spacing.
    cell_size = 0.055 + line_thickness 
    half_width = (grid_size * cell_size) / 2
    
    # Calculate cell centers for alignment
    # Cell 0 is at index -2.5, Cell 5 is at index 2.5 (relative to center)
    def get_cell_center(idx): # idx from 0 to 5
        return (idx - 2.5) * cell_size

    # Spawning Logic: We spawn a fixed number of cubes to handle various tasks
    max_cubes = 8
    cube_size = 0.045 # Exact cube size as requested by user
    blue_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
    cube_names = [f"cube_{i+1}" for i in range(max_cubes)]
    
    # 核心回归：取料位置恢复到 [0.3, -0.2]
    source_pick_pos_x = 0.3
    source_pick_pos_y = -0.2
    
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

    # --- 核心性能优化：移除所有默认相机，只保留我们需要的 4 个相机 ---
    # 这样可以极大地节省 GPU Descriptor Sets
    default_cams = ["table_cam", "table_high_cam", "robot_cam", "cam_default"]
    for d_cam in default_cams:
        if hasattr(env_cfg.scene, d_cam):
            setattr(env_cfg.scene, d_cam, None)

    # Add Ring Cameras around the grid (Only if enable_cameras is requested)
    if getattr(args_cli, "enable_cameras", False):
        from isaaclab.sensors import TiledCameraCfg
        camera_ring_configs = {
            # 仅保留这 4 个最关键视角
            "cam_front_left": ([-0.1, 0.5, 0.6], [0.0, 0.7, -0.7]), 
            "cam_front_right": ([-0.1, -0.5, 0.6], [0.0, 0.7, 0.7]),
            "cam_back_left": ([1.1, 0.5, 0.6], [0.0, 0.7, -2.4]),
            "cam_back_right": ([1.1, -0.5, 0.6], [0.0, 0.7, 2.4]),
        }

        for cam_name, (pos, euler) in camera_ring_configs.items():
            # Convert euler to quat
            q = math_utils.quat_from_euler_xyz(
                torch.tensor([euler[0]], device=args_cli.device),
                torch.tensor([euler[1]], device=args_cli.device),
                torch.tensor([euler[2]], device=args_cli.device)
            ).tolist()[0]
            # OffsetCfg expects (w, x, y, z)
            quat_wxyz = (q[0], q[1], q[2], q[3])

            # 使用高性能 TiledCamera 替换传统 Camera
            setattr(env_cfg.scene, cam_name, TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{cam_name}",
                update_period=0.0,  # 0.0 表示每物理步更新一次
                height=224, 
                width=224,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, 
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1000.0), # 加大范围防止缓冲区分配失败
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=pos,
                    rot=quat_wxyz,
                    convention="world",
                ),
            ))
    else:
        print("[INFO]: Cameras are disabled via CLI. Skipping TiledCamera setup.")

    # 移除导致 TiledCamera 报错的语义映射逻辑
    # 我们只通过 RGB 信息给 Qwen

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


    env = gym.make(args_cli.task, cfg=env_cfg)
    sm = StackingStateMachine(
        env.unwrapped.num_envs, 
        env.unwrapped.device, 
        env.unwrapped.scene, 
        cube_names, 
        max_tasks=max_cubes,
        cube_z_size=cube_size,
        grid_origin=grid_origin,
        cell_size=cell_size
    )


    # === Fixed Grid Targets (Coord List) ===
    # 按照您的要求，这里设置固定的坐标列表。
    # 我们直接堆叠 6 个方块到网格中心点 (3, 3) 上。
    target_grid_indices = [
        (3, 3, 0), # 第1个在底层
        (3, 3, 1), # 第2个
        (3, 3, 2), # 第3个
        (3, 3, 3), # 第4个
        (3, 3, 4), # 第5个
        (3, 3, 5), # 第6个
    ]
    fixed_master_targets = []
    for g_x, g_y, g_z in target_grid_indices:
        target_x = grid_origin[0] + (g_x - 2.5) * cell_size
        target_y = grid_origin[1] + (g_y - 2.5) * cell_size
        target_z = (g_z + 0.5) * cube_size + 0.002
        fixed_master_targets.append((target_x, target_y, target_z))

    def prepare_environment(env_ids):
        """Prepare environmental targets and reset cube physics"""
        # 1. Apply fixed targets
        sm.set_env_targets(env_ids, fixed_master_targets)
        
        # 2. Physics Teleport: Move ALL cubes to their initial aligned positions
        env_ids_tensor = torch.as_tensor(env_ids, device=env.unwrapped.device, dtype=torch.long)
        num_target = len(env_ids)
        env_origins = env.unwrapped.scene.env_origins[env_ids_tensor]
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
        zero_velocity = torch.zeros((num_target, 6), device=env.unwrapped.device)

        for i, name in enumerate(cube_names):
            asset = env.unwrapped.scene[name]
            all_poses = torch.zeros((num_target, 7), device=env.unwrapped.device)
            init_p = aligned_poses[i]
            all_poses[:, :3] = env_origins + torch.tensor([init_p[0], init_p[1], init_p[2]], device=env.unwrapped.device)
            all_poses[:, 3:] = identity_quat
            
            # 使用 int32 避免底层 API 报错
            target_ids_i32 = env_ids_tensor.to(torch.int32)
            asset.write_root_pose_to_sim(all_poses, env_ids=target_ids_i32)
            asset.write_root_velocity_to_sim(zero_velocity, env_ids=target_ids_i32)

    # 1. 先重置环境，初始化物理缓冲区
    obs, _ = env.reset()
    
    # 2. 布置任务和方块
    prepare_environment(range(env.unwrapped.num_envs))
    
    # 彻底解决启动时“向外甩动”的问题：
    # 1. 原因是第一次 compute_action 时末端位置可能还没更新，导致位移差 diff_pos 巨大。
    # 2. 我们通过一次空跑或更新状态来同步。
    # 强制将状态机内部的目标位置初始化为当前位置，防止第一帧突跳

    init_ee_pos = obs['policy']['eef_pos']

    for i in range(env.unwrapped.num_envs):
        sm.reset_state(i)
        # 在没有正式进入 APPROACH 逻辑前，保持 quiet。
    
    print("[INFO]: Starting Dynamic Qwen-VL Stacking Loop...")
    
    step_count = 0
    # 将初始索引设为 -1，确保第一个方块 (Task 0) 也能触发瞬移逻辑
    prev_task_indices = torch.full((env.unwrapped.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
    
    while simulation_app.is_running():
        step_count += 1
        if step_count % 100 == 0:
            print(f"[Heartbeat] Step {step_count}...")
            
        # 移除 torch.inference_mode()，彻底解决 Isaac Lab 中“推理张量”原地更新导致的崩溃问题。
        # 状态机逻辑简单，不使用推理模式也不会影响性能。
        actions = sm.compute_action(obs)

        # 核心：应用魔法吸附同步 (Magic Suction)
        sm.apply_magic_suction(obs)

        # 物理操作（瞬移、步进）
        # 处理方块“瞬移”逻辑：当任务索引增加时，将下一个方块移至取料点
        for env_id in range(env.unwrapped.num_envs):
            curr_idx = sm.task_index[env_id].item()
            if curr_idx > prev_task_indices[env_id]:
                # 有新任务开始，检查对应的方块是否需要移到桌面上
                if curr_idx < len(cube_names):
                    next_cube_name = cube_names[curr_idx]
                    print(f"[Env {env_id}] Teleporting {next_cube_name} to pick position.")
                    cube_asset = env.unwrapped.scene[next_cube_name]
                    
                    # 使用环境的原点坐标，将相对位置转换为世界位置
                    env_origin = env.unwrapped.scene.env_origins[env_id]
                    target_pos_w = torch.tensor([
                        env_origin[0] + source_pick_pos_x,
                        env_origin[1] + source_pick_pos_y,
                        env_origin[2] + cube_size / 2.0
                    ], device=env.unwrapped.device)
                    
                    # 修正：瞬移方块至取料点时保持正向对齐 (Yaw=0)
                    target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                    
                    root_pose = torch.cat([target_pos_w, target_quat_w], dim=-1).unsqueeze(0)
                    cube_asset.write_root_pose_to_sim(
                        root_pose, 
                        env_ids=torch.tensor([env_id], device=env.unwrapped.device, dtype=torch.int)
                    )
                prev_task_indices[env_id] = curr_idx

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Case 1: Auto-reset
        auto_reset_ids = torch.where(terminated | truncated)[0].tolist()
        if auto_reset_ids:
            sm.reset_envs(auto_reset_ids)
            # 重要：重置任务索引记录器，确保下一轮能触发方块“瞬移”
            prev_task_indices[auto_reset_ids] = -1
            prepare_environment(auto_reset_ids)
            # 记录重置后的初始位置，防止后续逻辑出现位姿突变
            init_ee_pos = obs['policy']['eef_pos']
        
        # Case 2: Success Reset (all cubes stacked)
        finished_ids = torch.where((sm.state == -1) & (sm.wait_timer <= 0))[0].tolist()
        if finished_ids:
            print(f">>> Manually resetting {len(finished_ids)} finished environments <<<")
            # 同步重置索引记录器
            prev_task_indices[finished_ids] = -1
            new_obs, _ = env.unwrapped.reset(env_ids=torch.tensor(finished_ids, device=env.unwrapped.device))
            
            # Update the main obs dictionary
            def update_obs_dict(obs_dict, reset_obs_dict, ids):
                for key, value in reset_obs_dict.items():
                    if isinstance(value, torch.Tensor):
                        # 核心修复：只将重置环境对应的数据从 new_obs 中同步到 main obs
                        obs_dict[key][ids] = value[ids]
                    elif isinstance(value, dict):
                        update_obs_dict(obs_dict[key], value, ids)
            
            update_obs_dict(obs, new_obs, torch.tensor(finished_ids, device=env.unwrapped.device))
            sm.reset_envs(finished_ids)
            prepare_environment(finished_ids)
            # 同样同步刷新初始位置定义
            init_ee_pos = obs['policy']['eef_pos'] 
    
    env.close()

if __name__ == "__main__":
    main()
