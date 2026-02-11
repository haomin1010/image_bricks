import os
import torch
import ray
import asyncio
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# Isaac Lab and Gym imports will be deferred inside the Actor to ensure 
# they are only loaded in the process that has the simulation_app.

@ray.remote(num_gpus=1) # Customize this based on project needs
class IsaacEnvServer:
    """
    Ray Actor that hosts the Isaac Lab simulation environment.
    It provides a multi-slot interface for concurrent VAGEN workers.
    """
    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config
        self.num_envs = env_config.get("num_envs", 64)
        self.device = env_config.get("device", "cuda:0")
        
        # Defer AppLauncher until init to ensure proper CLI arg handling if needed
        from isaaclab.app import AppLauncher
        
        # Setup AppLauncher args
        # We can pass options like headless here
        launcher_args = {
            "headless": env_config.get("headless", True),
            "num_envs": self.num_envs,
            "task": env_config.get("task", "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0"),
        }
        
        # Note: AppLauncher is a singleton in Isaac Lab. 
        # In a Ray Actor, it's safe as this is a dedicated process.
        self.app_launcher = AppLauncher(launcher_args)
        self.simulation_app = self.app_launcher.app
        
        # Now import the rest after app is launched
        import gymnasium as gym
        import isaaclab_tasks # noqa: F401
        from .env_cfg import get_stack_cube_env_cfg
        
        # 1. Get custom environment config
        env_cfg, cube_names, self.aligned_poses = get_stack_cube_env_cfg(
            task_name=self.config.get("task", "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0"),
            device=self.device,
            num_envs=self.num_envs,
            enable_cameras=self.config.get("enable_cameras", True)
        )
        
        # 2. Create environment
        self.env = gym.make(self.config["task"], cfg=env_cfg)
        
        # 3. Initialize the State Machine
        from .sm_logic import StackingStateMachine 
        self.sm = StackingStateMachine(
            num_envs=self.num_envs,
            device=self.device,
            scene=self.env.unwrapped.scene,
            cube_names=cube_names,
            cube_z_size=self.config.get("cube_size", 0.045)
        )
        
        # Slot management
        self.available_slots = list(range(self.num_envs))
        self.slot_busy = [False] * self.num_envs
        self.obs = None # Current global observation buffer
        
        # Initial reset
        self.obs, _ = self.env.reset()
        logging.info(f"IsaacEnvServer initialized with {self.num_envs} envs.")

    async def allocate_env_id(self) -> Optional[int]:
        if not self.available_slots:
            return None
        slot = self.available_slots.pop(0)
        self.slot_busy[slot] = True
        return slot

    async def release_env_id(self, slot_id: int):
        if 0 <= slot_id < self.num_envs:
            # 1. Mark as available
            self.slot_busy[slot_id] = False
            self.available_slots.append(slot_id)
            
            # 2. Reset Logic State to IDLE to prevent "zombie" actions
            # If we don't do this, the robot might keep moving if other workers step the sim.
            self.sm.state[slot_id] = -1  # IDLE
            self.sm.task_index[slot_id] = 0
            self.sm.num_tasks_per_env[slot_id] = 0 
            self.sm.attached_cube_idx[slot_id] = -1

    async def remote_reset(self, slot_id: int, seed: int = 42) -> Dict[str, Any]:
        """Reset a specific slot."""
        env_ids = torch.tensor([slot_id], device=self.device, dtype=torch.long)
        
        # 1. Reset SM state
        self.sm.reset_envs(env_ids)
        
        # 2. Reset Physics State (Teleport cubes)
        from .env_cfg import get_stack_cube_env_cfg 
        _, _, aligned_poses = get_stack_cube_env_cfg(None, self.device, self.num_envs)
        
        env_origin = self.env.unwrapped.scene.env_origins[slot_id]
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        zero_vel = torch.zeros((1, 6), device=self.device)
        
        for i, name in enumerate(self.sm.cube_names):
            asset = self.env.unwrapped.scene[name]
            pos = aligned_poses[i]
            target_pos_w = env_origin + torch.tensor([pos[0], pos[1], pos[2]], device=self.device)
            root_pose = torch.cat([target_pos_w, identity_quat], dim=-1).unsqueeze(0)
            asset.write_root_pose_to_sim(root_pose, env_ids=torch.tensor([slot_id], device=self.device, dtype=torch.int32))
            asset.write_root_velocity_to_sim(zero_vel, env_ids=torch.tensor([slot_id], device=self.device, dtype=torch.int32))

        # 3. Perform steps to settle (Correctly driving all environments)
        # We must use the state machine to generate actions for ALL envs, 
        # otherwise other running envs will freeze/drop objects.
        for _ in range(10):
            actions = self.sm.compute_action(self.obs)
            self.sm.apply_magic_suction(self.obs)
            self.obs, _, _, _, _ = self.env.step(actions)

        return self._get_slot_obs(slot_id)
        """Return current images for a slot."""
        obs = self._get_slot_obs(slot_id)
        return obs["images"]

    async def remote_step(self, slot_id: int, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a step based on a structured goal."""
        # Handle manual submission
        if goal.get("type") == "submit":
            return self._check_submission(slot_id)

        initial_task_idx = self.sm.task_index[slot_id].item()
        
        # 只有在空闲时才接受新目标
        if self.sm.state[slot_id] == -1 or (self.sm.state[slot_id] == 0 and self.sm.state_timer[slot_id] == 0):
            self._parse_and_set_goal(slot_id, goal)
        
        max_substeps = self.config.get("max_substeps", 1500) 
        substeps = 0
        while substeps < max_substeps:
            # 检查任务是否已经完成（可能由其它请求驱动了仿真）
            if self.sm.task_index[slot_id].item() > initial_task_idx or self.sm.state[slot_id] == -1:
                break

            # 驱动仿真步进
            actions = self.sm.compute_action(self.obs)
            self.sm.apply_magic_suction(self.obs)
            self.obs, _, _, _, _ = self.env.step(actions)
            substeps += 1
            
            # 关键：让出执行权，防止多请求堵塞 Actor
            await asyncio.sleep(0)
            
        return self._get_slot_obs(slot_id)

    def _check_submission(self, slot_id: int) -> Dict[str, Any]:
        """Evaluate success on explicit submission."""
        # Force a rendering updated
        obs = self._get_slot_obs(slot_id)
        
        # Determine if success
        # Success definition: All cubes stacked (state == -1 implies task list exhausted)
        # However, user might submit early.
        # We need to verify if task_index == num_tasks_per_env
        
        # Note: task_index increments after each successful placement.
        # So if task_index == num_tasks, we are good.
        
        task_idx = self.sm.task_index[slot_id].item()
        num_tasks = self.sm.num_tasks_per_env[slot_id].item()
        
        # Also check strict success
        is_success = (task_idx >= num_tasks)
        
        obs["done"] = True
        obs["reward"] = 1.0 if is_success else 0.0
        obs["info"]["success"] = is_success
        obs["obs_str"] = "Submitted. " + ("Success!" if is_success else "Incomplete or incorrect.")
        
        return obs

    def _get_slot_obs(self, slot_id: int) -> Dict[str, Any]:
        """Format the raw Isaac observation into VAGEN format."""
        images = []
        # TiledCamera 数据的标准获取路径
        for cam_name in ["cam_front_left", "cam_front_right", "cam_back_left", "cam_back_right"]:
            cam_sensor = self.env.unwrapped.scene[cam_name]
            img_tensor = cam_sensor.data.output["rgb"][slot_id].clone()
            img_np = img_tensor.cpu().numpy().astype(np.uint8)
            images.append(Image.fromarray(img_np))
        
        state = self.sm.state[slot_id].item()
        task_idx = self.sm.task_index[slot_id].item()
        obs_str = f"Current stack height: {task_idx}. "
        if state == -1:
            obs_str += "Mission completed! All cubes are stacked. "
        else:
            obs_str += f"Robot is moving cube {task_idx + 1}. "
        
        info = {
            "success": bool(state == -1), 
            "progress": task_idx / self.sm.max_tasks,
            "correct": True,
            "target_description": f"Please stack cube {task_idx + 1} at a valid position."
        }
        return {"images": images, "obs_str": obs_str, "reward": 1.0 if state == -1 else 0.0, "done": state == -1, "info": info}



    def _parse_and_set_goal(self, slot_id: int, goal: Dict[str, Any]):
        """Convert goal dict {x, y, z} to SM target."""
        # Note: SM uses grid logic or absolute? In stack_cube_sm.py it used grid (gx, gy, gz)
        # But in our EnvServer we can calculate world coords.
        g_x, g_y, g_z = goal.get("x", 3), goal.get("y", 3), goal.get("z", 0)
        
        # Recalculate cell centers matching env_cfg.py logic
        grid_origin = self.sm.grid_origin
        cell_size = self.sm.cell_size
        cube_size = self.sm.cube_z_size
        
        target_x = grid_origin[0].item() + (g_x - 2.5) * cell_size
        target_y = grid_origin[1].item() + (g_y - 2.5) * cell_size
        target_z = (g_z + 0.5) * cube_size + 0.002
        
        # Currently we just set one target for the current task_index
        # IMPORTANT: Increase num_tasks_per_env by 1 to allow the SM to proceed to this new task
        current_task_idx = self.sm.task_index[slot_id].item()
        
        self.sm.num_tasks_per_env[slot_id] = current_task_idx + 1
        
        self.sm.target_positions[slot_id, current_task_idx] = torch.tensor(
            [target_x, target_y, target_z], device=self.device
        )
        self.sm.state[slot_id] = self.sm.APPROACH_CUBE
        self.sm.state_timer[slot_id] = 0

