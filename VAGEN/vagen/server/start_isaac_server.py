#!/usr/bin/env python3
"""
Standalone Isaac server that runs in main thread.
Creates a Ray actor interface for distributed access.
"""
import os
import sys

# --- GPU Isolation ---
# This MUST happen before any imports (even ray) to prevent Isaac skipping devices
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

import ray
import logging
import time
from typing import Dict, Any, List
import asyncio

# --- Configure Logging to File ---
LOG_FILE = os.path.join(os.getcwd(), "isaac_server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Isaac Server Script Starting. CUDA_VISIBLE_DEVICES={cvd}")

@ray.remote(num_cpus=0.1, num_gpus=0)
class IsaacEnvServerProxy:
    """Lightweight proxy that forwards calls to main thread via asynchronous buffers."""
    def __init__(self, num_envs):
        self.ready = True
        self.num_envs = num_envs
        # ID pool to ensure reuse of slots
        self.free_ids = list(range(num_envs))
        self.allocated_ids = set()
        
        # Buffers for state sharing
        self.latest_images = {} # env_id -> PIL Images list
        self.commands = [] # List of (env_id, type, data)
        
        logger.info(f"IsaacEnvServerProxy initialized with {num_envs} environment slots")
    
    def is_alive(self):
        return self.ready
    
    def allocate_env_id(self):
        """Allocate a unique environment slot ID from the pool."""
        if not self.free_ids:
            raise RuntimeError(f"All {self.num_envs} environment slots are allocated.")
        env_id = self.free_ids.pop(0)
        self.allocated_ids.add(env_id)
        logger.info(f"Allocated environment slot {env_id} ({len(self.allocated_ids)}/{self.num_envs} used)")
        return env_id
    
    def release_env_id(self, env_id):
        """Release an environment slot ID back to the pool."""
        if env_id in self.allocated_ids:
            self.allocated_ids.remove(env_id)
            self.free_ids.append(env_id)
            self.free_ids.sort() # Keep IDs ordered
            if env_id in self.latest_images:
                del self.latest_images[env_id]
            logger.info(f"Released environment slot {env_id}")

    # --- Methods for Main Thread to push/pull data ---
    def update_state(self, env_id, images):
        self.latest_images[env_id] = images

    def get_pending_commands(self):
        if not self.commands:
            return []
        cmds = self.commands
        self.commands = []
        return cmds

    # --- Methods for Trainer (Gym Remote Env) ---
    async def remote_reset(self, env_id, seed):
        """Reset a specific environment slot."""
        logger.info(f"Trainer requested reset for slot {env_id}")
        self.commands.append((env_id, "reset", seed))
        # Wait slightly for the simulation loop to catch up (no strict sync)
        await asyncio.sleep(0.1)
        images = self.latest_images.get(env_id, [])
        return {
            "images": images,
            "info": {
                "target_description": "Stack cubes in the target configuration",
                "env_id": env_id
            }
        }
    
    async def remote_step(self, env_id, goal):
        """Execute a step in a specific environment slot."""
        # Note: In this Isaac config, we might not use 'goal' directly here
        self.commands.append((env_id, "step", goal))
        await asyncio.sleep(0.05)
        images = self.latest_images.get(env_id, [])
        return {
            "images": images,
            "reward": 0.0,
            "done": False,
            "obs_str": "Action executed",
            "info": {"success": False, "env_id": env_id}
        }
    
    async def render(self, env_id):
        """Render current state of a specific environment slot."""
        return self.latest_images.get(env_id, [])
    
    def reset_slot(self, slot_id):
        return {"observation": None, "info": {}}
    
    def step_slot(self, slot_id, action):
        return {"observation": None, "reward": 0.0, "done": False, "info": {}}
    
    def get_observation(self, slot_id):
        return {"observation": None}


def main():
    """Main entry point - runs in main thread as required by Isaac Sim."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()
    
    # Connect to Ray cluster
    NAMESPACE = "vagen_training"
    ray.init(address="auto", namespace=NAMESPACE, ignore_reinit_error=True)
    
    # --- GPU Isolation Fix ---
    # We DO NOT delete CUDA_VISIBLE_DEVICES. 
    # Instead, we ensure Isaac only sees the specific device assigned to it.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Default to GPU 0 if not specified
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"Isaac Sim sees GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Import Isaac AFTER Ray and ENV setup
    from isaaclab.app import AppLauncher
    import torch
    import numpy as np
    from PIL import Image
    
    config = {
        "num_envs": args.num_envs,
        "device": args.device,
        "task": args.task,
        "headless": args.headless,
        "enable_cameras": True,
        "cube_size": 0.045,
    }
    
    logger.info(f"Starting Isaac server with config: {config}")
    
    # Launch Isaac app in main thread (REQUIRED)
    launcher_args = {
        "headless": config["headless"],
        "num_envs": config["num_envs"],
        "task": config["task"],
        "enable_cameras": config.get("enable_cameras", True),
    }
    
    logger.info(f"Launching AppLauncher: {launcher_args}")
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    logger.info("Isaac Simulation App launched successfully")
    
    # Import after app launch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    
    # Import env config from package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from vagen.server.env_cfg import get_stack_cube_env_cfg
    from vagen.server.sm_logic import StackingStateMachine
    
    # Create environment
    logger.info("Creating Isaac Lab environment...")
    env_cfg, cube_names, aligned_poses = get_stack_cube_env_cfg(
        task_name=config["task"],
        device=config["device"],
        num_envs=config["num_envs"],
        enable_cameras=config.get("enable_cameras", True)
    )
    
    env = gym.make(config["task"], cfg=env_cfg)
    logger.info(f"Environment '{config['task']}' created")
    
    # Initialize state machine
    sm = StackingStateMachine(
        num_envs=config["num_envs"],
        device=config["device"],
        scene=env.unwrapped.scene,
        cube_names=cube_names,
        cube_z_size=config.get("cube_size", 0.045)
    )
    logger.info("State machine initialized")
    
    # Initial reset
    obs, _ = env.reset()
    logger.info("Environment reset complete")
    
    # Register actor with name (use default namespace to match main_ppo)
    proxy_actor = IsaacEnvServerProxy.options(
        name="IsaacEnvServer",
        lifetime="detached",
        get_if_exists=True
    ).remote(num_envs=config["num_envs"])
    
    logger.info("IsaacEnvServer proxy registered to Ray cluster")
    ray.get(proxy_actor.is_alive.remote())  # Ensure it's ready
    
    # Updated to 3 cameras as requested
    cam_names = ["cam_front_left", "cam_front_right", "cam_back_left"]

    # Keep simulation running
    logger.info("Isaac server entering main loop (Ctrl+C to exit)...")
    try:
        frame_count = 0
        while True:
            # Update simulation
            simulation_app.update()
            
            # Check for commands from Proxy (Trainer)
            commands = ray.get(proxy_actor.get_pending_commands.remote())
            for env_id, cmd_type, data in commands:
                if cmd_type == "reset":
                    env.reset(options={"env_ids": [env_id]})
                    logger.info(f"Executed reset for env_id {env_id}")
            
            # --- OPTIMIZED IMAGE CAPTURE ---
            # Instead of capturing 64 envs every single frame (huge overhead),
            # we only capture when there are trainers active OR every X frames for specific envs.
            # Here we simplify: capture all every 5 frames, but env 0 every frame for logging.
            should_capture_all = (frame_count % 5 == 0)
            
            for env_id in range(config["num_envs"]):
                if not should_capture_all and env_id != 0:
                    continue
                    
                img_list = []
                for cam_name in cam_names:
                    try:
                        cam = env.unwrapped.scene[cam_name]
                        rgb_tensor = cam.data.output['rgb'][env_id]
                        rgb_np = rgb_tensor.cpu().numpy().astype('uint8')
                        
                        if env_id == 0 and frame_count % 100 == 0:
                            logger.info(f"[ImageCheck] Env {env_id} {cam_name}: "
                                        f"min={rgb_np.min()}, max={rgb_np.max()}, mean={rgb_np.mean():.1f}")
                        
                        img_list.append(Image.fromarray(rgb_np))
                    except:
                        img_list.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                
                proxy_actor.update_state.remote(env_id, img_list)

            frame_count += 1
            if frame_count % 600 == 0:  # Log every 10s
                logger.info(f"Isaac server running... (frame {frame_count})")
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("Shutting down Isaac server...")
        env.close()
        simulation_app.close()
        logger.info("Isaac server shutdown complete")


if __name__ == "__main__":
    main()

