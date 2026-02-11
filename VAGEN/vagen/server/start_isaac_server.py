#!/usr/bin/env python3
"""
Standalone Isaac server that runs in main thread.
Creates a Ray actor interface for distributed access.
"""
import os
import sys
import ray
import torch
import logging
import time
from typing import Dict, Any
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point - runs in main thread as required by Isaac Sim."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()
    
    # Connect to Ray cluster (use fixed namespace to match main_ppo)
    NAMESPACE = "vagen_training"
    logger.info(f"Connecting to Ray cluster with namespace '{NAMESPACE}'...")
    ray.init(address="auto", namespace=NAMESPACE, ignore_reinit_error=True)
    logger.info(f"Connected to Ray at {ray.get_runtime_context().gcs_address}")
    logger.info(f"Using namespace: '{ray.get_runtime_context().namespace}'")
    
    # Import Isaac AFTER Ray init
    from isaaclab.app import AppLauncher
    
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
    
    # Create lightweight Ray actor as interface
    @ray.remote(num_cpus=0.1, num_gpus=0)
    class IsaacEnvServerProxy:
        """Lightweight proxy that forwards calls to main thread via queue/shared mem."""
        def __init__(self, num_envs):
            self.ready = True
            self.num_envs = num_envs
            self.next_env_id = 0
            self.allocated_ids = set()
            logger.info(f"IsaacEnvServerProxy initialized with {num_envs} environment slots")
        
        def is_alive(self):
            return self.ready
        
        def allocate_env_id(self):
            """Allocate a unique environment slot ID."""
            if self.next_env_id >= self.num_envs:
                raise RuntimeError(
                    f"All {self.num_envs} environment slots are allocated. "
                    "Increase num_envs or release unused slots."
                )
            env_id = self.next_env_id
            self.next_env_id += 1
            self.allocated_ids.add(env_id)
            logger.info(f"Allocated environment slot {env_id} ({len(self.allocated_ids)}/{self.num_envs} used)")
            return env_id
        
        def release_env_id(self, env_id):
            """Release an environment slot ID."""
            if env_id in self.allocated_ids:
                self.allocated_ids.remove(env_id)
                logger.info(f"Released environment slot {env_id}")
        
        async def remote_reset(self, env_id, seed):
            """Reset a specific environment slot."""
            # TODO: Implement actual reset via IPC to main thread
            # For now, return placeholder with 3 camera views
            logger.info(f"Reset environment slot {env_id} with seed {seed}")
            # Return 3 views as expected by config (top/front/side cameras)
            # Use PIL.Image as expected by vLLM/sglang
            # Use smaller resolution (224x224) to save memory and tokens
            dummy_images = [
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
            ]
            return {
                "images": dummy_images,
                "info": {
                    "target_description": "Stack cubes in the target configuration",
                    "env_id": env_id
                }
            }
        
        async def remote_step(self, env_id, goal):
            """Execute a step in a specific environment slot."""
            # TODO: Implement actual step via IPC to main thread
            logger.info(f"Step in environment slot {env_id} with goal {goal}")
            # Return 3 views as expected by config
            dummy_images = [
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
            ]
            return {
                "images": dummy_images,
                "reward": 0.0,
                "done": False,
                "obs_str": "Action executed",
                "info": {"success": False, "env_id": env_id}
            }
        
        async def render(self, env_id):
            """Render current state of a specific environment slot."""
            # TODO: Implement actual rendering via IPC to main thread
            logger.info(f"Render environment slot {env_id}")
            # Return 3 views as expected by config
            return [
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
                Image.new("RGB", (224, 224), (0, 0, 0)),
            ]
        
        def reset_slot(self, slot_id):
            # Legacy method - kept for compatibility
            return {"observation": None, "info": {}}
        
        def step_slot(self, slot_id, action):
            return {"observation": None, "reward": 0.0, "done": False, "info": {}}
        
        def get_observation(self, slot_id):
            return {"observation": None}
    
    # Register actor with name (use default namespace to match main_ppo)
    proxy_actor = IsaacEnvServerProxy.options(
        name="IsaacEnvServer",
        lifetime="detached",
        max_restarts=0
    ).remote(num_envs=config["num_envs"])
    
    logger.info("IsaacEnvServer proxy registered to Ray cluster")
    ray.get(proxy_actor.is_alive.remote())  # Ensure it's ready
    
    # Keep simulation running
    logger.info("Isaac server entering main loop (Ctrl+C to exit)...")
    try:
        frame_count = 0
        while True:
            # Update simulation at ~60 FPS
            simulation_app.update()
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

