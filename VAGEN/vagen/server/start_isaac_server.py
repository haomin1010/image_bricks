#!/usr/bin/env python3
"""
Standalone Isaac server that runs in main thread.
Creates a Ray actor interface for distributed access.
"""
import os
import sys
import ray
import logging
import time
from typing import Dict, Any, List
import asyncio


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
        self.step_done_events = {}  # env_id -> asyncio.Event
        self.step_condition = asyncio.Condition()
        self.step_done = {}  # env_id -> bool
        
        print(f"IsaacEnvServerProxy initialized with {num_envs} environment slots")
    
    def is_alive(self):
        return self.ready
    
    def allocate_env_id(self):
        """Allocate a unique environment slot ID from the pool."""
        if not self.free_ids:
            raise RuntimeError(f"All {self.num_envs} environment slots are allocated.")
        env_id = self.free_ids.pop(0)
        self.allocated_ids.add(env_id)
        print(f"Allocated environment slot {env_id} ({len(self.allocated_ids)}/{self.num_envs} used)")
        return env_id
    
    def release_env_id(self, env_id):
        """Release an environment slot ID back to the pool."""
        if env_id in self.allocated_ids:
            self.allocated_ids.remove(env_id)
            self.free_ids.append(env_id)
            self.free_ids.sort() # Keep IDs ordered
            if env_id in self.latest_images:
                del self.latest_images[env_id]
            print(f"Released environment slot {env_id}")

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
        print(f"Trainer requested reset for slot {env_id}")
        self.commands.append((env_id, "reset", seed))
        # Wait for the simulation loop to perform the reset and publish new images.
        # Poll for up to ~5 seconds; this provides a soft sync so trainers see the post-reset state.
        images = []
        for _ in range(50):
            images = self.latest_images.get(env_id, [])
            if images:
                break
            await asyncio.sleep(0.1)
        return {
            "images": images,
            "info": {
                "target_description": "Stack cubes in the target configuration",
                "env_id": env_id
            }
        }
    
    async def remote_step(self, env_id, goal):
        """Execute a step in a specific environment slot."""
        async with self.step_condition:
            self.commands.append((env_id, "step", goal))
            await self.step_condition.wait_for(lambda: self.step_done.get(env_id, False))
            ret = self.step_done[env_id]
            del self.step_done[env_id]
        images = self.latest_images.get(env_id, [])

        done_flag, new_task_av, new_task_idx = ret
        done_bool = bool(done_flag)

        info = {"success": done_bool, "env_id": env_id, "new_task_available": new_task_av, "new_task_index": new_task_idx}
        print(f"Proxy.remote_step returning for env={env_id} info={info}")
        return {
            "images": images,
            "reward": 1.0 if done_bool else 0.0,
            "done": done_bool,
            "obs_str": "Action executed",
            "info": info,
        }
    
    async def render(self, env_id):
        """Render current state of a specific environment slot."""
        return self.latest_images.get(env_id, [])
    
    async def _set_step_done(self, env_id, done, new_task_available=False, new_task_index=-1):
        """Mark a step done for a trainer waiting on env_id.

        This async method stores the payload and notifies the condition
        so any awaiting `remote_step` will wake and proceed.
        """
        payload = (bool(done), bool(new_task_available), int(new_task_index))
        self.step_done[env_id] = payload
        print(f"Proxy._set_step_done called env={env_id} payload={payload}")
        # Notify any waiters
        try:
            async with self.step_condition:
                self.step_condition.notify_all()
        except Exception as e:
            print(f"Warning: failed to notify step_condition: {e}")
    
    def get_observation(self, slot_id):
        return {"observation": None}


def main():
    """Main entry point - runs in main thread as required by Isaac Sim."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")
    # Allow enabling/disabling headless mode via CLI
    parser.add_argument("--headless", dest="headless", action="store_true", help="Run Isaac in headless mode (no GUI).")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run Isaac with GUI (disable headless).")
    parser.add_argument("--record", action="store_true", default=False, help="Record a video of env 0 and save to outputs/")
    args = parser.parse_args()
    
    # Connect to Ray cluster
    NAMESPACE = "vagen_training"
    ray.init(address="auto", namespace=NAMESPACE, ignore_reinit_error=True)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Isaac Sim sees GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Import Isaac AFTER Ray and ENV setup
    from isaaclab.app import AppLauncher
    import torch
    import numpy as np
    from PIL import Image
    import imageio
    _IMAGEIO_AVAILABLE = True
    
    config = {
        "num_envs": args.num_envs,
        "device": args.device,
        "task": args.task,
        "headless": args.headless,
        "enable_cameras": True,
        "cube_size": 0.045,
    }
    
    print(f"Starting Isaac server with config: {config}")
    
    # Launch Isaac app in main thread (REQUIRED)
    launcher_args = {
        "headless": config["headless"],
        "num_envs": config["num_envs"],
        "task": config["task"],
        "enable_cameras": config.get("enable_cameras", True),
    }
    
    print(f"Launching AppLauncher: {launcher_args}")
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    print("Isaac Simulation App launched successfully")
    
    # Import after app launch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    
    # Import env config from package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from vagen.server.server import get_stack_cube_env_cfg
    from vagen.server.server import StackingStateMachine
    
    # Create environment
    print("Creating Isaac Lab environment...")
    ret = get_stack_cube_env_cfg(
        task_name=config["task"],
        device=config["device"],
        num_envs=config["num_envs"],
        enable_cameras=config.get("enable_cameras", True)
    )

    if isinstance(ret, (tuple, list)) and len(ret) >= 3:
        env_cfg, cube_names, aligned_poses = ret[0], ret[1], ret[2]
    else:
        # Newer API returned a single env_cfg object. Build compatible
        # `cube_names` and `aligned_poses` from the env_cfg.scene where possible.
        env_cfg = ret
        cube_names = []
        aligned_poses = []
        scene = env_cfg.scene
        # Collect attributes named like cube_1, cube_2, ...
        candidates = [n for n in dir(scene) if n.startswith("cube_")]
        def _idx(n):
                return int(n.split("_")[1])
        candidates = sorted(set(candidates), key=_idx)
        if candidates:
            cube_names = candidates
        else:
            cube_names = [f"cube_{i+1}" for i in range(8)]
        for name in cube_names:
            cfg = getattr(scene, name)
            pos = getattr(cfg.init_state, "pos", None)
            rot = getattr(cfg.init_state, "rot", None)
    
    env = gym.make(config["task"], cfg=env_cfg)
    print(f"Environment '{config['task']}' created")
    print(f"Action space: {env.action_space}")
    grid_origin = [0.5, 0.0, 0.001]
    # Initialize state machine
    line_thickness = 0.001 # Use 1mm for precision
    # Increase cell size to be slightly larger than the cube (0.04m) 
    # for easier placement and better visual spacing.
    cell_size = 0.055 + line_thickness 
    sm = StackingStateMachine(
        env.unwrapped.num_envs, 
        env.unwrapped.device, 
        scene=env.unwrapped.scene,
        cube_names=cube_names,
        max_tasks=8,
        cube_z_size=config.get("cube_size", 0.045),
        grid_origin=grid_origin,
        cell_size=cell_size,
    )
    print("State machine initialized")
    
    step_initial_task_idx = {}  # env_id -> (initial task index, is_submit) for step completion check
    
    # Initial reset
    obs, _ = env.reset()
    print("Environment reset complete")
    
    # 彻底解决启动时“向外甩动”的问题：
    # 1. 原因是第一次 compute_action 时末端位置可能还没更新，导致位移差 diff_pos 巨大。
    # 2. 我们通过一次空跑或更新状态来同步。
    # 强制将状态机内部的目标位置初始化为当前位置，防止第一帧突跳
    init_ee_pos = obs['policy']['eef_pos']
    
    for i in range(env.unwrapped.num_envs):
        sm.reset_state(i)
        #sm.idle_pos[i] = init_ee_pos[i]  # Set IDLE position to initial EE position
    
    print("[INFO]: Starting Isaac server main loop...")

    proxy_actor = IsaacEnvServerProxy.options(
        name="IsaacEnvServer",
        lifetime="detached",
        get_if_exists=True
    ).remote(num_envs=config["num_envs"])
    
    print("IsaacEnvServer proxy registered to Ray cluster")
    ray.get(proxy_actor.is_alive.remote())  # Ensure it's ready
    
    # Camera names aligned to batch_gen.py and server.cfg
    cam_names = ["camera", "camera_front", "camera_side", "camera_iso", "camera_iso2"]

    # Keep simulation running
    print("Isaac server entering main loop (Ctrl+C to exit)...")
    try:
        frame_count = 0
        # 将初始索引设为 -1，确保第一个方块 (Task 0) 也能触发瞬移逻辑
        # 使用 Python 列表以便主循环中快速比较并日志化变化
        # Streaming writer (prefer) and also save PNG frames for accurate final encoding
        writer = None
        frames_dir = None
        frame_idx = 0
        frame_timestamps = []
        if args.record:
            out_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)

            ts_start = int(time.time())
            filename = f"isaac_record_{args.task.replace('/', '_')}_{ts_start}.mp4"
            out_path = os.path.join(out_dir, filename)
            if _IMAGEIO_AVAILABLE:
                writer = imageio.get_writer(out_path, fps=20, codec='libx264')
                print(f"Recording enabled - streaming to {out_path}")

            # Always create a frames dir so we can re-encode with measured FPS at shutdown
            frames_dir = os.path.join(out_dir, f"record_frames_{ts_start}")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Recording: saving frames to {frames_dir}")
            ts_capture_start = time.time()

        step_count = 0
        prev_task_indices = torch.full((env.unwrapped.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
    
        while simulation_app.is_running():
            # Update simulation
            simulation_app.update()
            
            # Check if any step commands have completed
            for env_id in list(step_initial_task_idx.keys()):
                task_idx_now = int(sm.task_index[env_id].item())
                sm_state_now = int(sm.state[env_id].item())

                init_val = step_initial_task_idx[env_id]
                if isinstance(init_val, tuple):
                    init_idx, was_submit = init_val
                else:
                    init_idx = int(init_val)
                    was_submit = False

                if (task_idx_now is not None and task_idx_now > init_idx) or sm_state_now == -1:
                    # Decide done flag: only return True for explicit submit actions
                    done_flag = True if was_submit else False
                    # Capture new-task snapshot BEFORE clearing it so trainer/LLM receives accurate info
                    num_tasks = int(sm.num_tasks_per_env[env_id].item())
                    new_av = bool(sm.new_task_available[env_id].item())
                    new_idx = int(sm.new_task_index[env_id].item())
                    proxy_actor._set_step_done.remote(env_id, done_flag, new_av, new_idx)
                    print(f"Marked step done for env {env_id} (task_idx: {task_idx_now} state: {sm_state_now}) done={done_flag}")
                    # Emit detailed debug snapshot for the trainer to see after success
                    print(
                        f"[Proxy->Trainer] env={env_id} STEP_DONE snapshot: task_index={task_idx_now} num_tasks={num_tasks} new_task_available={new_av} new_task_index={new_idx} state={sm_state_now}"
                    )

                    del step_initial_task_idx[env_id]
            
            # Check for commands from Proxy (Trainer)
            commands = ray.get(proxy_actor.get_pending_commands.remote())
            for env_id, cmd_type, data in commands:
                if cmd_type == "reset":
                    seed = data
                    # Try passing seed to env.reset; support common option keys
                    reset_options = {"env_ids": [env_id]}
                    reset_options["seed"] = int(seed)
                    env.reset(options=reset_options)

                    # Reset state machine for this environment
                    sm.reset_envs([env_id])
                    init_ee_pos = obs['policy']['eef_pos']

                elif cmd_type == "step":
                    goal = data
                    # Debug: log receipt of remote step goal immediately
                    print(f"[Proxy] Received step goal for env {env_id}: {goal}")
                    # Support explicit submission commands first (mirror actor behavior)
                    is_submit = isinstance(goal, dict) and goal.get("type") == "submit"
                    # Store initial task index and whether this was a submit
                    step_initial_task_idx[env_id] = (sm.task_index[env_id].item(), is_submit)
                    if is_submit:
                        task_idx = int(sm.task_index[env_id].item())
                        num_tasks = int(sm.num_tasks_per_env[env_id].item())
                        is_success = (task_idx >= num_tasks)
                        print(
                            f"[Proxy] Submission for env {env_id}: task_idx={task_idx} num_tasks={num_tasks} success={is_success}"
                        )
                        continue

                    # Strict parsing for placement goals: require dict with explicit 'x','y','z' keys
                    if not isinstance(goal, dict):
                        raise ValueError(f"Invalid goal type: {type(goal)}")
                    if not all(k in goal for k in ("x", "y", "z")):
                        raise KeyError("Goal must contain keys 'x','y','z'")

                    g_x = goal["x"]
                    g_y = goal["y"]
                    g_z = goal["z"]
                    # ensure numeric
                    g_x = float(g_x)
                    g_y = float(g_y)
                    g_z = float(g_z)

                    # Convert grid coords -> world coords using SM parameters
                    grid_origin = sm.grid_origin
                    cell_size = sm.cell_size
                    cube_size = sm.cube_z_size
                    env_origin = env.unwrapped.scene.env_origins[env_id]
                    target_x = grid_origin[0].item() + (g_x - 2.5) * cell_size
                    target_y = grid_origin[1].item() + (g_y - 2.5) * cell_size
                    target_z = (g_z + 0.5) * cube_size + 0.002

                    # Apply environment origin offset for multi-env support
                    target_pos_w = env_origin + torch.tensor([target_x, target_y, target_z], device=env_origin.device)

                    current_task_idx = int(sm.task_index[env_id].item())
                    sm.num_tasks_per_env[env_id] = current_task_idx + 1
                    sm.target_positions[env_id, current_task_idx] = target_pos_w
                    sm.state[env_id] = sm.APPROACH_CUBE
                    sm.state_timer[env_id] = 0

                    print(
                        f"[Proxy] Parsed goal for env {env_id}: grid=({g_x},{g_y},{g_z}) "
                        f"-> world=({target_x:.4f},{target_y:.4f},{target_z:.4f}) task_idx={current_task_idx}"
                    )
                    # Debug: print env origin and full world target for verification
                    try:
                        print(f"[Proxy] env_origin[{env_id}] = {env_origin.cpu().numpy() if hasattr(env_origin,'cpu') else env_origin} target_pos_w = {target_pos_w}")
                    except Exception:
                        print(f"[Proxy] env_origin[{env_id}] = {env_origin} target_pos_w = {target_pos_w}")
                    # Mark that a new task was delivered so teleport logic and trainers can react.
                    try:
                        sm.new_task_available[env_id] = True
                        sm.new_task_index[env_id] = current_task_idx
                        # read back to confirm
                        try:
                            rb = bool(sm.new_task_available[env_id].item())
                        except Exception:
                            rb = sm.new_task_available[env_id]
                        print(f"[Proxy] Set sm.new_task_available for env {env_id} -> index={current_task_idx} (readback={rb})")
                    except Exception as e:
                        print(f"[Proxy] Warning: failed to set new_task flags for env {env_id}: {e}")
            
            # --- IMAGE CAPTURE (only when needed) ---
            # Capture camera images only when recording is enabled or when the
            # proxy/trainer has pending commands requesting env images. This
            # avoids expensive per-frame readbacks for all envs/cameras.
            should_capture = args.record or (commands is not None and len(commands) > 0)
            if should_capture:
                # Build set of env_ids explicitly requested by proxy commands
                requested_envs = set()
                try:
                    requested_envs = {int(c[0]) for c in commands}
                except Exception:
                    requested_envs = set()

                for env_id in range(config["num_envs"]):
                    # If recording, always capture env 0 for movie; otherwise only
                    # capture envs explicitly requested by the proxy/trainer.
                    if not args.record and env_id not in requested_envs:
                        continue

                    img_list = []
                    for cam_name in cam_names:
                        cam = env.unwrapped.scene[cam_name]
                        rgb_tensor = cam.data.output['rgb'][env_id]
                        rgb_np = rgb_tensor.cpu().numpy().astype('uint8')
                        img_list.append(Image.fromarray(rgb_np))

                    proxy_actor.update_state.remote(env_id, img_list)

                    # If recording enabled, capture env 0 frames by concatenating cameras
                    if args.record and env_id == 0:
                        widths, heights = zip(*(i.size for i in img_list))
                        total_w = sum(widths)
                        max_h = max(heights)
                        concat = Image.new('RGB', (total_w, max_h))
                        x_off = 0
                        for im in img_list:
                            concat.paste(im, (x_off, 0))
                            x_off += im.size[0]

                        if frames_dir is not None:
                            frame_file = os.path.join(frames_dir, f"frame_{frame_idx:08d}.png")
                            concat.save(frame_file)
                            frame_timestamps.append(time.time())
                            frame_idx += 1
                            writer.append_data(np.array(concat))

            frame_count += 1
            # Drive state machine so proxy delivers actions when targets are set
            actions = sm.compute_action(obs)

            # Teleport-on-new-task: when task_index increases, move the next cube
            # to the pick position so the SM can grab it. This mirrors the
            # canonical implementation's behavior to ensure cubes are available.
            for env_id in range(config["num_envs"]):
                # If a new task was delivered, teleport the corresponding cube
                if getattr(sm, "new_task_available", None) is not None and sm.new_task_available[env_id].item():
                    new_idx = int(sm.new_task_index[env_id].item())
                    print(f"[Env {env_id}] Detected new_task_available=True new_idx={new_idx}")
                    if new_idx >= 0 and new_idx < len(cube_names):
                        next_cube_name = cube_names[new_idx]
                        try:
                            cube_asset = env.unwrapped.scene[next_cube_name]
                        except Exception as e:
                            print(f"[Env {env_id}] Warning: cube asset '{next_cube_name}' not found: {e}")
                            cube_asset = None

                        if cube_asset is not None:
                            # check current cube world z; skip if already on table
                            cube_pos_w = cube_asset.data.root_pos_w[env_id]
                            cube_z = float(cube_pos_w[2].item()) if isinstance(cube_pos_w[2], torch.Tensor) else float(cube_pos_w[2])
                            print(f"[Env {env_id}] cube '{next_cube_name}' z={cube_z:.4f}")
                            # For reliability during testing, force teleport the cube to the pick
                            # position whenever a new task arrives. This avoids edge cases where
                            # the cube is slightly off the expected location and the SM stalls.
                            print(f"[Env {env_id}] Forcing teleport of {next_cube_name} to pick position (test-mode)")
                            env_origin = env.unwrapped.scene.env_origins[env_id]
                            pick_offset = torch.tensor([
                                float(sm.source_pick_pos[0]),
                                float(sm.source_pick_pos[1]),
                                float(config.get("cube_size", 0.045)) / 2.0
                            ], device=env_origin.device)
                            target_pos_w = env_origin + pick_offset
                            # ensure float32 and same device
                            target_pos_w = target_pos_w.to(dtype=torch.float32, device=env_origin.device)
                            target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_origin.device, dtype=torch.float32)
                            root_pose = torch.cat([target_pos_w, target_quat_w], dim=-1).unsqueeze(0)

                            # Debug: print pre-write details (device, dtype, values)
                            try:
                                print(f"[Env {env_id}] TELEPORT PREWRITE env_origin={env_origin} pick_offset={pick_offset} target_pos_w={target_pos_w} target_quat_w={target_quat_w} root_pose={root_pose}")
                                print(f"[Env {env_id}] cube current pos (pre) = {cube_asset.data.root_pos_w[env_id]}")
                            except Exception:
                                pass

                            # Write pose
                            cube_asset.write_root_pose_to_sim(
                                root_pose,
                                env_ids=torch.tensor([env_id], device=env_origin.device, dtype=torch.int32)
                            )

                            # Read back and print to verify
                            try:
                                new_pos = cube_asset.data.root_pos_w[env_id]
                                print(f"[Env {env_id}] TELEPORT POSTWRITE cube pos (post) = {new_pos}")
                            except Exception:
                                print(f"[Env {env_id}] TELEPORT POSTWRITE: unable to read back cube pos")
                # clear new task flag
                sm.new_task_available[env_id] = False
                sm.new_task_index[env_id] = -1
            obs, _, _, _, _ = env.step(actions)

    except KeyboardInterrupt:
        print("Received shutdown signal")
    finally:
        print("Shutting down Isaac server...")
        # Close streaming writer if opened
        if writer is not None:
            writer.close()
            print(f"Saved streaming recording to {out_path}")

        # If we saved PNG frames, re-encode them into a single MP4 using measured FPS
        if frames_dir is not None and _IMAGEIO_AVAILABLE and frame_idx > 0:
            ts_end = time.time()
            elapsed = max(1e-3, ts_end - ts_capture_start)
            measured_fps = float(frame_idx) / elapsed
            final_filename = f"isaac_record_{args.task.replace('/', '_')}_{int(ts_capture_start)}.mp4"
            final_out_path = os.path.join(out_dir, final_filename)
            print(f"Encoding final MP4 at measured FPS={measured_fps:.2f} to {final_out_path}")
            writer2 = imageio.get_writer(final_out_path, fps=measured_fps, codec='libx264')
            for i in range(frame_idx):
                frame_file = os.path.join(frames_dir, f"frame_{i:08d}.png")
                img = imageio.imread(frame_file)
                writer2.append_data(img)
            writer2.close()
            print(f"Saved recording to {final_out_path}")
        elif frames_dir is not None and frame_idx > 0:
            print(f"Frames saved to {frames_dir}. Install imageio[ffmpeg] to auto-encode MP4.")

        env.close()
        simulation_app.close()
        print("Isaac server shutdown complete")


if __name__ == "__main__":
    main()

