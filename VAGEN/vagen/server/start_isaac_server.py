#!/usr/bin/env python3
"""
Standalone Isaac server that runs in main thread.
Creates a Ray actor interface for distributed access.
"""
import os
import sys
from pathlib import Path
import ray
import time
from typing import Any, Callable
import asyncio
import signal
from ray.exceptions import GetTimeoutError

# Map ISAAC_NUCLEUS_DIR / ISAACLAB_NUCLEUS_DIR to local mirrored assets.
_repo_assets_root = Path(__file__).resolve().parents[3] / "assets"
_configured_asset_root = os.environ.get("ISAACLAB_ASSET_ROOT")
if _configured_asset_root:
    configured_path = Path(_configured_asset_root)
    if not configured_path.exists() and _repo_assets_root.exists():
        os.environ["ISAACLAB_ASSET_ROOT"] = str(_repo_assets_root)
else:
    os.environ["ISAACLAB_ASSET_ROOT"] = str(_repo_assets_root)


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
        self.step_done = {}  # env_id -> payload dict
        
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

        done_bool = bool(ret.get("done", False))
        success_bool = bool(ret.get("success", False))
        timeout_bool = bool(ret.get("timeout", False))
        new_task_av = bool(ret.get("new_task_available", False))
        new_task_idx = int(ret.get("new_task_index", -1))

        info = {
            "success": success_bool,
            "timeout": timeout_bool,
            "env_id": env_id,
            "new_task_available": new_task_av,
            "new_task_index": new_task_idx,
        }
        print(f"Proxy.remote_step returning for env={env_id} info={info}")
        return {
            "images": images,
            "reward": 1.0 if success_bool else 0.0,
            "done": done_bool,
            "obs_str": "Action executed",
            "info": info,
        }

    async def remote_submit(self, env_id):
        """Backward-compatible submit API."""
        return await self.remote_step(env_id, {"type": "submit"})
    
    async def render(self, env_id):
        """Render current state of a specific environment slot."""
        return self.latest_images.get(env_id, [])
    
    async def _set_step_done(self, env_id, done, success=None, timeout=False, new_task_available=False, new_task_index=-1):
        """Mark a step done for a trainer waiting on env_id.

        This async method stores the payload and notifies the condition
        so any awaiting `remote_step` will wake and proceed.
        """
        if success is None:
            success = bool(done)
        payload = {
            "done": bool(done),
            "success": bool(success),
            "timeout": bool(timeout),
            "new_task_available": bool(new_task_available),
            "new_task_index": int(new_task_index),
        }
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


def _run_with_init_heartbeat(
    step_name: str,
    fn: Callable[[], Any],
) -> Any:
    """Run an init step with start/end timing logs."""
    start_t = time.perf_counter()
    print(f"[INIT] {step_name}: start")
    try:
        return fn()
    finally:
        elapsed = time.perf_counter() - start_t
        print(f"[INIT] {step_name}: done ({elapsed:.1f}s)")


def main():
    """Main entry point - runs in main thread as required by Isaac Sim."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="multipicture_assembling_from_begin")
    # Allow enabling/disabling headless mode via CLI
    parser.add_argument("--headless", dest="headless", action="store_true", help="Run Isaac in headless mode (no GUI).")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run Isaac with GUI (disable headless).")
    parser.add_argument("--record", action="store_true", default=False, help="Record video using gymnasium RecordVideo.")
    parser.add_argument(
        "--video-length",
        type=int,
        default=0,
        help="Recorded clip length in simulation steps (0 = keep recording until close/reset).",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=0,
        help="Start a new clip every N simulation steps (0 = start once at step 0).",
    )
    parser.add_argument("--ik-lambda-val", type=float, default=None)
    args = parser.parse_args()
    
    # Connect to Ray cluster
    NAMESPACE = "vagen_training"
    ray_init_t = time.perf_counter()
    print("[INIT] ray.init(address='auto'): start")
    ray.init(address="auto", namespace=NAMESPACE, ignore_reinit_error=True)
    print(f"[INIT] ray.init(address='auto'): done ({time.perf_counter() - ray_init_t:.1f}s)")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Isaac Sim sees GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Ensure IsaacLab editable installs still resolve after moving the repo.
    # We prepend the local IsaacLab source paths if they exist.
    repo_root = Path(__file__).resolve().parents[3]
    isaac_source_root = repo_root / "IsaacLab" / "source"
    isaac_modules = [
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_contrib",
        "isaaclab_mimic",
        "isaaclab_rl",
        "isaaclab_tasks",
    ]
    added_paths = []
    for mod in isaac_modules:
        candidate = isaac_source_root / mod
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
                added_paths.append(candidate_str)
    if added_paths:
        print(f"Added IsaacLab paths to sys.path: {added_paths}")

    # Import Isaac AFTER Ray and ENV setup
    from isaaclab.app import AppLauncher
    
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
    app_launcher = _run_with_init_heartbeat(
        "AppLauncher",
        lambda: AppLauncher(launcher_args),
    )
    simulation_app = app_launcher.app
    print("Isaac Simulation App launched successfully")
    
    # Import after app launch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    
    # Import env config from package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from vagen.server.server import VagenStackExecutionManager
    from isaaclab_tasks.manager_based.manipulation.assembling import build_env_cfg
    from isaaclab_tasks.manager_based.manipulation.assembling.config import resolve_task_id
    
    # Create environment
    print("Creating Isaac Lab environment...")
    resolved_task_id = resolve_task_id(config["task"])
    if resolved_task_id != config["task"]:
        print(
            f"[WARN]: Requested task '{config['task']}' is not registered. "
            f"Using '{resolved_task_id}' instead."
        )
    ret = _run_with_init_heartbeat(
        "build_env_cfg",
        lambda: build_env_cfg(
            task_name=resolved_task_id,
            cube_size=config.get("cube_size", 0.045),
        ),
    )

    # Keep scene env count aligned with server runtime config.
    if hasattr(ret, "scene") and hasattr(ret.scene, "num_envs"):
        before_envs = int(getattr(ret.scene, "num_envs", config["num_envs"]))
        ret.scene.num_envs = int(config["num_envs"])
        print(
            f"[INIT] Override env_cfg.scene.num_envs: {before_envs} -> {int(config['num_envs'])}"
        )

    if isinstance(ret, (tuple, list)) and len(ret) >= 3:
        env_cfg, cube_names = ret[0], ret[1]
    else:
        # Newer API returned a single env_cfg object. Build compatible
        # `cube_names` from the env_cfg.scene where possible.
        env_cfg = ret
        cube_names = []
        scene = env_cfg.scene
        # Collect attributes named like cube_1, cube_2, ...
        candidates = [n for n in dir(scene) if n.startswith("cube_")]

        def _idx(n):
            return int(n.split("_")[1])

        candidates = sorted(set(candidates), key=_idx)
        if candidates:
            cube_names = candidates
        else:
            cube_names = [f"cube_{i + 1}" for i in range(8)]

    env = _run_with_init_heartbeat(
        "gym.make",
        lambda: gym.make(resolved_task_id, cfg=env_cfg, render_mode="rgb_array" if args.record else None),
    )
    record_single_clip_mode = False
    video_prefix = ""
    record_clip_index = 0
    if args.record:
        out_dir = os.path.join(os.getcwd(), "outputs", "videos")
        os.makedirs(out_dir, exist_ok=True)
        video_interval = int(args.video_interval)
        video_length = int(args.video_length)
        if video_length < 0:
            video_length = 0
        video_prefix = f"isaac_record_{args.task.replace('/', '_')}_{int(time.time())}"
        video_run_dir = os.path.join(out_dir, video_prefix)
        os.makedirs(video_run_dir, exist_ok=True)
        if video_interval <= 0:
            # Disable automatic trigger and start recording manually on reset commands.
            step_trigger = lambda step: False
            record_single_clip_mode = True
            trigger_desc = "single-clip(manual-on-reset)"
        else:
            step_trigger = lambda step: step % video_interval == 0
            trigger_desc = f"periodic(every={video_interval})"
        video_kwargs = {
            "video_folder": video_run_dir,
            "step_trigger": step_trigger,
            "video_length": video_length,
            "name_prefix": video_prefix,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(
            "[INFO]: RecordVideo enabled "
            f"trigger={trigger_desc} video_length={video_length} output={video_run_dir}"
        )

    def _start_manual_recording(reason: str):
        nonlocal record_clip_index
        if not (args.record and record_single_clip_mode):
            return
        if not hasattr(env, "start_recording"):
            return
        if bool(getattr(env, "recording", False)):
            return
        clip_name = f"{video_prefix}-manual-{record_clip_index}"
        record_clip_index += 1
        try:
            env.start_recording(clip_name)
            # Capture one frame immediately so the clip is never empty if stopped quickly.
            env._capture_frame()
            print(f"[INFO]: Manual RecordVideo start '{clip_name}' reason={reason}")
        except Exception as e:
            print(f"[WARN]: Failed to start manual recording ({reason}): {e}")

    print(f"Environment '{config['task']}' created")
    print(f"Action space: {env.action_space}")
    line_thickness = 0.001  # Use 1mm for precision.
    # Keep source cell spacing in sync with scene grid visualization.
    cell_size = 0.055 + line_thickness
    exec_mgr = _run_with_init_heartbeat(
        "VagenStackExecutionManager",
        lambda: VagenStackExecutionManager(
            env=env,
            cube_names=cube_names,
            cube_size=config.get("cube_size", 0.045),
            ik_lambda_val=args.ik_lambda_val,
            cell_size=cell_size,
        ),
    )
    
    # Initial reset
    obs = _run_with_init_heartbeat(
        "exec_mgr.reset_all",
        lambda: exec_mgr.reset_all(),
    )
    print("Environment reset complete")
    
    exec_mgr.reset_state_for_all_envs()
    
    print("[INFO]: Starting Isaac server main loop...")

    proxy_actor = IsaacEnvServerProxy.options(
        name="IsaacEnvServer",
        lifetime="detached",
        get_if_exists=True
    ).remote(num_envs=config["num_envs"])
    
    print("IsaacEnvServer proxy registered to Ray cluster")
    ray.get(proxy_actor.is_alive.remote())  # Ensure it's ready
    
    # Keep simulation running
    print("Isaac server entering main loop (Ctrl+C to exit)...")
    shutdown_requested = False
    shutdown_reason = "normal exit"
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_shutdown(signum, _frame):
        nonlocal shutdown_requested, shutdown_reason
        if not shutdown_requested:
            shutdown_requested = True
            shutdown_reason = f"signal {signum}"
            print(f"Received signal {signum}, requesting graceful shutdown...")

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        while simulation_app.is_running() and not shutdown_requested:
            
            # Check if any step commands have completed.
            for done_event in exec_mgr.collect_completed_step_events():
                env_id = int(done_event["env_id"])
                proxy_actor._set_step_done.remote(
                    env_id,
                    bool(done_event["done"]),
                    success=bool(done_event["success"]),
                    timeout=bool(done_event["timeout"]),
                    new_task_available=bool(done_event["new_task_available"]),
                    new_task_index=int(done_event["new_task_index"]),
                )
                print(
                    f"Marked step done for env {env_id} "
                    f"(task_idx: {int(done_event['task_index'])} state: {int(done_event['state'])}) "
                    f"done={bool(done_event['done'])}"
                )
                print(
                    "[Proxy->Trainer] "
                    f"env={env_id} STEP_DONE snapshot: "
                    f"task_index={int(done_event['task_index'])} "
                    f"num_tasks={int(done_event['num_tasks'])} "
                    f"new_task_available={bool(done_event['new_task_available'])} "
                    f"new_task_index={int(done_event['new_task_index'])} "
                    f"state={int(done_event['state'])}"
                )
            
            # Check for commands from Proxy (Trainer)
            try:
                commands = ray.get(proxy_actor.get_pending_commands.remote(), timeout=0.2)
            except GetTimeoutError:
                commands = []
            for env_id, cmd_type, data in commands:
                if cmd_type == "reset":
                    seed = data
                    obs = exec_mgr.handle_reset(env_id=env_id, seed=seed)
                    _start_manual_recording(reason=f"remote_reset_env_{env_id}")
                elif cmd_type == "step":
                    goal = data
                    print(f"[Proxy] Received step goal for env {env_id}: {goal}")
                    result = exec_mgr.handle_step_goal(env_id=env_id, goal=goal)
                    if result.get("immediate_done", False):
                        payload = result["done_payload"]
                        proxy_actor._set_step_done.remote(
                            env_id,
                            payload["done"],
                            success=payload["success"],
                            timeout=payload["timeout"],
                            new_task_available=payload["new_task_available"],
                            new_task_index=payload["new_task_index"],
                        )
                        continue
            

            # Camera readback is expensive. Only capture frames when proxy
            # requests are pending; video recording is handled by RecordVideo.
            exec_mgr.capture_requested_images(commands=commands, proxy_actor=proxy_actor)

            obs = exec_mgr.step(obs)

    except KeyboardInterrupt:
        shutdown_reason = "keyboard interrupt"
        print("Received shutdown signal")
    except SystemExit as exc:
        shutdown_reason = f"system exit ({exc.code})"
        raise
    except Exception as exc:
        shutdown_reason = f"exception: {type(exc).__name__}: {exc}"
        print(f"Unhandled exception in Isaac server: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Shutting down Isaac server... reason={shutdown_reason}")
        # IMPORTANT: close env (RecordVideo flush) before shutting down SimulationApp.
        try:
            exec_mgr.close()
        except Exception as e:
            print(f"[WARN] exec_mgr.close() failed during shutdown: {e}")
        try:
            env.close()
            print("Environment closed (recording finalized if enabled).")
        except Exception as e:
            print(f"[WARN] env.close() failed during shutdown: {e}")
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[WARN] simulation_app.close() failed during shutdown: {e}")
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        print("Isaac server shutdown complete")


if __name__ == "__main__":
    main()
