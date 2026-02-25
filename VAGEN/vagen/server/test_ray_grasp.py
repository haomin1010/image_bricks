#!/usr/bin/env python3
"""Ray-based end-to-end grasp test via start_isaac_server APIs.

This script validates the full grasp pipeline exposed by
``vagen.server.start_isaac_server`` only.

Test flow:
1. Connect to Ray and get ``IsaacEnvServer`` actor.
2. Allocate an environment slot.
3. Reset and query images.
4. Send one or more placement goals via ``remote_step``.
5. Submit and verify completion.
6. Release the environment slot.

Use ``--one-click-debug true`` for a single-command local debug flow
with GUI server startup (equivalent to ``--no-headless``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.exceptions import GetTimeoutError


DEFAULT_TASK = "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0"
DEFAULT_GOALS = "2,2,0;3,2,0;3,3,0"
DEFAULT_CAMERAS = "0,1,2,3,4"
DEFAULT_RAY_HEAD_LOG = "/tmp/ray_head_start.log"


def parse_goals(raw: str) -> List[Dict[str, int]]:
    goals: List[Dict[str, int]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid goal '{chunk}', expected format x,y,z")
        try:
            x, y, z = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise ValueError(f"Goal '{chunk}' contains non-integer values") from exc
        goals.append({"x": x, "y": y, "z": z})
    if not goals:
        raise ValueError("No valid goals were parsed from --goals")
    return goals


def parse_camera_ids(raw: str) -> Optional[List[int]]:
    raw = raw.strip()
    if not raw:
        return None
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out if out else None


def parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in {"1", "true", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{text}'")


def tail_text(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return ""
    return "".join(lines[-max_lines:])


def ray_get_with_timeout(ref: Any, label: str, timeout_s: float) -> Any:
    try:
        return ray.get(ref, timeout=timeout_s)
    except GetTimeoutError as exc:
        raise TimeoutError(f"RPC timeout while waiting for {label} ({timeout_s:.1f}s)") from exc


def expect_keys(payload: Dict[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise AssertionError(f"{label} missing keys: {missing}; payload keys={list(payload.keys())}")


def ensure_ray_initialized(address: str, namespace: str, local_fallback: bool) -> None:
    if ray.is_initialized():
        return

    try:
        ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
        print(f"[INFO] Connected to Ray: address={address} namespace={namespace}")
        return
    except Exception as exc:
        if not (local_fallback and address == "auto"):
            raise
        print(f"[WARN] ray.init(address='auto') failed: {exc}")
        print("[INFO] Falling back to local Ray runtime.")
        ray.init(namespace=namespace, ignore_reinit_error=True)
        print(f"[INFO] Local Ray initialized: namespace={namespace}")


def stop_ray_cluster() -> None:
    subprocess.run(
        ["ray", "stop", "--force"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def kill_existing_server_process() -> None:
    if shutil.which("pkill") is None:
        return
    subprocess.run(
        ["pkill", "-f", "VAGEN/vagen/server/start_isaac_server.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def start_ray_head(log_path: str) -> None:
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        subprocess.run(
            ["ray", "start", "--head", "--disable-usage-stats"],
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(f"[INFO] Ray head started (log: {log_path})")


def prepare_one_click_debug(args: argparse.Namespace) -> bool:
    print("[INFO] One-click debug enabled: reset Ray, use auto-start, enforce --no-headless.")
    stop_ray_cluster()
    if args.one_click_kill_existing_server:
        kill_existing_server_process()
    start_ray_head(args.ray_head_log)
    # Enforce one-click behavior in Python entrypoint.
    args.auto_start = True
    args.server_headless = False
    args.local_fallback = False
    return True


def wait_for_actor(actor_name: str, timeout_s: float, poll_s: float = 1.0) -> Any:
    start = time.time()
    while True:
        try:
            return ray.get_actor(actor_name)
        except ValueError:
            elapsed = time.time() - start
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Actor '{actor_name}' not found after {timeout_s:.1f}s"
                )
            time.sleep(poll_s)


def start_server_if_needed(args: argparse.Namespace) -> Tuple[Optional[subprocess.Popen], Any]:
    actor_name = args.actor_name
    try:
        actor = ray.get_actor(actor_name)
        print(f"[INFO] Found existing actor '{actor_name}'.")
        return None, actor
    except ValueError:
        pass

    if not args.auto_start:
        raise RuntimeError(
            f"Actor '{actor_name}' not found. Start the server first, or pass --auto-start true."
        )

    server_script = Path(args.server_script).resolve()
    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")

    log_path = os.path.abspath(args.server_log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    cmd = [
        sys.executable,
        str(server_script),
        "--num-envs",
        str(args.server_num_envs),
        "--device",
        args.server_device,
        "--task",
        args.server_task,
    ]
    if args.server_headless:
        cmd.append("--headless")
    else:
        cmd.append("--no-headless")

    env = os.environ.copy()
    if args.server_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.server_cuda_visible_devices

    print(f"[INFO] Starting server process: {' '.join(cmd)}")
    print(f"[INFO] Server log: {log_path}")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        # Parent process does not need to hold this file descriptor.
        log_file.close()

    try:
        actor = wait_for_actor(actor_name, timeout_s=args.server_start_timeout_s, poll_s=1.0)
        print(f"[INFO] Actor '{actor_name}' registered.")
        return proc, actor
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        server_log_tail = tail_text(log_path)
        if server_log_tail:
            print("[ERROR] Server log tail:")
            print(server_log_tail)
        raise


def release_env_slot(actor: Any, env_id: int, timeout_s: float) -> None:
    try:
        ray_get_with_timeout(
            actor.release_env_id.remote(env_id),
            label=f"release_env_id({env_id})",
            timeout_s=timeout_s,
        )
        print(f"[INFO] Released env slot {env_id}.")
    except Exception as exc:
        print(f"[WARN] Failed to release env slot {env_id}: {exc}")


def image_count(payload: Dict[str, Any]) -> int:
    images = payload.get("images", [])
    if isinstance(images, list):
        return len(images)
    return 0


def run_grasp_test(actor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    goals = parse_goals(args.goals)
    query_cameras = parse_camera_ids(args.query_cameras)

    alive = ray_get_with_timeout(
        actor.is_alive.remote(),
        label="is_alive",
        timeout_s=args.rpc_timeout_s,
    )
    if not alive:
        raise RuntimeError("Actor responded but is_alive returned False")

    env_id = ray_get_with_timeout(
        actor.allocate_env_id.remote(),
        label="allocate_env_id",
        timeout_s=args.rpc_timeout_s,
    )
    print(f"[INFO] Allocated env slot: {env_id}")

    summary: Dict[str, Any] = {
        "env_id": int(env_id),
        "seed": int(args.seed),
        "goals": goals,
        "query_cameras": query_cameras,
        "steps": [],
    }

    try:
        reset_resp = ray_get_with_timeout(
            actor.remote_reset.remote(env_id, int(args.seed)),
            label="remote_reset",
            timeout_s=args.rpc_timeout_s,
        )
        expect_keys(reset_resp, ["images", "info"], "remote_reset")
        reset_images = image_count(reset_resp)
        print(f"[INFO] reset returned {reset_images} images.")
        if args.require_images and reset_images == 0:
            raise AssertionError("remote_reset returned zero images")
        summary["reset_images"] = reset_images
        summary["reset_info"] = reset_resp.get("info", {})

        query_resp = None
        try:
            query_resp = ray_get_with_timeout(
                actor.remote_query.remote(env_id, query_cameras),
                label="remote_query",
                timeout_s=args.rpc_timeout_s,
            )
        except AttributeError:
            # Fallback for older/newer servers without remote_query
            query_resp = ray_get_with_timeout(
                actor.render.remote(env_id),
                label="render",
                timeout_s=args.rpc_timeout_s,
            )
            query_resp = {"images": query_resp, "info": {"env_id": env_id}}

        expect_keys(query_resp, ["images", "info"], "remote_query")
        query_images = image_count(query_resp)
        print(f"[INFO] query returned {query_images} images.")
        if args.require_images and query_images == 0:
            raise AssertionError("remote_query returned zero images")
        summary["query_images"] = query_images

        for idx, goal in enumerate(goals):
            method_name = "remote_step"
            ref = actor.remote_step.remote(env_id, goal)

            resp = ray_get_with_timeout(
                ref,
                label=f"{method_name}#{idx}",
                timeout_s=args.step_timeout_s,
            )
            expect_keys(resp, ["images", "reward", "done", "info"], method_name)
            info = resp.get("info", {})
            step_item = {
                "method": method_name,
                "goal": goal,
                "done": bool(resp.get("done", False)),
                "reward": float(resp.get("reward", 0.0)),
                "image_count": image_count(resp),
                "success": bool(info.get("success", False)),
                "new_task_available": bool(info.get("new_task_available", False)),
                "new_task_index": int(info.get("new_task_index", -1)),
            }
            if args.require_images and step_item["image_count"] == 0:
                raise AssertionError(f"{method_name}#{idx} returned zero images")
            print(
                f"[INFO] {method_name}#{idx} goal={goal} done={step_item['done']} "
                f"reward={step_item['reward']:.3f} images={step_item['image_count']} "
                f"new_task={step_item['new_task_available']} idx={step_item['new_task_index']}"
            )
            summary["steps"].append(step_item)

        submit_resp = ray_get_with_timeout(
            actor.remote_submit.remote(env_id),
            label="remote_submit",
            timeout_s=args.submit_timeout_s,
        )
        expect_keys(submit_resp, ["images", "reward", "done", "info"], "remote_submit")
        submit_done = bool(submit_resp.get("done", False))
        submit_images = image_count(submit_resp)
        submit_reward = float(submit_resp.get("reward", 0.0))
        submit_info = submit_resp.get("info", {})
        print(
            f"[INFO] submit done={submit_done} reward={submit_reward:.3f} "
            f"images={submit_images}"
        )
        if args.require_images and submit_images == 0:
            raise AssertionError("remote_submit returned zero images")
        if not submit_done:
            raise AssertionError("remote_submit returned done=False, grasp pipeline not complete")

        summary["submit"] = {
            "done": submit_done,
            "reward": submit_reward,
            "image_count": submit_images,
            "info": submit_info,
        }
        return summary
    finally:
        release_env_slot(actor, int(env_id), timeout_s=args.rpc_timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ray E2E grasp test based on start_isaac_server API."
    )
    parser.add_argument(
        "--one-click-debug",
        type=parse_bool,
        default=False,
        help=(
            "One-click debug mode: stop old Ray, start ray head, force auto-start server "
            "with --no-headless. true/false."
        ),
    )
    parser.add_argument(
        "--one-click-keep-ray",
        type=parse_bool,
        default=False,
        help="Keep Ray cluster running after one-click mode exits. true/false.",
    )
    parser.add_argument(
        "--one-click-kill-existing-server",
        type=parse_bool,
        default=True,
        help="Kill lingering start_isaac_server.py before one-click start. true/false.",
    )
    parser.add_argument(
        "--ray-head-log",
        type=str,
        default=DEFAULT_RAY_HEAD_LOG,
        help="Log path used for 'ray start --head' in one-click mode.",
    )

    parser.add_argument("--address", type=str, default="auto", help="Ray cluster address.")
    parser.add_argument("--namespace", type=str, default="vagen_training", help="Ray namespace.")
    parser.add_argument("--actor-name", type=str, default="IsaacEnvServer", help="Ray actor name.")
    parser.add_argument(
        "--local-fallback",
        type=parse_bool,
        default=True,
        help="Fallback to local ray.init() when --address auto is unavailable. true/false.",
    )

    parser.add_argument("--seed", type=int, default=0, help="Seed used for remote_reset.")
    parser.add_argument(
        "--goals",
        type=str,
        default=DEFAULT_GOALS,
        help="Placement sequence as 'x,y,z;x,y,z;...'.",
    )
    parser.add_argument(
        "--query-cameras",
        type=str,
        default=DEFAULT_CAMERAS,
        help="Comma-separated camera IDs for remote_query, empty string to skip selection.",
    )
    parser.add_argument(
        "--require-images",
        type=parse_bool,
        default=True,
        help="Fail if any RPC returns no images. true/false.",
    )

    parser.add_argument("--rpc-timeout-s", type=float, default=30.0, help="Timeout for quick RPC calls.")
    parser.add_argument("--step-timeout-s", type=float, default=240.0, help="Timeout per place/step call.")
    parser.add_argument("--submit-timeout-s", type=float, default=120.0, help="Timeout for submit.")

    parser.add_argument(
        "--auto-start",
        type=parse_bool,
        default=False,
        help="Auto-start start_isaac_server.py when actor is missing. true/false.",
    )
    parser.add_argument(
        "--server-script",
        type=str,
        default=str(Path(__file__).with_name("start_isaac_server.py")),
        help="Path to start_isaac_server.py.",
    )
    parser.add_argument("--server-num-envs", type=int, default=8, help="--num-envs for auto-started server.")
    parser.add_argument("--server-device", type=str, default="cuda:0", help="--device for auto-started server.")
    parser.add_argument("--server-task", type=str, default=DEFAULT_TASK, help="--task for auto-started server.")
    parser.add_argument(
        "--server-headless",
        type=parse_bool,
        default=True,
        help="Run auto-started server in headless mode. true/false.",
    )
    parser.add_argument(
        "--server-cuda-visible-devices",
        type=str,
        default="",
        help="Optional CUDA_VISIBLE_DEVICES for auto-started server.",
    )
    parser.add_argument(
        "--server-start-timeout-s",
        type=float,
        default=180.0,
        help="Wait timeout for actor registration after auto-start.",
    )
    parser.add_argument(
        "--server-log",
        type=str,
        default="/tmp/isaac_server_test.log",
        help="Log file used when auto-start launches server.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    ray_head_started = False

    try:
        if args.one_click_debug:
            ray_head_started = prepare_one_click_debug(args)

        ensure_ray_initialized(
            address=args.address,
            namespace=args.namespace,
            local_fallback=args.local_fallback,
        )
        server_proc, actor = start_server_if_needed(args)
        try:
            summary = run_grasp_test(actor, args)
        finally:
            if server_proc is not None:
                if server_proc.poll() is None:
                    print("[INFO] Stopping auto-started server process.")
                    server_proc.terminate()
                    try:
                        server_proc.wait(timeout=15)
                    except Exception:
                        server_proc.kill()
                print(f"[INFO] Auto-started server log: {os.path.abspath(args.server_log)}")

        print("[PASS] Ray grasp test completed successfully.")
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()
        if ray_head_started and not args.one_click_keep_ray:
            stop_ray_cluster()


if __name__ == "__main__":
    raise SystemExit(main())
