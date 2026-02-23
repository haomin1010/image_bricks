#!/usr/bin/env python3
"""
Minimal GPU joint-suction regression loop.

This script reuses the server state machine and runs a single pick-and-place cycle
without Ray proxy, so we can quickly validate attach/detach behavior.
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--cube-size", type=float, default=0.045)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    launcher_args = {
        "headless": args.headless,
        "num_envs": args.num_envs,
        "task": args.task,
        "enable_cameras": False,
    }
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from vagen.server.server import StackingStateMachine, get_stack_cube_env_cfg

    env_cfg = get_stack_cube_env_cfg(
        task_name=args.task,
        device=args.device,
        num_envs=args.num_envs,
        enable_cameras=False,
    )
    if isinstance(env_cfg, (tuple, list)):
        env_cfg = env_cfg[0]

    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()

    cube_names = [n for n in dir(env_cfg.scene) if n.startswith("cube_")]
    cube_names = sorted(cube_names, key=lambda x: int(x.split("_")[1]))
    if not cube_names:
        cube_names = [f"cube_{i+1}" for i in range(8)]

    sm = StackingStateMachine(
        env.unwrapped.num_envs,
        env.unwrapped.device,
        scene=env.unwrapped.scene,
        cube_names=cube_names,
        max_tasks=8,
        cube_z_size=args.cube_size,
        grid_origin=[0.5, 0.0, 0.001],
        cell_size=0.056,
    )
    sm.set_stage(env.unwrapped.sim.stage)

    # One target task per env: place at grid center (2.5, 2.5, z=0).
    for env_id in range(env.unwrapped.num_envs):
        env_origin = env.unwrapped.scene.env_origins[env_id]
        target_x = sm.grid_origin[0].item() + (2.5 - 2.5) * sm.cell_size
        target_y = sm.grid_origin[1].item() + (2.5 - 2.5) * sm.cell_size
        target_z = (0.5 * args.cube_size) + 0.002
        target_pos_w = env_origin + torch.tensor([target_x, target_y, target_z], device=env_origin.device)
        sm.target_positions[env_id, 0] = target_pos_w
        sm.num_tasks_per_env[env_id] = 1
        sm.task_index[env_id] = 0
        sm.state[env_id] = sm.APPROACH_CUBE
        sm.state_timer[env_id] = 0

    cube_name = cube_names[0]
    cube_asset = env.unwrapped.scene[cube_name]
    init_cube_z = cube_asset.data.root_pos_w[:, 2].clone()
    max_cube_z = init_cube_z.clone()
    last_attached = sm.attached_cube_idx.clone()

    print(f"[TEST] task={args.task} device={args.device} num_envs={args.num_envs} steps={args.steps}", flush=True)
    print(f"[TEST] cube={cube_name} init_cube_z={init_cube_z.tolist()}", flush=True)

    success = False
    try:
        for step in range(args.steps):
            if not simulation_app.is_running():
                print("[TEST] simulation_app is not running, stop loop.", flush=True)
                break

            actions = sm.compute_action(obs)
            obs, _, _, _, _ = env.step(actions)
            sm.apply_magic_suction(obs)

            cube_z = cube_asset.data.root_pos_w[:, 2]
            max_cube_z = torch.maximum(max_cube_z, cube_z)

            changed = sm.attached_cube_idx != last_attached
            if torch.any(changed):
                idxs = torch.nonzero(changed, as_tuple=False).squeeze(-1).tolist()
                for env_id in idxs:
                    print(
                        f"[ATTACH_CHANGE] step={step} env={env_id} "
                        f"attached={int(sm.attached_cube_idx[env_id].item())} "
                        f"state={int(sm.state[env_id].item())}"
                    , flush=True)
                last_attached = sm.attached_cube_idx.clone()

            if step % args.print_every == 0 or step == args.steps - 1:
                env_id = 0
                print(
                    f"[STEP] step={step} state={int(sm.state[env_id].item())} "
                    f"task_idx={int(sm.task_index[env_id].item())}/{int(sm.num_tasks_per_env[env_id].item())} "
                    f"attached={int(sm.attached_cube_idx[env_id].item())} "
                    f"cube_z={float(cube_z[env_id].item()):.4f} "
                    f"max_cube_z={float(max_cube_z[env_id].item()):.4f}"
                , flush=True)
        lift_delta = max_cube_z - init_cube_z
        print(f"[RESULT] lift_delta={lift_delta.tolist()}", flush=True)
        print(f"[RESULT] final_task_idx={sm.task_index.tolist()} final_state={sm.state.tolist()}", flush=True)
        success = bool(torch.any(lift_delta > 0.03).item())
        print(f"[RESULT] success_lift_gt_3cm={success}", flush=True)
    finally:
        env.close()
        simulation_app.close()

    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
