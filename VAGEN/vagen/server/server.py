import os
from typing import Any

from PIL import Image
import torch

from .state_machine import StackingStateMachine

# Isaac Lab and Gym imports will be deferred inside the Actor to ensure
# they are only loaded in the process that has the simulation_app. 

class VagenStackExecutionManager:
    """Server-side execution manager for stack task control."""

    def __init__(
        self,
        env,
        cube_names: list[str],
        cube_size: float,
        ik_lambda_val: float | None = None,
        *,
        max_tasks: int | None = None,
        grid_origin: list[float] | tuple[float, float, float] = (0.5, 0.0, 0.001),
        cell_size: float = 0.056,
    ):
        self.env = env
        self.cube_names = cube_names
        self.cube_size = float(cube_size)
        self.num_envs = int(env.unwrapped.num_envs)
        self.device = env.unwrapped.device
        self.ik_lambda_val = ik_lambda_val
        self._step_initial_task_idx: dict[int, dict[str, Any]] = {}
        self._step_count: int = 0  # global sim-step counter for torque logging

        default_max_tasks = len(self.cube_names) if len(self.cube_names) > 0 else 8
        resolved_max_tasks = (
            max(1, int(os.getenv("VAGEN_MAX_TASKS", str(default_max_tasks))))
            if max_tasks is None
            else max(1, int(max_tasks))
        )
        self.sm = StackingStateMachine(
            self.num_envs,
            self.device,
            scene=self.env.unwrapped.scene,
            cube_names=self.cube_names,
            max_tasks=resolved_max_tasks,
            cube_z_size=self.cube_size,
            grid_origin=list(grid_origin),
            cell_size=float(cell_size),
            grid_size=max(1, len(self.cube_names)),
        )
        # Lazy import to avoid importing task modules before Isaac app startup.
        from isaaclab_tasks.manager_based.manipulation.assembling import mdp

        default_cam_names = ("camera", "camera_front", "camera_side", "camera_iso", "camera_iso2")
        self.cam_names = list(getattr(mdp, "STACK_CAMERA_NAMES", getattr(mdp, "DEFAULT_CAMERA_NAMES", default_cam_names)))

        # Shared state consumed by runtime terms.
        self.env.unwrapped._vagen_new_task_available = self.sm.new_task_available
        self.env.unwrapped._vagen_new_task_index = self.sm.new_task_index
        print(
            "[INFO]: Operational-space control enabled via action term "
            "(input action=[ee_pos(3), ee_quat_wxyz(4), gripper(1)], "
            "output=Franka joint efforts + parallel-gripper cmd). "
            f"ik_lambda_override={self.ik_lambda_val}"
        )
        print(
            f"[INFO]: State machine initialized by manager "
            f"(max_tasks={resolved_max_tasks}, cell_size={float(cell_size):.4f})"
        )
        self._gripper_center_marker = None
        self._gripper_marker_enabled = os.getenv("VAGEN_VIS_GRIPPER_CENTER", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if self._gripper_marker_enabled:
            self._init_gripper_center_marker()

    def _init_gripper_center_marker(self) -> None:
        """Initialize gripper-center marker visualization in world frame."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import SPHERE_MARKER_CFG

            marker_cfg = SPHERE_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/VAGEN/gripper_center"
            marker_cfg.markers["sphere"].radius = 0.008
            marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0)
            )
            self._gripper_center_marker = VisualizationMarkers(marker_cfg)
            self._gripper_center_marker.set_visibility(True)
            print("[INFO]: Enabled gripper-center marker visualization at /Visuals/VAGEN/gripper_center")
        except Exception as exc:
            self._gripper_center_marker = None
            print(f"[WARN]: Failed to initialize gripper-center marker visualization: {exc}")

    def _update_gripper_center_marker(self, policy_obs: dict | None) -> None:
        marker = self._gripper_center_marker
        if marker is None or not isinstance(policy_obs, dict):
            return
        ee_pos_env = policy_obs.get("ee_pos")
        env_origin = policy_obs.get("env_origin")
        if ee_pos_env is None or env_origin is None:
            return
        if ee_pos_env.ndim != 2 or ee_pos_env.shape[-1] != 3:
            return
        if env_origin.ndim != 2 or env_origin.shape[-1] != 3:
            return
        # ee_pos is env-local for this task. Convert to world before visualization.
        marker.visualize(translations=ee_pos_env + env_origin)

    def reset_all(self):
        obs, _ = self.env.reset()
        return obs

    def handle_reset(self, env_id: int, seed: int | None = None):
        reset_options = {"env_ids": [env_id]}
        if seed is not None:
            reset_options["seed"] = int(seed)
        obs, _ = self.env.reset(options=reset_options)
        self.sm.reset_envs([env_id])
        self._step_initial_task_idx.pop(int(env_id), None)
        return obs

    def _parse_goal_to_world(self, env_id: int, goal: dict) -> torch.Tensor:
        if not isinstance(goal, dict):
            raise ValueError(f"Invalid goal type: {type(goal)}")
        if not all(k in goal for k in ("x", "y", "z")):
            raise KeyError("Goal must contain keys 'x','y','z'")

        g_x, g_y, g_z = float(goal["x"]), float(goal["y"]), float(goal["z"])
        grid_origin = self.sm.grid_origin
        cell_size = self.sm.cell_size
        env_origin = self.env.unwrapped.scene.env_origins[env_id]
        target_x = grid_origin[0].item() + (g_x - 4.5) * cell_size
        target_y = grid_origin[1].item() + (g_y - 4.5) * cell_size
        target_z = (g_z + 0.5) * self.cube_size + 0.002
        return env_origin + torch.tensor([target_x, target_y, target_z], device=env_origin.device)

    def handle_step_goal(self, env_id: int, goal: Any) -> dict:
        is_submit = isinstance(goal, dict) and goal.get("type") == "submit"
        current_task_idx = int(self.sm.task_index[env_id].item())
        if is_submit:
            num_tasks = int(self.sm.num_tasks_per_env[env_id].item())
            is_success = current_task_idx >= num_tasks
            self._step_initial_task_idx[int(env_id)] = {
                "init_idx": int(current_task_idx),
                "was_submit": True,
                "submit_success": bool(is_success),
            }
            return {"immediate_done": False, "done_payload": None}

        max_tasks = int(getattr(self.sm, "max_tasks", self.sm.target_positions.shape[1]))
        if current_task_idx >= max_tasks:
            self._step_initial_task_idx.pop(int(env_id), None)
            self.sm.num_tasks_per_env[env_id] = max_tasks
            self.sm.state[env_id] = self.sm.IDLE
            return {
                "immediate_done": True,
                "done_payload": {
                    "done": True,
                    "success": False,
                    "timeout": True,
                    "new_task_available": False,
                    "new_task_index": -1,
                },
            }

        self._step_initial_task_idx[int(env_id)] = {
            "init_idx": int(current_task_idx),
            "was_submit": False,
            "submit_success": None,
        }
        target_pos_w = self._parse_goal_to_world(env_id, goal)
        self.sm.set_env_targets([env_id], target_pos_w)
        return {"immediate_done": False, "done_payload": None}

    def collect_completed_step_events(self) -> list[dict[str, int | bool]]:
        events: list[dict[str, int | bool]] = []
        for env_id in list(self._step_initial_task_idx.keys()):
            snapshot = self.get_step_snapshot(int(env_id))
            task_idx_now = int(snapshot["task_index"])
            sm_state_now = int(snapshot["state"])

            init_val = self._step_initial_task_idx[env_id]
            init_idx = int(init_val.get("init_idx", 0))
            was_submit = bool(init_val.get("was_submit", False))
            submit_success = init_val.get("submit_success", None)

            if (task_idx_now > init_idx) or sm_state_now == -1:
                done_flag = True if was_submit else False
                success_flag = done_flag if submit_success is None else bool(submit_success)
                events.append(
                    {
                        "env_id": int(env_id),
                        "done": bool(done_flag),
                        "success": bool(success_flag),
                        "timeout": False,
                        "new_task_available": bool(snapshot["new_task_available"]),
                        "new_task_index": int(snapshot["new_task_index"]),
                        "task_index": int(task_idx_now),
                        "state": int(sm_state_now),
                        "num_tasks": int(snapshot["num_tasks"]),
                    }
                )
                del self._step_initial_task_idx[env_id]
        return events

    def reset_state_for_all_envs(self) -> None:
        for env_id in range(self.num_envs):
            self.sm.reset_state(env_id)

    def get_step_snapshot(self, env_id: int) -> dict[str, int | bool]:
        return {
            "task_index": int(self.sm.task_index[env_id].item()),
            "state": int(self.sm.state[env_id].item()),
            "num_tasks": int(self.sm.num_tasks_per_env[env_id].item()),
            "new_task_available": bool(self.sm.new_task_available[env_id].item()),
            "new_task_index": int(self.sm.new_task_index[env_id].item()),
        }

    def capture_requested_images(self, commands: list, proxy_actor):
        if not commands:
            return
        requested_envs = {int(c[0]) for c in commands}
        for env_id in requested_envs:
            img_list = []
            for cam_name in self.cam_names:
                cam = self.env.unwrapped.scene[cam_name]
                rgb_tensor = cam.data.output["rgb"][env_id]
                rgb_np = rgb_tensor.cpu().numpy().astype("uint8")
                img_list.append(Image.fromarray(rgb_np))
            proxy_actor.update_state.remote(env_id, img_list)

    def compute_joint_actions(self, obs: dict) -> torch.Tensor:
        policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else None
        if isinstance(policy_obs, dict):
            self.env.unwrapped._vagen_policy_obs = policy_obs
            self._update_gripper_center_marker(policy_obs)
        ee_pos_des, ee_quat_des, gripper_cmd = self.sm.compute_ee_pose_targets(obs)
        quat_norm = torch.linalg.vector_norm(ee_quat_des, dim=-1, keepdim=True).clamp_min(1e-8)
        ee_quat_des = ee_quat_des / quat_norm
        gripper_cmd = gripper_cmd.to(dtype=ee_pos_des.dtype)
        return torch.cat([ee_pos_des, ee_quat_des, gripper_cmd.unsqueeze(-1)], dim=-1)

    def step(self, obs: dict):
        actions = self.compute_joint_actions(obs)
        obs, _, _, _, _ = self.env.step(actions)
        self._step_count += 1
        return obs

    def close(self):
        pass
