import os
from typing import Any

from PIL import Image

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
        self._step_count: int = 0  # global sim-step counter for torque logging

        default_max_tasks = len(self.cube_names) if len(self.cube_names) > 0 else 8
        resolved_max_tasks = (
            max(1, int(os.getenv("VAGEN_MAX_TASKS", str(default_max_tasks))))
            if max_tasks is None
            else max(1, int(max_tasks))
        )
        # Build runtime through environment-side wrapper.
        self.action_runtime = self.env.unwrapped.cfg.build_server_runtime(
            env=self.env,
            cube_names=self.cube_names,
            cube_size=self.cube_size,
            max_tasks=resolved_max_tasks,
            grid_origin=list(grid_origin),
            cell_size=float(cell_size),
            grid_size=max(1, len(self.cube_names)),
        )
        # Lazy import to avoid importing task modules before Isaac app startup.
        from isaaclab_tasks.manager_based.manipulation.assembling import mdp

        default_cam_names = ("camera", "camera_front", "camera_side", "camera_iso", "camera_iso2")
        self.cam_names = list(getattr(mdp, "STACK_CAMERA_NAMES", getattr(mdp, "DEFAULT_CAMERA_NAMES", default_cam_names)))

        print(
            "[INFO]: Runtime bootstrapped via env cfg "
            f"(cfg={type(self.env.unwrapped.cfg).__name__}, "
            f"impl={type(self.action_runtime).__name__}, "
            f"ik_lambda_override={self.ik_lambda_val})"
        )
        print(
            f"[INFO]: Action runtime initialized by manager "
            f"(impl={type(self.action_runtime).__name__}, max_tasks={resolved_max_tasks}, cell_size={float(cell_size):.4f})"
        )

    def reset_all(self):
        obs, _ = self.env.reset()
        return obs

    def handle_reset(self, env_id: int, seed: int | None = None):
        reset_options = {"env_ids": [env_id]}
        if seed is not None:
            reset_options["seed"] = int(seed)
        obs, _ = self.env.reset(options=reset_options)
        self.action_runtime.on_reset_env(env_id)
        return obs

    def handle_step_goal(self, env_id: int, goal: Any) -> dict:
        return self.action_runtime.handle_step_goal(env_id, goal)

    def collect_completed_step_events(self) -> list[dict[str, int | bool]]:
        return self.action_runtime.collect_completed_step_events()

    def reset_state_for_all_envs(self) -> None:
        self.action_runtime.reset_state_for_all_envs()

    def get_step_snapshot(self, env_id: int) -> dict[str, int | bool]:
        return self.action_runtime.get_step_snapshot(env_id)

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

    def step(self, obs: dict):
        obs = self.action_runtime.step(obs)
        self._step_count += 1
        return obs

    def close(self):
        pass
