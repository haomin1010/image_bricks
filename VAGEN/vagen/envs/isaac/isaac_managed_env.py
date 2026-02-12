"""
Isaac Managed Environment — GymImageEnv proxy backed by the EnvManager system.

This environment supports two action types from the VLM:

1. **Query** – ``{"query": [cam_id, ...]}``
   Returns images from the requested cameras.  No physics step.
2. **Place** – ``{"x": INT, "y": INT, "z": INT}``
   Places a brick and returns camera-0's image.
3. **Submit** – ``submit``
   Ends the episode.

Usage
-----
Register this class in the env_registry config::

    env_registry:
      brick_isaac: vagen.envs.brick_isaac.IsaacManagedEnv

The ``env_config`` dict in the dataset YAML should contain at least:

- ``num_total_envs`` (int): Total sub-envs in the Isaac DirectVectorEnv.
- ``n_cameras`` (int): Number of cameras available (IDs 0 .. n_cameras-1).
"""

from __future__ import annotations

import logging
import ray
import asyncio
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..gym_image_env import GymImageEnv
from .utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    query_result_template,
    system_prompt,
)
from .utils.utils import parse_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IsaacManagedEnvConfig:
    """Configuration for the Isaac-managed environment."""

    # Executor / manager settings
    num_total_envs: int = 64  # total sub-envs in the DirectVectorEnv

    # Cameras
    n_cameras: int = 3  # total cameras (IDs 0 .. n_cameras-1)
    image_size: Tuple[int, int] = (224, 224)

    # Step limits
    max_steps: int = 200

    # Prompt
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True

    # Reward shaping
    format_reward: float = 0.1
    success_reward: float = 1.0


_CONFIG_FIELDS = {f.name for f in fields(IsaacManagedEnvConfig)}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IsaacManagedEnv(GymImageEnv):
    """
    GymImageEnv proxy backed by the IsaacEnvServer Ray Actor.
    Each agent loop creates one ``IsaacManagedEnv`` instance. On ``reset()``
    a sub-env ID is allocated from the global server; on ``close()`` it is
    automatically returned when the instance is destroyed or explicitly closed.
    """

    _server_handle = None
    _server_lock = asyncio.Lock()

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter to known config fields
        known = {k: v for k, v in env_config.items() if k in _CONFIG_FIELDS}
        self.config = IsaacManagedEnvConfig(**known)

        # Instance state
        self._sub_env_id: Optional[int] = None
        self.total_reward: float = 0.0
        self.steps_taken: int = 0        
        self.trajectory: List[Dict[str, Any]] = []

    async def _get_server(self):
        """Lazily obtain the per-worker server handle singleton."""
        if IsaacManagedEnv._server_handle is not None:
            return IsaacManagedEnv._server_handle

        async with IsaacManagedEnv._server_lock:
            if IsaacManagedEnv._server_handle is not None:
                return IsaacManagedEnv._server_handle
            
            try:
                IsaacManagedEnv._server_handle = ray.get_actor("IsaacEnvServer")
                logger.info("Connected to existing IsaacEnvServer actor.")
            except ValueError:
                logger.info("IsaacEnvServer actor not found. It should be started by the main script.")
                raise RuntimeError("IsaacEnvServer actor not found. Ensure it is started as a detached actor.")
            
            return IsaacManagedEnv._server_handle

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        """Return the system-level prompt observation."""
        return {"obs_str": self._build_system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Allocate a sub-env ID (on first call) and reset it on the Isaac side.
        Returns camera-0's image as the initial observation.
        """
        server = await self._get_server()

        # Allocate once; reuse the same sub-env ID across resets
        if self._sub_env_id is None:
            self._sub_env_id = await server.allocate_env_id.remote()

        # Reset on the Isaac side
        response = await server.remote_reset.remote(
            self._sub_env_id, seed
        )
        all_images = response["images"]
        env_info = response["info"]

        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory = []

        # Build initial observation: camera 0 only
        target_desc = ""
        if "target_description" in env_info:
            target_desc = env_info["target_description"] + "\n"

        cam0_images = [all_images[0]] if len(all_images) > 0 else []
        obs_text = target_desc + init_observation_template(
            img_placeholder=self.config.image_placeholder,
        )
        obs = self._make_multi_image_obs(obs_text, cam0_images)
        return obs, env_info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Parse the LLM action and execute it.

        Supported actions:
        - **Query**:  ``{"query": [cam_id, ...]}`` — returns requested camera views.
        - **Place**:  ``{"x": INT, "y": INT, "z": INT}`` — places a brick, returns camera 0.
        - **Submit**: ``submit`` — ends the episode.

        Args:
            action_str: Raw LLM output string.

        Returns:
            ``(obs, reward, done, info)``
        """
        print(f"Received action string: {action_str}")  # Debug logging
        self.steps_taken += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        coordinate = parsed.get("coordinate")
        query_cameras = parsed.get("query_cameras")
        is_submit = parsed.get("is_submit", False)
        format_correct = parsed.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if query_cameras is not None:
            # ----------------------------------------------------------
            # Query action — return requested camera views
            # ----------------------------------------------------------
            # Validate camera IDs
            invalid_ids = [c for c in query_cameras if c >= self.config.n_cameras]
            if invalid_ids:
                # Some camera IDs out of range — treat as parse error
                server = await self._get_server()
                all_images = await server.render.remote(self._sub_env_id)
                cam0_images = [all_images[0]] if len(all_images) > 0 else []
                msg = (
                    f"Invalid camera ID(s): {invalid_ids}. "
                    f"Valid IDs are 0..{self.config.n_cameras - 1}."
                )
                obs = self._make_multi_image_obs(
                    action_template(
                        action_result=msg,
                        img_placeholder=self.config.image_placeholder,
                    ),
                    cam0_images,
                    action_str=action_str,
                )
                format_correct = False
                metrics["turn_metrics"]["action_is_valid"] = False
            else:
                server = await self._get_server()
                all_images = await server.render.remote(self._sub_env_id)
                selected_images = []
                for cam_id in query_cameras:
                    if cam_id < len(all_images):
                        selected_images.append(all_images[cam_id])
                    else:
                        selected_images.append(
                            Image.new("RGB", self.config.image_size, (0, 0, 0))
                        )

                img_phs = "\n".join(
                    self.config.image_placeholder for _ in query_cameras
                )
                obs_text = query_result_template(
                    camera_ids=query_cameras,
                    img_placeholders=img_phs,
                )
                obs = self._make_multi_image_obs(
                    obs_text, selected_images, action_str=action_str
                )
                metrics["turn_metrics"]["action_is_effective"] = True

        elif coordinate is not None:
            # ----------------------------------------------------------
            # Place action — execute placement, return camera 0
            # ----------------------------------------------------------
            goal = {
                "x": coordinate["x"],
                "y": coordinate["y"],
                "z": coordinate["z"],
            }
            server = await self._get_server()
            response = await server.remote_step.remote(self._sub_env_id, goal)

            all_images = response["images"]
            step_reward = response["reward"]
            step_done = response["done"]
            step_info = response["info"]

            reward += step_reward
            done = step_done
            metrics["turn_metrics"]["action_is_effective"] = True

            if step_info.get("success", False):
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True

            cam0_images = [all_images[0]] if len(all_images) > 0 else []
            obs_text = action_template(
                action_result=response["obs_str"],
                img_placeholder=self.config.image_placeholder,
            )
            obs = self._make_multi_image_obs(
                obs_text, cam0_images, action_str=action_str
            )
            info.update(step_info)

        elif is_submit:
            # ----------------------------------------------------------
            # Submit action — end the episode
            # ----------------------------------------------------------
            goal = {"type": "submit"}
            server = await self._get_server()
            response = await server.remote_step.remote(self._sub_env_id, goal)

            all_images = response["images"]
            step_reward = response["reward"]
            step_info = response["info"]

            reward += step_reward
            done = True  # Submit always ends the episode
            metrics["turn_metrics"]["action_is_effective"] = True

            if step_info.get("success", False):
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True

            cam0_images = [all_images[0]] if len(all_images) > 0 else []
            obs_text = action_template(
                action_result=response["obs_str"],
                img_placeholder=self.config.image_placeholder,
            )
            obs = self._make_multi_image_obs(
                obs_text, cam0_images, action_str=action_str
            )
            info.update(step_info)

        else:
            # ----------------------------------------------------------
            # Parse failure — show error with camera 0 view
            # ----------------------------------------------------------
            server = await self._get_server()
            all_images = await server.render.remote(self._sub_env_id)
            cam0_images = [all_images[0]] if len(all_images) > 0 else []

            msg = (
                "Could not parse your action. Valid formats:\n"
                f'  Query cameras: {{"query": [0, 1]}}  (IDs 0..{self.config.n_cameras - 1})\n'
                '  Place a brick: {"x": 2, "y": 3, "z": 0}\n'
                "  Submit: submit"
            )
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=msg,
                    img_placeholder=self.config.image_placeholder,
                ),
                cam0_images,
                action_str=action_str,
            )

        # Format reward for any correctly formatted action
        if format_correct:
            reward += self.config.format_reward

        # Step limit
        if self.steps_taken >= self.config.max_steps:
            done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        # Record step trajectory
        self.trajectory.append({
            "step_idx": self.steps_taken,
            "coordinate": coordinate,
            "query_cameras": query_cameras,
            "is_submit": is_submit,
            "reward": reward,
            "success": info["success"],
            "raw_action": parsed.get("action_content", ""),
            "thought": parsed.get("think_content", ""),
        })

        if done:
            info["trajectory"] = self.trajectory

        return obs, reward, done, info

    async def close(self) -> None:
        """Release the sub-env ID back to the global pool."""
        if self._sub_env_id is not None:
            try:
                server = await self._get_server()
                await server.release_env_id.remote(self._sub_env_id)
            except Exception as exc:
                logger.warning(
                    "Failed to release env_id=%d: %s", self._sub_env_id, exc
                )
            self._sub_env_id = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt string."""
        try:
            from .utils.prompt import get_checked_system_prompt
        except Exception:
            fmt = format_prompt(
                n_cameras=self.config.n_cameras,
                add_example=self.config.use_example_in_sys_prompt,
            )
            return system_prompt(n_cameras=self.config.n_cameras) + "\n" + fmt

        return get_checked_system_prompt(
            n_cameras=self.config.n_cameras,
            add_example=self.config.use_example_in_sys_prompt,
        )

    def _make_multi_image_obs(
        self,
        obs_str: str,
        images: List[Image.Image],
        action_str: str = "",
    ) -> Dict[str, Any]:
        """Wrap an observation string and a variable-length image list into
        the standard obs dict.

        Args:
            obs_str: Observation text (with ``<image>`` placeholders matching
                the number of *images*).
            images: Actual images to include in the observation.
            action_str: The previous VLM response (used to detect hallucinated
                vision tags and inject dummy images for alignment).
        """
        # Hallucination absorber: if the model hallucinated vision tags in its
        # response, inject dummy images so the tag count stays in sync.
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)

        if len(images) == 0 and hallucinated_tags == 0:
            return {"obs_str": obs_str}

        processed_images: List[Image.Image] = []

        # 1. Dummy images for hallucinated tags (align with previous response)
        for _ in range(hallucinated_tags):
            processed_images.append(
                Image.new("RGB", self.config.image_size, (0, 0, 0))
            )

        # 2. Actual images — resize if necessary
        for img in images:
            if img.size != self.config.image_size:
                img = img.resize(self.config.image_size, Image.Resampling.LANCZOS)
            processed_images.append(img)

        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: processed_images,
            },
        }