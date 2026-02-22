"""
Isaac Managed Environment — GymImageEnv proxy backed by the EnvManager system.
This environment is a drop-in replacement for :class:`BrickIsaac` in the VAGEN
agent loop.  Instead of holding a local simulator, it delegates all
reset / step / render calls to a shared :class:`IsaacSkillExecutor` via the
per-worker :class:`EnvCoordinator`.
From the agent loop's perspective this behaves identically to any other
``GymImageEnv``:  ``reset(seed)`` returns multi-view images + info, and
``step(action_str)`` parses the LLM output, executes the high-level goal
remotely, and returns the formatted observation.
Usage
-----
Register this class in the env_registry config::
    env_registry:
      brick_isaac: vagen.envs.brick_isaac.IsaacManagedEnv
The ``env_config`` dict in the dataset YAML should contain at least:
- ``num_total_envs`` (int): Total sub-envs in the Isaac DirectVectorEnv.
- Standard BrickIsaac fields (``n_views``, ``image_size``, ``max_steps``, …).
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

    # Observation
    n_views: int = 3
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
        """
        server = await self._get_server()

        # Allocate once; reuse the same sub-env ID across resets
        if self._sub_env_id is None:
            self._sub_env_id = await server.allocate_env_id.remote()

        # Reset on the Isaac side
        response = await server.remote_reset.remote(
            self._sub_env_id, seed
        )
        images = response["images"]
        env_info = response["info"]

        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory = []

        # Build initial observation text
        target_desc = ""
        if "target_description" in env_info:
            target_desc = env_info["target_description"] + "\n"

        obs_text = target_desc + init_observation_template(
            self._img_placeholders()
        )
        obs = self._make_multi_image_obs(obs_text, images)
        return obs, env_info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Parse the LLM's coordinate, execute the high-level goal remotely,
        and return the formatted observation.
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
        is_submit = parsed.get("is_submit", False)
        format_correct = parsed.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and (coordinate is not None or is_submit),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if coordinate is None and not is_submit:
            # Parse failure — render current state and show error
            server = await self._get_server()
            images = await server.render.remote(self._sub_env_id)
            # Prefer targeted feedback when think tag is missing
            if parsed.get("missing_think", False):
                msg = (
                    "Your response is missing the <think>...</think> reasoning tag. "
                    "Please include a short reasoning inside <think> and then the answer. "
                    "Example: <think>The bottom layer is missing at (2,3).</think>"
                    "<answer>{{\"x\": 2, \"y\": 3, \"z\": 0}}</answer>"
                )
            else:
                msg = (
                    "Could not parse your action. "
                    "Example: <think>The bottom layer is missing at (2,3).</think>"
                    "<answer>{{\"x\": 2, \"y\": 3, \"z\": 0}}</answer>"
                )

            obs = self._make_multi_image_obs(
                action_template(
                    action_result=msg,
                    img_placeholders=self._img_placeholders(),
                ),
                images,
                action_str=action_str,
            )
        elif is_submit:
            # Handle submission
            goal = {"type": "submit"}
            server = await self._get_server()
            response = await server.remote_step.remote(self._sub_env_id, goal)
            
            images = response["images"]
            step_reward = response["reward"]
            step_info = response["info"]

            reward += step_reward
            done = True # Submit always ends the episode
            metrics["turn_metrics"]["action_is_effective"] = True

            if step_info.get("success", False):
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True

            obs_text = action_template(
                action_result=response["obs_str"],
                img_placeholders=self._img_placeholders(),
            )
            obs = self._make_multi_image_obs(obs_text, images, action_str=action_str)
            info.update(step_info)
        else:
            # Submit high-level goal to the executor
            goal = {
                "x": coordinate["x"],
                "y": coordinate["y"],
                "z": coordinate["z"],
            }
            server = await self._get_server()
            response = await server.remote_step.remote(self._sub_env_id, goal)

            images = response["images"]
            step_reward = response["reward"]
            step_done = response["done"]
            step_info = response["info"]

            reward += step_reward
            # For non-submit high-level goals we treat the environment as continuing
            # (do not end the episode). The server signals step completion via
            # `step_done` and `step_info['success']`, but the episode should only
            # terminate on explicit submit actions. This prevents the LLM from
            # stopping planning after each successful low-level execution.
            # Exception: timeout terminates the episode like RL time-limit.
            done = True if step_info.get("timeout", False) else False
            metrics["turn_metrics"]["action_is_effective"] = True

            if step_info.get("success", False):
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True

            # Use server-generated description
            obs_text = action_template(
                action_result=response["obs_str"],
                img_placeholders=self._img_placeholders(),
            )
            obs = self._make_multi_image_obs(obs_text, images, action_str=action_str)
            info.update(step_info)

        # Format reward
        if format_correct and coordinate is not None:
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
            "is_submit": is_submit,
            "reward": reward,
            "success": info["success"],
            "raw_action": parsed.get("action_content", ""),
            "thought": parsed.get("think_content", "")
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
    # Helpers (same logic as BrickIsaac)
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt string."""
        # Use checked system prompt for Isaac environment: if the composed
        # prompt is malformed, return a concise corrective example so the
        # LLM replies in a parseable format instead of producing invalid text.
        try:
            from .utils.prompt import get_checked_system_prompt
        except Exception:
            # Fallback to original behavior on import error
            fmt = format_prompt(add_example=self.config.use_example_in_sys_prompt)
            return system_prompt() + "\n" + fmt

        return get_checked_system_prompt(add_example=self.config.use_example_in_sys_prompt)

    def _img_placeholders(self) -> str:
        """Return n image placeholder tokens separated by newlines."""
        # Only provide placeholders for the first few turns to avoid hitting 
        # token limits and causing desync between text and image list.
        # Max 5 turns of images = 15 images total.
        if self.steps_taken >= 6:
            return "(Latest images omitted to save context)"
            
        return "\n".join(
            self.config.image_placeholder for _ in range(self.config.n_views)
        )

    def _make_multi_image_obs(
        self, obs_str: str, images: List[Image.Image], action_str: str = ""
    ) -> Dict[str, Any]:
        """Wrap an observation string and images into the standard obs dict."""
        # Fundamental Fix: Hallucination Absorber
        # In multi-turn context, if the model hallucinates vision tags in its response,
        # the AgentLoop will store them in history, causing desync with the image list.
        # We detect tags in action_str and provide dummy images to keep the counts aligned.
        
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)
        
        target_count = self.config.n_views if self.steps_taken < 6 else 0
        total_needed = target_count + hallucinated_tags
        
        if total_needed == 0:
            return {"obs_str": obs_str}

        processed_images = []
        # 1. Add dummies for hallucinated tags first (to align with the previous turn's response)
        for _ in range(hallucinated_tags):
            processed_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            
        # 2. Add the actual new images
        if self.steps_taken < 6:
            for i in range(self.config.n_views):
                if i < len(images):
                    img = images[i]
                    if img.size != (224, 224):
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    processed_images.append(img)
                else:
                    processed_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: processed_images,
            },
        }
