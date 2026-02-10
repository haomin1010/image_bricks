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
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..gym_image_env import GymImageEnv
from .env_coordinator import EnvCoordinator
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
    image_size: Tuple[int, int] = (256, 256)

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
    GymImageEnv proxy backed by the :class:`EnvCoordinator` / :class:`IsaacSkillExecutor`
    architecture.

    Each agent loop creates one ``IsaacManagedEnv`` instance.  On ``reset()``
    a sub-env ID is allocated from the global executor; on ``close()`` it is
    returned.

    The coordinator singleton is lazily created on first use (see
    :meth:`EnvCoordinator.get_or_create`).
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter to known config fields
        known = {k: v for k, v in env_config.items() if k in _CONFIG_FIELDS}
        self.config = IsaacManagedEnvConfig(**known)

        # Lazily obtain the per-worker coordinator singleton.
        # This also triggers executor actor creation if needed.
        self._coordinator: EnvCoordinator = EnvCoordinator.get_or_create(
            env_config
        )

        # Episode state — set on reset()
        self._sub_env_id: Optional[int] = None
        self.total_reward: float = 0.0
        self.steps_taken: int = 0

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        """Return the system-level prompt observation."""
        return {"obs_str": self._build_system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Allocate a sub-env ID (on first call) and reset it on the Isaac side.

        Args:
            seed: Random seed for episode generation.

        Returns:
            ``(obs, info)`` — initial multi-view observation and metadata.
        """
        # Allocate once; reuse the same sub-env ID across resets
        if self._sub_env_id is None:
            self._sub_env_id = await self._coordinator.allocate_env_id()

        # Reset on the Isaac side
        images, env_info = await self._coordinator.request_reset(
            self._sub_env_id, seed
        )

        self.total_reward = 0.0
        self.steps_taken = 0

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
        self.steps_taken += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        coordinate = parsed.get("coordinate")
        format_correct = parsed.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and coordinate is not None,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if coordinate is None:
            # Parse failure — render current state and show error
            images = await self._coordinator.render(self._sub_env_id)
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=(
                        "Could not parse your action. "
                        'Please output valid JSON: {"x": INT, "y": INT, "z": INT}.'
                    ),
                    img_placeholders=self._img_placeholders(),
                ),
                images,
            )
        else:
            # Submit high-level goal to the executor
            goal = {
                "x": coordinate["x"],
                "y": coordinate["y"],
                "z": coordinate["z"],
            }
            images, step_reward, step_done, step_info = (
                await self._coordinator.request_step(self._sub_env_id, goal)
            )

            reward += step_reward
            done = step_done
            metrics["turn_metrics"]["action_is_effective"] = True

            if step_info.get("success", False):
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True

            # Build human-readable result description
            x, y, z = coordinate["x"], coordinate["y"], coordinate["z"]
            progress = step_info.get("progress", None)
            result_parts = [f"Brick placed at ({x}, {y}, {z})."]
            if step_info.get("correct", None) is True:
                result_parts.append("Correct position!")
            elif step_info.get("correct", None) is False:
                result_parts.append("Wrong position.")
            if step_info.get("duplicate", False):
                result_parts.append("(Duplicate — already placed here.)")
            if step_info.get("out_of_bounds", False):
                result_parts.append("(Out of bounds!)")
            if progress is not None:
                result_parts.append(f"Progress: {progress:.0%}")

            result_str = " ".join(result_parts)
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=result_str,
                    img_placeholders=self._img_placeholders(),
                ),
                images,
            )
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

        return obs, reward, done, info

    async def close(self) -> None:
        """Release the sub-env ID back to the global pool."""
        if self._sub_env_id is not None:
            try:
                await self._coordinator.release_env_id(self._sub_env_id)
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
        fmt = format_prompt(add_example=self.config.use_example_in_sys_prompt)
        return system_prompt() + "\n" + fmt

    def _img_placeholders(self) -> str:
        """Return n image placeholder tokens separated by newlines."""
        return "\n".join(
            self.config.image_placeholder for _ in range(self.config.n_views)
        )

    def _make_multi_image_obs(
        self, obs_str: str, images: List[Image.Image]
    ) -> Dict[str, Any]:
        """Wrap an observation string and images into the standard obs dict."""
        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: images,
            },
        }

