"""
Isaac Managed Environment - GymImageEnv proxy backed by the EnvManager system.

This environment supports two action types from the VLM:

1. **Query** - ``{"query": [cam_id, ...]}``
   Returns images from the requested cameras. No physics step.
2. **Place** - ``{"x": INT, "y": INT, "z": INT}``
   Places a brick and returns camera-0's image.
3. **Submit** - ``submit``
   Ends the episode.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PIL import Image

from ...gym_image_env import GymImageEnv
from ..reward_manager import IsaacRewardConfig, IsaacRewardManager
from ..task_spec import TaskSpec
from isaaclab_tasks.manager_based.manipulation.assembling.termination_manager import (
    IsaacTerminationConfig,
    IsaacTerminationManager,
)
from ..utils.utils import parse_response
from .actions import IsaacManagedEnvActionMixin, build_step_metrics
from .config import CONFIG_FIELDS, IsaacManagedEnvConfig
from .helpers import IsaacManagedEnvHelperMixin

logger = logging.getLogger(__name__)


class IsaacManagedEnv(IsaacManagedEnvActionMixin, IsaacManagedEnvHelperMixin, GymImageEnv):
    """
    GymImageEnv proxy backed by the IsaacEnvServer Ray Actor.
    Each agent loop creates one ``IsaacManagedEnv`` instance. On ``reset()``
    a sub-env ID is allocated from the global server; on ``close()`` it is
    automatically returned when the instance is destroyed or explicitly closed.
    """

    _server_handle = None
    _server_lock = asyncio.Lock()

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        known = {key: value for key, value in env_config.items() if key in CONFIG_FIELDS}
        self.config = IsaacManagedEnvConfig(**known)

        self._sub_env_id: int | None = None
        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory: list[dict[str, Any]] = []
        self._latest_scene_images: list[Image.Image] = []
        self._dataset_images_cache: list[Image.Image] = []
        self._current_task_spec: TaskSpec = TaskSpec.empty()
        self._current_ground_truth_path = None
        self._dataset_entries: list[dict[str, Any]] = []
        self._ground_truth_entries = []
        self._ground_truth_by_stem: dict[str, Any] = {}

        reward_config = IsaacRewardConfig(
            format_reward=self.config.format_reward,
            correct_placement_reward=self._get_correct_placement_reward(),
            floating_placement_penalty=self.config.floating_placement_penalty,
            non_candidate_penalty=self.config.non_candidate_penalty,
        )
        self.reward_manager = IsaacRewardManager(reward_config)
        self.termination_manager = IsaacTerminationManager(
            IsaacTerminationConfig(
                max_attempts_factor=self.config.max_attempts_factor,
                collapse_mock_after_attempt=self.config.collapse_mock_after_attempt,
            )
        )
        self._load_ground_truth_entries()

    async def system_prompt(self) -> dict[str, Any]:
        return {"obs_str": self._build_system_prompt()}

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        server = await self._get_server()

        if self._sub_env_id is None:
            self._sub_env_id = await server.allocate_env_id.remote()

        try:
            response = await asyncio.wait_for(
                server.remote_reset.remote(self._sub_env_id, seed),
                timeout=120.0,
            )
            env_info = response.get("info", {})
        except Exception as exc:
            logger.error("Failed to reset Isaac environment (timeout or crash): %s", exc)
            env_info = {}

        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory = []

        all_images = self._load_dataset_images(seed)
        self._dataset_images_cache = list(all_images)
        self._latest_scene_images = list(all_images)

        self._current_ground_truth_path = self._select_ground_truth_path(seed)
        self._current_task_spec = self._load_current_task_spec()
        self.reward_manager.reset(self._current_task_spec)
        self.termination_manager.reset(self._current_task_spec)

        target_desc = self._load_target_description(seed)
        env_info["target_description"] = target_desc

        logger.info(
            "reset: seed=%d dataset_index=%d gt=%s images=%d max_attempts=%d",
            seed,
            seed % max(len(self._dataset_entries), 1),
            self._current_ground_truth_path.name if self._current_ground_truth_path else "none",
            len(all_images),
            self.termination_manager.max_attempts,
        )

        return self._build_reset_observation(all_images, target_desc), env_info

    async def step(
        self,
        action_str: str,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.steps_taken += 1
        parsed = parse_response(action_str)

        coordinate = parsed.get("coordinate")
        is_submit = parsed.get("is_submit", False)
        format_correct = parsed.get("format_correct", False)
        query_cameras = parsed.get("query_cameras")

        reward = self.reward_manager.format_reward(format_correct)
        done = False
        info: dict[str, Any] = dict(parsed)
        metrics = build_step_metrics(format_correct)

        isaac_info: dict[str, Any] = {}
        placement_result = None
        termination_status = None

        if coordinate is not None:
            result = await self._handle_place_action(
                action_str=action_str,
                coordinate=coordinate,
                metrics=metrics,
            )
        elif is_submit:
            result = await self._handle_submit_action(
                action_str=action_str,
                metrics=metrics,
            )
        elif query_cameras is not None:
            result = await self._handle_query_action(
                action_str=action_str,
                query_cameras=query_cameras,
            )
        else:
            result = self._handle_invalid_action(action_str=action_str)

        obs = result.obs
        reward += result.reward
        done = result.done
        info.update(result.info_updates)
        isaac_info = result.isaac_info
        placement_result = result.placement_result
        termination_status = result.termination_status

        done, termination_status = self._apply_max_steps_guard(
            done=done,
            termination_status=termination_status,
            isaac_info=isaac_info,
            metrics=metrics,
        )
        termination_status = self._ensure_termination_status(
            termination_status=termination_status,
            isaac_info=isaac_info,
        )

        self._finalize_step_info(
            info=info,
            reward=reward,
            format_correct=format_correct,
            placement_result=placement_result,
            termination_status=termination_status,
            metrics=metrics,
        )
        self._record_trajectory_step(
            parsed=parsed,
            coordinate=coordinate,
            reward=reward,
            info=info,
            query_cameras=query_cameras,
            is_submit=is_submit,
            placement_result=placement_result,
            termination_status=termination_status,
        )

        if done:
            info["trajectory"] = self.trajectory

        return obs, reward, done, info
