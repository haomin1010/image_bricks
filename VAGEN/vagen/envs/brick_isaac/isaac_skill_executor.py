"""
Isaac Skill Executor — global singleton that drives the Isaac DirectVectorEnv.

This Ray Actor owns the Isaac DirectVectorEnv and translates high-level goals
(e.g. "place brick at (x, y, z)") into low-level multi-step action sequences.
It is the **only** component that directly communicates with Isaac.

All methods below are stubs returning mock/dummy data.  Replace them with real
Isaac logic when the simulator is ready.

Lifecycle
---------
1. Created once as a **named Ray Actor** (``name="isaac_skill_executor"``).
2. Workers discover it via ``ray.get_actor("isaac_skill_executor")``.
3. Workers call ``submit_goal`` / ``request_reset`` via Ray RPC.
4. Shutdown is explicit via ``shutdown()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import ray
from PIL import Image

logger = logging.getLogger(__name__)

# Default actor name used for discovery
EXECUTOR_ACTOR_NAME = "isaac_skill_executor"


@ray.remote(num_gpus=1, max_concurrency=1000)
class IsaacSkillExecutor:
    """
    Global Ray Actor that manages a single Isaac DirectVectorEnv instance.

    The executor maintains a pool of sub-environment IDs.  Workers request
    IDs via :meth:`allocate_env_id` and return them via :meth:`release_env_id`.

    High-level goals are submitted through :meth:`submit_goal`, which
    internally executes the multi-step skill on the Isaac side and returns
    the result once the skill completes.

    .. note::

        All methods in this class are **stubs**.  They return mock data so
        that the VAGEN side can be tested end-to-end without a real Isaac
        environment.  Replace the method bodies with actual Isaac SDK calls
        when the simulator is ready.
    """

    def __init__(self, num_envs: int, env_config: Dict[str, Any]):
        """
        Initialise the executor.

        Args:
            num_envs: Total number of sub-environments managed by this
                executor (i.e. the size of the DirectVectorEnv).
            env_config: Configuration dict forwarded to the Isaac env
                constructor.  Currently unused in the stub.
        """
        self.num_envs = num_envs
        self.env_config = env_config

        # --- Sub-env ID pool (thread-safe) ---
        self._id_lock = threading.Lock()
        self._free_ids: List[int] = list(range(num_envs))
        self._allocated_ids: set = set()

        # --- Isaac environment placeholder ---
        # TODO: Replace with real Isaac DirectVectorEnv initialisation.
        #   e.g. self._isaac_env = IsaacDirectVectorEnv(num_envs, **env_config)
        self._isaac_env = None

        # Image defaults (used by stubs)
        self._n_views: int = env_config.get("n_views", 3)
        self._image_size: Tuple[int, int] = tuple(
            env_config.get("image_size", (256, 256))
        )

        logger.info(
            "IsaacSkillExecutor initialised with %d sub-envs (stub mode)",
            num_envs,
        )

    # ------------------------------------------------------------------
    # Sub-env ID allocation
    # ------------------------------------------------------------------

    def allocate_env_id(self) -> int:
        """Allocate a free sub-environment ID.

        Returns:
            An integer env ID that the caller can use with
            :meth:`submit_goal` and :meth:`request_reset`.

        Raises:
            RuntimeError: If no free IDs are available.
        """
        with self._id_lock:
            if not self._free_ids:
                raise RuntimeError(
                    f"No free sub-env IDs.  All {self.num_envs} are allocated."
                )
            env_id = self._free_ids.pop(0)
            self._allocated_ids.add(env_id)
            return env_id

    def release_env_id(self, env_id: int) -> None:
        """Return a sub-environment ID to the free pool.

        Args:
            env_id: The ID previously obtained from :meth:`allocate_env_id`.
        """
        with self._id_lock:
            if env_id in self._allocated_ids:
                self._allocated_ids.discard(env_id)
                self._free_ids.append(env_id)
                logger.debug("Released env_id=%d", env_id)
            else:
                logger.warning(
                    "release_env_id called with unknown/already-free id=%d",
                    env_id,
                )

    # ------------------------------------------------------------------
    # High-level goal interface (STUBS — fill with real Isaac logic)
    # ------------------------------------------------------------------

    def submit_goal(
        self, env_id: int, goal: Dict[str, Any]
    ) -> Tuple[List[Any], float, bool, Dict[str, Any]]:
        """
        Execute a high-level skill for the given sub-env.

        This method blocks (from the caller's perspective it is an async
        Ray RPC) until the skill finishes, then returns the result.

        Args:
            env_id: Sub-environment index in ``[0, num_envs)``.
            goal: High-level goal dict.  For brick placement this is
                ``{"x": int, "y": int, "z": int}``.

        Returns:
            A 4-tuple ``(images, reward, done, info)`` where:

            - *images*: ``List[PIL.Image.Image]`` of length ``n_views``.
            - *reward*: Scalar float reward for this high-level step.
            - *done*: Whether the episode has ended.
            - *info*: Auxiliary metadata dict.

        .. note::

            **STUB** — returns dummy grey images, zero reward, done=False.
            Replace with:
            1. Translate *goal* into a sequence of low-level actions.
            2. Step the Isaac env repeatedly until the skill finishes.
            3. Render images and compute reward.
        """
        # TODO: Implement real skill execution.
        #
        # Pseudo-code:
        # ---------------------------------------------------------------
        # skill = self._create_skill(env_id, goal)
        # while not skill.is_done():
        #     low_level_actions = self._compute_all_actions()  # for all sub-envs
        #     obs_batch = self._isaac_env.step(low_level_actions)
        #     skill.update(obs_batch[env_id])
        # images = self._render(env_id)
        # reward = skill.compute_reward()
        # done = self._check_episode_done(env_id)
        # info = skill.get_info()
        # return images, reward, done, info
        # ---------------------------------------------------------------

        logger.info(
            "IsaacSkillExecutor.submit_goal(env_id=%d, goal=%s) [STUB]",
            env_id,
            goal,
        )

        images = self._make_dummy_images()
        reward = 0.0
        done = False
        info: Dict[str, Any] = {
            "placed": (goal.get("x", 0), goal.get("y", 0), goal.get("z", 0)),
            "stub": True,
        }
        return images, reward, done, info

    def request_reset(
        self, env_id: int, seed: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Reset the specified sub-environment.

        Args:
            env_id: Sub-environment index.
            seed: Random seed that determines the target / scenario.

        Returns:
            A 2-tuple ``(images, info)`` where:

            - *images*: ``List[PIL.Image.Image]`` initial observation views.
            - *info*: Metadata dict (e.g. target shape description).

        .. note::

            **STUB** — returns dummy images and minimal info.
            Replace with:
            1. Reset the Isaac sub-env (possibly via ``env_ids`` mask).
            2. Render initial images.
            3. Collect metadata.
        """
        # TODO: Implement real Isaac reset for a single sub-env.
        #
        # Pseudo-code:
        # ---------------------------------------------------------------
        # self._isaac_env.reset(env_ids=[env_id], seeds=[seed])
        # images = self._render(env_id)
        # info = {"seed": seed, "target_name": ..., "target_cells": ...}
        # return images, info
        # ---------------------------------------------------------------

        logger.info(
            "IsaacSkillExecutor.request_reset(env_id=%d, seed=%d) [STUB]",
            env_id,
            seed,
        )

        images = self._make_dummy_images()
        info: Dict[str, Any] = {"seed": seed, "stub": True}
        return images, info

    def render(self, env_id: int) -> List[Any]:
        """
        Render the current state of a sub-environment.

        Args:
            env_id: Sub-environment index.

        Returns:
            A list of ``n_views`` PIL images.

        .. note::

            **STUB** — returns grey dummy images.
        """
        # TODO: Replace with real Isaac rendering.
        return self._make_dummy_images()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all resources held by the executor.

        .. note::

            **STUB** — no-op.  Replace with Isaac env teardown.
        """
        # TODO: self._isaac_env.close() or equivalent.
        logger.info("IsaacSkillExecutor.shutdown() [STUB]")

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary (useful for debugging).

        Returns:
            Dict with num_envs, allocated count, and free count.
        """
        with self._id_lock:
            return {
                "num_envs": self.num_envs,
                "allocated": len(self._allocated_ids),
                "free": len(self._free_ids),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_dummy_images(self) -> List[Image.Image]:
        """Generate placeholder grey images."""
        return [
            Image.new("RGB", self._image_size, color=(180, 180, 180))
            for _ in range(self._n_views)
        ]

