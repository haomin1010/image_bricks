"""
Environment Coordinator — per-worker async bridge to the global IsaacSkillExecutor.

Each VAGEN ``AgentLoopWorker`` (Ray actor) holds **one** ``EnvCoordinator``
singleton.  It provides a thin async interface that
:class:`IsaacManagedEnv` instances call into.  Internally it forwards every
request as a Ray RPC to the global :class:`IsaacSkillExecutor` named actor.

The coordinator is **lazy-initialised**: the first ``IsaacManagedEnv`` that is
created in a worker process will trigger ``EnvCoordinator.get_or_create()``,
which discovers (or creates) the executor actor automatically.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import ray

from .isaac_skill_executor import EXECUTOR_ACTOR_NAME, IsaacSkillExecutor

logger = logging.getLogger(__name__)


class EnvCoordinator:
    """
    Lightweight per-worker coordinator.

    Responsibilities
    ----------------
    * Hold a reference to the global :class:`IsaacSkillExecutor` Ray Actor.
    * Proxy ``submit_goal`` / ``request_reset`` as async Ray RPCs.
    * Manage sub-env ID allocation on behalf of ``IsaacManagedEnv`` instances
      running in this worker.

    Thread safety
    -------------
    All public methods are ``async`` and should be called from the worker's
    ``asyncio`` event loop.  The singleton factory ``get_or_create`` uses a
    threading lock for the (rare) case where two coroutines race on first
    access.
    """

    # ---- Singleton management ----
    _instance: Optional["EnvCoordinator"] = None
    _init_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_create(cls, env_config: Dict[str, Any]) -> "EnvCoordinator":
        """Return the per-worker singleton, creating it if necessary.

        On first call the method also discovers (or creates) the global
        :class:`IsaacSkillExecutor` named Ray Actor.

        Args:
            env_config: Environment config dict.  Must contain
                ``num_total_envs`` when the executor needs to be created for
                the first time.

        Returns:
            The singleton ``EnvCoordinator`` instance for this worker.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._init_lock:
            # Double-check after acquiring lock
            if cls._instance is not None:
                return cls._instance

            executor_handle = cls._get_or_create_executor(env_config)
            coordinator = cls(executor_handle=executor_handle)
            cls._instance = coordinator
            logger.info("EnvCoordinator singleton created for this worker.")
            return coordinator

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton (useful for tests)."""
        cls._instance = None

    # ---- Construction ----

    def __init__(self, executor_handle: ray.actor.ActorHandle):
        self._executor = executor_handle

    # ------------------------------------------------------------------
    # Public async interface (called by IsaacManagedEnv)
    # ------------------------------------------------------------------

    async def allocate_env_id(self) -> int:
        """Request a free sub-env ID from the global executor.

        Returns:
            An integer sub-env ID.

        Raises:
            RuntimeError: If no free IDs are available.
        """
        env_id: int = await self._executor.allocate_env_id.remote()
        logger.debug("Allocated env_id=%d", env_id)
        return env_id

    async def release_env_id(self, env_id: int) -> None:
        """Return a sub-env ID to the global pool.

        Args:
            env_id: The ID previously obtained from :meth:`allocate_env_id`.
        """
        await self._executor.release_env_id.remote(env_id)
        logger.debug("Released env_id=%d", env_id)

    async def request_reset(
        self, env_id: int, seed: int
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Reset a sub-environment on the Isaac side.

        Args:
            env_id: Sub-environment index.
            seed: Random seed for the episode.

        Returns:
            ``(images, info)`` — same contract as the executor's
            ``request_reset``.
        """
        result = await self._executor.request_reset.remote(env_id, seed)
        return result

    async def request_step(
        self, env_id: int, goal: Dict[str, Any]
    ) -> Tuple[List[Any], float, bool, Dict[str, Any]]:
        """Submit a high-level goal and wait for the skill to complete.

        Args:
            env_id: Sub-environment index.
            goal: High-level goal dict, e.g. ``{"x": 1, "y": 2, "z": 0}``.

        Returns:
            ``(images, reward, done, info)`` — same contract as the
            executor's ``submit_goal``.
        """
        result = await self._executor.submit_goal.remote(env_id, goal)
        return result

    async def render(self, env_id: int) -> List[Any]:
        """Request a render of the current sub-env state.

        Args:
            env_id: Sub-environment index.

        Returns:
            A list of PIL images.
        """
        images = await self._executor.render.remote(env_id)
        return images

    # ------------------------------------------------------------------
    # Executor discovery / creation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_create_executor(
        env_config: Dict[str, Any],
    ) -> ray.actor.ActorHandle:
        """Discover the named executor actor, creating it if needed.

        Uses a double-check pattern to handle the race where multiple
        workers try to create the actor simultaneously.

        Args:
            env_config: Must contain ``num_total_envs`` if the executor
                needs to be created.

        Returns:
            A Ray actor handle for the :class:`IsaacSkillExecutor`.
        """
        try:
            handle = ray.get_actor(EXECUTOR_ACTOR_NAME)
            logger.info(
                "Found existing IsaacSkillExecutor actor '%s'.",
                EXECUTOR_ACTOR_NAME,
            )
            return handle
        except ValueError:
            pass  # Actor does not exist yet — try to create it.

        num_envs = int(env_config.get("num_total_envs", 64))
        logger.info(
            "Creating IsaacSkillExecutor actor '%s' with %d sub-envs …",
            EXECUTOR_ACTOR_NAME,
            num_envs,
        )

        try:
            handle = IsaacSkillExecutor.options(
                name=EXECUTOR_ACTOR_NAME,
                lifetime="detached",
            ).remote(num_envs=num_envs, env_config=env_config)
            return handle
        except ValueError:
            # Another worker won the race — just look it up.
            handle = ray.get_actor(EXECUTOR_ACTOR_NAME)
            logger.info(
                "Lost creation race; attached to existing '%s'.",
                EXECUTOR_ACTOR_NAME,
            )
            return handle

