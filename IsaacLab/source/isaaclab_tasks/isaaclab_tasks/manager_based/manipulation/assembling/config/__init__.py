"""Unified config management interface for assembling environments."""

from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg

from .franka_stack_env_cfg import FRANKA_STACK_TASK_ID
from .franka_stack_env_cfg import FrankaStackEnvCfg
from .teleport_stack_env_cfg import TeleportStackEnvCfg

DEFAULT_ENV_CFG_ENTRY_POINT = f"{__name__}.franka_stack_env_cfg:FrankaStackEnvCfg"
SUPPORTED_TASK_IDS = (FRANKA_STACK_TASK_ID,)


def resolve_task_id(task_name: str | None = None) -> str:
    if task_name is None or task_name in SUPPORTED_TASK_IDS:
        return FRANKA_STACK_TASK_ID if task_name is None else task_name
    print(
        f"[WARN]: Unknown assembling task_name='{task_name}'. "
        f"Falling back to '{FRANKA_STACK_TASK_ID}'."
    )
    return FRANKA_STACK_TASK_ID


def list_supported_task_ids() -> tuple[str, ...]:
    return SUPPORTED_TASK_IDS


def build_env_cfg(*, task_name: str | None = None, cube_size: float = 0.0203 * 2.0) -> AssemblingEnvCfg:
    _ = resolve_task_id(task_name)
    _ = cube_size
    return FrankaStackEnvCfg()


for task_id in SUPPORTED_TASK_IDS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DEFAULT_ENV_CFG_ENTRY_POINT},
        disable_env_checker=True,
    )


__all__ = [
    "FRANKA_STACK_TASK_ID",
    "DEFAULT_ENV_CFG_ENTRY_POINT",
    "SUPPORTED_TASK_IDS",
    "FrankaStackEnvCfg",
    "TeleportStackEnvCfg",
    "build_env_cfg",
    "list_supported_task_ids",
    "resolve_task_id",
]
