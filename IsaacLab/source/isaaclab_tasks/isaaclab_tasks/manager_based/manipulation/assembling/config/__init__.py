"""Unified config management interface for assembling environments."""

from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg

from .franka_stack_env_cfg import FRANKA_STACK_TASK_ID
from .franka_stack_env_cfg import FrankaStackEnvCfg
from .teleport_stack_env_cfg import TELEPORT_STACK_TASK_ID
from .teleport_stack_env_cfg import TeleportStackEnvCfg

TASK_TO_ENV_CFG_ENTRY_POINT = {
    FRANKA_STACK_TASK_ID: f"{__name__}.franka_stack_env_cfg:FrankaStackEnvCfg",
    TELEPORT_STACK_TASK_ID: f"{__name__}.teleport_stack_env_cfg:TeleportStackEnvCfg",
}
DEFAULT_TASK_ID = FRANKA_STACK_TASK_ID
DEFAULT_ENV_CFG_ENTRY_POINT = TASK_TO_ENV_CFG_ENTRY_POINT[DEFAULT_TASK_ID]
SUPPORTED_TASK_IDS = tuple(TASK_TO_ENV_CFG_ENTRY_POINT.keys())


def resolve_task_id(task_name: str | None = None) -> str:
    if task_name is None or task_name in SUPPORTED_TASK_IDS:
        return DEFAULT_TASK_ID if task_name is None else task_name
    print(
        f"[WARN]: Unknown assembling task_name='{task_name}'. "
        f"Falling back to '{DEFAULT_TASK_ID}'."
    )
    return DEFAULT_TASK_ID


def list_supported_task_ids() -> tuple[str, ...]:
    return SUPPORTED_TASK_IDS


def build_env_cfg(*, task_name: str | None = None, cube_size: float = 0.0203 * 2.0) -> AssemblingEnvCfg:
    resolved_task = resolve_task_id(task_name)
    _ = cube_size
    if resolved_task == TELEPORT_STACK_TASK_ID:
        return TeleportStackEnvCfg()
    return FrankaStackEnvCfg()


for task_id in SUPPORTED_TASK_IDS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": TASK_TO_ENV_CFG_ENTRY_POINT[task_id]},
        disable_env_checker=True,
    )


__all__ = [
    "FRANKA_STACK_TASK_ID",
    "TELEPORT_STACK_TASK_ID",
    "TASK_TO_ENV_CFG_ENTRY_POINT",
    "DEFAULT_TASK_ID",
    "DEFAULT_ENV_CFG_ENTRY_POINT",
    "SUPPORTED_TASK_IDS",
    "FrankaStackEnvCfg",
    "TeleportStackEnvCfg",
    "build_env_cfg",
    "list_supported_task_ids",
    "resolve_task_id",
]
