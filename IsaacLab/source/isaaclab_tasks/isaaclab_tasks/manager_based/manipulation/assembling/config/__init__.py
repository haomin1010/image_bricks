"""Unified config management interface for assembling environments."""

from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.manager_based.manipulation.assembling.assembling_env_cfg import AssemblingEnvCfg

from .assembling_joint_pos_env_cfg import ASSEMBLING_TASK_ID
from .assembling_joint_pos_env_cfg import AssemblingCubeStackEnvCfg

DEFAULT_ENV_CFG_ENTRY_POINT = f"{__name__}.assembling_joint_pos_env_cfg:AssemblingCubeStackEnvCfg"
SUPPORTED_TASK_IDS = (ASSEMBLING_TASK_ID,)


def resolve_task_id(task_name: str | None = None) -> str:
    if task_name is None or task_name in SUPPORTED_TASK_IDS:
        return ASSEMBLING_TASK_ID if task_name is None else task_name
    print(
        f"[WARN]: Unknown assembling task_name='{task_name}'. "
        f"Falling back to '{ASSEMBLING_TASK_ID}'."
    )
    return ASSEMBLING_TASK_ID


def list_supported_task_ids() -> tuple[str, ...]:
    return SUPPORTED_TASK_IDS


def build_env_cfg(*, task_name: str | None = None, cube_size: float = 0.045) -> AssemblingEnvCfg:
    _ = resolve_task_id(task_name)
    _ = cube_size
    return AssemblingCubeStackEnvCfg()


for task_id in SUPPORTED_TASK_IDS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DEFAULT_ENV_CFG_ENTRY_POINT},
        disable_env_checker=True,
    )


__all__ = [
    "ASSEMBLING_TASK_ID",
    "DEFAULT_ENV_CFG_ENTRY_POINT",
    "SUPPORTED_TASK_IDS",
    "build_env_cfg",
    "list_supported_task_ids",
    "resolve_task_id",
]
