"""Termination terms for assembling tasks.

These terms consume runtime-provided tensors exposed on ``env.unwrapped`` by
the server runtime (e.g. ``TeleportRuntime.bind_shared_state``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TERM_REASON_NONE = 0
TERM_REASON_SUBMIT = 1
TERM_REASON_MAX_ATTEMPTS = 2
TERM_REASON_TELEPORT_FAILED = 3
TERM_REASON_ISAAC_DONE = 4
TERM_REASON_REPEAT_COORDINATE = 5


def server_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when runtime marks the current env as done."""
    signal = getattr(env.unwrapped, "_vagen_done_signal", None)
    if isinstance(signal, torch.Tensor):
        return signal.to(device=torch.device(env.device), dtype=torch.bool).clone()
    return torch.zeros((env.num_envs,), device=torch.device(env.device), dtype=torch.bool)


def server_done_by_reason(env: ManagerBasedRLEnv, reason_code: int) -> torch.Tensor:
    """Terminate only when runtime reports done with the target reason code."""
    done = server_done(env)
    reason = getattr(env.unwrapped, "_vagen_termination_reason_code", None)
    if not isinstance(reason, torch.Tensor):
        return torch.zeros((env.num_envs,), device=torch.device(env.device), dtype=torch.bool)
    reason_match = reason.to(device=torch.device(env.device), dtype=torch.long) == int(reason_code)
    return torch.logical_and(done, reason_match)


def done_submit(env: ManagerBasedRLEnv) -> torch.Tensor:
    return server_done_by_reason(env, TERM_REASON_SUBMIT)


def done_max_attempts(env: ManagerBasedRLEnv) -> torch.Tensor:
    return server_done_by_reason(env, TERM_REASON_MAX_ATTEMPTS)


def done_teleport_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return server_done_by_reason(env, TERM_REASON_TELEPORT_FAILED)


def done_isaac_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    return server_done_by_reason(env, TERM_REASON_ISAAC_DONE)


def done_repeat_coordinate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return server_done_by_reason(env, TERM_REASON_REPEAT_COORDINATE)


__all__ = [
    "TERM_REASON_NONE",
    "TERM_REASON_SUBMIT",
    "TERM_REASON_MAX_ATTEMPTS",
    "TERM_REASON_TELEPORT_FAILED",
    "TERM_REASON_ISAAC_DONE",
    "TERM_REASON_REPEAT_COORDINATE",
    "server_done",
    "server_done_by_reason",
    "done_submit",
    "done_max_attempts",
    "done_teleport_failed",
    "done_isaac_done",
    "done_repeat_coordinate",
]
