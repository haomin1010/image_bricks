from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .task_spec import TaskSpec


@dataclass(frozen=True)
class IsaacTerminationConfig:
    max_attempts_factor: float = 1.5
    collapse_mock_after_attempt: int = -1



@dataclass(frozen=True)
class TerminationStatus:
    done: bool
    reason: str | None
    success: bool
    collapsed: bool
    submitted: bool
    reached_max_attempts: bool
    task_completed: bool
    placement_attempts: int
    max_attempts: int


class IsaacTerminationManager:
    """Episode termination rules for the brick stacking task."""

    COLLAPSE_KEYS = ("collapsed", "collapse", "has_collapsed", "tower_collapsed")

    def __init__(self, config: IsaacTerminationConfig):
        self.config = config
        self._task_spec = TaskSpec.empty()
        self._placement_attempts = 0
        self._max_attempts = 1

    def reset(self, task_spec: TaskSpec) -> None:
        self._task_spec = task_spec
        self._placement_attempts = 0
        self._max_attempts = max(
            1,
            int(math.ceil(max(1, task_spec.total_blocks) * float(self.config.max_attempts_factor))),
        )

    @property
    def placement_attempts(self) -> int:
        return int(self._placement_attempts)

    @property
    def max_attempts(self) -> int:
        return int(self._max_attempts)

    def evaluate(
        self,
        *,
        placement_attempted: bool,
        submit_requested: bool,
        isaac_info: dict[str, Any] | None,
        task_completed: bool,
        external_done: bool = False,
    ) -> TerminationStatus:
        if placement_attempted:
            self._placement_attempts += 1

        collapsed = self._detect_collapse(isaac_info)
        reached_max_attempts = self._placement_attempts >= self._max_attempts
        success = bool(submit_requested and task_completed and not collapsed)

        if collapsed:
            return self._make_status(
                done=True,
                reason="collapse",
                success=False,
                collapsed=True,
                submitted=submit_requested,
                reached_max_attempts=reached_max_attempts,
                task_completed=task_completed,
            )

        if submit_requested:
            return self._make_status(
                done=True,
                reason="submit",
                success=success,
                collapsed=False,
                submitted=True,
                reached_max_attempts=reached_max_attempts,
                task_completed=task_completed,
            )

        if reached_max_attempts:
            return self._make_status(
                done=True,
                reason="max_attempts",
                success=False,
                collapsed=False,
                submitted=False,
                reached_max_attempts=True,
                task_completed=task_completed,
            )

        if external_done:
            return self._make_status(
                done=True,
                reason="isaac_done",
                success=False,
                collapsed=False,
                submitted=False,
                reached_max_attempts=False,
                task_completed=task_completed,
            )

        return self._make_status(
            done=False,
            reason=None,
            success=False,
            collapsed=False,
            submitted=False,
            reached_max_attempts=False,
            task_completed=task_completed,
        )

    def _detect_collapse(self, isaac_info: dict[str, Any] | None) -> bool:
        if isinstance(isaac_info, dict):
            for key in self.COLLAPSE_KEYS:
                if bool(isaac_info.get(key, False)):
                    return True

        mock_after_attempt = int(self.config.collapse_mock_after_attempt)
        if mock_after_attempt > 0 and self._placement_attempts >= mock_after_attempt:
            return True
        return False

    def _make_status(
        self,
        *,
        done: bool,
        reason: str | None,
        success: bool,
        collapsed: bool,
        submitted: bool,
        reached_max_attempts: bool,
        task_completed: bool,
    ) -> TerminationStatus:
        return TerminationStatus(
            done=done,
            reason=reason,
            success=success,
            collapsed=collapsed,
            submitted=submitted,
            reached_max_attempts=reached_max_attempts,
            task_completed=task_completed,
            placement_attempts=int(self._placement_attempts),
            max_attempts=int(self._max_attempts),
        )
