from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..reward_manager import PlacementRewardResult
from ..task_spec import BrickPosition
from isaaclab_tasks.manager_based.manipulation.assembling.termination_manager import TerminationStatus

logger = logging.getLogger(__name__)


def build_step_metrics(format_correct: bool) -> dict[str, Any]:
    return {
        "turn_metrics": {
            "action_is_valid": format_correct,
            "action_is_effective": False,
        },
        "traj_metrics": {
            "success": False,
            "termination_reason": None,
            "task_completed": False,
            "collapse_detected": False,
        },
    }


@dataclass
class ActionExecutionResult:
    obs: dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info_updates: dict[str, Any] = field(default_factory=dict)
    isaac_info: dict[str, Any] = field(default_factory=dict)
    placement_result: PlacementRewardResult | None = None
    termination_status: TerminationStatus | None = None


class IsaacManagedEnvActionMixin:
    async def _execute_server_goal(
        self,
        goal: dict[str, Any],
        *,
        timeout: float = 300.0,
        error_context: str,
    ) -> tuple[dict[str, Any], bool]:
        server = await self._get_server()
        try:
            step_response = await asyncio.wait_for(
                server.remote_step.remote(self._sub_env_id, goal),
                timeout=timeout,
            )
            isaac_info = dict(step_response.get("info", {}) or {})
            external_done = bool(step_response.get("done", False))
            return isaac_info, external_done
        except Exception as exc:
            logger.error("%s: %s", error_context, exc)
            return {"timeout": True}, True

    async def _handle_place_action(
        self,
        *,
        action_str: str,
        coordinate: dict[str, int],
        metrics: dict[str, Any],
    ) -> ActionExecutionResult:
        goal = {"x": coordinate["x"], "y": coordinate["y"], "z": coordinate["z"]}
        isaac_info, external_done = await self._execute_server_goal(
            goal,
            error_context="Failed to step Isaac (timeout or crash)",
        )

        placement_result = self.reward_manager.evaluate_placement(BrickPosition.from_mapping(goal))
        termination_status = self.termination_manager.evaluate(
            placement_attempted=True,
            submit_requested=False,
            isaac_info=isaac_info,
            task_completed=self.reward_manager.task_completed,
            external_done=external_done,
        )
        isaac_images = await self._render_env_images()
        obs_text = self._build_placement_obs_text(coordinate, placement_result, termination_status)

        metrics["turn_metrics"]["action_is_effective"] = True
        metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
        metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
        metrics["traj_metrics"]["termination_reason"] = termination_status.reason

        return ActionExecutionResult(
            obs=self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str),
            reward=placement_result.reward_delta,
            done=termination_status.done,
            info_updates={
                "timeout": bool(isaac_info.get("timeout", False)),
                "placement_outcome": placement_result.outcome,
                "placement_feedback": placement_result.feedback,
            },
            isaac_info=isaac_info,
            placement_result=placement_result,
            termination_status=termination_status,
        )

    async def _handle_submit_action(
        self,
        *,
        action_str: str,
        metrics: dict[str, Any],
    ) -> ActionExecutionResult:
        isaac_info, external_done = await self._execute_server_goal(
            {"type": "submit"},
            error_context="Failed to submit Isaac (timeout or crash)",
        )

        termination_status = self.termination_manager.evaluate(
            placement_attempted=False,
            submit_requested=True,
            isaac_info=isaac_info,
            task_completed=self.reward_manager.task_completed,
            external_done=external_done,
        )
        isaac_images = await self._render_env_images()
        obs_text = self._build_submit_obs_text(termination_status)

        metrics["turn_metrics"]["action_is_effective"] = True
        metrics["traj_metrics"]["success"] = termination_status.success
        metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
        metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
        metrics["traj_metrics"]["termination_reason"] = termination_status.reason

        return ActionExecutionResult(
            obs=self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str),
            done=termination_status.done,
            info_updates={"timeout": bool(isaac_info.get("timeout", False))},
            isaac_info=isaac_info,
            termination_status=termination_status,
        )

    async def _handle_query_action(
        self,
        *,
        action_str: str,
        query_cameras: list[int],
    ) -> ActionExecutionResult:
        scene_images = await self._render_env_images()
        selected_ids = [cam_id for cam_id in query_cameras if 0 <= cam_id < len(scene_images)]
        selected_images = [scene_images[cam_id] for cam_id in selected_ids]
        if not selected_images:
            selected_images = self._make_fallback_images(count=1, color=(30, 30, 30))
            selected_ids = []

        label_lines = [
            f"Camera {cam_id}: {self.config.image_placeholder}" for cam_id in selected_ids
        ] or [f"Camera unavailable: {self.config.image_placeholder}"]
        obs_text = (
            "[System]: Query result.\n"
            + "\n".join(label_lines)
            + "\nYou may place the next cube or submit when done."
        )
        return ActionExecutionResult(
            obs=self._make_multi_image_obs(obs_text, selected_images, action_str=action_str),
        )

    def _handle_invalid_action(self, *, action_str: str) -> ActionExecutionResult:
        return ActionExecutionResult(obs=self._build_invalid_action_observation(action_str))

    def _apply_max_steps_guard(
        self,
        *,
        done: bool,
        termination_status: TerminationStatus | None,
        isaac_info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[bool, TerminationStatus | None]:
        if self.steps_taken < self.config.max_steps or done:
            return done, termination_status

        if termination_status is None:
            termination_status = self.termination_manager.evaluate(
                placement_attempted=False,
                submit_requested=False,
                isaac_info=isaac_info,
                task_completed=self.reward_manager.task_completed,
                external_done=False,
            )
        termination_status = TerminationStatus(
            done=True,
            reason="max_steps_guard",
            success=False,
            collapsed=termination_status.collapsed,
            submitted=termination_status.submitted,
            reached_max_attempts=termination_status.reached_max_attempts,
            task_completed=termination_status.task_completed,
            placement_attempts=termination_status.placement_attempts,
            max_attempts=termination_status.max_attempts,
        )
        metrics["traj_metrics"]["termination_reason"] = termination_status.reason
        return True, termination_status

    def _ensure_termination_status(
        self,
        *,
        termination_status: TerminationStatus | None,
        isaac_info: dict[str, Any],
    ) -> TerminationStatus:
        if termination_status is not None:
            return termination_status
        return self.termination_manager.evaluate(
            placement_attempted=False,
            submit_requested=False,
            isaac_info=isaac_info,
            task_completed=self.reward_manager.task_completed,
            external_done=False,
        )

    def _finalize_step_info(
        self,
        *,
        info: dict[str, Any],
        reward: float,
        format_correct: bool,
        placement_result: PlacementRewardResult | None,
        termination_status: TerminationStatus,
        metrics: dict[str, Any],
    ) -> None:
        metrics["traj_metrics"]["success"] = termination_status.success
        metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
        metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
        metrics["traj_metrics"]["termination_reason"] = termination_status.reason
        info["metrics"] = metrics
        info["success"] = termination_status.success
        info["termination"] = {
            "reason": termination_status.reason,
            "collapsed": termination_status.collapsed,
            "submitted": termination_status.submitted,
            "task_completed": termination_status.task_completed,
            "placement_attempts": termination_status.placement_attempts,
            "max_attempts": termination_status.max_attempts,
        }
        info["reward_breakdown"] = {
            "format_reward": self.reward_manager.format_reward(format_correct),
            "placement_reward": 0.0 if placement_result is None else placement_result.reward_delta,
        }
        self.total_reward += reward
        info["total_reward"] = self.total_reward
        info["remaining_target_blocks"] = len(self.reward_manager.remaining_target_positions())

    def _record_trajectory_step(
        self,
        *,
        parsed: dict[str, Any],
        coordinate: dict[str, int] | None,
        reward: float,
        info: dict[str, Any],
        query_cameras: list[int] | None,
        is_submit: bool,
        placement_result: PlacementRewardResult | None,
        termination_status: TerminationStatus,
    ) -> None:
        self.trajectory.append(
            {
                "step_idx": self.steps_taken,
                "coordinate": coordinate,
                "query_cameras": query_cameras,
                "is_submit": is_submit,
                "reward": reward,
                "success": info["success"],
                "raw_action": parsed.get("action_content", ""),
                "thought": parsed.get("think_content", ""),
                "placement_outcome": None if placement_result is None else placement_result.outcome,
                "termination_reason": termination_status.reason,
                "placement_attempts": termination_status.placement_attempts,
                "total_reward": self.total_reward,
            }
        )
