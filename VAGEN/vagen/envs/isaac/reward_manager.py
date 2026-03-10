from __future__ import annotations

from dataclasses import dataclass

from .task_spec import BrickPosition, TaskSpec


@dataclass(frozen=True)
class IsaacRewardConfig:
    format_reward: float = 0.1
    correct_placement_reward: float = 1.0
    floating_placement_penalty: float = -10.0
    non_candidate_penalty: float = -5.0


@dataclass(frozen=True)
class PlacementRewardResult:
    reward_delta: float
    outcome: str
    is_supported: bool
    is_candidate: bool
    already_occupied: bool
    state_changed: bool
    feedback: str


class IsaacRewardManager:
    """Rule-based reward evaluator for the brick stacking task."""

    def __init__(self, config: IsaacRewardConfig):
        self.config = config
        self._task_spec = TaskSpec.empty()
        self._occupied_positions: set[BrickPosition] = set()

    def reset(self, task_spec: TaskSpec) -> None:
        self._task_spec = task_spec
        self._occupied_positions = set()

    def format_reward(self, format_correct: bool) -> float:
        return float(self.config.format_reward) if format_correct else 0.0

    @property
    def occupied_positions(self) -> set[BrickPosition]:
        return set(self._occupied_positions)

    @property
    def target_positions(self) -> set[BrickPosition]:
        return set(self._task_spec.positions)

    @property
    def task_completed(self) -> bool:
        return self._occupied_positions == set(self._task_spec.positions)

    def remaining_target_positions(self) -> set[BrickPosition]:
        return set(self._task_spec.positions) - self._occupied_positions

    def candidate_positions(self) -> set[BrickPosition]:
        candidates: set[BrickPosition] = set()
        for position in self.remaining_target_positions():
            support = position.below()
            if support is None or support in self._occupied_positions:
                candidates.add(position)
        return candidates

    def evaluate_placement(self, position: BrickPosition) -> PlacementRewardResult:
        support = position.below()
        is_supported = support is None or support in self._occupied_positions
        already_occupied = position in self._occupied_positions
        candidates = self.candidate_positions()
        is_candidate = position in candidates

        if not is_supported:
            return PlacementRewardResult(
                reward_delta=float(self.config.floating_placement_penalty),
                outcome="floating",
                is_supported=False,
                is_candidate=False,
                already_occupied=already_occupied,
                state_changed=False,
                feedback="The brick is floating because the cell directly below is empty.",
            )

        if is_candidate:
            self._occupied_positions.add(position)
            return PlacementRewardResult(
                reward_delta=float(self.config.correct_placement_reward),
                outcome="correct_candidate",
                is_supported=True,
                is_candidate=True,
                already_occupied=already_occupied,
                state_changed=True,
                feedback="The brick is supported and matches one valid target candidate.",
            )

        state_changed = False
        if not already_occupied:
            self._occupied_positions.add(position)
            state_changed = True

        if already_occupied:
            feedback = "That cell is already occupied, so this placement cannot advance the target structure."
            outcome = "occupied_cell"
        else:
            feedback = "The brick is supported, but the position is not a valid target candidate."
            outcome = "non_candidate"

        return PlacementRewardResult(
            reward_delta=float(self.config.non_candidate_penalty),
            outcome=outcome,
            is_supported=True,
            is_candidate=False,
            already_occupied=already_occupied,
            state_changed=state_changed,
            feedback=feedback,
        )
