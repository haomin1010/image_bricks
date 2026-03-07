from __future__ import annotations

import json

from vagen.envs.isaac.reward_manager import IsaacRewardConfig, IsaacRewardManager
from vagen.envs.isaac.task_spec import BrickPosition, TaskSpec, load_task_spec
from vagen.envs.isaac.termination_manager import IsaacTerminationConfig, IsaacTerminationManager


def test_load_task_spec_from_convex_json(tmp_path):
    json_path = tmp_path / "0001.json"
    json_path.write_text(
        json.dumps(
            {
                "dimensions": {"length": 8, "width": 8, "height": 8},
                "blocks": [
                    {"id": 1, "x": 1, "y": 2, "z": 0},
                    {"id": 2, "x": 1, "y": 2, "z": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    task_spec = load_task_spec(json_path)

    assert task_spec.dimensions == (8, 8, 8)
    assert task_spec.total_blocks == 2
    assert BrickPosition(1, 2, 0) in task_spec.positions
    assert BrickPosition(1, 2, 1) in task_spec.positions


def test_reward_manager_distinguishes_candidate_floating_and_non_candidate():
    task_spec = TaskSpec(
        source_path=None,
        dimensions=(8, 8, 8),
        positions=frozenset(
            {
                BrickPosition(0, 0, 0),
                BrickPosition(0, 0, 1),
                BrickPosition(1, 0, 0),
            }
        ),
    )
    manager = IsaacRewardManager(
        IsaacRewardConfig(
            format_reward=0.1,
            correct_placement_reward=1.0,
            floating_placement_penalty=-10.0,
            non_candidate_penalty=-5.0,
        )
    )
    manager.reset(task_spec)

    floating = manager.evaluate_placement(BrickPosition(0, 0, 1))
    assert floating.outcome == "floating"
    assert floating.reward_delta == -10.0
    assert BrickPosition(0, 0, 1) not in manager.occupied_positions

    correct = manager.evaluate_placement(BrickPosition(0, 0, 0))
    assert correct.outcome == "correct_candidate"
    assert correct.reward_delta == 1.0
    assert BrickPosition(0, 0, 0) in manager.occupied_positions
    assert BrickPosition(0, 0, 1) in manager.candidate_positions()

    wrong = manager.evaluate_placement(BrickPosition(2, 2, 0))
    assert wrong.outcome == "non_candidate"
    assert wrong.reward_delta == -5.0
    assert BrickPosition(2, 2, 0) in manager.occupied_positions
    assert not manager.task_completed


def test_termination_manager_handles_attempt_limit_submit_and_mock_collapse():
    task_spec = TaskSpec(
        source_path=None,
        dimensions=(8, 8, 8),
        positions=frozenset({BrickPosition(0, 0, 0), BrickPosition(0, 0, 1)}),
    )

    limit_manager = IsaacTerminationManager(IsaacTerminationConfig(max_attempts_factor=1.5))
    limit_manager.reset(task_spec)
    assert limit_manager.max_attempts == 3

    status = limit_manager.evaluate(
        placement_attempted=True,
        submit_requested=False,
        isaac_info={},
        task_completed=False,
    )
    assert not status.done

    status = limit_manager.evaluate(
        placement_attempted=True,
        submit_requested=False,
        isaac_info={},
        task_completed=False,
    )
    assert not status.done

    status = limit_manager.evaluate(
        placement_attempted=True,
        submit_requested=False,
        isaac_info={},
        task_completed=False,
    )
    assert status.done
    assert status.reason == "max_attempts"

    submit_manager = IsaacTerminationManager(IsaacTerminationConfig(max_attempts_factor=1.5))
    submit_manager.reset(task_spec)
    submit_status = submit_manager.evaluate(
        placement_attempted=False,
        submit_requested=True,
        isaac_info={},
        task_completed=True,
    )
    assert submit_status.done
    assert submit_status.success
    assert submit_status.reason == "submit"

    collapse_manager = IsaacTerminationManager(
        IsaacTerminationConfig(max_attempts_factor=1.5, collapse_mock_after_attempt=2)
    )
    collapse_manager.reset(task_spec)
    first = collapse_manager.evaluate(
        placement_attempted=True,
        submit_requested=False,
        isaac_info={},
        task_completed=False,
    )
    second = collapse_manager.evaluate(
        placement_attempted=True,
        submit_requested=False,
        isaac_info={},
        task_completed=False,
    )
    assert not first.done
    assert second.done
    assert second.reason == "collapse"
