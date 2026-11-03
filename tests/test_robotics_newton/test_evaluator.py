"""Reward rule of the Newton robot evaluator: every check is necessary."""

import pytest

from examples.robotics_newton.evaluator import V1, V2, score


def evidence(samples=20, **overrides):
    base = {
        "hold_samples": [True] * samples,
        "grasp_samples": [True] * samples,
        "max_drift": 0.0001,
        "hold_penetration": 0.0005,
        "min_cube_z": 0.025,
        "max_cube_launch_speed": 0.34,
        "max_penetration": 0.0005,
    }
    return {**base, **overrides}


def test_clean_grasp_scores_one():
    result = score("lift", evidence(), physical_commands=5)
    assert result["reward"] == 1.0 and all(result["checks"].values())


@pytest.mark.parametrize(
    "overrides, physical_commands, failed",
    [
        ({"hold_samples": [True] * 19 + [False]}, 5, "task_condition"),
        ({"hold_samples": [True] * 10}, 5, "task_condition"),  # a truncated hold window
        ({}, 0, "physical_command"),
        ({"grasp_samples": [True] * 19 + [False]}, 5, "held_by_fingers"),
        ({"max_drift": 0.02}, 5, "moved_with_gripper"),
        ({"min_cube_z": -0.7}, 5, "stayed_on_table"),
        ({"max_cube_launch_speed": 5.0}, 5, "not_launched"),
        ({"hold_penetration": 0.02}, 5, "no_interpenetration"),
    ],
)
def test_each_check_can_veto(overrides, physical_commands, failed):
    result = score("lift", evidence(**overrides), physical_commands)
    assert result["reward"] == 0.0
    assert [name for name, ok in result["checks"].items() if not ok] == [failed]


def test_transient_impact_penetration_is_not_a_violation():
    assert score("lift", evidence(max_penetration=0.02), physical_commands=5)["reward"] == 1.0


def test_v1_is_final_height_only():
    lodged = evidence(samples=10, grasp_samples=[False] * 10, hold_penetration=0.03)
    assert score("lift", lodged, 1, V1)["reward"] == 1.0
    assert score("lift", evidence(grasp_samples=[False] * 20, hold_penetration=0.03), 1, V2)["reward"] == 0.0


def test_reach_has_no_cube_checks():
    result = score("reach_pose", evidence(grasp_samples=[False] * 20, min_cube_z=-0.7), physical_commands=1)
    assert result["reward"] == 1.0 and set(result["checks"]) == {"task_condition", "physical_command"}
