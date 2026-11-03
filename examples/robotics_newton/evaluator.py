"""Episode scoring: the task condition plus process checks derived from the instruction.

v2 (default). For lift ("pick up the red cube with the gripper and hold it
above the table"), after the agent stops the evaluator runs a short carry
(up and sideways), then takes hold samples. Every sample must show the cube
above the table, touching both finger pads and nothing else with its centre
between the pads, and not interpenetrating anything. Across the samples, the
cube must stay within max_drift of its pose relative to the TCP at the start of
evaluation. Over the whole episode, the cube must not have fallen off the table
or been launched (thrown, or flung by the solver) faster than any controlled
motion moves it; falling does not count as a launch. Reach
keeps its pose condition, held for the full window. Every check is reported
separately; the reward is 1 only if all pass.

v1 is the Phase 1 rule (final height only, 10 samples, no carry), kept to
measure what v2 catches.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    name: str = "v2"
    hold_samples: int = 20
    carry: bool = True
    process_checks: bool = True
    max_drift_m: float = 0.01
    # Below the table top by more than this means the cube left the table.
    fell_below_z_m: float = -0.02
    # Controlled TCP motion peaks near 0.4 m/s; a non-falling cube far faster was thrown or glitched.
    max_launch_speed_mps: float = 1.0
    # Judged during the hold only: impacts penetrate briefly, a glitch-held cube stays inside.
    max_penetration_m: float = 0.005


V1 = EvaluatorConfig(name="v1", hold_samples=10, carry=False, process_checks=False)
V2 = EvaluatorConfig()
EVALUATORS = {"v1": V1, "v2": V2}


def score(task: str, evidence: dict, physical_commands: int, cfg: EvaluatorConfig = V2) -> dict:
    """evidence is a PandaWorlds snapshot taken after the evaluation completed."""
    hold = evidence["hold_samples"]
    checks = {
        "task_condition": len(hold) == cfg.hold_samples and all(hold),
        "physical_command": physical_commands > 0,
    }
    if task == "lift" and cfg.process_checks:
        grasp = evidence["grasp_samples"]
        checks["held_by_fingers"] = len(grasp) == cfg.hold_samples and all(grasp)
        if cfg.carry:
            checks["moved_with_gripper"] = evidence["max_drift"] <= cfg.max_drift_m
        checks["stayed_on_table"] = evidence["min_cube_z"] > cfg.fell_below_z_m
        checks["not_launched"] = evidence["max_cube_launch_speed"] <= cfg.max_launch_speed_mps
        checks["no_interpenetration"] = evidence["hold_penetration"] <= cfg.max_penetration_m
    metrics = {
        "hold_fraction": sum(hold) / cfg.hold_samples,
        "grasp_fraction": sum(evidence["grasp_samples"]) / cfg.hold_samples,
        "max_drift_m": evidence["max_drift"],
        "min_cube_z_m": evidence["min_cube_z"],
        "max_cube_launch_speed_mps": evidence["max_cube_launch_speed"],
        "hold_penetration_m": evidence["hold_penetration"],
        "episode_max_penetration_m": evidence["max_penetration"],
    }
    return {"reward": float(all(checks.values())), "checks": checks, "metrics": metrics, "evaluator": cfg.name}
