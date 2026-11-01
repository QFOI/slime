"""Host-owned physical evidence, evaluated only after agent termination."""

import json
import os
from pathlib import Path

state = json.loads(Path("/artifacts/evaluator-state.json").read_text())
events = [json.loads(line) for line in Path("/artifacts/device-events.jsonl").read_text().splitlines()]
hold = state["hold_samples"]
physical = (
    state["success"] is True
    and state["physical_commands"] > 0
    and len(hold) == 10
    and all(value is True for value in hold)
    and state["hold_duration_sec"] >= 0.5
)
depth_programs = sum(
    event["event"] == "python_end"
    and event["data"].get("status") == "completed"
    and any(
        a.get("units") == "m" and a.get("convention") == "optical_z" for a in event["data"].get("input_artifacts", [])
    )
    for event in events
)
expert_ok = os.environ["ROBOT_PROFILE"] != "expert" or depth_programs > 0
reward = int(physical and expert_ok)
Path("/logs/verifier/reward.json").write_text(
    json.dumps(
        {
            "reward": reward,
            "physical_success": int(physical),
            "depth_programs": depth_programs,
        }
    )
)
