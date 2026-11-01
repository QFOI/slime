"""Small Harbor package reader. The task package is the sole data source.

JSONL rows used by slime contain references and a digest, never a second copy of
the simulator configuration or the verifier. Only single-step robot tasks are
supported; unsupported Harbor execution features fail explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import tomllib
from pydantic import BaseModel, ConfigDict, Field


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    driver: str = "robot_rollout.robosuite_driver:create_driver"
    task: str
    seed: int = 17
    command_budget: int = Field(default=80, gt=0)


@dataclass(frozen=True)
class HarborTask:
    path: Path
    instruction: str
    config: dict
    robot: RobotSpec
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> HarborTask:
        root = Path(path).expanduser().resolve(strict=True)
        for name in ("instruction.md", "task.toml", "environment/robot.json", "tests/test.sh"):
            item = root / name
            if not item.is_file() or not item.resolve().is_relative_to(root):
                raise ValueError(f"Missing or escaping Harbor task file: {name}")
        digest = hashlib.sha256()
        for item in sorted(root.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"Task packages must not contain symlinks: {item.relative_to(root)}")
            # Local imports, Finder and Git metadata must not invalidate an
            # otherwise identical dataset when it moves to a rollout worker.
            if {"__pycache__", ".git"} & set(item.relative_to(root).parts) or item.name == ".DS_Store":
                continue
            if item.is_file():
                digest.update(item.relative_to(root).as_posix().encode() + b"\0" + item.read_bytes() + b"\0")
        config = tomllib.loads((root / "task.toml").read_text())
        if config.get("steps"):
            raise ValueError("Robotics currently requires a single-step Harbor task")
        if config.get("metadata", {}).get("task_type") != "robotics":
            raise ValueError("Harbor task must declare metadata.task_type = 'robotics'")
        for phase, default in (("agent", 1200), ("verifier", 60)):
            seconds = config.get(phase, {}).get("timeout_sec", default)
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError(f"Invalid Harbor {phase}.timeout_sec")
        instruction = (root / "instruction.md").read_text().strip()
        if not instruction:
            raise ValueError("Empty Harbor instruction")
        return cls(
            root,
            instruction,
            config,
            RobotSpec.model_validate_json((root / "environment/robot.json").read_text()),
            digest.hexdigest(),
        )

    @property
    def name(self) -> str:
        return self.config.get("task", {}).get("name", self.path.name)

    def timeout(self, phase: str) -> float:
        return float(self.config.get(phase, {}).get("timeout_sec", 1200 if phase == "agent" else 60))

    def row(self) -> dict:
        return {
            "prompt": self.instruction,
            "metadata": {
                "task_type": "robotics",
                "task_path": str(self.path),
                "task_digest": self.digest,
            },
        }


def read_reward(directory: Path) -> dict[str, float]:
    """Harbor reward.json/reward.txt contract; missing/invalid != task failure."""
    if (directory / "reward.json").is_file():
        rewards = json.loads((directory / "reward.json").read_text())
    elif (directory / "reward.txt").is_file():
        rewards = {"reward": float((directory / "reward.txt").read_text().strip())}
    else:
        raise ValueError("Verifier produced neither reward.json nor reward.txt")
    if not isinstance(rewards, dict) or "reward" not in rewards:
        raise ValueError("Verifier must produce a reward mapping with a primary 'reward'")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rewards.values()):
        raise ValueError("Harbor rewards must be finite numbers")
    if not 0 <= rewards["reward"] <= 1:
        raise ValueError("Robot task reward must lie in [0, 1]")
    return {k: float(v) for k, v in rewards.items()}
