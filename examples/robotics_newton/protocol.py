"""Episode semantics of the intuitive robot tool surface.

Follows the robot-rollout contract visible from slime (prompts/intuitive.md,
tests/test_robotics/smoke.py, the Harbor verifiers): move_to is a relative
base-frame translation plus axis-angle rotation anchored to the measured TCP;
actions must cite the newest observation_id; command_id is idempotent; physical
commands are budgeted and executed serially; batch_actions is a literal list.
After the episode, the evaluator samples the task condition 10 times at 20 Hz
and the reward follows the Harbor verifier: success needs every hold sample,
the full 0.5 s window, and at least one physical command.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .sim.scene import FINGER_OPEN, Layout

INSTRUCTIONS = {
    # Verbatim from examples/robotics/tasks/*/instruction.md.
    "lift": (
        "Pick up the red cube with the Panda gripper and hold it above the table for at least 0.5 simulated "
        "seconds. Ground each motion in the current camera observations and robot measurements; verify the "
        "grasp and lift visually."
    ),
    "reach_pose": (
        "Move the Panda end effector to the public goal pose returned by observe. Satisfy both position and "
        "orientation and hold the pose for at least 0.5 simulated seconds. Use the robot tools; do not inspect "
        "the simulator or evaluator."
    ),
}


class Backend(Protocol):
    """Per-world physical operations; implemented by server.SimWorker."""

    async def move(self, world: int, translation: np.ndarray, rotvec: np.ndarray, duration: float) -> dict: ...

    async def gripper(self, world: int, width: float) -> dict: ...

    async def observe(self, world: int) -> tuple[dict, dict[str, np.ndarray]]: ...

    async def snapshot(self, world: int) -> dict: ...

    async def evaluate(self, world: int) -> dict: ...


@dataclasses.dataclass(frozen=True)
class Limits:
    max_translation_m: float = 0.25
    max_rotation_rad: float = 1.0
    min_tcp_z_m: float = 0.005
    max_batch_physical: int = 3
    command_budget: int = 80
    # Planned TCP speeds; a move's duration is the longest of the three.
    linear_speed: float = 0.25
    angular_speed: float = 1.0
    min_duration: float = 0.4


DEFAULT_LIMITS = Limits()


@dataclasses.dataclass
class Reply:
    data: dict
    images: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.data.get("status") == "error"


class ToolError(ValueError):
    pass


def pose(position, quat_xyzw) -> dict:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return {"position_m": [float(v) for v in position], "quaternion_wxyz": [w, x, y, z]}


def _vector(arguments: dict, key: str, default=None) -> np.ndarray:
    value = arguments.get(key, default)
    if value is None:
        raise ToolError(f"{key} is required")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ToolError(f"{key} must be a list of 3 numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ToolError(f"{key} must contain finite numbers")
    return np.asarray(value, dtype=np.float64)


class Episode:
    """One episode on one leased world; tool calls are serialized."""

    def __init__(
        self,
        episode_id: str,
        world: int,
        layout: Layout,
        backend: Backend,
        *,
        limits: Limits = DEFAULT_LIMITS,
        cameras: list[dict] | None = None,
    ):
        self.id = episode_id
        self.world = world
        self.layout = layout
        self.backend = backend
        self.limits = limits
        self.cameras = cameras or []
        self.observation_id: str | None = None
        self.physical_commands = 0
        self.gripper_command = "open"
        self.events: list[dict] = []
        self._observations = 0
        self._commands: dict[str, tuple[str, Reply]] = {}
        self._lock = asyncio.Lock()
        self._finished = False

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.layout.task]

    async def call(self, name: str, arguments: dict | None) -> Reply:
        handlers = {
            "observe": self._observe,
            "move_to": self._move_to,
            "gripper": self._gripper,
            "batch_actions": self._batch_actions,
        }
        async with self._lock:
            if self._finished:
                return Reply({"status": "error", "error": "episode already finished"})
            if name not in handlers:
                return Reply({"status": "error", "error": f"unknown tool {name!r}"})
            if not isinstance(arguments, dict):
                return Reply({"status": "error", "error": "arguments must be an object"})
            try:
                reply = await handlers[name](arguments)
            except ToolError as exc:
                reply = Reply({"status": "error", "error": str(exc)})
            self.events.append({"tool": name, "arguments": arguments, "reply": reply.data})
            return reply

    async def finish(self) -> dict:
        """Freeze the episode and score it; no tool call is accepted afterwards."""
        async with self._lock:
            self._finished = True
            result = await self.backend.evaluate(self.world)
        hold = result["hold_samples"]
        success = bool(hold) and len(hold) == result["sample_count"] and all(hold)
        state = {
            "task": self.layout.task,
            "seed": self.layout.seed,
            "simulator": "Newton SolverMuJoCo",
            "success": success,
            "hold_samples": hold,
            "hold_duration_sec": result["hold_duration_sec"],
            "cube_world_position": [float(v) for v in result["cube_position"]],
            "tcp_pose": pose(result["tcp_position"], result["tcp_quat_xyzw"]),
            "observation_id": self.observation_id,
            "physical_commands": self.physical_commands,
        }
        if self.layout.goal_position is not None:
            state["reach_goal"] = pose(self.layout.goal_position, self.layout.goal_quat_xyzw)
        physical = success and self.physical_commands > 0 and state["hold_duration_sec"] >= 0.5
        return {
            "reward": float(physical),
            "rewards": {"reward": float(physical), "physical_success": float(success)},
            "evaluator_state": state,
        }

    # ----- tools -------------------------------------------------------------

    def _new_observation_id(self) -> str:
        self._observations += 1
        self.observation_id = f"{self.id[:16]}:obs-{self._observations}"
        return self.observation_id

    def _check_observation(self, arguments: dict) -> None:
        if self.observation_id is None:
            raise ToolError("call observe before acting")
        if arguments.get("observation_id") != self.observation_id:
            raise ToolError(f"stale or missing observation_id; the newest is {self.observation_id}")

    def _replay(self, tool: str, arguments: dict) -> tuple[str, Reply | None]:
        command_id = arguments.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise ToolError("command_id must be a non-empty string")
        identity = json.dumps(
            {"tool": tool, **{k: v for k, v in arguments.items() if k not in ("observation_id", "note")}},
            sort_keys=True,
        )
        if command_id in self._commands:
            previous, reply = self._commands[command_id]
            if previous != identity:
                raise ToolError(f"command_id {command_id!r} was already used with different arguments")
            return identity, reply
        return identity, None

    def _state(self, snap: dict) -> dict:
        return {
            "tcp_pose": pose(snap["tcp_position"], snap["tcp_quat_xyzw"]),
            "gripper": {"width_m": round(snap["gripper_width"], 5), "command": self.gripper_command},
        }

    def _robot_card(self) -> dict:
        lim = self.limits
        return {
            "robot": "Franka FR3 arm with Franka Hand (simulated)",
            "base_frame": "x forward over the table, y left, z up; origin at the robot base",
            "table_top_z_m": 0.0,
            "tcp": "centre point between the fingertip pads",
            "gripper": {"max_width_m": 2 * FINGER_OPEN, "actions": ["open", "close"]},
            "limits": {
                "max_translation_m": lim.max_translation_m,
                "max_rotation_rad": lim.max_rotation_rad,
                "min_tcp_z_m": lim.min_tcp_z_m,
                "max_batch_physical_actions": lim.max_batch_physical,
            },
            "command_budget": lim.command_budget,
            "cameras": self.cameras,
        }

    def _task(self) -> dict:
        task = {"name": self.layout.task, "instruction": self.instruction}
        if self.layout.goal_position is not None:
            task["goal_tcp_pose"] = pose(self.layout.goal_position, self.layout.goal_quat_xyzw)
        return task

    async def _observe(self, arguments: dict) -> Reply:
        snap, images = await self.backend.observe(self.world)
        data = {
            "status": "ok",
            "observation_id": self._new_observation_id(),
            "state": self._state(snap),
            "task": self._task(),
            "robot_card": self._robot_card(),
            "physical_commands": self.physical_commands,
            "images": [
                {"camera": name, "height": img.shape[0], "width": img.shape[1]} for name, img in images.items()
            ],
        }
        return Reply(data, images)

    def _rejected(self, identity: str, receipt: dict, reason: str) -> Reply:
        receipt.update(status="rejected", reason=reason)
        reply = Reply({"status": "ok", "observation_id": self.observation_id, "receipt": receipt})
        self._commands[receipt["command_id"]] = (identity, reply)
        return reply

    async def _move_to(self, arguments: dict) -> Reply:
        identity, replay = self._replay("move_to", arguments)
        if replay is not None:
            return replay
        self._check_observation(arguments)
        if arguments.get("frame", "base") != "base":
            raise ToolError("only frame='base' is supported")
        translation = _vector(arguments, "translation_m")
        rotvec = _vector(arguments, "rotation_delta_rad", [0.0, 0.0, 0.0])
        lim = self.limits
        receipt = {
            "command_id": arguments["command_id"],
            "tool": "move_to",
            "requested": {"translation_m": translation.tolist(), "rotation_delta_rad": rotvec.tolist()},
        }
        if self.physical_commands >= lim.command_budget:
            return self._rejected(identity, receipt, "physical command budget exhausted")
        if np.linalg.norm(translation) > lim.max_translation_m:
            return self._rejected(identity, receipt, f"translation norm exceeds {lim.max_translation_m} m")
        if np.linalg.norm(rotvec) > lim.max_rotation_rad:
            return self._rejected(identity, receipt, f"rotation norm exceeds {lim.max_rotation_rad} rad")
        before = await self.backend.snapshot(self.world)
        if before["tcp_position"][2] + translation[2] < lim.min_tcp_z_m:
            return self._rejected(identity, receipt, f"target TCP z below {lim.min_tcp_z_m} m")

        duration = max(
            lim.min_duration,
            float(np.linalg.norm(translation)) / lim.linear_speed,
            float(np.linalg.norm(rotvec)) / lim.angular_speed,
        )
        snap = await self.backend.move(self.world, translation, rotvec, duration)
        self.physical_commands += 1
        target = Rotation.from_quat(snap["target_quat_xyzw"])
        achieved = Rotation.from_quat(snap["tcp_quat_xyzw"])
        receipt.update(
            status=snap["status"],
            target_tcp_pose=pose(snap["target_position"], snap["target_quat_xyzw"]),
            achieved_tcp_pose=pose(snap["tcp_position"], snap["tcp_quat_xyzw"]),
            residual={
                "position_m": float(np.linalg.norm(snap["target_position"] - snap["tcp_position"])),
                "rotation_rad": float((target * achieved.inv()).magnitude()),
            },
            sim_duration_sec=round(snap["elapsed"], 4),
        )
        reply = Reply(
            {
                "status": "ok",
                "observation_id": self._new_observation_id(),
                "receipt": receipt,
                "state": self._state(snap),
            }
        )
        self._commands[receipt["command_id"]] = (identity, reply)
        return reply

    async def _gripper(self, arguments: dict) -> Reply:
        identity, replay = self._replay("gripper", arguments)
        if replay is not None:
            return replay
        self._check_observation(arguments)
        action = arguments.get("action")
        if action not in ("open", "close"):
            raise ToolError("action must be 'open' or 'close'")
        receipt = {"command_id": arguments["command_id"], "tool": "gripper", "requested": {"action": action}}
        if self.physical_commands >= self.limits.command_budget:
            return self._rejected(identity, receipt, "physical command budget exhausted")
        snap = await self.backend.gripper(self.world, FINGER_OPEN if action == "open" else 0.0)
        self.physical_commands += 1
        self.gripper_command = "open" if action == "open" else "closed"
        receipt.update(
            status=snap["status"], width_m=round(snap["gripper_width"], 5), sim_duration_sec=round(snap["elapsed"], 4)
        )
        reply = Reply(
            {
                "status": "ok",
                "observation_id": self._new_observation_id(),
                "receipt": receipt,
                "state": self._state(snap),
            }
        )
        self._commands[receipt["command_id"]] = (identity, reply)
        return reply

    async def _batch_actions(self, arguments: dict) -> Reply:
        self._check_observation(arguments)
        actions = arguments.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ToolError("actions must be a non-empty list")
        tools = [a.get("tool") if isinstance(a, dict) else None for a in actions]
        if any(t not in ("move_to", "gripper", "observe") for t in tools):
            raise ToolError("batch entries must be move_to, gripper or observe")
        if "observe" in tools[:-1]:
            raise ToolError("observe may only be the last batch entry")
        physical = [a for a in actions if a["tool"] != "observe"]
        if len(physical) > self.limits.max_batch_physical:
            raise ToolError(f"at most {self.limits.max_batch_physical} physical actions per batch")
        if "gripper" in tools[: len(physical) - 1]:
            raise ToolError("after a gripper command only observe may follow")
        receipts = []
        for action in physical:
            inner = dict(action.get("arguments") or {}, observation_id=self.observation_id)
            handler = self._move_to if action["tool"] == "move_to" else self._gripper
            try:
                reply = await handler(inner)
            except ToolError as exc:
                receipts.append({"tool": action["tool"], "status": "error", "error": str(exc)})
                break
            receipts.append(reply.data["receipt"])
            if reply.data["receipt"]["status"] != "completed":
                break
        observation = await self._observe({})
        observation.data["batch_receipt"] = receipts
        return observation
