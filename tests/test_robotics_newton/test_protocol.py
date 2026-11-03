"""Tool-surface semantics of the Newton robot episodes, against a fake backend."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("newton")

from examples.robotics_newton.protocol import Episode, Limits
from examples.robotics_newton.sim.scene import sample_layout


class FakeBackend:
    """Kinematic stand-in: moves complete exactly, the hold result is scripted."""

    def __init__(self, hold=(True,) * 10):
        self.tcp = np.array([0.5, 0.0, 0.2])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw: 180 deg about x, pointing down
        self.width = 0.08
        self.hold = list(hold)
        self.moves = []

    def _snap(self, status="completed"):
        return {
            "status": status,
            "tcp_position": self.tcp.copy(),
            "tcp_quat_xyzw": self.quat.copy(),
            "target_position": self.tcp.copy(),
            "target_quat_xyzw": self.quat.copy(),
            "cube_position": np.array([0.5, 0.0, 0.1]),
            "gripper_width": self.width,
            "elapsed": 0.5,
        }

    async def move(self, world, translation, rotvec, duration):
        self.moves.append((translation.tolist(), rotvec.tolist(), duration))
        self.tcp = self.tcp + translation
        return self._snap()

    async def gripper(self, world, width):
        self.width = 2 * width
        return self._snap()

    async def observe(self, world):
        return self._snap(), {"front": np.zeros((4, 4, 3), np.uint8)}

    async def snapshot(self, world):
        return self._snap()

    async def evaluate(self, world):
        return {**self._snap(), "hold_samples": self.hold, "sample_count": 10, "hold_duration_sec": 0.5}


def make_episode(task="lift", **kwargs):
    backend = FakeBackend(**kwargs)
    return Episode("e" * 32, 0, sample_layout(task, 3), backend, limits=Limits(command_budget=3)), backend


def move(obs_id, command_id, translation=(0.0, 0.0, 0.05), **extra):
    return {"observation_id": obs_id, "command_id": command_id, "translation_m": list(translation), **extra}


def test_actions_require_the_newest_observation():
    async def run():
        episode, _ = make_episode()
        assert (await episode.call("move_to", move(None, "a"))).is_error
        first = (await episode.call("observe", {})).data["observation_id"]
        reply = await episode.call("move_to", move(first, "a"))
        assert reply.data["receipt"]["status"] == "completed"
        assert reply.data["observation_id"] != first
        stale = await episode.call("move_to", move(first, "b"))
        assert stale.is_error and "stale" in stale.data["error"]

    asyncio.run(run())


def test_command_id_is_idempotent_and_immutable():
    async def run():
        episode, backend = make_episode()
        obs = (await episode.call("observe", {})).data["observation_id"]
        first = await episode.call("move_to", move(obs, "a"))
        again = await episode.call("move_to", move(obs, "a"))  # old observation id is fine for a replay
        assert again.data == first.data and len(backend.moves) == 1
        changed = await episode.call("move_to", move(first.data["observation_id"], "a", translation=(0, 0, 0.06)))
        assert changed.is_error and "different arguments" in changed.data["error"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments, reason",
    [
        ({"translation_m": [0.3, 0.0, 0.0]}, "translation norm"),
        ({"translation_m": [0.0, 0.0, 0.0], "rotation_delta_rad": [0.0, 0.0, 1.5]}, "rotation norm"),
        ({"translation_m": [0.0, 0.0, -0.2]}, "below"),
    ],
)
def test_limit_violations_are_rejected_without_motion(arguments, reason):
    async def run():
        episode, backend = make_episode()
        obs = (await episode.call("observe", {})).data["observation_id"]
        reply = await episode.call("move_to", {"observation_id": obs, "command_id": "x", **arguments})
        assert reply.data["receipt"]["status"] == "rejected" and reason in reply.data["receipt"]["reason"]
        assert reply.data["observation_id"] == obs and not backend.moves and episode.physical_commands == 0

    asyncio.run(run())


def test_command_budget():
    async def run():
        episode, backend = make_episode()
        obs = (await episode.call("observe", {})).data["observation_id"]
        for i in range(3):
            obs = (await episode.call("move_to", move(obs, f"m{i}", translation=(0, 0, 0.01)))).data["observation_id"]
        reply = await episode.call("move_to", move(obs, "m3"))
        assert reply.data["receipt"]["status"] == "rejected" and "budget" in reply.data["receipt"]["reason"]
        assert len(backend.moves) == 3

    asyncio.run(run())


def test_batch_actions_rules_and_execution():
    async def run():
        episode, backend = make_episode()
        obs = (await episode.call("observe", {})).data["observation_id"]
        gripper_then_move = [
            {"tool": "gripper", "arguments": {"command_id": "g", "action": "close"}},
            {"tool": "move_to", "arguments": {"command_id": "m", "translation_m": [0, 0, 0.01]}},
        ]
        assert (await episode.call("batch_actions", {"observation_id": obs, "actions": gripper_then_move})).is_error
        ok = [
            {"tool": "move_to", "arguments": {"command_id": "m", "translation_m": [0, 0, 0.01]}},
            {"tool": "gripper", "arguments": {"command_id": "g", "action": "close"}},
            {"tool": "observe"},
        ]
        reply = await episode.call("batch_actions", {"observation_id": obs, "actions": ok})
        assert [r["status"] for r in reply.data["batch_receipt"]] == ["completed", "completed"]
        assert reply.images and episode.gripper_command == "closed" and len(backend.moves) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "hold, physical, reward",
    [((True,) * 10, True, 1.0), ((True,) * 9 + (False,), True, 0.0), ((True,) * 10, False, 0.0)],
)
def test_reward_follows_harbor_verifier(hold, physical, reward):
    async def run():
        episode, _ = make_episode(hold=hold)
        obs = (await episode.call("observe", {})).data["observation_id"]
        if physical:
            await episode.call("move_to", move(obs, "a"))
        result = await episode.finish()
        assert result["reward"] == reward
        assert (await episode.call("observe", {})).is_error  # frozen after finish

    asyncio.run(run())


def test_layouts_are_deterministic_and_task_specific():
    assert sample_layout("lift", 7) == sample_layout("lift", 7)
    assert sample_layout("lift", 7) != sample_layout("lift", 8)
    assert sample_layout("lift", 7).goal_position is None
    assert sample_layout("reach_pose", 7).goal_position is not None
