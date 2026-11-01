"""An interrupted harness must lose robot access before container teardown."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.robotics import runner
from examples.robotics.config import load_experiment
from examples.robotics.task import HarborTask

ROOT = Path(__file__).resolve().parents[2] / "examples/robotics"


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
def test_interrupt_freezes_before_teardown(monkeypatch, tmp_path, failure):
    events = []
    cfg = load_experiment(ROOT / "configs/robotics.yaml")
    cfg.output_dir = str(tmp_path)
    task = HarborTask.load(ROOT / "tasks/panda-reach")

    async def freeze():
        events.append("freeze")

    @asynccontextmanager
    async def session(*_args):
        try:
            yield {"type": "http", "url": "http://localhost:1/mcp"}, freeze
        finally:
            events.append("session_exit")

    @asynccontextmanager
    async def sandbox(*_args):
        try:
            yield AsyncMock()
        finally:
            events.append("container_exit")

    @asynccontextmanager
    async def gateway(*_args, **_kwargs):
        yield SimpleNamespace(token="test", url="http://localhost:2", requests=0, limit_reached=False)

    async def fail(*_args, **_kwargs):
        events.append("harness")
        raise failure("interrupted")

    monkeypatch.setattr(runner, "robot_session", session)
    monkeypatch.setattr(runner, "DockerSandbox", sandbox)
    monkeypatch.setattr(runner, "Gateway", gateway)
    monkeypatch.setattr(runner, "DeepSeekHarness", lambda: SimpleNamespace(run=fail))
    verifier = AsyncMock()
    monkeypatch.setattr(runner, "verify", verifier)
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(runner.run_episode(cfg, task, agent="dsh", profile="expert"))
    else:
        result = asyncio.run(runner.run_episode(cfg, task, agent="dsh", profile="expert"))
        assert result["status"] == "aborted" and "reward" not in result
    assert events == ["harness", "freeze", "container_exit", "session_exit"]
    verifier.assert_not_awaited()
