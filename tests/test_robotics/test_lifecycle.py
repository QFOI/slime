"""An interrupted harness must lose robot access before container teardown."""

import asyncio
import json
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
    cfg.session_python = "/sim-venv/bin/python"
    task = HarborTask.load(ROOT / "tasks/panda-reach")

    async def freeze():
        events.append("freeze")

    @asynccontextmanager
    async def session(*_args):
        assert _args[-1] == cfg.session_python
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


@pytest.mark.parametrize(
    "agent,subtype",
    [
        ("cc", kind)
        for kind in ("error_max_turns", "error_max_budget_usd", "blocking_limit", "error_during_execution", "success")
    ]
    + [
        ("dsh", kind) for kind in ("model_request_limit", "timeout", "max-tokens", "error_during_execution", "success")
    ],
)
def test_terminal_budget_preserves_independent_scoring(monkeypatch, tmp_path, agent, subtype):
    cfg = load_experiment(ROOT / "configs/robotics.yaml")
    cfg.output_dir = str(tmp_path)
    task = HarborTask.load(ROOT / "tasks/panda-reach")
    freeze = AsyncMock()

    @asynccontextmanager
    async def session(*_args):
        yield {"type": "http", "url": "http://localhost:1/mcp"}, freeze

    async def copy_out(_source, target):
        target.mkdir(parents=True)
        if agent == "dsh":
            event = {"status": subtype, "exit_code": 0, "final_text": "done"}
            if subtype == "max-tokens":
                event.update(status="error", turn_end={"kind": "max-tokens"})
            if subtype == "error_during_execution":
                event.update(status="error", exit_code=1)
            record = target / "dsh/session/result.json"
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps(event))
            return
        event = {"type": "result", "subtype": subtype, "is_error": subtype != "success", "result": "done"}
        if subtype == "blocking_limit":
            # Pinned CC emits subtype=success with is_error=true for full context.
            event.update(subtype="success", terminal_reason="blocking_limit", result="Prompt is too long")
        (target / "trajectory.jsonl").write_text(json.dumps(event) + "\n")

    @asynccontextmanager
    async def sandbox(*_args):
        sb = AsyncMock()
        sb.exec.return_value = (0, cfg.runtime.claude.expected_version, "")
        sb.copy_out.side_effect = copy_out
        yield sb

    @asynccontextmanager
    async def gateway(*_args, **_kwargs):
        yield SimpleNamespace(token="test", url="http://localhost:2", requests=8, limit_reached=False)

    async def verify(*_args):
        freeze.assert_awaited_once()
        return {"reward": 0.0}

    verifier = AsyncMock(side_effect=verify)
    monkeypatch.setattr(runner, "robot_session", session)
    monkeypatch.setattr(runner, "DockerSandbox", sandbox)
    monkeypatch.setattr(runner, "Gateway", gateway)
    monkeypatch.setattr(runner, "verify", verifier)
    monkeypatch.setattr(
        runner,
        "ClaudeCodeHarness" if agent == "cc" else "DeepSeekHarness",
        lambda: SimpleNamespace(
            run=AsyncMock(return_value=0 if subtype == "success" else 1),
            collect_metrics=AsyncMock(return_value={"completed": subtype == "success"}),
        ),
    )
    result = asyncio.run(runner.run_episode(cfg, task, agent=agent, profile="intuitive"))
    if subtype == "error_during_execution":
        assert result["status"] == "aborted" and "reward" not in result
        verifier.assert_not_awaited()
    else:
        assert result["status"] == ("completed" if subtype == "success" else "truncated")
        assert result["reward"] == 0.0 and not result["grading_solved"]
        verifier.assert_awaited_once()
