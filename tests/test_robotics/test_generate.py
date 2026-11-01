"""The standard slime Sample hook preserves indexing and never fabricates RL data."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.robotics import generate as module
from examples.robotics.task import HarborTask
from slime.utils.types import Sample

ROOT = Path(__file__).resolve().parents[2] / "examples/robotics"


def test_standard_evaluation_sample(monkeypatch):
    captured = {}

    async def run(experiment, task, **kwargs):
        captured.update(temperature=experiment.runtime.backend.sampling.temperature, **kwargs)
        return {"trial_id": "trial", "status": "completed", "reward": 1.0, "response": "done"}

    monkeypatch.setattr(module, "run_episode", run)
    sample = Sample(index=7, group_index=3, **HarborTask.load(ROOT / "tasks/panda-reach").row())
    sample.metadata.update(agent="dsh", profile="expert")
    result = asyncio.run(
        module.generate(
            SimpleNamespace(robotics_config=ROOT / "configs/robotics.yaml"),
            sample,
            {"temperature": 0.4},
            evaluation=True,
        )
    )
    assert result == [sample]
    assert sample.index == 7 and sample.group_index == 3
    assert sample.reward == 1 and sample.status == Sample.Status.COMPLETED
    assert sample.remove_sample and sample.loss_mask == [0]
    assert captured == {"temperature": 0.4, "agent": "dsh", "profile": "expert"}


def test_external_api_rollout_cannot_enter_training():
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(module.generate(SimpleNamespace(), Sample(), {}, evaluation=False))


def test_changed_task_rejected_before_agent_start():
    sample = Sample(**HarborTask.load(ROOT / "tasks/panda-reach").row())
    sample.metadata["task_digest"] = "stale"
    with pytest.raises(ValueError, match="digest changed"):
        asyncio.run(
            module.generate(
                SimpleNamespace(robotics_config=ROOT / "configs/robotics.yaml"), sample, {}, evaluation=True
            )
        )
