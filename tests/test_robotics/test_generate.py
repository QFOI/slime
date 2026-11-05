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
        asyncio.run(
            module.generate(
                SimpleNamespace(robotics_config=ROOT / "configs/robotics.yaml"), Sample(), {}, evaluation=False
            )
        )


def test_changed_task_rejected_before_agent_start():
    sample = Sample(**HarborTask.load(ROOT / "tasks/panda-reach").row())
    sample.metadata["task_digest"] = "stale"
    with pytest.raises(ValueError, match="digest changed"):
        asyncio.run(
            module.generate(
                SimpleNamespace(robotics_config=ROOT / "configs/robotics.yaml"), sample, {}, evaluation=True
            )
        )


def test_training_sampling_does_not_enter_external_dsh_schema(monkeypatch):
    from examples.robotics import training
    from robot_rollout.config import Config

    recorded = {}

    class Gateway:
        def __init__(self, args, cfg, agent, sampling_params, **kwargs):
            recorded["sampling"] = sampling_params

        def samples(self, sample, result):
            sample.remove_sample = False
            return [sample]

    async def run(cfg, task, **kwargs):
        # DSH revalidates this schema in its child process.
        Config.model_validate(cfg.runtime.model_dump())
        assert cfg.runtime.backend.sampling.top_k != -1
        assert cfg.runtime.model is None and not cfg.runtime.models
        assert kwargs["agent"] == "dsh" and isinstance(kwargs["model_gateway"], Gateway)
        return {"trial_id": "trial", "status": "completed", "reward": 1.0}

    monkeypatch.setattr(training, "TrainingGateway", Gateway)
    monkeypatch.setattr(module, "run_episode", run)
    sample = Sample(index=0, **HarborTask.load(ROOT / "tasks/panda-reach").row())
    sample.metadata.update(agent="dsh", profile="expert")
    asyncio.run(
        module.generate(
            SimpleNamespace(robotics_config=ROOT / "configs/training.yaml"), sample, {"top_k": -1, "temperature": 0.8}
        )
    )
    assert recorded["sampling"] == {"top_k": -1, "temperature": 0.8}


def test_slime_eval_disabled_top_k_maps_to_external_api_none(monkeypatch):
    from robot_rollout.config import Config

    async def run(cfg, task, **kwargs):
        Config.model_validate(cfg.runtime.model_dump())
        assert cfg.runtime.backend.sampling.top_k is None
        return {"trial_id": "trial", "status": "completed", "reward": 0.0}

    monkeypatch.setattr(module, "run_episode", run)
    sample = Sample(index=0, **HarborTask.load(ROOT / "tasks/panda-reach").row())
    sample.metadata.update(agent="dsh", profile="expert")
    asyncio.run(
        module.generate(
            SimpleNamespace(robotics_config=ROOT / "configs/robotics.yaml"), sample, {"top_k": -1}, evaluation=True
        )
    )


@pytest.mark.parametrize("status", ["aborted", "completed", "truncated"])
def test_training_eval_does_not_disguise_infrastructure_failure_as_zero(monkeypatch, status):
    from examples.robotics import training

    class Gateway(training.TrainingGateway):
        def __init__(self, *_args, **_kwargs):
            self.error = None
            self.turns = [object()]

    async def run(*_args, **_kwargs):
        if status == "aborted":
            return {"trial_id": "trial", "status": status, "error": "harness failed"}
        return {"trial_id": "trial", "status": status, "reward": 0.0}

    monkeypatch.setattr(training, "TrainingGateway", Gateway)
    monkeypatch.setattr(module, "run_episode", run)
    sample = Sample(index=0, **HarborTask.load(ROOT / "tasks/panda-reach").row())
    invocation = module.generate(
        SimpleNamespace(robotics_config=ROOT / "configs/training.yaml"), sample, {}, evaluation=True
    )
    if status == "aborted":
        with pytest.raises(RuntimeError, match="harness failed"):
            asyncio.run(invocation)
    else:
        assert asyncio.run(invocation)[0].reward == 0
