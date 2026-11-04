"""slime custom-generate hook for robot evaluation and on-policy training."""

from __future__ import annotations

import os

from slime.utils.types import Sample

from .config import load_experiment
from .runner import run_episode
from .task import HarborTask


async def generate(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample]:
    metadata = sample.metadata or {}
    path = (
        metadata.get("robotics_config") or getattr(args, "robotics_config", None) or os.environ.get("ROBOTICS_CONFIG")
    )
    if not path:
        raise ValueError("Set ROBOTICS_CONFIG or sample.metadata.robotics_config")
    cfg = load_experiment(path)
    if not evaluation and not cfg.training.enabled:
        raise ValueError("External API robotics is evaluation-only; enable training.enabled for slime-managed SGLang")
    task = HarborTask.load(metadata["task_path"])
    if metadata.get("task_digest") != task.digest:
        raise ValueError("Harbor task digest changed; regenerate the slime manifest")
    agent = metadata.get("agent", cfg.agents[0])
    profile = metadata.get("profile", cfg.profiles[0])
    if agent not in cfg.agents or profile not in cfg.profiles:
        raise ValueError("Sample selects an agent/profile outside the experiment")
    # External API sampling has a different schema (e.g. SGLang top_k=-1
    # means disabled). Training values go directly to the training gateway,
    # never through the external API config revalidated by DSH.
    if not cfg.training.enabled:
        for name in ("temperature", "top_p", "top_k", "seed"):
            if sampling_params.get(name) is not None:
                value = sampling_params[name]
                if name == "top_k" and value == -1:
                    value = None  # disabled in the external API schema
                setattr(cfg.runtime.backend.sampling, name, value)
    if sampling_params.get("max_new_tokens"):
        cfg.runtime.claude.max_output_tokens = cfg.runtime.dsh.max_output_tokens = sampling_params["max_new_tokens"]
    if cfg.training.enabled:
        from .training import TrainingGateway

        # The actor belongs to slime, never the external model registry. Eval
        # during training uses the same current actor, tools and verifier.
        cfg.runtime.model, cfg.runtime.models = None, {}
        cfg.runtime.backend.model = "slime-actor"
        gateway = TrainingGateway(args, cfg, agent, sampling_params, template_kwargs=sample.apply_chat_template_kwargs)
        result = await run_episode(cfg, task, agent=agent, profile=profile, model_gateway=gateway)
        if not evaluation:
            return gateway.samples(sample, result)
    else:
        result = await run_episode(cfg, task, agent=agent, profile=profile)
    sample.tokens, sample.response_length, sample.loss_mask, sample.rollout_log_probs = [0, 0], 1, [0], [0.0]
    sample.response = result.get("response", "")
    sample.reward = result.get("reward", 0.0)
    sample.remove_sample = True  # same evaluation sentinel as coding_agent_rl; never train on it
    sample.session_id = result["trial_id"]
    sample.status = {"completed": Sample.Status.COMPLETED, "truncated": Sample.Status.TRUNCATED}.get(
        result["status"], Sample.Status.ABORTED
    )
    sample.metadata = {**metadata, **result}
    return [sample]
