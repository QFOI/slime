"""slime custom-generate hook, matching coding_agent_rl's evaluation contract.

Robot rollout evaluation uses full multimodal APIs. It does not fabricate
trainable tokens or pass image-omitting coding adapters to a vision model.
"""

from __future__ import annotations

import os

from slime.utils.types import Sample

from .config import load_experiment
from .runner import run_episode
from .task import HarborTask


async def generate(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample]:
    if not evaluation:
        raise ValueError(
            "Robotics is rollout/evaluation-only. Use evaluation=True; external API tokens are not RL samples."
        )
    metadata = sample.metadata or {}
    path = (
        metadata.get("robotics_config") or getattr(args, "robotics_config", None) or os.environ.get("ROBOTICS_CONFIG")
    )
    if not path:
        raise ValueError("Set ROBOTICS_CONFIG or sample.metadata.robotics_config")
    cfg = load_experiment(path)
    task = HarborTask.load(metadata["task_path"])
    if metadata.get("task_digest") != task.digest:
        raise ValueError("Harbor task digest changed; regenerate the slime manifest")
    agent = metadata.get("agent", cfg.agents[0])
    profile = metadata.get("profile", cfg.profiles[0])
    if agent not in cfg.agents or profile not in cfg.profiles:
        raise ValueError("Sample selects an agent/profile outside the experiment")
    # Slime sampling values take precedence, as in the coding-agent adapter.
    for name in ("temperature", "top_p", "top_k", "seed"):
        if sampling_params.get(name) is not None:
            setattr(cfg.runtime.backend.sampling, name, sampling_params[name])
    if sampling_params.get("max_new_tokens"):
        cfg.runtime.claude.max_output_tokens = cfg.runtime.dsh.max_output_tokens = sampling_params["max_new_tokens"]
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
