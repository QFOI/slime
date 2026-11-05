"""Evaluate a frozen SGLang VL actor through the production training gateway.

Start SGLang separately with the same checkpoint and context length. This runs
real harnesses, tools and the independent verifier; no optimizer is created.
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from .config import load_experiment
from .runner import run_episode
from .task import HarborTask
from .training import TrainingGateway


async def evaluate(options):
    cfg = load_experiment(options.config)
    if not cfg.training.enabled:
        raise ValueError("Frozen actor evaluation requires training.enabled: true")
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg.output_dir = str(root / "episodes")
    # External API model registries do not select the frozen local actor.
    cfg.runtime.model = None
    cfg.runtime.models = {}
    cfg.runtime.backend.model = "slime-actor"
    (root / "experiment.yaml").write_text(yaml.safe_dump(cfg.model_dump()))
    args = SimpleNamespace(
        hf_checkpoint=options.checkpoint,
        sglang_router_ip=options.host,
        sglang_router_port=options.port,
        rollout_max_context_len=options.context_length,
        sglang_context_length=options.context_length,
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3",
        rollout_top_p=options.top_p,
        use_score_centering=False,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    sampling = {
        "temperature": options.temperature,
        "top_p": options.top_p,
        "top_k": -1,
        "max_new_tokens": options.max_output_tokens,
    }
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(index, task_path, agent, profile, repeat):
        async with sem:
            gateway = TrainingGateway(args, cfg, agent, sampling)
            result = await run_episode(
                cfg, HarborTask.load(task_path), agent=agent, profile=profile, model_gateway=gateway
            )
            gateway.validate_result(result)
            turns = []
            for encoded, turn, ill_formed in gateway.turns:
                assert len(turn.output_ids) == len(turn.output_log_probs) > 0
                assert all(math.isfinite(x) for x in turn.output_log_probs)
                turns.append(
                    {
                        "prompt_tokens": len(turn.prompt_ids),
                        "output_tokens": len(turn.output_ids),
                        "image_count": len(encoded.image_data),
                        "ill_formed": ill_formed,
                        "finish_reason": turn.finish_reason,
                    }
                )
            record = {**result, "case_index": index, "repeat": repeat, "sampled_turns": turns}
            (Path(result["run_dir"]) / "sampling-evidence.json").write_text(json.dumps(turns, indent=2) + "\n")
            print(
                "EPISODE",
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "case_index",
                            "task",
                            "agent",
                            "profile",
                            "status",
                            "reward",
                            "rewards",
                            "agent_metrics",
                        )
                    }
                ),
                flush=True,
            )
            return record

    cases = itertools.product(cfg.tasks, cfg.agents, cfg.profiles, range(cfg.repeats))
    started = time.monotonic()
    results = await asyncio.gather(*(one(i, *case) for i, case in enumerate(cases)))
    summary = {
        "model": options.model_name,
        "model_revision": options.model_revision,
        "sampling": sampling,
        "context_length": options.context_length,
        "episodes": len(results),
        "elapsed_sec": time.monotonic() - started,
        "cases": results,
        "prompt_sha256": {
            p: hashlib.sha256((Path(__file__).parent / "prompts" / f"{p}.md").read_bytes()).hexdigest()
            for p in cfg.profiles
        },
    }
    (root / "actor-eval.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "EVAL_COMPLETE",
        json.dumps(
            {
                "episodes": len(results),
                "passed": sum(r["reward"] == 1 for r in results),
                "elapsed_sec": summary["elapsed_sec"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for this experiment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
