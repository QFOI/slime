"""Rollout-only evaluation: run the slime hook (rollout.generate) on prompt rows, without Ray or Megatron.

Needs a running SGLang server and robot server (never with --allow-privileged):
  python -m examples.robotics_newton.server --world-count 64 --device cuda:1 --port 18100
  python -m sglang.launch_server --model-path /models/Qwen3.5-4B --port 30000 --mem-fraction-static 0.4
  python -m examples.robotics_newton.rollout_eval --hf-checkpoint /models/Qwen3.5-4B \\
      --prompt-data heldout.jsonl --n-samples-per-prompt 4 --output results.jsonl

The hook, prompt loading (slime.utils.data.Dataset) and sampling defaults are
slime's; this script only replaces the Ray rollout manager that would call
them under train.py --debug-rollout-only. It writes one jsonl line per episode
and prints the success rate and failure breakdown per task.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import json
import re
import statistics
import time
from argparse import Namespace

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.data import Dataset
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample

from .rollout import generate

_IMAGE_RUN = re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>")
CATEGORIES = ("success", "format", "no_action", "perception", "control", "infrastructure")


def build_args(cli) -> Namespace:
    """The slime arguments the hook and GenerateState read, with slime's defaults where it has them."""
    host, port = cli.sglang_url.rsplit(":", 1)
    return Namespace(
        hf_checkpoint=cli.hf_checkpoint,
        sglang_router_ip=host,
        sglang_router_port=int(port),
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3" if cli.enable_thinking else None,
        sglang_dp_size=None,
        sglang_server_concurrency=cli.concurrency,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        use_distributed_post=False,
        rollout_temperature=cli.temperature,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=cli.max_response_len,
        rollout_max_context_len=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": cli.enable_thinking},
        partial_rollout=False,
        robot_server_url=cli.robot_server_url,
        robot_max_turns=cli.max_turns,
        robot_max_turn_tokens=cli.max_turn_tokens,
    )


def record(sample: Sample, seconds: float) -> dict:
    robot = sample.metadata["robot"]
    return {
        **{k: v for k, v in robot.items() if k != "calls"},
        "calls": robot["calls"],
        "status": sample.status.value,
        "response_length": sample.response_length,
        "trainable_tokens": sum(sample.loss_mask or []),
        "images": len((sample.multimodal_inputs or {}).get("images") or []),
        "seconds": round(seconds, 2),
        "transcript": _IMAGE_RUN.sub("<image>", sample.response),
    }


def summarize(rows: list[dict]) -> dict:
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    summary = {}
    for task, episodes in sorted(by_task.items()):
        failures = collections.Counter(e["failure"] for e in episodes)
        scored = len(episodes) - failures["infrastructure"]
        summary[task] = {
            "episodes": len(episodes),
            "success_rate": failures["success"] / scored if scored else None,  # infrastructure failures excluded
            "failures": {c: failures[c] for c in CATEGORIES if failures[c]},
            "end_reasons": dict(collections.Counter(e["end_reason"] for e in episodes)),
            "mean_turns": statistics.mean(e["turns"] for e in episodes),
            "mean_physical_commands": statistics.mean(e["physical_commands"] for e in episodes),
            "format_error_episodes": sum(e["format_errors"] > 0 for e in episodes) / len(episodes),
            "mean_response_tokens": statistics.mean(e["response_length"] for e in episodes),
            "mean_seconds": statistics.mean(e["seconds"] for e in episodes),
        }
        failed_checks = collections.Counter(
            name for e in episodes for name, ok in (e.get("checks") or {}).items() if not ok
        )
        summary[task]["failed_checks"] = dict(failed_checks)
    return summary


async def run(cli) -> list[dict]:
    args = build_args(cli)
    state = GenerateState(args, concurrency=cli.concurrency)
    init_http_client(args, concurrency=cli.concurrency)
    dataset = Dataset(
        cli.prompt_data,
        state.tokenizer,
        state.processor,
        max_length=None,
        prompt_key="messages",
        tool_key="tools",
        metadata_key="metadata",
        apply_chat_template=True,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )
    prompts = dataset.origin_samples[: cli.limit] if cli.limit else dataset.origin_samples
    semaphore = asyncio.Semaphore(cli.concurrency)
    rows: list[dict] = []
    output = open(cli.output, "w") if cli.output else None

    async def episode(prompt: Sample, index: int) -> None:
        sample = copy.deepcopy(prompt)
        sample.index = index
        async with semaphore:
            start = time.monotonic()
            try:
                sample = await generate(args, sample, dict(state.sampling_params))
                row = record(sample, time.monotonic() - start)
            except Exception as exc:  # e.g. SGLang unreachable; keep the other episodes going
                row = {
                    **{k: 0 for k in ("turns", "physical_commands", "format_errors", "response_length")},
                    "task": prompt.metadata["task"],
                    "seed": prompt.metadata["seed"],
                    "failure": "infrastructure",
                    "end_reason": "error",
                    "error": repr(exc)[:500],
                    "seconds": round(time.monotonic() - start, 2),
                }
        rows.append(row)
        if output:
            output.write(json.dumps(row) + "\n")
            output.flush()
        print(
            f"[{len(rows)}/{len(prompts) * cli.n_samples_per_prompt}] {row['task']} seed {row['seed']}: "
            f"{row['failure']} ({row['end_reason']}, {row['turns']} turns, {row['physical_commands']} commands)",
            flush=True,
        )

    try:
        await asyncio.gather(
            *(
                episode(prompt, i * cli.n_samples_per_prompt + j)
                for i, prompt in enumerate(prompts)
                for j in range(cli.n_samples_per_prompt)
            )
        )
    finally:
        if output:
            output.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--sglang-url", default="127.0.0.1:30000")
    parser.add_argument("--robot-server-url", default="http://127.0.0.1:18100")
    parser.add_argument("--n-samples-per-prompt", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-turn-tokens", type=int, default=1024)
    parser.add_argument("--max-response-len", type=int, default=24576, help="whole-episode response budget")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", default=None)
    cli = parser.parse_args()
    start = time.monotonic()
    rows = asyncio.run(run(cli))
    print(json.dumps(summarize(rows), indent=2))
    print(f"{len(rows)} episodes in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
