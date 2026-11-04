"""CPU training integration, not a GPU Megatron/SGLang acceptance test.

Runs official CC/DSH, real MCP, MuJoCo and verifier. A tiny trainable RGB policy
samples between an allowed controller action and stopping. The finite grammar
has a single stochastic token (first byte); remaining bytes are deterministic.
Its actual normalized token probabilities cross the production /generate
adapter. slime BatchBuilder + GRPO + PPO update the policy, then repeat rollout
with updated weights and save/reload a checkpoint. No reward is invented.

This tests plumbing and gradients; it is not a pretrained LLM performance test,
does not run a native SGLang engine, and does not test Megatron weight sync.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from aiohttp import web
from aiohttp.test_utils import TestServer
from examples.robotics.config import load_experiment
from examples.robotics.generate import generate
from examples.robotics.task import HarborTask
from PIL import Image
from tests.test_robotics.smoke import ReachModel
from tests.test_robotics.training_fixtures import PixelProcessor

from slime.data.batch_builder import BatchBuilder
from slime.utils.ppo_utils import compute_policy_loss, get_grpo_returns
from slime.utils.types import Sample


class RGBPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, pixels):
        return self.head(pixels.mean(dim=(0, 2, 3))).log_softmax(-1)


class Request:
    path = "/v1/chat/completions"

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class CPUModel:
    def __init__(self):
        self.policy = RGBPolicy()
        self.processor = PixelProcessor()
        self.controller = ReachModel()
        self.version = 0
        self.images_seen = 0

    async def reply(self, request):
        body = await request.json()
        prompt = body.get("text") or self.processor.tokenizer.decode(body["input_ids"])
        parsed = json.loads(prompt)
        planned = await self.controller.reply(Request({**parsed, "model": "cpu-protocol-fixture"}))
        message = json.loads(planned.body)["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if calls:
            function = calls[0]["function"]
            arguments = json.loads(function["arguments"])
            parameters = "".join(
                f"<parameter={key}>{value if isinstance(value, str) else json.dumps(value)}</parameter>"
                for key, value in arguments.items()
            )
            response = f"<tool_call><function={function['name']}>{parameters}</function></tool_call>"
        else:
            response = message.get("content") or "Task finished."
        images = []
        for uri in body.get("image_data") or []:
            with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
                images.append(image.convert("RGB"))
        self.images_seen += len(images)
        # Initial observe and expert depth exercise are mandatory grammar
        # prefixes. The policy chooses whether to perform the physical action.
        can_choose = bool(calls) and ("__move_to" in response) and bool(images)
        first_logp = 0.0
        if can_choose:
            pixels = self.processor(text=[prompt], images=images)["pixel_values"]
            with torch.no_grad():
                logps = self.policy(pixels)
                rng = torch.Generator().manual_seed(body["sampling_params"]["sampling_seed"])
                choice = int(torch.multinomial(logps.exp(), 1, generator=rng))
                first_logp = float(logps[choice])
            if choice == 0:
                response = "Task finished."
        ids = self.processor.tokenizer.encode(response)
        if len(ids) > body["sampling_params"]["max_new_tokens"]:
            raise ValueError("CPU fixture output exceeds the configured generation budget")
        logps = [first_logp] + [0.0] * (len(ids) - 1)
        return web.json_response(
            {
                "meta_info": {
                    "prompt_tokens": len(self.processor.tokenizer.encode(prompt)),
                    "completion_tokens": len(ids),
                    "output_token_logprobs": list(zip(logps, ids, strict=True)),
                    "finish_reason": {"type": "stop"},
                    "weight_version": str(self.version),
                }
            }
        )


def optimize(model, samples, optimizer):
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=4,
        reward_key=None,
        use_score_centering=False,
        use_rollout_routing_replay=False,
    )
    data = BatchBuilder(args).convert(samples)
    old = [torch.tensor(x) for x in data["rollout_log_probs"]]
    advantages = get_grpo_returns(torch.tensor(data["rewards"]), old)
    optimizer.zero_grad()
    loss = torch.zeros(())
    max_replay_error = 0.0
    stochastic_turns = 0
    for sample, old_logps, advantage, denominator in zip(
        samples, old, advantages, data["rollout_mask_sums"], strict=True
    ):
        if sample.rollout_log_probs[0] == 0:
            continue
        stochastic_turns += 1
        choice = int(sample.response.startswith("<tool_call>"))
        logp = model.policy(sample.multimodal_train_inputs["pixel_values"])[choice]
        current = torch.cat((logp.reshape(1), logp.new_zeros(sample.response_length - 1)))
        max_replay_error = max(max_replay_error, float((current.detach() - old_logps).abs().max()))
        pg, _ = compute_policy_loss(old_logps - current, advantage, 0.2, 0.28)
        mask = torch.tensor(sample.loss_mask)
        loss = loss + (pg * mask).sum() / max(denominator, 1)
    loss = loss / len(set(data["rollout_ids"]))
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0))
    vision_grad = float(model.policy.head.weight.grad.norm())
    before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    assert torch.isfinite(loss) and grad_norm > 0 and vision_grad > 0
    assert max_replay_error < 1e-6
    optimizer.step()
    delta = sum(float((value - before[key]).abs().sum()) for key, value in model.policy.state_dict().items())
    assert delta > 0
    model.version += 1
    return {
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "vision_grad_norm": vision_grad,
        "parameter_delta_l1": delta,
        "max_replay_logprob_error": max_replay_error,
        "stochastic_turns": stochastic_turns,
        "sample_segments": len(samples),
        "trainable_tokens": sum(sum(s.loss_mask) for s in samples),
        "weight_version": model.version,
    }


async def main(options):
    torch.set_num_threads(2)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment("examples/robotics/configs/training.yaml")
    cfg.tasks = cfg.tasks[:1]
    cfg.output_dir = str(output / "episodes")
    cfg.runtime.claude.context_window = cfg.runtime.dsh.context_window = 400000
    cfg.runtime.claude.max_output_tokens = cfg.runtime.dsh.max_output_tokens = 4096
    cfg.runtime.run.timeout_sec = 240
    config_file = output / "smoke.local.yaml"
    config_file.write_text(yaml.safe_dump(cfg.model_dump()))
    task = HarborTask.load(cfg.tasks[0])
    model = CPUModel()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.02)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/generate", model.reply)
    reports = []
    async with TestServer(app) as server:
        args = SimpleNamespace(
            robotics_config=str(config_file),
            hf_checkpoint="cpu-protocol-fixture",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=server.port,
            rollout_max_context_len=400000,
            sglang_context_length=400000,
            sglang_tool_call_parser="qwen3_coder",
            rollout_top_p=1.0,
            use_score_centering=False,
        )
        with patch("examples.robotics.training.load_training_processor", return_value=model.processor):
            for step in range(options.steps):
                samples, outcomes = [], []
                for group, (agent, profile) in enumerate(
                    (a, p) for a in options.agents.split(",") for p in options.profiles.split(",")
                ):
                    for seed in (0, 1):
                        base = Sample(index=group * 2 + seed, group_index=group, **task.row())
                        base.metadata.update(agent=agent, profile=profile)
                        segments = await generate(
                            args,
                            base,
                            {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "top_k": -1,
                                "max_new_tokens": 4096,
                                "sampling_seed": seed,
                            },
                        )
                        assert all(s.weight_versions == [str(step)] for s in segments)
                        assert any(s.multimodal_train_inputs for s in segments)
                        assert all(not s.remove_sample for s in segments)
                        result = segments[0].metadata
                        if agent == "dsh" and profile == "expert":
                            assert result["agent_metrics"]["ptc_successful_subcalls"] > 0
                        outcome = {
                            "step": step,
                            "agent": agent,
                            "profile": profile,
                            "seed": seed,
                            "reward": segments[0].reward,
                            "trial_id": result["trial_id"],
                            "status": result["status"],
                            "segments": len(segments),
                            "image_turns": sum(s.metadata["image_count"] > 0 for s in segments),
                        }
                        print(json.dumps(outcome), flush=True)
                        outcomes.append(outcome)
                        samples.extend(segments)
                assert set(o["reward"] for o in outcomes) == {0.0, 1.0}
                report = {"step": step, "outcomes": outcomes, **optimize(model, samples, optimizer)}
                reports.append(report)
                print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)
                torch.save(
                    {
                        "policy": model.policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "version": model.version,
                    },
                    output / "checkpoint.pt",
                )
                (output / "evidence.json").write_text(
                    json.dumps(
                        {"kind": "cpu_trainable_protocol_fixture", "gpu_training_verified": False, "steps": reports},
                        indent=2,
                    )
                )
    restored = RGBPolicy()
    restored.load_state_dict(torch.load(output / "checkpoint.pt", weights_only=True)["policy"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.policy.state_dict().items())
    print(
        json.dumps(
            {"checkpoint_reload_equal": True, "images_seen": model.images_seen, "completed_steps": len(reports)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="examples/robotics/runs/training-cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--agents", default="cc,dsh")
    parser.add_argument("--profiles", default="intuitive,expert")
    asyncio.run(main(parser.parse_args()))
