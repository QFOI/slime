"""Actual CC/DSH + HTTP MCP + MuJoCo, with a deterministic protocol-test model.

This validates rollout plumbing and physics; it is not an LLM quality score.
The policy sees only the published goal and measured TCP, never hidden state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import yaml
from aiohttp import web
from examples.robotics.__main__ import evaluate
from examples.robotics.config import load_experiment
from robot_rollout.bridge import send_message, to_anthropic
from robot_rollout.demo import MockModel
from scipy.spatial.transform import Rotation


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)
    elif isinstance(value, str):
        # Tool text can be JSON followed by camera labels. Parse only complete
        # JSON objects, never execute content returned by a model/tool.
        for start, char in enumerate(value):
            if char == "{":
                try:
                    decoded, _ = json.JSONDecoder().raw_decode(value[start:])
                except json.JSONDecodeError:
                    continue
                yield from objects(decoded)
                break


class ReachModel(MockModel):
    async def reply(self, request):
        body = await request.json()
        self.requests.append(body)
        openai = request.path.endswith("completions")
        names = [t["function"]["name"] if openai else t["name"] for t in body.get("tools", [])]
        results = (
            [m for m in body["messages"] if m["role"] == "tool"]
            if openai
            else [
                b
                for m in body["messages"]
                if isinstance(m.get("content"), list)
                for b in m["content"]
                if b.get("type") == "tool_result"
            ]
        )
        observations = [
            x for result in results for x in objects(result) if "observation_id" in x and "state" in x and "task" in x
        ]
        ptc = "run_code" in names
        expert = ptc or any(n.endswith("__run_python") for n in names)
        name, args = "observe", {}
        if observations:
            obs = observations[-1]
            if expert and len(results) == 1:
                name = "run_python"
                args = {
                    "mode": "offline",
                    "timeout_sec": 30,
                    "code": "depth = artifacts.load(observation['artifacts'][0]['depth']['artifact_id'])\nresult = {'depth_median_m': float(np.median(depth[np.isfinite(depth) & (depth > 0)]))}",
                }
            else:
                pose, goal = obs["state"]["tcp_pose"], obs["task"]["goal_tcp_pose"]
                delta = np.array(goal["position_m"]) - pose["position_m"]

                def orientation(q):
                    return Rotation.from_quat(np.array(q)[[1, 2, 3, 0]])

                rotation = (
                    orientation(goal["quaternion_wxyz"]) * orientation(pose["quaternion_wxyz"]).inv()
                ).as_rotvec()
                if np.linalg.norm(delta) < 0.008 and np.linalg.norm(rotation) < 0.04:
                    name = None
                else:
                    limits = obs["robot_card"]["limits"]
                    delta *= min(1, 0.95 * limits["max_translation_m"] / max(np.linalg.norm(delta), 1e-12))
                    rotation *= min(1, 0.95 * limits["max_rotation_rad"] / max(np.linalg.norm(rotation), 1e-12))
                    name, args = "move_to", {
                        "observation_id": obs["observation_id"],
                        "command_id": f"smoke-{len(results)}",
                        "note": "Protocol test: track the public pose target from measured TCP.",
                        "frame": "base",
                        "translation_m": delta.tolist(),
                        "rotation_delta_rad": rotation.tolist(),
                    }
        if len(results) > 15:
            name = None
        message = {"role": "assistant", "content": "Measured target reached." if name is None else None}
        if name:
            wire_name = "mcp__robot__" + name
            if ptc:
                args = {
                    "code": f"const reply = await tools.{wire_name}({json.dumps(args)}); return reply.structuredContent;",
                    "description": "Exercise the public robot control contract",
                }
                wire_name = "run_code"
            assert wire_name in names, (wire_name, names)
            message["tool_calls"] = [
                {
                    "id": f"call_{len(results)}",
                    "type": "function",
                    "function": {"name": wire_name, "arguments": json.dumps(args)},
                }
            ]
        response = {
            "id": "smoke-response",
            "model": body["model"],
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if name else "stop"}],
            "usage": {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": 150,
                "total_tokens": self.input_tokens + 150,
            },
        }
        if openai:
            return web.json_response(response)
        return await send_message(request, to_anthropic(response, body["model"]), stream=body.get("stream", False))


async def main(args):
    cfg = load_experiment(Path("examples/robotics/configs/robotics.yaml"))
    cfg.output_dir = str(Path(args.output).resolve())
    cfg.runtime.claude.compaction.enabled = cfg.runtime.dsh.compaction_enabled = False
    cfg.runtime.claude.thinking_budget = 0
    cfg.runtime.run.timeout_sec = 240
    cfg.runtime.model, cfg.runtime.models = None, {}
    cfg.tasks = cfg.tasks[:1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for agent in args.agents.split(","):
        for profile in args.profiles.split(","):
            async with ReachModel() as model:
                cfg.runtime.backend.kind, cfg.runtime.backend.base_url = args.provider, model.url
                cfg.runtime.backend.api_key_env, cfg.runtime.backend.model = None, "protocol-test-model"
                cfg.agents, cfg.profiles = [agent], [profile]
                cfg.output_dir = str(output / f"{agent}-{profile}")
                config_path = output / f"{agent}-{profile}.local.yaml"
                config_path.write_text(yaml.safe_dump(cfg.model_dump()))
                # Exercise YAML → standard slime generate hook → Sample, not
                # just the lower-level episode runner.
                report = await evaluate(config_path)
                result = report["cases"][0]
                image_count = sum(1 for x in objects(model.requests) if x.get("type") in {"image", "image_url"})
                result["test_model"] = "deterministic-public-pose-controller"
                result["observed_api_image_blocks"] = image_count
                rows.append(result)
                print(json.dumps(result, indent=2), flush=True)
    Path(args.output, "smoke.json").write_text(json.dumps(rows, indent=2))
    assert all(
        r["status"] == "completed" and r.get("reward") == 1 and r["observed_api_image_blocks"] > 0 for r in rows
    )
    assert all(
        r["agent_metrics"]["ptc_successful_subcalls"] > 0
        for r in rows
        if r["agent"] == "dsh" and r["profile"] == "expert"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--agents", default="cc,dsh")
    parser.add_argument("--profiles", default="intuitive,expert")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--output", default="examples/robotics/runs/smoke")
    asyncio.run(main(parser.parse_args()))
