"""slime custom generate hook: a Qwen3.5 VLM operates a Newton robot through tools.

--custom-generate-function-path examples.robotics_newton.rollout.generate

Token-in/token-out multi-turn loop on the geo3k_vlm_multi_turn pattern, with
the environment behind HTTP (server.py) and awaited, so the episodes of a
rollout run concurrently. Per turn: SGLang /generate on the full token
sequence plus every image so far (with logprobs); parse the Qwen3.5 tool calls;
run them on the robot server; append the tool responses (JSON text and camera
images, rendered by the chat template) as non-trainable tokens. The episode
ends when the model answers without a tool call, or on max turns, the token
budget, a truncated turn, or an exhausted command budget. The robot server then
scores the frozen episode. The reward, every evaluator check and the episode
statistics land on the Sample (metadata["robot"]).

Prompt rows come from tasks.py: text-only chat messages, the tool schemas, and
metadata {"task", "seed"}. Images arrive through observe. All samples of a
prompt share the seed, so a GRPO group starts from one scene.

Settings, read from args (e.g. a --custom-config-path yaml):
  robot_server_url       robot server base URL     (http://127.0.0.1:18100)
  robot_max_turns        model turns per episode   (24)
  robot_max_turn_tokens  generation cap per turn   (1024)
The tool-call parser is --sglang-tool-call-parser (default qwen3_coder).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import io
import json
import logging

import aiohttp
import torch
from PIL import Image

from slime.agent.parsing import parse_model_output
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import build_processor_kwargs, encode_image_for_rollout_engine, process_vision_info
from slime.utils.types import Sample

from .client import RobotClient
from .tools import TOOLS

logger = logging.getLogger(__name__)

_PARAM_TYPES = {
    tool["function"]["name"]: {k: v.get("type") for k, v in tool["function"]["parameters"]["properties"].items()}
    for tool in TOOLS
}
# A tool turn is rendered after this stand-in history and cut off at its end (the geo3k trick).
_PREFIX_MESSAGES = [{"role": "system", "content": "-"}, {"role": "user", "content": "-"}]
_FORMAT_ERROR = (
    "Could not parse a tool call. Call exactly one tool per block as "
    "<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>, "
    "or reply without a tool call to end the episode."
)
_TRUNCATED = ("turn_truncated", "token_budget")
# Any of these in a reply that yields no tool call is a failed attempt at one, not a final answer.
_CALL_MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=")


@dataclasses.dataclass(frozen=True)
class RobotRolloutConfig:
    server_url: str = "http://127.0.0.1:18100"
    max_turns: int = 24
    max_turn_tokens: int = 1024

    @classmethod
    def from_args(cls, args) -> RobotRolloutConfig:
        default = cls()
        return cls(
            server_url=getattr(args, "robot_server_url", None) or default.server_url,
            max_turns=int(getattr(args, "robot_max_turns", None) or default.max_turns),
            max_turn_tokens=int(getattr(args, "robot_max_turn_tokens", None) or default.max_turn_tokens),
        )


class RobotUnavailable(RuntimeError):
    """The robot server failed; the sample says nothing about the policy."""


_clients: dict[tuple[str, int], RobotClient] = {}


def _client(url: str) -> RobotClient:
    # aiohttp sessions are bound to the loop that created them.
    key = (url, id(asyncio.get_running_loop()))
    if key not in _clients:
        _clients[key] = RobotClient(url)
    return _clients[key]


async def _robot(call):
    try:
        return await call
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
        raise RobotUnavailable(str(exc)) from exc


def _rounded(value, digits: int = 4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    return value


def _coerce(name: str, arguments: dict) -> dict:
    """JSON-decode string values the schema types as arrays, objects or numbers (XML-fallback parses)."""
    types = _PARAM_TYPES.get(name, {})
    coerced = {}
    for key, value in arguments.items():
        if isinstance(value, str) and types.get(key) in ("array", "object", "number", "integer", "boolean"):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        coerced[key] = value
    return coerced


def _receipts(data: dict) -> list[dict]:
    if "receipt" in data:
        return [data["receipt"]]
    return list(data.get("batch_receipt") or [])


def _tool_message(reply: dict) -> dict:
    """One tool response: the reply JSON, then each camera image under its label."""
    content = [{"type": "text", "text": json.dumps(_rounded(reply["data"]))}]
    for image in reply.get("images") or []:
        content.append({"type": "text", "text": f"\n{image['camera']} camera:"})
        content.append(
            {"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(image["data"]))).convert("RGB")}
        )
    return {"role": "tool", "content": content}


def _error_message(error: str) -> dict:
    return {"role": "tool", "content": [{"type": "text", "text": json.dumps({"status": "error", "error": error})}]}


def _encode_tool_turn(messages: list[dict], last_token: int, tokenizer, processor, template_kwargs: dict):
    """Token ids, images and processor tensors for the tool turn that follows an assistant turn."""
    prefix = tokenizer.apply_chat_template(
        _PREFIX_MESSAGES, tokenize=False, add_generation_prompt=False, **template_kwargs
    )
    full = tokenizer.apply_chat_template(
        _PREFIX_MESSAGES + messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    assert full.startswith(prefix), "chat template does not render the history as a prefix"
    # The template closes an assistant turn with "<|im_end|>\n"; generation stops after <|im_end|>.
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text = ("\n" if last_token == im_end else "<|im_end|>\n") + full[len(prefix) :]
    if not processor:
        return tokenizer.encode(text, add_special_tokens=False), [], None
    images = process_vision_info(messages, processor)["images"] or []
    output = processor(text=text, **build_processor_kwargs({"images": images or None, "videos": None}))
    train = {k: v for k, v in output.items() if k not in ("input_ids", "attention_mask")} or None
    return list(output["input_ids"][0]), images, train


def _merge_train_inputs(chunks: list[dict]) -> dict | None:
    merged: dict[str, list[torch.Tensor]] = {}
    for chunk in chunks:
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                merged.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in merged.items()} or None


def classify(robot: dict) -> str:
    """Failure category of a finished episode (heuristic; see the README).

    format: a turn ran past the per-turn cap, or no physical command was executed and a call was malformed.
    no_action: the model never moved the robot.
    perception (lift): it never closed the gripper at the cube, so it did not locate the cube.
    control: everything else that failed: the robot acted near the goal but the condition was not met.
    """
    if robot.get("reward") == 1.0:
        return "success"
    if robot["end_reason"] == "turn_truncated" or (robot["physical_commands"] == 0 and robot["format_errors"]):
        return "format"
    if robot["physical_commands"] == 0:
        return "no_action"
    if robot["task"] == "lift" and not robot["diagnostics"].get("closed_at_cube"):
        return "perception"
    return "control"


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    assert not getattr(args, "partial_rollout", False), "robot episodes cannot be resumed from a partial rollout"
    cfg = RobotRolloutConfig.from_args(args)
    state = GenerateState(args)
    tokenizer, processor = state.tokenizer, state.processor
    template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    tool_parser = getattr(args, "sglang_tool_call_parser", None) or "qwen3_coder"
    reasoning_parser = getattr(args, "sglang_reasoning_parser", None)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    metadata = sample.metadata or {}
    tools = metadata.get("tools") or TOOLS

    # Always start from the (text-only) prompt, so a retried sample does not keep a previous attempt.
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    sample.tokens, sample.response, sample.response_length = list(prompt_ids), "", 0
    sample.loss_mask, sample.rollout_log_probs, sample.reward = None, None, None
    images, image_data, train_chunks = [], [], []
    budget = (
        args.rollout_max_context_len - len(prompt_ids)
        if getattr(args, "rollout_max_context_len", None)
        else sampling_params.get("max_new_tokens")
    )
    robot = {
        "task": metadata["task"],
        "seed": int(metadata["seed"]),
        "end_reason": "max_turns",
        "turns": 0,
        "tool_calls": 0,
        "format_errors": 0,
        "tool_errors": 0,
        "rejected_commands": 0,
        "physical_commands": 0,
        "calls": [],
    }
    client = _client(cfg.server_url)
    episode_id = None
    try:
        started = await _robot(client.start(robot["task"], robot["seed"]))
        episode_id = robot["episode_id"] = started["episode_id"]
        for turn in range(cfg.max_turns):
            max_new = cfg.max_turn_tokens if budget is None else min(cfg.max_turn_tokens, budget)
            if max_new <= 0:
                robot["end_reason"] = "token_budget"
                break
            payload = {
                "input_ids": sample.tokens,
                "sampling_params": {**sampling_params, "max_new_tokens": max_new},
                "return_logprob": True,
            }
            if image_data:
                payload["image_data"] = image_data
            output = await post(url, payload)
            meta = output["meta_info"]
            pairs = meta.get("output_token_logprobs") or []
            new_tokens = [int(pair[1]) for pair in pairs]
            sample.append_response_tokens(
                args, tokens=new_tokens, log_probs=[float(pair[0]) for pair in pairs], trainable=True, meta_info=meta
            )
            robot["turns"] += 1
            if budget is not None:
                budget -= len(new_tokens)
            finish = meta["finish_reason"]["type"]
            if finish == "abort":
                sample.status = Sample.Status.ABORTED
                return sample
            if finish == "length":
                robot["end_reason"] = "token_budget" if budget is not None and budget <= 0 else "turn_truncated"
                break

            parsed = parse_model_output(
                output["text"],
                tools_schema=tools,
                tool_parser_name=tool_parser,
                reasoning_parser_name=reasoning_parser,
            )
            malformed = parsed.ill_formed or (
                not parsed.tool_uses and any(tag in output["text"] for tag in _CALL_MARKUP)
            )
            if not parsed.tool_uses and not malformed:
                robot["end_reason"] = "final_answer"
                break
            messages, budget_exhausted = [], False
            if malformed:
                robot["format_errors"] += 1
                messages.append(_error_message(_FORMAT_ERROR))
            for use in parsed.tool_uses:
                name = use["name"]
                robot["calls"].append(name)
                if name not in _PARAM_TYPES:
                    robot["format_errors"] += 1
                    messages.append(_error_message(f"unknown tool {name!r}; the tools are {sorted(_PARAM_TYPES)}"))
                    continue
                reply = await _robot(client.call(episode_id, name, _coerce(name, use["input"])))
                robot["tool_calls"] += 1
                robot["tool_errors"] += bool(reply["is_error"])
                for receipt in _receipts(reply["data"]):
                    if receipt.get("status") == "rejected":
                        robot["rejected_commands"] += 1
                        budget_exhausted |= "budget" in receipt.get("reason", "")
                messages.append(_tool_message(reply))
            if budget_exhausted:
                robot["end_reason"] = "command_budget"
                break
            if turn + 1 == cfg.max_turns:
                break  # no further turn would read this observation

            obs_ids, obs_images, obs_train = _encode_tool_turn(
                messages, sample.tokens[-1], tokenizer, processor, template_kwargs
            )
            if budget is not None and len(obs_ids) >= budget:
                robot["end_reason"] = "token_budget"
                break
            sample.append_response_tokens(args, tokens=obs_ids, trainable=False)
            if budget is not None:
                budget -= len(obs_ids)
            images.extend(obs_images)
            image_data.extend(encode_image_for_rollout_engine(image) for image in obs_images)
            if obs_train:
                train_chunks.append(obs_train)

        result = await _robot(client.finish(episode_id))
        episode_id = None  # finish releases the world
        sample.reward = float(result["reward"])
        evaluator_state = result["evaluator_state"]
        robot.update(
            reward=sample.reward,
            success=evaluator_state["success"],
            checks=evaluator_state["checks"],
            metrics=evaluator_state["metrics"],
            physical_commands=evaluator_state["physical_commands"],
            diagnostics=result.get("diagnostics", {}),
        )
        robot["failure"] = classify(robot)
        sample.status = Sample.Status.TRUNCATED if robot["end_reason"] in _TRUNCATED else Sample.Status.COMPLETED
    except RobotUnavailable as exc:
        logger.warning("robot server failed for %s seed %s: %s", robot["task"], robot["seed"], exc)
        robot.update(end_reason="robot_unavailable", failure="infrastructure", error=str(exc)[:500])
        sample.reward, sample.remove_sample = 0.0, True
        sample.status = Sample.Status.FAILED
    finally:
        if episode_id is not None:  # cancelled, aborted or failed before finish
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.release(episode_id), timeout=10)
        sample.multimodal_inputs = {"images": images, "videos": None} if images else None
        sample.multimodal_train_inputs = _merge_train_inputs(train_chunks)
        sample.response = tokenizer.decode(sample.tokens[len(prompt_ids) :], skip_special_tokens=False)
        sample.metadata = {**metadata, "robot": robot}
    return sample


# slime cancels the task on abort (instead of waiting for SGLang); the finally above releases the world.
generate.abort_mode = "request"
