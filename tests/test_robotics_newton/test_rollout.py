"""The slime rollout hook against a scripted SGLang server and a scripted robot server.

Needs slime's rollout dependencies (sglang, transformers) and a Qwen3.5 checkpoint
directory for the tokenizer, processor and chat template; set
ROBOTICS_NEWTON_HF_CHECKPOINT (default ~/models/Qwen3.5-0.8B).
"""

import asyncio
import base64
import io
import json
import os
import re
import socket
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sglang")
pytest.importorskip("transformers")
CHECKPOINT = Path(os.environ.get("ROBOTICS_NEWTON_HF_CHECKPOINT", Path.home() / "models/Qwen3.5-0.8B"))
if not (CHECKPOINT / "tokenizer_config.json").exists():
    pytest.skip(f"no Qwen3.5 checkpoint at {CHECKPOINT}", allow_module_level=True)

from aiohttp import web
from examples.robotics_newton import rollout
from examples.robotics_newton.rollout_eval import build_args
from examples.robotics_newton.tasks import row
from examples.robotics_newton.tools import TOOLS
from PIL import Image

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.data import Dataset
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample

IMAGE_TOKENS = 64  # a 256x256 camera image: 16x16 patches merged 2x2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _png(seed: int) -> str:
    pixels = np.random.default_rng(seed).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def call(name: str, **arguments) -> str:
    """A tool call in the exact form Qwen3.5's chat template renders it."""
    params = "".join(
        f"<parameter={k}>\n{json.dumps(v) if isinstance(v, (list, dict)) else v}\n</parameter>\n"
        for k, v in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


class FakeSGLang:
    """Answers /generate from a script; each entry is text or (text, finish_type)."""

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.script = list(script)
        self.requests = []
        self.block = asyncio.Event()
        self.block.set()

    async def handle(self, request):
        payload = await request.json()
        self.requests.append(payload)
        await self.block.wait()
        entry = self.script.pop(0)
        text, finish = entry if isinstance(entry, tuple) else (entry, "stop")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if finish == "stop":
            tokens.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        pairs = [[-0.01 * (i + 1), t, None] for i, t in enumerate(tokens)]
        return web.json_response(
            {"text": text, "meta_info": {"finish_reason": {"type": finish}, "output_token_logprobs": pairs}}
        )


class FakeRobot:
    """Minimal robot server: observe returns two camera images; the finish result is scripted."""

    def __init__(self, *, reward=1.0, physical_commands=1, fail_tools=False, budget_left=10):
        self.reward = reward
        self.physical_commands = physical_commands
        self.fail_tools = fail_tools
        self.budget_left = budget_left
        self.calls, self.finished, self.released = [], [], []
        self.observations = 0

    async def start(self, request):
        body = await request.json()
        return web.json_response({"episode_id": f"ep-{body['task']}-{body['seed']}"})

    async def tool(self, request):
        name, arguments = request.match_info["name"], await request.json()
        self.calls.append((name, arguments))
        if self.fail_tools:
            return web.Response(status=500, text="boom")
        self.observations += 1
        data = {"status": "ok", "observation_id": f"obs-{self.observations}"}
        images = []
        if name == "observe":
            images = [
                {"camera": c, "width": 256, "height": 256, "data": _png(i)} for i, c in enumerate(("front", "wrist"))
            ]
        else:
            self.budget_left -= 1
            status = "completed" if self.budget_left >= 0 else "rejected"
            data["receipt"] = {"command_id": arguments.get("command_id"), "status": status, "residual": 0.000123456}
            if status == "rejected":
                data["receipt"]["reason"] = "physical command budget exhausted"
        return web.json_response({"data": data, "images": images, "is_error": False})

    async def finish(self, request):
        self.finished.append(request.match_info["id"])
        state = {
            "success": self.reward == 1.0,
            "checks": {"task_condition": self.reward == 1.0, "physical_command": self.physical_commands > 0},
            "metrics": {"hold_fraction": self.reward},
            "physical_commands": self.physical_commands,
        }
        return web.json_response(
            {"reward": self.reward, "evaluator_state": state, "diagnostics": {"closed_at_cube": True}}
        )

    async def release(self, request):
        self.released.append(request.match_info["id"])
        return web.json_response({"released": True})


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def args(loop):
    args = build_args(
        Namespace(
            hf_checkpoint=str(CHECKPOINT),
            sglang_url="127.0.0.1:1",  # replaced per test
            robot_server_url="http://127.0.0.1:1",
            enable_thinking=False,
            concurrency=8,
            temperature=1.0,
            max_response_len=8192,
            max_turns=6,
            max_turn_tokens=512,
        )
    )
    GenerateState(args, concurrency=8)
    init_http_client(args, concurrency=8)
    return args


@pytest.fixture(scope="module")
def prompt(args, tmp_path_factory):
    path = tmp_path_factory.mktemp("rows") / "rows.jsonl"
    path.write_text(json.dumps(row("lift", 7, "train")) + "\n")
    state = GenerateState(args)
    dataset = Dataset(
        str(path),
        state.tokenizer,
        state.processor,
        max_length=None,
        prompt_key="messages",
        tool_key="tools",
        metadata_key="metadata",
        apply_chat_template=True,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )
    return dataset.origin_samples[0]


def run_episode(loop, args, prompt, sglang, robot, *, cancel_on_request=None, **overrides):
    """Serve both fakes on free ports and run one generate() call."""

    async def main():
        runners = []
        for routes in (
            [web.post("/generate", sglang.handle)],
            [
                web.post("/episodes", robot.start),
                web.post("/episodes/{id}/tools/{name}", robot.tool),
                web.post("/episodes/{id}/finish", robot.finish),
                web.delete("/episodes/{id}", robot.release),
            ],
        ):
            app = web.Application(client_max_size=64 * 1024**2)
            app.add_routes(routes)
            runner = web.AppRunner(app)
            await runner.setup()
            port = _free_port()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            runners.append((runner, port))
        (_, sglang_port), (_, robot_port) = runners
        run_args = Namespace(
            **{
                **vars(args),
                "sglang_router_port": sglang_port,
                "robot_server_url": f"http://127.0.0.1:{robot_port}",
                **overrides,
            }
        )
        sample = Sample(
            prompt=prompt.prompt, metadata=dict(prompt.metadata), multimodal_inputs=prompt.multimodal_inputs
        )
        params = dict(GenerateState(args).sampling_params)
        try:
            task = asyncio.ensure_future(rollout.generate(run_args, sample, params))
            if cancel_on_request is None:
                return await task
            while len(sglang.requests) < cancel_on_request:
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return sample
        finally:
            sglang.block.set()  # let a held request finish so the fake server shuts down promptly
            for runner, _ in runners:
                await runner.cleanup()

    return loop.run_until_complete(main())


def _collapse_images(text: str) -> str:
    return re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", text)


def test_episode_tokens_match_the_chat_template(loop, args, prompt):
    state = GenerateState(args)
    tok = state.tokenizer
    move = {"observation_id": "obs-1", "command_id": "m1", "translation_m": [0.0, 0.0, 0.05]}
    sglang = FakeSGLang(tok, [call("observe"), "Moving up.\n\n" + call("move_to", **move), "The cube is lifted."])
    robot = FakeRobot()
    sample = run_episode(loop, args, prompt, sglang, robot)

    meta = sample.metadata["robot"]
    assert sample.reward == 1.0 and sample.status == Sample.Status.COMPLETED
    assert meta["end_reason"] == "final_answer" and meta["turns"] == 3 and meta["tool_calls"] == 2
    assert meta["failure"] == "success" and robot.finished == ["ep-lift-7"] and not robot.released
    assert robot.calls[1] == ("move_to", move)  # typed arguments reached the robot

    # Token bookkeeping: model tokens are trainable with SGLang's logprobs, tool turns are masked.
    prompt_len = len(tok.encode(prompt.prompt, add_special_tokens=False))
    assert len(sample.tokens) == prompt_len + sample.response_length
    assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
    for request, nxt in zip(sglang.requests, sglang.requests[1:] + [None], strict=True):
        start = len(request["input_ids"]) - prompt_len
        end = len(nxt["input_ids"]) - prompt_len if nxt else sample.response_length
        assert request["input_ids"] == sample.tokens[: prompt_len + start]  # each request extends the last
        mask = sample.loss_mask[start:end]
        model = mask.index(0) if 0 in mask else len(mask)
        assert mask == [1] * model + [0] * (len(mask) - model)
        assert sample.rollout_log_probs[start : start + model] == pytest.approx(
            [-0.01 * (i + 1) for i in range(model)]
        )

    # Images: 2 after observe, carried into every later request, 64 placeholder tokens each.
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    assert [len(r.get("image_data", [])) for r in sglang.requests] == [0, 2, 2]
    for request in sglang.requests:
        assert request["input_ids"].count(pad) == IMAGE_TOKENS * len(request.get("image_data", []))
    assert len(sample.multimodal_inputs["images"]) == 2
    grid = sample.multimodal_train_inputs["image_grid_thw"]
    assert grid.tolist() == [[1, 16, 16]] * 2
    assert sample.multimodal_train_inputs["pixel_values"].shape[0] == int(grid.prod(dim=1).sum())

    # The whole sequence is what the chat template renders for the same conversation.
    observe_text = json.dumps({"status": "ok", "observation_id": "obs-1"})
    move_text = json.dumps(
        {
            "status": "ok",
            "observation_id": "obs-2",
            "receipt": {"command_id": "m1", "status": "completed", "residual": 0.0001},
        }
    )
    image = {"type": "image", "image": "x"}
    messages = row("lift", 7, "train")["messages"] + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "observe", "arguments": {}}}]},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": observe_text},
                {"type": "text", "text": "\nfront camera:"},
                image,
                {"type": "text", "text": "\nwrist camera:"},
                image,
            ],
        },
        {
            "role": "assistant",
            "content": "Moving up.",
            "tool_calls": [{"function": {"name": "move_to", "arguments": move}}],
        },
        {"role": "tool", "content": [{"type": "text", "text": move_text}]},
        {"role": "assistant", "content": "The cube is lifted."},
    ]
    expected = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, enable_thinking=False)
    assert _collapse_images(tok.decode(sample.tokens)) == expected.removesuffix("\n")


def test_malformed_and_unknown_calls_get_feedback(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    headless = "<parameter=translation_m>\n[0, 0, 0.1]\n</parameter>\n</function>\n</tool_call>"  # no opening tags
    script = ["<tool_call>\nobserve()\n</tool_call>", call("teleport", where="cube"), headless, "I give up."]
    sglang = FakeSGLang(tok, script)
    robot = FakeRobot(reward=0.0, physical_commands=0)
    sample = run_episode(loop, args, prompt, sglang, robot)
    meta = sample.metadata["robot"]
    assert meta["format_errors"] == 3 and meta["tool_calls"] == 0 and not robot.calls
    assert meta["end_reason"] == "final_answer" and meta["failure"] == "format" and sample.reward == 0.0
    assert "Could not parse a tool call" in tok.decode(sglang.requests[1]["input_ids"])


def test_truncated_turn_is_scored_and_marked(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    sglang = FakeSGLang(tok, [call("observe"), ("I will think about this for a very long", "length")])
    robot = FakeRobot(reward=0.0, physical_commands=0)
    sample = run_episode(loop, args, prompt, sglang, robot)
    meta = sample.metadata["robot"]
    assert sample.status == Sample.Status.TRUNCATED and meta["end_reason"] == "turn_truncated"
    assert meta["failure"] == "format" and robot.finished and sample.response_length == len(sample.loss_mask)


def test_token_budget_stops_before_an_observation_overflows(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    prompt_len = len(tok.encode(prompt.prompt, add_special_tokens=False))
    sglang = FakeSGLang(tok, [call("observe"), "unused"])
    robot = FakeRobot(reward=0.0, physical_commands=0)
    sample = run_episode(loop, args, prompt, sglang, robot, rollout_max_context_len=prompt_len + 100)
    meta = sample.metadata["robot"]
    assert meta["end_reason"] == "token_budget" and sample.status == Sample.Status.TRUNCATED
    assert len(sglang.requests) == 1 and all(sample.loss_mask)  # the observation was not appended
    assert sample.multimodal_inputs is None and robot.finished


def test_command_budget_ends_the_episode(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    move = {"observation_id": "obs-1", "command_id": "m1", "translation_m": [0.0, 0.0, 0.01]}
    sglang = FakeSGLang(tok, [call("observe"), call("move_to", **move)])
    robot = FakeRobot(reward=0.0, budget_left=0)
    sample = run_episode(loop, args, prompt, sglang, robot)
    meta = sample.metadata["robot"]
    assert meta["end_reason"] == "command_budget" and meta["rejected_commands"] == 1 and robot.finished


def test_cancellation_releases_the_world(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    sglang = FakeSGLang(tok, [call("observe"), "never sent"])
    robot = FakeRobot()

    original = sglang.handle

    async def handle(request):
        if len(sglang.requests) == 1:
            sglang.block.clear()  # hang on the second turn
        return await original(request)

    sglang.handle = handle
    run_episode(loop, args, prompt, sglang, robot, cancel_on_request=2)
    assert robot.released == ["ep-lift-7"] and not robot.finished


def test_robot_failure_marks_the_sample_unusable(loop, args, prompt):
    tok = GenerateState(args).tokenizer
    sglang = FakeSGLang(tok, [call("observe")])
    robot = FakeRobot(fail_tools=True)
    sample = run_episode(loop, args, prompt, sglang, robot)
    meta = sample.metadata["robot"]
    assert sample.status == Sample.Status.FAILED and sample.remove_sample and sample.reward == 0.0
    assert meta["failure"] == "infrastructure" and robot.released == ["ep-lift-7"]


@pytest.mark.parametrize(
    "robot, expected",
    [
        ({"reward": 1.0}, "success"),
        ({"end_reason": "turn_truncated", "physical_commands": 3, "format_errors": 0}, "format"),
        ({"physical_commands": 0, "format_errors": 1}, "format"),
        ({"physical_commands": 0, "format_errors": 0}, "no_action"),
        ({"task": "lift", "diagnostics": {"closed_at_cube": False}}, "perception"),
        ({"task": "lift", "diagnostics": {"closed_at_cube": True}}, "control"),
        ({"task": "reach_pose", "diagnostics": {}}, "control"),
    ],
)
def test_failure_categories(robot, expected):
    base = {"reward": 0.0, "end_reason": "final_answer", "physical_commands": 2, "format_errors": 0, "task": "lift"}
    assert rollout.classify({**base, "diagnostics": {}, **robot}) == expected
