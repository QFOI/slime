"""CC/DSH → slime-managed SGLang → token-exact, image-bearing training Samples.

Each model invocation is a separate Sample. This deliberately avoids merging
re-rendered histories, compaction branches, or tensors from different views.
slime's rollout_id-aware BatchBuilder normalizes reward and loss per episode.
External paid APIs remain evaluation-only; they are never an RL actor.
"""

from __future__ import annotations

import asyncio
import secrets
from functools import lru_cache

from aiohttp import web

from slime.agent.adapters import anthropic, openai
from slime.agent.adapters.common import Session, call_sglang_generate
from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import REPLAY_FIELDS
from slime.utils.types import Sample

from .vision import encode_prompt, image_content


@lru_cache(maxsize=2)
def load_training_processor(checkpoint):
    from slime.utils.processing_utils import load_processor

    processor = load_processor(checkpoint, trust_remote_code=True)
    if processor is None or not hasattr(processor, "image_processor"):
        raise ValueError("Robotics training requires a vision-language HF checkpoint with an image processor")
    return processor


def validate_training_args(args):
    for field in ("hf_checkpoint", "sglang_router_ip", "sglang_router_port", "rollout_max_context_len"):
        if not getattr(args, field, None):
            raise ValueError(f"Robotics training requires slime --{field.replace('_', '-')}")
    if getattr(args, "partial_rollout", False):
        raise ValueError("Robotics training does not support partial rollout/resuming a physical session")
    if getattr(args, "group_rm", False):
        raise ValueError("Robotics training uses its independent verifier; --group-rm must be disabled")


class TrainingGateway:
    """One loopback-authenticated server per episode; both official wire protocols."""

    def __init__(self, args, experiment, agent, sampling_params, *, template_kwargs=None, processor=None):
        validate_training_args(args)
        self.args = args
        self.processor = processor if processor is not None else load_training_processor(args.hf_checkpoint)
        self.tokenizer = self.processor.tokenizer
        self.token = secrets.token_urlsafe(32)
        self.requests = 0
        self.limit_reached = False
        self.error = None
        self.turns = []
        self.request_limit = min(
            experiment.max_model_requests,
            experiment.runtime.dsh.max_model_requests if agent == "dsh" else experiment.max_model_requests,
        )
        self.session = Session(
            sampling_defaults=dict(sampling_params),
            max_context_tokens=int(args.rollout_max_context_len),
            apply_chat_template_kwargs={
                **(getattr(args, "apply_chat_template_kwargs", None) or {}),
                **(template_kwargs or {}),
            },
        )
        options = dict(
            tokenizer=self.tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None),
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None),
            rollout_args=args,
        )
        self.wires = {"cc": anthropic.AnthropicAdapter(**options), "dsh": openai.OpenAIAdapter(**options)}
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/messages", self.reply)
        self.app.router.add_post("/v1/chat/completions", self.reply)
        self.app.router.add_post("/v1/messages/count_tokens", self.count_tokens)
        self.lock = asyncio.Lock()
        self.runner = None

    async def __aenter__(self):
        self.runner = web.AppRunner(self.app, access_log=None, handler_cancellation=True, shutdown_timeout=10)
        try:
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            await site.start()
            self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        except BaseException:
            await self.runner.cleanup()
            raise
        return self

    async def __aexit__(self, *_):
        await self.runner.cleanup()

    def authorize(self, request):
        auth = request.headers.get("Authorization", "")
        key = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("X-Api-Key", "")
        if not secrets.compare_digest(key, self.token):
            raise web.HTTPUnauthorized(text="Invalid episode token")

    def translate(self, body, wire_name):
        if wire_name == "cc":
            self.wires["cc"]._preprocess_body(body)
            return (
                anthropic._translate_messages(
                    body.get("messages") or [], body.get("system"), content_transform=image_content
                ),
                anthropic._tools_to_chat_tools(body.get("tools")),
            )
        return (
            openai._translate_messages(body.get("messages") or [], content_transform=image_content),
            openai._tools_to_chat_tools(body.get("tools")),
        )

    async def count_tokens(self, request):
        self.authorize(request)
        messages, tools = self.translate(await request.json(), "cc")
        encoded = encode_prompt(messages, tools, self.processor, self.session.apply_chat_template_kwargs)
        return web.json_response({"input_tokens": len(encoded.token_ids)})

    async def reply(self, request):
        self.authorize(request)
        async with self.lock:
            if self.requests >= self.request_limit or self.limit_reached:
                self.limit_reached = True
                return web.json_response({"error": {"message": "Episode model budget exhausted"}}, status=429)
            if self.error:
                return web.json_response({"error": {"message": "Episode model adapter failed"}}, status=500)
            self.requests += 1
            wire_name = "cc" if request.path.endswith("messages") else "dsh"
            wire = self.wires[wire_name]
            try:
                body = await request.json()
                translated, tools = self.translate(body, wire_name)
                encoded = encode_prompt(translated, tools, self.processor, self.session.apply_chat_template_kwargs)
                # Training CLI sampling is authoritative; harness defaults must
                # not silently change the behavior policy or top-p support.
                for key in ("temperature", "top_p", "top_k"):
                    body.pop(key, None)
                turn = await call_sglang_generate(
                    encoded.token_ids,
                    self.session,
                    body,
                    adapter=wire,
                    session_id=self.token,
                    multimodal_payload=(
                        {"text": encoded.text, "image_data": encoded.image_data} if encoded.image_data else None
                    ),
                )
                if turn.finish_reason == "abort":
                    raise RuntimeError("SGLang aborted a physical episode; partial rollout is unsupported")
                if not turn.output_ids:
                    self.limit_reached = True
                    return web.json_response({"error": {"message": "Model context exhausted"}}, status=429)
                raw = self.tokenizer.decode(turn.output_ids, skip_special_tokens=False)
                parsed = parse_model_output(
                    raw,
                    tools_schema=tools,
                    tool_parser_name=wire.tool_parser,
                    reasoning_parser_name=wire.reasoning_parser,
                )
                reply = wire._build_reply(parsed, turn.finish_reason, translated, tools)
                response = await wire._respond(
                    request, body, reply, len(encoded.token_ids), len(turn.output_ids), body.get("stream", False)
                )
                self.turns.append((encoded, turn, parsed.ill_formed))
                return response
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Do not include the HTTP body, model credentials, or image data.
                self.error = f"Training adapter {type(exc).__name__}: {exc}"
                return web.json_response({"error": {"message": self.error}}, status=500)

    def samples(self, base, result):
        if self.error or result["status"] not in {"completed", "truncated"} or "reward" not in result:
            raise RuntimeError(
                f"Robot training episode failed ({result['trial_id']}): {result.get('error', self.error)}"
            )
        if not self.turns:
            raise RuntimeError("Robot training episode produced no sampled tokens")
        rid = base.rollout_id if base.rollout_id is not None else base.index
        if rid is None:
            raise ValueError("slime must assign an index or rollout_id before robotics generation")
        samples = []
        for number, (encoded, turn, ill_formed) in enumerate(self.turns):
            sample = Sample(
                index=base.index,
                group_index=base.group_index,
                rollout_id=rid,
                prompt=encoded.text,
                label=base.label,
                tokens=turn.prompt_ids + turn.output_ids,
                response=self.tokenizer.decode(turn.output_ids, skip_special_tokens=False),
                response_length=len(turn.output_ids),
                loss_mask=[1] * len(turn.output_ids),
                rollout_log_probs=turn.output_log_probs,
                reward=float(result["reward"]),
                multimodal_train_inputs=encoded.train_inputs,
                status=(
                    Sample.Status.TRUNCATED
                    if result["status"] == "truncated" or turn.finish_reason == "length"
                    else Sample.Status.COMPLETED
                ),
                session_id=result["trial_id"],
                weight_versions=turn.weight_versions,
                spec_info=turn.spec_info,
                prefix_cache_info=turn.prefix_cache_info,
                metadata={
                    **(base.metadata or {}),
                    **result,
                    "model_turn": number,
                    "image_count": len(encoded.image_data),
                    "ill_formed": ill_formed,
                },
            )
            for key in REPLAY_FIELDS:
                if turn.replay and key in turn.replay:
                    setattr(sample, key, turn.replay[key])
            if getattr(base, "trace", None) is not None:
                sample.trace = base.trace
            samples.append(sample)
        return samples
