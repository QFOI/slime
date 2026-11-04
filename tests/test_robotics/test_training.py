"""Training boundaries over real HTTP, including images, token replay and GRPO."""

import asyncio
import base64
import io
from types import SimpleNamespace

import aiohttp
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer
from examples.robotics.config import Experiment
from examples.robotics.training import TrainingGateway, validate_training_args
from examples.robotics.vision import image_content
from PIL import Image
from tests.test_robotics.training_fixtures import PixelProcessor

from slime.agent.adapters import anthropic, openai
from slime.utils.types import Sample


def image_uri(color):
    out = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def args_for(server):
    return SimpleNamespace(
        hf_checkpoint="fixture",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=server.port,
        rollout_max_context_len=400000,
        sglang_context_length=400000,
        rollout_top_p=1.0,
        use_score_centering=False,
    )


def test_wire_tool_images_keep_order_and_bytes():
    a, b = image_uri("red"), image_uri("blue")
    blocks = [
        {"type": "text", "text": "camera"},
        {"type": "image_url", "image_url": {"url": a}},
        {"type": "image_url", "image_url": {"url": b}},
    ]
    left = openai._translate_messages([{"role": "tool", "content": blocks}], content_transform=image_content)
    right = anthropic._translate_messages(
        [{"role": "user", "content": [{"type": "tool_result", "content": blocks}]}],
        None,
        content_transform=image_content,
    )
    assert left == right
    assert [i["image"] for i in left[0]["content"] if i["type"] == "image"] == [a, b]
    assert (
        image_content(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": a.split(",")[1]}}
        )[0]["image"]
        == a
    )


@pytest.mark.parametrize("uri", ["file:///secret", "https://remote/image.png", "/tmp/picture.png"])
def test_training_images_cannot_read_host_files_or_remote_urls(uri):
    with pytest.raises(ValueError, match="inline base64"):
        image_content([{"type": "image_url", "image_url": {"url": uri}}])


@pytest.mark.parametrize("agent", ["cc", "dsh"])
@pytest.mark.parametrize("fault", [None, "length", "logprobs", "count"])
def test_multimodal_turns_use_real_sampler_tokens_and_independent_pixels(agent, fault):
    async def run():
        processor, requests = PixelProcessor(), []

        async def generate(request):
            body = await request.json()
            requests.append(body)
            tokens = processor.tokenizer.encode("done")
            meta = {
                "prompt_tokens": len(processor.tokenizer.encode(body["text"])) + (1 if fault == "length" else 0),
                "finish_reason": {"type": "stop"},
                "completion_tokens": len(tokens),
                "output_token_logprobs": [[-0.125, t] for t in tokens],
                "weight_version": "3",
            }
            if fault == "logprobs":
                meta.pop("output_token_logprobs")
            if fault == "count":
                meta["completion_tokens"] += 1
            return web.json_response({"meta_info": meta})

        app = web.Application()
        app.router.add_post("/generate", generate)
        async with TestServer(app) as server:
            gateway = TrainingGateway(
                args_for(server),
                Experiment(tasks=["unused"]),
                agent,
                {"max_new_tokens": 20, "temperature": 0.7},
                processor=processor,
            )
            async with gateway, aiohttp.ClientSession() as client:
                for color in ("red", "blue"):
                    content = [
                        {"type": "text", "text": "view"},
                        {"type": "image_url", "image_url": {"url": image_uri(color)}},
                    ]
                    body = {"messages": [{"role": "user", "content": content}], "temperature": 0.01}
                    path = "/v1/messages" if agent == "cc" else "/v1/chat/completions"
                    headers = {"Authorization": "Bearer " + gateway.token}
                    async with client.post(gateway.url + path, json=body, headers=headers) as response:
                        assert response.status == (500 if fault else 200)
                        await response.read()
                    if fault:
                        break
                async with client.post(gateway.url + path, json=body) as response:
                    assert response.status == 401
            result = {"status": "completed", "trial_id": "trial", "reward": 1}
            if fault:
                with pytest.raises(RuntimeError, match="adapter"):
                    gateway.samples(Sample(index=2), result)
                return
            samples = gateway.samples(Sample(index=2, group_index=1, rollout_id=9), result)
            assert len(samples) == 2 and all(s.rollout_id == 9 and s.group_index == 1 for s in samples)
            for sample in samples:
                assert sample.tokens[-4:] == processor.tokenizer.encode("done")
                assert sample.loss_mask == [1] * 4 and sample.rollout_log_probs == [-0.125] * 4
                assert not sample.remove_sample and sample.weight_versions == ["3"]
            assert not torch.equal(
                samples[0].multimodal_train_inputs["pixel_values"], samples[1].multimodal_train_inputs["pixel_values"]
            )
            assert all("input_ids" not in r and r["image_data"] for r in requests)
            assert all(r["sampling_params"]["temperature"] == 0.7 for r in requests)

    asyncio.run(run())


def test_grpo_normalizes_unequal_segments_per_episode():
    from slime.data.batch_builder import BatchBuilder

    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        reward_key=None,
        use_score_centering=False,
        use_rollout_routing_replay=False,
    )
    samples = [
        Sample(
            index=i,
            rollout_id=i,
            group_index=0,
            reward=r,
            tokens=[1, 2],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.5],
        )
        for i, r in [(0, 1), (0, 1), (0, 1), (1, 0)]
    ]
    samples[1].multimodal_train_inputs = {"pixel_values": torch.ones(1, 3, 8, 8)}
    data = BatchBuilder(args).convert(samples)
    assert data["rewards"] == [0.5, 0.5, 0.5, -0.5]
    assert data["rollout_mask_sums"] == [3, 3, 3, 1]
    assert data["multimodal_train_inputs"][0] is None
    assert torch.equal(data["multimodal_train_inputs"][1]["pixel_values"], torch.ones(1, 3, 8, 8))


def test_training_requires_actor_and_rejects_physical_restarts():
    with pytest.raises(ValueError, match="hf-checkpoint"):
        validate_training_args(SimpleNamespace())
    args = args_for(SimpleNamespace(port=8000))
    args.partial_rollout = True
    with pytest.raises(ValueError, match="partial rollout"):
        validate_training_args(args)


def test_multimodal_training_data_survives_slime_disk_transport(tmp_path):
    from slime.data.tensor import TensorRef
    from slime.data.transport import pack_rollout_payload

    args = SimpleNamespace(rollout_data_dir=str(tmp_path), rollout_data_transport="straw")
    pixels = torch.randn(2, 3, 8, 8)
    original = Sample(
        index=3,
        group_index=1,
        rollout_id=3,
        tokens=[8, 9, 10],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.42],
        reward=1,
        multimodal_train_inputs={"pixel_values": pixels},
        weight_versions=["7"],
    )
    loaded = pack_rollout_payload([original], args, 0).load()[0]
    value = loaded.multimodal_train_inputs["pixel_values"]
    if isinstance(value, TensorRef):
        value = value.load()
    assert torch.equal(value, pixels)
    assert list(loaded.tokens) == original.tokens and loaded.rollout_id == original.rollout_id
    assert loaded.weight_versions == ["7"] and loaded.remove_sample is False
