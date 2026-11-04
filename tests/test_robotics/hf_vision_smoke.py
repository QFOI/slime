"""Real HF Qwen3.5 image/token replay and backward on a tiny random CPU model.

Requires a locally downloaded Qwen3.5 processor/config, not pretrained weights.
This isolates multimodal tensor correctness; it does not validate Megatron or
the native SGLang engine, and does not measure robot task performance.
"""

import argparse
import json
from pathlib import Path

import torch
from examples.robotics.training import load_training_processor
from examples.robotics.vision import encode_prompt
from tests.test_robotics.test_training import image_uri
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

from slime.utils.ppo_utils import compute_policy_loss
from slime_plugins.models.qwen3_5_vl_utils import build_packed_mrope_position_ids


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    processor = load_training_processor(args.processor)
    config = AutoConfig.from_pretrained(args.processor)
    for key, value in {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "layer_types": ["full_attention", "full_attention"],
        "mtp_num_hidden_layers": 0,
    }.items():
        setattr(config.text_config, key, value)
    config.text_config.rope_parameters.update(partial_rotary_factor=1.0, mrope_section=[2, 3, 3])
    for key, value in {
        "hidden_size": 32,
        "intermediate_size": 64,
        "out_hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "deepstack_visual_indexes": [],
    }.items():
        setattr(config.vision_config, key, value)
    model = Qwen3_5ForConditionalGeneration(config).float().eval()
    model.set_attn_implementation("eager")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    user = {"role": "user", "content": "Choose the next robot action from these camera views."}
    views = [
        {
            "role": "tool",
            "content": [{"type": "text", "text": f"camera {color}"}, {"type": "image", "image": image_uri(color)}],
        }
        for color in ("red", "blue")
    ]
    encoded = encode_prompt([user, *views], None, processor, {"enable_thinking": False})
    grid = encoded.train_inputs["image_grid_thw"]
    expected_image_tokens = int(grid.prod(-1).sum()) // config.vision_config.spatial_merge_size**2
    assert encoded.token_ids.count(config.image_token_id) == expected_image_tokens
    assert len(encoded.image_data) == 2
    compact = encode_prompt([user, views[-1]], None, processor, {"enable_thinking": False})
    assert compact.train_inputs["image_grid_thw"].shape[0] == 1
    tokens = torch.tensor([encoded.token_ids])

    # Exercise the same position builder used by slime's native Megatron VL
    # wrapper. HF's recent mm_token_type_ids are sequence-shaped and must not
    # enter Megatron's concatenated image-tensor inputs.
    def hf_inputs(ids):
        positions = build_packed_mrope_position_ids(
            ids,
            [0, ids.shape[1]],
            encoded.train_inputs["image_grid_thw"],
            None,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        )
        return {"input_ids": ids, "position_ids": positions, **encoded.train_inputs, "use_cache": False}

    sampled_ids, old_logps = [], []
    with torch.no_grad():
        for _ in range(2):
            logits = model(**hf_inputs(tokens), logits_to_keep=1).logits[0, -1]
            logps = (logits / 0.8).log_softmax(-1)
            token = torch.multinomial(logps.exp(), 1)
            sampled_ids.append(int(token))
            old_logps.append(logps[token].squeeze())
            tokens = torch.cat((tokens, token.reshape(1, 1)), dim=1)
    output = model(**hf_inputs(tokens[:, :-1]), logits_to_keep=2)
    current = (output.logits[0] / 0.8).log_softmax(-1).gather(1, torch.tensor(sampled_ids)[:, None]).squeeze(1)
    old = torch.stack(old_logps)
    error = float((current.detach() - old).abs().max())
    assert error < 2e-5, error
    pg, _ = compute_policy_loss(old - current, torch.ones_like(current), 0.2, 0.28)
    loss = pg.mean()
    loss.backward()
    vision_grad = sum(
        float(p.grad.float().norm()) for n, p in model.named_parameters() if ".visual." in n and p.grad is not None
    )
    assert vision_grad > 0 and torch.isfinite(loss)
    before = model.lm_head.weight.detach().clone()
    optimizer.step()
    delta = float((model.lm_head.weight.detach() - before).abs().sum())
    assert delta > 0
    report = {
        "kind": "random_tiny_hf_qwen35_cpu",
        "gpu_training_verified": False,
        "processor": type(processor).__name__,
        "image_count": len(encoded.image_data),
        "image_token_count": expected_image_tokens,
        "prompt_tokens": len(encoded.token_ids),
        "sampled_tokens": len(sampled_ids),
        "replay_logprob_error": error,
        "vision_grad_norm_sum": vision_grad,
        "lm_parameter_delta_l1": delta,
        "compacted_image_count": compact.train_inputs["image_grid_thw"].shape[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--output", type=Path, default=Path("examples/robotics/runs/hf-vision-smoke.json"))
    main(parser.parse_args())
