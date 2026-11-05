# hy-stem training integration validation — 2026-11-04

**Historical CPU validation: CPU training integration passed.** Subsequent real GPU training and checkpoint recovery are documented in the [GPU validation report](../gpu-training/README.md). The results below remain CPU-only evidence.

The host has no CUDA GPU and runs PyTorch `2.6.0+cpu`; Megatron is absent. The complete Qwen3.5-0.8B checkpoint has been staged on shared storage. [preflight.json](preflight.json) records the remaining resource/runtime failures. Do not treat these CPU results as a pretrained robot policy benchmark or as evidence that GPU training, weight synchronization or checkpoint recovery works.

Staged checkpoint: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, 1,746,942,600 safetensors bytes / 488 tensors. Model weights are not committed to this repository.

## Official harnesses and real robot simulation

[cpu-training.json](cpu-training.json) records two successive optimization steps, each using CC/DSH × intuitive/expert × two sampled trajectories on Panda Reach: **16 episodes completed**, with eight valid reward=1 and eight valid reward=0 outcomes. These mixed outcomes are deliberate training-fixture behavior, not a claimed 100% robot success rate. The simulator and independent verifier compute all rewards; no score is injected.

The CPU model is an explicit finite-grammar protocol fixture. A trainable RGB policy chooses between stopping and a public-pose controller action. Its stochastic first token has the actual normalized policy log probability; remaining grammar tokens have probability 1. This fixture is served through the production SGLang-protocol adapter, but it is **not the native SGLang inference engine or an LLM**.

| Check | Result |
|---|---|
| Official CC / DSH | 2.1.296 / 0.1.0-rc.6 |
| Real MuJoCo + HTTP MCP + independent verifier | 16 episodes completed |
| Expert depth Python / DSH PTC | Exercised successfully |
| Model image inputs | 80 images across all requests |
| Trainable segments / tokens | 24 / 4,648 per step |
| Step 1 gradient norm / parameter L1 change | 0.00264835 / 0.15999826 |
| Step 2 gradient norm / parameter L1 change | 0.00277462 / 0.16014910 |
| Sampler/replayed logprob error | 0 on both steps |
| Weight version | 0 → 1 → 2; second rollout used the updated policy |
| Saved CPU policy checkpoint reload | Exact tensor equality |

The test uses the real slime `Sample`, `BatchBuilder`, rollout-level GRPO normalization, `get_grpo_returns` and `compute_policy_loss`. It updates a CPU policy with Adam and saves/reloads that policy/optimizer state. It does **not** execute `train.py`, a Megatron actor, distributed weight transfer or a Megatron checkpoint.

## Actual Hugging Face vision tensors and backward

[hf-vision.json](hf-vision.json) uses the official Qwen3.5 processor and a randomly initialized, reduced Qwen3.5 HF model: two full-attention text layers and one vision layer. Two images expand to 128 image tokens. The test uses slime's native packed MRoPE position builder, samples two real model tokens, and replays them for a policy-gradient update.

- Maximum sampled/recomputed logprob error: `9.536743e-7`.
- Vision gradient norm sum: `4.542698`; language-model parameters also change.
- A separately rendered compacted prompt retains only its one surviving image/grid.
- This validates image/token/position alignment and gradient flow on CPU; it does not use the staged pretrained weights or exercise the full hybrid model architecture.

## Regression and runtime details

`pytest tests/test_robotics tests/test_agent -q`: **263 passed, 2 skipped**. Coverage includes image preservation in both protocols, incomplete/invalid token replay rejection, distinct images per turn, actual slime disk-transport round trip, unequal trajectory segment counts under GRPO, and the existing coding-agent harness/adapter tests. Black, isort, Ruff, shell syntax and diff checks pass; the launch script's dry-run generates a manifest and train command.

CPU validation packages: Python 3.12.13, torch 2.6.0+cpu, torchvision 0.21.0+cpu, transformers 5.22.1, Ray 2.54.0, straw-queue 0.1.8. SGLang 0.5.9 is installed **only for its official function-call parser**. This is not the GPU runtime lock; the root slime Dockerfile currently uses SGLang v0.5.21 and its pinned/patched Megatron tree.

The initial CPU integration attempt exposed DSH rejecting slime's `top_k=-1`. The final implementation sends training sampling settings directly to SGLang, bypassing the external-API sampling schema. Its regression and the subsequent two-step matrix pass. Earlier failed episode artifacts were retained. All test-created robot sessions and containers were stopped after the run.

## Reproduction

```bash
python -m pytest tests/test_robotics tests/test_agent -q
python -m tests.test_robotics.train_cpu_smoke --steps 2
python -m tests.test_robotics.hf_vision_smoke --processor /path/to/Qwen3.5-0.8B
bash examples/robotics/train.sh --dry-run
```

See [TRAINING.md](../../TRAINING.md) for the GPU launch recipe and current validation scope. No external API credential is needed for these training integration tests.
