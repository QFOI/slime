# Real GPU training validation — 2026-11-04

Two completed jobs passed **eight real Megatron updates**, actor-to-SGLang weight synchronization, three evaluation rounds, eight distributed checkpoint saves and a full checkpoint restart on **two NVIDIA H20 GPUs**. They used the pretrained **Qwen/Qwen3.5-0.8B** vision-language model, official CC/DSH CLIs and the real Panda Reach MuJoCo environment. This supersedes the GPU-pending status of the earlier [CPU validation](../training/README.md).

**This is training infrastructure validation, not evidence of improved robot performance.** All task rewards and GRPO advantages were zero, including after recovery. The explicit `entropy_coef=0.001` produced the nonzero gradients; `weight_decay=0` excludes weight decay as the explanation for parameter changes. No rewards, successful tool calls, images or model outputs were substituted with fixtures.

## Initial run

Evidence: [initial.json](initial.json), [initial-checkpoint.json](initial-checkpoint.json), [runtime.json](runtime.json).

| Update | Harness / profile | Episodes | Sample segments | Generated tokens | Image exposures in training contexts | Gradient norm | Serving weight version |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | CC / intuitive | 4 | 32 | 6,669 | 62 | 0.01115843 | 1 |
| 1 | CC / expert | 4 | 21 | 3,632 | 4 | 0.01669075 | 2 |
| 2 | DSH / intuitive | 4 | 32 | 6,355 | 40 | 0.01465319 | 3 |
| 3 | DSH / expert | 4 | 32 | 24,440 | 0 | 0.00676867 | 4 |

Image exposures count repeated images in successive request histories, not distinct camera frames. Every retained output has matching token IDs, loss masks and finite sampled log probabilities. Image-bearing samples retain nonempty `pixel_values`. The corresponding train dumps contain finite recomputed log probabilities with matching lengths. Mean absolute differences per rollout are 0.00424–0.00957; the largest token difference is 0.53275. The two implementations use different BF16 kernels; this is measured agreement, not bitwise equivalence or a proof of convergence.

All 16 training episodes and eight evaluation episodes were independently scored after robot access was frozen. Evaluation statuses were one `completed` and seven `truncated`; none was `aborted`. A completed harness episode can still have task reward 0. Both evaluation rounds scored 0.

DSH expert had PTC enabled and generated `run_code` calls, but the initial policy repeatedly submitted Python to DSH's JavaScript runtime. **Successful PTC subcalls were zero in this initial run.** The resulting tool errors were returned to the model and the sampled trajectories retained. The expert prompt now explicitly distinguishes JavaScript `run_code` from Python `run_python`, with a valid observation example. The original prompt hashes are recorded in the evidence; the initial run must not be attributed to the revised prompt.

The saved visual patch embedding has 136,007 changed BF16 elements relative to the pinned HF checkpoint (L1 difference 0.33708438), and its FP32 master parameters and Adam moments also change. The language final-normalization FP32 master changes even though its BF16 copy rounds to the same values at this small learning rate. Checkpoint metadata includes RNG state, scheduler progress 16, and dataset sample index 16.

## Checkpoint recovery

The successful recovery job loaded iteration 3 into a fresh training runtime and completed updates 4–7, using the revised expert prompt and full layer recomputation. It exited 0. Optimizer/RNG loading remained enabled, `finetune` was false, and `--use-checkpoint-opt-param-scheduler` retained the saved scheduler. The dataset cursor advanced from sample index 16 / epoch 0 to sample index 32 / epoch 1; both checkpoints retain 1,311 optimizer tensor keys and RNG metadata. This is a continuation with a prompt clarification, not an exact replay experiment. Evidence: [resume.json](resume.json), [recovery-state.json](recovery-state.json).

| Update | Harness / profile | Episodes | Sample segments | Generated tokens | Image exposures | Gradient norm |
|---|---|---:|---:|---:|---:|---:|
| 4 | CC / intuitive | 4 | 29 | 5,971 | 40 | 0.02260245 |
| 5 | CC / expert | 4 | 13 | 2,345 | 6 | 0.02100279 |
| 6 | DSH / intuitive | 4 | 11 | 2,219 | 8 | 0.05213429 |
| 7 | DSH / expert | 4 | 31 | 16,272 | 0 | 0.01300062 |

The final evaluation had three `completed` episodes and one `truncated`, all independently scored 0. Across both runs there are 32 training episodes, 12 evaluation episodes, 201 trainable segments and 67,903 generated training tokens; none of the 44 episodes is `aborted`. Serving weight-version counters restart at 1 in the new serving session and advance to 4; these counters are session-local and must not be interpreted as global training iterations or evidence that original HF weights were reloaded.

The prompt clarification did **not** resolve this small model's PTC behavior: all five resumed DSH expert episodes still had zero successful PTC subcalls and no image-bearing training segment. The harness correctly returned model-program errors and trained on the resulting failure trajectories. Successful PTC/depth execution in the earlier CPU protocol fixture is separate evidence, not a successful pretrained-Qwen result.

## Runtime and scope

The existing outer Ray cluster only reserves two GPUs and launches an isolated GPU container. Training and SGLang run inside its separate Ray runtime; unrelated outer jobs remain running. The exact package/model pins and parallelism are in [runtime.json](runtime.json). The GPU image was pinned to Megatron commit `4b4acac9a1d28ea6829c8d4f566d75698a21249d` plus the repository's two SGLang 0.5.21 Megatron patches. A newer Megatron tree in a floating image was incompatible; do not assume an arbitrary `latest` image is equivalent.

The simulator uses a separate Python environment with NumPy 1.26.4; the GPU environment uses NumPy 2.3.5. Both use robot-rollout 0.5.1. CC is 2.1.296; DSH and its ACP adapter are 0.1.0-rc.6. The simulation/robot MCP contract and independent verifier are unchanged.

The first run had **eight recovered CUDA allocator OOM warnings**; the resumed run with full layer recomputation had **four**. Neither had a fatal `OutOfMemoryError`; each had one nonfatal SGLang `freeze_gc` startup connection warning. Both exited 0. Trajectories differ, so these counts are not a controlled memory-performance comparison, and recomputation does not eliminate allocation retries. No lower-memory GPU, TP/PP/CP > 1, multi-node model training, Lift training, real robot training, compaction, multiagent, or learning improvement is claimed by this test.

## Reproduction

Run inside a compatible prepared Linux GPU runtime with two visible GPUs, Docker access, the robot sandbox image, OSMesa and Bubblewrap. Set `session_python` in a copy of the YAML if simulation dependencies use a separate environment. Launching directly in the outer scheduler's unrelated Python environment is not equivalent.

```bash
export ROBOT_HF_CHECKPOINT=/models/Qwen3.5-0.8B
export ROBOT_TRAIN_CONFIG="$PWD/examples/robotics/configs/training-smoke.yaml"
export ROBOT_TRAIN_OUTPUT=/runs/robotics-first
export ROBOT_TRAIN_STEPS=4
bash examples/robotics/train.sh \
  --rollout-max-response-len 1024 --eval-max-response-len 1024 \
  --eval-interval 2 --entropy-coef 0.001 --weight-decay 0

# --num-rollout is the total target count, not the number of additional updates.
ROBOT_TRAIN_OUTPUT=/runs/robotics-resumed ROBOT_TRAIN_STEPS=8 \
  bash examples/robotics/train.sh \
  --load /runs/robotics-first/checkpoints \
  --use-checkpoint-opt-param-scheduler \
  --rollout-max-response-len 1024 --eval-max-response-len 1024 \
  --eval-interval 4 --entropy-coef 0.001 --weight-decay 0
```

Keep optimizer/RNG loading enabled; do not add `--finetune`, `--no-load-optim` or `--no-load-rng`. Extending the target update count changes Megatron's inferred scheduler length: `--use-checkpoint-opt-param-scheduler` explicitly retains the saved schedule and progress. Without it Megatron correctly rejects the mismatch; this was exposed by the first recovery attempt. This test uses a constant learning rate, so retaining the old schedule length does not terminate updates. Retain the four manifest rows and their order when validating dataset progress. The current launcher enables layer recomputation; the initial historical run above did not.

[summarize.py](summarize.py) validates completed local debug dumps and exports sanitized evidence. [checkpoint_delta.py](checkpoint_delta.py) reads selected distributed-checkpoint tensors and compares them with the pinned HF safetensors file. Large model/checkpoint/debug files, prompts, camera images, host addresses and credentials are not embedded in the published evidence.

```bash
python examples/robotics/validation/gpu-training/summarize.py /runs/robotics-first
python examples/robotics/validation/gpu-training/checkpoint_delta.py \
  /runs/robotics-first/checkpoints /models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors \
  /runs/robotics-first/checkpoint-delta.json
```

The changed integration and native-VL regression suite passed **66 tests, one skipped**. Black, Ruff, shell syntax and diff checks passed. The smoke YAML generates exactly four manifest rows. Earlier failed attempts were retained as diagnostics and are not counted as passing runs.
