#!/usr/bin/env bash
# Single-node Qwen3.5-0.8B VL / GRPO bring-up. Requires a slime training image.
# No pkill, Ray stop, dependency replacement, or external API credentials.
set -euo pipefail
ROBOT_SLIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROBOT_SLIME_ROOT"
ROBOT_TRAIN_CONFIG="${ROBOT_TRAIN_CONFIG:-$ROBOT_SLIME_ROOT/examples/robotics/configs/training.yaml}"
ROBOT_HF_CHECKPOINT="${ROBOT_HF_CHECKPOINT:-/models/Qwen3.5-0.8B}"
ROBOT_TRAIN_OUTPUT="${ROBOT_TRAIN_OUTPUT:-$ROBOT_SLIME_ROOT/examples/robotics/runs/train-$(date +%Y%m%d-%H%M%S)}"
ROBOT_TRAIN_STEPS="${ROBOT_TRAIN_STEPS:-3}"
ROBOT_PYTHON="${ROBOT_PYTHON:-python}"
mkdir -p "$ROBOT_TRAIN_OUTPUT"
export ROBOTICS_CONFIG="$ROBOT_TRAIN_CONFIG"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Reuse the upstream native language-model dimensions, with its native VL wrapper.
source scripts/models/qwen3.5-0.8B.sh
MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"
"$ROBOT_PYTHON" -m examples.robotics manifest "$ROBOT_TRAIN_CONFIG" --output "$ROBOT_TRAIN_OUTPUT/tasks.jsonl"

ROBOT_TRAIN_COMMAND=(
  "$ROBOT_PYTHON" train.py
  "${MODEL_ARGS[@]}"
  --train-backend megatron
  --hf-checkpoint "$ROBOT_HF_CHECKPOINT" --load "$ROBOT_HF_CHECKPOINT"
  --save "$ROBOT_TRAIN_OUTPUT/checkpoints" --save-interval 1
  --actor-num-nodes 1 --actor-num-gpus-per-node 1
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 2
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --micro-batch-size 1
  --attention-dropout 0 --hidden-dropout 0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --prompt-data "$ROBOT_TRAIN_OUTPUT/tasks.jsonl" --input-key prompt --metadata-key metadata
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking": false}'
  --custom-generate-function-path examples.robotics.generate.generate
  --num-rollout "$ROBOT_TRAIN_STEPS"
  --rollout-batch-size 1 --n-samples-per-prompt 4 --global-batch-size 4
  --rollout-max-context-len 16384 --rollout-max-response-len 2048
  --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1
  --sglang-context-length 16384 --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3 --sglang-server-concurrency 2
  --sglang-mem-fraction-static 0.6 --sglang-max-running-requests 2
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28
  --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
  --eval-interval 1 --skip-eval-before-train
  --eval-prompt-data robotics "$ROBOT_TRAIN_OUTPUT/tasks.jsonl"
  --n-samples-per-eval-prompt 1 --eval-max-response-len 2048
  --save-debug-rollout-data "$ROBOT_TRAIN_OUTPUT/rollout-{rollout_id}.pt"
  --save-debug-train-data "$ROBOT_TRAIN_OUTPUT/train-{rollout_id}.pt"
)
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${ROBOT_TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi
"$ROBOT_PYTHON" -m examples.robotics.preflight "$ROBOT_TRAIN_CONFIG" \
  --checkpoint "$ROBOT_HF_CHECKPOINT" --gpus 2 --output "$ROBOT_TRAIN_OUTPUT/preflight.json"
exec "${ROBOT_TRAIN_COMMAND[@]}" "$@"
