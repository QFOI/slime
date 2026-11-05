"""slime rollout-only run (train.py --debug-rollout-only) of a Qwen3.5 VLM on the Newton robot tasks.

Start the robot server first, on its own GPU and never with --allow-privileged:
  python -m examples.robotics_newton.server --world-count 64 --device cuda:1 --port 18100
then:
  MODEL_NAME=Qwen3.5-4B SLIME_SCRIPT_NUM_GPUS=1 python examples/robotics_newton/run_robotics_newton.py

Rollout data is saved with --save-debug-rollout-data. Without the slime Docker
image (Megatron, Ray), rollout_eval.py runs the same hook directly.
"""

import json
import os
import urllib.request

import yaml

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3.5-4B")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))
ROBOT_SERVER_URL = os.environ.get("ROBOT_SERVER_URL", "http://127.0.0.1:18100")
DATA_ROOT = "/root/datasets/robotics_newton"
PROMPT_DATA = f"{DATA_ROOT}/heldout.jsonl"
CONFIG = f"{DATA_ROOT}/rollout_config.yaml"


def prepare():
    U.exec_command("mkdir -p /root/models " + DATA_ROOT)
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"python -m examples.robotics_newton.tasks --split heldout --count 64 --output {PROMPT_DATA}")
    with open("examples/robotics_newton/rollout_config.yaml") as f:
        config = yaml.safe_load(f)
    with open(CONFIG, "w") as f:
        yaml.safe_dump({**config, "robot_server_url": ROBOT_SERVER_URL}, f)
    with urllib.request.urlopen(f"{ROBOT_SERVER_URL}/stats", timeout=10) as response:
        print("robot server:", json.load(response))


def execute():
    rollout_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--prompt-data {PROMPT_DATA} "
        "--input-key messages "
        "--tool-key tools "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--custom-generate-function-path examples.robotics_newton.rollout.generate "
        f"--custom-config-path {CONFIG} "
        "--sglang-tool-call-parser qwen3_coder "
        "--rollout-shuffle "
        "--num-rollout 1 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 24576 "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
        "--debug-rollout-only "
        "--save-debug-rollout-data /root/robotics_newton_rollout/{rollout_id}.pt "
    )
    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 "
    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --rollout-num-gpus {NUM_GPUS} "
    execute_train(
        train_args=f"{rollout_args} {sglang_args} {misc_args}",
        num_gpus_per_node=NUM_GPUS,
        # scripts/models/qwen3.5-4B-vl.sh: only the leading letter is lower case.
        megatron_model_type=f"{MODEL_NAME[0].lower()}{MODEL_NAME[1:]}-vl",
    )


if __name__ == "__main__":
    prepare()
    execute()
