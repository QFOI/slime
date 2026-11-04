"""Prompt rows for slime: one jsonl row per (task, seed).

python -m examples.robotics_newton.tasks --tasks lift reach_pose --split heldout --count 64 --output heldout.jsonl

Each row holds text-only chat messages (images arrive through observe), the
tool schemas, and metadata {"task", "seed", "split"}. Train and held-out seeds
come from disjoint ranges. Load with:
  --input-key messages --tool-key tools --metadata-key metadata --apply-chat-template
"""

from __future__ import annotations

import argparse
import json

from .tools import TOOLS

INSTRUCTIONS = {
    # Verbatim from examples/robotics/tasks/*/instruction.md.
    "lift": (
        "Pick up the red cube with the Panda gripper and hold it above the table for at least 0.5 simulated "
        "seconds. Ground each motion in the current camera observations and robot measurements; verify the "
        "grasp and lift visually."
    ),
    "reach_pose": (
        "Move the Panda end effector to the public goal pose returned by observe. Satisfy both position and "
        "orientation and hold the pose for at least 0.5 simulated seconds. Use the robot tools; do not inspect "
        "the simulator or evaluator."
    ),
}

SPLITS = {"train": range(0, 100_000), "heldout": range(1_000_000, 1_100_000)}

# The intuitive-mode rules of examples/robotics/prompts/intuitive.md, plus how an episode ends here.
SYSTEM_PROMPT = """\
You control a robot through typed tools. First call observe. Use the labelled RGB views and measured TCP pose to \
complete the task. Inspect each returned observation before deciding the next action. The independent verifier \
decides success from physical evidence; your final text does not change the score.

Only observe, move_to, gripper and batch_actions are available. Depth, joint information, code and hidden simulator \
state are unavailable. batch_actions is a bounded literal list of tool calls, not a programming language.

move_to takes RELATIVE base-frame XYZ displacement in metres and RELATIVE base-frame axis-angle rotation in radians. \
Rotation acts about the current TCP; it is not Euler angles or an absolute quaternion. Use the measured pose as the \
anchor. Position and rotation bounds are vector-norm limits. Check the actual residual after every motion; never \
blindly repeat a delta following partial execution.

Use the newest observation_id for actions and a unique command_id for each physical command. Reusing an identical \
command_id retrieves its receipt; changing its arguments is forbidden. gripper accepts open or close. A closed \
gripper is not proof of a grasp: check the images after closure and lifting.

batch_actions allows at most three physical commands followed by observe. After a gripper command only observe may \
follow. Batch only motions that do not require an intervening visual decision. Stop on partial, rejected or unknown \
execution and inspect the new observation.

Make one tool call per reply. When the requested physical condition is visibly met, reply without a tool call to \
end the episode; the evaluator then checks the robot's state."""


def row(task: str, seed: int, split: str) -> dict:
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": INSTRUCTIONS[task]}],
        "tools": TOOLS,
        "metadata": {"task": task, "seed": seed, "split": split},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=sorted(INSTRUCTIONS), default=sorted(INSTRUCTIONS))
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--count", type=int, default=1024, help="seeds per task")
    parser.add_argument("--offset", type=int, default=0, help="first seed index within the split")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = SPLITS[args.split][args.offset : args.offset + args.count]
    if len(seeds) < args.count:
        raise SystemExit(f"split {args.split} has only {len(SPLITS[args.split])} seeds")
    with open(args.output, "w") as f:
        for seed in seeds:
            for task in args.tasks:
                f.write(json.dumps(row(task, seed, args.split)) + "\n")
    print(f"wrote {len(seeds) * len(args.tasks)} rows to {args.output}")


if __name__ == "__main__":
    main()
