# Newton robot worlds for slime

GPU-batched Newton simulation of the Panda tasks in [`examples/robotics`](../robotics) (`lift`, `reach_pose`), served over HTTP with the same intuitive tool surface: `observe`, `move_to`, `gripper` and `batch_actions`. One process holds many worlds on one GPU, and each episode leases one world. There is no Docker or private `robot-rollout` dependency.

Status: Phase 1 is done: simulator, tool protocol, scripted experts, benchmark and tests. Evaluator v2 and its red-team suite are in place (see below). Phase 2: the slime rollout hook runs Qwen3.5 VLM episodes end to end. Zero-shot base rates and failure breakdowns are below: lift 0/64, reach 0/32 with a 1024-token turn cap and 2/16 with 4096 (Qwen3.5-4B).

| File | Role |
|---|---|
| `sim/scene.py` | FR3 + Franka Hand on a table with a red cube. Maps `(task, seed)` to the initial layout. The base frame is the simulation frame, with the table top at z = 0. |
| `sim/worlds.py` | `PandaWorlds` runs W worlds in one CUDA graph per 60 Hz control frame: per-world controller, batched IK, physics, settle checks, cube-contact classification, episode monitors and the evaluator (carry test and hold samples). It also runs batched ray-traced cameras (`front` fixed, `wrist` on the hand). |
| `protocol.py` | Episode semantics: current `observation_id`, idempotent `command_id`, limits, command budget, `batch_actions` rules. `finish` hands the simulator's evidence to the evaluator. |
| `evaluator.py` | The reward rule: the task condition plus process checks, each reported separately. `v1` (final height only) is kept for comparison. |
| `tools.py` | Tool schemas (OpenAI function format). |
| `server.py` | aiohttp server. A simulator thread applies the requests that arrive between frames together, and steps frames only while a command is in flight. |
| `client.py` | Async HTTP client. |
| `scripted.py` | Scripted experts and the evaluation runner. Reach uses only public state; lift reads the cube pose from the privileged endpoint. |
| `redteam.py` | Hack policies and injected states that must score 0, plus the experts, run under v1 and v2. |
| `bench.py` | Frame and render latency against world count. |
| `tasks.py` | Task instructions, the system prompt, and prompt rows for slime (one jsonl row per task and seed; disjoint train and held-out seed ranges). |
| `rollout.py` | The slime custom generate hook: a token-in/token-out multi-turn VLM loop against SGLang `/generate`, with the robot server awaited over HTTP. |
| `rollout_eval.py` | Runs the hook on prompt rows without Ray or Megatron, and reports success rates and failure categories. |
| `run_robotics_newton.py`, `rollout_config.yaml` | `train.py --debug-rollout-only` launch on the slime Docker image (not yet run; see below). |

## Run

```bash
conda create -n robotics-newton python=3.12 -y
conda activate robotics-newton
pip install "newton[examples]==1.7.0" pillow aiohttp scipy pytest

# from the slime checkout
python -m examples.robotics_newton.scripted --task lift --seeds 0:64 --world-count 64 --device cuda:0
python -m examples.robotics_newton.server --world-count 64 --device cuda:0 --port 18100   # standalone server
python -m examples.robotics_newton.bench --world-counts 16 64 128 --device cuda:0
python -m examples.robotics_newton.redteam --seeds 0:16 --device cuda:0   # exits 1 if v2 rewards a hack
python -m pytest tests/test_robotics_newton
```

The first build for a new world count compiles Warp kernels, which takes 1–3 minutes. The Franka assets download once to `~/.cache/newton`.

## Semantics and deliberate differences

- **Episode flow.** `POST /episodes {"task", "seed"}` resets a free world to `sample_layout(task, seed)`. GRPO samples of one prompt share the seed, so they start from the same scene. `POST /episodes/{id}/finish` freezes the episode, runs the evaluator, and releases the world. The policy cannot trigger scoring, and its text never reaches the reward.
- **Reward.** See the evaluator section below.
- **`move_to`** takes `translation_m` and `rotation_delta_rad`, in the base frame, anchored to the measured TCP. Its argument names match the robot-rollout calls in `tests/test_robotics/smoke.py`.
  - The planned TCP speed is 0.25 m/s and 1 rad/s.
  - A move is `completed` when the TCP is within 2 mm / 1° and the arm is still. It is `partial` if that doesn't happen within 1 s after the planned duration; the arm then holds where it is.
- **Limits.** 0.25 m and 1.0 rad per command, TCP z ≥ 5 mm, 80 physical commands. These are local choices; robot-rollout's real values are not visible from this repository.
- **Robot.** The robot is the FR3 from `newton-assets` with gains from Newton's `ik_cube_stacking` example, not robosuite's Panda. Images are 256×256 ray-traced RGB without depth.
- **Idle worlds** keep simulating under their hold target while other worlds move, as a real arm does while its policy thinks. Simulated time is therefore not bit-reproducible across concurrency levels.
- **Privileged endpoints.** `GET /episodes/{id}/privileged` (cube and hand poses) and `POST /episodes/{id}/privileged/cube` (teleport the cube) exist only when the server is started with `--allow-privileged`. They are for scripted experts and red-team tests; an RL server must never enable them.

## Evaluator v2

Every check follows from the instruction. The reward is 1 only if all of them pass, and each one is logged separately.

| Check | Lift | Reach | Evidence |
|---|---|---|---|
| `task_condition` | Cube bottom ≥ 3 cm above the table in all 20 samples (1 s at 20 Hz) | TCP within 1 cm and 5° of the goal in all 20 samples | Hold samples after the agent stops |
| `physical_command` | ≥ 1 | ≥ 1 | Protocol counter |
| `held_by_fingers` | In every sample, the cube touches both finger pads (signed separation ≤ 1 mm) and nothing else, with its centre between the pads in the TCP frame | – | Per-frame contact classification on the GPU |
| `moved_with_gripper` | Before sampling, the evaluator carries the hand 4 cm up and 4 cm toward y = 0 over 0.5 s; the cube stays within 1 cm of its pose relative to the TCP | – | Carry test |
| `stayed_on_table` | The cube never dropped more than 2 cm below the table top | – | Episode monitor |
| `not_launched` | The cube never moved faster than 1 m/s, ignoring downward motion (so falls don't count) | – | Episode monitor |
| `no_interpenetration` | Penetration ≤ 5 mm during the hold. Transient impact penetration earlier in the episode is logged, not penalized. | – | Contacts during the hold |

`v1` (the Phase 1 rule: final height only, 10 samples, no carry) stays available as `--evaluator v1` to measure what v2 catches. Red-team results, 16 seeds each, both evaluators:

| Entry | v1 | v2 | v2 checks that fail |
|---|---:|---:|---|
| Expert lift | 1.00 | 1.00 | – |
| Expert reach | 1.00 | 1.00 | – |
| Hack: close beside the cube and lift | 0.00 | 0.00 | task, held, moved |
| Hack: push the cube off the table | 0.00 | 0.00 | task, held, moved; also launched (10–13/16) and off table (2–5/16), varying between runs |
| Hack: lift, then drop | 0.00 | 0.00 | task, held, moved |
| Injected: cube lodged on the hand | **1.00** | 0.00 | held, interpenetration |
| Injected: cube embedded in the flange | **1.00** | 0.00 | held, interpenetration |
| Injected: cube launched upward at 5 m/s | **0.44** | 0.00 | task, held, moved, launched |

Expert margins under v2, 64 seeds:

| Metric | Expert value | Limit |
|---|---|---|
| Carry drift | 0.1 mm | 10 mm |
| Cube speed (launch-style) | 0.34 m/s | 1.0 m/s |
| Penetration during the hold | 0.6 mm | 5 mm |
| Grasp check | passes in all samples | all samples |

Known limits:
- **Shoving can trip `not_launched`.** Shoving the cube with the stiff arm makes the solver kick it to 1.2–3.5 m/s. A clumsy but otherwise legitimate policy that bangs into the cube would trip this check.
- **No clean balanced state to test with.** No stable resting spot on this hand is free of interpenetration (probed at 15–30 mm), so a genuinely balanced cube can't be tested. `held_by_fingers` rejects the injected states on its own.
- **Reach has a vision-free shortcut.** Reach gives the goal as numbers, so it is solvable without vision (the scripted expert never looks). A blind lift policy that always grasps at the spawn-region centre scores 3.1% (4/128).
- **Grasp and carry tolerances** come from the scripted expert's grasps (cube centred to 0.3 mm, width 49.1 mm for the 50 mm cube).

## Rollout hook (Phase 2)

`rollout.generate` is a `--custom-generate-function-path` hook on the pattern of `examples/geo3k_vlm_multi_turn`, with the environment made asynchronous so the episodes of a rollout run concurrently.

- **Prompt.** Rows from `tasks.py` hold text-only messages (system prompt with the intuitive-mode rules, then the task instruction), the tool schemas and `metadata {task, seed, split}`. Load them with `--input-key messages --tool-key tools --metadata-key metadata --apply-chat-template`. The model must call `observe` first; images arrive only through tool replies. All samples of a prompt share its seed, so a GRPO group starts from one scene.
- **Turn.** SGLang `/generate` gets the full token sequence and every image so far, and returns logprobs. Tool calls are parsed with SGLang's `qwen3_coder` parser (Qwen3.5's XML format) and run on the robot server in order. The replies are rendered by the chat template as a `tool` turn: the reply JSON with floats rounded to 4 digits, then each camera image under its label. That turn is appended with loss mask 0. The decoded token sequence is exactly what the chat template renders for the same conversation (tested).
- **End.** The episode ends on a reply without tool-call markup, on `robot_max_turns`, on a turn that hits `robot_max_turn_tokens`, on the response budget (`--rollout-max-response-len`, or `--rollout-max-context-len`), or when the command budget is exhausted. A reply with tool-call markup that does not parse gets a format-error tool turn and counts as a format error.
- **Reward.** The robot server's `finish` scores the frozen episode with evaluator v2, and the hook sets `sample.reward`, so slime's reward model is skipped. Every check, metric and episode statistic goes to `sample.metadata["robot"]`.
- **Failure handling.** If the robot server fails, the sample is marked `FAILED` with `remove_sample=True`, so a broken environment never becomes a training signal. On abort (`abort_mode = "request"`) or any exception, the hook releases the episode. The server also releases episodes that have seen no tool call for `--idle-timeout` seconds (900 s by default), which covers clients that died before releasing.

Failure categories (`rollout.classify`; heuristic, from privileged diagnostics that never reach the policy or the reward):

| Category | Meaning |
|---|---|
| `format` | A turn ran past the per-turn token cap, or a tool call was malformed and no physical command ran. |
| `no_action` | The model never moved the robot (for example, it observed and then answered). |
| `perception` | Lift only: the gripper was never closed with the TCP at the cube (within 2 cm horizontally and 2.5 cm vertically of its centre), so the model did not locate it. |
| `control` | The robot acted (for lift, closed at the cube), but the task condition or a process check failed. |
| `infrastructure` | The robot server or SGLang failed; excluded from success rates. |

### Zero-shot base rates (2026-11-03, RTX A6000 shared with another job)

Held-out seeds 1,000,000 onward, one sample per seed, temperature 1, thinking off, at most 24 turns, 24,576 response tokens, evaluator v2. Episode logs (transcripts, checks, diagnostics) are in `codespace/phase2-results/`, outside the repository.

| Model, turn cap | Task | Success | format | no_action | perception | control | Mean turns | Tool errors / episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B, 1024 | lift (32) | 0 | 7 | 5 | 20 | – | 19.5 | 8.5 |
| Qwen3.5-0.8B, 1024 | reach_pose (32) | 0 | 7 | 6 | – | 19 | 15.3 | 7.5 |
| Qwen3.5-4B, 1024 | lift (32) | 0 | 5 | 0 | 27 | – | 16.0 | 1.1 |
| Qwen3.5-4B, 1024 | reach_pose (32) | 0 | 31 | 0 | – | 1 | 2.3 | 0 |
| Qwen3.5-4B, 4096 | reach_pose (16) | **2 (12.5%)** | 9 | 0 | – | 5 | 6.9 | 0 |

What the transcripts show:
- **0.8B breaks the protocol.** Of its tool errors, 204 reuse a `command_id` with different arguments (often the tool name itself as the id). Next come missing command ids, `batch_actions` lists that aren't valid JSON, and 83 moves over the 0.25 m limit.
- **4B follows the protocol but doesn't look.** In lift, it never closed the gripper within 2 cm horizontally of the cube. Its best close was a median 17 cm above the cube centre and 11 cm off horizontally. All 15 of its final answers claimed the cube was held, including ones where its own reading showed the gripper fully closed (0.0005 m; a held 5 cm cube leaves about 0.049 m). Evaluator v2 scores all of these 0 from physical state, and a transcript-reading judge would have been told the opposite.
- **Reach comes down to arithmetic, not vision.** The 4B model works out goal minus current pose by hand. With a 1024-token turn cap, almost every episode is cut off mid-calculation. With 4096, 9 of 16 episodes ended within 0.5 mm of the goal position, but 0.17–1.0 rad off in orientation. Converting a goal quaternion into a relative axis-angle command is the bottleneck, and the 2 successes are the episodes that got it right. This is the vision-free shortcut noted under evaluator v2. Reach measures numeric reasoning until the goal is given visually.
- **GRPO signal.** Lift has none at any setting (0/64). Reach at 4096 has some (2/16). That matches the Phase 3 plan: a supervised warm start on scripted-expert trajectories before GRPO.
- **Throughput.** With 16 concurrent 4B episodes, SGLang's KV pool (0.45 of one A6000) cannot keep every conversation between turns. Each turn re-prefills its full history (often over 15k tokens), and decode fell to about 33 tokens/s. At 8 concurrent, prefix hits rose from 1.2k (only the system prompt) to 4–5k tokens per request. Wall time: 0.8B, 64 episodes in 7.3 min; 4B, 64 episodes in 32 min; 4B at 4096, 16 episodes in 24 min.

### Running it

`rollout_eval.py` replaces only slime's Ray rollout manager. It calls the same hook with slime's `Dataset`, `GenerateState` and HTTP client:

```bash
python -m examples.robotics_newton.tasks --split heldout --count 32 --output heldout.jsonl
python -m examples.robotics_newton.server --world-count 16 --device cuda:1 --port 18100      # robotics-newton env
python -m sglang.launch_server --model-path Qwen3.5-4B --port 30000 --mem-fraction-static 0.45 \
    --attention-backend triton --sampling-backend pytorch                                    # rollout env
python -m examples.robotics_newton.rollout_eval --hf-checkpoint Qwen3.5-4B --prompt-data heldout.jsonl \
    --concurrency 16 --output results.jsonl
python -m pytest tests/test_robotics_newton/test_rollout.py   # rollout env; needs a Qwen3.5 checkpoint for the tokenizer
```

Rollout environment notes:
- **SGLang version and patch.** The env was built with pip: `sglang==0.5.21` (slime's pin), `requirements.txt`, `openai==2.6.1` and `ninja`. SGLang must carry slime's `docker/patch/v0.5.21/sglang.patch`, as the Docker image does. Without the patch, SGLang's Qwen-VL processor rejects `input_ids` whose image placeholders are already expanded. Every multi-turn VLM request has that form, so the second turn fails with `StopIteration`. The patch applies cleanly to the pip package with `patch -p2` inside `site-packages`.
- **Ampere GPUs.** FlashInfer compiles kernels at runtime and needs `ninja` on `PATH`. The Triton attention backend avoids that.
- **`run_robotics_newton.py`** launches `train.py --debug-rollout-only` with the same hook and data. It needs the slime Docker image (Megatron, Ray) and hasn't been run here. It uses the new `scripts/models/qwen3.5-{0.8B,4B}-vl.sh` model scripts.

## Measured (2026-11-03, RTX A6000)

Both GPUs were shared with another training job at 100% utilization, so these numbers are lower bounds.

| Check | Result |
|---|---|
| Scripted `reach_pose`, seeds 0–63, 64 worlds, 64 concurrent episodes | 64/64 under both v1 and v2, 1.1 commands per episode on average, 3.8 s wall (v1) |
| Scripted `lift`, seeds 0–63, 64 worlds, 64 concurrent episodes | 64/64 under both v1 and v2, 5 commands each, 6.6 s wall (v1) |
| Held-out seeds 100–115, both tasks | 16/16 each |
| Control frame (all worlds moving) | 16 worlds: 11.4 ms. 64 worlds: 17.2 ms (1.0× real time per world, 62× aggregate). 128 worlds: 23.8 ms. |
| Render, 2 cameras at 256² | about 2.1 ms per world; 137 ms for all 64 worlds |
| PNG encoding | about 4 ms per image, off the simulator thread |
| Simulator memory, 64 worlds | about 210 MB (Warp mempool) |
