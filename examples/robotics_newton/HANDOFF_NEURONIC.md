# Handoff: Newton robot RL experiments on Neuronic

Written 2026-11-03 for a new Claude Code session on Princeton's Neuronic cluster. It continues work that was done on a 2× RTX A6000 workstation. Read this file, then [README.md](README.md) in this directory, before running anything.

## 0. Rules from the user

- **Never modify `main`.** No commits, pushes, merges, rebases or resets on `main`, locally or on GitHub. All work goes on `robotics-newton`. `main` is at `3744f4f6`.
- **Commit and push only when the user asks**, and ask again each time. Don't change git config. Commit with
  `git -c user.name="Yixun Hu" -c user.email="yh4742@princeton.edu" commit ...`
  and end the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push with an explicit refspec:
  `git push git@github.com:QFOI/slime.git robotics-newton:robotics-newton`.
- **Never start a policy-facing robot server with `--allow-privileged`.** That flag exposes the cube pose and a cube teleport. The only legitimate use is a server that only scripted experts talk to (Phase 3 data generation, red-team tests).
- **Keep results outside the repository**, in a results directory on scratch. Commit only code and the README result tables.
- **Implementation subagents get `model: "opus"`.** This is the user's standing preference.
- **Report faithfully.** Give each success rate with its denominator, and give the failure categories and failed evaluator checks with it. Check claims against simulator state, never against what the model says: the 4B model claimed success in all 15 of its failed lift answers.

## 1. Where things stand

The project integrates the Newton GPU simulator into slime as an RL environment for a VLM tool-calling robot policy. Each action is one model turn calling `observe`, `move_to`, `gripper` or `batch_actions`, and a Newton server holds many robot worlds on one GPU. The tasks are `lift` (pick up a 5 cm red cube) and `reach_pose` (move the TCP to a numeric goal pose). Both run on an FR3 + Franka Hand on a table.

Branch `robotics-newton`, three commits on top of `main`:

| Commit | Content |
|---|---|
| `dafa3aaf` | Phase 1: batched Newton worlds, tool protocol, HTTP server, scripted experts, benchmark, tests |
| `d2ac543d` | Evaluator v2 (task condition plus process checks on true sim state) and the red-team suite |
| `f9814e32` | Phase 2: slime custom-generate hook (`rollout.py`), prompt rows (`tasks.py`), `rollout_eval.py`, `train.py --debug-rollout-only` launcher, Qwen3.5 VL model scripts |

Phase status (full plan in section 5):

| Phase | Status |
|---|---|
| 1. Sim, tools, scripted experts | Done: experts 64/64 on both tasks; 17 ms per control frame at 64 worlds |
| Evaluator v2 + red team | Done: every hack scores 0, both experts score 1 |
| 2. Zero-shot rollouts through the hook | Done with `rollout_eval.py` (no Ray or Megatron). `train.py --debug-rollout-only` **not yet run** |
| 3. SFT warm start | Not started. Needs new code (section 4, E4) |
| 4. GRPO | Not started |
| 5. Real-to-sim | Blocked on real data; not for this session unless the user says so |

Zero-shot base rates (held-out seeds from 1,000,000, one sample per seed, temperature 1, thinking off, ≤ 24 turns, 24,576 response tokens, evaluator v2):

| Model, per-turn token cap | Task | Success | Main failure |
|---|---|---:|---|
| Qwen3.5-0.8B, 1024 | lift (32) | 0 | breaks the protocol: reuses command ids, sends malformed batches |
| Qwen3.5-0.8B, 1024 | reach_pose (32) | 0 | same |
| Qwen3.5-4B, 1024 | lift (32) | 0 | perception: closes a median 17 cm above the cube |
| Qwen3.5-4B, 1024 | reach_pose (32) | 0 | 31/32 cut off by the turn cap mid-arithmetic |
| Qwen3.5-4B, 4096 | reach_pose (16) | **2 (12.5%)** | 9/16 within 0.5 mm in position but 0.17–1.0 rad off in orientation |

The two reach successes (seeds 1000007 and 1000002) were replayed and checked: their tool calls, fed back through the simulator, score reward 1 again. Lift has no GRPO signal at all, which is why Phase 3 exists. Reach has a vision-free shortcut, because its goal is given as numbers.

## 2. Code map

Everything is in `examples/robotics_newton/` (README has the full table) and `tests/test_robotics_newton/`.

- `sim/scene.py`, `sim/worlds.py`: scene, seed → layout, `PandaWorlds` (one CUDA graph per 60 Hz control frame), cameras (256² ray-traced RGB: `front` fixed, `wrist` on the hand).
- `protocol.py`, `server.py`, `client.py`, `tools.py`: episode semantics and the HTTP server. `POST /episodes` leases a world and waits for a free one if all are leased. `POST /episodes/{id}/finish` scores the episode and releases the world. Idle episodes are reaped after `--idle-timeout` (900 s).
- `evaluator.py`: the reward. `redteam.py`: hacks that must score 0.
- `scripted.py`: experts. Reach uses public state only; lift reads the cube pose from the privileged endpoint.
- `tasks.py`: system prompt, instructions, prompt rows. The splits are disjoint: `train` = seeds 0–99,999, `heldout` = seeds 1,000,000–1,099,999. Never train on held-out seeds.
- `rollout.py`: the hook (`--custom-generate-function-path examples.robotics_newton.rollout.generate`). It is token-in/token-out against SGLang `/generate`, parses tool calls with `qwen3_coder`, and appends tool turns with loss mask 0. It sets `sample.reward` from the server's `finish`, so slime's RM is skipped, and puts all checks and diagnostics in `sample.metadata["robot"]`. Settings: `robot_server_url`, `robot_max_turns`, `robot_max_turn_tokens` (via `--custom-config-path`, see `rollout_config.yaml`).
- `rollout_eval.py`: runs the hook on prompt rows with slime's `Dataset` and `GenerateState`, but without Ray. It writes one jsonl row per episode as episodes finish. It opens `--output` with `"w"`, so give each job a new file.
- `run_robotics_newton.py`: `train.py --debug-rollout-only` launcher. It assumes the slime Docker image layout (`/root/models`, `/root/datasets`) and runs `hf download` inside `prepare()`.

## 3. Cluster bring-up

### 3.1 Check the cluster first

The workstation had no Docker or Apptainer and was shared with another job. Neuronic's specs are behind the CS portal login, and public pages only say it is 33 Slurm GPU nodes aimed at FP32 workloads. Find out the following and write it down:

```bash
sinfo -o "%P %G %l %D %N" ; scontrol show partition
nvidia-smi --query-gpu=name,memory.total --format=csv     # inside a GPU job
apptainer --version ; which docker
nproc ; free -g ; df -h ~ /scratch* 2>/dev/null           # home quota vs scratch
curl -sI https://huggingface.co | head -1                 # inside a GPU job: does a compute node reach the internet?
```

These answers decide a lot:
- **GPU model and memory.** A 48 GB card (A40/L40 class) is enough for rollouts. Full fine-tuning Qwen3.5-4B needs about 84 GB of training state plus activations, which means the distributed optimizer across ≥ 4 GPUs or `--optimizer-cpu-offload`. FP8 needs Hopper or Ada; on Ampere use bf16 only.
- **Time limit.** It sets how long a job can run. `rollout_eval.py` can't resume, so split long evals into prompt files with `tasks.py --offset/--count`.
- **Internet on compute nodes.** Without it, download models, the Newton assets, pip wheels and the container image on the login node first.

### 3.2 Storage

Put the conda envs, caches, models and results on scratch, not in a small home quota:

```bash
export NEWTON_CACHE_PATH=$SCRATCH/cache/newton     # Franka assets (93 MB); downloaded on first scene build
export WARP_CACHE_PATH=$SCRATCH/cache/warp         # compiled kernels (~450 MB); first build per world count takes 1–3 min
export HF_HOME=$SCRATCH/cache/hf
```

`$SCRATCH` is a placeholder; use whatever scratch path Neuronic provides. Set these in every job script, and keep them identical between the login node and compute nodes so the caches are reused.

### 3.3 Code and data

```bash
git clone git@github.com:QFOI/slime.git && cd slime && git checkout robotics-newton   # verify: git log -1 shows f9814e32 or later
hf download Qwen/Qwen3.5-4B --local-dir $SCRATCH/models/Qwen3.5-4B
hf download Qwen/Qwen3.5-0.8B --local-dir $SCRATCH/models/Qwen3.5-0.8B
```

The Phase 2 episode logs and the replay/video scripts live on the workstation, outside the repository. The user can copy them over (fill in the login host):

```bash
scp -r yixunhu@10.249.7.49:/home/yixunhu/codespace/phase2-results yh4742@<neuronic-login>:<scratch>/robotics_newton/
scp yixunhu@10.249.7.49:/home/yixunhu/codespace/newton-viz/{render_model_reach.py,render_scripted_lift.py} yh4742@<neuronic-login>:<scratch>/newton-viz/
```

- **`phase2-results/`** (3.4 MB): `eval_0.8B.jsonl`, `eval_4B.jsonl`, `eval_4B_reach4096.jsonl` (one row per episode: reward, checks, diagnostics, failure, transcript, …) and the prompt files `heldout32.jsonl` and `heldout16_reach.jsonl`.
- **`render_model_reach.py`** replays a transcript's tool calls on the same seed and writes an mp4. It expects `newton-viz/` to sit next to the `slime/` checkout, and it imports `LocalBackend` from `render_scripted_lift.py`. Note that its transcript parser treats the text before the first `<|im_start|>` as turn 1, because the transcript is response-only.

### 3.4 Environments

There are two envs, because Newton and SGLang pin different numpy and torch versions. The robot server runs in the first, SGLang and the hook in the second. They only talk over HTTP on localhost, so start both inside the same Slurm job.

**`robotics-newton`** (simulator and robot server). The workstation versions were Python 3.12, newton 1.7.0, warp-lang 1.19.0, mujoco and mujoco-warp 3.15.0, numpy 2.5.4.

```bash
conda create -n robotics-newton python=3.12 -y && conda activate robotics-newton
pip install "newton[examples]==1.7.0" pillow aiohttp scipy pytest pytest-asyncio
```

**`slime-rollout`** (SGLang, the hook, `rollout_eval.py`). The workstation versions were Python 3.12, sglang 0.5.21, torch 2.13.0, transformers 5.12.1, flashinfer-python 0.6.18, sglang-router 0.3.2, ray 2.59.0, qwen-vl-utils 0.0.14, openai 2.6.1, ninja 1.13.2.

```bash
conda create -n slime-rollout python=3.12 -y && conda activate slime-rollout
pip install sglang==0.5.21 && pip install -r requirements.txt && pip install openai==2.6.1 ninja pytest pytest-asyncio
pip install -e . --no-deps
# Required: slime's SGLang patch. Unpatched SGLang rejects multi-turn VLM requests (StopIteration on turn 2).
cd "$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')" \
  && patch -p2 < /path/to/slime/docker/patch/v0.5.21/sglang.patch
```

Megatron is not in either env. Training (E3–E5) needs the slime image, through Apptainer if Neuronic has it:

```bash
apptainer pull slime.sif docker://slimerl/slime:<tag>
```

`docker/version.txt` says `nightly-dev-20261027a`, so the tag is probably `nightly-dev-20261027a-cu130` or `latest`; check that it exists. Without containers, there is `build_conda.sh`, which builds SGLang, Megatron and the rest from source. It has not been tried; it assumes `/root` and sets `TORCH_CUDA_ARCH_LIST` to Hopper and Blackwell, so change that for the cluster's GPUs.

### 3.5 Smoke checks (expected results)

Run on one GPU node, from the slime checkout:

```bash
# robotics-newton env
python -m pytest tests/test_robotics_newton                                  # 28 pass, test_rollout.py skipped (needs sglang)
python -m examples.robotics_newton.scripted --task reach_pose --seeds 0:64 --world-count 64 --device cuda:0   # 64/64
python -m examples.robotics_newton.scripted --task lift --seeds 0:64 --world-count 64 --device cuda:0         # 64/64
python -m examples.robotics_newton.redteam --seeds 0:16 --device cuda:0     # exits 0; every hack scores 0 under v2
# slime-rollout env
ROBOTICS_NEWTON_HF_CHECKPOINT=$SCRATCH/models/Qwen3.5-0.8B python -m pytest tests/test_robotics_newton/test_rollout.py   # 14 pass
```

If the experts drop below 64/64, stop and investigate before running any model. That would mean the simulator or its versions differ from the workstation's.

## 4. Experiments, in order

Each experiment ends with a short written report to the user: the numbers, where the results are, and what was not verified. Ask before moving to the next one if anything is surprising.

### E1. Idle-GPU benchmark (leftover from Phase 0)

The workstation numbers are lower bounds, because both GPUs were at 100% from another job.

```bash
python -m examples.robotics_newton.bench --world-counts 16 64 128 256 --device cuda:0
```

Record ms per control frame and render ms per world, and add them to the README's "Measured" table with the GPU model. Workstation reference: 11.4 / 17.2 / 23.8 ms per frame at 16 / 64 / 128 worlds, and about 2.1 ms per world to render 2 cameras. The result also tells you how many worlds one server GPU can hold for E3–E5.

### E2. Reproduce and widen the zero-shot baseline

Goal: tight base rates, and above all the **per-group signal** GRPO needs. That is the fraction of seeds where n = 8 samples give some successes and some failures.

```bash
# slime-rollout env unless noted. Paths are examples.
python -m examples.robotics_newton.tasks --tasks reach_pose --split heldout --count 64 --output $RUN/heldout64_reach.jsonl
python -m examples.robotics_newton.tasks --tasks lift --split heldout --count 64 --output $RUN/heldout64_lift.jsonl
python -m examples.robotics_newton.server --world-count 128 --device cuda:0 --port 18100          # robotics-newton env, no --allow-privileged
python -m sglang.launch_server --model-path $SCRATCH/models/Qwen3.5-4B --port 30000 \
    --mem-fraction-static 0.85 --attention-backend triton --sampling-backend pytorch           # on its own GPU(s)
python -m examples.robotics_newton.rollout_eval --hf-checkpoint $SCRATCH/models/Qwen3.5-4B \
    --prompt-data $RUN/heldout64_reach.jsonl --n-samples-per-prompt 8 --max-turn-tokens 4096 \
    --concurrency 32 --output $RUN/zeroshot_4B_reach_n8.jsonl
```

- **First, a quick reproduction.** Run the copied `heldout16_reach.jsonl` at `--max-turn-tokens 4096 --n-samples-per-prompt 1`. Expect about 2/16. With n = 16 the interval is wide, so 0–5/16 is consistent with the old run.
- **Then reach**, 64 seeds × 8 samples at a 4096-token turn cap. Report success, pass@8, and the histogram of successes per seed (0–8). Lift: 64 seeds × 8. Expect about 0; report the perception diagnostics (`diagnostics.closed_at_cube` and `diagnostics.closest_close_offset_m`, the TCP offset from the cube centre at its closest gripper close).
- **Optional: Qwen3.5-9B zero-shot** on the same files, if the GPUs fit it. It gives a model-size data point. Ask the user first.
- **Throughput.** On the workstation, the KV pool (0.45 of a shared A6000) couldn't hold 16 conversations between turns. Every turn re-prefilled 15k+ tokens and decode fell to about 33 tokens/s; 16 episodes at a 4096 cap took 24 min. On a dedicated GPU, raise `--mem-fraction-static`. Scale out with SGLang data parallelism (`--dp-size N`, or sglang-router in front of N servers). Then raise `--concurrency` only while the prefix-cache hit stays high. The robot server should have at least as many worlds as concurrent episodes; extra episodes wait for a free world.
- **Count infrastructure failures separately.** Exclude rows with `failure == "infrastructure"` from success rates and report how many there were.

### E3. `train.py --debug-rollout-only` through the real slime stack

Goal: show that the hook runs under Ray's RolloutManager with SGLang engines that slime launches. Its success rates should match `rollout_eval.py`, and the saved Samples should look right.

- Run inside the slime image (Apptainer, `--nv`, with the repo and scratch bind-mounted). Start the robot server from the `robotics-newton` env outside the container, in the same job.
- `run_robotics_newton.py` hard-codes `/root/models`, `/root/datasets` and an `hf download`. Either bind-mount scratch over those paths, or make the paths environment variables (a small change; commit it only with approval).
- Run it with `MODEL_NAME=Qwen3.5-4B SLIME_SCRIPT_NUM_GPUS=<n> python examples/robotics_newton/run_robotics_newton.py`. It uses batch 32 × n 4, so up to 128 episodes run at once; start the server with `--world-count 128`.
- Check the saved `/…/{rollout_id}.pt`:
  - `tokens` and `loss_mask` lengths agree;
  - `loss_mask` is 1 only on assistant tokens;
  - `multimodal_train_inputs` is present and the number of images matches the observe calls;
  - `reward` matches `metadata["robot"]["reward"]`.
- This is also the first real test of slime's dense Qwen3.5-VL path (`scripts/models/qwen3.5-4B-vl.sh` → `slime_plugins.models.qwen3_5_vl`). Nothing upstream exercises it.

### E4. Phase 3: SFT warm start on scripted-expert trajectories

Goal: a 4B checkpoint whose held-out lift success is neither 0 nor 1 for most seeds at n = 8. This needs new code, in two parts.

**1. The trajectory generator** (new file, e.g. `examples/robotics_newton/sft_data.py`).
- Run the scripted experts on **train** seeds. Lift needs a separate server with `--allow-privileged` that only the expert uses.
- Turn each successful episode into exactly the token sequence the policy would see in RL:
  - prompt rows from `tasks.py`;
  - assistant turns that are the expert's calls in Qwen3.5's tool-call XML (observe first, real `observation_id`s and fresh `command_id`s);
  - tool turns rendered by `rollout.py`'s own encoder (`_tool_message`, `_encode_tool_turn`), with images;
  - a short final answer with no tool call.
- **Don't re-render the conversation separately; drive the hook itself.** Put a teacher in place of SGLang: a fake `/generate` that returns the expert's next turn as tokens. `tests/test_robotics_newton/test_rollout.py` already has a fake SGLang server to copy. That way the tokens, loss masks and `multimodal_train_inputs` are identical to RL by construction. Keep only reward-1 episodes.
- **Trajectory design matters more than the amount.** The lift expert goes straight to a pose computed from the privileged cube position, so a cloned policy has to infer that position from the images.
  - Make the expert observe after the approach, so the wrist view shows the cube, and make small corrective moves, so the data shows closed-loop behaviour.
  - Randomize step sizes.
  - Keep assistant text minimal. Don't write reasoning that cites privileged numbers the policy can't see.

**2. The VLM SFT rollout function.** `slime.rollout.sft_rollout.generate_rollout` is text-only: it builds tokens from messages with `MultiTurnLossMaskGenerator` and drops images. So the SFT run needs a small `--rollout-function-path` that loads the pre-built Samples, with `--loss-type sft_loss`. Use `scripts/run-qwen3-4B-base-sft.sh` for the SFT flags and `examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py` for the VL Megatron flags. Convert the HF checkpoint with `tools/convert_hf_to_torch_dist.py`; this is untested for dense Qwen3.5-VL.

**Exit check:**
- Run E2's held-out evaluation on the SFT checkpoint (convert back with `tools/convert_torch_dist_to_hf.py`, then serve it with SGLang).
- Report success, the successes-per-seed histogram, the failure categories and the failed checks.
- Watch for a policy that learned the spawn-region centre (blind lift scores 3.1%) rather than to look. Compare against that rate, and check `closest_close_offset_m` across seeds.

### E5. Phase 4: GRPO

Reach comes first, because it already has zero-shot signal at a 4096-token turn cap. Lift starts from the E4 checkpoint.

- Use the same hook for training and for evaluation (`evaluation=True`) on held-out rows.
- Use n = 8 per seed with train-split seeds, `--advantage-estimator grpo`.
- Start from the geo3k multi-turn recipe: `--eps-clip 0.2 --eps-clip-high 0.28 --lr 1e-6`, KL 0, `--colocate`, and `--optimizer-cpu-offload` if memory needs it.
- Lower `--sglang-mem-fraction-static` if the robot server shares a GPU with SGLang; 64 worlds use about 210 MB plus Warp's compile workspace.
- **Watch for reward hacking.** Track every evaluator check over training, not just the reward. Rerun `redteam.py` if anything in the sim or evaluator changes. Read a few transcripts per eval, and render one or two episodes with `render_model_reach.py` (or adapt it for lift).
- **Exit:** held-out success rises above the E2 or E4 baseline with non-overlapping intervals.

### Not for this session unless the user asks

- **Task-design fixes.** Visual reach goals, distractor objects and instruction variety would close the shortcuts. They change the tasks, so every baseline has to be rerun.
- **The shove false positive in `not_launched`.** The stiff arm kicks a shoved cube to 1.2–3.5 m/s.
- **A VLM auditor of transcripts.**
- **Real-to-sim** (README plan section 5). The user still has to answer what real data exists.
- **Della.** This handoff replaces the Della plan.

## 5. Gotchas seen so far

- **SGLang without slime's patch** fails on the second turn of every VLM episode (`StopIteration` in the Qwen-VL processor).
- **Turn-token cap.** At 1024, 4B reach fails almost entirely by truncation. Use 4096 for 4B.
- **Robot server vs SGLang start-up.** The robot server compiles Warp kernels for 1–3 min the first time it sees a world count. Wait for `GET /stats` before starting episodes.
- **Simulated time** is not bit-reproducible across concurrency levels. Idle worlds keep simulating under hold while their LLM thinks, by design. Replays match the live runs to within about 0.3 mm, not exactly.
- **The model's self-report is unreliable.** The 4B model cites its command receipt (error relative to its own command) as if it were the error to the goal. Only `finish` decides.
- **FlashInfer** compiles kernels at runtime and needs `ninja` on `PATH`. The Triton attention backend avoids this.
- **`ruff format --check`** reports differences, but the repo's formatters are black and isort. Use those.

## 6. Bringing results back

- **Results** go under `<scratch>/robotics_newton/<experiment>/`: the jsonl episode logs, the prompt files, the server and SGLang logs, and the exact commands, in a `RUN.md`.
- **README.** Update the README result tables on `robotics-newton` and keep this file's status table current. Commit only after the user approves.
- **Every success rate** comes with its n, its failure categories, its failed checks and the GPU it ran on.
