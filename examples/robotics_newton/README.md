# Newton robot worlds for slime

GPU-batched Newton simulation of the Panda tasks in [`examples/robotics`](../robotics) (`lift`, `reach_pose`), served over HTTP with the same intuitive tool surface: `observe`, `move_to`, `gripper` and `batch_actions`. One process holds many worlds on one GPU, and each episode leases one world. There is no Docker or private `robot-rollout` dependency.

Status: Phase 1 is done: simulator, tool protocol, scripted experts, benchmark and tests. The slime rollout hook (a token-level VLM loop modelled on `geo3k_vlm_multi_turn`) is the next step.

| File | Role |
|---|---|
| `sim/scene.py` | FR3 + Franka Hand on a table with a red cube. Maps `(task, seed)` to the initial layout. The base frame is the simulation frame, with the table top at z = 0. |
| `sim/worlds.py` | `PandaWorlds` runs W worlds in one CUDA graph per 60 Hz control frame: per-world controller, batched IK, physics, settle and evaluation checks. It also runs batched ray-traced cameras (`front` fixed, `wrist` on the hand). |
| `protocol.py` | Episode semantics: current `observation_id`, idempotent `command_id`, limits, command budget, `batch_actions` rules, Harbor-style reward. |
| `tools.py` | Tool schemas (OpenAI function format). |
| `server.py` | aiohttp server. A simulator thread applies the requests that arrive between frames together, and steps frames only while a command is in flight. |
| `client.py` | Async HTTP client. |
| `scripted.py` | Scripted experts and the evaluation runner. Reach uses only public state; lift reads the cube pose from the privileged endpoint. |
| `bench.py` | Frame and render latency against world count. |

## Run

```bash
conda create -n robotics-newton python=3.12 -y
conda activate robotics-newton
pip install "newton[examples]==1.7.0" pillow aiohttp scipy pytest

# from the slime checkout
python -m examples.robotics_newton.scripted --task lift --seeds 0:64 --world-count 64 --device cuda:0
python -m examples.robotics_newton.server --world-count 64 --device cuda:0 --port 18100   # standalone server
python -m examples.robotics_newton.bench --world-counts 16 64 128 --device cuda:0
python -m pytest tests/test_robotics_newton
```

The first build for a new world count compiles Warp kernels, which takes 1–3 minutes. The Franka assets download once to `~/.cache/newton`.

## Semantics and deliberate differences

- **Episode flow.** `POST /episodes {"task", "seed"}` resets a free world to `sample_layout(task, seed)`. GRPO samples of one prompt share the seed, so they start from the same scene. `POST /episodes/{id}/finish` freezes the episode, samples the task condition 10 times at 20 Hz, and releases the world.
- **Reward.** The reward follows the Harbor verifiers: all 10 hold samples, the full 0.5 s window, and at least one physical command.
  - Lift success means the cube's bottom face is ≥ 3 cm above the table.
  - Reach success means TCP error ≤ 1 cm and ≤ 5°.
- **`move_to`** takes `translation_m` and `rotation_delta_rad`, in the base frame, anchored to the measured TCP. Its argument names match the robot-rollout calls in `tests/test_robotics/smoke.py`.
  - The planned TCP speed is 0.25 m/s and 1 rad/s.
  - A move is `completed` when the TCP is within 2 mm / 1° and the arm is still. It is `partial` if that doesn't happen within 1 s after the planned duration; the arm then holds where it is.
- **Limits.** 0.25 m and 1.0 rad per command, TCP z ≥ 5 mm, 80 physical commands. These are local choices; robot-rollout's real values are not visible from this repository.
- **Robot.** The robot is the FR3 from `newton-assets` with gains from Newton's `ik_cube_stacking` example, not robosuite's Panda. Images are 256×256 ray-traced RGB without depth.
- **Idle worlds** keep simulating under their hold target while other worlds move, as a real arm does while its policy thinks. Simulated time is therefore not bit-reproducible across concurrency levels.
- **Privileged endpoint.** `GET /episodes/{id}/privileged` returns the cube pose only when the server is started with `--allow-privileged`. It is for scripted experts only.

## Measured (2026-11-03, RTX A6000)

Both GPUs were shared with another training job at 100% utilization, so these numbers are lower bounds.

| Check | Result |
|---|---|
| Scripted `reach_pose`, seeds 0–63, 64 worlds, 64 concurrent episodes | 64/64, 1.1 commands per episode on average, 3.8 s wall |
| Scripted `lift`, seeds 0–63, 64 worlds, 64 concurrent episodes | 64/64, 5 commands each, 6.6 s wall |
| Held-out seeds 100–115, both tasks | 16/16 each |
| Control frame (all worlds moving) | 16 worlds: 11.4 ms. 64 worlds: 17.2 ms (1.0× real time per world, 62× aggregate). 128 worlds: 23.8 ms. |
| Render, 2 cameras at 256² | about 2.1 ms per world; 137 ms for all 64 worlds |
| PNG encoding | about 4 ms per image, off the simulator thread |
| Simulator memory, 64 worlds | about 210 MB (Warp mempool) |
