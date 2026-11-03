"""Scripted experts that drive the robot server through its public tools.

reach_pose uses only public information (the goal pose in the task block).
lift reads the cube pose from the privileged endpoint, then issues the same
observe/move_to/gripper calls a policy would. Both validate the simulator and
protocol end to end and can later produce demonstration trajectories.

python -m examples.robotics_newton.scripted --task lift --seeds 0:64 --world-count 64 --device cuda:0
python -m examples.robotics_newton.scripted --task reach_pose --seeds 0:64 --server http://127.0.0.1:18100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
from aiohttp import web
from scipy.spatial.transform import Rotation

from .client import RobotClient
from .evaluator import EVALUATORS, V2, EvaluatorConfig
from .sim.scene import DOWN


def _rotation(pose: dict) -> Rotation:
    w, x, y, z = pose["quaternion_wxyz"]
    return Rotation.from_quat([x, y, z, w])


class Driver:
    """Tracks the newest observation id and measured TCP across tool calls."""

    def __init__(self, client: RobotClient, episode_id: str):
        self.client = client
        self.episode_id = episode_id
        self.observation_id = None
        self.tcp = None
        self.commands = 0
        self.receipts = []
        self.last_images = []
        self.task = None

    async def call(self, name: str, arguments: dict | None = None) -> dict:
        reply = await self.client.call(self.episode_id, name, arguments)
        if reply["is_error"]:
            raise RuntimeError(f"{name} failed: {reply['data'].get('error')}")
        data = reply["data"]
        self.observation_id = data.get("observation_id", self.observation_id)
        if "state" in data:
            self.tcp = data["state"]["tcp_pose"]
        if "task" in data:
            self.task = data["task"]
        if "receipt" in data:
            self.receipts.append(data["receipt"])
        if reply["images"]:
            self.last_images = reply["images"]
        return data

    async def observe(self) -> dict:
        return await self.call("observe")

    async def move_to_pose(self, position, rotation: Rotation, *, max_translation=0.24) -> None:
        """Reach an absolute pose with relative commands; split long translations."""
        for _ in range(4):
            delta = np.asarray(position) - self.tcp["position_m"]
            rotvec = (rotation * _rotation(self.tcp).inv()).as_rotvec()
            if np.linalg.norm(delta) < 0.003 and np.linalg.norm(rotvec) < 0.015:
                return
            delta *= min(1.0, max_translation / max(np.linalg.norm(delta), 1e-12))
            rotvec *= min(1.0, 0.95 / max(np.linalg.norm(rotvec), 1e-12))
            self.commands += 1
            await self.call(
                "move_to",
                {
                    "observation_id": self.observation_id,
                    "command_id": f"scripted-{self.commands}",
                    "frame": "base",
                    "translation_m": delta.tolist(),
                    "rotation_delta_rad": rotvec.tolist(),
                },
            )

    async def gripper(self, action: str) -> None:
        self.commands += 1
        await self.call(
            "gripper",
            {"observation_id": self.observation_id, "command_id": f"scripted-{self.commands}", "action": action},
        )


async def reach_expert(driver: Driver) -> None:
    await driver.observe()
    goal = driver.task["goal_tcp_pose"]
    await driver.move_to_pose(goal["position_m"], _rotation(goal))
    await driver.observe()


async def lift_expert(driver: Driver) -> None:
    await driver.observe()
    cube = await driver.client.privileged(driver.episode_id)
    position = np.asarray(cube["cube_position"])
    cube_yaw = Rotation.from_quat(cube["cube_quat_xyzw"]).as_euler("xyz")[2]
    # The cube is 90-degree symmetric: pick the grasp yaw nearest the current TCP.
    current = _rotation(driver.tcp)
    grasp = min(
        (Rotation.from_euler("z", cube_yaw + k * np.pi / 2) * DOWN for k in range(-2, 3)),
        key=lambda r: (r * current.inv()).magnitude(),
    )
    await driver.move_to_pose(position + [0.0, 0.0, 0.10], grasp)
    await driver.gripper("open")
    await driver.move_to_pose(position, grasp)
    await driver.gripper("close")
    await driver.move_to_pose(position + [0.0, 0.0, 0.15], grasp)
    await driver.observe()


EXPERTS = {"reach_pose": reach_expert, "lift": lift_expert}


async def run_episode(client: RobotClient, task: str, policy, seed: int, image_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    episode = await client.start(task, seed)
    driver = Driver(client, episode["episode_id"])
    error = None
    try:
        await policy(driver)
    except Exception as exc:  # recorded; the episode is still scored
        error = f"{type(exc).__name__}: {exc}"
    result = await client.finish(episode["episode_id"])
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        for image in driver.last_images:
            (image_dir / f"{task}-{seed:04d}-{image['camera']}.png").write_bytes(base64.b64decode(image["data"]))
    state = result["evaluator_state"]
    return {
        "task": task,
        "seed": seed,
        "reward": result["reward"],
        "task_condition": state["checks"]["task_condition"],
        "failed_checks": [name for name, ok in state["checks"].items() if not ok],
        "metrics": state["metrics"],
        "physical_commands": state["physical_commands"],
        "receipt_status": [r["status"] for r in driver.receipts],
        "error": error,
        "wall_sec": round(time.perf_counter() - started, 3),
    }


@contextlib.asynccontextmanager
async def local_server(world_count: int, device=None, evaluator: EvaluatorConfig = V2):
    """In-process simulator + HTTP server with privileged endpoints; yields its base URL."""
    from .server import RobotServer, SimWorker

    worker = SimWorker(world_count, device=device)
    await asyncio.to_thread(worker.start)
    runner = web.AppRunner(RobotServer(worker, evaluator=evaluator, allow_privileged=True).app())
    await runner.setup()
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        await web.TCPSite(runner, "127.0.0.1", port).start()
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()
        worker.stop()


async def run_seeds(client: RobotClient, task: str, policy, seeds, *, concurrency=64, image_dir=None, save_images=0):
    semaphore = asyncio.Semaphore(concurrency)
    first = min(seeds)

    async def one(seed):
        async with semaphore:
            images = image_dir if image_dir is not None and seed < first + save_images else None
            return await run_episode(client, task, policy, seed, images)

    return await asyncio.gather(*(one(seed) for seed in seeds))


def summarize(task: str, episodes: list[dict]) -> dict:
    return {
        "task": task,
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["reward"] for e in episodes])),
        "task_condition_rate": float(np.mean([e["task_condition"] for e in episodes])),
        "errors": sum(e["error"] is not None for e in episodes),
        "mean_physical_commands": float(np.mean([e["physical_commands"] for e in episodes])),
        "failed_seeds": [e["seed"] for e in episodes if e["reward"] != 1.0],
        "failed_checks": dict(Counter(name for e in episodes for name in e["failed_checks"])),
    }


async def evaluate(args) -> dict:
    lo, hi = (int(v) for v in args.seeds.split(":"))
    async with contextlib.AsyncExitStack() as stack:
        base_url = args.server or await stack.enter_async_context(
            local_server(args.world_count, args.device, EVALUATORS[args.evaluator])
        )
        client = RobotClient(base_url)
        stack.push_async_callback(client.close)
        started = time.perf_counter()
        image_dir = args.output / "images" if args.output else None
        episodes = await run_seeds(
            client,
            args.task,
            EXPERTS[args.task],
            range(lo, hi),
            concurrency=args.concurrency,
            image_dir=image_dir,
            save_images=args.save_images,
        )
        wall = time.perf_counter() - started
        stats = await client.stats()
    summary = summarize(args.task, episodes)
    summary["wall_sec"] = round(wall, 2)
    summary["server"] = {k: stats[k] for k in ("world_count", "frames", "mean_step_ms", "renders", "mean_render_ms")}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.task}-episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        (args.output / f"{args.task}-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(EXPERTS), required=True)
    parser.add_argument("--seeds", default="0:64", help="half-open range lo:hi")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--server", default=None, help="existing server URL; default starts one in-process")
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--evaluator", choices=sorted(EVALUATORS), default="v2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-images", type=int, default=4, help="save final images of the first N seeds")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args)), indent=2))


if __name__ == "__main__":
    main()
