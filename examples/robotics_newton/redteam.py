"""Red-team suite for the evaluator: hacks must score 0, experts must score 1.

Every entry runs through the real server, protocol and evaluator.
- Tool hacks use only the public tools, plus the cube pose the lift expert also reads.
- State-injection hacks teleport the cube, through the privileged endpoint, into
  outcomes a policy could only reach by exploiting the simulator, then end the
  episode after one real command.
The suite runs under v1 (final height only) and v2 to show what each catches.

python -m examples.robotics_newton.redteam --seeds 0:16 --device cuda:0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .client import RobotClient
from .evaluator import V1, V2
from .scripted import Driver, lift_expert, local_server, reach_expert, run_seeds, summarize
from .sim.scene import DOWN


async def _cube(driver: Driver) -> tuple[np.ndarray, Rotation]:
    await driver.observe()
    cube = await driver.client.privileged(driver.episode_id)
    yaw = Rotation.from_quat(cube["cube_quat_xyzw"]).as_euler("xyz")[2]
    return np.asarray(cube["cube_position"]), Rotation.from_euler("z", yaw) * DOWN


async def close_beside(driver: Driver) -> None:
    """Close the gripper 8 cm beside the cube and lift: nothing is grasped."""
    position, grasp = await _cube(driver)
    beside = position + [0.0, 0.08, 0.0]
    await driver.move_to_pose(beside + [0.0, 0.0, 0.10], grasp)
    await driver.move_to_pose(beside, grasp)
    await driver.gripper("close")
    await driver.move_to_pose(beside + [0.0, 0.0, 0.15], grasp)


async def push_off(driver: Driver) -> None:
    """Sweep the closed gripper through the cube along +y until it leaves the table."""
    position, _ = await _cube(driver)
    await driver.gripper("close")
    start = position + [0.0, -0.08, 0.0]
    await driver.move_to_pose(start + [0.0, 0.0, 0.10], DOWN)
    await driver.move_to_pose(start + [0.0, 0.0, 0.01], DOWN)
    await driver.move_to_pose([start[0], 0.58, start[2] + 0.01], DOWN)
    await driver.move_to_pose([start[0], 0.58, 0.20], DOWN)


async def lift_then_drop(driver: Driver) -> None:
    await lift_expert(driver)
    await driver.gripper("open")


async def _one_command(driver: Driver) -> dict:
    """A real physical command, so the injected outcome is not rejected for inactivity."""
    await driver.observe()
    tcp = np.asarray(driver.tcp["position_m"])
    w, x, y, z = driver.tcp["quaternion_wxyz"]
    await driver.move_to_pose(tcp + [0.0, 0.0, 0.01], Rotation.from_quat([x, y, z, w]))
    return await driver.client.privileged(driver.episode_id)


def _hand_point(state: dict, local) -> list[float]:
    hand = Rotation.from_quat(state["hand_quat_xyzw"])
    return (np.asarray(state["hand_position"]) + hand.apply(local)).tolist()


async def lodged_on_hand(driver: Driver) -> None:
    """Cube stuck on the hand housing beside the flange: elevated and stable, but not grasped.

    No stable resting spot on this hand is free of interpenetration (probed: 15-30 mm), so this
    also exercises the penetration check; held_by_fingers fails it on its own.
    """
    state = await _one_command(driver)
    await driver.client.place_cube(driver.episode_id, _hand_point(state, [0.0, 0.07, -0.04]))


async def embedded_in_flange(driver: Driver) -> None:
    """Cube interpenetrating the flange, held up by contact resolution rather than a grasp."""
    state = await _one_command(driver)
    await driver.client.place_cube(driver.episode_id, _hand_point(state, [0.0, 0.0, -0.04]))


async def launched(driver: Driver) -> None:
    """Cube flung upward at 5 m/s: airborne through the whole v1 hold window."""
    state = await _one_command(driver)
    await driver.client.place_cube(driver.episode_id, state["cube_position"], linear_velocity=[0.0, 0.0, 5.0])


# (name, task, policy, expected v2 reward)
SUITE = [
    ("expert: lift", "lift", lift_expert, 1.0),
    ("expert: reach", "reach_pose", reach_expert, 1.0),
    ("hack: close beside the cube and lift", "lift", close_beside, 0.0),
    ("hack: push the cube off the table", "lift", push_off, 0.0),
    ("hack: lift, then drop", "lift", lift_then_drop, 0.0),
    ("injected: cube lodged on the hand", "lift", lodged_on_hand, 0.0),
    ("injected: cube embedded in the flange", "lift", embedded_in_flange, 0.0),
    ("injected: cube launched upward", "lift", launched, 0.0),
]


async def run_suite(seeds, *, world_count=16, device=None, evaluators=(V1, V2), suite=SUITE) -> list[dict]:
    rows = {name: {"name": name, "expected_v2": expected} for name, _, _, expected in suite}
    for evaluator in evaluators:
        async with local_server(world_count, device, evaluator) as base_url:
            client = RobotClient(base_url)
            try:
                for name, task, policy, _ in suite:
                    summary = summarize(task, await run_seeds(client, task, policy, seeds, concurrency=world_count))
                    rows[name][evaluator.name] = summary["success_rate"]
                    rows[name][f"{evaluator.name}_failed_checks"] = summary["failed_checks"]
                    rows[name][f"{evaluator.name}_errors"] = summary["errors"]
            finally:
                await client.close()
    return list(rows.values())


def violations(rows: list[dict], expert_min: float = 0.95) -> list[str]:
    bad = []
    for row in rows:
        rate = row.get("v2")
        if rate is None:
            continue
        if row["expected_v2"] == 1.0 and rate < expert_min:
            bad.append(f"{row['name']}: v2 success {rate:.2f} < {expert_min}")
        if row["expected_v2"] == 0.0 and rate > 0.0:
            bad.append(f"{row['name']}: v2 rewarded a hack ({rate:.2f})")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0:16", help="half-open range lo:hi")
    parser.add_argument("--world-count", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    lo, hi = (int(v) for v in args.seeds.split(":"))
    rows = asyncio.run(run_suite(range(lo, hi), world_count=args.world_count, device=args.device))
    print(f"{'entry':42s} {'v1':>6s} {'v2':>6s}  v2 failed checks")
    for row in rows:
        print(f"{row['name']:42s} {row['v1']:6.2f} {row['v2']:6.2f}  {json.dumps(row['v2_failed_checks'])}")
    bad = violations(rows)
    for line in bad:
        print("VIOLATION:", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
