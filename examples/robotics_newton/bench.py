"""Throughput of the batched worlds: control frames and camera renders vs world count.

Every world keeps a small random move in flight, so each frame does full work.

python -m examples.robotics_newton.bench --world-counts 16 64 128 --device cuda:0
"""

from __future__ import annotations

import argparse
import io
import json
import time

import numpy as np
import warp as wp
from PIL import Image

from .sim.scene import sample_layout
from .sim.worlds import PandaWorlds


def bench(world_count: int, device, frames: int, renders: int) -> dict:
    started = time.perf_counter()
    worlds = PandaWorlds(world_count, device=device)
    build_sec = time.perf_counter() - started
    ids = list(range(world_count))
    worlds.reset(ids, [sample_layout("lift", seed) for seed in ids])
    rng = np.random.default_rng(0)

    def new_moves(targets):
        n = len(targets)
        worlds.begin_move(targets, rng.uniform(-0.05, 0.05, (n, 3)), rng.uniform(-0.2, 0.2, (n, 3)), [0.5] * n)

    new_moves(ids)
    for _ in range(10):
        worlds.step()
    completed = 0
    started = time.perf_counter()
    for _ in range(frames):
        result = worlds.step()
        done = [w for w in ids if result[w]]
        if done:
            completed += len(done)
            new_moves(done)
    step_sec = (time.perf_counter() - started) / frames

    worlds.render(ids)
    started = time.perf_counter()
    for _ in range(renders):
        images = worlds.render(ids)
    render_sec = (time.perf_counter() - started) / renders

    sample = images[0]["front"]
    started = time.perf_counter()
    for _ in range(20):
        Image.fromarray(sample).save(io.BytesIO(), format="PNG")
    png_sec = (time.perf_counter() - started) / 20

    return {
        "world_count": world_count,
        "build_sec": round(build_sec, 1),
        "frame_ms": round(1000 * step_sec, 2),
        "realtime_factor_per_world": round(worlds.frame_dt / step_sec, 2),
        "realtime_factor_aggregate": round(world_count * worlds.frame_dt / step_sec, 1),
        "moves_completed": completed,
        "render_ms_all_worlds": round(1000 * render_sec, 2),
        "render_ms_per_world": round(1000 * render_sec / world_count, 3),
        "cameras": [c.spec.name for c in worlds.cameras],
        "png_ms_per_image": round(1000 * png_sec, 2),
        "mempool_mb": round(wp.get_mempool_used_mem_current(worlds.device) / 2**20, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-counts", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--device", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--renders", type=int, default=20)
    args = parser.parse_args()
    wp.config.quiet = True
    for count in args.world_counts:
        print(json.dumps(bench(count, args.device, args.frames, args.renders)), flush=True)


if __name__ == "__main__":
    main()
