"""HTTP robot server: one GPU process, many Newton worlds, one episode per world.

python -m examples.robotics_newton.server --world-count 64 --device cuda:0 --port 18100

POST   /episodes                    {"task", "seed"} -> episode id, instruction, tools
POST   /episodes/{id}/tools/{name}  tool arguments    -> {"data", "images", "is_error"}
POST   /episodes/{id}/finish        evaluate, release -> {"reward", "rewards", "evaluator_state"}
DELETE /episodes/{id}               release without evaluation
GET    /episodes/{id}/privileged    cube and hand poses   } only with --allow-privileged:
POST   /episodes/{id}/privileged/cube  teleport the cube  } scripted experts and red-team tests
GET    /stats

The simulator runs on a dedicated thread. Requests that arrive between control
frames are applied together; frames are stepped only while some world has a
command in flight, and every world advances on each frame. Episodes with no
tool call for --idle-timeout seconds are released, so a client that died or was
cancelled before it could release its episode does not leak the world.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import io
import logging
import queue
import threading
import time
import uuid
from typing import Any

import numpy as np
from aiohttp import web
from PIL import Image

from .evaluator import EVALUATORS, V2, EvaluatorConfig
from .protocol import DEFAULT_LIMITS, Episode, Limits
from .sim.scene import DEFAULT_SCENE, TASKS, sample_layout
from .sim.worlds import DEFAULT_CAMERAS, DEFAULT_SIM, PandaWorlds
from .tools import TOOLS

logger = logging.getLogger(__name__)

_STATUS = {1: "completed", 2: "partial"}


@dataclasses.dataclass
class _Request:
    kind: str
    world: int
    payload: dict
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


def _settle(future: asyncio.Future, value: Any = None, error: BaseException | None = None) -> None:
    if future.done():
        return
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)


class SimWorker:
    """Owns PandaWorlds on one thread and implements protocol.Backend."""

    def __init__(
        self, world_count: int, *, device=None, scene=DEFAULT_SCENE, sim=DEFAULT_SIM, cameras=DEFAULT_CAMERAS
    ):
        self.world_count = world_count
        self.sim = sim
        self.cameras = cameras
        self._factory = lambda: PandaWorlds(world_count, device=device, scene=scene, sim=sim, cameras=cameras)
        self._queue: queue.SimpleQueue[_Request | None] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="newton-sim", daemon=True)
        self.stats = {"frames": 0, "step_sec": 0.0, "renders": 0, "render_sec": 0.0, "max_in_flight": 0}

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError("Newton worlds failed to build") from self._error

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)

    async def _submit(self, kind: str, world: int, **payload) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put(_Request(kind, world, payload, loop, future))
        return await future

    # ----- protocol.Backend ---------------------------------------------------

    async def reset(self, world: int, layout) -> None:
        await self._submit("reset", world, layout=layout)

    async def move(self, world, translation, rotvec, duration) -> dict:
        return await self._submit("move", world, translation=translation, rotvec=rotvec, duration=duration)

    async def gripper(self, world, width) -> dict:
        return await self._submit("gripper", world, width=width)

    async def observe(self, world) -> tuple[dict, dict[str, np.ndarray]]:
        return await self._submit("observe", world)

    async def snapshot(self, world) -> dict:
        return await self._submit("snapshot", world)

    async def evaluate(self, world, carry, samples) -> dict:
        snap = await self._submit("evaluate", world, carry=carry, samples=samples)
        snap["hold_duration_sec"] = len(snap["hold_samples"]) * self.sim.sample_every / self.sim.fps
        return snap

    async def place_cube(self, world: int, position, quat_xyzw, velocity) -> None:
        await self._submit("place_cube", world, position=position, quat_xyzw=quat_xyzw, velocity=velocity)

    # ----- simulator thread ---------------------------------------------------

    def _reply(self, request: _Request, value: Any = None, error: BaseException | None = None) -> None:
        request.loop.call_soon_threadsafe(_settle, request.future, value, error)

    def _drain(self, block: bool) -> list[_Request | None]:
        items = []
        try:
            items.append(self._queue.get(timeout=1.0) if block else self._queue.get_nowait())
            while True:
                items.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return items

    def _run(self) -> None:
        try:
            worlds = self._factory()
        except BaseException as exc:  # surfaced by start()
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        in_flight: dict[int, _Request] = {}
        while True:
            requests = self._drain(block=not in_flight)
            if None in requests:
                for request in [*in_flight.values(), *(r for r in requests if r is not None)]:
                    self._reply(request, error=RuntimeError("simulator stopped"))
                return
            observes = []
            for request in requests:
                w, p = request.world, request.payload
                try:
                    if request.kind == "reset":
                        # A released episode may have left a command in flight; it can never finish now.
                        if (abandoned := in_flight.pop(w, None)) is not None:
                            self._reply(abandoned, error=RuntimeError("world was reset"))
                        worlds.reset([w], [p["layout"]])
                        self._reply(request)
                    elif request.kind == "snapshot":
                        self._reply(request, worlds.snapshot([w])[0])
                    elif request.kind == "observe":
                        observes.append(request)
                    elif request.kind == "move":
                        worlds.begin_move([w], [p["translation"]], [p["rotvec"]], [p["duration"]])
                        in_flight[w] = request
                    elif request.kind == "gripper":
                        worlds.begin_gripper([w], [p["width"]])
                        in_flight[w] = request
                    elif request.kind == "evaluate":
                        worlds.begin_evaluate([w], [p["carry"]], [p["samples"]])
                        in_flight[w] = request
                    elif request.kind == "place_cube":
                        worlds.place_cube([w], [p["position"]], [p["quat_xyzw"]], [p["velocity"]])
                        self._reply(request)
                    else:
                        raise ValueError(f"unknown request {request.kind}")
                except Exception as exc:
                    logger.exception("simulator request %s failed", request.kind)
                    self._reply(request, error=exc)
            if observes:
                started = time.perf_counter()
                ids = [r.world for r in observes]
                images, snaps = worlds.render(ids), worlds.snapshot(ids)
                for request, snap in zip(observes, snaps, strict=True):
                    self._reply(request, (snap, images[request.world]))
                self.stats["renders"] += 1
                self.stats["render_sec"] += time.perf_counter() - started
            if in_flight:
                self.stats["max_in_flight"] = max(self.stats["max_in_flight"], len(in_flight))
                started = time.perf_counter()
                result = worlds.step()
                self.stats["frames"] += 1
                self.stats["step_sec"] += time.perf_counter() - started
                done = [w for w in in_flight if result[w]]
                if done:
                    for w, snap in zip(done, worlds.snapshot(done), strict=True):
                        snap["status"] = _STATUS[int(result[w])]
                        self._reply(in_flight.pop(w), snap)


def _png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class RobotServer:
    def __init__(
        self,
        worker: SimWorker,
        *,
        limits: Limits = DEFAULT_LIMITS,
        evaluator: EvaluatorConfig = V2,
        allow_privileged: bool = False,
        idle_timeout_sec: float | None = 900.0,
    ):
        self.worker = worker
        self.limits = limits
        self.evaluator = evaluator
        self.allow_privileged = allow_privileged
        self.idle_timeout_sec = idle_timeout_sec
        self.episodes: dict[str, Episode] = {}
        self.free: asyncio.Queue[int] | None = None
        self._reaper: asyncio.Task | None = None
        self.cameras = [
            {"name": c.name, "width": c.width, "height": c.height, "mount": "wrist" if c.mount_body else "fixed"}
            for c in worker.cameras
        ]

    def app(self) -> web.Application:
        app = web.Application(client_max_size=16 * 1024**2)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        app.router.add_post("/episodes", self.start_episode)
        app.router.add_post("/episodes/{id}/tools/{name}", self.call_tool)
        app.router.add_post("/episodes/{id}/finish", self.finish_episode)
        app.router.add_delete("/episodes/{id}", self.release_episode)
        app.router.add_get("/episodes/{id}/privileged", self.privileged)
        app.router.add_post("/episodes/{id}/privileged/cube", self.place_cube)
        app.router.add_get("/stats", self.get_stats)
        return app

    async def _startup(self, _app) -> None:
        self.free = asyncio.Queue()
        for world in range(self.worker.world_count):
            self.free.put_nowait(world)
        if self.idle_timeout_sec:
            self._reaper = asyncio.create_task(self._reap_idle(self.idle_timeout_sec))

    async def _cleanup(self, _app) -> None:
        if self._reaper is not None:
            self._reaper.cancel()

    async def _reap_idle(self, timeout: float) -> None:
        while True:
            await asyncio.sleep(min(30.0, timeout / 4))
            for episode in list(self.episodes.values()):
                if episode.idle_for(timeout):
                    logger.warning(
                        "releasing episode %s on world %d after %.0f s idle", episode.id, episode.world, timeout
                    )
                    episode.close()
                    self._release(episode)

    def _episode(self, request: web.Request) -> Episode:
        episode = self.episodes.get(request.match_info["id"])
        if episode is None:
            raise web.HTTPNotFound(text="unknown episode")
        return episode

    async def start_episode(self, request: web.Request) -> web.Response:
        body = await request.json()
        task, seed = body.get("task"), body.get("seed")
        if task not in TASKS or isinstance(seed, bool) or not isinstance(seed, int):
            raise web.HTTPBadRequest(text=f"task must be one of {TASKS} and seed an integer")
        layout = sample_layout(task, seed)
        world = await self.free.get()
        try:
            await self.worker.reset(world, layout)
        except BaseException:
            self.free.put_nowait(world)
            raise
        episode = Episode(
            uuid.uuid4().hex,
            world,
            layout,
            self.worker,
            limits=self.limits,
            evaluator=self.evaluator,
            cameras=self.cameras,
        )
        self.episodes[episode.id] = episode
        return web.json_response(
            {
                "episode_id": episode.id,
                "world": world,
                "task": task,
                "seed": seed,
                "instruction": episode.instruction,
                "tools": TOOLS,
            }
        )

    async def call_tool(self, request: web.Request) -> web.Response:
        episode = self._episode(request)
        arguments = await request.json() if request.can_read_body else {}
        reply = await episode.call(request.match_info["name"], arguments)
        loop = asyncio.get_running_loop()
        encoded = await asyncio.gather(*(loop.run_in_executor(None, _png, img) for img in reply.images.values()))
        images = [
            {"camera": name, "width": img.shape[1], "height": img.shape[0], "mime_type": "image/png", "data": data}
            for (name, img), data in zip(reply.images.items(), encoded, strict=True)
        ]
        return web.json_response({"data": reply.data, "images": images, "is_error": reply.is_error})

    def _release(self, episode: Episode) -> None:
        if self.episodes.pop(episode.id, None) is not None:
            self.free.put_nowait(episode.world)

    async def finish_episode(self, request: web.Request) -> web.Response:
        episode = self._episode(request)
        try:
            result = await episode.finish()
        finally:
            self._release(episode)
        return web.json_response({**result, "events": episode.events})

    async def release_episode(self, request: web.Request) -> web.Response:
        self._release(self._episode(request))
        return web.json_response({"released": True})

    async def privileged(self, request: web.Request) -> web.Response:
        if not self.allow_privileged:
            raise web.HTTPForbidden(text="privileged state is disabled")
        episode = self._episode(request)
        snap = await self.worker.snapshot(episode.world)
        keys = ("cube_position", "cube_quat_xyzw", "hand_position", "hand_quat_xyzw")
        return web.json_response({key: snap[key].tolist() for key in keys})

    async def place_cube(self, request: web.Request) -> web.Response:
        if not self.allow_privileged:
            raise web.HTTPForbidden(text="privileged state is disabled")
        episode = self._episode(request)
        body = await request.json()
        await self.worker.place_cube(
            episode.world,
            body["position"],
            body.get("quat_xyzw", [0, 0, 0, 1]),
            body.get("linear_velocity", [0, 0, 0]),
        )
        return web.json_response({"placed": True})

    async def get_stats(self, _request: web.Request) -> web.Response:
        stats = dict(self.worker.stats)
        stats["world_count"] = self.worker.world_count
        stats["episodes"] = len(self.episodes)
        stats["mean_step_ms"] = 1000 * stats["step_sec"] / max(stats["frames"], 1)
        stats["mean_render_ms"] = 1000 * stats["render_sec"] / max(stats["renders"], 1)
        return web.json_response(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--allow-privileged", action="store_true")
    parser.add_argument("--evaluator", choices=sorted(EVALUATORS), default="v2")
    parser.add_argument("--idle-timeout", type=float, default=900.0, help="seconds; 0 disables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    worker = SimWorker(args.world_count, device=args.device)
    worker.start()
    try:
        server = RobotServer(
            worker,
            evaluator=EVALUATORS[args.evaluator],
            allow_privileged=args.allow_privileged,
            idle_timeout_sec=args.idle_timeout,
        )
        web.run_app(server.app(), host=args.host, port=args.port)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
