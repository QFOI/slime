"""HTTP server behaviour that needs no GPU: abandoned episodes give their world back."""

import asyncio

import pytest

pytest.importorskip("newton")
pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer

from examples.robotics_newton.server import RobotServer

from .test_protocol import FakeBackend


class FakeWorker(FakeBackend):
    world_count = 1
    cameras = []

    async def reset(self, world, layout):
        pass


def test_idle_episodes_are_released():
    async def run():
        server = RobotServer(FakeWorker(), idle_timeout_sec=0.2)
        async with TestClient(TestServer(server.app())) as client:
            first = await (await client.post("/episodes", json={"task": "lift", "seed": 1})).json()
            await asyncio.sleep(0.5)  # the reaper checks every timeout / 4
            assert not server.episodes
            # The only world is free again, and the abandoned episode is gone.
            second = await asyncio.wait_for(client.post("/episodes", json={"task": "lift", "seed": 2}), 2)
            assert second.status == 200
            stale = await client.post(f"/episodes/{first['episode_id']}/tools/observe", json={})
            assert stale.status == 404

    asyncio.run(run())
