"""Async client for server.py, shared by scripted experts and rollout environments."""

from __future__ import annotations

import aiohttp


class RobotClient:
    def __init__(self, base_url: str, *, timeout_sec: float = 600):
        self.base_url = base_url.rstrip("/")
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec))

    async def close(self) -> None:
        await self._session.close()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        async with self._session.request(method, self.base_url + path, **kwargs) as response:
            if response.status != 200:
                raise RuntimeError(f"{method} {path} -> {response.status}: {(await response.text())[:500]}")
            return await response.json()

    async def start(self, task: str, seed: int) -> dict:
        return await self._request("POST", "/episodes", json={"task": task, "seed": seed})

    async def call(self, episode_id: str, name: str, arguments: dict | None = None) -> dict:
        """Returns {"data", "images": [{camera, width, height, mime_type, data(base64)}], "is_error"}."""
        return await self._request("POST", f"/episodes/{episode_id}/tools/{name}", json=arguments or {})

    async def finish(self, episode_id: str) -> dict:
        return await self._request("POST", f"/episodes/{episode_id}/finish")

    async def release(self, episode_id: str) -> dict:
        return await self._request("DELETE", f"/episodes/{episode_id}")

    async def privileged(self, episode_id: str) -> dict:
        return await self._request("GET", f"/episodes/{episode_id}/privileged")

    async def place_cube(self, episode_id: str, position, quat_xyzw=(0, 0, 0, 1), linear_velocity=(0, 0, 0)) -> dict:
        body = {"position": list(position), "quat_xyzw": list(quat_xyzw), "linear_velocity": list(linear_velocity)}
        return await self._request("POST", f"/episodes/{episode_id}/privileged/cube", json=body)

    async def stats(self) -> dict:
        return await self._request("GET", "/stats")
