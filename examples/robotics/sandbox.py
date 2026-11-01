"""Docker implementation of slime.agent.sandbox.Sandbox, with per-trial cleanup.

The CLI has no host filesystem mounts. The separately owned robot process and
verifier evidence are reached only through the declared MCP service. Linux host
networking permits the CLI to reach loopback-only model and MCP gateways.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import uuid
from pathlib import Path

from slime.agent.sandbox import FileContent, SandboxTerminatedError


async def docker(*args: str, timeout: float = 120, check: bool = True) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    result = process.returncode, out.decode(errors="replace"), err.decode(errors="replace")
    if check and process.returncode:
        raise RuntimeError(f"docker {args[0]} failed ({process.returncode}): {result[2][-2000:]}")
    return result


class DockerSandbox:
    def __init__(self, image: str, *, network: str = "host"):
        self.image = image
        self.network = network
        self.sandbox_id = "slime-robot-" + uuid.uuid4().hex

    async def __aenter__(self):
        try:
            await docker(
                "run",
                "-d",
                "--name",
                self.sandbox_id,
                "--network",
                self.network,
                "--init",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "CHOWN",
                "--cap-add",
                "SETUID",
                "--cap-add",
                "SETGID",
                "--cap-add",
                "DAC_OVERRIDE",
                "--pids-limit",
                "256",
                "--memory",
                "4g",
                self.image,
                "sleep",
                "infinity",
            )
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        await docker("rm", "-f", self.sandbox_id, check=False, timeout=30)

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        argv = ["exec", "--user", user]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        result = await docker(*argv, self.sandbox_id, "bash", "-c", cmd, timeout=timeout, check=False)
        if result[0] and "is not running" in result[2]:
            raise SandboxTerminatedError("Robot agent container stopped")
        if check and result[0]:
            # Commands may contain temporary gateway auth; never echo them.
            raise RuntimeError(f"Agent command failed ({result[0]}): {result[2][-2000:]}")
        return result

    async def write_file(self, sandbox_path: str, content: FileContent, *, user="root"):
        target = Path(sandbox_path)
        await self.exec("mkdir -p " + shlex.quote(str(target.parent)), check=True)
        with tempfile.TemporaryDirectory(prefix="slime-robot-copy-") as tmp:
            source = Path(tmp) / "payload"
            if isinstance(content, Path):
                source.write_bytes(content.read_bytes())
            elif isinstance(content, bytes):
                source.write_bytes(content)
            else:
                source.write_text(content)
            os.chmod(source, 0o600)
            await docker("cp", str(source), f"{self.sandbox_id}:{sandbox_path}")
        await self.exec(f"chown {shlex.quote(user)} {shlex.quote(sandbox_path)}", check=True)

    async def read_file(self, sandbox_path: str, *, user="root") -> str:
        code, out, err = await self.exec(
            f"if test -f {shlex.quote(sandbox_path)}; then cat {shlex.quote(sandbox_path)}; fi",
            user=user,
        )
        if code:
            raise RuntimeError(f"Sandbox read failed: {err[-1000:]}")
        return out

    async def copy_out(self, sandbox_path: str, target: Path):
        target.mkdir(parents=True, exist_ok=True)
        await docker("cp", f"{self.sandbox_id}:{sandbox_path}/.", str(target))
