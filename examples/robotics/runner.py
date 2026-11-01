"""Prepare → official harness → freeze → Harbor verifier → cleanup.

Model text is never a reward. The simulator is a separate process and the
verifier is injected into a fresh container only after the agent is destroyed.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp

from robot_rollout.bridge import Gateway
from robot_rollout.trace import Trace, write_json

from slime.agent.harness import ClaudeCodeHarness, DeepSeekHarness
from slime.agent.sandbox import EXIT_TIME_BUDGET_EXCEEDED

from .config import Experiment, cc_options, episode_config
from .sandbox import DockerSandbox, docker
from .task import HarborTask, read_reward


@asynccontextmanager
async def robot_session(task: HarborTask, profile: str, directory: Path, timeout: float):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    spec = {
        "robot": task.robot.model_dump(),
        "profile": profile,
        "directory": str(directory / "robot"),
        "token": token,
    }
    write_json(directory / "session.json", spec)
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "NUMBA_DISABLE_JIT": "1"}
    process = None

    async def freeze():
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    with (directory / "session.log").open("wb") as log:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "examples.robotics.session",
                "--config",
                str(directory / "session.json"),
                "--port",
                str(port),
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            url = f"http://127.0.0.1:{port}/mcp"
            async with asyncio.timeout(timeout), aiohttp.ClientSession() as client:
                while True:
                    if process.returncode is not None:
                        raise RuntimeError("Robot session startup failed; see session.log")
                    try:
                        async with client.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 401:
                                break
                    except (aiohttp.ClientError, TimeoutError):
                        pass
                    await asyncio.sleep(0.1)
            yield {"type": "http", "url": url, "headers": {"Authorization": "Bearer " + token}}, freeze
        finally:
            await freeze()
            (directory / "session.json").unlink(missing_ok=True)


async def verify(task: HarborTask, image: str, directory: Path, profile: str) -> dict[str, float]:
    async with DockerSandbox(image, network="none") as sb:
        await sb.exec("mkdir -p /tests /artifacts /logs/verifier", check=True)
        await docker("cp", str(task.path / "tests") + "/.", f"{sb.sandbox_id}:/tests")
        await docker("cp", str(directory / "robot") + "/.", f"{sb.sandbox_id}:/artifacts")
        code, out, err = await sb.exec(
            "bash /tests/test.sh", env={"ROBOT_PROFILE": profile}, timeout=task.timeout("verifier")
        )
        target = directory / "logs/verifier"
        await sb.copy_out("/logs/verifier", target)
        (target / "stdout.txt").write_text(out)
        (target / "stderr.txt").write_text(err)
        if code:
            raise RuntimeError(f"Harbor verifier failed with exit {code}; see logs/verifier")
        return read_reward(target)


def cc_result(path: Path) -> dict:
    results = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            results.append(event)
    if not results or results[-1].get("is_error"):
        raise RuntimeError("CC did not produce a successful final result event")
    return results[-1]


async def run_episode(experiment: Experiment, task: HarborTask, *, agent: str, profile: str) -> dict:
    ident = uuid.uuid4().hex
    directory = Path(experiment.output_dir) / ident
    directory.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    result = {
        "trial_id": ident,
        "task": task.name,
        "task_digest": task.digest,
        "agent": agent,
        "profile": profile,
        "status": "starting",
        "run_dir": str(directory),
    }
    try:
        async with robot_session(task, profile, directory, experiment.startup_timeout_sec) as (mcp, freeze):
            cfg = episode_config(experiment, agent=agent, profile=profile, instruction=task.instruction, mcp=mcp)
            limit = (
                min(experiment.max_model_requests, cfg.dsh.max_model_requests)
                if agent == "dsh"
                else experiment.max_model_requests
            )
            max_output = cfg.claude.max_output_tokens if agent == "cc" else cfg.dsh.max_output_tokens
            trace = Trace(directory / "api.jsonl", [], cfg.run.api_trace)
            async with Gateway(cfg.backend, trace, max_output=max_output, request_limit=limit) as gateway:
                async with DockerSandbox(experiment.image) as sb:
                    await sb.exec("mkdir -p /workspace/.harness", check=True)
                    if agent == "cc":
                        _, version, _ = await sb.exec("claude --version", check=True)
                        if version.split()[0] != cfg.claude.expected_version:
                            raise RuntimeError(f"CC version mismatch: expected {cfg.claude.expected_version}")
                        result["cc_version"] = version.strip()
                        await sb.write_file(
                            "/workspace/.harness/mcp.json", json.dumps({"mcpServers": cfg.mcp_servers})
                        )
                        await sb.write_file("/workspace/.harness/system.md", cfg.prompt.system)
                        harness, options = ClaudeCodeHarness(), cc_options(cfg)
                    else:
                        harness, options = DeepSeekHarness(), {"config": cfg.model_dump()}
                    budget = min(task.timeout("agent"), cfg.run.timeout_sec)
                    # The shared sandbox transport has its own detached-command deadline.
                    # The outer guard also bounds setup and unresponsive transport calls.
                    try:
                        async with asyncio.timeout(budget + 120):
                            exit_code = await harness.run(
                                sb,
                                workdir="/workspace",
                                session_id=gateway.token,
                                adapter_url=gateway.url,
                                time_budget_sec=max(1, int(budget)),
                                prompt=cfg.prompt.user,
                                model_label=cfg.backend.model,
                                options=options,
                            )
                    finally:
                        # Freeze before diagnostics or container teardown, also
                        # on cancellation, an exception or the outer deadline.
                        await freeze()
                    metrics = await harness.collect_metrics(sb, workdir="/workspace")
                    result.update(agent_exit_code=exit_code, agent_metrics=metrics, model_requests=gateway.requests)
                    await sb.copy_out("/workspace/.harness", directory / "agent")
                    if (
                        exit_code == EXIT_TIME_BUDGET_EXCEEDED
                        or gateway.limit_reached
                        or metrics.get("compact_limit_reached")
                    ):
                        result["status"] = "truncated"
                    elif exit_code:
                        raise RuntimeError(f"Agent execution failed with exit {exit_code}")
                    elif agent == "cc":
                        result["response"] = cc_result(directory / "agent/trajectory.jsonl").get("result", "")
                    elif not metrics.get("completed"):
                        raise RuntimeError("DSH did not persist a completed turn")
                    else:
                        records = list((directory / "agent/dsh").glob("*/result.json"))
                        dsh_result = json.loads(records[0].read_text())
                        result["response"] = dsh_result.get("final_text", "")
                        result["dsh_versions"] = dsh_result.get("dsh_versions")
        # Agent and robot are stopped. No tool can modify evidence during judging.
        result["rewards"] = await verify(task, experiment.image, directory, profile)
        result["reward"] = result["rewards"]["reward"]
        result["grading_solved"] = result["reward"] == 1
        if result["status"] == "starting":
            result["status"] = "completed"
    except asyncio.CancelledError:
        result["status"] = "cancelled"
        raise
    except Exception as exc:
        result.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_sec"] = round(time.monotonic() - started, 3)
        write_json(directory / "result.json", result)
    return result
