"""Official DeepSeek Harness via the pinned robot-rollout ACP host.

DSH was not present in slime 540ea919. This optional integration implements the
same Sandbox/BaseHarness contract as Claude Code; it does not reimplement the
agent loop. robot-rollout supplies the rc.6 MCP image/PTC fixes and runtime lock.
"""

from __future__ import annotations

import json
import shlex

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, run_agent


class DeepSeekHarness(BaseHarness):
    name = "dsh"

    async def install_cli(self, sb: Sandbox) -> None:
        # Installation belongs to the immutable runtime image, with npm ci.
        await sb.exec("python -c 'import robot_rollout.dsh'", check=True)

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        from robot_rollout.config import Config

        # The task already selected its model. Re-resolving a registry here (or
        # inside the child) would bypass this episode's gateway and credentials.
        cfg = Config.model_validate({**ctx.options["config"], "model": None, "models": {}})
        cfg.agent = "dsh"
        cfg.backend.kind = "openai"
        cfg.backend.base_url = ctx.adapter_url + "/v1"
        cfg.backend.model = ctx.model_label
        cfg.backend.api_key_env = "SLIME_DSH_GATEWAY_TOKEN"
        cfg.backend.headers = {}
        cfg.run.workdir = ctx.workdir
        cfg.run.output_dir = f"{ctx.workdir}/.harness/dsh"
        await sb.write_file(f"{ctx.workdir}/.harness/dsh.json", cfg.model_dump_json(), user="agent")

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        path = f"{ctx.workdir}/.harness/dsh.json"
        cmd = shlex.join(["python", "-m", "robot_rollout", "run", "-c", path, "--prompt", prompt])
        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env={"SLIME_DSH_GATEWAY_TOKEN": ctx.session_id},
            time_budget_sec=time_budget_sec,
        )

    async def collect_metrics(self, sb: Sandbox, *, workdir: str) -> dict[str, int | float]:
        _, text, _ = await sb.exec(
            "python -c "
            + shlex.quote(
                "import glob,json; p=glob.glob(" + repr(f"{workdir}/.harness/dsh/*/result.json") + "); "
                "r=json.load(open(p[0])) if len(p)==1 else {}; "
                "print(json.dumps({'completed':int(r.get('status')=='success'),"
                "'ptc_subcalls':r.get('ptc_subcalls',0),"
                "'ptc_successful_subcalls':r.get('ptc_successful_subcalls',0)}))"
            ),
            user="agent",
            check=True,
        )
        return json.loads(text)
