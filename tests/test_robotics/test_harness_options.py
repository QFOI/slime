"""Per-episode harness settings must not leak across concurrent samples."""

import asyncio
import json
from unittest.mock import AsyncMock

from robot_rollout.config import Config

from slime.agent.harness import ClaudeCodeHarness, DeepSeekHarness, HarnessContext
from slime.agent.harness import claude_code as cc_module


def test_robot_cc_options_ignore_ambient_coding_flags(monkeypatch):
    launch = AsyncMock(return_value=0)
    monkeypatch.setattr(cc_module, "run_agent", launch)
    monkeypatch.setenv("SLIME_AGENT_CC_EXTRA_ARGS", "--tools Bash")
    monkeypatch.setenv("SLIME_AGENT_CC_EXTRA_ENVS", '{"ANTHROPIC_BASE_URL":"http://wrong"}')
    ctx = HarnessContext(
        workdir="/workspace",
        session_id="own-token",
        adapter_url="http://own-gateway",
        options={"args": ["--tools", ""], "max_compactions": 0},
    )
    asyncio.run(ClaudeCodeHarness().launch_and_wait(None, ctx, "solve", 60))
    values = launch.call_args.kwargs
    assert "--tools Bash" not in values["start_cmd"]
    assert values["env"]["ANTHROPIC_BASE_URL"] == ctx.adapter_url
    assert values["env"]["ANTHROPIC_AUTH_TOKEN"] == ctx.session_id


def test_dsh_model_registry_cannot_bypass_episode_gateway():
    cfg = Config(
        model="remote",
        models={"remote": {"kind": "anthropic", "base_url": "https://upstream.example", "model": "other"}},
    )
    sandbox = AsyncMock()
    ctx = HarnessContext(
        workdir="/workspace",
        session_id="own-token",
        adapter_url="http://own-gateway",
        model_label="chosen-model",
        options={"config": cfg.model_dump()},
    )
    asyncio.run(DeepSeekHarness().write_config(sandbox, ctx))
    # Reparse exactly as the child CLI does; registry resolution must stay off.
    saved = Config.model_validate(json.loads(sandbox.write_file.call_args.args[1]))
    assert saved.model is None and saved.models == {}
    assert saved.backend.base_url == "http://own-gateway/v1"
    assert saved.backend.kind == "openai" and saved.backend.model == "chosen-model"
    assert saved.backend.api_key_env == "SLIME_DSH_GATEWAY_TOKEN"
