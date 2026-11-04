"""Experiment configuration stays separate from Harbor task data."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robot_rollout.config import Config, Prompt, prompt_text


class Training(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[str] = Field(min_length=1)
    agents: list[Literal["cc", "dsh"]] = Field(default_factory=lambda: ["cc", "dsh"], min_length=1)
    profiles: list[Literal["intuitive", "expert"]] = Field(
        default_factory=lambda: ["intuitive", "expert"], min_length=1
    )
    repeats: int = Field(default=1, gt=0)
    concurrency: int = Field(default=1, gt=0)
    image: str = "slime-robotics:0.1"
    output_dir: str = "runs"
    startup_timeout_sec: float = Field(default=180, gt=0)
    max_model_requests: int = Field(default=60, gt=0)
    runtime: Config = Field(default_factory=Config)
    training: Training = Field(default_factory=Training)

    @model_validator(mode="after")
    def profile_boundary(self):
        cc = self.runtime.claude
        if cc.extra_args or cc.settings or cc.plugin_dirs or cc.add_dirs:
            raise ValueError("Robot profiles own CLI tool/settings boundaries; use explicit supported config fields")
        if cc.multiagent.enabled or cc.multiagent.teams:
            raise ValueError("Robot profiles serialize physical control; multiagent must be disabled")
        if self.runtime.evaluation.cases or self.runtime.evaluation.cases_file:
            raise ValueError("Use Harbor tasks, not a second robot-rollout case dataset")
        if "robot" in self.runtime.mcp_servers:
            raise ValueError("The robot MCP server is task-owned; use another name for custom expert tools")
        return self


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path).expanduser().resolve()
    cfg = Experiment.model_validate(yaml.safe_load(path.read_text()))
    cfg.tasks = [str((path.parent / p).resolve()) for p in cfg.tasks]
    cfg.output_dir = str((path.parent / cfg.output_dir).resolve())
    prompt = cfg.runtime.prompt
    for field in ("user_file", "system_file", "append_file"):
        if value := getattr(prompt, field):
            setattr(prompt, field, str((path.parent / value).resolve()))
    for part in prompt.system_parts:
        if part.file:
            part.file = str((path.parent / part.file).resolve())
    return cfg


def tool_names(profile: str) -> list[str]:
    contract = json.loads(files("robot_rollout").joinpath("resources/robot-tools.json").read_text())
    intuitive = {"observe", "move_to", "gripper", "batch_actions"}
    return ["mcp__robot__" + t["name"] for t in contract if profile == "expert" or t["name"] in intuitive]


def episode_config(experiment: Experiment, *, agent: str, profile: str, instruction: str, mcp: dict) -> Config:
    cfg = experiment.runtime.model_copy(deep=True)
    if cfg.prompt.user is None and cfg.prompt.user_file is None:
        cfg.prompt.user = "Complete the task and verify its final physical state."
    user, system, append = prompt_text(cfg.prompt)
    preset = (Path(__file__).parent / "prompts" / f"{profile}.md").read_text()
    cfg.agent = agent
    cfg.prompt = Prompt(
        system_mode="replace",
        system="\n\n".join(x for x in (preset, system, append) if x),
        user="\n\n".join(x for x in (instruction, user) if x),
    )
    cfg.claude.tools = []
    extra_servers = cfg.mcp_servers if profile == "expert" else {}
    cfg.claude.allowed_tools = tool_names(profile) + [f"mcp__{name}__*" for name in extra_servers]
    cfg.claude.permission_mode = "dontAsk"
    cfg.claude.strict_mcp = True
    cfg.claude.setting_sources = []
    cfg.dsh.tools_mode = "native" if profile == "intuitive" else "ptc"
    cfg.dsh.builtin_tools = []
    cfg.dsh.max_parallel_tool_calls = cfg.dsh.max_parallel_subcalls = 1
    cfg.dsh.runtime_dir = "/opt/dsh"
    cfg.dsh.node = "/opt/node/bin/node"
    cfg.mcp_servers = {**extra_servers, "robot": mcp}
    return cfg


def cc_options(cfg: Config) -> dict:
    cc = cfg.claude
    args = [
        "--tools",
        "",
        "--allowedTools",
        ",".join(cc.allowed_tools),
        "--disallowedTools",
        "Agent,Task,TeamCreate,TeamDelete,SendMessage",
        "--strict-mcp-config",
        "--mcp-config",
        "/workspace/.harness/mcp.json",
        "--setting-sources",
        "",
        "--settings",
        "/home/agent/.claude/settings.json",
        "--system-prompt-file",
        "/workspace/.harness/system.md",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    for flag, value in (
        ("--max-turns", cc.max_turns),
        ("--max-budget-usd", cc.max_budget_usd),
        ("--effort", cc.effort),
    ):
        if value is not None:
            args.extend([flag, str(value)])
    env = {
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "ENABLE_TOOL_SEARCH": "false",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "0",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(cc.max_output_tokens),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(cc.context_window),
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(cc.compaction.threshold_pct),
        "DISABLE_AUTO_COMPACT": "0" if cc.compaction.enabled else "1",
        "DISABLE_COMPACT": "0" if cc.compaction.enabled else "1",
        **cc.extra_env,
    }
    if cc.thinking_budget is not None:
        env["MAX_THINKING_TOKENS"] = str(cc.thinking_budget)
    return {
        "args": args,
        "env": env,
        "max_compactions": cc.compaction.max_compactions,
        "settings": {"autoCompactEnabled": cc.compaction.enabled},
        "launch_flags": "--permission-mode dontAsk --output-format stream-json --include-partial-messages --include-hook-events --verbose",
    }
