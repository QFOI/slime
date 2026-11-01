"""Harbor, profile and reward boundaries; no model credentials or simulator."""

import json
import shutil
from pathlib import Path

import pytest

from examples.robotics.config import Experiment, cc_options, episode_config, load_experiment
from examples.robotics.task import HarborTask, read_reward

ROOT = Path(__file__).resolve().parents[2] / "examples/robotics"


@pytest.fixture
def task_dir(tmp_path):
    return Path(shutil.copytree(ROOT / "tasks/panda-reach", tmp_path / "task"))


def test_harbor_is_the_only_task_source(task_dir):
    task = HarborTask.load(task_dir)
    row = task.row()
    assert row["prompt"] == task.instruction
    assert set(row["metadata"]) == {"task_type", "task_path", "task_digest"}
    (task_dir / "tests/verify.py").write_text("# changed verifier\n")
    assert task.digest != HarborTask.load(task_dir).digest


def test_harbor_rejects_escaping_package(task_dir, tmp_path):
    (tmp_path / "secret").write_text("not part of the task")
    (task_dir / "environment/link").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="symlinks"):
        HarborTask.load(task_dir)


def test_task_digest_ignores_local_generated_metadata(task_dir):
    before = HarborTask.load(task_dir).digest
    cache = task_dir / "tests/__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "verify.cpython-312.pyc").write_bytes(b"local interpreter cache")
    (task_dir / ".DS_Store").write_bytes(b"local Finder metadata")
    assert HarborTask.load(task_dir).digest == before


@pytest.mark.parametrize("value", [None, {}, {"reward": True}, {"reward": float("nan")}, {"reward": 2}])
def test_missing_or_invalid_reward_is_not_zero(tmp_path, value):
    if value is not None:
        (tmp_path / "reward.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_reward(tmp_path)


def test_harbor_reward_formats(tmp_path):
    (tmp_path / "reward.txt").write_text("0.25\n")
    assert read_reward(tmp_path) == {"reward": 0.25}
    (tmp_path / "reward.json").write_text('{"reward": 1, "physical_success": 1}')
    assert read_reward(tmp_path) == {"reward": 1, "physical_success": 1}


@pytest.mark.parametrize("agent", ["cc", "dsh"])
@pytest.mark.parametrize("profile", ["intuitive", "expert"])
def test_profile_is_enforced_by_tools_and_prompt(agent, profile):
    experiment = load_experiment(ROOT / "configs/robotics.yaml")
    cfg = episode_config(
        experiment,
        agent=agent,
        profile=profile,
        instruction="test task",
        mcp={"type": "http", "url": "http://localhost:1/mcp"},
    )
    assert cfg.claude.tools == cfg.dsh.builtin_tools == []
    assert cfg.claude.multiagent.enabled is False
    if profile == "intuitive":
        assert cfg.dsh.tools_mode == "native"
        assert len(cfg.claude.allowed_tools) == 4
        assert "Do not write or execute code" in cfg.prompt.system
    else:
        assert cfg.dsh.tools_mode == "ptc"
        assert "mcp__robot__run_python" in cfg.claude.allowed_tools
        assert "RGB-D" in cfg.prompt.system
    options = cc_options(cfg)
    assert options["max_compactions"] == 2
    assert "--strict-mcp-config" in options["args"]
    assert options["args"][options["args"].index("--settings") + 1] == "/home/agent/.claude/settings.json"
    assert cfg.prompt.user.startswith("test task\n")


def test_extra_cli_flags_cannot_reenable_code_in_intuitive():
    with pytest.raises(ValueError, match="typed configuration|boundaries"):
        Experiment(tasks=["x"], runtime={"claude": {"extra_args": ["--tools", "Bash"]}})


def test_custom_mcp_only_mounted_for_expert():
    cfg = Experiment(
        tasks=["x"],
        runtime={"tools": {"servers": {"depth_camera": {"type": "http", "url": "http://localhost:9000/mcp"}}}},
    )
    robot = {"type": "http", "url": "http://localhost:9001/mcp"}
    intuitive = episode_config(cfg, agent="cc", profile="intuitive", instruction="task", mcp=robot)
    expert = episode_config(cfg, agent="cc", profile="expert", instruction="task", mcp=robot)
    assert set(intuitive.mcp_servers) == {"robot"}
    assert set(expert.mcp_servers) == {"robot", "depth_camera"}
    assert "mcp__depth_camera__*" in expert.claude.allowed_tools


def test_official_harbor_accepts_task_packages():
    # Install Harbor's pinned source for this optional cross-framework check.
    task_module = pytest.importorskip("harbor.models.task.task")
    for path in (ROOT / "tasks").iterdir():
        official = task_module.Task(path)
        assert task_module.Task.is_valid_dir(path)
        assert official.instruction == HarborTask.load(path).instruction + "\n"
