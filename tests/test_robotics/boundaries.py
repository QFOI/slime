"""Actual CC/MuJoCo regression for no-action scoring and the compaction stop hook."""

import asyncio
import json
from pathlib import Path

from aiohttp import web

from examples.robotics.config import load_experiment
from examples.robotics.runner import run_episode
from examples.robotics.task import HarborTask
from tests.test_robotics.smoke import ReachModel


class StopModel(ReachModel):
    async def reply(self, request):
        body = await request.json()
        self.requests.append(body)
        return web.json_response(
            {
                "id": "stop-response",
                "model": body["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I cannot complete this task."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
            }
        )


async def main():
    cfg = load_experiment("examples/robotics/configs/robotics.yaml")
    cfg.runtime.model, cfg.runtime.models = None, {}
    cfg.runtime.claude.thinking_budget = 0
    cfg.runtime.claude.compaction.threshold_pct = 50
    cfg.runtime.claude.compaction.max_compactions = 0
    cfg.runtime.run.timeout_sec = 90
    cfg.output_dir = str(Path("examples/robotics/runs/boundaries").resolve())
    task = HarborTask.load(cfg.tasks[0])
    results = []
    for label, model in (("no-action", StopModel()), ("compaction-stop", ReachModel(input_tokens=190000))):
        async with model:
            cfg.runtime.backend.base_url = model.url
            cfg.runtime.backend.api_key_env = None
            cfg.runtime.backend.model = "boundary-test-model"
            result = await run_episode(cfg, task, agent="cc", profile="intuitive")
        results.append({"test": label, **result})
        print(json.dumps(results[-1]), flush=True)
        assert result.get("reward") == 0, result
        assert result["status"] == ("completed" if label == "no-action" else "truncated"), result
        if label == "compaction-stop":
            assert result["agent_metrics"]["compact_limit_reached"] == 1, result
    Path(cfg.output_dir, "report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
