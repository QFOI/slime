"""python -m examples.robotics {validate,manifest,rollout} experiment.yaml"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from .config import load_experiment
from .task import HarborTask


async def evaluate(path: Path) -> dict:
    from slime.utils.types import Sample

    from .generate import generate

    cfg = load_experiment(path)
    cases = list(itertools.product(cfg.tasks, cfg.agents, cfg.profiles, range(cfg.repeats)))
    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def run(index, task_path, agent, profile, repeat):
        task = HarborTask.load(task_path)
        sample = Sample(index=index, group_index=index, **task.row())
        sample.metadata.update(agent=agent, profile=profile, repeat=repeat)
        async with semaphore:
            samples = await generate(SimpleNamespace(robotics_config=str(path)), sample, {}, evaluation=True)
        result = samples[0].metadata
        print(
            json.dumps(
                {k: result[k] for k in ("trial_id", "task", "agent", "profile", "status")}
                | {"reward": result.get("reward")}
            ),
            flush=True,
        )
        return result

    results = await asyncio.gather(*(run(i, *case) for i, case in enumerate(cases)))
    report = {
        "total": len(results),
        "completed": sum(r["status"] == "completed" for r in results),
        "passed": sum(r.get("reward") == 1 for r in results),
        "cases": results,
    }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["validate", "manifest", "rollout"])
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.config.resolve()
    cfg = load_experiment(path)
    tasks = [HarborTask.load(p) for p in cfg.tasks]
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "tasks": [{"name": t.name, "digest": t.digest} for t in tasks],
                    "agents": cfg.agents,
                    "profiles": cfg.profiles,
                },
                indent=2,
            )
        )
    elif args.command == "manifest":
        rows = []
        for task, agent, profile in itertools.product(tasks, cfg.agents, cfg.profiles):
            row = task.row()
            row["metadata"].update(agent=agent, profile=profile, robotics_config=str(path))
            rows.append(row)
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        if args.output:
            args.output.write_text(payload)
        else:
            print(payload, end="")
    else:
        report = asyncio.run(evaluate(path))
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        if report["completed"] != report["total"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
