import ast
import collections
import json
import math
import re
import sys
from pathlib import Path

import torch

# Run against a completed, trusted local slime debug run. No prompts, images,
# host paths or credentials are copied into the resulting evidence document.
root = Path(sys.argv[1])
report = {"train_steps": [], "rollouts": [], "evals": [], "episodes": []}
log = re.sub(r"\x1b\[[0-9;]*m", "", (root / "driver.log").read_text())
report["diagnostics"] = {
    "allocator_retry_warnings": log.count("memory allocation failed with OOM"),
    "fatal_cuda_oom_exceptions": log.count("OutOfMemoryError:"),
    "nonfatal_freeze_gc_warnings": log.count("post-warmup freeze_gc failed"),
}
report["timings"] = []
for line in log.splitlines():
    match = re.search(r"model.py:\d+ - step (\d+): (\{.*\})", line)
    if match:
        metrics = ast.literal_eval(match[2])
        assert math.isfinite(metrics["train/grad_norm"]) and metrics["train/grad_norm"] > 0
        report["train_steps"].append({"step": int(match[1]), "metrics": metrics})
    match = re.search(r"train_metric_utils.py:\d+ - perf (\d+): (\{.*\})", line)
    if match:
        report["timings"].append({"step": int(match[1]), "metrics": ast.literal_eval(match[2])})
for path in sorted(root.glob("rollout-*.pt")):
    if ".train-recovery." in path.name:
        continue
    samples = torch.load(path, map_location="cpu", weights_only=False)["samples"]
    if "eval_" in path.name:
        assert all(s["metadata"]["status"] in {"completed", "truncated"} for s in samples)
        report["evals"].append(
            {
                "file": path.name,
                "episodes": len(samples),
                "status": dict(collections.Counter(s["metadata"]["status"] for s in samples)),
                "rewards": [s["reward"] for s in samples],
            }
        )
        continue
    row = {
        "file": path.name,
        "segments": len(samples),
        "episodes": len({s["rollout_id"] for s in samples}),
        "profile": sorted({s["metadata"]["profile"] for s in samples}),
        "agent": sorted({s["metadata"]["agent"] for s in samples}),
        "output_tokens": sum(s["response_length"] for s in samples),
        "image_exposures": sum(s["metadata"]["image_count"] for s in samples),
        "weight_versions": sorted({str(v) for s in samples for v in (s.get("weight_versions") or [])}),
    }
    for s in samples:
        assert len(s["rollout_log_probs"]) == s["response_length"] == len(s["loss_mask"])
        assert all(math.isfinite(v) for v in s["rollout_log_probs"])
        assert s["metadata"]["status"] in {"completed", "truncated"}
        assert len(s["tokens"]) > s["response_length"]
        if s["metadata"]["image_count"]:
            assert s["multimodal_train_inputs"]["pixel_values"].numel() > 0
    train = root / path.name.replace("rollout-", "train-")
    if train.exists():
        trained = torch.load(train, map_location="cpu", weights_only=False)["samples"]
        assert len(trained) == len(samples)
        diffs = []
        for before, after in zip(samples, trained, strict=True):
            assert before["response_length"] == int(after["response_lengths"])
            a, b = torch.as_tensor(before["rollout_log_probs"]), torch.as_tensor(after["log_probs"])
            assert a.shape == b.shape and torch.isfinite(b).all()
            diffs.append((a - b).abs())
        diff = torch.cat(diffs)
        row["replayed_logprob_abs_diff"] = {"mean": diff.mean().item(), "max": diff.max().item()}
    report["rollouts"].append(row)
for path in sorted((root / "episodes").glob("*/result.json")):
    value = json.loads(path.read_text())
    assert value["status"] in {"completed", "truncated"}, value.get("status")
    report["episodes"].append(
        {
            k: value[k]
            for k in (
                "task",
                "task_digest",
                "agent",
                "profile",
                "status",
                "reward",
                "rewards",
                "model_requests",
                "elapsed_sec",
                "cc_version",
                "dsh_versions",
                "agent_metrics",
            )
            if k in value
        }
    )
report["exit"] = json.loads((root / "exit.json").read_text()) if (root / "exit.json").exists() else None
if (root / "prompt-digests.json").exists():
    report["prompt_sha256"] = json.loads((root / "prompt-digests.json").read_text())
report["checkpoint_iterations"] = sorted(int(p.name.split("_")[-1]) for p in (root / "checkpoints").glob("iter_*"))
assert report["exit"] == {"exit_code": 0}, "Run did not finish successfully"
assert report["train_steps"] and report["evals"], "Missing training or evaluation evidence"
assert len(report["train_steps"]) == len(report["rollouts"])
assert {s["step"] for s in report["train_steps"]} == set(report["checkpoint_iterations"])
assert all("replayed_logprob_abs_diff" in r for r in report["rollouts"])
assert report["diagnostics"]["fatal_cuda_oom_exceptions"] == 0
(root / "sanitized-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
