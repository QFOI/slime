"""Read-only single-node training prerequisites. Does not start or stop Ray."""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

from .config import load_experiment


def inspect(config, checkpoint, required_gpus):
    cfg = load_experiment(config)
    failures = []
    if not cfg.training.enabled:
        failures.append("training.enabled must be true")
    gpus = torch.cuda.device_count()
    if gpus < required_gpus:
        failures.append(f"Need {required_gpus} local CUDA GPUs; found {gpus}")
    if cfg.session_python and not Path(cfg.session_python).is_file():
        failures.append("session_python must point to an existing simulation Python interpreter")
    packages = {}
    for package, module in (
        ("ray", "ray"),
        ("sglang", "sglang"),
        ("megatron-core", "megatron.core"),
        ("transformers", "transformers"),
        ("straw-queue", "straw"),
    ):
        try:
            present = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            present = False
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "source checkout" if present else None
        if not present:
            failures.append(f"Missing training dependency: {module}")
    if not (Path(checkpoint) / "config.json").is_file():
        failures.append("HF checkpoint must be a local downloaded model directory")
    elif not any(Path(checkpoint).glob("*.safetensors")):
        failures.append("HF checkpoint contains no safetensors weights")
    try:
        result = subprocess.run(["docker", "image", "inspect", cfg.image], capture_output=True, timeout=20)
        image_ok = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        image_ok = False
    if not image_ok:
        failures.append("Docker or the configured robotics image is unavailable")
    return {
        "ready": not failures,
        "cuda_gpus": gpus,
        "torch": torch.__version__,
        "packages": packages,
        "robot_image_available": image_ok,
        "session_python": cfg.session_python,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.config, args.checkpoint, args.gpus)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
