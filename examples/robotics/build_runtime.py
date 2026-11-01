"""Build a Linux CLI image from verified, locally installed robot-rollout runtimes.

No API credentials, simulator data, or task verifiers enter this image.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROBOT_ROLLOUT_COMMIT = "addbad62ac202332aad8b14eb10ea481a0402936"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--robot-rollout", type=Path, required=True)
    parser.add_argument("--tag", default="slime-robotics:0.1")
    args = parser.parse_args()
    root = args.robot_rollout.resolve()
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ROBOT_ROLLOUT_COMMIT:
        raise ValueError(f"Expected robot-rollout {ROBOT_ROLLOUT_COMMIT}, got {revision}")
    archive = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "archive",
            revision,
            "src",
            "pyproject.toml",
            "README.md",
            "LICENSE",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="slime-robot-runtime-") as tmp:
        stage = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(stage / "runtime", filter="data")
        shutil.copy2(root / ".tools/claude-2.1.296/claude", stage / "claude")
        shutil.copytree(root / ".tools/dsh-0.1.0-rc.6", stage / "dsh", symlinks=True)
        shutil.copytree(root / ".tools/node", stage / "node", symlinks=True)
        shutil.copy2(Path(__file__).with_name("Dockerfile"), stage / "Dockerfile")
        subprocess.run(["docker", "build", "--tag", args.tag, str(stage)], check=True)


if __name__ == "__main__":
    main()
