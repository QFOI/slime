"""Red-team suite on GPU: v2 scores every hack 0 and both experts 1; v1 is fooled by injected states."""

import asyncio

import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

if not wp.is_cuda_available():
    pytest.skip("Newton worlds need a CUDA device", allow_module_level=True)

from examples.robotics_newton.redteam import run_suite, violations


def test_redteam_suite():
    rows = {row["name"]: row for row in asyncio.run(run_suite(range(4), world_count=8))}
    assert violations(list(rows.values()), expert_min=1.0) == []
    # The reason v2 exists: the final-height rule accepts these.
    assert rows["injected: cube lodged on the hand"]["v1"] == 1.0
    assert rows["injected: cube embedded in the flange"]["v1"] == 1.0
