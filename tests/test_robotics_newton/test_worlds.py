"""GPU check of the batched Newton worlds: tracking, gripper, evaluator, cameras."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

if not wp.is_cuda_available():
    pytest.skip("Newton worlds need a CUDA device", allow_module_level=True)

from examples.robotics_newton.sim.scene import sample_layout
from examples.robotics_newton.sim.worlds import PandaWorlds


def settle(worlds, ids, max_frames=400):
    for _ in range(max_frames):
        result = worlds.step()
        if all(result[w] for w in ids):
            return result
    raise AssertionError("commands did not finish")


def test_worlds_track_commands_independently():
    worlds = PandaWorlds(3)
    worlds.reset([0, 1, 2], [sample_layout("lift", s) for s in range(3)])
    before = worlds.snapshot([2])[0]["tcp_position"]

    worlds.begin_move([0, 1], [[0.05, 0.0, -0.05], [0.0, 0.08, 0.0]], [[0.0, 0.0, 0.3], [0.0, 0.0, 0.0]], [0.8, 0.6])
    result = settle(worlds, [0, 1])
    assert list(result[:2]) == [1, 1]  # completed, not partial
    for snap in worlds.snapshot([0, 1]):
        assert np.linalg.norm(snap["target_position"] - snap["tcp_position"]) < 0.003
    # The idle world held its pose while the others moved.
    assert np.linalg.norm(worlds.snapshot([2])[0]["tcp_position"] - before) < 0.002

    worlds.begin_gripper([0], [0.0])
    settle(worlds, [0])
    assert worlds.snapshot([0])[0]["gripper_width"] < 0.005

    worlds.begin_evaluate([1])
    settle(worlds, [1])
    assert worlds.snapshot([1])[0]["hold_samples"] == [False] * 10  # cube still on the table

    images = worlds.render([0, 2])
    assert set(images) == {0, 2}
    for cams in images.values():
        assert set(cams) == {"front", "wrist"}
        assert all(img.shape == (256, 256, 3) and img.std() > 1 for img in cams.values())
