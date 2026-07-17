from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.sb3_extensions.replay.config import resolve_transition_replay_config
from thesis_rl.sb3_extensions.replay.prioritized import _SumTree


def test_per_sum_tree_uses_float64_and_prefix_sampling() -> None:
    tree = _SumTree(3)
    tree.set(0, 1.0)
    tree.set(1, 2.0)
    tree.set(2, 3.0)

    assert tree.tree.dtype == np.float64
    assert tree.total == pytest.approx(6.0)
    assert tree.find_prefix(0.5) == 0
    assert tree.find_prefix(1.5) == 1
    assert tree.find_prefix(5.5) == 2


def test_per_requires_positive_beta_horizon() -> None:
    with pytest.raises(ValueError, match="beta_anneal_steps"):
        resolve_transition_replay_config(
            {"enabled": True, "prioritized": True},
            algorithm_name="td3_sb3",
        )
