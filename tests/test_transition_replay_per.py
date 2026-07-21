from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from thesis_rl.sb3_extensions.replay.config import resolve_transition_replay_config
from thesis_rl.sb3_extensions.replay.prioritized import PrioritizedNStepReplayBuffer, _SumTree


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


def test_per_insertion_priority_tracks_exact_maximum_across_updates_and_overwrite() -> None:
    buffer = PrioritizedNStepReplayBuffer(
        # SB3 divides the supplied capacity by n_envs; retain two storage rows
        # so the final insertion exercises ring overwrite.
        buffer_size=4,
        observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=2,
        n_steps=1,
        gamma=0.99,
        beta_anneal_steps=10,
    )

    observation = np.zeros((2, 1), dtype=np.float32)
    action = np.zeros((2, 1), dtype=np.float32)
    reward = np.zeros(2, dtype=np.float32)
    done = np.zeros(2, dtype=bool)
    infos = [{}, {}]

    def add() -> None:
        buffer.add(observation, observation, action, reward, done, infos)
        active = buffer.raw_priorities[: buffer._active_count()]
        expected = float(np.max(active))
        assert buffer._insertion_priority() == pytest.approx(expected)

    add()
    buffer.update_priorities(np.array([0, 1]), np.array([2.0, 5.0]))
    add()
    buffer.update_priorities(np.array([1, 2, 3]), np.array([1.0, 3.0, 4.0]))
    add()

    assert np.all(buffer.raw_priorities[0] == pytest.approx(4.0))
    assert buffer._insertion_priority() == pytest.approx(float(np.max(buffer.raw_priorities)))
