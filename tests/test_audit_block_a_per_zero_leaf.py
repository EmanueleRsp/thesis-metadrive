"""Audit 2026-09-06, block A7: PER sampling must never emit non-finite weights.

A prefix search can land on a zero-priority leaf (a masked data-abort slot, or
drift in the incrementally maintained sums). Its probability is 0, so the
importance weight is ``inf`` and the max-normalised batch becomes ``NaN``/0
without any exception, after which the critic parameters are NaN for good.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from thesis_rl.sb3_extensions.replay.prioritized import PrioritizedNStepReplayBuffer


def _buffer(n_envs: int = 1) -> PrioritizedNStepReplayBuffer:
    return PrioritizedNStepReplayBuffer(
        buffer_size=8 * n_envs,
        observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=n_envs,
        n_steps=1,
        gamma=1.0,
        beta_anneal_steps=10,
    )


def _fill(buffer: PrioritizedNStepReplayBuffer, count: int) -> None:
    n_envs = buffer.n_envs
    observation = np.zeros((n_envs, 1), dtype=np.float32)
    action = np.zeros((n_envs, 1), dtype=np.float32)
    for _ in range(count):
        buffer.add(
            observation,
            observation,
            action,
            np.zeros(n_envs),
            np.zeros(n_envs, dtype=bool),
            [{} for _ in range(n_envs)],
        )


def test_zero_priority_leaves_are_never_sampled_and_weights_stay_finite() -> None:
    buffer = _buffer()
    _fill(buffer, 6)
    # Mask half the active leaves the way a data abort does.
    for index in (1, 3, 5):
        buffer.raw_priorities.flat[index] = 0.0
        buffer._tree.set(index, 0.0)

    for _ in range(50):
        storage, _env, weights = buffer._sample_addresses(16)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0.0)
        assert not set(storage.tolist()) & {1, 3, 5}


def test_forced_zero_leaf_hit_is_redrawn_instead_of_producing_nan(monkeypatch) -> None:
    """Force the first prefix search onto a zero leaf and check the redraw path."""

    buffer = _buffer()
    _fill(buffer, 4)
    buffer.raw_priorities.flat[2] = 0.0
    buffer._tree.set(2, 0.0)
    real_find = buffer._tree.find_prefix_batch
    calls = {"count": 0}

    def _rigged(masses: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            return np.full(masses.shape, 2, dtype=np.int64)
        return real_find(masses)

    monkeypatch.setattr(buffer._tree, "find_prefix_batch", _rigged)

    storage, _env, weights = buffer._sample_addresses(4)

    assert calls["count"] >= 2
    assert 2 not in storage.tolist()
    assert np.all(np.isfinite(weights))


def test_inconsistent_tree_raises_instead_of_looping_forever(monkeypatch) -> None:
    buffer = _buffer()
    _fill(buffer, 4)
    buffer.raw_priorities.flat[2] = 0.0
    buffer._tree.set(2, 0.0)
    monkeypatch.setattr(
        buffer._tree,
        "find_prefix_batch",
        lambda masses: np.full(np.shape(masses), 2, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="zero-priority leaves"):
        buffer._sample_addresses(4)
