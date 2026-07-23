from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from thesis_rl.sb3_extensions.rollout.masked import MaskedRolloutBuffer


def _make_buffer(buffer_size: int = 3, n_envs: int = 2) -> MaskedRolloutBuffer:
    return MaskedRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.9,
        n_envs=n_envs,
    )


def _add(
    buffer: MaskedRolloutBuffer,
    *,
    reward: list[float],
    episode_start: list[bool],
    valid_mask: np.ndarray | None = None,
) -> None:
    n_envs = buffer.n_envs
    buffer.add(
        obs=np.zeros((n_envs, 1), dtype=np.float32),
        action=np.zeros((n_envs, 1), dtype=np.float32),
        reward=np.asarray(reward, dtype=np.float32),
        episode_start=np.asarray(episode_start),
        value=torch.zeros(n_envs),
        log_prob=torch.zeros(n_envs),
        valid_mask=valid_mask,
    )


def test_data_abort_boundary_bootstraps_reward_and_marks_next_row_invalid() -> None:
    buffer = _make_buffer()

    _add(buffer, reward=[1.0, 1.0], episode_start=[True, True])
    buffer.close_previous_transition_as_data_abort(env_index=1, terminal_value=5.0)
    assert buffer.rewards[0, 1] == pytest.approx(1.0 + 0.9 * 5.0)
    assert buffer.rewards[0, 0] == pytest.approx(1.0)

    _add(
        buffer,
        reward=[1.0, 0.0],
        episode_start=[False, True],
        valid_mask=np.array([True, False]),
    )
    assert not buffer.full
    assert bool(buffer.valid_transitions[1, 0])
    assert not bool(buffer.valid_transitions[1, 1])
    # The boundary flag must still be genuine even though the row is invalid:
    # it is what stops the previous row's GAE from crossing the abort.
    assert bool(buffer.episode_starts[1, 1])

    _add(buffer, reward=[1.0, 2.0], episode_start=[False, True])
    assert buffer.full
    assert buffer.pos == 3


def test_close_previous_transition_raises_without_a_preceding_valid_row() -> None:
    buffer = _make_buffer()
    with pytest.raises(ValueError, match="No previous rollout transition"):
        buffer.close_previous_transition_as_data_abort(env_index=0, terminal_value=1.0)

    _add(buffer, reward=[1.0, 1.0], episode_start=[True, True])
    _add(
        buffer,
        reward=[1.0, 0.0],
        episode_start=[False, True],
        valid_mask=np.array([True, False]),
    )
    with pytest.raises(ValueError, match="preceding valid transition"):
        buffer.close_previous_transition_as_data_abort(env_index=1, terminal_value=1.0)


def test_invalid_cell_is_never_yielded_by_get() -> None:
    buffer = _make_buffer()
    _add(buffer, reward=[1.0, 1.0], episode_start=[True, True])
    buffer.close_previous_transition_as_data_abort(env_index=1, terminal_value=5.0)
    _add(
        buffer,
        reward=[1.0, 0.0],
        episode_start=[False, True],
        valid_mask=np.array([True, False]),
    )
    # Plant a sentinel in the invalid cell to prove it never reaches a minibatch.
    buffer.observations[1, 1] = np.array([999.0], dtype=np.float32)
    _add(buffer, reward=[1.0, 2.0], episode_start=[False, True])

    buffer.compute_returns_and_advantage(
        last_values=torch.zeros((2, 1)),
        dones=np.array([False, False]),
    )

    samples = list(buffer.get(batch_size=None))
    all_obs = np.concatenate([sample.observations.numpy() for sample in samples], axis=0)
    assert all_obs.shape[0] == buffer.buffer_size * buffer.n_envs - 1
    assert not np.any(np.isclose(all_obs.flatten(), 999.0))


def test_abort_boundary_does_not_leak_advantage_across_the_reset() -> None:
    buffer = _make_buffer()
    _add(buffer, reward=[1.0, 1.0], episode_start=[True, True])
    buffer.close_previous_transition_as_data_abort(env_index=1, terminal_value=5.0)
    _add(
        buffer,
        reward=[1.0, 0.0],
        episode_start=[False, True],
        valid_mask=np.array([True, False]),
    )
    _add(buffer, reward=[1.0, 2.0], episode_start=[False, True])

    buffer.compute_returns_and_advantage(
        last_values=torch.zeros((2, 1)),
        dones=np.array([False, False]),
    )

    # Row 0 for env 1 bootstraps purely from its own (boosted) reward and value,
    # since episode_starts[1, 1]=True cuts the recursion there: no advantage
    # from the invalid row (or the fresh episode after it) can leak backward.
    expected_row0_env1 = buffer.rewards[0, 1] - buffer.values[0, 1]
    assert buffer.advantages[0, 1] == pytest.approx(expected_row0_env1)
