"""Hand-calculated N-step arithmetic tests for the deployed PER buffer.

`PrioritizedNStepReplayBuffer` (`src/thesis_rl/sb3_extensions/replay/prioritized.py`)
duplicates the N-step return/discount computation of the vendored
`NStepReplayBuffer` (`third_party/stable-baselines3/stable_baselines3/common/buffers.py`)
instead of reusing it. The fork's own test suite
(`third_party/stable-baselines3/tests/test_n_step_replay.py`) exercises that
arithmetic only for the base class. Since production TD3-SB3/SAC-SB3 configs
set `prioritized: true` (`conf/agent/planner/algorithm/td3_sb3.yaml`,
`conf/agent/planner/algorithm/sac_sb3.yaml`), the subclass actually deployed
had no dedicated numeric regression test. These tests close that gap by
mirroring the fork's hand-calculation style against
`PrioritizedNStepReplayBuffer` directly, per
`docs/specifications/transition_replay_v1_specification.md` AC-003/004/005/006/009.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from thesis_rl.sb3_extensions.replay.prioritized import PrioritizedNStepReplayBuffer

OBS_SPACE = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)


def create_per_buffer(
    buffer_size: int = 10, n_steps: int = 3, gamma: float = 0.99
) -> PrioritizedNStepReplayBuffer:
    return PrioritizedNStepReplayBuffer(
        buffer_size=buffer_size,
        observation_space=OBS_SPACE,
        action_space=ACT_SPACE,
        device="cpu",
        n_envs=1,
        n_steps=n_steps,
        gamma=gamma,
        beta_anneal_steps=10,
    )


def fill_buffer(
    buffer: PrioritizedNStepReplayBuffer,
    rewards: list[float],
    done_at: int | None = None,
    truncated_at: int | None = None,
) -> None:
    """Fill with one transition per reward; optional done/truncation index."""

    for i, reward in enumerate(rewards):
        obs = np.full((1, 4), i, dtype=np.float32)
        next_obs = np.full((1, 4), i + 1, dtype=np.float32)
        action = np.zeros((1, 2), dtype=np.float32)
        done = np.array([1.0 if i == done_at else 0.0])
        infos = [{"TimeLimit.truncated": i == truncated_at}]
        buffer.add(obs, next_obs, action, np.array([reward], dtype=np.float32), done, infos)


def compute_expected_nstep_reward(
    gamma: float, rewards: list[float], stop_idx: int | None = None
) -> float:
    """Discounted sum of `rewards`, stopping (inclusive) at `stop_idx` if set."""

    last_sum = 0.0
    for step in reversed(range(len(rewards))):
        next_non_terminal = step != stop_idx
        last_sum = rewards[step] + gamma * next_non_terminal * last_sum
    return last_sum


def sample_from_start(buffer: PrioritizedNStepReplayBuffer, base_idx: int = 0):
    """Bypass PER address sampling to read the deterministic transition at `base_idx`."""

    obs, actions, next_obs, dones, rewards, discounts = buffer._get_samples_with_env_indices(
        np.array([base_idx]), np.array([0]), None
    )
    return rewards.item(), dones.item(), discounts.item()


def test_per_nstep_three_step_hand_calculation() -> None:
    """AC-004: G = 1 + 2*gamma + 3*gamma**2, discount = gamma**3, done = 0."""

    gamma = 0.9
    buffer = create_per_buffer(n_steps=3, gamma=gamma)
    fill_buffer(buffer, rewards=[1.0, 2.0, 3.0, 1.0, 1.0])

    reward, done, discount = sample_from_start(buffer, base_idx=0)

    expected_return = 1.0 + 2.0 * gamma + 3.0 * gamma**2
    np.testing.assert_allclose(reward, expected_return, rtol=1e-6)
    np.testing.assert_allclose(discount, gamma**3, rtol=1e-6)
    assert done == 0.0


def test_per_nstep_one_step_equivalence() -> None:
    """AC-003: with n_steps=1 the return degenerates to the immediate reward."""

    gamma = 0.97
    buffer = create_per_buffer(n_steps=1, gamma=gamma)
    fill_buffer(buffer, rewards=[5.0, 1.0, 1.0])

    reward, done, discount = sample_from_start(buffer, base_idx=0)

    assert reward == pytest.approx(5.0)
    assert discount == pytest.approx(gamma)
    assert done == 0.0


@pytest.mark.parametrize("position", [1, 2, 3])
def test_per_nstep_true_termination_at_each_position(position: int) -> None:
    """AC-005: termination at window step `position` (1-indexed) truncates the return, done=1."""

    gamma = 0.95
    buffer = create_per_buffer(n_steps=3, gamma=gamma)
    rewards = [1.0, 2.0, 3.0, 1.0, 1.0]
    done_idx = position - 1
    fill_buffer(buffer, rewards=rewards, done_at=done_idx)

    reward, done, _discount = sample_from_start(buffer, base_idx=0)

    expected = compute_expected_nstep_reward(gamma, rewards[:3], stop_idx=done_idx)
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    assert done == 1.0


@pytest.mark.parametrize("position", [1, 2, 3])
def test_per_nstep_truncation_at_each_position(position: int) -> None:
    """AC-006: truncation at window step `position` (1-indexed) truncates the return, done=0."""

    gamma = 0.95
    buffer = create_per_buffer(n_steps=3, gamma=gamma)
    rewards = [1.0, 2.0, 3.0, 1.0, 1.0]
    truncated_idx = position - 1
    fill_buffer(buffer, rewards=rewards, truncated_at=truncated_idx)

    reward, done, discount = sample_from_start(buffer, base_idx=0)

    expected = compute_expected_nstep_reward(gamma, rewards[:3], stop_idx=truncated_idx)
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    assert done == 0.0
    effective_m = position
    np.testing.assert_allclose(discount, gamma**effective_m, rtol=1e-6)


def test_per_nstep_frontier_isolation_on_ring_wraparound() -> None:
    """AC-009: sampling near the write frontier never crosses into stale/unwritten rows."""

    gamma = 0.99
    n_steps = 3
    buffer = create_per_buffer(buffer_size=10, n_steps=n_steps, gamma=gamma)
    fill_buffer(buffer, rewards=[1.0] * 10)
    # Overwrite the first two rows; base_idx=0 now sits right behind the write head.
    fill_buffer(buffer, rewards=[1.0, 1.0])

    reward, done, discount = sample_from_start(buffer, base_idx=0)

    frontier_stop = buffer.pos - 1
    expected = compute_expected_nstep_reward(gamma, [1.0] * n_steps, stop_idx=frontier_stop)
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    assert done == 0.0
    np.testing.assert_allclose(discount, gamma ** (frontier_stop + 1), rtol=1e-6)

    # A real termination on the frontier row must still be honored (done=1),
    # not masked by the temporary anti-crossing truncation.
    buffer.dones[buffer.pos - 1, :] = True
    reward, done, _discount = sample_from_start(buffer, base_idx=0)
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    assert done == 1.0
