from __future__ import annotations

import pickle

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


def test_per_batched_prefix_sampling_matches_scalar_traversal_at_boundaries() -> None:
    tree = _SumTree(7)
    priorities = np.array([0.25, 1.75, 0.0, 3.5, 2.0, 0.5, 4.0], dtype=np.float64)
    for index, priority in enumerate(priorities):
        tree.set(index, float(priority))
    masses = np.array(
        [0.0, 0.25, 0.250000000001, 1.0, 2.0, 5.5, tree.total - 1.0e-12],
        dtype=np.float64,
    )

    expected = np.asarray([tree.find_prefix(float(mass)) for mass in masses], dtype=np.int64)

    assert np.array_equal(tree.find_prefix_batch(masses), expected)


def test_per_batched_priority_update_matches_scalar_reduced_updates() -> None:
    buffer = PrioritizedNStepReplayBuffer(
        buffer_size=8,
        observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        n_steps=1,
        gamma=0.99,
        beta_anneal_steps=10,
    )
    reference = PrioritizedNStepReplayBuffer(
        buffer_size=8,
        observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        n_steps=1,
        gamma=0.99,
        beta_anneal_steps=10,
    )
    observation = np.zeros((1, 1), dtype=np.float32)
    action = np.zeros((1, 1), dtype=np.float32)
    for _ in range(5):
        for replay in (buffer, reference):
            replay.add(observation, observation, action, np.zeros(1), np.zeros(1, dtype=bool), [{}])

    indices = np.array([4, 1, 4, 2, 1, 0], dtype=np.int64)
    priorities = np.array([0.5, 2.0, 3.0, 1.5, 4.0, 0.75], dtype=np.float64)
    buffer.update_priorities(indices, priorities)
    reduced: dict[int, float] = {}
    for index, priority in zip(indices, priorities, strict=True):
        reduced[int(index)] = max(reduced.get(int(index), 0.0), float(priority) + reference.epsilon)
    for index, priority in reduced.items():
        reference._set_raw_priority(index, priority)

    assert np.array_equal(buffer.raw_priorities, reference.raw_priorities)
    assert np.array_equal(buffer._tree.tree, reference._tree.tree)
    assert buffer._current_max_raw_priority == reference._current_max_raw_priority
    assert buffer._current_max_count == reference._current_max_count


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


def test_per_insertion_priority_never_falls_below_the_specified_floor() -> None:
    """`REQ-013`: the priming priority is `max(1, p_max_current)`, not `p_max_current`.

    `AC-016` repeats it and traces it to Schaul et al.'s maximum insertion
    priority, whose whole point -- REQ-013's own rationale -- is that a new
    transition can be sampled *before* a TD error has been computed for it. Raw
    priorities are `|TD error| + epsilon`, so a run whose residuals sit below 1
    has a buffer maximum below 1, and priming at that maximum puts unseen
    transitions level with seen ones instead of above them.
    """

    buffer = PrioritizedNStepReplayBuffer(
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

    buffer.add(observation, observation, action, reward, done, infos)
    buffer.add(observation, observation, action, reward, done, infos)
    # Every row now holds a TD-error magnitude below the floor, which is the
    # ordinary case for a critic loss under 1.
    buffer.update_priorities(np.array([0, 1, 2, 3]), np.array([0.4, 0.4, 0.4, 0.4]))

    assert float(np.max(buffer.raw_priorities)) < 1.0
    assert buffer._insertion_priority() == pytest.approx(1.0)

    buffer.add(observation, observation, action, reward, done, infos)
    assert np.all(buffer.raw_priorities[0] == pytest.approx(1.0))


def test_per_masked_overwrite_releases_the_maximum_it_evicts() -> None:
    """A masked data-abort slot evicts a priced row, and the maximum must follow.

    The masked branch of `add` writes the leaf directly, so it used to skip the
    max bookkeeping entirely. `_current_max_count` then over-counted, and the
    error is **permanently sticky**: the decrement in `_update_current_max` fires
    only when the overwritten value equals the recorded maximum, which can never
    happen again once no row holds it. Every unseen transition is primed at a
    maximum the buffer does not contain, over-weighting it in sampling for the
    rest of the run.
    """

    buffer = PrioritizedNStepReplayBuffer(
        # SB3 divides the supplied capacity by n_envs: two storage rows, so the
        # third insertion wraps onto the row holding the maximum.
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
    aborted = np.asarray([False, False])

    buffer.add(observation, observation, action, reward, done, infos)
    buffer.add(observation, observation, action, reward, done, infos)
    # Row 0 becomes the sole holder of the maximum; row 1 stays at the 1.0 both
    # rows were primed with.
    buffer.update_priorities(np.array([0, 1]), np.array([9.0, 9.0]))
    assert buffer._insertion_priority() == pytest.approx(9.0)

    buffer.add(observation, observation, action, reward, done, infos, valid_mask=aborted)

    assert np.max(buffer.raw_priorities) == pytest.approx(1.0)
    assert buffer._insertion_priority() == pytest.approx(float(np.max(buffer.raw_priorities)))

    # A whole buffer cycle of aborts leaves no priced row at all. The maximum has
    # to fall back to the empty-buffer value, because priming an insertion at
    # 0.0 would make `_set_raw_priority` reject it outright.
    buffer.add(observation, observation, action, reward, done, infos, valid_mask=aborted)
    assert np.max(buffer.raw_priorities) == pytest.approx(0.0)
    assert buffer._insertion_priority() == pytest.approx(1.0)

    buffer.add(observation, observation, action, reward, done, infos)
    assert np.max(buffer.raw_priorities) == pytest.approx(1.0)


def test_per_data_abort_leaf_is_non_addressable_and_closes_previous_transition() -> None:
    buffer = PrioritizedNStepReplayBuffer(
        buffer_size=8,
        observation_space=spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=2,
        n_steps=3,
        gamma=0.9,
        beta_anneal_steps=10,
    )
    obs = np.asarray([[1.0], [2.0]], dtype=np.float32)
    action = np.zeros((2, 1), dtype=np.float32)
    buffer.add(obs, obs + 1.0, action, np.ones(2), np.zeros(2, dtype=bool), [{}, {}])
    buffer.close_previous_transition_as_data_abort(
        env_index=1, final_observation=np.asarray([9.0], dtype=np.float32)
    )
    buffer.add(
        obs + 2.0,
        obs + 3.0,
        action,
        np.asarray([2.0, 0.0], dtype=np.float32),
        np.zeros(2, dtype=bool),
        [{}, {}],
        valid_mask=np.asarray([True, False]),
    )

    assert bool(buffer.timeouts[0, 1])
    np.testing.assert_array_equal(buffer.next_observations[0, 1], np.asarray([9.0]))
    assert not bool(buffer.valid_transitions[1, 1])
    assert buffer.raw_priorities[1, 1] == 0.0
    assert buffer._tree.tree[buffer._tree.capacity + 3] == 0.0


def test_per_sparse_persistence_restores_active_rows_without_serializing_capacity() -> None:
    buffer = PrioritizedNStepReplayBuffer(
        buffer_size=1_000,
        observation_space=spaces.Box(-1.0, 1.0, shape=(128,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=2,
        n_steps=1,
        gamma=0.99,
        beta_anneal_steps=10,
        seed=7,
    )
    observation = np.arange(256, dtype=np.float32).reshape(2, 128)
    action = np.zeros((2, 2), dtype=np.float32)
    for reward in (1.0, 2.0, 3.0):
        buffer.add(
            observation,
            observation + reward,
            action,
            np.full(2, reward, dtype=np.float32),
            np.zeros(2, dtype=bool),
            [{}, {}],
        )
    buffer.update_priorities(np.array([0, 1, 2]), np.array([2.0, 3.0, 4.0]))

    payload = pickle.dumps(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(payload)

    assert len(payload) < 20_000
    assert restored.buffer_size == buffer.buffer_size
    assert restored.pos == buffer.pos
    assert restored.full is buffer.full
    assert np.array_equal(restored.observations, buffer.observations)
    assert np.array_equal(restored.next_observations, buffer.next_observations)
    assert np.array_equal(restored.raw_priorities, buffer.raw_priorities)
    assert restored._tree.tree == pytest.approx(buffer._tree.tree)
    assert restored.sample(4).indices.cpu().numpy() == pytest.approx(
        buffer.sample(4).indices.cpu().numpy()
    )
