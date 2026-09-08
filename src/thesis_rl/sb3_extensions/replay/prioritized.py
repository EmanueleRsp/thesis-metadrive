"""Vector-aware proportional prioritized N-step replay for the local SB3 fork."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer


class _SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = 1
        while self.capacity < int(capacity):
            self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def set(self, index: int, value: float) -> None:
        node = int(index) + self.capacity
        delta = float(value) - self.tree[node]
        self.tree[node] = float(value)
        node //= 2
        while node:
            self.tree[node] += delta
            node //= 2

    def set_batch(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Set distinct leaves and propagate their summed deltas by tree level."""

        leaf_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        leaf_values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(leaf_indices) != len(leaf_values):
            raise ValueError("Sum-tree batch indices and values must have matching lengths.")
        if len(leaf_indices) == 0:
            return
        if len(np.unique(leaf_indices)) != len(leaf_indices):
            raise ValueError("Sum-tree batch indices must be distinct.")
        nodes = leaf_indices + self.capacity
        deltas = leaf_values - self.tree[nodes]
        self.tree[nodes] = leaf_values
        nodes //= 2
        while len(nodes) > 0 and np.any(nodes):
            parents, inverse = np.unique(nodes, return_inverse=True)
            parent_deltas = np.zeros(len(parents), dtype=np.float64)
            np.add.at(parent_deltas, inverse, deltas)
            self.tree[parents] += parent_deltas
            nodes = parents // 2
            deltas = parent_deltas

    def find_prefix(self, mass: float) -> int:
        node = 1
        value = float(mass)
        while node < self.capacity:
            left = node * 2
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node - self.capacity

    def find_prefix_batch(self, masses: np.ndarray) -> np.ndarray:
        """Resolve independent prefix masses with the scalar traversal rules.

        Every lane takes the same fixed tree depth, so the node decisions can
        be evaluated as NumPy vectors without changing the float64 comparison,
        subtraction, or tie behavior of :meth:`find_prefix`.
        """

        values = np.asarray(masses, dtype=np.float64).copy()
        nodes = np.ones(values.shape, dtype=np.int64)
        while nodes.size and int(nodes.flat[0]) < self.capacity:
            left_nodes = nodes * 2
            left_values = self.tree[left_nodes]
            choose_left = values <= left_values
            values = np.where(choose_left, values, values - left_values)
            nodes = np.where(choose_left, left_nodes, left_nodes + 1)
        return nodes - self.capacity


class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    """NStepReplayBuffer-compatible buffer with proportional PER sampling."""

    def __init__(
        self,
        *args: Any,
        n_steps: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_final: float = 1.0,
        beta_anneal_steps: int = 1,
        epsilon: float = 1.0e-6,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if bool(kwargs.get("optimize_memory_usage", False)):
            raise ValueError("PrioritizedNStepReplayBuffer requires optimize_memory_usage=false.")
        super().__init__(*args, n_steps=n_steps, gamma=gamma, **kwargs)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta_initial = float(beta_initial)
        self.beta_final = float(beta_final)
        self.beta_anneal_steps = int(beta_anneal_steps)
        self.epsilon = float(epsilon)
        self.beta_progress_env_steps = 0
        self.rng = np.random.default_rng(seed)
        self.raw_priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        # A vector collection tick may have no transition for one slot after a
        # typed runtime data-abort. The physical row remains only to preserve
        # the other slots' chronology; an invalid leaf is never addressable,
        # sampled, prioritized, or interpreted as a replay transition.
        self.valid_transitions = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        # An unseen transition is primed at the exact current raw maximum (or 1.0
        # for an empty buffer); see `_insertion_priority` for why there is no
        # floor above it. Tracking the multiplicity avoids scanning the entire
        # allocation at every vector insertion; `_recompute_current_max` is the
        # fallback for the one case the counter cannot resolve incrementally.
        self._current_max_raw_priority = 1.0
        self._current_max_count = 0
        self._tree = _SumTree(self.buffer_size * self.n_envs)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("PER alpha must be in [0, 1].")
        if not (0.0 <= self.beta_initial <= self.beta_final <= 1.0):
            raise ValueError("PER beta values must satisfy 0 <= beta_initial <= beta_final <= 1.")
        if self.beta_anneal_steps <= 0:
            raise ValueError("PER beta_anneal_steps must be positive.")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("PER epsilon must be finite and positive.")

    def _flat(self, storage_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        return np.asarray(storage_indices, dtype=np.int64) * self.n_envs + np.asarray(
            env_indices, dtype=np.int64
        )

    def _active_count(self) -> int:
        return self.buffer_size if self.full else self.pos

    def _set_raw_priority(self, flat_index: int, raw_priority: float) -> None:
        value = float(raw_priority)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("PER raw priority must be finite and positive.")
        index = int(flat_index)
        previous = float(self.raw_priorities.flat[index])
        self.raw_priorities.flat[index] = value
        self._update_current_max(previous, value)
        self._tree.set(index, value**self.alpha)

    def _set_raw_priorities_batch(
        self, flat_indices: np.ndarray, raw_priorities: np.ndarray
    ) -> None:
        """Apply distinct valid raw priorities with equivalent max bookkeeping."""

        indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        values = np.asarray(raw_priorities, dtype=np.float64).reshape(-1)
        if len(indices) != len(values):
            raise ValueError("PER batch indices and priorities must have matching lengths.")
        if len(indices) == 0:
            return
        if len(np.unique(indices)) != len(indices):
            raise ValueError("PER batch indices must be distinct after reduction.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("PER raw priority must be finite and positive.")

        previous = self.raw_priorities.flat[indices].copy()
        previous_max = self._current_max_raw_priority
        remaining_max_count = self._current_max_count - int(
            np.count_nonzero(previous == previous_max)
        )
        retained_max_count = remaining_max_count + int(np.count_nonzero(values == previous_max))
        self.raw_priorities.flat[indices] = values
        highest_new = float(np.max(values))
        if highest_new > previous_max:
            self._current_max_raw_priority = highest_new
            self._current_max_count = int(np.count_nonzero(values == highest_new))
        elif retained_max_count > 0:
            self._current_max_count = retained_max_count
        else:
            self._recompute_current_max()
        self._tree.set_batch(indices, values**self.alpha)

    def _recompute_current_max(self) -> None:
        """Rescan for the maximum, treating "no positive priority" as an empty buffer.

        A masked data-abort slot writes 0.0, so a buffer cycle in which every
        slot aborted leaves no priced row at all. `_insertion_priority` has to
        prime the next transition at 1.0 there, exactly as it does before the
        first insertion: recording a maximum of 0.0 would instead make
        `_set_raw_priority` reject that insertion as non-positive.
        """

        highest = float(np.max(self.raw_priorities))
        if highest <= 0.0:
            self._current_max_raw_priority = 1.0
            self._current_max_count = 0
            return
        self._current_max_raw_priority = highest
        self._current_max_count = int(np.count_nonzero(self.raw_priorities == highest))

    def _update_current_max(self, previous: float, value: float) -> None:
        if value > self._current_max_raw_priority:
            self._current_max_raw_priority = value
            self._current_max_count = 1
            return
        if value == self._current_max_raw_priority:
            if previous != self._current_max_raw_priority:
                self._current_max_count += 1
            return
        if previous != self._current_max_raw_priority:
            return
        self._current_max_count -= 1
        if self._current_max_count > 0:
            return
        self._recompute_current_max()

    def _insertion_priority(self) -> float:
        """The exact current maximum, with 1.0 standing in for an empty buffer.

        Deliberately **not** `max(1.0, current)`. Raw priorities are absolute
        `|TD error| + epsilon` in reward units, and the sampler is otherwise
        exactly equivariant to a rescaling of them: leaves hold `p ** alpha`,
        each address's probability is its leaf over the total, and the
        importance weights are max-normalised within the batch, so multiplying
        every priority by a constant changes no address and no weight. A
        constant floor would be the one reward-scale-dependent term in it, and
        `conf/scalarization/default.yaml` is a file this project recalibrates.

        A floor also buys nothing it is supposed to buy. The share of draws
        reaching rows that have never been priced is pinned by arrivals over
        draws — `n_envs / (gradient_steps * batch_size)` — whatever value they
        are primed at, so the priming level sets only the latency to a first
        sample, and that latency is orders of magnitude inside a row's
        residence. Where a floor does bind it destroys the machinery it sits
        in: `_current_max_raw_priority` collapses onto the injected constant
        and stops measuring the critic.
        """

        return self._current_max_raw_priority if self._current_max_count > 0 else 1.0

    def add(self, *args: Any, valid_mask: np.ndarray | None = None, **kwargs: Any) -> None:
        storage_index = int(self.pos)
        super().add(*args, **kwargs)
        if valid_mask is None:
            mask = np.ones(self.n_envs, dtype=bool)
        else:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != (self.n_envs,):
                raise ValueError("Replay valid_mask must have one entry per environment.")
        self.valid_transitions[storage_index] = mask
        maximum = self._insertion_priority()
        for env_index in range(self.n_envs):
            flat_index = storage_index * self.n_envs + env_index
            if mask[env_index]:
                self._set_raw_priority(flat_index, maximum)
            else:
                # The overwritten row may have been the one holding the running
                # maximum. Release it through the same bookkeeping the valid
                # branch uses: skipping it left `_current_max_count` counting a
                # row that no longer exists, and that error never self-corrects,
                # because the decrement fires only on a value equal to the
                # recorded maximum.
                previous = float(self.raw_priorities.flat[flat_index])
                self.raw_priorities.flat[flat_index] = 0.0
                self._update_current_max(previous, 0.0)
                self._tree.set(flat_index, 0.0)

    def close_previous_transition_as_data_abort(
        self, *, env_index: int, final_observation: np.ndarray
    ) -> None:
        """Retroactively make the preceding valid row a bootstrappable boundary."""

        if not 0 <= int(env_index) < self.n_envs:
            raise IndexError("Replay environment index is out of range.")
        if self.pos == 0 and not self.full:
            raise ValueError("No previous replay transition exists for data-abort closure.")
        row = (int(self.pos) - 1) % self.buffer_size
        if not bool(self.valid_transitions[row, int(env_index)]):
            raise ValueError("Data-abort closure requires a preceding valid transition.")
        self.dones[row, int(env_index)] = 0.0
        self.timeouts[row, int(env_index)] = 1.0
        self.next_observations[row, int(env_index)] = np.asarray(
            final_observation, dtype=self.next_observations.dtype
        )

    def _sample_addresses(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_count = self._active_count() * self.n_envs
        if active_count <= 0:
            raise ValueError("Cannot sample from an empty prioritized replay buffer.")
        total = self._tree.total
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("PER total priority mass must be finite and positive.")
        bounds = np.linspace(0.0, total, int(batch_size) + 1)
        masses = self.rng.uniform(bounds[:-1], bounds[1:])
        flat_indices = self._tree.find_prefix_batch(masses)
        flat_indices %= active_count
        probabilities = self._tree.tree[flat_indices + self._tree.capacity] / total
        # A prefix search can land on a zero-priority leaf (a masked data-abort
        # slot, or float drift in the incrementally maintained sums), and the
        # active-range wrap above can move an index onto one. Such a leaf has
        # probability 0, so its importance weight is `inf`, the max-normalisation
        # turns the batch into `NaN`/0 and the critic parameters become NaN with
        # no exception. Re-sample those addresses from the positive mass instead
        # (audit 2026-09-06, A7).
        invalid = ~(probabilities > 0.0)
        attempts = 0
        while np.any(invalid):
            attempts += 1
            if attempts > 16:
                raise ValueError(
                    "PER sampling repeatedly landed on zero-priority leaves; "
                    "the sum tree is inconsistent with its leaves."
                )
            redraw = self.rng.uniform(0.0, total, size=int(np.count_nonzero(invalid)))
            redrawn = self._tree.find_prefix_batch(redraw) % active_count
            flat_indices[invalid] = redrawn
            probabilities = self._tree.tree[flat_indices + self._tree.capacity] / total
            invalid = ~(probabilities > 0.0)
        storage_indices = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        beta = self.current_beta()
        weights = (active_count * probabilities) ** (-beta)
        weights /= max(float(np.max(weights)), 1.0e-12)
        if not np.all(np.isfinite(weights)):
            raise ValueError("PER importance-sampling weights must be finite.")
        return storage_indices, env_indices, weights.astype(np.float32)

    def current_beta(self) -> float:
        progress = min(max(self.beta_progress_env_steps, 0) / self.beta_anneal_steps, 1.0)
        return float(self.beta_initial + (self.beta_final - self.beta_initial) * progress)

    def set_beta_progress(self, env_steps: int) -> None:
        self.beta_progress_env_steps = max(int(env_steps), 0)

    def __getstate__(self) -> dict[str, Any]:
        """Persist only written replay rows until the ring is full.

        SB3 allocates every replay row at construction.  Pickling that full
        allocation for a short smoke/resume checkpoint is pure I/O overhead:
        zero-filled, not-yet-addressable rows cannot influence sampling.  The
        active prefix and all scalar/RNG state are sufficient to recreate the
        exact buffer and priority tree on load.
        """

        state = dict(self.__dict__)
        active_count = self._active_count()
        state["_thesis_sparse_replay_version"] = 1
        state["_thesis_sparse_replay_capacity"] = int(self.buffer_size)
        state["_thesis_sparse_replay_active_count"] = int(active_count)
        if not self.full:
            for name in (
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "timeouts",
                "raw_priorities",
                "valid_transitions",
            ):
                values = np.asarray(state[name])
                state[name] = values[:active_count].copy()
        # It is deterministically derived from raw priorities and should not
        # duplicate a multi-megabyte allocation in every checkpoint.
        state.pop("_tree", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an exact sparse replay checkpoint produced by ``__getstate__``."""

        version = state.pop("_thesis_sparse_replay_version", None)
        if version is None:
            self.__dict__.update(state)
            if not hasattr(self, "valid_transitions"):
                self.valid_transitions = np.asarray(self.raw_priorities > 0.0, dtype=bool)
            return
        if version != 1:
            raise ValueError(f"Unsupported sparse replay persistence version: {version}")
        capacity = int(state.pop("_thesis_sparse_replay_capacity"))
        active_count = int(state.pop("_thesis_sparse_replay_active_count"))
        if active_count < 0 or active_count > capacity:
            raise ValueError("Sparse replay checkpoint has an invalid active row count.")
        self.__dict__.update(state)
        if not hasattr(self, "valid_transitions"):
            self.valid_transitions = np.asarray(self.raw_priorities > 0.0, dtype=bool)
        if not bool(self.full):
            for name in (
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "timeouts",
                "raw_priorities",
                "valid_transitions",
            ):
                persisted = np.asarray(getattr(self, name))
                restored = np.zeros((capacity, *persisted.shape[1:]), dtype=persisted.dtype)
                restored[:active_count] = persisted
                setattr(self, name, restored)
        self.buffer_size = capacity
        self._tree = _SumTree(self.buffer_size * self.n_envs)
        active_priorities = self.raw_priorities[: self._active_count()].reshape(-1)
        for index, priority in enumerate(active_priorities):
            if float(priority) > 0.0:
                self._tree.set(index, float(priority) ** self.alpha)

    def sample(self, batch_size: int, env: Any = None):
        from stable_baselines3.common.type_aliases import ReplayBufferSamples

        storage_indices, env_indices, weights = self._sample_addresses(batch_size)
        samples = self._get_samples_with_env_indices(storage_indices, env_indices, env)
        return ReplayBufferSamples(
            *samples,
            weights=self.to_torch(weights.reshape(-1, 1)),
            indices=self.to_torch(self._flat(storage_indices, env_indices)),
        )

    def _get_samples_with_env_indices(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Any
    ):
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(
            original_timeout_values, np.logical_not(self.dones[last_valid_index])
        )
        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)
        dones_seq = self.dones[indices, env_indices[:, None]]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        done_idx = np.where(done_or_truncated.any(axis=1), done_idx, self.n_steps - 1)
        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]
        target_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)
        n_step_returns = (rewards_seq * discounts * mask).sum(axis=1, keepdims=True)
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        final_dones = self.dones[last_indices, env_indices][:, None] * (
            1.0 - self.timeouts[last_indices, env_indices][:, None]
        )
        self.timeouts[last_valid_index] = original_timeout_values
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        actions = self.actions[batch_inds, env_indices]
        return (
            self.to_torch(obs),
            self.to_torch(actions),
            self.to_torch(next_obs),
            self.to_torch(final_dones),
            self.to_torch(n_step_returns),
            self.to_torch(target_discounts),
        )

    def update_priorities(self, indices: Any, priorities: Any) -> None:
        flat_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        raw_priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if len(flat_indices) != len(raw_priorities):
            raise ValueError("PER priority indices and values must have matching lengths.")
        values = raw_priorities + self.epsilon
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("PER updated priority must be finite and positive.")
        unique_indices, inverse = np.unique(flat_indices, return_inverse=True)
        reduced_values = np.zeros(len(unique_indices), dtype=np.float64)
        np.maximum.at(reduced_values, inverse, values)
        self._set_raw_priorities_batch(unique_indices, reduced_values)
