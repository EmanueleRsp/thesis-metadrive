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
        # The specification assigns an unseen transition the exact current raw
        # maximum (or 1.0 for an empty buffer).  Tracking its multiplicity avoids
        # scanning the entire allocation at every vector insertion.
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
        self._current_max_raw_priority = float(np.max(self.raw_priorities))
        self._current_max_count = int(
            np.count_nonzero(self.raw_priorities == self._current_max_raw_priority)
        )

    def _insertion_priority(self) -> float:
        return self._current_max_raw_priority if self._current_max_count > 0 else 1.0

    def add(self, *args: Any, **kwargs: Any) -> None:
        storage_index = int(self.pos)
        super().add(*args, **kwargs)
        maximum = self._insertion_priority()
        for env_index in range(self.n_envs):
            self._set_raw_priority(storage_index * self.n_envs + env_index, maximum)

    def _sample_addresses(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_count = self._active_count() * self.n_envs
        if active_count <= 0:
            raise ValueError("Cannot sample from an empty prioritized replay buffer.")
        total = self._tree.total
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("PER total priority mass must be finite and positive.")
        bounds = np.linspace(0.0, total, int(batch_size) + 1)
        masses = self.rng.uniform(bounds[:-1], bounds[1:])
        flat_indices = np.asarray([self._tree.find_prefix(mass) for mass in masses], dtype=np.int64)
        flat_indices %= active_count
        storage_indices = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        probabilities = np.asarray(
            [self._tree.tree[int(index) + self._tree.capacity] / total for index in flat_indices],
            dtype=np.float64,
        )
        beta = self.current_beta()
        weights = (active_count * probabilities) ** (-beta)
        weights /= max(float(np.max(weights)), 1.0e-12)
        return storage_indices, env_indices, weights.astype(np.float32)

    def current_beta(self) -> float:
        progress = min(max(self.beta_progress_env_steps, 0) / self.beta_anneal_steps, 1.0)
        return float(self.beta_initial + (self.beta_final - self.beta_initial) * progress)

    def set_beta_progress(self, env_steps: int) -> None:
        self.beta_progress_env_steps = max(int(env_steps), 0)

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
        reduced: dict[int, float] = {}
        for index, priority in zip(flat_indices, raw_priorities, strict=True):
            value = float(priority) + self.epsilon
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("PER updated priority must be finite and positive.")
            reduced[int(index)] = max(reduced.get(int(index), 0.0), value)
        for index, priority in reduced.items():
            self._set_raw_priority(index, priority)
