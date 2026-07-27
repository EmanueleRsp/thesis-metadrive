"""PPO rollout buffer that can retroactively truncate one env at a boundary.

A typed runtime scenario data-abort (see ``RuntimeScenarioNotEvaluableError``)
must discard only its failed step and retrospectively close the preceding
valid transition as a bootstrappable truncation boundary, mirroring the
existing off-policy PER buffer's ``close_previous_transition_as_data_abort``
convention and this codebase's existing ``TimeLimit``-truncation reward
bootstrap. The vanilla ``RolloutBuffer`` from Stable-Baselines3 has no notion
of "this cell was never a real transition," so the failed env's row would
either have to be invented (prohibited by ADR-024) or the whole shared
rollout discarded.

This buffer keeps SB3's shared, lockstep write cursor (every ``add()`` call
still advances ``self.pos`` by exactly one for every env, so the outer
training loop's iteration/collection-length assumptions are untouched) and
adds a per-cell validity mask instead. An invalid cell only ever writes its
``episode_starts`` flag (needed so the *previous* row's GAE computation
correctly sees a boundary and stops propagating advantage across it); every
other field is left at its zero-initialized default and is never read,
because the minibatch sampler filters invalid cells out before they can ever
reach a gradient step. No change to ``compute_returns_and_advantage`` itself
is needed: the boundary is already enforced purely through the
``episode_starts`` flag, exactly like an ordinary mid-rollout episode
termination.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer


class MaskedRolloutBuffer(RolloutBuffer):
    """``RolloutBuffer`` variant with a per-cell validity mask for data-aborts."""

    def reset(self) -> None:
        super().reset()
        self.valid_transitions = np.ones((self.buffer_size, self.n_envs), dtype=bool)

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        if valid_mask is None:
            mask = np.ones(self.n_envs, dtype=bool)
        else:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != (self.n_envs,):
                raise ValueError("Rollout valid_mask must have one entry per environment.")

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        episode_start_np = np.array(episode_start)
        # The invalid cell's episode_starts flag is the only thing that must
        # be genuine: it is what makes the *previous* row's GAE computation
        # correctly stop propagating advantage across the abort boundary.
        self.episode_starts[self.pos] = episode_start_np
        self.valid_transitions[self.pos] = mask

        if np.any(mask):
            obs_np = np.array(obs)
            action_np = np.array(action)
            reward_np = np.array(reward)
            value_np = value.clone().cpu().numpy().flatten()
            log_prob_np = log_prob.clone().cpu().numpy()
            self.observations[self.pos, mask] = obs_np[mask]
            self.actions[self.pos, mask] = action_np[mask]
            self.rewards[self.pos, mask] = reward_np[mask]
            self.values[self.pos, mask] = value_np[mask]
            self.log_probs[self.pos, mask] = log_prob_np[mask]

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def close_previous_transition_as_data_abort(
        self, *, env_index: int, terminal_value: float
    ) -> None:
        """Bootstrap the preceding valid row's reward at a data-abort boundary.

        Mirrors the existing ``TimeLimit``-truncation convention (bootstrap
        the stored reward with ``gamma * V(final_observation)``) instead of
        crossing the reset with a synthetic transition. The caller marks the
        *next* transition written for this env as an episode start, exactly
        as for an ordinary truncated episode.

        Unlike the off-policy replay buffer (a circular buffer spanning the
        whole run, where ``pos == 0`` genuinely means "nothing has ever been
        collected"), this on-policy buffer is periodically reset to
        ``pos == 0`` by SB3 at the start of every ``n_steps`` rollout
        collection cycle. An abort on an env's first step of a fresh
        collection cycle is therefore a routine occurrence, not a caller
        bug: the "previous" transition belongs to the just-flushed prior
        rollout, whose boundary reward was already correctly bootstrapped by
        SB3's own standard end-of-rollout truncation handling before this
        buffer was reset. There is nothing left to retroactively fix in
        that case, so this is a no-op rather than an error.
        """

        if not 0 <= int(env_index) < self.n_envs:
            raise IndexError("Rollout environment index is out of range.")
        row = int(self.pos) - 1
        if row < 0:
            return
        if not bool(self.valid_transitions[row, int(env_index)]):
            raise ValueError("Data-abort closure requires a preceding valid transition.")
        self.rewards[row, int(env_index)] += float(self.gamma * terminal_value)

    def get(self, batch_size: int | None = None):
        assert self.full, ""
        flat_valid = self.swap_and_flatten(self.valid_transitions).reshape(-1)
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = indices[flat_valid[indices]]
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = indices.shape[0]

        start_idx = 0
        while start_idx < indices.shape[0]:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
