"""Gymnasium monitor-only integration for Rulebook v2.

The wrapper deliberately receives snapshot and transition functions from the
environment adapter.  This keeps MetaDrive/ScenarioNet extraction out of the
normative core and makes the transactional boundary testable in isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym

from thesis_rl.rulebook.v2.memory import apply_cache_delta
from thesis_rl.rulebook.v2.types import EpisodeCache, EnvSnapshot, RulebookMemory, RulebookResult


Snapshotter = Callable[[Any], EnvSnapshot]
TransitionEvaluator = Callable[..., tuple[RulebookResult, RulebookMemory, Any]]


class RulebookV2MonitorWrapper(gym.Wrapper):
    """Attach the v2 rule vector while preserving the native Gym reward."""

    def __init__(
        self,
        env: gym.Env,
        *,
        snapshotter: Snapshotter,
        transition_evaluator: TransitionEvaluator,
        initial_memory: RulebookMemory,
        initial_cache: EpisodeCache,
    ) -> None:
        super().__init__(env)
        self._snapshotter = snapshotter
        self._transition_evaluator = transition_evaluator
        self._initial_memory = initial_memory
        self._initial_cache = initial_cache
        self._memory = initial_memory
        self._cache = initial_cache
        self._pre_snapshot: EnvSnapshot | None = None

    @property
    def memory(self) -> RulebookMemory:
        return self._memory

    @property
    def cache(self) -> EpisodeCache:
        return self._cache

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._memory = self._initial_memory
        self._cache = self._initial_cache
        self._pre_snapshot = self._snapshotter(self.env)
        return observation, info

    def step(self, action: Any):
        if self._pre_snapshot is None:
            raise RuntimeError("RulebookV2MonitorWrapper.step called before reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        post_snapshot = self._snapshotter(self.env)
        result, next_memory, cache_delta = self._transition_evaluator(
            pre_state=self._pre_snapshot,
            post_state=post_snapshot,
            memory=self._memory,
            cache=self._cache,
        )
        next_cache = apply_cache_delta(self._cache, cache_delta)
        # Commit only after evaluator and cache validation complete.
        self._memory = next_memory
        self._cache = next_cache
        self._pre_snapshot = post_snapshot
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        info_dict["rule_reward_vector"] = result.margins
        info_dict["rule_components"] = {
            name: component.to_dict() for name, component in result.components.items()
        }
        info_dict["rulebook"] = result.to_dict()
        return observation, reward, terminated, truncated, info_dict

