"""Gymnasium monitor-only integration for Rulebook v2.

The wrapper deliberately receives snapshot and transition functions from the
environment adapter.  This keeps MetaDrive/ScenarioNet extraction out of the
normative core and makes the transactional boundary testable in isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.rulebook.v2.memory import apply_cache_delta
from thesis_rl.rulebook.v2.types import (
    MACRO_RULE_ORDER,
    EpisodeCache,
    EnvSnapshot,
    RulebookMemory,
    RulebookResult,
)


Snapshotter = Callable[[Any], EnvSnapshot]
TransitionEvaluator = Callable[..., tuple[RulebookResult, RulebookMemory, Any]]


@dataclass(frozen=True, slots=True)
class RulebookV2Adapter:
    """Environment-owned live hooks required by the v2 wrapper.

    Adapters must construct canonical snapshots from the live simulator and
    must not consult future tracks or ``current_sdc_route`` at runtime.
    """

    snapshotter: Snapshotter
    transition_evaluator: TransitionEvaluator
    initial_memory: RulebookMemory
    initial_cache: EpisodeCache


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
        self._causal_scene_context: CausalSceneContext | None = None

    @property
    def memory(self) -> RulebookMemory:
        return self._memory

    @property
    def cache(self) -> EpisodeCache:
        return self._cache

    @property
    def causal_scene_context(self) -> CausalSceneContext:
        """Committed observation-safe state for the current control step.

        It never includes the just-evaluated ``RulebookResult`` and is updated
        only after the memory/cache transaction succeeds.
        """

        if self._causal_scene_context is None:
            raise RuntimeError("Causal scene context is unavailable before reset")
        return self._causal_scene_context

    def _publish_causal_context(self, snapshot: object) -> None:
        """Publish only canonical snapshots to the owning environment/engine."""

        if not isinstance(snapshot, EnvSnapshot):
            self._causal_scene_context = None
            return
        context = CausalSceneContext(
            episode_cache=self._cache,
            snapshot=snapshot,
            memory=self._memory,
        )
        self._causal_scene_context = context
        # ``BaseObservation`` has access to MetaDrive's engine, while the
        # environment owns the episode lifecycle. Both references point to the
        # same immutable committed object and neither carries RulebookResult.
        owner = self.env.unwrapped
        setattr(owner, "causal_scene_context", context)
        engine = getattr(owner, "engine", None)
        if engine is not None:
            setattr(engine, "causal_scene_context", context)
        on_commit = getattr(owner, "_on_causal_context_committed", None)
        if callable(on_commit):
            on_commit(context)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._memory = self._initial_memory
        self._cache = self._initial_cache
        self._pre_snapshot = self._snapshotter(self.env)
        self._publish_causal_context(self._pre_snapshot)
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
        self._publish_causal_context(post_snapshot)
        refresh = getattr(self.env.unwrapped, "_refresh_causal_observation", None)
        if callable(refresh):
            observation = refresh(observation)
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        info_dict["rule_reward_vector"] = result.margins
        macro_names = [rule.value for rule in MACRO_RULE_ORDER]
        info_dict["rule_metadata"] = {
            "version": "v2",
            "rule_names": macro_names,
            "priorities": list(range(len(macro_names))),
            "saturation_ratio_by_rule": {
                name: cost for name, cost in zip(macro_names, result.costs)
            },
        }
        info_dict["rule_components"] = {
            name: component.to_dict() for name, component in result.components.items()
        }
        info_dict["rulebook"] = result.to_dict()
        return observation, reward, terminated, truncated, info_dict
