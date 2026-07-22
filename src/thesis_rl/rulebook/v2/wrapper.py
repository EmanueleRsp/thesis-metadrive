"""Gymnasium monitor-only integration for Rulebook v2.

The wrapper deliberately receives snapshot and transition functions from the
environment adapter.  This keeps MetaDrive/ScenarioNet extraction out of the
normative core and makes the transactional boundary testable in isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.reward.scalarization import RulebookScalarizer, ScalarizationResult
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
AdapterFactory = Callable[[], "RulebookV2Adapter"]


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
    """Attach v2 diagnostics and optionally replace the native Gym reward."""

    def __init__(
        self,
        env: gym.Env,
        *,
        snapshotter: Snapshotter,
        transition_evaluator: TransitionEvaluator,
        initial_memory: RulebookMemory | None,
        initial_cache: EpisodeCache | None,
        scalarizer: RulebookScalarizer | None = None,
        adapter_factory: AdapterFactory | None = None,
        rule_margin_log_path: str | None = None,
        runtime_info_debug_enabled: bool = False,
        runtime_info_debug_path: str | None = None,
    ) -> None:
        super().__init__(env)
        self._snapshotter = snapshotter
        self._transition_evaluator = transition_evaluator
        self._initial_memory = initial_memory
        self._initial_cache = initial_cache
        self._scalarizer = scalarizer
        self._adapter_factory = adapter_factory
        self._rule_margin_log_path = Path(rule_margin_log_path) if rule_margin_log_path else None
        if self._rule_margin_log_path is not None:
            self._rule_margin_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_info_debug_enabled = bool(runtime_info_debug_enabled)
        self._runtime_info_debug_path = (
            Path(runtime_info_debug_path)
            if runtime_info_debug_path
            else Path("runtime_info_debug.jsonl")
        )
        if self._runtime_info_debug_enabled:
            self._runtime_info_debug_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = initial_memory
        self._cache = initial_cache
        self._pre_snapshot: EnvSnapshot | None = None
        self._causal_scene_context: CausalSceneContext | None = None

    @property
    def memory(self) -> RulebookMemory:
        if self._memory is None:
            raise RuntimeError("Rulebook v2 memory is unavailable before reset")
        return self._memory

    @property
    def cache(self) -> EpisodeCache:
        if self._cache is None:
            raise RuntimeError("Rulebook v2 cache is unavailable before reset")
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
        if self._adapter_factory is not None:
            adapter = self._adapter_factory()
            if not isinstance(adapter, RulebookV2Adapter):
                raise TypeError("Rulebook v2 adapter factory must return RulebookV2Adapter")
            self._snapshotter = adapter.snapshotter
            self._transition_evaluator = adapter.transition_evaluator
            self._initial_memory = adapter.initial_memory
            self._initial_cache = adapter.initial_cache
        self._memory = self._initial_memory
        self._cache = self._initial_cache
        self._pre_snapshot = self._snapshotter(self.env)
        self._publish_causal_context(self._pre_snapshot)
        return observation, info

    def step(self, action: Any):
        if self._pre_snapshot is None:
            raise RuntimeError("RulebookV2MonitorWrapper.step called before reset")
        pre_snapshot = self._pre_snapshot
        observation, reward, terminated, truncated, info = self.env.step(action)
        phase_started = time.perf_counter()
        post_snapshot = self._snapshotter(self.env)
        snapshot_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        result, next_memory, cache_delta = self._transition_evaluator(
            pre_state=pre_snapshot,
            post_state=post_snapshot,
            memory=self._memory,
            cache=self._cache,
        )
        next_cache = apply_cache_delta(self._cache, cache_delta)
        evaluator_seconds = time.perf_counter() - phase_started
        scalarization_result: ScalarizationResult | None = None
        phase_started = time.perf_counter()
        if self._scalarizer is not None:
            if not result.complete_evaluation:
                raise RuntimeError("Rulebook v2 scalarization requires complete_evaluation=True.")
            scalarization_result = self._scalarizer(result.margins)
        scalarization_seconds = time.perf_counter() - phase_started
        # Commit only after evaluator and cache validation complete.
        self._memory = next_memory
        self._cache = next_cache
        self._pre_snapshot = post_snapshot
        self._publish_causal_context(post_snapshot)
        refresh = getattr(self.env.unwrapped, "_refresh_causal_observation", None)
        phase_started = time.perf_counter()
        if callable(refresh):
            observation = refresh(observation)
        observation_refresh_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        info_dict["terminated"] = bool(terminated)
        info_dict["truncated"] = bool(truncated)
        native_reward = float(reward)
        info_dict["env_reward"] = native_reward
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
        if scalarization_result is not None:
            reward = scalarization_result.reward
            info_dict["scalar_reward"] = scalarization_result.reward
            # The runtime logging/metric contract uses this explicit name to
            # distinguish the scalarized Rulebook reward from the native one.
            info_dict["scalar_rule_reward"] = scalarization_result.reward
            info_dict["selected_reward"] = scalarization_result.reward
            info_dict["scalarization"] = scalarization_result.to_dict()
        self._append_diagnostics(
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            native_reward=native_reward,
            info_dict=info_dict,
            result=result,
            scalarization_result=scalarization_result,
        )
        info_seconds = time.perf_counter() - phase_started
        info_dict["_thesis_rulebook_timing_seconds"] = {
            "snapshot": snapshot_seconds,
            "evaluator": evaluator_seconds,
            "scalarization": scalarization_seconds,
            "observation_refresh": observation_refresh_seconds,
            "info_and_diagnostics": info_seconds,
            **{
                f"transition_{name}": float(value)
                for name, value in getattr(cache_delta, "diagnostic_timing_seconds", {}).items()
            },
        }
        return observation, reward, terminated, truncated, info_dict

    def _append_diagnostics(
        self,
        *,
        pre_snapshot: object,
        post_snapshot: object,
        native_reward: float,
        info_dict: Mapping[str, Any],
        result: RulebookResult,
        scalarization_result: ScalarizationResult | None,
    ) -> None:
        """Persist one JSON-safe Rulebook/scalarization record per transition."""

        scenario_id = getattr(post_snapshot, "scenario_id", None)
        step_index = getattr(post_snapshot, "step_index", None)
        sim_time_s = getattr(post_snapshot, "sim_time_s", None)
        payload: dict[str, Any] = {
            "scenario_id": scenario_id,
            "step": step_index,
            "sim_time_s": sim_time_s,
            "pre_step": getattr(pre_snapshot, "step_index", None),
            "env_reward": native_reward,
            "termination": {
                key: info_dict[key]
                for key in (
                    "crash",
                    "crash_vehicle",
                    "crash_object",
                    "crash_human",
                    "crash_sidewalk",
                    "collision",
                    "out_of_road",
                    "physical_out_of_road",
                    "crossed_continuous_line",
                    "termination_reason",
                    "route_lateral",
                    "dist_to_left_side",
                    "dist_to_right_side",
                    "on_lane",
                    "contact_results",
                    "geometric_full_footprint_exit",
                    "geometric_outside_area_m2",
                    "geometric_ego_area_m2",
                    "terminated",
                    "truncated",
                )
                if key in info_dict
            },
            "rulebook": result.to_dict(),
        }
        if scalarization_result is not None:
            payload["scalar_rule_reward"] = scalarization_result.reward
            payload["scalarization"] = scalarization_result.to_dict()

        if self._rule_margin_log_path is not None:
            with self._rule_margin_log_path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True)
                handle.write("\n")

        if self._runtime_info_debug_enabled:
            runtime_payload = {
                "scenario_id": scenario_id,
                "step": step_index,
                "sim_time_s": sim_time_s,
                "ego_actor_id": getattr(getattr(post_snapshot, "ego", None), "actor_id", None),
                "actor_count": len(getattr(post_snapshot, "actors", ())),
                "contact_onset_count": len(getattr(post_snapshot, "contact_onset_records", ())),
                "active_contact_count": len(getattr(post_snapshot, "active_contact_ids", ())),
                "rulebook_margins": list(result.margins),
                "rulebook_costs": list(result.costs),
                "raw_progress_m": result.raw_progress_m,
                "scalar_rule_reward": (
                    None if scalarization_result is None else scalarization_result.reward
                ),
            }
            with self._runtime_info_debug_path.open("a", encoding="utf-8") as handle:
                json.dump(runtime_payload, handle, ensure_ascii=True)
                handle.write("\n")
