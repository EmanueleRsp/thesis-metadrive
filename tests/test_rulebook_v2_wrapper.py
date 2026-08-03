from __future__ import annotations

import json

import gymnasium as gym
import pytest
from shapely.geometry import Polygon

from thesis_rl.agent.agent import Agent
from thesis_rl.reward.scalarization import RulebookScalarizer, ScalarizationConfig
from thesis_rl.rulebook.v2.types import (
    EpisodeCache,
    RulebookMemory,
    RulebookResult,
    TaskRouteRecord,
)
from thesis_rl.rulebook.v2.types import CacheDelta, ConflictZoneRecord, MovementKey
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper


class _Env(gym.Env):
    def __init__(self, *, terminated=False, truncated=False, step_info=None):
        self.t = 0
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._step_info = dict(step_info or {"native": True})

    def reset(self, **kwargs):
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        return self.t, 3.5, self._terminated, self._truncated, dict(self._step_info)


def test_wrapper_preserves_native_reward_and_commits_after_transition():
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    cache = EpisodeCache("s", route)

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        assert kwargs["pre_state"] == 0 and kwargs["post_state"] == 1
        delta = type("Delta", (), {"new_conflict_zones": ()})()
        return (
            RulebookResult((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0, {}, True),
            RulebookMemory(),
            delta,
        )

    wrapped = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=snapshot,
        transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(),
        initial_cache=cache,
    )
    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)
    assert reward == 3.5
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert info["rule_reward_vector"] == (0.0, 0.0, 0.0, 0.1)
    assert info["rule_metadata"]["rule_names"] == [
        "collision_impact",
        "dynamic_interaction_safety",
        "road_traffic_compliance",
        "route_progress",
    ]
    assert info["rule_metadata"]["priorities"] == [0, 1, 2, 3]
    assert Agent._extract_rule_margins(info) == [
        ("collision_impact", 0, 0.0),
        ("dynamic_interaction_safety", 1, 0.0),
        ("road_traffic_compliance", 2, 0.0),
        ("route_progress", 3, 0.1),
    ]


def test_wrapper_exposes_native_termination_and_truncation_flags() -> None:
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    cache = EpisodeCache("s", route)

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        _ = kwargs
        return (
            RulebookResult((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, {}, True),
            RulebookMemory(),
            CacheDelta(),
        )

    for terminated, truncated in ((True, False), (False, True)):
        wrapped = RulebookV2MonitorWrapper(
            _Env(terminated=terminated, truncated=truncated),
            snapshotter=snapshot,
            transition_evaluator=evaluate_transition,
            initial_memory=RulebookMemory(),
            initial_cache=cache,
        )
        wrapped.reset()
        _, _, returned_terminated, returned_truncated, info = wrapped.step(0)
        assert (returned_terminated, returned_truncated) == (terminated, truncated)
        assert (info["terminated"], info["truncated"]) == (terminated, truncated)


def test_wrapper_preserves_physical_road_diagnostics(tmp_path) -> None:
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    cache = EpisodeCache("s", route)
    log_path = tmp_path / "rule_margins.jsonl"

    wrapped = RulebookV2MonitorWrapper(
        _Env(
            step_info={
                "physical_out_of_road": False,
                "geometric_full_footprint_exit": False,
                "geometric_outside_area_m2": 0.0,
                "geometric_ego_area_m2": 8.0,
                "route_lateral": 5.0,
                "dist_to_left_side": 3.0,
                "dist_to_right_side": 3.0,
                "on_lane": True,
                "contact_results": ["ROAD_EDGE_BOUNDARY"],
            }
        ),
        snapshotter=lambda env: env.t,
        transition_evaluator=lambda **_kwargs: (
            RulebookResult((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, {}, True),
            RulebookMemory(),
            CacheDelta(),
        ),
        initial_memory=RulebookMemory(),
        initial_cache=cache,
        rule_margin_log_path=str(log_path),
    )
    wrapped.reset()
    _observation, _reward, _terminated, _truncated, info = wrapped.step(0)

    assert info["route_lateral"] == 5.0
    assert info["dist_to_right_side"] == 3.0
    assert info["contact_results"] == ["ROAD_EDGE_BOUNDARY"]
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["termination"]["geometric_full_footprint_exit"] is False
    assert payload["termination"]["geometric_outside_area_m2"] == 0.0


def test_wrapper_uses_scalarizer_after_complete_rulebook_evaluation(tmp_path):
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    cache = EpisodeCache("s", route)

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        result = RulebookResult((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0, {}, True)
        return result, RulebookMemory(), CacheDelta()

    wrapped = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=snapshot,
        transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(),
        initial_cache=cache,
        scalarizer=RulebookScalarizer(ScalarizationConfig()),
        rule_margin_log_path=str(tmp_path / "rule_margins.jsonl"),
        runtime_info_debug_enabled=True,
        runtime_info_debug_path=str(tmp_path / "runtime_info_debug.jsonl"),
    )
    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)
    assert reward == pytest.approx(0.025)
    assert info["env_reward"] == 3.5
    assert info["scalar_reward"] == pytest.approx(0.025)
    assert info["scalar_rule_reward"] == pytest.approx(0.025)
    assert info["scalarization"]["mode"] == "bounded_satisfaction_rank"
    margin_record = (tmp_path / "rule_margins.jsonl").read_text(encoding="utf-8").strip()
    assert '"scalar_rule_reward": 0.025' in margin_record
    runtime_record = (tmp_path / "runtime_info_debug.jsonl").read_text(encoding="utf-8").strip()
    assert '"rulebook_margins": [0.0, 0.0, 0.0, 0.1]' in runtime_record


def test_wrapper_does_not_commit_memory_or_snapshot_when_cache_commit_fails():
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    existing_zone = ConflictZoneRecord(
        "z",
        Polygon(((2, 0), (3, 0), (3, 1), (2, 1))),
        MovementKey("a", "n", "e"),
        MovementKey("b", "n", "f"),
        2.0,
        3.0,
        0.0,
    )
    cache = EpisodeCache("s", route, conflict_zones={"z": existing_zone})
    initial_memory = RulebookMemory()
    bad_zone = ConflictZoneRecord(
        "z",
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
        MovementKey("a", "n", "e"),
        MovementKey("b", "n", "f"),
        0.0,
        1.0,
        0.0,
    )

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        return (
            RulebookResult((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0, {}, True),
            RulebookMemory(),
            CacheDelta((bad_zone,)),
        )

    wrapped = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=snapshot,
        transition_evaluator=evaluate_transition,
        initial_memory=initial_memory,
        initial_cache=cache,
    )
    wrapped.reset()
    with pytest.raises(ValueError):
        wrapped.step(0)
    assert wrapped.memory == initial_memory
    assert wrapped.cache == cache
    assert wrapped._pre_snapshot == 0


def test_wrapper_instances_keep_memory_and_cache_isolated_per_environment():
    route_one = TaskRouteRecord("s1", ("lane",), "pg", "v2", "hash-1")
    route_two = TaskRouteRecord("s2", ("lane",), "pg", "v2", "hash-2")

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        return (
            RulebookResult((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, {}, True),
            kwargs["memory"],
            CacheDelta(),
        )

    first = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=snapshot,
        transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(),
        initial_cache=EpisodeCache("s1", route_one),
    )
    second = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=snapshot,
        transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(),
        initial_cache=EpisodeCache("s2", route_two),
    )
    first.reset()
    second.reset()
    assert first.memory == second.memory == RulebookMemory()
    assert first.cache.scenario_id == "s1"
    assert second.cache.scenario_id == "s2"
