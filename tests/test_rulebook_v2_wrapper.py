from __future__ import annotations

import gymnasium as gym
import pytest

from thesis_rl.rulebook.v2.types import EpisodeCache, RulebookMemory, RulebookResult, TaskRouteRecord
from thesis_rl.rulebook.v2.types import CacheDelta, ConflictZoneRecord, MovementKey
from shapely.geometry import Polygon
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper


class _Env(gym.Env):
    def __init__(self):
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        return self.t, 3.5, False, False, {"native": True}


def test_wrapper_preserves_native_reward_and_commits_after_transition():
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    cache = EpisodeCache("s", route)

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        assert kwargs["pre_state"] == 0 and kwargs["post_state"] == 1
        return RulebookResult((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0, {}, True), RulebookMemory(), type("Delta", (), {"new_conflict_zones": ()})()

    wrapped = RulebookV2MonitorWrapper(
        _Env(), snapshotter=snapshot, transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(), initial_cache=cache,
    )
    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)
    assert reward == 3.5
    assert info["rule_reward_vector"] == (0.0, 0.0, 0.0, 0.1)


def test_wrapper_does_not_commit_memory_or_snapshot_when_cache_commit_fails():
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    existing_zone = ConflictZoneRecord(
        "z", Polygon(((2, 0), (3, 0), (3, 1), (2, 1))),
        MovementKey("a", "n", "e"), MovementKey("b", "n", "f"), 2.0, 3.0, 0.0,
    )
    cache = EpisodeCache("s", route, conflict_zones={"z": existing_zone})
    initial_memory = RulebookMemory()
    bad_zone = ConflictZoneRecord(
        "z", Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
        MovementKey("a", "n", "e"), MovementKey("b", "n", "f"), 0.0, 1.0, 0.0,
    )

    def snapshot(env):
        return env.t

    def evaluate_transition(**kwargs):
        return RulebookResult((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0, {}, True), RulebookMemory(previous_route_s_m=9.0), CacheDelta((bad_zone,))

    wrapped = RulebookV2MonitorWrapper(
        _Env(), snapshotter=snapshot, transition_evaluator=evaluate_transition,
        initial_memory=initial_memory, initial_cache=cache,
    )
    wrapped.reset()
    with pytest.raises(ValueError):
        wrapped.step(0)
    assert wrapped.memory == initial_memory
    assert wrapped.cache == cache
    assert wrapped._pre_snapshot == 0
