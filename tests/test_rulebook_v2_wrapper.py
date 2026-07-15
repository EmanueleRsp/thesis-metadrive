from __future__ import annotations

import gymnasium as gym

from thesis_rl.rulebook.v2.types import EpisodeCache, RulebookMemory, RulebookResult, TaskRouteRecord
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

