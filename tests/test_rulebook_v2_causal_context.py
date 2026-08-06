from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import numpy as np
from shapely.geometry import box

from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    EnvSnapshot,
    EpisodeCache,
    RulebookMemory,
    RulebookResult,
    TaskRouteRecord,
)
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper

_MISSION_ROUTE = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))


def test_causal_context_import_does_not_trigger_rulebook_wrapper_cycle() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from thesis_rl.contracts.causal_scene_context import CausalSceneContext; "
            "assert CausalSceneContext.__name__ == 'CausalSceneContext'",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


class _Env(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.step_index = 0
        self.committed_steps: list[int] = []

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.step_index = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_index += 1
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def _on_causal_context_committed(self, context) -> None:
        self.committed_steps.append(context.snapshot.step_index)

    def _refresh_causal_observation(self, previous_observation):
        del previous_observation
        return np.asarray([self.step_index], dtype=np.float32)


def _snapshot(env: _Env) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (float(env.step_index), 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        box(-1.0, -0.5, 1.0, 0.5),
        "lane-0",
        20.0,
    )
    return EnvSnapshot(
        "scenario", env.step_index, env.step_index * 0.1, ego, (ego,), (), frozenset(), {}
    )


def test_wrapper_publishes_committed_context_without_rulebook_result() -> None:
    cache = EpisodeCache("scenario", TaskRouteRecord("scenario", ("lane-0",), "task", "v1", "hash"))

    def evaluate(**kwargs):
        memory = kwargs["memory"]
        return (
            RulebookResult(
                margins=(0.0, 0.0, 0.0, 0.0),
                costs=(0.0, 0.0, 0.0),
                raw_progress_m=0.0,
                components={},
                complete_evaluation=True,
            ),
            memory,
            CacheDelta(),
        )

    wrapper = RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=_snapshot,
        transition_evaluator=evaluate,
        initial_memory=RulebookMemory(),
        initial_cache=cache,
        mission_route=_MISSION_ROUTE,
    )
    wrapper.reset()
    wrapper.step(np.zeros(1, dtype=np.float32))

    context = wrapper.causal_scene_context
    assert context.snapshot.step_index == 1
    assert wrapper.unwrapped.causal_scene_context is context
    assert not hasattr(context, "result")


def test_wrapper_refreshes_observation_after_memory_context_commit() -> None:
    cache = EpisodeCache("scenario", TaskRouteRecord("scenario", ("lane-0",), "task", "v1", "hash"))

    def evaluate(**kwargs):
        return (
            RulebookResult(
                margins=(0.0, 0.0, 0.0, 0.0),
                costs=(0.0, 0.0, 0.0),
                raw_progress_m=0.0,
                components={},
                complete_evaluation=True,
            ),
            kwargs["memory"],
            CacheDelta(),
        )

    environment = _Env()
    wrapper = RulebookV2MonitorWrapper(
        environment,
        snapshotter=_snapshot,
        transition_evaluator=evaluate,
        initial_memory=RulebookMemory(),
        initial_cache=cache,
        mission_route=_MISSION_ROUTE,
    )
    wrapper.reset()
    observation, *_ = wrapper.step(np.zeros(1, dtype=np.float32))

    assert environment.committed_steps == [0, 1]
    assert np.array_equal(observation, np.asarray([1], dtype=np.float32))
