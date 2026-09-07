"""`REQ-RB51-19`: RULEBOOK-V5.1 §7's reported-never-priced counters.

Two of the three exist to make a *claim falsifiable* rather than to describe a
run, and that is why they are asserted rather than merely emitted.
"""

from __future__ import annotations

import gymnasium as gym
import pytest
from shapely.geometry import Polygon

from thesis_rl.reward.scalarization import (
    SIX_LEVEL_PRIORITY_BASE,
    SIX_LEVEL_VECTOR_SCHEMA_ID,
    RulebookScalarizer,
    ScalarizationConfig,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    EpisodeCache,
    EnvSnapshot,
    RulebookMemory,
    RulebookResult,
    TaskRouteRecord,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper


_EGO_SPEED_MPS = 3.0


class _Env(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.t = 0
        self.speed_mps = _EGO_SPEED_MPS

    def reset(self, **_kwargs):
        self.t = 0
        return [0.0], {}

    def step(self, _action):
        self.t += 1
        return [0.0], 0.0, False, False, {}


def _snapshot(step: int, speed_mps: float = _EGO_SPEED_MPS) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (float(step), 0.0),
        0.0,
        0.0,
        (speed_mps, 0.0),
        Polygon(((-2.0, -0.9), (2.0, -0.9), (2.0, 0.9), (-2.0, 0.9))),
        None,
        20.0,
    )
    return EnvSnapshot("s", step, step * 0.1, ego, (), (), frozenset(), {})


def _wrapper(margins: tuple[float, ...]) -> RulebookV2MonitorWrapper:
    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")

    def evaluate_transition(**_kwargs):
        result = RulebookResult(margins, (0.0, 0.0, 0.0, 0.0, 0.0), 1.0, {}, True)
        return result, RulebookMemory(), CacheDelta()

    return RulebookV2MonitorWrapper(
        _Env(),
        snapshotter=lambda env: _snapshot(env.t, env.speed_mps),
        transition_evaluator=evaluate_transition,
        initial_memory=RulebookMemory(),
        initial_cache=EpisodeCache("s", route),
        mission_route=RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))),
        scalarizer=RulebookScalarizer(
            ScalarizationConfig(
                mode="six_level_priority_weighted_rank",
                priority_base=SIX_LEVEL_PRIORITY_BASE,
                vector_schema_id=SIX_LEVEL_VECTOR_SCHEMA_ID,
            )
        ),
    )


def test_l4_clip_counter_stays_zero_on_a_reachable_step() -> None:
    """It should be zero for **any** agent trajectory.

    MetaDrive caps every vehicle at `max_speed_km_h = 80 = v_ref`, so the §4.1
    clip cannot bind on a step the agent produces. A non-zero count is not a
    statistic: it says the cap was overridden and that the Test A figures no
    longer bound what the agent can earn.
    """

    wrapped = _wrapper((0.0, 0.0, 0.0, 0.4, 0.0, 0.0))
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)
    assert info["l4_clip_binding_steps"] == 0


def test_l4_clip_counter_fires_when_the_clip_binds() -> None:
    wrapped = _wrapper((0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)
    assert info["l4_clip_binding_steps"] == 1


def test_l5_reached_counts_only_steps_where_nothing_above_it_is_charged() -> None:
    """If this stays zero in practice the restructure bought nothing.

    Moving the relaxable lane rules below progress buys exactly one ordering; a
    level that never decides anything has not bought it. The counter is what
    makes that reportable instead of assumed.
    """

    clean = _wrapper((0.0, 0.0, 0.0, 0.2, -0.5, -0.5))
    clean.reset()
    _, _, _, _, info = clean.step(0)
    assert info["l5_reached_steps"] == 1

    blocked = _wrapper((0.0, -0.3, 0.0, 0.2, -0.5, -0.5))
    blocked.reset()
    _, _, _, _, info = blocked.step(0)
    assert info["l5_reached_steps"] == 0


def test_counters_accumulate_across_steps_and_speed_is_published() -> None:
    wrapped = _wrapper((0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    wrapped.reset()
    for expected in (1, 2, 3):
        _, _, _, _, info = wrapped.step(0)
        assert info["l4_clip_binding_steps"] == expected
        assert info["l5_reached_steps"] == expected
    # Required by limitation 13: L6 rewards speed while `speed_limit` is admitted
    # on Waymo only, so the per-source divergence must be measured, not assumed.
    assert info["ego_speed_mps"] == pytest.approx(_EGO_SPEED_MPS)
    assert info["mean_ego_speed_mps"] == pytest.approx(_EGO_SPEED_MPS)


def test_control_line_diagnostics_are_published_at_reset_not_on_every_step() -> None:
    """Static per scenario, so it belongs in the reset info and nowhere else.

    The evaluation loop's `metadata_keys` reads this key from **both** the reset
    info and the last step info, and only the second used to fire: the wrapper
    returned the inner reset info unmodified, so the route the exec plan
    intended ("same mechanism as `scenario_uid`") was dead and a payload that
    cannot change was pickled through the worker pipe on every step instead.
    """

    wrapped = _wrapper((0.0, 0.0, 0.0, 0.2, 0.0, 0.0))
    _, reset_info = wrapped.reset()
    _, _, _, _, step_info = wrapped.step(0)

    assert isinstance(reset_info["rulebook_control_line_diagnostics"], dict)
    assert "rulebook_control_line_diagnostics" not in step_info


def test_reset_clears_the_episode_counters() -> None:
    """The counters are per-episode, and one wrapper serves every episode of a slot.

    Accumulating across `reset()` makes each of the three unreadable in a
    different way: `l4_clip_binding_steps` exists to be zero, so an override in
    any earlier episode reports forever; `l5_reached_steps` grows with the
    slot's age rather than with the episode; and `mean_ego_speed_mps` averages
    scenarios the episode never visited, which is exactly the per-source
    grouping limitation 13 asks for.
    """

    wrapped = _wrapper((0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    wrapped.reset()
    for _ in range(3):
        wrapped.step(0)

    second_episode_speed_mps = 9.0
    wrapped.env.speed_mps = second_episode_speed_mps
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["l4_clip_binding_steps"] == 1
    assert info["l5_reached_steps"] == 1
    assert info["ego_speed_mps"] == pytest.approx(second_episode_speed_mps)
    assert info["mean_ego_speed_mps"] == pytest.approx(second_episode_speed_mps)
