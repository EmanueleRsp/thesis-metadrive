"""`REQ-RB51-19`: RULEBOOK-V5.1 §7's reported-never-priced counters.

Two of the three exist to make a *claim falsifiable* rather than to describe a
run, and that is why they are asserted rather than merely emitted.
"""

from __future__ import annotations

import itertools
from typing import Sequence

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
    ComponentStatus,
    EpisodeCache,
    EnvSnapshot,
    RuleComponentResult,
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


def _progress_component(outside: float | None) -> dict[str, RuleComponentResult]:
    """A `progress` component carrying `REQ-EF-15`'s corridor diagnostic.

    ``None`` reproduces the case the evaluator actually emits when it has no ego
    footprint or no assigned corridor: the key is **absent**, not zero.
    """

    diagnostics: dict[str, float] = {}
    if outside is not None:
        diagnostics["route_outside_fraction"] = outside
        diagnostics["route_adherence"] = 1.0 - outside
    return {
        "progress": RuleComponentResult(
            "progress", 0.0, {}, True, True, ComponentStatus.SATISFIED, diagnostics
        )
    }


def _wrapper(
    margins: tuple[float, ...],
    *,
    route_outside: Sequence[float | None] | None = None,
) -> RulebookV2MonitorWrapper:
    """``route_outside`` supplies one corridor fraction per step, cycling if short."""

    route = TaskRouteRecord("s", ("lane",), "pg", "v2", "hash")
    calls = itertools.count()

    def evaluate_transition(**_kwargs):
        components: dict[str, RuleComponentResult] = {}
        if route_outside is not None:
            index = next(calls)
            components = _progress_component(route_outside[index % len(route_outside)])
        result = RulebookResult(margins, (0.0, 0.0, 0.0, 0.0, 0.0), 1.0, components, True)
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
    """A margin well inside the clip must not be counted as binding.

    *An earlier revision of this docstring* said the counter "should be zero for
    **any** agent trajectory", because MetaDrive caps every vehicle at
    `max_speed_km_h = 80 = v_ref`, and therefore that a non-zero count meant the
    cap had been overridden. That inference does not hold: the cap bounds the
    ego's *ground travel*, while `s` is a **projection** onto the route polyline,
    and ADR-035 sizes the difference at a factor of 2
    (`ROUTE_CONTINUITY_JUMP_FACTOR`, because cutting the inside of a bend
    advances the centreline coordinate faster than the ego). `AC-RB5.1-05` is
    `NOT ESTABLISHED` for that reason.

    The assertion is unchanged and still correct: it says only that a margin of
    0.4 is not the clip binding.
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


# ---------------------------------------------------------------------------
# `REQ-EF-15`'s corridor diagnostic, aggregated per episode (open item `D14`)
# ---------------------------------------------------------------------------


def test_route_adherence_is_not_reported_when_the_component_never_measured_it() -> None:
    """Zero excursions and zero measurements must not look alike.

    The progress evaluator omits `route_outside_fraction` when it has no ego
    footprint or no assigned corridor, so an episode can end having never
    measured adherence at all. Reporting `0` there would record "the ego never
    left its corridor" on evidence that does not exist.
    """

    wrapped = _wrapper((0.0, 0.0, 0.0, 0.2, 0.0, 0.0))
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["route_outside_evaluated_steps"] == 0
    assert info["mean_route_outside_fraction"] is None
    assert info["mean_route_adherence"] is None
    assert info["route_fully_outside_max_run"] == 0


def test_route_adherence_means_are_reported_when_it_is_measured() -> None:
    wrapped = _wrapper((0.0, 0.0, 0.0, 0.2, 0.0, 0.0), route_outside=[0.0, 0.5, 0.25])
    wrapped.reset()
    for _ in range(3):
        _, _, _, _, info = wrapped.step(0)

    assert info["route_outside_evaluated_steps"] == 3
    assert info["mean_route_outside_fraction"] == pytest.approx(0.25)
    assert info["mean_route_adherence"] == pytest.approx(0.75)
    # Two of the three steps had *some* area outside; none was fully outside.
    assert info["route_outside_steps"] == 2
    assert info["route_fully_outside_steps"] == 0


def test_longest_fully_outside_run_separates_a_clipped_corner_from_a_wrong_road() -> None:
    """The headline aggregate, and the reason a mean is not enough.

    Both trajectories below spend the same *number* of steps entirely off the
    assigned corridor, so their means and their totals are identical. Only the
    longest consecutive run tells them apart, and the distinction is the whole
    question: brushing a corridor edge on four separate corners is ordinary
    driving, while four consecutive steps is the beginning of being on another
    road with the projection still paying `R4`.
    """

    scattered = _wrapper(
        (0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
        route_outside=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    )
    scattered.reset()
    for _ in range(8):
        _, _, _, _, scattered_info = scattered.step(0)

    sustained = _wrapper(
        (0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
        route_outside=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    sustained.reset()
    for _ in range(8):
        _, _, _, _, sustained_info = sustained.step(0)

    assert scattered_info["route_fully_outside_steps"] == 4
    assert sustained_info["route_fully_outside_steps"] == 4
    assert scattered_info["mean_route_outside_fraction"] == pytest.approx(
        sustained_info["mean_route_outside_fraction"]
    )

    assert scattered_info["route_fully_outside_max_run"] == 1
    assert sustained_info["route_fully_outside_max_run"] == 4


def test_the_longest_run_survives_a_later_shorter_one() -> None:
    """The maximum, not the last run: a return to the corridor must not erase it."""

    wrapped = _wrapper(
        (0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
        route_outside=[1.0, 1.0, 1.0, 0.0, 1.0],
    )
    wrapped.reset()
    for _ in range(5):
        _, _, _, _, info = wrapped.step(0)

    assert info["route_fully_outside_max_run"] == 3
    assert info["route_fully_outside_steps"] == 4


def test_reset_clears_the_route_adherence_counters() -> None:
    """Per-episode, like every other counter here: one wrapper serves every episode.

    Left accumulating, `route_fully_outside_max_run` would report the worst
    excursion of any episode the slot has ever run, which is the one reading it
    must never give: the question it answers is whether *this* policy left *this*
    mission's corridor.
    """

    wrapped = _wrapper((0.0, 0.0, 0.0, 0.2, 0.0, 0.0), route_outside=[1.0, 1.0, 1.0])
    wrapped.reset()
    for _ in range(3):
        wrapped.step(0)

    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["route_outside_evaluated_steps"] == 1
    assert info["route_fully_outside_steps"] == 1
    assert info["route_fully_outside_max_run"] == 1
    assert info["mean_route_outside_fraction"] == pytest.approx(1.0)
