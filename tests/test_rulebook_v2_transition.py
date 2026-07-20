from __future__ import annotations

from dataclasses import replace

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.transition import (
    RulebookTransitionConfig,
    align_episode_cache_to_live_elevation,
    evaluate_transition,
    initial_memory_for_snapshot,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    EpisodeCache,
    EnvSnapshot,
    ContactOnsetRecord,
    RulebookMemory,
    TaskRouteRecord,
)


def _snapshot(step: int, time_s: float, x: float) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, 0.0),
        0.0,
        0.0,
        (1.0, 0.0),
        Polygon(((x - 1.0, -1.0), (x + 1.0, -1.0), (x + 1.0, 1.0), (x - 1.0, 1.0))),
        "lane-a",
        10.0,
    )
    return EnvSnapshot("scenario", step, time_s, ego, (), (), frozenset(), {})


def _cache() -> EpisodeCache:
    route = RoutePolyline(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane-a",
        Polygon(((-1.0, -2.0), (21.0, -2.0), (21.0, 2.0), (-1.0, 2.0))),
        route,
        (),
    )
    return EpisodeCache(
        "scenario",
        TaskRouteRecord("scenario", ("lane-a",), "test", "v2", "hash"),
        route_lanes=(lane,),
        route_polyline=route,
    )


def test_transition_invokes_complete_registry_and_keeps_vehicle_yield_not_applicable() -> None:
    cache = _cache()
    pre = _snapshot(0, 0.0, 1.0)
    post = _snapshot(1, 0.1, 1.1)
    memory = initial_memory_for_snapshot(pre, cache)
    result, next_memory, cache_delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=memory,
        cache=cache,
        config=RulebookTransitionConfig(),
    )
    assert result.complete_evaluation
    assert set(result.components) == {
        "collision",
        "rss",
        "ttc",
        "clearance",
        "offroad",
        "wrongway",
        "solid_line",
        "dashed_line",
        "signal",
        "stop",
        "crosswalk",
        "vehicle_yield",
        "progress",
        "collision_impact",
        "dynamic_interaction_safety",
        "road_traffic_compliance",
    }
    assert result.components["vehicle_yield"].applicable is False
    assert next_memory.previous_route_s_m > memory.previous_route_s_m
    assert cache_delta.new_conflict_zones == ()


def test_transition_evaluates_contact_onset_from_post_snapshot() -> None:
    cache = _cache()
    pre = _snapshot(0, 0.0, 1.0)
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (2.0, 0.0),
        0.0,
        0.0,
        (-1.0, 0.0),
        Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0))),
        "lane-a",
        10.0,
    )
    pre = replace(pre, actors=(other,))
    post = replace(
        _snapshot(1, 0.1, 1.1),
        actors=(other,),
        contact_onset_records=(
            ContactOnsetRecord("other", ActorClass.VEHICLE),
        ),
        active_contact_ids=frozenset({"other"}),
    )
    memory = initial_memory_for_snapshot(pre, cache)
    result, _, _ = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=memory,
        cache=cache,
        config=RulebookTransitionConfig(),
    )
    assert result.components["collision"].raw["new_collision"] is True
    assert result.components["collision"].applicable is True


def test_transition_rejects_non_positive_simulation_step() -> None:
    cache = _cache()
    pre = _snapshot(0, 1.0, 1.0)
    post = _snapshot(1, 1.0, 1.1)
    try:
        evaluate_transition(
            pre_state=pre,
            post_state=post,
            memory=RulebookMemory(previous_route_s_m=1.0, previous_sim_time_s=1.0),
            cache=cache,
            config=RulebookTransitionConfig(),
        )
    except ValueError as error:
        assert "timestep" in str(error)
    else:
        raise AssertionError("non-positive timestep must fail fast")


def test_cache_elevation_alignment_preserves_relative_route_shape() -> None:
    cache = _cache()
    shifted_route = RoutePolyline(((0.0, 0.0, 50.0), (20.0, 0.0, 50.0)))
    shifted_lane = RouteLaneRecord(
        "lane-a",
        cache.route_lanes[0].polygon_xy,
        shifted_route,
        (),
    )
    shifted_cache = EpisodeCache(
        "scenario",
        cache.task_route,
        route_lanes=(shifted_lane,),
        route_polyline=shifted_route,
    )
    aligned = align_episode_cache_to_live_elevation(shifted_cache, _snapshot(0, 0.0, 1.0))
    assert aligned.route_polyline is not None
    assert aligned.route_polyline.points_xyz[0][2] == 0.0
    assert aligned.route_polyline.points_xyz[-1][2] == 0.0
