from __future__ import annotations

from dataclasses import replace

import pytest
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact
from thesis_rl.rulebook.v2.transition import (
    RulebookTransitionConfig,
    align_episode_cache_to_live_elevation,
    evaluate_transition,
    initial_memory_for_snapshot,
)
from thesis_rl.rulebook.v2 import transition as transition_module
from thesis_rl.rulebook.v2.memory import apply_cache_delta
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    EpisodeCache,
    EnvSnapshot,
    ContactOnsetRecord,
    MapFeatureClass,
    MapFeatureRecord,
    RulebookMemory,
    TaskRouteRecord,
    ApproachControl,
    MovementKey,
    TrafficControlRecord,
    MovementPriority,
    MovementPriorityRecord,
    RoundaboutPriorityRecord,
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
        "rss_lateral",
        "ttc",
        "clearance",
        "offroad",
        "wrongway",
        "wrong_carriageway",
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


def test_transition_benchmark_can_skip_vehicle_yield_input_construction(monkeypatch) -> None:
    cache = _cache()
    pre = _snapshot(0, 0.0, 1.0)
    post = _snapshot(1, 0.1, 1.1)
    memory = initial_memory_for_snapshot(pre, cache)

    def unexpected_vehicle_yield(**_kwargs):
        raise AssertionError("vehicle-yield construction must be skipped")

    monkeypatch.setattr(transition_module, "_vehicle_yield_inputs", unexpected_vehicle_yield)
    result, _, cache_delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=memory,
        cache=cache,
        config=RulebookTransitionConfig(disable_vehicle_yield_for_benchmark=True),
    )

    assert "vehicle_yield" not in result.components
    assert cache_delta.diagnostic_timing_seconds["vehicle_yield"] == 0.0


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
        contact_onset_records=(ContactOnsetRecord("other", ActorClass.VEHICLE),),
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


def test_transition_reuses_lane_associations_across_rulebook_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crosswalk, vehicle-yield and RSS share each snapshot's associations exactly."""

    cache = _cache()
    pre = _snapshot(0, 0.0, 1.0)
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (3.0, 0.0),
        0.0,
        0.0,
        (1.0, 0.0),
        Polygon(((2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0))),
        "lane-a",
        10.0,
    )
    pre = replace(pre, actors=(other,))
    post = replace(_snapshot(1, 0.1, 1.1), actors=(other,))
    memory = initial_memory_for_snapshot(pre, cache)
    original = transition_module.associate_route_lane
    calls = 0

    def counted_association(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(transition_module, "associate_route_lane", counted_association)

    result, _, _ = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=memory,
        cache=cache,
        config=RulebookTransitionConfig(
            rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
            expected_config_hash="calibration",
        ),
    )

    assert result.complete_evaluation
    assert calls == 4  # ego + one vehicle for each of pre/post snapshots


def test_transition_penalizes_a_red_signal_crossed_during_step() -> None:
    cache = _cache()
    control = TrafficControlRecord(
        "signal:p",
        ApproachControl.SIGNAL,
        ("lane-a",),
        MovementKey("lane-a", "node", "lane-a"),
        LineString(((10.0, -2.0), (10.0, 2.0))),
        10.0,
        0.0,
        ("p",),
    )
    cache = replace(cache, traffic_control_catalog=(control,))
    pre = _snapshot(0, 0.0, 8.0)
    post = replace(_snapshot(1, 0.1, 10.0), signal_states_by_physical_id={"p": "RED"})
    pre = replace(pre, signal_states_by_physical_id={"p": "RED"})
    memory = initial_memory_for_snapshot(pre, cache)

    result, next_memory, _ = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=memory,
        cache=cache,
        config=RulebookTransitionConfig(
            rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
            expected_config_hash="calibration",
        ),
    )

    assert result.components["signal"].cost == 1.0
    assert "signal:p" in next_memory.resolved_signal_group_ids


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


def test_transition_vehicle_yield_uses_each_scoped_priority_predicate() -> None:
    """Live transition wiring must reach the pure evaluator for all §7.9 predicates."""

    cache = _cache()
    lane_b = RouteLaneRecord(
        "lane-b",
        Polygon(((8.0, -10.0), (12.0, -10.0), (12.0, 10.0), (8.0, 10.0))),
        RoutePolyline(((10.0, -10.0, 0.0), (10.0, 10.0, 0.0))),
        (),
    )
    ego_key = MovementKey("lane-a", "junction:lane-a->lane-a", "lane-a")
    other_key = MovementKey("lane-b", "junction:lane-b->lane-b", "lane-b")
    cache = replace(cache, route_lanes=(cache.route_lanes[0], lane_b))
    pre = _snapshot(0, 0.0, 6.0)
    post = _snapshot(1, 0.1, 6.5)
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 3.0),
        0.0,
        -1.57079632679,
        (0.0, -2.0),
        Polygon(((9.0, 2.0), (11.0, 2.0), (11.0, 4.0), (9.0, 4.0))),
        "lane-b",
        10.0,
    )
    pre = replace(pre, actors=(other,))
    post = replace(post, actors=(other,))
    calibration = RulebookTransitionConfig(
        rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
        expected_config_hash="calibration",
    )
    for cache_variant in (
        # Explicit pairwise priority.
        replace(
            cache,
            movement_priority_records=(
                MovementPriorityRecord(ego_key, other_key, MovementPriority.OTHER_HAS_PRIORITY),
            ),
        ),
        # Validated roundabout entry/circulating relation.
        replace(
            cache, roundabout_priority_records=(RoundaboutPriorityRecord("r", "lane-a", "lane-b"),)
        ),
        # Ego STOP versus an uncontrolled other movement.
        replace(
            cache,
            traffic_control_catalog=(
                TrafficControlRecord(
                    "stop",
                    ApproachControl.STOP,
                    ("lane-a",),
                    ego_key,
                    LineString(((8.0, -2.0), (8.0, 2.0))),
                    8.0,
                    0.0,
                    (),
                ),
            ),
        ),
    ):
        result, _memory, cache_delta = evaluate_transition(
            pre_state=pre,
            post_state=post,
            memory=initial_memory_for_snapshot(pre, cache_variant),
            cache=cache_variant,
            config=calibration,
        )
        assert result.components["vehicle_yield"].applicable
        assert cache_delta.new_conflict_zones

    # Occupancy alone is also a deterministic priority predicate.
    occupied = replace(
        other,
        position_xy=(10.0, 0.0),
        footprint=Polygon(((9.0, -1.0), (11.0, -1.0), (11.0, 1.0), (9.0, 1.0))),
    )
    result, _memory, _delta = evaluate_transition(
        pre_state=replace(pre, actors=(occupied,)),
        post_state=replace(post, actors=(occupied,)),
        memory=initial_memory_for_snapshot(replace(pre, actors=(occupied,)), cache),
        cache=cache,
        config=calibration,
    )
    assert result.components["vehicle_yield"].applicable


def _yield_geometry_cache() -> EpisodeCache:
    cache = _cache()
    lane_b = RouteLaneRecord(
        "lane-b",
        Polygon(((8.0, -10.0), (12.0, -10.0), (12.0, 10.0), (8.0, 10.0))),
        RoutePolyline(((10.0, -10.0, 0.0), (10.0, 10.0, 0.0))),
        (),
    )
    return replace(cache, route_lanes=(cache.route_lanes[0], lane_b))


def _yield_config() -> RulebookTransitionConfig:
    return RulebookTransitionConfig(
        rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
        expected_config_hash="calibration",
    )


def test_transition_clears_vehicle_yield_illegal_entry_latch_after_ego_fully_exits_zone() -> None:
    """Regression test for ADR-025: a stale vehicle-yield illegal-entry latch.

    Section 2.8.2's "ahead" filter (epsilon = 0.05 m) excludes a conflict
    zone from selection long before the ego footprint (length 2 m here, via
    ``_snapshot``) actually stops intersecting it, so
    ``select_first_ahead_or_occupied_zone`` returns ``None`` for the zone at
    the exact step ``ego_occupied`` becomes ``False``. The illegal-entry
    latch must still clear at that step, independent of which zone (if any)
    is selected for the step's own cost evaluation.
    """

    cache = _yield_geometry_cache()
    config = _yield_config()
    # Centred at (10, 3) -- outside lane-a's lateral extent, so association
    # resolves unambiguously to lane-b -- with a footprint reaching down to
    # y=-1 so it overlaps the lane-a/lane-b conflict zone (occupancy alone is
    # a deterministic priority predicate, as in
    # `test_transition_vehicle_yield_uses_each_scoped_priority_predicate`).
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 3.0),
        0.0,
        -1.57079632679,
        (0.0, -2.0),
        Polygon(((9.0, -1.0), (11.0, -1.0), (11.0, 4.0), (9.0, 4.0))),
        "lane-b",
        10.0,
    )

    # The conflict zone spans route s in roughly [7.99, 12.01] (lane-a's
    # width intersected with lane-b's width).
    # x=9.0: ego enters the zone (front_s=10, occupied) while "other" already
    # occupies it -> illegal entry recorded.
    # x=10.5: still occupied, still "ahead" (front_s=11.5 <= 12.06).
    # x=11.5: still occupied (front_s=12.5, rear_s=10.5 < 12.01), but already
    # NOT "ahead" (front_s=12.5 > 12.06) -- demonstrates that the "occupied"
    # branch does not depend on the "ahead" filter, so the zone stays
    # selected through this step.
    # x=13.5: ego has fully left (rear_s=12.5 > 12.01) -> `ego_occupied`
    # becomes False at the exact step "ahead" has long been False too, so
    # `select_first_ahead_or_occupied_zone` returns None. The latch must
    # clear here regardless.
    ego_positions = (6.0, 9.0, 10.5, 11.5, 13.5, 15.0)
    pre = replace(_snapshot(0, 0.0, ego_positions[0]), actors=(other,))
    memory = initial_memory_for_snapshot(pre, cache)
    illegal_entry_observed = False
    for index, x in enumerate(ego_positions[1:], start=1):
        post = replace(_snapshot(index, index * 0.1, x), actors=(other,))
        result, memory, delta = evaluate_transition(
            pre_state=pre,
            post_state=post,
            memory=memory,
            cache=cache,
            config=config,
        )
        cache = apply_cache_delta(cache, delta)
        if ("other", result.components["vehicle_yield"].raw["zone_id"]) in (
            memory.vehicle_yield_illegal_entries
        ):
            illegal_entry_observed = True
        pre = post

    assert illegal_entry_observed, "test setup must actually record an illegal entry first"
    assert memory.vehicle_yield_illegal_entries == frozenset()


def test_transition_clears_crosswalk_illegal_entry_latch_after_ego_fully_exits_zone() -> None:
    """Regression test for the same bug class as ADR-025, on the crosswalk
    component: ``select_first_ahead_or_occupied_zone``'s epsilon-scoped
    "ahead" filter can exclude the crosswalk zone from selection before the
    ego footprint (length 2 m here, via ``_snapshot``) actually stops
    intersecting it, so the illegal-entry latch must clear once ego's
    footprint has geometrically left the zone, independent of which zone (if
    any) is selected for the step's own cost evaluation.
    """

    base_cache = _cache()
    crosswalk_polygon = Polygon(((8.0, -5.0), (12.0, -5.0), (12.0, 5.0), (8.0, 5.0)))
    feature = MapFeatureRecord("crosswalk-1", MapFeatureClass.CROSSWALK, crosswalk_polygon, 0.0)
    cache = replace(base_cache, map_feature_catalog={"crosswalk-1": feature})
    config = RulebookTransitionConfig(
        rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
        expected_config_hash="calibration",
    )
    # A stationary pedestrian sitting inside the crosswalk zone the whole
    # time keeps the temporal gap unsafe (worst > 0) whenever the zone is
    # selected and occupied, so ego's first entry is recorded as illegal.
    ped = ActorSnapshot(
        "ped",
        ActorClass.PEDESTRIAN,
        (10.0, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        Polygon(((9.5, -0.5), (10.5, -0.5), (10.5, 0.5), (9.5, 0.5))),
        None,
        2.0,
    )
    ego_positions = (6.0, 9.0, 10.5, 11.5, 13.5, 15.0)
    pre = replace(_snapshot(0, 0.0, ego_positions[0]), actors=(ped,))
    memory = initial_memory_for_snapshot(pre, cache)
    illegal_entry_observed = False
    for index, x in enumerate(ego_positions[1:], start=1):
        post = replace(_snapshot(index, index * 0.1, x), actors=(ped,))
        result, memory, delta = evaluate_transition(
            pre_state=pre,
            post_state=post,
            memory=memory,
            cache=cache,
            config=config,
        )
        cache = apply_cache_delta(cache, delta)
        if (
            "ped",
            result.components["crosswalk"].raw["zone_id"],
        ) in memory.crosswalk_illegal_entries:
            illegal_entry_observed = True
        pre = post

    assert illegal_entry_observed, (
        "test setup must actually record an illegal crosswalk entry first"
    )
    assert memory.crosswalk_illegal_entries == frozenset()


def test_transition_vehicle_yield_latches_illegal_entry_from_pre_state_even_if_actor_exits_same_step() -> (
    None
):
    """Regression for the DEC-005 wiring bug (REQ-VY-01): an actor occupying
    the conflict zone in the pre-state, but gone from it by the post-state
    (it exits during the same control step), must still gate the
    illegal-entry latch when ego enters that same step."""

    cache = _yield_geometry_cache()
    pre_ego = _snapshot(0, 0.0, 6.0)
    post_ego = _snapshot(1, 0.1, 9.0)
    occupying = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 0.0),
        0.0,
        1.57079632679,
        (0.0, 5.0),
        Polygon(((9.0, -1.0), (11.0, -1.0), (11.0, 1.0), (9.0, 1.0))),
        "lane-b",
        10.0,
    )
    exited = replace(
        occupying,
        position_xy=(10.0, 6.0),
        footprint=Polygon(((9.0, 5.0), (11.0, 5.0), (11.0, 7.0), (9.0, 7.0))),
    )
    pre = replace(pre_ego, actors=(occupying,))
    post = replace(post_ego, actors=(exited,))
    result, next_memory, _delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=initial_memory_for_snapshot(pre, cache),
        cache=cache,
        config=_yield_config(),
    )
    zone_id = result.components["vehicle_yield"].raw["zone_id"]
    assert result.components["vehicle_yield"].applicable is True
    assert result.components["vehicle_yield"].cost == 1.0
    assert ("other", zone_id) in next_memory.vehicle_yield_illegal_entries


def test_transition_vehicle_yield_no_latch_when_pre_state_gap_is_sufficient() -> None:
    """Counterpart of the regression above: an occupying actor whose
    pre-state occupancy of the zone clears well before ego's predicted entry
    must not create a latch (sufficient r_gap^-)."""

    cache = _yield_geometry_cache()
    pre_ego = _snapshot(0, 0.0, 6.0)
    post_ego = _snapshot(1, 0.1, 6.5)
    clearing_early = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 0.0),
        0.0,
        1.57079632679,
        (0.0, 40.0),
        Polygon(((9.0, -1.0), (11.0, -1.0), (11.0, 1.0), (9.0, 1.0))),
        "lane-b",
        10.0,
    )
    already_cleared = replace(
        clearing_early,
        position_xy=(10.0, 4.0),
        footprint=Polygon(((9.0, 3.0), (11.0, 3.0), (11.0, 5.0), (9.0, 5.0))),
    )
    pre = replace(pre_ego, actors=(clearing_early,))
    post = replace(post_ego, actors=(already_cleared,))
    result, next_memory, _delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=initial_memory_for_snapshot(pre, cache),
        cache=cache,
        config=_yield_config(),
    )
    assert next_memory.vehicle_yield_illegal_entries == frozenset()
    assert result.components["vehicle_yield"].cost == 0.0


def test_transition_vehicle_yield_approach_cost_clears_once_actor_exits_before_entry() -> None:
    """REQ-VY-03: if ego never enters the zone and the priority actor exits,
    the continuous approach cost must fall back to zero (post-state view)."""

    cache = _yield_geometry_cache()
    pre_ego = _snapshot(0, 0.0, 2.0)
    post_ego = _snapshot(1, 0.1, 2.2)
    occupying = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 0.0),
        0.0,
        1.57079632679,
        (0.0, 5.0),
        Polygon(((9.0, -1.0), (11.0, -1.0), (11.0, 1.0), (9.0, 1.0))),
        "lane-b",
        10.0,
    )
    exited = replace(
        occupying,
        position_xy=(10.0, 6.0),
        footprint=Polygon(((9.0, 5.0), (11.0, 5.0), (11.0, 7.0), (9.0, 7.0))),
    )
    pre = replace(pre_ego, actors=(occupying,))
    post = replace(post_ego, actors=(exited,))
    result, next_memory, _delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=initial_memory_for_snapshot(pre, cache),
        cache=cache,
        config=_yield_config(),
    )
    assert result.components["vehicle_yield"].cost == 0.0
    assert next_memory.vehicle_yield_illegal_entries == frozenset()


def test_vehicle_conflict_pair_cache_reuses_complete_canonical_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    lane_b = RouteLaneRecord(
        "lane-b",
        Polygon(((8.0, -10.0), (12.0, -10.0), (12.0, 10.0), (8.0, 10.0))),
        RoutePolyline(((10.0, -10.0, 0.0), (10.0, 10.0, 0.0))),
        (),
    )
    ego_key = MovementKey("lane-a", "junction:lane-a->lane-a", "lane-a")
    other_key = MovementKey("lane-b", "junction:lane-b->lane-b", "lane-b")
    cache = replace(
        cache,
        route_lanes=(cache.route_lanes[0], lane_b),
        movement_priority_records=(
            MovementPriorityRecord(ego_key, other_key, MovementPriority.OTHER_HAS_PRIORITY),
        ),
    )
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (10.0, 3.0),
        0.0,
        -1.57079632679,
        (0.0, -2.0),
        Polygon(((9.0, 2.0), (11.0, 2.0), (11.0, 4.0), (9.0, 4.0))),
        "lane-b",
        10.0,
    )
    other_same_movement = replace(
        other,
        actor_id="other-same-movement",
        position_xy=(10.0, 5.0),
        footprint=Polygon(((9.0, 4.0), (11.0, 4.0), (11.0, 6.0), (9.0, 6.0))),
    )
    pre = replace(_snapshot(0, 0.0, 6.0), actors=(other, other_same_movement))
    post = replace(_snapshot(1, 0.1, 6.5), actors=(other, other_same_movement))
    config = RulebookTransitionConfig(
        rss_calibration=RSSCalibrationArtifact("calibration", 4.0),
        expected_config_hash="calibration",
    )
    calls = 0
    candidate_counts: list[int] = []
    original = transition_module.build_vehicle_conflict_zone_candidates
    original_select = transition_module.select_first_ahead_or_occupied_zone

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(transition_module, "build_vehicle_conflict_zone_candidates", counted)

    def count_candidates(*args, **kwargs):
        candidate_counts.append(len(kwargs["candidates"]))
        return original_select(*args, **kwargs)

    monkeypatch.setattr(transition_module, "select_first_ahead_or_occupied_zone", count_candidates)
    first, memory, delta = evaluate_transition(
        pre_state=pre,
        post_state=post,
        memory=initial_memory_for_snapshot(pre, cache),
        cache=cache,
        config=config,
    )
    cached = apply_cache_delta(cache, delta)
    second_post = replace(
        post, step_index=2, sim_time_s=0.2, ego=replace(post.ego, position_xy=(7.0, 0.0))
    )
    second, _memory, _delta = evaluate_transition(
        pre_state=post,
        post_state=second_post,
        memory=memory,
        cache=cached,
        config=config,
    )

    assert calls == 1  # one local pair build despite two actors on the same movement
    assert candidate_counts == [1, 1]
    assert len(cached.vehicle_conflict_pairs) == 1
    assert (
        first.components["vehicle_yield"].raw["zone_id"]
        == second.components["vehicle_yield"].raw["zone_id"]
    )


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


def test_cache_elevation_alignment_preserves_lane_start_points_for_checkpoints() -> None:
    """Regression: `align_episode_cache_to_live_elevation` used to rebuild the
    route via ``RoutePolyline``'s plain constructor, silently resetting
    ``lane_start_points_xyz`` (video overlay v1 REQ-001 checkpoint markers,
    docs/implementation/evaluation_video_route_and_ego_trail_overlay_v1_exec_plan.md)
    to its empty default whenever a nonzero elevation offset applied -- the
    common case for Waymo scenarios per this function's docstring."""

    cache = _cache()
    route_with_checkpoints = RoutePolyline.from_lane_centerlines(
        (((0.0, 0.0, 50.0), (20.0, 0.0, 50.0)),)
    )
    lane = RouteLaneRecord(
        "lane-a",
        cache.route_lanes[0].polygon_xy,
        route_with_checkpoints,
        (),
    )
    cache_with_checkpoints = EpisodeCache(
        "scenario",
        cache.task_route,
        route_lanes=(lane,),
        route_polyline=route_with_checkpoints,
    )
    aligned = align_episode_cache_to_live_elevation(cache_with_checkpoints, _snapshot(0, 0.0, 1.0))
    assert aligned.route_polyline is not None
    assert aligned.route_polyline.lane_start_points_xyz == ((0.0, 0.0, 0.0),)
