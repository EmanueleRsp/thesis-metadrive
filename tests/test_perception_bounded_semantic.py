from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

import thesis_rl.envs.observations.causal_semantic as causal_semantic
from test_causal_semantic_batch import _Vehicle, _actor, _context, _route
from thesis_rl.envs.observations.causal_semantic import PerceptionBoundedSemanticBatchBuilder
from thesis_rl.envs.observations.causal_semantic import CausalSemanticObservationError
from thesis_rl.rulebook.v2.types import ConflictZoneRecord, MovementKey, RoundaboutPriorityRecord


def test_v12_cache_preserves_an_actual_occlusion_gap(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    actor = _actor("other", (15.0, 0.0), (0.0, 0.0))
    visible_ids = iter(
        (frozenset({"other"}), frozenset({"other"}), frozenset(), frozenset({"other"}))
    )
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=next(visible_ids)),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    builder.build(_Vehicle(), _context(0, ego, (actor,), route, lanes))
    builder.build(
        _Vehicle(), _context(1, ego, (replace(actor, position_xy=(16.0, 0.0)),), route, lanes)
    )
    builder.build(
        _Vehicle(), _context(2, ego, (replace(actor, position_xy=(17.0, 0.0)),), route, lanes)
    )
    reacquired = builder.build(
        _Vehicle(), _context(3, ego, (replace(actor, position_xy=(18.0, 0.0)),), route, lanes)
    )

    slot = int(np.flatnonzero(reacquired.dynamic_mask[:, -1])[0])
    assert reacquired.dynamic_mask[slot].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert reacquired.dynamic[slot, 3].tolist() == [0.0] * 22


def test_v12_reacquired_track_reuses_its_persistent_slot(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    actor = _actor("other", (15.0, 0.0), (0.0, 0.0))
    visible_ids = iter((frozenset({"other"}), frozenset(), frozenset({"other"})))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=next(visible_ids)),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    first = builder.build(_Vehicle(), _context(0, ego, (actor,), route, lanes))
    first_slot = int(np.flatnonzero(first.dynamic_mask[:, -1])[0])
    builder.build(_Vehicle(), _context(1, ego, (), route, lanes))
    reacquired = builder.build(
        _Vehicle(), _context(2, ego, (replace(actor, position_xy=(16.0, 0.0)),), route, lanes)
    )

    assert np.flatnonzero(reacquired.dynamic_mask[:, -1]).tolist() == [first_slot]


def test_v13_route_width_is_local_and_adjacency_fields_are_gone(monkeypatch) -> None:
    """OBS-V1.3 (DEC-002, ADR-033): the two permanently-zero adjacent-lane
    fields are removed rather than emitted as fail-closed constants."""

    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    batch = builder.build(_Vehicle(), _context(0, ego, (), route, lanes))

    assert batch.route[0, 5] == pytest.approx(4.0 / 6.0)
    assert batch.lane_road.shape == (14,)


def test_v12_rejects_route_samples_without_a_containing_lane(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=(), brake_mps2=4.0)

    with pytest.raises(CausalSemanticObservationError, match="route-lane polygon"):
        builder.build(_Vehicle(), _context(0, ego, (), route, lanes))


def test_v12_conflict_candidate_is_retained_over_noncritical_overflow(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    conflict = _actor("conflict", (10.0, 0.0), lane_id="lane-1")
    ordinary = tuple(_actor(f"ordinary-{index:02d}", (20.0 + index, 1.0)) for index in range(16))
    actors = (conflict, *ordinary)
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset(actor.actor_id for actor in actors)),
    )
    zone = ConflictZoneRecord(
        "conflict-zone",
        conflict.footprint,
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-1", "node", "lane-1"),
        8.0,
        12.0,
        0.0,
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    batch = builder.build(_Vehicle(), _context(0, ego, actors, route, lanes, zones=(zone,)))

    assert builder._slot_actor[0] == "conflict"
    assert batch.dynamic_mask[:, -1].sum() == 16.0
    assert builder.diagnostics.capacity_dropped["dynamic"] == 1


def test_v12_compliance_trace_is_right_aligned_and_does_not_expose_rulebook_timers(
    monkeypatch,
) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    batch = builder.build(_Vehicle(), _context(0, ego, (), route, lanes))

    assert batch.context_history.shape == (21, 23)
    assert batch.context_history_mask.tolist() == [0.0] * 20 + [1.0]
    assert batch.signal_onset_state.shape == (3,)
    assert np.isfinite(batch.context_history).all()


def test_v12_compliance_trace_preserves_actual_step_gaps(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    builder.build(_Vehicle(), _context(0, ego, (), route, lanes))
    batch = builder.build(_Vehicle(), _context(3, ego, (), route, lanes))

    assert batch.context_history_mask.tolist() == [0.0] * 17 + [1.0, 0.0, 0.0, 1.0]


def test_v13_repeated_build_for_the_same_step_does_not_duplicate_the_context_row(
    monkeypatch,
) -> None:
    """REQ-AF-03 (OBS-AUDIT-FIX-001): MetaDrive calls ``observe()`` inside
    ``step()`` with the previous committed context and the Rulebook wrapper calls
    it again after the commit.  The second build for one step must return the
    batch already built for it instead of appending a second history row."""

    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset()),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)

    context_0 = _context(0, ego, (), route, lanes)
    first = builder.build(_Vehicle(), context_0)
    repeated = builder.build(_Vehicle(), context_0)
    assert repeated is first

    # Reproduce the production call pattern: at every step the previous context
    # is observed once more before the new one is committed.  Before the fix the
    # 21-slot deque held two entries per step and the window collapsed to ~11
    # distinct steps.
    previous = context_0
    for step in range(1, 25):
        builder.build(_Vehicle(), previous)
        previous = _context(step, ego, (), route, lanes)
        batch = builder.build(_Vehicle(), previous)
    assert batch.context_history_mask.tolist() == [1.0] * 21
    assert builder._last_context_row_step == 24


def test_v13_dynamic_station_difference_uses_committed_mission_station(monkeypatch) -> None:
    """REQ-AF-02 (OBS-AUDIT-FIX-001): on a route that returns next to itself,
    an un-anchored ego projection snaps to the far branch; the committed mission
    station is the single station authority for the dynamic block."""

    from shapely.geometry import box as _box

    from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

    # Out along y=0 for 100 m, then back along y=3 m.
    route = RoutePolyline(
        ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 3.0, 0.0), (0.0, 3.0, 0.0))
    )
    lane = RouteLaneRecord("lane-0", _box(-2.0, -2.0, 102.0, 5.0), route)
    lanes = (lane,)
    # Ego on the outbound branch at s = 50 m, 1.6 m to the left: the return
    # branch (y = 3) is 1.4 m away, so the nearest-point projection picks it.
    ego = _actor("ego", (50.0, 1.6), (5.0, 0.0))
    other = _actor("other", (60.0, 0.0), (0.0, 0.0))
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset({"other"})),
    )
    context = _context(0, ego, (other,), route, lanes)
    committed = replace(
        context,
        snapshot=replace(
            context.snapshot,
            mission_snapshot=replace(context.snapshot.mission_snapshot, s_m=50.0),
        ),
    )
    naive_ego_s = route.project(ego.position_xy).s_m
    assert naive_ego_s > 100.0, "fixture must make the un-anchored projection pick the far branch"

    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)
    batch = builder.build(_Vehicle(), committed)

    slot = int(np.flatnonzero(batch.dynamic_mask[:, -1])[0])
    assert batch.dynamic[slot, -1, causal_semantic.DYNAMIC_ROUTE_STATION_INDEX] == pytest.approx(
        (60.0 - 50.0) / 50.0
    )


def test_v12_interaction_type_is_unknown_without_source_taxonomy(monkeypatch) -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    zone = ConflictZoneRecord(
        "zone",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        10.0,
        20.0,
        0.0,
    )
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda _vehicle: SimpleNamespace(actor_ids=frozenset({"other"})),
    )
    builder = PerceptionBoundedSemanticBatchBuilder(route=route, route_lanes=lanes, brake_mps2=4.0)
    context = _context(0, ego, (other,), route, lanes, zones=(zone,))

    assert builder._conflict_zone_type_index(context, zone) is None

    roundabout_context = replace(
        context,
        episode_cache=replace(
            context.episode_cache,
            roundabout_priority_records=(RoundaboutPriorityRecord("r", "lane-0", "lane-x"),),
        ),
    )
    zone_with_roundabout_lane = replace(
        zone,
        other_movement_key=MovementKey("lane-x", "node", "lane-x"),
    )
    assert builder._conflict_zone_type_index(roundabout_context, zone_with_roundabout_lane) == 3
