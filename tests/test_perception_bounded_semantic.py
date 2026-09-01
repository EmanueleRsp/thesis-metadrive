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
