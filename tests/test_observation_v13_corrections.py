"""OBS-V1.3 causal-correctness matrix (PLAN-OBS-CORR-V1.3, TEST-001..TEST-033).

Each test here pins one acceptance criterion of the correction plan. The module
is deliberately separate from ``test_perception_bounded_semantic.py``, which
holds the OBS-V1.2 contract tests that must keep passing unchanged.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, box

import thesis_rl.envs.observations.causal_semantic as causal_semantic
from test_causal_semantic_batch import _Vehicle, _actor, _mission_snapshot
from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.envs.observations.causal_semantic import (
    CausalSemanticBatchBuilder,
    PerceptionBoundedSemanticBatchBuilder,
)
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ApproachControl,
    ConflictZoneRecord,
    EnvSnapshot,
    EpisodeCache,
    MapFeatureClass,
    MapFeatureRecord,
    MovementKey,
    RulebookMemory,
    StaticSubclass,
    TaskRouteRecord,
    TrafficControlRecord,
)
from thesis_rl.rulebook.v2.context.live_adapter import actor_snapshot_from_payload

LANE_WIDTH_M = 4.0
EGO_LENGTH_M = 2.0


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _route_x() -> tuple[RoutePolyline, tuple[RouteLaneRecord, ...]]:
    """A 100 m route along +X with a 4 m lane; the historical fixture shape."""

    centerline = RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    lane = RouteLaneRecord("lane-0", box(-1.75, -2.0, 100.0, 2.0), centerline)
    return centerline, (lane,)


def _route_y() -> tuple[RoutePolyline, tuple[RouteLaneRecord, ...]]:
    """The same road rotated 90 degrees: a 100 m route along +Y.

    The lane is still 4 m wide, but its world-frame Y extent is now its
    *length*. Any width measurement taken from the polygon bounding box
    returns 100 m here instead of 4 m.
    """

    centerline = RoutePolyline(((0.0, 0.0, 0.0), (0.0, 100.0, 0.0)))
    lane = RouteLaneRecord("lane-0", box(-2.0, -1.75, 2.0, 100.0), centerline)
    return centerline, (lane,)


def _ego(position=(0.0, 0.0), velocity=(5.0, 0.0), heading=0.0) -> ActorSnapshot:
    footprint = box(
        position[0] - EGO_LENGTH_M / 2.0,
        position[1] - 0.5,
        position[0] + EGO_LENGTH_M / 2.0,
        position[1] + 0.5,
    )
    if heading:
        footprint = affinity.rotate(footprint, heading, origin=Point(position), use_radians=True)
    return ActorSnapshot(
        actor_id="ego",
        actor_class=ActorClass.VEHICLE,
        position_xy=position,
        position_z=0.0,
        heading_rad=heading,
        velocity_xy=velocity,
        footprint=footprint,
        live_lane_id="lane-0",
        configured_speed_cap_mps=20.0,
    )


def _feature(
    feature_id: str,
    geometry,
    feature_class: MapFeatureClass = MapFeatureClass.LANE_MARKING_SOLID,
) -> MapFeatureRecord:
    return MapFeatureRecord(
        feature_id=feature_id,
        feature_class=feature_class,
        geometry=geometry,
        elevation_m=0.0,
    )


def _control(
    group_id: str,
    route_s_m: float,
    *,
    control_type: ApproachControl = ApproachControl.SIGNAL,
    lanes: tuple[str, ...] = ("lane-0",),
    physical_ids: tuple[str, ...] = ("light-0",),
) -> TrafficControlRecord:
    return TrafficControlRecord(
        control_group_id=group_id,
        control_type=control_type,
        controlled_lane_ids=lanes,
        movement_key=MovementKey("lane-0", "node", "lane-0"),
        control_line=LineString(((route_s_m, -2.0), (route_s_m, 2.0))),
        route_s_m=route_s_m,
        elevation_m=0.0,
        physical_control_ids=physical_ids,
    )


def _ctx(
    step: int,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    route: RoutePolyline,
    lanes: tuple[RouteLaneRecord, ...],
    *,
    zones: tuple[ConflictZoneRecord, ...] = (),
    features: dict[str, MapFeatureRecord] | None = None,
    controls: tuple[TrafficControlRecord, ...] = (),
    signals: dict[str, str] | None = None,
    memory: RulebookMemory | None = None,
) -> CausalSceneContext:
    cache = EpisodeCache(
        "scene",
        TaskRouteRecord("scene", ("lane-0",), "test", "v1", "geometry"),
        conflict_zones={zone.zone_id: zone for zone in zones},
        map_feature_catalog=features or {},
        traffic_control_catalog=controls,
        route_lanes=lanes,
        route_polyline=route,
    )
    snapshot = EnvSnapshot(
        "scene",
        step,
        step * 0.1,
        ego,
        (ego, *actors),
        (),
        frozenset(),
        signals or {},
        mission_snapshot=_mission_snapshot(route, ego, step),
    )
    return CausalSceneContext(cache, snapshot, memory or RulebookMemory(), route)


def _builder(route, lanes, **kwargs) -> PerceptionBoundedSemanticBatchBuilder:
    return PerceptionBoundedSemanticBatchBuilder(
        route=route, route_lanes=lanes, brake_mps2=4.0, **kwargs
    )


def _all_visible(monkeypatch, ids: frozenset[str] | None = None) -> None:
    monkeypatch.setattr(
        causal_semantic,
        "first_hit_lidar_sweep",
        lambda vehicle: SimpleNamespace(
            actor_ids=ids if ids is not None else frozenset({"other", "static", "vru"})
        ),
    )


def _no_signals_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        causal_semantic,
        "mapped_signal_visibility",
        lambda vehicle, ids, **_kwargs: {i: False for i in ids},
    )


def _signals_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        causal_semantic,
        "mapped_signal_visibility",
        lambda vehicle, ids, **_kwargs: {i: True for i in ids},
    )


# --------------------------------------------------------------------------
# REQ-001 — lane width (TEST-001, TEST-002)
# --------------------------------------------------------------------------


def test_001_lane_width_matches_the_analytic_width_on_an_x_aligned_lane(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    builder = _builder(route, lanes)

    assert builder._lane_width("lane-0") == pytest.approx(LANE_WIDTH_M, abs=1e-6)


def test_002_lane_width_is_invariant_to_a_ninety_degree_map_rotation(monkeypatch) -> None:
    """REQ-001: the world-frame Y extent of a north-south lane is its length."""

    route, lanes = _route_y()
    _all_visible(monkeypatch, frozenset())
    builder = _builder(route, lanes)

    assert builder._lane_width("lane-0") == pytest.approx(LANE_WIDTH_M, abs=1e-6)


# --------------------------------------------------------------------------
# REQ-002 — nearest boundary (TEST-003)
# --------------------------------------------------------------------------


def _three_left_boundaries() -> dict[str, MapFeatureRecord]:
    return {
        "near": _feature("near", LineString(((-20.0, 1.0), (20.0, 1.0)))),
        "mid": _feature(
            "mid", LineString(((-20.0, 3.0), (20.0, 3.0))), MapFeatureClass.LANE_MARKING_DASHED
        ),
        "far": _feature(
            "far", LineString(((-20.0, 7.0), (20.0, 7.0))), MapFeatureClass.ROAD_BOUNDARY
        ),
    }


@pytest.mark.parametrize("order", [("near", "mid", "far"), ("far", "mid", "near")])
def test_003_nearest_boundary_wins_regardless_of_catalog_order(monkeypatch, order) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    source = _three_left_boundaries()
    features = {key: source[key] for key in order}
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features=features))

    # ego footprint half-width 0.5, nearest boundary at y=1.0 -> clearance 0.5
    assert batch.lane_road[1] == pytest.approx(0.5 / 50.0, abs=1e-6)
    assert batch.lane_road[3:7].tolist() == [1.0, 0.0, 0.0, 0.0]  # solid


# --------------------------------------------------------------------------
# REQ-003 — signed clearance (TEST-004)
# --------------------------------------------------------------------------


def test_004_overlapping_boundary_reports_negative_penetration(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    crossing = {"crossing": _feature("crossing", LineString(((-20.0, 0.2), (20.0, 0.2))))}
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features=crossing))

    # footprint spans y in [-0.5, 0.5]; the part beyond y=0.2 is 0.3 m deep
    assert batch.lane_road[1] == pytest.approx(-0.3 / 50.0, abs=1e-6)


def test_004b_touching_boundary_reports_exactly_zero(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    touching = {"touching": _feature("touching", LineString(((-20.0, 0.5), (20.0, 0.5))))}
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features=touching))

    assert batch.lane_road[1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# REQ-004 — boundary side (TEST-005)
# --------------------------------------------------------------------------


def test_005_boundary_side_follows_the_nearest_point_not_the_representative_point(
    monkeypatch,
) -> None:
    """An L-shaped boundary whose representative point sits on the right while
    its nearest point sits on the left."""

    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    shape = LineString(((-0.5, 0.9), (1.5, 0.9), (1.5, -9.0), (6.0, -9.0)))
    assert shape.representative_point().y < 0.0  # the trap the old code fell into
    builder = _builder(route, lanes)

    batch = builder.build(
        _Vehicle(),
        _ctx(0, _ego(), (), route, lanes, features={"L": _feature("L", shape)}),
    )

    assert batch.lane_road[1] == pytest.approx(0.4 / 50.0, abs=1e-6)  # left, 0.4 m
    assert batch.lane_road[2] == pytest.approx(1.0, abs=1e-6)  # right: nothing


# --------------------------------------------------------------------------
# REQ-005 — local static dimensions (TEST-006)
# --------------------------------------------------------------------------


def test_006_static_map_feature_dimensions_use_a_local_window(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    rail = _feature(
        "rail", LineString(((-100.0, 9.0), (100.0, 9.0))), MapFeatureClass.ROAD_BOUNDARY
    )
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features={"rail": rail}))

    expected = 2.0 * math.sqrt(10.0**2 - 9.0**2) / 10.0
    assert batch.static_mask[0] == 1.0
    # the local window is a segmented buffer, so the chord is marginally short
    assert batch.static[0, 4] == pytest.approx(expected, abs=1e-2)


# --------------------------------------------------------------------------
# REQ-006 — no world-frame leak (TEST-007, TEST-008)
# --------------------------------------------------------------------------


def _rotate_scene(angle_rad: float):
    """Rigidly rotate route, lane and a boundary feature about the origin."""

    def rot(x, y):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return (c * x - s * y, s * x + c * y)

    centerline = RoutePolyline(((0.0, 0.0, 0.0), (*rot(100.0, 0.0), 0.0)))
    lane_polygon = affinity.rotate(
        box(-1.75, -2.0, 100.0, 2.0), angle_rad, origin=Point(0.0, 0.0), use_radians=True
    )
    lane = RouteLaneRecord("lane-0", lane_polygon, centerline)
    rail = affinity.rotate(
        LineString(((-20.0, 6.0), (60.0, 6.0))),
        angle_rad,
        origin=Point(0.0, 0.0),
        use_radians=True,
    )
    ego = _ego(velocity=rot(5.0, 0.0), heading=angle_rad)
    return centerline, (lane,), ego, {"rail": _feature("rail", rail, MapFeatureClass.ROAD_BOUNDARY)}


def test_007_static_tokens_are_invariant_under_a_rigid_scene_rotation(monkeypatch) -> None:
    """REQ-006: with a hardcoded zero heading for map features the emitted
    sin/cos pair is ``(-sin psi_ego, cos psi_ego)`` — the ego world heading."""

    _all_visible(monkeypatch, frozenset())
    batches = []
    for angle in (0.0, math.pi / 3.0):
        route, lanes, ego, features = _rotate_scene(angle)
        builder = _builder(route, lanes)
        batches.append(builder.build(_Vehicle(), _ctx(0, ego, (), route, lanes, features=features)))

    assert batches[0].static_mask.tolist() == batches[1].static_mask.tolist()
    assert np.allclose(batches[0].static, batches[1].static, atol=1e-5)


def test_008_static_map_feature_heading_is_the_local_tangent(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    # A boundary running at +45 degrees near the ego.
    diagonal = _feature(
        "diag", LineString(((0.0, 4.0), (20.0, 24.0))), MapFeatureClass.ROAD_BOUNDARY
    )
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features={"d": diagonal}))

    assert batch.static[0, 2] == pytest.approx(math.sin(math.pi / 4.0), abs=1e-3)
    assert batch.static[0, 3] == pytest.approx(math.cos(math.pi / 4.0), abs=1e-3)


# --------------------------------------------------------------------------
# REQ-007 — no Rulebook latch (TEST-009)
# --------------------------------------------------------------------------


def _zone_scene():
    route, lanes = _route_x()
    ego = _ego()
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    zone = ConflictZoneRecord(
        "zone-a",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        10.0,
        20.0,
        0.0,
    )
    return route, lanes, ego, other, zone


def test_009_a_populated_rulebook_memory_cannot_change_the_observation(monkeypatch) -> None:
    route, lanes, ego, other, zone = _zone_scene()
    _all_visible(monkeypatch, frozenset({"other"}))
    poisoned = RulebookMemory(
        crosswalk_illegal_entries=frozenset({("other", "zone-a"), ("zone-a", "other")}),
        vehicle_yield_illegal_entries=frozenset({("other", "zone-a"), ("zone-a", "other")}),
        preexisting_ego_occupancy_zone_ids=frozenset({"zone-a"}),
        active_signal_group_id="group-x",
        resolved_signal_group_ids=frozenset({"group-x"}),
        dashed_line_timer_s=1.7,
        stop_best_timer_s=0.9,
    )

    clean = _builder(route, lanes).build(
        _Vehicle(), _ctx(0, ego, (other,), route, lanes, zones=(zone,))
    )
    dirty = _builder(route, lanes).build(
        _Vehicle(), _ctx(0, ego, (other,), route, lanes, zones=(zone,), memory=poisoned)
    )

    assert np.array_equal(clean.interactions, dirty.interactions)
    assert np.array_equal(clean.controls, dirty.controls)
    assert np.array_equal(clean.context_history, dirty.context_history)


# --------------------------------------------------------------------------
# REQ-008 — pre-existing occupancy reconstructed (TEST-010, TEST-011)
# --------------------------------------------------------------------------


def test_010_preexisting_occupancy_is_derived_from_geometry(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset({"other"}))
    zone = ConflictZoneRecord(
        "zone-a",
        box(-5.0, -2.0, 5.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        0.0,
        5.0,
        0.0,
    )
    inside = _ego(position=(0.0, 0.0))
    other = _actor("other", (4.0, 0.0), (0.0, 0.0))
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, inside, (other,), route, lanes, zones=(zone,)))

    assert batch.interactions_mask[0] == 1.0
    assert batch.interactions[0, causal_semantic.INTERACTION_PREEXISTING_INDEX] == 1.0


def test_011_a_zone_entered_later_is_not_preexisting_and_reset_clears_it(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset({"other"}))
    zone = ConflictZoneRecord(
        "zone-a",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        10.0,
        20.0,
        0.0,
    )
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    builder = _builder(route, lanes)

    builder.build(
        _Vehicle(), _ctx(0, _ego(position=(0.0, 0.0)), (other,), route, lanes, zones=(zone,))
    )
    entered = builder.build(
        _Vehicle(), _ctx(1, _ego(position=(15.0, 0.0)), (other,), route, lanes, zones=(zone,))
    )

    index = causal_semantic.INTERACTION_PREEXISTING_INDEX
    assert entered.interactions[0, index] == 0.0

    builder.reset()
    assert builder._preexisting_zone_occupancy == {}


# --------------------------------------------------------------------------
# REQ-009 — front-bumper distance (TEST-012, TEST-013)
# --------------------------------------------------------------------------


def test_012_control_distance_is_measured_from_the_front_bumper(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    control = _control("g0", 30.0, control_type=ApproachControl.STOP, physical_ids=())
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, controls=(control,)))

    expected = (30.0 - EGO_LENGTH_M / 2.0) / 80.0
    assert batch.controls_mask[0] == 1.0
    assert batch.controls[0, 2] == pytest.approx(expected, abs=1e-6)


def test_013_context_row_and_control_token_agree_on_the_distance(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    control = _control("g0", 30.0, control_type=ApproachControl.STOP, physical_ids=())
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, controls=(control,)))

    assert batch.context_history[-1, 22] == pytest.approx(batch.controls[0, 2], abs=1e-6)


# --------------------------------------------------------------------------
# REQ-010 / REQ-011 — control type and signal state (TEST-014, TEST-015)
# --------------------------------------------------------------------------


def test_014_an_occluded_signal_is_not_encoded_as_a_stop_control(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    control = _control("g0", 30.0)
    builder = _builder(route, lanes)

    batch = builder.build(
        _Vehicle(),
        _ctx(0, _ego(), (), route, lanes, controls=(control,), signals={"light-0": "RED"}),
    )

    row = batch.context_history[-1]
    assert row[15:17].tolist() == [1.0, 0.0]  # signal, not stop
    assert row[17:22].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]  # unknown
    assert batch.controls[0, 5:7].tolist() == [1.0, 0.0]  # token type: signal
    assert batch.controls[0, 7:12].tolist() == [0.0] * 5  # state payload zeroed


def test_015_a_stop_control_sets_the_not_signal_state(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    control = _control("g0", 30.0, control_type=ApproachControl.STOP, physical_ids=())
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, controls=(control,)))

    assert batch.controls[0, 5:7].tolist() == [0.0, 1.0]  # stop
    assert batch.controls[0, 7:12].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]  # not-signal at index 4


# --------------------------------------------------------------------------
# REQ-012 — signed route lateral offset (TEST-016)
# --------------------------------------------------------------------------


def test_016_route_lateral_offset_keeps_the_left_right_sign(monkeypatch) -> None:
    route, lanes = _route_x()
    left = _actor("other", (15.0, 3.0), (0.0, 0.0))
    right = _actor("other", (15.0, -3.0), (0.0, 0.0))
    _all_visible(monkeypatch, frozenset({"other"}))

    batch_left = _builder(route, lanes).build(_Vehicle(), _ctx(0, _ego(), (left,), route, lanes))
    batch_right = _builder(route, lanes).build(_Vehicle(), _ctx(0, _ego(), (right,), route, lanes))

    index = causal_semantic.DYNAMIC_ROUTE_LATERAL_INDEX
    slot_l = int(np.flatnonzero(batch_left.dynamic_mask[:, -1])[0])
    slot_r = int(np.flatnonzero(batch_right.dynamic_mask[:, -1])[0])
    assert batch_left.dynamic[slot_l, -1, index] == pytest.approx(3.0 / 50.0, abs=1e-6)
    assert batch_right.dynamic[slot_r, -1, index] == pytest.approx(-3.0 / 50.0, abs=1e-6)


# --------------------------------------------------------------------------
# REQ-013 — interaction ranking (TEST-017, TEST-018)
# --------------------------------------------------------------------------


def _many_zones(count: int) -> tuple[ConflictZoneRecord, ...]:
    return tuple(
        ConflictZoneRecord(
            f"zone-{index:02d}",
            box(30.0 + index, -2.0, 32.0 + index, 2.0),
            MovementKey("lane-0", "node", "lane-0"),
            MovementKey("lane-0", "node", "lane-0"),
            30.0 + index,
            32.0 + index,
            0.0,
        )
        for index in range(count)
    )


def test_017_a_critical_interaction_survives_alphabetical_overflow(monkeypatch) -> None:
    """The ego sits inside ``zone-99``, which sorts last by id; with nine
    candidates it must still occupy a token."""

    route, lanes = _route_x()
    occupied = ConflictZoneRecord(
        "zone-99",
        box(-5.0, -2.0, 5.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        0.0,
        5.0,
        0.0,
    )
    zones = (*_many_zones(8), occupied)
    actors = tuple(_actor(f"a{index}", (31.0 + index, 0.0)) for index in range(8))
    near = _actor("a8", (3.0, 0.0))
    everyone = (*actors, near)
    _all_visible(monkeypatch, frozenset(actor.actor_id for actor in everyone))
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), everyone, route, lanes, zones=zones))

    assert batch.interactions_mask.sum() == 8.0
    ego_inside = batch.interactions[:, causal_semantic.INTERACTION_EGO_INSIDE_INDEX]
    assert ego_inside.sum() >= 1.0


def test_018_interaction_ranking_is_stable_under_input_permutation(monkeypatch) -> None:
    route, lanes = _route_x()
    zones = _many_zones(4)
    actors = tuple(_actor(f"a{index}", (31.0 + index, 0.0)) for index in range(4))
    _all_visible(monkeypatch, frozenset(actor.actor_id for actor in actors))

    first = _builder(route, lanes).build(
        _Vehicle(), _ctx(0, _ego(), actors, route, lanes, zones=zones)
    )
    second = _builder(route, lanes).build(
        _Vehicle(),
        _ctx(0, _ego(), tuple(reversed(actors)), route, lanes, zones=tuple(reversed(zones))),
    )

    assert np.array_equal(first.interactions, second.interactions)


# --------------------------------------------------------------------------
# REQ-014 — overflow diagnostics (TEST-019)
# --------------------------------------------------------------------------


def test_019_overflow_diagnostics_cover_all_four_groups(monkeypatch) -> None:
    route, lanes = _route_x()
    zones = _many_zones(9)
    actors = tuple(_actor(f"a{index:02d}", (31.0 + index, 0.0)) for index in range(9))
    features = {
        f"rail-{index}": _feature(
            f"rail-{index}",
            LineString(((-20.0, 5.0 + index), (20.0, 5.0 + index))),
            MapFeatureClass.ROAD_BOUNDARY,
        )
        for index in range(10)
    }
    controls = tuple(_control(f"g{index}", 10.0 + index, physical_ids=()) for index in range(10))
    _all_visible(monkeypatch, frozenset(actor.actor_id for actor in actors))
    _no_signals_visible(monkeypatch)
    builder = _builder(route, lanes)

    builder.build(
        _Vehicle(),
        _ctx(0, _ego(), actors, route, lanes, zones=zones, features=features, controls=controls),
    )

    diagnostics = builder.diagnostics
    for group in ("dynamic", "static", "controls", "interactions"):
        assert group in diagnostics.total_candidates
        assert group in diagnostics.selected_candidates
        assert group in diagnostics.capacity_dropped
    assert diagnostics.capacity_dropped["controls"] == 2
    assert diagnostics.capacity_dropped["static"] == 2


# --------------------------------------------------------------------------
# REQ-015 — real crosswalk entry/exit (TEST-020)
# --------------------------------------------------------------------------


def test_020_crosswalk_entry_and_exit_are_the_true_route_limits(monkeypatch) -> None:
    route, lanes = _route_x()
    crosswalk = _feature("cw", box(20.0, -3.0, 26.0, 3.0), MapFeatureClass.CROSSWALK)
    vru = replace(
        _actor("vru", (23.0, 1.0), (0.0, 0.0)),
        actor_class=ActorClass.PEDESTRIAN,
        configured_speed_cap_mps=None,
    )
    _all_visible(monkeypatch, frozenset({"vru"}))
    builder = _builder(route, lanes)

    batch = builder.build(
        _Vehicle(), _ctx(0, _ego(), (vru,), route, lanes, features={"cw": crosswalk})
    )

    assert batch.interactions_mask[0] == 1.0
    entry, exit_ = batch.interactions[0, 2] * 50.0, batch.interactions[0, 3] * 50.0
    assert entry == pytest.approx(20.0 - EGO_LENGTH_M / 2.0, abs=1e-3)
    assert exit_ == pytest.approx(26.0 - EGO_LENGTH_M / 2.0, abs=1e-3)
    assert exit_ > entry


# --------------------------------------------------------------------------
# REQ-016 — control ranking (TEST-021)
# --------------------------------------------------------------------------


def test_021_a_control_on_the_ego_lane_ranks_before_a_nearer_foreign_control(
    monkeypatch,
) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    foreign = _control("foreign", 20.0, lanes=("lane-9",), physical_ids=())
    mine = _control("mine", 60.0, physical_ids=())
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, controls=(foreign, mine)))

    assert batch.controls[0, causal_semantic.CONTROL_GOVERNS_INDEX] == 1.0


# --------------------------------------------------------------------------
# REQ-025 — bounded control horizon (TEST-030)
# --------------------------------------------------------------------------


def test_030_a_control_beyond_the_horizon_is_not_exposed_via_approach_control(
    monkeypatch,
) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset({"other"}))
    distant = _control("far", 200.0, control_type=ApproachControl.STOP, physical_ids=())
    route_long = RoutePolyline(((0.0, 0.0, 0.0), (300.0, 0.0, 0.0)))
    lanes_long = (RouteLaneRecord("lane-0", box(-1.75, -2.0, 300.0, 2.0), route_long),)
    zone = ConflictZoneRecord(
        "zone-a",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        10.0,
        20.0,
        0.0,
    )
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    builder = _builder(route_long, lanes_long)

    batch = builder.build(
        _Vehicle(),
        _ctx(0, _ego(), (other,), route_long, lanes_long, zones=(zone,), controls=(distant,)),
    )

    start = causal_semantic.INTERACTION_OTHER_APPROACH_CONTROL_SLICE.start
    # Inside the horizon the local map legitimately establishes "no control";
    # what must never happen is the distant stop leaking in as `stop`.
    assert batch.interactions[0, start + 1] == 0.0  # not `stop`
    assert batch.interactions[0, start + 0] == 1.0  # none


# --------------------------------------------------------------------------
# REQ-026 — continuity contiguity (TEST-031)
# --------------------------------------------------------------------------


def test_031_continuity_indicators_reset_after_a_step_gap(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    _no_signals_visible(monkeypatch)
    control = _control("g0", 30.0, control_type=ApproachControl.STOP, physical_ids=())
    dashed = {
        "d": _feature(
            "d", LineString(((-20.0, 0.2), (20.0, 0.2))), MapFeatureClass.LANE_MARKING_DASHED
        )
    }
    builder = _builder(route, lanes)

    builder.build(
        _Vehicle(), _ctx(0, _ego(), (), route, lanes, controls=(control,), features=dashed)
    )
    contiguous = builder.build(
        _Vehicle(), _ctx(1, _ego(), (), route, lanes, controls=(control,), features=dashed)
    )
    assert contiguous.context_history[-1, 12] == 1.0
    assert contiguous.context_history[-1, 14] == 1.0

    gapped = builder.build(
        _Vehicle(), _ctx(5, _ego(), (), route, lanes, controls=(control,), features=dashed)
    )
    assert gapped.context_history[-1, 12] == 0.0
    assert gapped.context_history[-1, 14] == 0.0


# --------------------------------------------------------------------------
# REQ-027 — conflict-zone geometry is actor independent (TEST-032)
# --------------------------------------------------------------------------


def test_032_conflict_zone_geometry_does_not_depend_on_the_triggering_actor(
    monkeypatch,
) -> None:
    """Pins the property §4.3 of the ExecPlan relies on: the zone polygon is a
    function of the ego corridor and the approach lane only."""

    from thesis_rl.rulebook.v2.geometry.conflict_zones import (
        MovementCorridor,
        build_vehicle_conflict_zone_candidates,
    )
    from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation

    ego_lane = RouteLaneRecord(
        "lane-0", box(-2.0, -2.0, 100.0, 2.0), RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    )
    other_lane = RouteLaneRecord(
        "lane-1",
        box(40.0, -50.0, 44.0, 50.0),
        RoutePolyline(((42.0, -50.0, 0.0), (42.0, 50.0, 0.0))),
    )
    corridors = {
        name: MovementCorridor(
            movement_key=MovementKey(lane.lane_id, "node", lane.lane_id),
            polygon=lane.polygon_xy,
            elevation_at_xy=PolylineElevation(lane.centerline.points_xyz),
        )
        for name, lane in (("ego", ego_lane), ("other", other_lane))
    }

    first = build_vehicle_conflict_zone_candidates(
        scenario_id="scene", ego_corridor=corridors["ego"], other_corridor=corridors["other"]
    )
    second = build_vehicle_conflict_zone_candidates(
        scenario_id="scene", ego_corridor=corridors["ego"], other_corridor=corridors["other"]
    )

    assert [candidate.polygon.wkt for candidate in first] == [
        candidate.polygon.wkt for candidate in second
    ]
    assert first, "the fixture must actually produce a conflict zone"


# --------------------------------------------------------------------------
# REQ-023 — a non-projectable static actor degrades its token only (TEST-029)
# --------------------------------------------------------------------------


def test_029_unprojectable_static_actor_is_dropped_without_failing_the_build(
    monkeypatch,
) -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 6.0)))
    lanes = (RouteLaneRecord("lane-0", box(-1.75, -2.0, 11.75, 2.0), route),)
    stray = replace(
        _actor("static", (10.0, 0.0)),
        actor_class=ActorClass.STATIC_COLLIDABLE,
        position_z=2.9,
    )
    _all_visible(monkeypatch, frozenset({"static"}))
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (stray,), route, lanes))

    assert batch.static_mask.sum() == 0.0
    assert np.isfinite(batch.static).all()


# --------------------------------------------------------------------------
# REQ-022 — dead code (TEST-028)
# --------------------------------------------------------------------------


def test_028_the_builder_exposes_no_unreachable_v12_helpers() -> None:
    for name in ("_build_compliance_v12", "_active_dashed_key", "_active_dashed_feature_id"):
        assert not hasattr(CausalSemanticBatchBuilder, name), name
        assert not hasattr(PerceptionBoundedSemanticBatchBuilder, name), name


def test_028b_dashed_feature_lookup_filters_elevation_and_takes_the_nearest(
    monkeypatch,
) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    upper = MapFeatureRecord(
        feature_id="upper",
        feature_class=MapFeatureClass.LANE_MARKING_DASHED,
        geometry=LineString(((-20.0, 0.0), (20.0, 0.0))),
        elevation_m=9.0,
    )
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes, features={"u": upper}))

    assert batch.context_history[-1, 11] == 0.0


# --------------------------------------------------------------------------
# REQ-021 — static taxonomy (TEST-027)
# --------------------------------------------------------------------------


def test_027_static_taxonomy_separates_a_detected_obstacle_from_a_map_boundary(
    monkeypatch,
) -> None:
    route, lanes = _route_x()
    obstacle = replace(_actor("static", (12.0, 0.0)), actor_class=ActorClass.STATIC_COLLIDABLE)
    rail = _feature("rail", LineString(((-20.0, 6.0), (20.0, 6.0))), MapFeatureClass.ROAD_BOUNDARY)
    _all_visible(monkeypatch, frozenset({"static"}))
    builder = _builder(route, lanes)

    batch = builder.build(
        _Vehicle(), _ctx(0, _ego(), (obstacle,), route, lanes, features={"rail": rail})
    )

    types = batch.static[:2, causal_semantic.STATIC_TYPE_SLICE]
    assert batch.static_mask[:2].tolist() == [1.0, 1.0]
    assert not np.array_equal(types[0], types[1])


# --------------------------------------------------------------------------
# REQ-028 — every static taxonomy slot is reachable (TEST-034..TEST-036)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subclass", "expected_index"),
    [
        (StaticSubclass.TRAFFIC_CONE, causal_semantic.STATIC_TYPE_TRAFFIC_CONE),
        (StaticSubclass.TRAFFIC_BARRIER, causal_semantic.STATIC_TYPE_TRAFFIC_BARRIER),
        (StaticSubclass.TRAFFIC_WARNING, causal_semantic.STATIC_TYPE_OTHER_OBSTACLE),
        (StaticSubclass.OTHER, causal_semantic.STATIC_TYPE_OTHER_OBSTACLE),
        (None, causal_semantic.STATIC_TYPE_OTHER_OBSTACLE),
    ],
)
def test_034_static_subclass_selects_its_taxonomy_slot(
    monkeypatch, subclass, expected_index
) -> None:
    route, lanes = _route_x()
    obstacle = replace(
        _actor("static", (12.0, 0.0)),
        actor_class=ActorClass.STATIC_COLLIDABLE,
        static_subclass=subclass,
    )
    _all_visible(monkeypatch, frozenset({"static"}))
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (obstacle,), route, lanes))

    expected = np.zeros(5, dtype=np.float32)
    expected[expected_index] = 1.0
    assert batch.static_mask[0] == 1.0
    assert np.array_equal(batch.static[0, causal_semantic.STATIC_TYPE_SLICE], expected)


def test_035_every_static_taxonomy_slot_is_reachable(monkeypatch) -> None:
    """No slot may be dead: the previous 5-way one-hot had two constant columns."""

    route, lanes = _route_x()
    actors = tuple(
        replace(
            _actor(f"static_{index}", (10.0 + 3.0 * index, 0.0)),
            actor_class=ActorClass.STATIC_COLLIDABLE,
            static_subclass=subclass,
        )
        for index, subclass in enumerate(
            (StaticSubclass.TRAFFIC_CONE, StaticSubclass.TRAFFIC_BARRIER, StaticSubclass.OTHER)
        )
    )
    features = {
        "rail": _feature(
            "rail", LineString(((-20.0, 6.0), (20.0, 6.0))), MapFeatureClass.ROAD_BOUNDARY
        ),
        "verge": _feature(
            "verge",
            LineString(((-20.0, -6.0), (20.0, -6.0))),
            MapFeatureClass.OTHER_NON_DRIVABLE,
        ),
    }
    _all_visible(monkeypatch, frozenset(actor.actor_id for actor in actors))
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), actors, route, lanes, features=features))

    visible = batch.static[batch.static_mask == 1.0, causal_semantic.STATIC_TYPE_SLICE]
    observed = {int(np.argmax(row)) for row in visible}
    assert observed == {0, 1, 2, 3, 4}


def test_036_static_subclass_is_rejected_on_non_static_actors() -> None:
    """The refinement must not silently attach to a vehicle or a pedestrian."""

    with pytest.raises(ValueError, match="STATIC_COLLIDABLE"):
        actor_snapshot_from_payload(
            {
                "actor_id": "vehicle",
                "actor_class": ActorClass.VEHICLE,
                "position_xy": (0.0, 0.0),
                "position_z": 0.0,
                "heading_rad": 0.0,
                "velocity_xy": (0.0, 0.0),
                "length_m": 4.0,
                "width_m": 2.0,
                "configured_speed_cap_mps": 10.0,
                "static_subclass": StaticSubclass.TRAFFIC_CONE,
            }
        )


# --------------------------------------------------------------------------
# REQ-017..020 — OBS-V1.3 dimensions (TEST-022..TEST-026)
# --------------------------------------------------------------------------


def test_022_to_025_group_shapes_match_obs_v13(monkeypatch) -> None:
    route, lanes = _route_x()
    _all_visible(monkeypatch, frozenset())
    builder = _builder(route, lanes)

    batch = builder.build(_Vehicle(), _ctx(0, _ego(), (), route, lanes))

    assert batch.controls.shape == (8, 15)
    assert batch.context_history.shape == (21, 23)
    assert batch.interactions.shape == (8, 33)
    assert batch.lane_road.shape == (12,)


def test_026_flat_dimension_and_token_count() -> None:
    from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV12

    assert SemanticObservationSchemaV12.flat_dim == 3009
    assert SemanticObservationSchemaV12.raw_token_count == 143


# --------------------------------------------------------------------------
# DEC-008 — semantic_v2 must stay frozen (TEST-033)
# --------------------------------------------------------------------------


def test_033_legacy_semantic_v2_builder_is_unchanged() -> None:
    """The legacy builder keeps the OBS-V1.1 contract: 14 lane/road values, a
    35-wide interaction token, and the historical bounding-box lane width."""

    route, lanes = _route_y()
    ego = _ego()
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)

    batch = builder.build(_Vehicle(), _ctx(0, ego, (), route, lanes))

    assert batch.lane_road.shape == (14,)
    assert batch.interactions.shape == (8, 35)
    assert builder._lane_width("lane-0") == pytest.approx(101.75)
