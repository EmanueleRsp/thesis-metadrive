from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.envs.observations.causal_semantic import (
    CausalSemanticBatchBuilder,
    CausalSemanticObservationError,
)
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ConflictZoneRecord,
    EpisodeCache,
    EnvSnapshot,
    MapFeatureClass,
    MapFeatureRecord,
    MovementKey,
    RulebookMemory,
    TaskRouteRecord,
)


class _Vehicle:
    steering = 0.1
    throttle_brake = 0.2
    last_velocity = (0.0, 0.0)
    yaw_rate = 0.0


def _route(offset_x: float = 0.0) -> tuple[RoutePolyline, tuple[RouteLaneRecord, ...]]:
    centerline = RoutePolyline(((offset_x, 0.0, 0.0), (offset_x + 100.0, 0.0, 0.0)))
    lane = RouteLaneRecord("lane-0", box(offset_x - 1.75, -2.0, offset_x + 100.0, 2.0), centerline)
    return centerline, (lane,)


def _actor(
    actor_id: str,
    position: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    *,
    lane_id: str | None = "lane-0",
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id=actor_id,
        actor_class=ActorClass.VEHICLE,
        position_xy=position,
        position_z=0.0,
        heading_rad=0.0,
        velocity_xy=velocity,
        footprint=box(position[0] - 1.0, position[1] - 0.5, position[0] + 1.0, position[1] + 0.5),
        live_lane_id=lane_id,
        configured_speed_cap_mps=20.0,
    )


def _context(
    step: int,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    route: RoutePolyline,
    lanes: tuple[RouteLaneRecord, ...],
    *,
    zones: tuple[ConflictZoneRecord, ...] = (),
    map_features: dict[str, MapFeatureRecord] | None = None,
) -> CausalSceneContext:
    task_route = TaskRouteRecord("scene", ("lane-0",), "test", "v1", "geometry")
    cache = EpisodeCache(
        "scene",
        task_route,
        conflict_zones={zone.zone_id: zone for zone in zones},
        map_feature_catalog=map_features or {},
        route_lanes=lanes,
        route_polyline=route,
    )
    snapshot = EnvSnapshot("scene", step, step * 0.1, ego, (ego, *actors), (), frozenset(), {})
    return CausalSceneContext(cache, snapshot, RulebookMemory())


def test_builder_emits_schema_groups_without_legacy_vector_recycling() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other = _actor("other", (15.0, 0.0), (-2.0, 0.0))
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)

    batch = builder.build(_Vehicle(), _context(0, ego, (other,), route, lanes))

    assert batch.dynamic.shape == (16, 5, 22)
    assert batch.route.shape == (10, 7)
    assert batch.interactions.shape == (8, 35)
    assert batch.ego_history_mask.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert np.isfinite(batch.dynamic).all()
    assert np.isfinite(batch.temporal).all()


def test_actor_history_is_causal_and_slot_persists_then_expires() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    actor = _actor("other", (15.0, 0.0), (0.0, 0.0))
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)

    first = builder.build(_Vehicle(), _context(0, ego, (actor,), route, lanes))
    first_slot = int(np.flatnonzero(first.dynamic_mask[:, -1])[0])
    actor_step = replace(actor, position_xy=(16.0, 0.0))
    second = builder.build(_Vehicle(), _context(1, ego, (actor_step,), route, lanes))
    assert int(np.flatnonzero(second.dynamic_mask[:, -1])[0]) == first_slot
    assert second.dynamic_mask[first_slot, -2] == 1.0

    for step in range(2, 7):
        absent = builder.build(_Vehicle(), _context(step, ego, (), route, lanes))
    assert absent.dynamic_mask[first_slot].sum() == 0.0
    assert builder._slot_actor.get(first_slot) is None


def test_new_actor_has_no_backfilled_history() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)
    builder.build(_Vehicle(), _context(0, ego, (), route, lanes))

    newcomer = _actor("new", (20.0, 0.0), (0.0, 0.0))
    batch = builder.build(_Vehicle(), _context(1, ego, (newcomer,), route, lanes))
    slot = int(np.flatnonzero(batch.dynamic_mask[:, -1])[0])
    assert batch.dynamic_mask[slot].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_cpa_is_computed_from_current_relative_state() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)
    batch = builder.build(_Vehicle(), _context(0, ego, (other,), route, lanes))
    slot = int(np.flatnonzero(batch.dynamic_mask[:, -1])[0])
    assert batch.dynamic[slot, -1, 19] == pytest.approx(1.0)
    assert batch.dynamic[slot, -1, 20] == pytest.approx(1.0)


def test_ambiguous_zone_does_not_emit_pairwise_interaction() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    ambiguous = ConflictZoneRecord(
        "ambiguous",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        None,
        10.0,
        20.0,
        0.0,
    )
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)
    batch = builder.build(_Vehicle(), _context(0, ego, (other,), route, lanes, zones=(ambiguous,)))
    assert batch.interactions_mask.sum() == 0.0


def test_causal_vehicle_conflict_populates_interaction_token() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other = _actor("other", (15.0, 0.0), (0.0, 0.0))
    zone = ConflictZoneRecord(
        "vehicle-zone",
        box(10.0, -2.0, 20.0, 2.0),
        MovementKey("lane-0", "node", "lane-0"),
        MovementKey("lane-0", "node", "lane-0"),
        10.0,
        20.0,
        0.0,
    )
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)
    batch = builder.build(_Vehicle(), _context(0, ego, (other,), route, lanes, zones=(zone,)))
    assert batch.interactions_mask.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.isfinite(batch.interactions[0]).all()


def test_missing_current_ego_field_fails_closed_instead_of_zero_filling() -> None:
    route, lanes = _route()
    ego = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)
    with pytest.raises(CausalSemanticObservationError, match="last_velocity"):
        builder.build(object(), _context(0, ego, (), route, lanes))


def test_unprojectable_dynamic_actor_reports_elevation_diagnostics() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 6.0)))
    lanes = (RouteLaneRecord("lane-0", box(-1.75, -2.0, 11.75, 2.0), route),)
    ego = replace(_actor("ego", (0.0, 0.0), (5.0, 0.0)), position_z=0.0)
    actor = replace(_actor("other", (10.0, 0.0)), position_z=2.9)
    builder = CausalSemanticBatchBuilder(route=route, route_lanes=lanes)

    with pytest.raises(CausalSemanticObservationError) as error:
        builder.build(_Vehicle(), _context(3, ego, (actor,), route, lanes))

    message = str(error.value)
    assert "scenario_id='scene'" in message
    assert "step=3" in message
    assert "actor_id='other'" in message
    assert "actor_ego_vertical_delta_m=2.900000" in message
    assert "nearest_planar_z_m=6.000000" in message
    assert "minimum_route_vertical_delta_m=3.100000" in message
    assert "compatible_route_segment_count=0" in message


def _static_feature(feature_id: str, elevation: float) -> MapFeatureRecord:
    points = ((10.0, 1.0, elevation), (12.0, 1.0, elevation))
    return MapFeatureRecord(
        feature_id=feature_id,
        feature_class=MapFeatureClass.ROAD_BOUNDARY,
        geometry=LineString(tuple((x, y) for x, y, _ in points)),
        elevation_m=elevation,
        elevation_profile_xyz=points,
    )


def _static_fixture(
    features: dict[str, MapFeatureRecord],
) -> tuple[CausalSemanticBatchBuilder, CausalSceneContext]:
    route, lanes = _route()
    ego = replace(_actor("ego", (0.0, 0.0), (5.0, 0.0)), position_z=0.1515)
    return CausalSemanticBatchBuilder(route=route, route_lanes=lanes), _context(
        3, ego, (), route, lanes, map_features=features
    )


def test_route_compatible_static_feature_emits_token_and_mask() -> None:
    builder, context = _static_fixture({"compatible": _static_feature("compatible", 0.1515)})

    batch = builder.build(_Vehicle(), context)

    assert batch.static_mask[0] == 1.0
    assert builder.diagnostics.route_incompatible_static_features == 0


def test_route_incompatible_static_feature_is_omitted_and_masked() -> None:
    builder, context = _static_fixture({"upper": _static_feature("upper", 3.1448)})

    batch = builder.build(_Vehicle(), context)

    assert batch.static_mask.sum() == 0.0
    assert builder.diagnostics.route_incompatible_static_features == 1


def test_mixed_static_features_keep_compatible_feature() -> None:
    compatible = _static_feature("compatible", 0.1515)
    incompatible = _static_feature("upper", 3.1448)
    builder, context = _static_fixture({"upper": incompatible, "compatible": compatible})

    batch = builder.build(_Vehicle(), context)

    assert batch.static_mask[0] == 1.0
    assert batch.static_mask[1] == 0.0
    assert builder.diagnostics.route_incompatible_static_features == 1


def test_all_incompatible_static_features_preserve_shape_and_empty_mask() -> None:
    first = _static_feature("upper-a", 3.1448)
    second = _static_feature("upper-b", 3.1448)
    builder, context = _static_fixture({"upper-a": first, "upper-b": second})

    batch = builder.build(_Vehicle(), context)

    assert batch.static.shape == (8, 13)
    assert batch.static_mask.shape == (8,)
    assert batch.static_mask.sum() == 0.0
    assert builder.diagnostics.route_incompatible_static_features == 2


def test_structural_static_projection_error_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A projection failure that is NOT a vertical-compatibility gap (the
    synthetic monkeypatch below is unrelated to the feature's actual, real
    vertical compatibility) must not be silently swallowed as an
    incompatible-feature skip. It is re-raised as
    ``CausalSemanticObservationError`` (a ``ValueError`` subclass) carrying
    the original failure plus route-projection diagnostics, mirroring the
    same structural-vs-incompatible distinction already established by
    ``_build_static_v12``."""

    builder, context = _static_fixture({"broken": _static_feature("broken", 0.1515)})
    original_project = RoutePolyline.project

    def fail_feature_projection(self, point_xy, *, position_z=None, previous_s_m=None):
        if point_xy == (10.0, 1.0):
            raise ValueError("synthetic structural projection failure")
        return original_project(self, point_xy, position_z=position_z, previous_s_m=previous_s_m)

    monkeypatch.setattr(RoutePolyline, "project", fail_feature_projection)
    with pytest.raises(CausalSemanticObservationError, match="route projection unavailable"):
        builder.build(_Vehicle(), context)


def test_waymo_feature_16_equivalent_geometry_is_omitted() -> None:
    builder, context = _static_fixture({"16": _static_feature("16", 3.1447820165)})

    batch = builder.build(_Vehicle(), context)

    assert batch.static_mask.sum() == 0.0
    assert builder.diagnostics.route_incompatible_static_features == 1


def test_global_translation_preserves_relative_observation() -> None:
    route_a, lanes_a = _route()
    route_b, lanes_b = _route(100.0)
    ego_a = _actor("ego", (0.0, 0.0), (5.0, 0.0))
    other_a = _actor("other", (15.0, 0.0), (0.0, 0.0))
    ego_b = _actor("ego", (100.0, 0.0), (5.0, 0.0))
    other_b = _actor("other", (115.0, 0.0), (0.0, 0.0))
    batch_a = CausalSemanticBatchBuilder(route=route_a, route_lanes=lanes_a).build(
        _Vehicle(), _context(0, ego_a, (other_a,), route_a, lanes_a)
    )
    batch_b = CausalSemanticBatchBuilder(route=route_b, route_lanes=lanes_b).build(
        _Vehicle(), _context(0, ego_b, (other_b,), route_b, lanes_b)
    )
    assert np.allclose(batch_a.dynamic, batch_b.dynamic)
    assert np.allclose(batch_a.route, batch_b.route)
