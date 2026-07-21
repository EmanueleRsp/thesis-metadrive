from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import box

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
) -> CausalSceneContext:
    task_route = TaskRouteRecord("scene", ("lane-0",), "test", "v1", "geometry")
    cache = EpisodeCache(
        "scene",
        task_route,
        conflict_zones={zone.zone_id: zone for zone in zones},
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
