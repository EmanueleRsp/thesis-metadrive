"""Pure canonical vehicle--vehicle conflict-zone candidate construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.geometry.canonical import (
    CanonicalGeometryError,
    canonical_geometry_wkb,
    canonicalize_geometry,
)
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import ElevationAtXY, vertically_compatible_at_xy
from thesis_rl.rulebook.v2.types import MovementKey


OFFROAD_AREA_EPSILON_M2 = 1.0e-4
SIGNED_DISTANCE_EPSILON_M = 5.0e-2


@dataclass(frozen=True, slots=True)
class MovementCorridor:
    """Static, speed-independent corridor identified by a stable movement key."""

    movement_key: MovementKey
    polygon: BaseGeometry
    elevation_at_xy: ElevationAtXY


@dataclass(frozen=True, slots=True)
class ConflictZoneCandidate:
    zone_id: str
    component_index: int
    polygon: BaseGeometry
    ego_movement_key: MovementKey
    other_movement_key: MovementKey


@dataclass(frozen=True, slots=True)
class CrosswalkZoneCandidate:
    zone_id: str
    component_index: int
    polygon: BaseGeometry
    ego_movement_key: MovementKey
    crosswalk_id: str


class _ZoneCandidate(Protocol):
    component_index: int
    polygon: BaseGeometry


@dataclass(frozen=True, slots=True)
class RouteConflictZoneCandidate:
    """A candidate whose canonical route entry/exit interval has been derived."""

    candidate: _ZoneCandidate
    route_entry_s_m: float
    route_exit_s_m: float


def _polygonal_components(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    elif geometry.geom_type == "GeometryCollection":
        for child in geometry.geoms:
            yield from _polygonal_components(child)


def _intersection_coordinates(geometry: BaseGeometry) -> Iterable[tuple[float, float]]:
    if geometry.geom_type == "Point":
        yield geometry.x, geometry.y
    elif geometry.geom_type in {"LineString", "LinearRing"}:
        yield from ((float(x), float(y)) for x, y, *_ in geometry.coords)
    elif geometry.geom_type.startswith("Multi") or geometry.geom_type == "GeometryCollection":
        for child in geometry.geoms:
            yield from _intersection_coordinates(child)


def _vertical_overlap_compatible(
    polygon: BaseGeometry,
    first_elevation: ElevationAtXY,
    second_elevation: ElevationAtXY,
) -> bool:
    # Geometry repair/canonicalization can legitimately return a MultiPolygon
    # even when the original corridor intersection was a single polygon.  Do
    # not assume a scalar ``.exterior`` here: sample every polygonal component
    # (including holes) for the vertical compatibility check.
    points: list[Point] = []
    for component in _polygonal_components(polygon):
        points.append(component.representative_point())
        points.extend(Point(xy) for xy in component.exterior.coords[:-1])
        for interior in component.interiors:
            points.extend(Point(xy) for xy in interior.coords[:-1])
    if not points:
        return False
    return all(
        vertically_compatible_at_xy(
            first_elevation_at_xy=first_elevation,
            second_elevation_at_xy=second_elevation,
            x=point.x,
            y=point.y,
        )
        for point in points
    )


def _movement_key_payload(key: MovementKey) -> dict[str, str]:
    return {
        "approach_lane_id": key.approach_lane_id,
        "conflict_node_id": key.conflict_node_id,
        "exit_lane_id": key.exit_lane_id,
    }


def _vehicle_zone_id(
    *,
    scenario_id: str,
    ego_key: MovementKey,
    other_key: MovementKey,
    component_index: int,
    polygon: BaseGeometry,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "namespace": "vehicle_yield",
        "feature_type": "conflict_zone",
        "ego_movement_key": _movement_key_payload(ego_key),
        "other_movement_key": _movement_key_payload(other_key),
        "component_index": component_index,
        "canonical_wkb_hex": canonical_geometry_wkb(polygon).hex(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _crosswalk_zone_id(
    *,
    scenario_id: str,
    ego_key: MovementKey,
    crosswalk_id: str,
    component_index: int,
    polygon: BaseGeometry,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "namespace": "crosswalk",
        "feature_type": "conflict_zone",
        "ego_movement_key": _movement_key_payload(ego_key),
        "crosswalk_id": crosswalk_id,
        "component_index": component_index,
        "canonical_wkb_hex": canonical_geometry_wkb(polygon).hex(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_vehicle_conflict_zone_candidates(
    *,
    scenario_id: str,
    ego_corridor: MovementCorridor,
    other_corridor: MovementCorridor,
) -> tuple[ConflictZoneCandidate, ...]:
    """Intersect two static corridors and return DEC-012-ordered candidates."""

    if not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    # RULEBOOK-V4.7 §2.8 requires canonical movement corridors before the
    # boolean operation.  In particular, snapping only a post-intersection
    # sliver can turn an otherwise valid derived component into an empty
    # geometry and incorrectly make an ordinary no-zone case fatal.
    ego_polygon = canonicalize_geometry(ego_corridor.polygon)
    other_polygon = canonicalize_geometry(other_corridor.polygon)
    ego_min_x, ego_min_y, ego_max_x, ego_max_y = ego_polygon.bounds
    other_min_x, other_min_y, other_max_x, other_max_y = other_polygon.bounds
    # Exact broad phase after canonical validation: disjoint axis-aligned
    # bounds imply a disjoint polygon intersection.  Keeping validation ahead
    # of this rejection preserves the Rulebook's invalid-geometry contract.
    if (
        ego_max_x < other_min_x
        or ego_min_x > other_max_x
        or ego_max_y < other_min_y
        or ego_min_y > other_max_y
    ):
        return ()
    overlap = ego_polygon.intersection(other_polygon)
    components: list[BaseGeometry] = []
    for component in _polygonal_components(overlap):
        if component.area <= OFFROAD_AREA_EPSILON_M2:
            continue
        try:
            canonical = canonicalize_geometry(component)
        except CanonicalGeometryError:
            # Both source corridors have already passed canonical validation.
            # A component which disappears only after the derived boolean
            # result is snapped is not a valid conflict zone (§2.8.2).
            continue
        if _vertical_overlap_compatible(
            canonical,
            ego_corridor.elevation_at_xy,
            other_corridor.elevation_at_xy,
        ):
            components.append(canonical)
    ordered = sorted(components, key=canonical_geometry_wkb)
    return tuple(
        ConflictZoneCandidate(
            zone_id=_vehicle_zone_id(
                scenario_id=scenario_id,
                ego_key=ego_corridor.movement_key,
                other_key=other_corridor.movement_key,
                component_index=index,
                polygon=polygon,
            ),
            component_index=index,
            polygon=polygon,
            ego_movement_key=ego_corridor.movement_key,
            other_movement_key=other_corridor.movement_key,
        )
        for index, polygon in enumerate(ordered)
    )


def build_crosswalk_conflict_zone_candidates(
    *,
    scenario_id: str,
    ego_corridor: MovementCorridor,
    crosswalk_id: str,
    crosswalk_polygon: BaseGeometry,
    crosswalk_elevation_at_xy: ElevationAtXY,
) -> tuple[CrosswalkZoneCandidate, ...]:
    """Build the static crosswalk candidates without introducing another movement key."""

    if not scenario_id or not crosswalk_id:
        raise ValueError("scenario_id and crosswalk_id must be non-empty")
    ego_polygon = canonicalize_geometry(ego_corridor.polygon)
    canonical_crosswalk = canonicalize_geometry(crosswalk_polygon)
    overlap = ego_polygon.intersection(canonical_crosswalk)
    components: list[BaseGeometry] = []
    for component in _polygonal_components(overlap):
        if component.area <= OFFROAD_AREA_EPSILON_M2:
            continue
        try:
            canonical = canonicalize_geometry(component)
        except CanonicalGeometryError:
            continue
        if _vertical_overlap_compatible(
            canonical,
            ego_corridor.elevation_at_xy,
            crosswalk_elevation_at_xy,
        ):
            components.append(canonical)
    ordered = sorted(components, key=canonical_geometry_wkb)
    return tuple(
        CrosswalkZoneCandidate(
            zone_id=_crosswalk_zone_id(
                scenario_id=scenario_id,
                ego_key=ego_corridor.movement_key,
                crosswalk_id=crosswalk_id,
                component_index=index,
                polygon=polygon,
            ),
            component_index=index,
            polygon=polygon,
            ego_movement_key=ego_corridor.movement_key,
            crosswalk_id=crosswalk_id,
        )
        for index, polygon in enumerate(ordered)
    )


def route_interval_for_zone(
    *, route: RoutePolyline, polygon: BaseGeometry
) -> tuple[float, float] | None:
    """Return route entry/exit of a zone buffered by the frozen geometry epsilon."""

    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Conflict-zone polygon must be non-empty and valid")
    buffered_zone = polygon.buffer(GEOMETRY_EPSILON_M)
    zone_min_x, zone_min_y, zone_max_x, zone_max_y = buffered_zone.bounds
    route_coordinates: list[float] = []
    for index, (first, second) in enumerate(zip(route.points_xyz, route.points_xyz[1:])):
        # This is an exact broad-phase rejection: a segment whose AABB does not
        # overlap the buffered zone cannot intersect it.  Avoiding construction
        # and GEOS intersection for the usual far-away route segments is
        # particularly important for long ScenarioNet polylines.
        segment_min_x = min(first[0], second[0])
        segment_max_x = max(first[0], second[0])
        segment_min_y = min(first[1], second[1])
        segment_max_y = max(first[1], second[1])
        if (
            segment_max_x < zone_min_x
            or segment_min_x > zone_max_x
            or segment_max_y < zone_min_y
            or segment_min_y > zone_max_y
        ):
            continue
        segment = LineString(((first[0], first[1]), (second[0], second[1])))
        intersection = segment.intersection(buffered_zone)
        length = route._segment_lengths_m[index]
        for x, y in _intersection_coordinates(intersection):
            fraction = min(
                1.0,
                max(
                    0.0,
                    (
                        (x - first[0]) * (second[0] - first[0])
                        + (y - first[1]) * (second[1] - first[1])
                    )
                    / (length * length),
                ),
            )
            route_coordinates.append(route._segment_starts_m[index] + fraction * length)
    if not route_coordinates:
        return None
    return min(route_coordinates), max(route_coordinates)


def attach_route_intervals(
    *, route: RoutePolyline, candidates: tuple[_ZoneCandidate, ...]
) -> tuple[RouteConflictZoneCandidate, ...]:
    """Discard zone components not traversed by the canonical ego route."""

    attached: list[RouteConflictZoneCandidate] = []
    for candidate in candidates:
        interval = route_interval_for_zone(route=route, polygon=candidate.polygon)
        if interval is not None:
            attached.append(
                RouteConflictZoneCandidate(
                    candidate=candidate,
                    route_entry_s_m=interval[0],
                    route_exit_s_m=interval[1],
                )
            )
    return tuple(attached)


def select_first_ahead_or_occupied_zone(
    *,
    candidates: tuple[RouteConflictZoneCandidate, ...],
    ego_footprint: BaseGeometry,
    ego_front_s_m: float,
) -> RouteConflictZoneCandidate | None:
    """Apply §2.8.2 selection: occupied component, otherwise first ahead."""

    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Ego footprint must be non-empty and valid")
    occupied = [
        (candidate, ego_footprint.intersection(candidate.candidate.polygon).area)
        for candidate in candidates
    ]
    occupied = [(candidate, area) for candidate, area in occupied if area > 0.0]
    if occupied:
        return min(
            occupied,
            key=lambda item: (
                -item[1],
                item[0].route_entry_s_m,
                item[0].candidate.component_index,
            ),
        )[0]
    ahead = [
        candidate
        for candidate in candidates
        if candidate.route_exit_s_m >= ego_front_s_m - SIGNED_DISTANCE_EPSILON_M
    ]
    if not ahead:
        return None
    return min(
        ahead,
        key=lambda candidate: (candidate.route_entry_s_m, candidate.candidate.component_index),
    )


def worst_case_temporal_gap_violation(
    *,
    ego_interval: OccupancyInterval,
    other_intervals: tuple[tuple[str, OccupancyInterval], ...],
    gap_scale_s: float,
) -> float:
    """Worst-of temporal-gap risk between an ego occupancy interval and others.

    Returns ``max_i [1 - gap_i / gap_scale_s]_+`` in ``[0, 1]``; an interval
    pair that cannot be temporally ordered (concurrent/unknown separation)
    contributes ``1.0``. Shared by the vehicle-yield pre-state and
    post-state passes (Rulebook v2 spec Section 7.9).
    """
    worst = 0.0
    for _, interval in other_intervals:
        if ego_interval.end_s is not None and interval.start_s >= ego_interval.end_s:
            gap = interval.start_s - ego_interval.end_s
        elif interval.end_s is not None and interval.end_s <= ego_interval.start_s:
            gap = ego_interval.start_s - interval.end_s
        else:
            gap = None
        worst = max(
            worst,
            1.0 if gap is None else min(max((gap_scale_s - gap) / gap_scale_s, 0.0), 1.0),
        )
    return worst
