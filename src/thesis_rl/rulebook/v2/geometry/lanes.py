"""Canonical route-lane association and longitudinal footprint coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import atan2, isfinite, sin, cos

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.types import MovementKey


LANE_ANGLE_EQUIVALENCE_EPSILON_RAD = 1.0e-6
LANE_LATERAL_EQUIVALENCE_EPSILON_M = 1.0e-3
SIGNED_DISTANCE_EPSILON_M = 5.0e-2


@lru_cache(maxsize=1024)
def _lane_coverage_polygon(polygon: BaseGeometry) -> BaseGeometry:
    """Return the frozen lane-association tolerance envelope once per geometry."""

    return polygon.buffer(GEOMETRY_EPSILON_M)


def _covers_lane_position(polygon: BaseGeometry, position_xy: tuple[float, float]) -> bool:
    """Use cached coverage geometry without requiring Shapely hash support."""

    try:
        envelope = _lane_coverage_polygon(polygon)
    except TypeError:
        envelope = polygon.buffer(GEOMETRY_EPSILON_M)
    return envelope.covers(Point(position_xy))


@dataclass(frozen=True, slots=True)
class RouteLaneRecord:
    lane_id: str
    polygon_xy: BaseGeometry
    centerline: RoutePolyline
    successor_lane_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("Route lane_id must be non-empty")
        if self.polygon_xy.is_empty or not self.polygon_xy.is_valid:
            raise ValueError("Route lane polygon must be non-empty and valid")
        if any(not isinstance(lane_id, str) or not lane_id for lane_id in self.successor_lane_ids):
            raise ValueError("Route lane successor IDs must be non-empty strings")


def derive_lane_movement_key(
    lane: RouteLaneRecord,
    *,
    assigned_route_lane_ids: tuple[str, ...] = (),
) -> MovementKey | None:
    """Derive a movement key only when the exit is topologically unambiguous.

    The assigned route may disambiguate a branching lane for the ego.  For
    other actors, a lane with multiple successors remains unresolved rather
    than selecting an arbitrary branch.
    """

    route = tuple(assigned_route_lane_ids)
    route_successors = tuple(
        route[index + 1]
        for index, lane_id in enumerate(route[:-1])
        if lane_id == lane.lane_id and route[index + 1] in lane.successor_lane_ids
    )
    if len(route_successors) == 1:
        exit_lane_id = route_successors[0]
    elif len(lane.successor_lane_ids) == 1:
        exit_lane_id = lane.successor_lane_ids[0]
    elif not lane.successor_lane_ids:
        exit_lane_id = lane.lane_id
    else:
        return None
    conflict_node_id = f"junction:{lane.lane_id}->{exit_lane_id}"
    return MovementKey(lane.lane_id, conflict_node_id, exit_lane_id)


@dataclass(frozen=True, slots=True)
class LaneAssociation:
    lane_id: str
    route_projection_s_m: float
    tangent_xy: tuple[float, float]
    lateral_distance_m: float
    heading_misalignment_rad: float


@dataclass(frozen=True, slots=True)
class FootprintRouteCoordinates:
    center_s_m: float
    front_s_m: float
    rear_s_m: float
    center_tangent_xy: tuple[float, float]
    center_lateral_m: float
    lateral_min_m: float
    lateral_max_m: float


def associate_route_lane(
    *,
    position_xy: tuple[float, float],
    position_z: float,
    heading_rad: float,
    route_lanes: tuple[RouteLaneRecord, ...],
) -> LaneAssociation | None:
    """Associate only a vertically compatible route lane, or return NA."""

    if not all(isfinite(value) for value in (*position_xy, position_z, heading_rad)):
        raise ValueError("Lane association pose must be finite")
    candidates: list[LaneAssociation] = []
    position_x, position_y = position_xy
    for lane in route_lanes:
        min_x, min_y, max_x, max_y = lane.polygon_xy.bounds
        if (
            position_x < min_x - GEOMETRY_EPSILON_M
            or position_x > max_x + GEOMETRY_EPSILON_M
            or position_y < min_y - GEOMETRY_EPSILON_M
            or position_y > max_y + GEOMETRY_EPSILON_M
        ):
            continue
        if not _covers_lane_position(lane.polygon_xy, position_xy):
            continue
        try:
            projection = lane.centerline.project(position_xy, position_z=position_z)
        except ValueError as error:
            if "vertically compatible" in str(error):
                continue
            raise
        tangent_heading = atan2(projection.tangent_xy[1], projection.tangent_xy[0])
        misalignment = abs(
            atan2(sin(heading_rad - tangent_heading), cos(heading_rad - tangent_heading))
        )
        candidates.append(
            LaneAssociation(
                lane_id=lane.lane_id,
                route_projection_s_m=projection.s_m,
                tangent_xy=projection.tangent_xy,
                lateral_distance_m=projection.lateral_distance_m,
                heading_misalignment_rad=misalignment,
            )
        )
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.heading_misalignment_rad,
            abs(candidate.lateral_distance_m),
            candidate.lane_id,
        ),
    )
    if len(ordered) >= 2:
        first, second = ordered[:2]
        if (
            abs(first.heading_misalignment_rad - second.heading_misalignment_rad)
            <= LANE_ANGLE_EQUIVALENCE_EPSILON_RAD
            and abs(abs(first.lateral_distance_m) - abs(second.lateral_distance_m))
            <= LANE_LATERAL_EQUIVALENCE_EPSILON_M
        ):
            return None
    return ordered[0] if ordered else None


def footprint_route_coordinates(
    footprint: BaseGeometry, route: RoutePolyline, *, position_z: float
) -> FootprintRouteCoordinates:
    """Project center then vertices on its local route branch as required by §2.9.2.

    The center's tangent/lateral projection also anchors the shared
    tangent/normal frame used by the R2 scoped lateral-RSS metric
    (rulebook v4.8 specification §7): ``lateral_min_m``/``lateral_max_m`` are
    the min/max of each vertex's ``lateral_distance_m`` (signed normal-axis
    offset), mirroring ``front_s_m``/``rear_s_m`` on the tangent axis.
    """

    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Footprint must be non-empty and valid")
    center = footprint.centroid
    center_projection = route.project((center.x, center.y), position_z=position_z)
    vertex_projections = [
        route.project((x, y), position_z=position_z, previous_s_m=center_projection.s_m)
        for x, y, *_ in footprint.exterior.coords[:-1]
    ]
    if not vertex_projections:
        raise ValueError("Footprint must contain exterior vertices")
    return FootprintRouteCoordinates(
        center_s_m=center_projection.s_m,
        front_s_m=max(projection.s_m for projection in vertex_projections),
        rear_s_m=min(projection.s_m for projection in vertex_projections),
        center_tangent_xy=center_projection.tangent_xy,
        center_lateral_m=center_projection.lateral_distance_m,
        lateral_min_m=min(projection.lateral_distance_m for projection in vertex_projections),
        lateral_max_m=max(projection.lateral_distance_m for projection in vertex_projections),
    )


def bumper_to_bumper_gap(
    ego: FootprintRouteCoordinates, other: FootprintRouteCoordinates
) -> tuple[float, bool]:
    """Return non-negative gap and canonical front-vehicle predicate."""

    is_front = other.rear_s_m >= ego.front_s_m - SIGNED_DISTANCE_EPSILON_M
    return max(0.0, other.rear_s_m - ego.front_s_m), is_front


def lateral_edge_to_edge_gap(
    ego: FootprintRouteCoordinates, other: FootprintRouteCoordinates
) -> tuple[float, int]:
    """Return the non-negative lateral edge-to-edge gap and side of ``other``.

    Mirrors :func:`bumper_to_bumper_gap` on the normal axis instead of the
    tangent axis (rulebook v4.8 specification §7, ``d_i^lat``). The lateral
    axis is symmetric (no inherent "front"), so the side is reported as
    ``+1`` when ``other`` is on the positive-normal side of ``ego``
    (``other.center_lateral_m >= ego.center_lateral_m``), else ``-1``.
    """

    if other.center_lateral_m >= ego.center_lateral_m:
        return max(0.0, other.lateral_min_m - ego.lateral_max_m), 1
    return max(0.0, ego.lateral_min_m - other.lateral_max_m), -1
