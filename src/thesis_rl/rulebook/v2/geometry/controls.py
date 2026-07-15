"""Canonical traffic-control line derivation and signed-distance orientation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M


@dataclass(frozen=True, slots=True)
class ControlLine:
    geometry: LineString
    route_s_m: float
    point_xy: tuple[float, float]
    route_tangent_xy: tuple[float, float]
    elevation_m: float

    def signed_distance_m(self, point_xy: tuple[float, float]) -> float:
        """Positive upstream of the line and negative after it along the route."""

        return -(
            (point_xy[0] - self.point_xy[0]) * self.route_tangent_xy[0]
            + (point_xy[1] - self.point_xy[1]) * self.route_tangent_xy[1]
        )


def _line_components(geometry: BaseGeometry) -> tuple[LineString, ...]:
    if geometry.geom_type == "LineString":
        return (geometry,)
    if geometry.geom_type == "MultiLineString":
        return tuple(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return tuple(line for child in geometry.geoms for line in _line_components(child))
    return ()


def derive_control_line(
    *,
    control_point_xy: tuple[float, float],
    control_point_z: float,
    controlled_lane: RouteLaneRecord,
) -> ControlLine:
    """Derive exactly the orthogonal lane section mandated by §2.9.6."""

    if not all(isfinite(value) for value in (*control_point_xy, control_point_z)):
        raise ValueError("Control point coordinates must be finite")
    projection = controlled_lane.centerline.project(control_point_xy)
    if abs(control_point_z - projection.z_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
        raise ValueError("Control point is vertically incompatible with controlled lane")
    tangent = projection.tangent_xy
    normal = (-tangent[1], tangent[0])
    min_x, min_y, max_x, max_y = controlled_lane.polygon_xy.bounds
    extent = hypot(max_x - min_x, max_y - min_y) + 1.0
    infinite_section = LineString(
        (
            (control_point_xy[0] - extent * normal[0], control_point_xy[1] - extent * normal[1]),
            (control_point_xy[0] + extent * normal[0], control_point_xy[1] + extent * normal[1]),
        )
    )
    components = _line_components(infinite_section.intersection(controlled_lane.polygon_xy))
    if not components:
        raise ValueError("Control-line section does not intersect controlled lane polygon")
    control_point = Point(control_point_xy)
    distances = [component.distance(control_point) for component in components]
    containing = [
        component
        for component in components
        if component.distance(control_point) <= GEOMETRY_EPSILON_M
    ]
    candidates = containing or [
        component
        for component, distance in zip(components, distances)
        if distance <= min(distances) + GEOMETRY_EPSILON_M
    ]
    if len(candidates) != 1:
        raise ValueError("Control-line intersection has multiple unresolved components")
    line = candidates[0]
    if line.is_empty or line.length <= 0.0:
        raise ValueError("Control-line intersection is empty or degenerate")
    return ControlLine(
        geometry=line,
        route_s_m=projection.s_m,
        point_xy=control_point_xy,
        route_tangent_xy=tangent,
        elevation_m=projection.z_m,
    )
