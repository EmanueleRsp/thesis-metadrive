"""Canonical traffic-control line derivation and signed-distance orientation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import control_point_level_compatible


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


class ControlLineOffRouteError(ValueError):
    """The control line does not cross the canonical route.

    This is the normal case for a control governing another approach of the
    same intersection, not a data defect: such a control simply does not govern
    the ego on this task route.  It is a distinct type so adapters can skip it
    without also swallowing genuine geometry errors.
    """


def _route_curvilinear_crossing_s_m(
    *,
    control_line: LineString,
    route: RoutePolyline,
    control_line_z_m: float,
) -> float:
    """Return `route_s` per v4.7 §2.9.6/§2.9.5.

    The value is the minimum curvilinear coordinate among the vertically
    compatible intersection points between the control line and the
    ``RoutePolyline`` buffered by ``eps_geom``.

    REQ-EF-16: the elevation compared here is the *control line's*, taken from
    the controlled lane centerline at q_c, not the control point's.  The line
    lies on the carriageway by construction, so this is the level at which the
    crossing must be judged.  Passing the raw control-point elevation instead
    leaked a stop sign's mounting height into a grade-separation test and could
    discard a legitimate crossing on the ego's own level.

    Deriving it from the *controlled lane's* centerline instead — as the
    original implementation did — yields a lane-local coordinate, which is
    silently wrong for every control that is not on the first lane of the
    route: it was then compared against an ego front bumper measured on the
    concatenated route, so the control appeared already passed.
    """

    route_xy = LineString(tuple((x, y) for x, y, _ in route.points_xyz))
    crossing = control_line.intersection(route_xy.buffer(GEOMETRY_EPSILON_M))
    if crossing.is_empty:
        raise ControlLineOffRouteError("Control line does not cross the canonical route")
    candidates: list[float] = []
    for x, y, *_ in _crossing_coordinates(crossing):
        try:
            projection = route.project((x, y), position_z=control_line_z_m)
        except ValueError:
            # Vertically incompatible crossing (grade-separated overpass):
            # not a crossing of *this* route level.
            continue
        candidates.append(projection.s_m)
    if not candidates:
        raise ControlLineOffRouteError(
            "Control line crosses the route only at vertically incompatible levels"
        )
    return min(candidates)


def _crossing_coordinates(geometry: BaseGeometry) -> tuple[tuple[float, ...], ...]:
    if geometry.geom_type == "Point":
        return ((geometry.x, geometry.y),)
    if geometry.geom_type == "LineString":
        return tuple(geometry.coords)
    if geometry.geom_type in {"MultiPoint", "MultiLineString", "GeometryCollection"}:
        return tuple(
            coordinate for child in geometry.geoms for coordinate in _crossing_coordinates(child)
        )
    return ()


def derive_control_line(
    *,
    control_point_xy: tuple[float, float],
    control_point_z: float,
    controlled_lane: RouteLaneRecord,
    route: RoutePolyline,
) -> ControlLine:
    """Derive exactly the orthogonal lane section mandated by §2.9.6.

    The line's own geometry still comes from the controlled lane (§2.9.6 is
    unchanged).  Only ``route_s_m`` changes frame: it is now the canonical
    route coordinate of the crossing (§2.9.5), not a lane-local one.
    """

    if not all(isfinite(value) for value in (*control_point_xy, control_point_z)):
        raise ValueError("Control point coordinates must be finite")
    projection = controlled_lane.centerline.project(control_point_xy)
    # REQ-EF-16: an asymmetric mounting-height-aware bound, not abs(dz) <= tol.
    # ``controlled_lane`` comes from the *authoritative* association in the data
    # (``STOP_SIGN.lane`` / ``TRAFFIC_LIGHT.lane``), and the route-crossing test
    # in ``_route_curvilinear_crossing_s_m`` independently confirms the level, so
    # this is a plausibility bound rather than the level discriminator it was
    # written as.
    if not control_point_level_compatible(
        control_point_z=control_point_z, surface_z=projection.z_m
    ):
        raise ValueError("Control point is vertically incompatible with controlled lane")
    tangent = projection.tangent_xy
    normal = (-tangent[1], tangent[0])
    # §2.9.6 steps 3 and 5 are anchored on q_c, the projection of the control
    # point onto the lane centerline -- not on the raw control point.  Waymo's
    # ``STOP_SIGN.position`` is the physical sign, which sits at the roadside
    # outside the lane polygon (traffic lights use ``stop_point``, which is on
    # the carriageway).  Building the section through the raw point therefore
    # offset the stop line both laterally and longitudinally, and made the
    # step-5 component selection measure distances from a point outside the
    # polygon, where two components can tie within eps_geom and abort with
    # "multiple unresolved components".  q_c lies on the centerline and hence
    # inside the polygon, so the containing component is unique.
    centerline_point = controlled_lane.centerline.point_at(projection.s_m)
    section_origin_xy = (centerline_point[0], centerline_point[1])
    min_x, min_y, max_x, max_y = controlled_lane.polygon_xy.bounds
    extent = hypot(max_x - min_x, max_y - min_y) + 1.0
    infinite_section = LineString(
        (
            (
                section_origin_xy[0] - extent * normal[0],
                section_origin_xy[1] - extent * normal[1],
            ),
            (
                section_origin_xy[0] + extent * normal[0],
                section_origin_xy[1] + extent * normal[1],
            ),
        )
    )
    components = _line_components(infinite_section.intersection(controlled_lane.polygon_xy))
    if not components:
        raise ValueError("Control-line section does not intersect controlled lane polygon")
    control_point = Point(section_origin_xy)
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
        route_s_m=_route_curvilinear_crossing_s_m(
            control_line=line, route=route, control_line_z_m=projection.z_m
        ),
        # q_c, so ``signed_distance_m`` is measured from the stop line itself
        # rather than from the roadside sign position.
        point_xy=section_origin_xy,
        route_tangent_xy=tangent,
        elevation_m=projection.z_m,
    )
