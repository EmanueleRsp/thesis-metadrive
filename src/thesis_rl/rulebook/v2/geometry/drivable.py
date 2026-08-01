"""Per-step 2.5D drivable surface construction required by §2.9.4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import hypot, isfinite

import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M


@dataclass(frozen=True, slots=True)
class DrivableLaneRecord:
    lane_id: str
    centerline: RoutePolyline
    polygon_xy: BaseGeometry | None
    lane_width_m: float | None

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("Drivable lane_id must be non-empty")
        if self.polygon_xy is not None and (
            self.polygon_xy.is_empty or not self.polygon_xy.is_valid
        ):
            raise ValueError("Drivable lane polygon must be non-empty and valid")
        if self.polygon_xy is None and (
            self.lane_width_m is None or not isfinite(self.lane_width_m) or self.lane_width_m <= 0.0
        ):
            raise ValueError("Lane without polygon requires a finite, positive lane_width_m")

    def resolved_polygon(self) -> BaseGeometry:
        if self.polygon_xy is not None:
            return self.polygon_xy
        centerline_xy = LineString(tuple((x, y) for x, y, _ in self.centerline.points_xyz))
        assert self.lane_width_m is not None
        polygon = centerline_xy.buffer(
            self.lane_width_m / 2.0,
            cap_style="flat",
            join_style="mitre",
        )
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Centerline/width lane fallback produced an invalid polygon")
        return polygon


# Adjacent lane polygons are built independently of one another -- from
# per-point left/right widths on Waymo, from block geometry on PG, from a
# mitre buffer of the centerline in the fallback path -- so their shared
# edges do not coincide to floating-point accuracy. Their union therefore
# retains hairline interior holes (measured: 681 holes of 1e-4..5e-4 m^2 on a
# single PG map) that are a representation defect, not real off-road gaps. A
# morphological closing removes interior features narrower than 2x this
# value while leaving genuine map-edge gaps intact; 0.05 m was measured
# insufficient at the seams present in PG maps, 0.10 m removed every
# seam-driven activation. Closing can only add surface, so it can only lower
# an off-road cost, never raise one.
DRIVABLE_SEAM_CLOSING_M = 0.10


@lru_cache(maxsize=128)
def _union_selected_surfaces(selected: tuple[BaseGeometry, ...]) -> BaseGeometry:
    """Cache exact unions for a stable lane set within worker processes.

    Rulebook R3 evaluates the same vertically compatible lane layer for many
    consecutive control steps. Caching only the final Shapely operation keeps
    the normative union unchanged while avoiding repeated expensive topology
    work. The bounded cache is process-local, matching the environment worker
    lifecycle.
    """

    union = shapely.union_all(selected)
    closed = union.buffer(DRIVABLE_SEAM_CLOSING_M).buffer(-DRIVABLE_SEAM_CLOSING_M)
    if closed.is_empty or not closed.is_valid:
        # Degenerate closing result: fall back to the raw union, which is
        # strictly no worse than the pre-closing behaviour.
        return union
    return closed


def drivable_surface_for_ego(
    *,
    ego_footprint: BaseGeometry,
    ego_position_xy: tuple[float, float],
    ego_position_z: float,
    lanes: tuple[DrivableLaneRecord, ...],
) -> BaseGeometry:
    """Union every and only vertically compatible drivable lane for this step."""

    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Ego footprint must be non-empty and valid")
    if not all(isfinite(value) for value in (*ego_position_xy, ego_position_z)):
        raise ValueError("Ego pose must be finite")
    selected: list[BaseGeometry] = []
    for lane in lanes:
        projection = lane.centerline.project(ego_position_xy)
        if abs(ego_position_z - projection.z_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
            continue
        # The normative surface is the union of every vertically compatible
        # drivable lane. Do not prefilter by current footprint intersection:
        # an ego that has left a lane still needs a non-empty reference surface
        # so the off-road area fraction can be evaluated instead of failing as
        # if the map were unavailable.
        selected.append(lane.resolved_polygon())
    if not selected:
        return shapely.GeometryCollection()
    try:
        return _union_selected_surfaces(tuple(selected))
    except TypeError:
        # Preserve compatibility with Shapely geometry implementations that do
        # not expose a stable hash, without changing the normative result.
        return shapely.union_all(selected)


# REQ-RBCOST-009 / DEC-RBCOST-004: +-60 degree cone around the route tangent.
# A lane whose centerline tangent at the ego's projection agrees with the
# route tangent within this cone is "aligned"; one that opposes it within the
# same cone is "opposing". Lanes in neither cone (crossing branches inside a
# junction) contribute to neither surface.
DIRECTION_ALIGNMENT_COS_THRESHOLD = 0.5  # cos(60 degrees)


@dataclass(frozen=True, slots=True)
class CarriagewaySurfaces:
    aligned: BaseGeometry
    opposing: BaseGeometry


def carriageway_surfaces_for_ego(
    *,
    ego_position_xy: tuple[float, float],
    ego_position_z: float,
    route_tangent_xy: tuple[float, float],
    lanes: tuple[DrivableLaneRecord, ...],
) -> CarriagewaySurfaces:
    """Split vertically compatible lanes into route-aligned and opposing surfaces.

    Mirrors ``drivable_surface_for_ego``'s vertical-compatibility gate but
    additionally partitions by direction; it does not reuse or alter that
    function's return value, which remains the untouched normative off-road
    reference surface (REQ-RBCOST-001).
    """

    if not all(isfinite(value) for value in (*ego_position_xy, ego_position_z)):
        raise ValueError("Ego pose must be finite")
    route_norm = hypot(*route_tangent_xy)
    if not isfinite(route_norm) or route_norm <= 0.0:
        raise ValueError("Route tangent must be a finite, non-zero vector")
    route_unit = (route_tangent_xy[0] / route_norm, route_tangent_xy[1] / route_norm)
    aligned: list[BaseGeometry] = []
    opposing: list[BaseGeometry] = []
    for lane in lanes:
        projection = lane.centerline.project(ego_position_xy)
        if abs(ego_position_z - projection.z_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
            continue
        lane_tangent = projection.tangent_xy
        lane_norm = hypot(*lane_tangent)
        if lane_norm <= 0.0:
            continue
        cos_angle = (lane_tangent[0] * route_unit[0] + lane_tangent[1] * route_unit[1]) / lane_norm
        if cos_angle >= DIRECTION_ALIGNMENT_COS_THRESHOLD:
            aligned.append(lane.resolved_polygon())
        elif cos_angle <= -DIRECTION_ALIGNMENT_COS_THRESHOLD:
            opposing.append(lane.resolved_polygon())
    return CarriagewaySurfaces(
        aligned=_union_geometries(tuple(aligned)),
        opposing=_union_geometries(tuple(opposing)),
    )


def _union_geometries(geometries: tuple[BaseGeometry, ...]) -> BaseGeometry:
    if not geometries:
        return shapely.GeometryCollection()
    try:
        return _union_selected_surfaces(geometries)
    except TypeError:
        return shapely.union_all(geometries)
