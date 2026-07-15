"""Per-step 2.5D drivable surface construction required by §2.9.4."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
            self.lane_width_m is None
            or not isfinite(self.lane_width_m)
            or self.lane_width_m <= 0.0
        ):
            raise ValueError("Lane without polygon requires a finite, positive lane_width_m")

    def resolved_polygon(self) -> BaseGeometry:
        if self.polygon_xy is not None:
            return self.polygon_xy
        centerline_xy = LineString(tuple((x, y) for x, y, _ in self.centerline.points_xyz))
        polygon = centerline_xy.buffer(
            self.lane_width_m / 2.0,
            cap_style="flat",
            join_style="mitre",
        )
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Centerline/width lane fallback produced an invalid polygon")
        return polygon


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
        polygon = lane.resolved_polygon()
        if ego_footprint.intersects(polygon) or polygon.buffer(1.0e-2).contains(
            ego_footprint.centroid
        ):
            selected.append(polygon)
    if not selected:
        return shapely.GeometryCollection()
    return shapely.union_all(selected)
