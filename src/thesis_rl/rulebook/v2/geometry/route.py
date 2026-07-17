"""Wrapper-owned canonical 3D task-route polyline and projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from statistics import median
from typing import TYPE_CHECKING, Mapping

from thesis_rl.rulebook.v2.geometry.canonical import PRECISION_GRID_M
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M

if TYPE_CHECKING:
    from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord


GEOMETRY_EPSILON_M = 1.0e-2


@dataclass(frozen=True, slots=True)
class RouteProjection:
    s_m: float
    tangent_xy: tuple[float, float]
    z_m: float
    lateral_distance_m: float
    segment_index: int


@dataclass(frozen=True, slots=True)
class RoutePolyline:
    """Canonical 3D route with XY arc length and deterministic projections."""

    points_xyz: tuple[tuple[float, float, float], ...]
    _segment_starts_m: tuple[float, ...] = field(init=False, repr=False)
    _segment_lengths_m: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        consolidated = self._consolidate_points(self.points_xyz)
        if len(consolidated) < 2:
            raise ValueError("RoutePolyline requires at least two distinct XY points")
        starts: list[float] = []
        lengths: list[float] = []
        total = 0.0
        for first, second in zip(consolidated, consolidated[1:]):
            length = hypot(second[0] - first[0], second[1] - first[1])
            if length <= PRECISION_GRID_M:
                raise ValueError("RoutePolyline contains a collapsed segment after consolidation")
            starts.append(total)
            lengths.append(length)
            total += length
        object.__setattr__(self, "points_xyz", consolidated)
        object.__setattr__(self, "_segment_starts_m", tuple(starts))
        object.__setattr__(self, "_segment_lengths_m", tuple(lengths))

    @classmethod
    def from_lane_centerlines(
        cls, lane_centerlines: tuple[tuple[tuple[float, float, float], ...], ...]
    ) -> "RoutePolyline":
        if not lane_centerlines:
            raise ValueError("Task route must contain at least one lane centerline")
        return cls(tuple(point for centerline in lane_centerlines for point in centerline))

    @staticmethod
    def _consolidate_points(
        points_xyz: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[float, float, float], ...]:
        if not points_xyz:
            raise ValueError("RoutePolyline requires centerline points")
        if not all(isfinite(value) for point in points_xyz for value in point):
            raise ValueError("RoutePolyline centerline coordinates must be finite")
        clusters: list[list[tuple[float, float, float]]] = [[points_xyz[0]]]
        for point in points_xyz[1:]:
            previous = clusters[-1][-1]
            if hypot(point[0] - previous[0], point[1] - previous[1]) <= PRECISION_GRID_M:
                clusters[-1].append(point)
            else:
                clusters.append([point])

        consolidated: list[tuple[float, float, float]] = []
        for cluster in clusters:
            elevations = [point[2] for point in cluster]
            if max(elevations) - min(elevations) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
                raise ValueError("RoutePolyline cluster has vertically incompatible elevations")
            consolidated.append(
                (
                    sum(point[0] for point in cluster) / len(cluster),
                    sum(point[1] for point in cluster) / len(cluster),
                    float(median(elevations)),
                )
            )
        return tuple(consolidated)

    @property
    def length_m(self) -> float:
        return self._segment_starts_m[-1] + self._segment_lengths_m[-1]

    def point_at(self, s_m: float) -> tuple[float, float, float]:
        """Return the canonical centerline point at clamped XY arc length."""

        if not isfinite(s_m):
            raise ValueError("Route point arc length must be finite")
        target = min(max(float(s_m), 0.0), self.length_m)
        index = min(
            range(len(self._segment_lengths_m)),
            key=lambda candidate: abs(
                target - (self._segment_starts_m[candidate] + self._segment_lengths_m[candidate])
            ),
        )
        start = self._segment_starts_m[index]
        length = self._segment_lengths_m[index]
        fraction = min(1.0, max(0.0, (target - start) / length))
        first, second = self.points_xyz[index], self.points_xyz[index + 1]
        return tuple(first[axis] + fraction * (second[axis] - first[axis]) for axis in range(3))

    def project(
        self,
        point_xy: tuple[float, float],
        *,
        position_z: float | None = None,
        previous_s_m: float | None = None,
    ) -> RouteProjection:
        """Project with the fixed vertical, distance, continuity and index ties."""

        if not all(isfinite(value) for value in point_xy):
            raise ValueError("Route projection XY coordinates must be finite")
        if position_z is not None and not isfinite(position_z):
            raise ValueError("Route projection position_z must be finite when supplied")
        if previous_s_m is not None and not isfinite(previous_s_m):
            raise ValueError("Route projection previous_s_m must be finite when supplied")

        candidates: list[RouteProjection] = []
        for index, (first, second) in enumerate(zip(self.points_xyz, self.points_xyz[1:])):
            length = self._segment_lengths_m[index]
            tangent = ((second[0] - first[0]) / length, (second[1] - first[1]) / length)
            offset_x = point_xy[0] - first[0]
            offset_y = point_xy[1] - first[1]
            fraction = min(1.0, max(0.0, (offset_x * tangent[0] + offset_y * tangent[1]) / length))
            projected_x = first[0] + fraction * length * tangent[0]
            projected_y = first[1] + fraction * length * tangent[1]
            z_m = first[2] + fraction * (second[2] - first[2])
            if (
                position_z is not None
                and abs(position_z - z_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M
            ):
                continue
            lateral_distance = (point_xy[0] - projected_x) * -tangent[1] + (
                point_xy[1] - projected_y
            ) * tangent[0]
            candidates.append(
                RouteProjection(
                    s_m=self._segment_starts_m[index] + fraction * length,
                    tangent_xy=tangent,
                    z_m=z_m,
                    lateral_distance_m=lateral_distance,
                    segment_index=index,
                )
            )
        if not candidates:
            raise ValueError("Route projection has no vertically compatible segment")

        def planar_distance(candidate: RouteProjection) -> float:
            first = self.points_xyz[candidate.segment_index]
            along = candidate.s_m - self._segment_starts_m[candidate.segment_index]
            projected_x = first[0] + along * candidate.tangent_xy[0]
            projected_y = first[1] + along * candidate.tangent_xy[1]
            return hypot(point_xy[0] - projected_x, point_xy[1] - projected_y)

        minimum_distance = min(planar_distance(candidate) for candidate in candidates)
        tied = [
            candidate
            for candidate in candidates
            if planar_distance(candidate) <= minimum_distance + GEOMETRY_EPSILON_M
        ]
        if previous_s_m is None:
            return min(tied, key=lambda candidate: (candidate.s_m, candidate.segment_index))
        return min(
            tied,
            key=lambda candidate: (
                abs(candidate.s_m - previous_s_m),
                candidate.segment_index,
            ),
        )


def build_assigned_route_polyline(
    assigned_route_lane_ids: tuple[str, ...],
    route_lanes: Mapping[str, "RouteLaneRecord"],
) -> RoutePolyline:
    """Build the reset-time route from frozen lane IDs and canonical geometry.

    The lane sequence is task metadata, not a runtime trajectory. Missing lanes
    or non-contiguous geometry fail closed before control.
    """

    if not assigned_route_lane_ids:
        raise ValueError("Assigned route lane IDs must be non-empty")
    if any(not lane_id for lane_id in assigned_route_lane_ids):
        raise ValueError("Assigned route lane IDs must be non-empty strings")
    missing = [lane_id for lane_id in assigned_route_lane_ids if lane_id not in route_lanes]
    if missing:
        raise ValueError("Assigned route lane is missing: " + ",".join(missing))

    lanes = [route_lanes[lane_id] for lane_id in assigned_route_lane_ids]
    for previous, following in zip(lanes, lanes[1:]):
        previous_end = previous.centerline.points_xyz[-1]
        following_start = following.centerline.points_xyz[0]
        if (
            hypot(previous_end[0] - following_start[0], previous_end[1] - following_start[1])
            > GEOMETRY_EPSILON_M
            or abs(previous_end[2] - following_start[2]) > VERTICAL_COMPATIBILITY_TOLERANCE_M
        ):
            raise ValueError(
                "Assigned route lane sequence is not contiguous: "
                f"{previous.lane_id}->{following.lane_id}"
            )
    return RoutePolyline.from_lane_centerlines(tuple(lane.centerline.points_xyz for lane in lanes))
