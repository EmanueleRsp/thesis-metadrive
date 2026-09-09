"""Wrapper-owned canonical 3D task-route polyline and projections."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import hypot, isfinite
from statistics import median
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

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
class RouteProjectionDiagnostics:
    """Read-only geometry facts for a rejected vertically filtered projection."""

    nearest_planar_segment_index: int
    nearest_planar_s_m: float
    nearest_planar_z_m: float
    nearest_planar_distance_m: float
    minimum_vertical_difference_m: float | None
    vertically_compatible_segment_count: int | None
    route_min_z_m: float
    route_max_z_m: float


@dataclass(frozen=True, slots=True)
class RoutePolyline:
    """Canonical 3D route with XY arc length and deterministic projections."""

    points_xyz: tuple[tuple[float, float, float], ...]
    # REQ-001 (video overlay v1, 2026-07-31): raw lane-start points from the
    # frozen lane sequence that produced this route, one per lane in order,
    # used only to draw discrete "planned checkpoint" markers in evaluation
    # GIFs. Diagnostic metadata only; not read by any rulebook geometry or
    # projection logic, so it participates in neither consolidation nor
    # equality-relevant computation.
    lane_start_points_xyz: tuple[tuple[float, float, float], ...] = ()
    _segment_starts_m: tuple[float, ...] = field(init=False, repr=False)
    _segment_lengths_m: tuple[float, ...] = field(init=False, repr=False)
    # Closed elevation range of the consolidated points. Every projection's
    # ``z_m`` is a convex combination of two consecutive point elevations, so it
    # lies inside this range (up to one rounding); callers that only need the
    # vertical-compatibility verdict can decide most polylines from it without
    # projecting (F8, `drivable.py`).
    z_range_m: tuple[float, float] = field(init=False, repr=False, compare=False)
    # F8: per-segment NumPy arrays for the vectorised projection, built on the
    # first call (most polylines are lane centerlines that are never projected
    # on). Excluded from equality and hashing: derived data, and arrays do not
    # hash.
    _segment_arrays: Any = field(init=False, repr=False, compare=False, default=None)

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
        elevations = [point[2] for point in consolidated]
        object.__setattr__(self, "z_range_m", (min(elevations), max(elevations)))

    @classmethod
    def from_lane_centerlines(
        cls, lane_centerlines: tuple[tuple[tuple[float, float, float], ...], ...]
    ) -> "RoutePolyline":
        if not lane_centerlines:
            raise ValueError("Task route must contain at least one lane centerline")
        return cls(
            tuple(point for centerline in lane_centerlines for point in centerline),
            lane_start_points_xyz=tuple(centerline[0] for centerline in lane_centerlines),
        )

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
        """Return the canonical centerline point at clamped XY arc length.

        The segment is the one that *contains* ``target``; selecting instead
        the segment with the nearest endpoint (the historical behaviour)
        returned the preceding vertex for every ``target`` in the first half
        of a segment, because the interpolation fraction was then clamped to
        ``1.0``.  With MetaDrive's 2 m polyline sampling that produced a
        sawtooth error of up to 1 m, biased backwards, in the route waypoints
        and curvature exposed to the policy observation.
        """

        if not isfinite(s_m):
            raise ValueError("Route point arc length must be finite")
        target = min(max(float(s_m), 0.0), self.length_m)
        # ``_segment_starts_m`` is strictly increasing, so the containing
        # segment is the last one whose start does not exceed ``target``.
        index = bisect_right(self._segment_starts_m, target) - 1
        index = min(max(index, 0), len(self._segment_lengths_m) - 1)
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
        max_s_jump_m: float | None = None,
    ) -> RouteProjection:
        """Project with the fixed vertical, distance, continuity and index ties.

        ``max_s_jump_m`` bounds how far the arc-length coordinate may move from
        ``previous_s_m``.  Without it, ``previous_s_m`` only breaks ties among
        candidates already within ``eps_geom`` of the minimum planar distance,
        so on a self-intersecting or closely parallel route (a roundabout, in
        practice) a strictly-closer far branch wins outright and the coordinate
        jumps, producing spurious route progress.

        The bound is a **preference, not a gate**: when no candidate is
        plausible the unbounded selection is kept, for the reason the inline
        comment at the selection gives.  An earlier revision of this docstring
        claimed the projection "fails closed" instead, which contradicted the
        code ten lines below it and the decision that introduced the bound
        (ADR-035, which documents it as a preference and records the three
        transition tests that failing closed broke).  Note also that **no
        production call site supplies this argument**: ``e63e0bf`` withdrew it
        from the progress evaluator on 2026-08-03 and the mission tracker
        projects without a jump envelope (`C50`, `C52`).
        """

        if not all(isfinite(value) for value in point_xy):
            raise ValueError("Route projection XY coordinates must be finite")
        if position_z is not None and not isfinite(position_z):
            raise ValueError("Route projection position_z must be finite when supplied")
        if previous_s_m is not None and not isfinite(previous_s_m):
            raise ValueError("Route projection previous_s_m must be finite when supplied")
        if max_s_jump_m is not None:
            if previous_s_m is None:
                raise ValueError("Route projection max_s_jump_m requires previous_s_m")
            if not isfinite(max_s_jump_m) or max_s_jump_m < 0.0:
                raise ValueError("Route projection max_s_jump_m must be finite and non-negative")

        (
            first_x,
            first_y,
            first_z,
            delta_z,
            lengths,
            tangent_x,
            tangent_y,
            starts,
        ) = self._projection_arrays()
        px = float(point_xy[0])
        py = float(point_xy[1])
        # One pass over every segment, with the reference's arithmetic per
        # segment: clamped fraction, projected point, interpolated z, signed
        # lateral distance and the planar distance recomputed from ``s``.
        offset_x = px - first_x
        offset_y = py - first_y
        fraction = np.clip((offset_x * tangent_x + offset_y * tangent_y) / lengths, 0.0, 1.0)
        along = fraction * lengths
        projected_x = first_x + along * tangent_x
        projected_y = first_y + along * tangent_y
        z_m = first_z + fraction * delta_z
        s_m = starts + along
        if position_z is not None:
            mask = np.abs(position_z - z_m) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
            if not mask.any():
                raise ValueError("Route projection has no vertically compatible segment")
        else:
            mask = np.ones(len(lengths), dtype=bool)
        if max_s_jump_m is not None:
            assert previous_s_m is not None
            # A *preference*, not a hard gate.  When no candidate is plausible
            # the unbounded selection is kept: the bound exists to stop a far
            # branch from winning while a plausible one is available, and
            # turning its absence into a failure would convert a rare geometric
            # situation into an episode abort with no compensating benefit.
            plausible = mask & (np.abs(s_m - previous_s_m) <= max_s_jump_m + GEOMETRY_EPSILON_M)
            if plausible.any():
                mask = plausible
        candidate_indices = np.flatnonzero(mask)
        along_from_s = s_m[candidate_indices] - starts[candidate_indices]
        distance = np.hypot(
            px - (first_x[candidate_indices] + along_from_s * tangent_x[candidate_indices]),
            py - (first_y[candidate_indices] + along_from_s * tangent_y[candidate_indices]),
        )
        tied = candidate_indices[distance <= distance.min() + GEOMETRY_EPSILON_M]
        tied_s = s_m[tied]
        key = tied_s if previous_s_m is None else np.abs(tied_s - previous_s_m)
        # Lexicographic minimum: the key first, the segment index on ties,
        # exactly as the reference's ``min`` over ``(key, segment_index)``.
        index = int(tied[np.lexsort((tied, key))[0]])
        return RouteProjection(
            s_m=float(s_m[index]),
            tangent_xy=(float(tangent_x[index]), float(tangent_y[index])),
            z_m=float(z_m[index]),
            lateral_distance_m=float(
                (px - projected_x[index]) * -tangent_y[index]
                + (py - projected_y[index]) * tangent_x[index]
            ),
            segment_index=index,
        )

    def _projection_arrays(self) -> tuple[np.ndarray, ...]:
        arrays = self._segment_arrays
        if arrays is None:
            points = np.asarray(self.points_xyz, dtype=np.float64)
            lengths = np.asarray(self._segment_lengths_m, dtype=np.float64)
            first_x = points[:-1, 0]
            first_y = points[:-1, 1]
            first_z = points[:-1, 2]
            arrays = (
                first_x,
                first_y,
                first_z,
                points[1:, 2] - first_z,
                lengths,
                (points[1:, 0] - first_x) / lengths,
                (points[1:, 1] - first_y) / lengths,
                np.asarray(self._segment_starts_m, dtype=np.float64),
            )
            object.__setattr__(self, "_segment_arrays", arrays)
        return arrays

    def projection_diagnostics(
        self,
        point_xy: tuple[float, float],
        *,
        position_z: float | None = None,
    ) -> RouteProjectionDiagnostics:
        """Describe projection feasibility without changing projection selection."""

        if not all(isfinite(value) for value in point_xy):
            raise ValueError("Route projection XY coordinates must be finite")
        if position_z is not None and not isfinite(position_z):
            raise ValueError("Route projection position_z must be finite when supplied")

        candidates: list[tuple[int, float, float, float, float]] = []
        for index, (first, second) in enumerate(zip(self.points_xyz, self.points_xyz[1:])):
            length = self._segment_lengths_m[index]
            tangent = ((second[0] - first[0]) / length, (second[1] - first[1]) / length)
            offset_x = point_xy[0] - first[0]
            offset_y = point_xy[1] - first[1]
            fraction = min(1.0, max(0.0, (offset_x * tangent[0] + offset_y * tangent[1]) / length))
            projected_x = first[0] + fraction * length * tangent[0]
            projected_y = first[1] + fraction * length * tangent[1]
            z_m = first[2] + fraction * (second[2] - first[2])
            candidates.append(
                (
                    index,
                    fraction,
                    z_m,
                    hypot(point_xy[0] - projected_x, point_xy[1] - projected_y),
                    abs(float(position_z) - z_m) if position_z is not None else 0.0,
                )
            )
        nearest = min(candidates, key=lambda item: (item[3], item[0]))
        vertical_differences = [item[4] for item in candidates] if position_z is not None else []
        compatible_count = (
            sum(item[4] <= VERTICAL_COMPATIBILITY_TOLERANCE_M for item in candidates)
            if position_z is not None
            else None
        )
        return RouteProjectionDiagnostics(
            nearest_planar_segment_index=nearest[0],
            nearest_planar_s_m=self._segment_starts_m[nearest[0]]
            + nearest[1] * self._segment_lengths_m[nearest[0]],
            nearest_planar_z_m=nearest[2],
            nearest_planar_distance_m=nearest[3],
            minimum_vertical_difference_m=min(vertical_differences)
            if vertical_differences
            else None,
            vertically_compatible_segment_count=compatible_count,
            route_min_z_m=min(point[2] for point in self.points_xyz),
            route_max_z_m=max(point[2] for point in self.points_xyz),
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
