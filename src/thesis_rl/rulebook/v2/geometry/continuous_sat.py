"""Deterministic polygon decomposition and constant-velocity continuous SAT."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable

import shapely
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry


AREA_EPSILON_M2 = 1.0e-4
INTERVAL_EPSILON_S = 1.0e-6


@dataclass(frozen=True, slots=True)
class OccupancyInterval:
    start_s: float
    end_s: float | None

    @property
    def is_open_end(self) -> bool:
        return self.end_s is None


def _signed_area(vertices: list[tuple[float, float]]) -> float:
    return sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(vertices, vertices[1:] + vertices[:1])
    ) / 2.0


def _normalized_ring(coordinates: Iterable[tuple[float, float]], *, ccw: bool) -> list[tuple[float, float]]:
    ring = [(float(x), float(y)) for x, y, *_ in coordinates]
    if ring[0] == ring[-1]:
        ring.pop()
    if len(ring) < 3 or abs(_signed_area(ring)) <= AREA_EPSILON_M2:
        raise ValueError("Polygon ring is degenerate")
    if (_signed_area(ring) > 0.0) != ccw:
        ring.reverse()
    return ring


def _bridge_hole(
    outer: list[tuple[float, float]], hole: list[tuple[float, float]], polygon: Polygon
) -> list[tuple[float, float]]:
    pairs = sorted(
        (
            (hypot(outer_i[0] - hole_i[0], outer_i[1] - hole_i[1]), outer_index, hole_index)
            for outer_index, outer_i in enumerate(outer)
            for hole_index, hole_i in enumerate(hole)
        ),
        key=lambda value: (value[0], outer[value[1]], hole[value[2]]),
    )
    for _, outer_index, hole_index in pairs:
        bridge = shapely.LineString((outer[outer_index], hole[hole_index]))
        if polygon.covers(bridge):
            hole_cycle = hole[hole_index:] + hole[:hole_index] + [hole[hole_index]]
            return outer[: outer_index + 1] + hole_cycle + outer[outer_index:]
    raise ValueError("Polygon hole has no visible deterministic bridge")


def deterministic_convex_decomposition(polygon: BaseGeometry) -> tuple[Polygon, ...]:
    """Bridge holes then ear-clip with the frozen lexicographic ear tie-break."""

    if polygon.geom_type != "Polygon" or polygon.is_empty or not polygon.is_valid:
        raise ValueError("Convex decomposition requires one valid, non-empty polygon")
    outer = _normalized_ring(polygon.exterior.coords, ccw=True)
    holes = sorted(
        (_normalized_ring(interior.coords, ccw=False) for interior in polygon.interiors),
        key=lambda hole: min(hole),
    )
    merged = outer
    for hole in holes:
        merged = _bridge_hole(merged, hole, polygon)
    indices = list(range(len(merged)))
    triangles: list[Polygon] = []
    while len(indices) > 3:
        ears: list[tuple[tuple[float, float, int], int, Polygon]] = []
        for position, current_index in enumerate(indices):
            previous_index = indices[(position - 1) % len(indices)]
            next_index = indices[(position + 1) % len(indices)]
            previous, current, following = (
                merged[previous_index],
                merged[current_index],
                merged[next_index],
            )
            cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
                current[1] - previous[1]
            ) * (following[0] - current[0])
            triangle = Polygon((previous, current, following))
            if cross <= 0.0 or triangle.area <= AREA_EPSILON_M2 or not polygon.covers(triangle):
                continue
            if any(
                triangle.contains(Point(merged[other_index]))
                for other_index in indices
                if other_index not in {previous_index, current_index, next_index}
            ):
                continue
            ears.append(((current[0], current[1], current_index), position, triangle))
        if not ears:
            raise ValueError("Deterministic ear clipping found no valid ear")
        _, position, triangle = min(ears, key=lambda ear: ear[0])
        triangles.append(triangle)
        indices.pop(position)
    last_triangle = Polygon(tuple(merged[index] for index in indices))
    if last_triangle.area > AREA_EPSILON_M2 and polygon.covers(last_triangle):
        triangles.append(last_triangle)
    if not triangles:
        raise ValueError("Convex decomposition produced no non-degenerate triangles")
    return tuple(sorted(triangles, key=lambda tri: (tri.centroid.x, tri.centroid.y, tri.area)))


def _sat_interval(
    actor: Polygon, velocity_xy: tuple[float, float], zone: Polygon, horizon_s: float
) -> tuple[float, float] | None:
    lower, upper = 0.0, horizon_s
    axes: list[tuple[float, float]] = []
    for polygon in (actor, zone):
        coordinates = list(polygon.exterior.coords)
        for first, second in zip(coordinates, coordinates[1:]):
            edge_x, edge_y = second[0] - first[0], second[1] - first[1]
            length = hypot(edge_x, edge_y)
            if length > 0.0:
                axes.append((-edge_y / length, edge_x / length))
    for axis_x, axis_y in axes:
        actor_values = [axis_x * x + axis_y * y for x, y, *_ in actor.exterior.coords[:-1]]
        zone_values = [axis_x * x + axis_y * y for x, y, *_ in zone.exterior.coords[:-1]]
        actor_min, actor_max = min(actor_values), max(actor_values)
        zone_min, zone_max = min(zone_values), max(zone_values)
        speed = axis_x * velocity_xy[0] + axis_y * velocity_xy[1]
        for coefficient, bound in ((speed, zone_max - actor_min), (-speed, actor_max - zone_min)):
            if abs(coefficient) <= 1.0e-12:
                if bound < 0.0:
                    return None
            elif coefficient > 0.0:
                upper = min(upper, bound / coefficient)
            else:
                lower = max(lower, bound / coefficient)
        if lower > upper + INTERVAL_EPSILON_S:
            return None
    return max(0.0, lower), min(horizon_s, upper)


def predict_occupancy_interval(
    *, actor_footprint: Polygon, actor_velocity_xy: tuple[float, float], zone: Polygon, horizon_s: float
) -> OccupancyInterval | None:
    """Return `NO_INTERVAL` as ``None`` or the selected finite/open interval."""

    if not isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("Occupancy horizon must be finite and positive")
    if not all(isfinite(value) for value in actor_velocity_xy):
        raise ValueError("Actor velocity must be finite")
    intervals = [
        interval
        for actor_triangle in deterministic_convex_decomposition(actor_footprint)
        for zone_triangle in deterministic_convex_decomposition(zone)
        if (interval := _sat_interval(actor_triangle, actor_velocity_xy, zone_triangle, horizon_s))
        is not None
    ]
    if not intervals:
        return None
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + INTERVAL_EPSILON_S:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    selected = next((interval for interval in merged if interval[0] <= 0.0 <= interval[1]), merged[0])
    if selected[1] >= horizon_s - INTERVAL_EPSILON_S and translate(
        actor_footprint,
        actor_velocity_xy[0] * horizon_s, actor_velocity_xy[1] * horizon_s
    ).intersects(zone):
        return OccupancyInterval(selected[0], None)
    return OccupancyInterval(selected[0], selected[1])
