"""Validated piecewise-linear elevation functions derived from 3D centerlines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PolylineElevation:
    """Elevation at XY via the nearest non-degenerate 3D centerline segment."""

    points_xyz: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points_xyz) < 2:
            raise ValueError("Elevation centerline requires at least two points")
        if not all(isfinite(value) for point in self.points_xyz for value in point):
            raise ValueError("Elevation centerline coordinates must be finite")
        for first, second in zip(self.points_xyz, self.points_xyz[1:]):
            if first[:2] == second[:2]:
                raise ValueError("Elevation centerline cannot contain consecutive duplicate XY points")

    def __call__(self, x: float, y: float) -> float:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Elevation query coordinates must be finite")
        candidates: list[tuple[float, int, float, float]] = []
        for index, (first, second) in enumerate(zip(self.points_xyz, self.points_xyz[1:])):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length_squared = dx * dx + dy * dy
            fraction = ((x - first[0]) * dx + (y - first[1]) * dy) / length_squared
            clamped_fraction = min(1.0, max(0.0, fraction))
            projection_x = first[0] + clamped_fraction * dx
            projection_y = first[1] + clamped_fraction * dy
            distance_squared = (x - projection_x) ** 2 + (y - projection_y) ** 2
            elevation = first[2] + clamped_fraction * (second[2] - first[2])
            candidates.append((distance_squared, index, clamped_fraction, elevation))
        _, _, _, elevation = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        return elevation
