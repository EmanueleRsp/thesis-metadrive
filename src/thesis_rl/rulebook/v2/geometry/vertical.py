"""Frozen 2.5D compatibility predicates for Rulebook v2 geometry."""

from __future__ import annotations

from math import isfinite
from typing import Protocol


VERTICAL_COMPATIBILITY_TOLERANCE_M = 3.0


class ElevationAtXY(Protocol):
    """Canonical elevation accessor supplied by a validated adapter."""

    def __call__(self, x: float, y: float) -> float: ...


def vertically_compatible_at_xy(
    *,
    first_elevation_at_xy: ElevationAtXY,
    second_elevation_at_xy: ElevationAtXY,
    x: float,
    y: float,
    tolerance_m: float = VERTICAL_COMPATIBILITY_TOLERANCE_M,
) -> bool:
    """Return whether two validated geometries are compatible at an XY point.

    Invalid coordinates, elevations, or tolerances are contract violations and
    raise rather than becoming a planar match.
    """

    if not all(isfinite(value) for value in (x, y, tolerance_m)) or tolerance_m <= 0.0:
        raise ValueError("XY coordinates and vertical tolerance must be finite; tolerance must be positive")
    first_z = float(first_elevation_at_xy(x, y))
    second_z = float(second_elevation_at_xy(x, y))
    if not isfinite(first_z) or not isfinite(second_z):
        raise ValueError("Elevation accessors must return finite values")
    return abs(first_z - second_z) <= tolerance_m
