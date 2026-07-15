"""Canonical oriented bounding-box footprints for live dynamic actors."""

from __future__ import annotations

from math import cos, isfinite, sin

from shapely.geometry import Polygon


def oriented_bounding_box(
    *,
    center_xy: tuple[float, float],
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> Polygon:
    """Construct the frozen canonical OBB from a live actor pose and size."""

    values = (*center_xy, heading_rad, length_m, width_m)
    if not all(isfinite(value) for value in values):
        raise ValueError("OBB center, heading, length and width must be finite")
    if length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("OBB length and width must be strictly positive")

    half_length = length_m / 2.0
    half_width = width_m / 2.0
    forward = (cos(heading_rad), sin(heading_rad))
    left = (-forward[1], forward[0])
    corners = tuple(
        (
            center_xy[0] + longitudinal * forward[0] + lateral * left[0],
            center_xy[1] + longitudinal * forward[1] + lateral * left[1],
        )
        for longitudinal, lateral in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        )
    )
    footprint = Polygon(corners)
    if not footprint.is_valid or footprint.is_empty:
        raise ValueError("Canonical OBB construction produced an invalid footprint")
    return footprint
