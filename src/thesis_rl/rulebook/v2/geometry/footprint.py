"""Canonical oriented bounding-box footprints for live dynamic actors."""

from __future__ import annotations

from math import cos, isfinite, sin

from shapely.geometry import LineString, Polygon


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


def front_bumper_segment(footprint: Polygon, *, heading_rad: float) -> LineString:
    """Return the OBB edge with maximum body-frame longitudinal coordinate."""

    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Footprint must be non-empty and valid")
    if not isfinite(heading_rad):
        raise ValueError("Footprint heading must be finite")
    center = footprint.centroid
    forward = (cos(heading_rad), sin(heading_rad))
    vertices = list(footprint.exterior.coords[:-1])
    longitudinal = [
        (x - center.x) * forward[0] + (y - center.y) * forward[1]
        for x, y, *_ in vertices
    ]
    maximum = max(longitudinal)
    front_vertices = [
        vertex
        for vertex, coordinate in zip(vertices, longitudinal)
        if abs(coordinate - maximum) <= 1.0e-9
    ]
    if len(front_vertices) != 2:
        raise ValueError("Footprint does not expose exactly two front-bumper vertices")
    return LineString(front_vertices)


def swept_front_bumper(
    pre_footprint: Polygon,
    post_footprint: Polygon,
    *,
    pre_heading_rad: float,
    post_heading_rad: float,
) -> Polygon:
    """Return the convex swept front-bumper region used by crossing events."""

    pre_bumper = front_bumper_segment(pre_footprint, heading_rad=pre_heading_rad)
    post_bumper = front_bumper_segment(post_footprint, heading_rad=post_heading_rad)
    swept = pre_bumper.union(post_bumper).convex_hull
    if swept.is_empty or not swept.is_valid:
        raise ValueError("Swept front bumper is empty or invalid")
    return swept
