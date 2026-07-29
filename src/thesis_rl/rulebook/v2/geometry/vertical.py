"""Frozen 2.5D compatibility predicates for Rulebook v2 geometry."""

from __future__ import annotations

from math import isfinite
from typing import Protocol


VERTICAL_COMPATIBILITY_TOLERANCE_M = 3.0

# REQ-EF-16.  A traffic control's reference point is not necessarily on the
# carriageway: Waymo's ``STOP_SIGN.position`` is the *physical sign*, mounted on
# a post above the road, whereas ``TRAFFIC_LIGHT.stop_point`` lies exactly on the
# lane surface.  Measured over 250 Waymo scenarios: signal stop points have
# dz == 0.000 for all 1319 samples, while stop signs have median dz = +2.567 m
# (p05 +1.693, p95 +2.971, max +3.668) -- i.e. a mounting height, not a level
# difference.  Bounding it separately is what lets the grade-separation
# tolerance keep meaning grade separation.  7.0 m covers overhead gantries and
# mast arms (MUTCD requires 5.2 m clearance to the bottom of an overhead sign,
# so a sign centroid reaches ~6-7 m); above that, over the carriageway, the
# object belongs to another level rather than to this road.
CONTROL_MOUNTING_HEIGHT_MAX_M = 7.0

# REQ-EF-16.  How far *below* the surface a control reference point may sit.  A
# control is never mounted below the road it governs, so this covers only lane
# elevation noise and must stay well under a grade separation: reusing the 3.0 m
# grade tolerance downward would admit a control belonging to an underpass.
# Measured over the same 250 scenarios, p01 of the stop-sign offset is -0.251 m
# and only 0.4% of samples fall below -0.5 m, so 1.0 m is generous for noise
# while still rejecting a mount on a road below this one (about -2.5 m).
CONTROL_BELOW_SURFACE_TOLERANCE_M = 1.0


def control_point_level_compatible(
    *,
    control_point_z: float,
    surface_z: float,
    tolerance_m: float = CONTROL_BELOW_SURFACE_TOLERANCE_M,
    mounting_height_max_m: float = CONTROL_MOUNTING_HEIGHT_MAX_M,
) -> bool:
    """Return whether a control reference point can belong to ``surface_z``.

    The predicate is deliberately *asymmetric*.  The control point's ground
    projection is unknown but bounded: it lies in
    ``[control_point_z - mounting_height_max_m, control_point_z]``, because a
    control is mounted at or above the surface it governs and never below it.
    The point is level-compatible when that interval reaches the surface, i.e.
    ``-tolerance <= dz <= mounting_height_max``.  ``tolerance`` is the downward
    noise allowance only, deliberately far smaller than the grade-separation
    tolerance.

    A symmetric ``abs(dz) <= VERTICAL_COMPATIBILITY_TOLERANCE_M`` test is wrong
    in both directions: it rejects a legitimately tall mount (3.8% of measured
    stop signs) and accepts a control belonging to an underpass a couple of
    metres *below* this road.
    """

    if (
        not isfinite(control_point_z)
        or not isfinite(surface_z)
        or not isfinite(tolerance_m)
        or not isfinite(mounting_height_max_m)
    ):
        raise ValueError("Control elevation compatibility requires finite elevations and bounds")
    if tolerance_m <= 0.0 or mounting_height_max_m < 0.0:
        raise ValueError("Vertical tolerance must be positive and mounting bound non-negative")
    delta_z = control_point_z - surface_z
    return -tolerance_m <= delta_z <= mounting_height_max_m


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
        raise ValueError(
            "XY coordinates and vertical tolerance must be finite; tolerance must be positive"
        )
    first_z = float(first_elevation_at_xy(x, y))
    second_z = float(second_elevation_at_xy(x, y))
    if not isfinite(first_z) or not isfinite(second_z):
        raise ValueError("Elevation accessors must return finite values")
    return abs(first_z - second_z) <= tolerance_m
