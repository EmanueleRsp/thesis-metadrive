"""Canonical geometry primitives shared by all Rulebook v2 evaluators."""

from thesis_rl.rulebook.v2.geometry.canonical import (
    canonical_geometry_wkb,
    canonicalize_geometry,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.vertical import (
    ElevationAtXY,
    vertically_compatible_at_xy,
)

__all__ = [
    "ElevationAtXY",
    "PolylineElevation",
    "canonical_geometry_wkb",
    "canonicalize_geometry",
    "stable_geometry_id",
    "oriented_bounding_box",
    "vertically_compatible_at_xy",
]
