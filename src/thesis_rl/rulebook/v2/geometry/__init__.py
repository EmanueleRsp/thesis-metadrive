"""Canonical geometry primitives shared by all Rulebook v2 evaluators."""

from thesis_rl.rulebook.v2.geometry.canonical import (
    canonical_geometry_wkb,
    canonicalize_geometry,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.drivable import DrivableLaneRecord, drivable_surface_for_ego
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.lanes import (
    FootprintRouteCoordinates,
    LaneAssociation,
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    footprint_route_coordinates,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, RouteProjection
from thesis_rl.rulebook.v2.geometry.vertical import (
    ElevationAtXY,
    vertically_compatible_at_xy,
)

__all__ = [
    "ElevationAtXY",
    "DrivableLaneRecord",
    "FootprintRouteCoordinates",
    "LaneAssociation",
    "PolylineElevation",
    "RoutePolyline",
    "RouteProjection",
    "RouteLaneRecord",
    "associate_route_lane",
    "bumper_to_bumper_gap",
    "canonical_geometry_wkb",
    "canonicalize_geometry",
    "drivable_surface_for_ego",
    "stable_geometry_id",
    "oriented_bounding_box",
    "footprint_route_coordinates",
    "vertically_compatible_at_xy",
]
