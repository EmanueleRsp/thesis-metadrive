"""Canonical geometry primitives shared by all Rulebook v2 evaluators."""

from thesis_rl.rulebook.v2.geometry.canonical import (
    canonical_geometry_wkb,
    canonicalize_geometry,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.drivable import DrivableLaneRecord, drivable_surface_for_ego
from thesis_rl.rulebook.v2.geometry.controls import ControlLine, derive_control_line
from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    OccupancyInterval,
    deterministic_convex_decomposition,
    predict_occupancy_interval,
)
from thesis_rl.rulebook.v2.geometry.ctrv import (
    YawRateEstimate,
    estimate_yaw_rate,
    predict_actor_occupancy_interval,
    predict_conflict_zone_occupancy_intervals,
    predict_rotating_occupancy_interval,
    predict_vehicle_occupancy_interval,
    propagate_ctrv_footprint,
    unwrap_headings,
    wrap_heading_delta,
)
from thesis_rl.rulebook.v2.geometry.conflict_zones import (
    ConflictZoneCandidate,
    CrosswalkZoneCandidate,
    MovementCorridor,
    RouteConflictZoneCandidate,
    attach_route_intervals,
    build_crosswalk_conflict_zone_candidates,
    build_vehicle_conflict_zone_candidates,
    route_interval_for_zone,
    select_first_ahead_or_occupied_zone,
)
from thesis_rl.rulebook.v2.geometry.footprint import (
    front_bumper_segment,
    oriented_bounding_box,
    swept_front_bumper,
)
from thesis_rl.rulebook.v2.geometry.lanes import (
    derive_lane_movement_key,
    FootprintRouteCoordinates,
    LaneAssociation,
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    footprint_route_coordinates,
)
from thesis_rl.rulebook.v2.geometry.route import (
    RoutePolyline,
    RouteProjectionDiagnostics,
    RouteProjection,
    build_assigned_route_polyline,
)
from thesis_rl.rulebook.v2.geometry.vertical import (
    ElevationAtXY,
    vertically_compatible_at_xy,
)

__all__ = [
    "ElevationAtXY",
    "DrivableLaneRecord",
    "ControlLine",
    "OccupancyInterval",
    "ConflictZoneCandidate",
    "CrosswalkZoneCandidate",
    "RouteConflictZoneCandidate",
    "FootprintRouteCoordinates",
    "LaneAssociation",
    "PolylineElevation",
    "RoutePolyline",
    "RouteProjection",
    "RouteProjectionDiagnostics",
    "build_assigned_route_polyline",
    "RouteLaneRecord",
    "derive_lane_movement_key",
    "MovementCorridor",
    "associate_route_lane",
    "bumper_to_bumper_gap",
    "build_vehicle_conflict_zone_candidates",
    "build_crosswalk_conflict_zone_candidates",
    "attach_route_intervals",
    "canonical_geometry_wkb",
    "canonicalize_geometry",
    "drivable_surface_for_ego",
    "deterministic_convex_decomposition",
    "derive_control_line",
    "front_bumper_segment",
    "stable_geometry_id",
    "oriented_bounding_box",
    "predict_occupancy_interval",
    "YawRateEstimate",
    "estimate_yaw_rate",
    "predict_actor_occupancy_interval",
    "predict_conflict_zone_occupancy_intervals",
    "predict_rotating_occupancy_interval",
    "predict_vehicle_occupancy_interval",
    "propagate_ctrv_footprint",
    "unwrap_headings",
    "wrap_heading_delta",
    "route_interval_for_zone",
    "select_first_ahead_or_occupied_zone",
    "swept_front_bumper",
    "footprint_route_coordinates",
    "vertically_compatible_at_xy",
]
