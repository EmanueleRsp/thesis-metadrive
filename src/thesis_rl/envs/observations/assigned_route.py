"""Causal waypoint adapter backed by frozen assigned route geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi

import numpy as np

from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


@dataclass(frozen=True, slots=True)
class AssignedRouteWaypointAdapter:
    """Produce fixed-spacing route waypoints without native navigation state."""

    route: RoutePolyline
    num_waypoints: int = 10
    spacing_m: float = 5.0

    def __post_init__(self) -> None:
        if self.num_waypoints <= 0 or not isfinite(self.spacing_m) or self.spacing_m <= 0.0:
            raise ValueError("Assigned route waypoint configuration is invalid")

    def observe(self, vehicle: object) -> np.ndarray:
        position = np.asarray(getattr(vehicle, "position"), dtype=float).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            raise ValueError("Vehicle position must contain finite XY coordinates")
        position_z = float(position[2]) if position.size >= 3 else None
        projection = self.route.project(
            (float(position[0]), float(position[1])), position_z=position_z
        )
        points = []
        for index in range(self.num_waypoints):
            world_point = self.route.point_at(projection.s_m + index * self.spacing_m)
            convert = getattr(vehicle, "convert_to_local_coordinates", None)
            if not callable(convert):
                raise ValueError("Vehicle must expose causal world-to-local conversion")
            local = np.asarray(
                convert(np.asarray(world_point[:2], dtype=float), position[:2]), dtype=float
            ).reshape(-1)
            if local.size < 2 or not np.all(np.isfinite(local[:2])):
                raise ValueError("Assigned route local waypoint must be finite")
            points.append(local[:2])
        return np.asarray(points, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MapRouteNavigationObservation22:
    """The specified 22-dimensional map-route navigation block."""

    waypoint_adapter: AssignedRouteWaypointAdapter
    lateral_scale_m: float = 4.0
    waypoint_scale_m: float = 50.0

    OUTPUT_DIM = 22

    def __post_init__(self) -> None:
        if self.waypoint_adapter.num_waypoints != 10:
            raise ValueError("MapRouteNavigationObservation22 requires ten waypoints")
        if (
            not isfinite(self.lateral_scale_m)
            or self.lateral_scale_m <= 0.0
            or not isfinite(self.waypoint_scale_m)
            or self.waypoint_scale_m <= 0.0
        ):
            raise ValueError("Route scales must be positive and finite")

    def observe(self, vehicle: object) -> np.ndarray:
        waypoints = self.waypoint_adapter.observe(vehicle) / self.waypoint_scale_m
        position = np.asarray(getattr(vehicle, "position"), dtype=float).reshape(-1)
        projection = self.waypoint_adapter.route.project(
            (float(position[0]), float(position[1])),
            position_z=float(position[2]) if position.size >= 3 else None,
        )
        heading = float(getattr(vehicle, "heading_theta"))
        route_heading = atan2(projection.tangent_xy[1], projection.tangent_xy[0])
        lateral = float(np.clip(projection.lateral_distance_m / self.lateral_scale_m, -1.0, 1.0))
        heading_error = (heading - route_heading + pi) % (2.0 * pi) - pi
        result = np.concatenate(
            [waypoints.reshape(-1), np.asarray([lateral, heading_error / pi], dtype=np.float32)]
        ).astype(np.float32)
        if result.shape != (self.OUTPUT_DIM,) or not np.all(np.isfinite(result)):
            raise ValueError("Map route navigation output does not satisfy the 22D contract")
        return result
