from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.envs.observations.assigned_route import (
    AssignedRouteWaypointAdapter,
    MapRouteNavigationObservation22,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


class _Vehicle:
    position = np.asarray([2.0, 1.0, 0.0])
    heading_theta = 0.0

    @staticmethod
    def convert_to_local_coordinates(point: np.ndarray, origin: np.ndarray) -> np.ndarray:
        return np.asarray(point) - np.asarray(origin)


def test_assigned_route_waypoints_use_frozen_polyline_and_fixed_spacing() -> None:
    adapter = AssignedRouteWaypointAdapter(
        RoutePolyline(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
        num_waypoints=3,
        spacing_m=5.0,
    )

    waypoints = adapter.observe(_Vehicle())

    assert waypoints.shape == (3, 2)
    np.testing.assert_allclose(waypoints, [[0.0, -1.0], [5.0, -1.0], [10.0, -1.0]])


def test_assigned_route_waypoints_fail_closed_without_local_transform() -> None:
    adapter = AssignedRouteWaypointAdapter(RoutePolyline(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match="world-to-local"):
        adapter.observe(type("Vehicle", (), {"position": (1.0, 0.0, 0.0)})())


def test_map_route_navigation_observation_has_exact_22_dimensions() -> None:
    navigation = MapRouteNavigationObservation22(
        AssignedRouteWaypointAdapter(
            RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))),
        )
    )

    observation = navigation.observe(_Vehicle())

    assert observation.shape == (22,)
    assert np.all(np.isfinite(observation))
    np.testing.assert_allclose(observation[-2:], [0.25, 0.0])
