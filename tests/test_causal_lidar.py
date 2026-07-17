from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from thesis_rl.envs.observations.assigned_route import (
    AssignedRouteWaypointAdapter,
    MapRouteNavigationObservation22,
)
from thesis_rl.envs.observations.causal_lidar import CausalLidarFrameBuilder
from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


class _Sensor:
    def __init__(self, rays: int):
        self.rays = rays

    def perceive(self, *_args, num_lasers: int, **_kwargs):
        return SimpleNamespace(
            cloud_points=np.full(num_lasers, 0.5, dtype=np.float32),
            detected_objects=(),
        )

    def get_surrounding_vehicles_info(self, *_args):
        return [0.0] * 16


class _Vehicle:
    position = np.asarray([2.0, 0.0, 0.0])
    heading_theta = 0.0
    heading_error = 0.0
    speed_km_h = 30.0
    steering = 0.0
    MAX_STEERING = 1.0
    yaw_rate = 0.0
    last_current_action = [(0.0, 0.0)]
    config = {
        "lidar": {"gaussian_noise": 0.0, "dropout_prob": 0.0},
        "side_detector": {"gaussian_noise": 0.0, "dropout_prob": 0.0},
        "lane_line_detector": {"gaussian_noise": 0.0, "dropout_prob": 0.0},
    }

    def __init__(self):
        self.engine = SimpleNamespace(
            np_random=np.random.default_rng(3),
            physics_world=SimpleNamespace(static_world=object(), dynamic_world=object()),
            get_sensor=lambda name: _Sensor(
                {"lidar": 240, "side_detector": 12, "lane_line_detector": 12}[name]
            ),
        )

    @staticmethod
    def convert_to_local_coordinates(point, origin):
        return np.asarray(point) - np.asarray(origin)


def test_causal_lidar_frame_builder_produces_exact_308_dimensions() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    builder = CausalLidarFrameBuilder(
        MapRouteNavigationObservation22(AssignedRouteWaypointAdapter(route)),
        RayNoiseWrapper(enabled=False),
    )

    frame = builder.build(_Vehicle())

    assert frame.shape == (308,)
    assert np.all(np.isfinite(frame))


def test_causal_lidar_builder_rejects_missing_sensor_block() -> None:
    vehicle = _Vehicle()

    class _BadSensor(_Sensor):
        def perceive(self, *_args, **_kwargs):
            return SimpleNamespace(cloud_points=np.zeros(1, dtype=np.float32), detected_objects=())

    vehicle.engine.get_sensor = lambda _name: _BadSensor(1)
    builder = CausalLidarFrameBuilder(
        MapRouteNavigationObservation22(
            AssignedRouteWaypointAdapter(RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))))
        ),
        RayNoiseWrapper(enabled=False),
    )
    with pytest.raises(ValueError, match="returned"):
        builder.build(vehicle)
