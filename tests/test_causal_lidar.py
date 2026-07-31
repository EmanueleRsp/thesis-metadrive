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
    """Fakes both real MetaDrive sensor return contracts.

    `DistanceDetector.perceive` (used by `side_detector`/`lane_line_detector`)
    returns a `detect_result` namedtuple exposing `.cloud_points`. `Lidar.
    perceive` (used by the `lidar` sensor) returns a plain
    `(cloud_points, detected_objects)` tuple with no such attribute — see
    `CausalLidarFrameBuilder._lidar_blocks`, which must unpack it positionally
    rather than via `getattr(result, "cloud_points", result)`.
    """

    def __init__(self, rays: int, *, is_lidar: bool = False):
        self.rays = rays
        self.is_lidar = is_lidar

    def perceive(self, *_args, num_lasers: int, **_kwargs):
        cloud = np.full(num_lasers, 0.5, dtype=np.float32)
        if self.is_lidar:
            return cloud, ()
        return SimpleNamespace(cloud_points=cloud, detected_objects=())

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
                {"lidar": 240, "side_detector": 12, "lane_line_detector": 12}[name],
                is_lidar=(name == "lidar"),
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


class _MappingLikeConfig:
    """Duck-typed stand-in for `metadrive.utils.config.Config`.

    Regression for the discovery (2026-07-30) that real MetaDrive vehicles
    carry a `Config` object, not a `dict`, so `isinstance(config, dict)`
    rejected every real vehicle at runtime while unit tests only ever passed a
    plain dict. `CausalLidarFrameBuilder.build` now duck-types on `.get`.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def test_causal_lidar_builder_accepts_mapping_like_config_not_a_dict_subclass() -> None:
    vehicle = _Vehicle()
    vehicle.config = _MappingLikeConfig(dict(_Vehicle.config))
    builder = CausalLidarFrameBuilder(
        MapRouteNavigationObservation22(
            AssignedRouteWaypointAdapter(RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))))
        ),
        RayNoiseWrapper(enabled=False),
    )

    frame = builder.build(vehicle)

    assert frame.shape == (308,)


def test_causal_lidar_builder_rejects_non_mapping_config() -> None:
    vehicle = _Vehicle()
    vehicle.config = object()
    builder = CausalLidarFrameBuilder(
        MapRouteNavigationObservation22(
            AssignedRouteWaypointAdapter(RoutePolyline(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))))
        ),
        RayNoiseWrapper(enabled=False),
    )
    with pytest.raises(ValueError, match="mapping"):
        builder.build(vehicle)


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
