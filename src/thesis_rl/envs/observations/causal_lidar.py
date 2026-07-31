"""Causal 308D frame builder for the stacked LiDAR observation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from thesis_rl.envs.observations.assigned_route import MapRouteNavigationObservation22
from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper


@dataclass(slots=True)
class CausalLidarFrameBuilder:
    """Assemble the frozen 308D sensor/state contract without native navigation."""

    route_navigation: MapRouteNavigationObservation22
    ray_noise: RayNoiseWrapper
    distance_m: float = 50.0
    num_lidar_rays: int = 240
    num_side_rays: int = 12
    num_lane_rays: int = 12
    num_nearby_vehicles: int = 4

    FRAME_DIM = 308

    def __post_init__(self) -> None:
        if self.distance_m <= 0.0:
            raise ValueError("Causal LiDAR distance must be positive")
        if (self.num_lidar_rays, self.num_side_rays, self.num_lane_rays) != (240, 12, 12):
            raise ValueError("Causal LiDAR ray dimensions are frozen at 240/12/12")
        if self.num_nearby_vehicles != 4:
            raise ValueError("Causal LiDAR nearby-vehicle count is frozen at four")

    def __call__(self, vehicle: object) -> np.ndarray:
        return self.build(vehicle)

    def build(self, vehicle: object) -> np.ndarray:
        engine = getattr(vehicle, "engine", None)
        if engine is None:
            raise RuntimeError("Causal LiDAR frame requires vehicle.engine")
        config = getattr(vehicle, "config", {})
        if not callable(getattr(config, "get", None)):
            raise ValueError("Vehicle sensor configuration must be a mapping")
        self.ray_noise.validate_native_noise_disabled(
            {
                name: config.get(name, {})
                for name in ("lidar", "side_detector", "lane_line_detector")
            }
        )
        ego = self._ego_state(vehicle)
        navigation = self.route_navigation.observe(vehicle)
        side = self._ray_block(engine, vehicle, "side_detector", self.num_side_rays, "static_world")
        lane = self._ray_block(
            engine, vehicle, "lane_line_detector", self.num_lane_rays, "static_world"
        )
        lidar, nearby = self._lidar_blocks(engine, vehicle)
        frame = np.concatenate((ego, navigation, side, lane, nearby, lidar)).astype(np.float32)
        if frame.shape != (self.FRAME_DIM,) or not np.all(np.isfinite(frame)):
            raise ValueError("Causal LiDAR frame violates the finite 308D contract")
        if not np.all((-1.0 <= frame) & (frame <= 1.0)):
            raise ValueError("Causal LiDAR frame violates the normalized range contract")
        return frame

    @staticmethod
    def _ego_state(vehicle: object) -> np.ndarray:
        speed = float(getattr(vehicle, "speed_km_h")) / 120.0
        steering = float(getattr(vehicle, "steering")) / max(
            float(getattr(vehicle, "MAX_STEERING")), 1e-6
        )
        actions = getattr(vehicle, "last_current_action", ())
        previous = actions[-2] if len(actions) >= 2 else (0.0, 0.0)
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0)) / 2.0
        return np.asarray(
            [
                float(np.clip(getattr(vehicle, "heading_error", 0.0), -1.0, 1.0)),
                float(np.clip(speed, 0.0, 1.0)),
                float(np.clip(steering, -1.0, 1.0)),
                float(np.clip(float(previous[1]), -1.0, 1.0)),
                float(np.clip(float(previous[0]), -1.0, 1.0)),
                float(np.clip(yaw_rate, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _ray_block(
        self, engine: object, vehicle: object, sensor_name: str, rays: int, world_name: str
    ) -> np.ndarray:
        sensor = engine.get_sensor(sensor_name)
        physics_world = getattr(getattr(vehicle, "engine", None), "physics_world")
        world = getattr(physics_world, world_name)
        result = sensor.perceive(
            vehicle,
            world,
            num_lasers=rays,
            distance=self.distance_m,
            show=False,
        )
        cloud = np.asarray(getattr(result, "cloud_points", result), dtype=np.float32).reshape(-1)
        if cloud.size != rays:
            raise ValueError(f"{sensor_name} returned {cloud.size} rays, expected {rays}")
        return self.ray_noise.perturb(cloud, getattr(engine, "np_random"))

    def _lidar_blocks(self, engine: object, vehicle: object) -> tuple[np.ndarray, np.ndarray]:
        sensor = engine.get_sensor("lidar")
        physics_world = getattr(getattr(vehicle, "engine", None), "physics_world")
        result = sensor.perceive(
            vehicle,
            physics_world.dynamic_world,
            num_lasers=self.num_lidar_rays,
            distance=self.distance_m,
            show=False,
        )
        # `metadrive.component.sensors.lidar.Lidar.perceive` returns a plain
        # `(cloud_points, detected_objects)` tuple, not the `detect_result`
        # namedtuple `DistanceDetector.perceive` (used by `_ray_block`)
        # returns; it has no `.cloud_points`/`.detected_objects` attributes,
        # so attribute access must not be attempted here.
        raw_cloud, detected = result
        cloud = np.asarray(raw_cloud, dtype=np.float32).reshape(-1)
        if cloud.size != self.num_lidar_rays:
            raise ValueError(f"lidar returned {cloud.size} rays, expected {self.num_lidar_rays}")
        nearby = sensor.get_surrounding_vehicles_info(
            vehicle, detected, self.distance_m, self.num_nearby_vehicles, False
        )
        nearby_array = np.asarray(nearby, dtype=np.float32).reshape(-1)
        if nearby_array.size != self.num_nearby_vehicles * 4:
            raise ValueError("lidar nearby-vehicle block must have 16 values")
        return self.ray_noise.perturb(cloud, getattr(engine, "np_random")), np.clip(
            nearby_array, -1.0, 1.0
        )
