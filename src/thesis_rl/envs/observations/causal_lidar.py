"""Causal 310D frame builder for the stacked LiDAR observation.

310, not the previously frozen 308: `OBS-LIDAR-V2.0.2` adds the posted speed
limit of the ego's associated route lane and its explicit availability flag,
required by `RULEBOOK-V5.1`'s `speed_limit` sub-rule. Checkpoint compatibility is
intentionally broken (`DEC-RB51-001`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from thesis_rl.envs.observations.assigned_route import MapRouteNavigationObservation22
from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane
from thesis_rl.rulebook.v2.transition import associated_speed_limit_mps


# The scale the ego speed channel already uses, so the agent can compare its own
# speed with the limit without a change of units.
LIDAR_SPEED_SCALE_KMH = 120.0


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
    # The assigned route's lanes, carrying the posted limit admitted by
    # provenance. Empty is legitimate and is not an error: it makes the feature
    # unavailable, which is the correct reading on PG and on any record whose
    # lanes carry no real-map limit.
    route_lanes: tuple[RouteLaneRecord, ...] = field(default_factory=tuple)

    FRAME_DIM = 310

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
        ego = np.concatenate((self._ego_state(vehicle), self._speed_limit_block(vehicle)))
        navigation = self.route_navigation.observe(vehicle)
        side = self._ray_block(engine, vehicle, "side_detector", self.num_side_rays, "static_world")
        lane = self._ray_block(
            engine, vehicle, "lane_line_detector", self.num_lane_rays, "static_world"
        )
        lidar, nearby = self._lidar_blocks(engine, vehicle)
        frame = np.concatenate((ego, navigation, side, lane, nearby, lidar)).astype(np.float32)
        if frame.shape != (self.FRAME_DIM,) or not np.all(np.isfinite(frame)):
            raise ValueError("Causal LiDAR frame violates the finite 310D contract")
        if not np.all((-1.0 <= frame) & (frame <= 1.0)):
            raise ValueError("Causal LiDAR frame violates the normalized range contract")
        return frame

    def _speed_limit_block(self, vehicle: object) -> np.ndarray:
        """The posted limit and its explicit availability flag.

        Two values, not one: a sentinel inside the normalized channel would be
        indistinguishable from a real limit at that value. The limit is resolved
        through the **rulebook's own** lookup, because RULEBOOK-V5.0 §7 requires
        the "unavailable" encoding to fire under exactly the condition that makes
        the sub-rule inapplicable -- otherwise the observation would assert a norm
        no cost backs, which is worse than saying nothing.

        MetaDrive's own `lane.speed_limit` is deliberately **not** read here: on
        PG it is a constructor default written under a `_kmh` key in m/s, and on
        Waymo it comes through `ScenarioLane`, whose cap ADR-068 also prohibits.
        """

        if not self.route_lanes:
            return np.zeros(2, dtype=np.float32)
        position = getattr(vehicle, "position", None)
        heading = getattr(vehicle, "heading_theta", None)
        if position is None or heading is None:
            return np.zeros(2, dtype=np.float32)
        association = associate_route_lane(
            position_xy=(float(position[0]), float(position[1])),
            position_z=float(position[2]) if len(position) > 2 else 0.0,
            heading_rad=float(heading),
            route_lanes=self.route_lanes,
        )
        limit_mps = associated_speed_limit_mps(self.route_lanes, association)
        if limit_mps is None:
            return np.zeros(2, dtype=np.float32)
        normalized = float(np.clip(limit_mps * 3.6 / LIDAR_SPEED_SCALE_KMH, 0.0, 1.0))
        return np.asarray((normalized, 1.0), dtype=np.float32)

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
