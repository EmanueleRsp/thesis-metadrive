"""Physical eligibility gates for the OBS-V1.2 semantic tracker baseline.

The helpers in this module intentionally separate physical admission from the
ideal semantic measurement read after admission.  They never use MetaDrive's
LiDAR broad-phase object list as visibility evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, hypot, pi, sin
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class FirstHitLidarSweep:
    """Resolved first-hit information from one planar LiDAR sweep."""

    actor_ids: frozenset[str]
    hit_count: int
    beam_count: int

    @property
    def detected_object_ids(self) -> frozenset[str]:
        """Compatibility alias for the stable object-ID admission set."""

        return self.actor_ids


def _object_id(value: Any) -> str | None:
    identifier = getattr(value, "id", None)
    return identifier if isinstance(identifier, str) and identifier else None


def _scenario_id_mapping(engine: Any) -> Mapping[str, str] | None:
    """Return MetaDrive's runtime-object to source-actor id mapping, if any."""

    traffic_manager = getattr(engine, "traffic_manager", None)
    mapping = getattr(traffic_manager, "obj_id_to_scenario_id", None)
    return mapping if isinstance(mapping, Mapping) else None


def _stable_object_id(mapping: Mapping[str, str] | None, value: Any) -> str | None:
    """Resolve a MetaDrive object to the identity used by ``ActorSnapshot``.

    ScenarioNet replay spawns every actor under a random MetaDrive object name
    unless ``force_reuse_object_name`` is set, while the Rulebook's live adapter
    (`metadrive_live._stable_actor_id`) publishes the *source* actor id from
    ``traffic_manager.obj_id_to_scenario_id``.  Admission compares the two, so
    the sweep must apply the same normalization or the sets are disjoint by
    construction and no actor is ever admitted (OBS-AUDIT-FIX-001, REQ-AF-01).
    Objects outside the mapping (PG traffic, static props) keep their object id,
    which is also what the live adapter does.
    """

    identifier = _object_id(value)
    if identifier is None or mapping is None:
        return identifier
    mapped = mapping.get(identifier)
    return mapped if isinstance(mapped, str) and mapped else identifier


class FirstHitLidarAdapter:
    """Run the approved 240-beam, 50 m, 1.2 m physical LiDAR sweep."""

    def __init__(
        self,
        *,
        num_beams: int = 240,
        distance_m: float = 50.0,
        height_m: float = 1.2,
    ) -> None:
        if num_beams != 240:
            raise ValueError("OBS-V1.2 fixes the LiDAR beam count at 240")
        if distance_m != 50.0:
            raise ValueError("OBS-V1.2 fixes the LiDAR range at 50 m")
        if height_m != 1.2:
            raise ValueError("OBS-V1.2 fixes the LiDAR height at 1.2 m")
        self.num_beams = num_beams
        self.distance_m = distance_m
        self.height_m = height_m

    def sweep(self, vehicle: object) -> FirstHitLidarSweep:
        """Resolve only actual closest Bullet hits to stable MetaDrive object IDs."""

        engine = getattr(vehicle, "engine", None)
        get_sensor = getattr(engine, "get_sensor", None)
        physics_world = getattr(engine, "physics_world", None)
        dynamic_world = getattr(physics_world, "dynamic_world", None)
        if not callable(get_sensor) or dynamic_world is None:
            raise RuntimeError("OBS-V1.2 LiDAR requires vehicle.engine dynamic Bullet world")

        sensor = get_sensor("lidar")
        if sensor is None:
            raise RuntimeError("OBS-V1.2 LiDAR sensor is unavailable")

        # Lidar.perceive() replaces DistanceDetector's hit-result list with a
        # broad-phase list. Calling the base implementation retains the actual
        # first Bullet hit for every active beam.
        from metadrive.component.sensors.distance_detector import DistanceDetector
        from metadrive.utils.utils import get_object_from_node

        result = DistanceDetector.perceive(
            sensor,
            vehicle,
            dynamic_world,
            num_lasers=self.num_beams,
            distance=self.distance_m,
            height=self.height_m,
            show=False,
        )
        cloud_points = tuple(getattr(result, "cloud_points", ()))
        if len(cloud_points) != self.num_beams:
            raise RuntimeError(
                f"OBS-V1.2 LiDAR returned {len(cloud_points)} beams, expected {self.num_beams}"
            )

        # Fetch the mapping once: MetaDrive rebuilds it on every property read.
        mapping = _scenario_id_mapping(engine)
        ego_object_id = _object_id(vehicle)
        ego_id = _stable_object_id(mapping, vehicle)
        actor_ids: set[str] = set()
        hit_count = 0
        for hit in getattr(result, "detected_objects", ()):
            has_hit = getattr(hit, "hasHit", None)
            if callable(has_hit) and not has_hit():
                continue
            node_getter = getattr(hit, "getNode", None)
            if not callable(node_getter):
                continue
            node = node_getter()
            if node is None:
                continue
            hit_count += 1
            candidate = get_object_from_node(node)
            candidate_object_id = _object_id(candidate)
            if candidate_object_id is None or candidate_object_id == ego_object_id:
                continue
            candidate_id = _stable_object_id(mapping, candidate)
            if candidate_id is not None and candidate_id != ego_id:
                actor_ids.add(candidate_id)
        return FirstHitLidarSweep(frozenset(actor_ids), hit_count, self.num_beams)


@dataclass(frozen=True, slots=True)
class SignalVisibility:
    """Current physical observability of one mapped traffic signal."""

    physical_id: str
    visible: bool


class SymbolicSignalVisibilityAdapter:
    """Range/FOV/occlusion gate for a virtual traffic-light head anchor.

    OBS-V1.2 SS6.2 documents 80 m / 65 degrees / 1.2 m as the baseline signal
    camera geometry; ADR-045 approves deviating from these defaults through
    ``conf/obs/semantic_v3.yaml`` (``signal_range_m`` /
    ``signal_fov_degrees`` / ``signal_camera_height_m``) for experiments that
    intentionally study a different sensor placement or field of view.
    """

    def __init__(
        self,
        *,
        range_m: float = 80.0,
        horizontal_fov_deg: float = 65.0,
        camera_height_m: float = 1.2,
        ray_tolerance: float = 1.0e-4,
    ) -> None:
        if range_m <= 0.0 or horizontal_fov_deg <= 0.0 or horizontal_fov_deg > 360.0:
            raise ValueError(
                "Signal camera range must be positive and FOV must be in (0, 360] degrees"
            )
        self.range_m = range_m
        self.horizontal_fov_rad = horizontal_fov_deg * pi / 180.0
        self.camera_height_m = camera_height_m
        self.ray_tolerance = ray_tolerance

    def visible_ids(self, vehicle: object, physical_ids: Iterable[str]) -> dict[str, bool]:
        """Return visibility for requested physical IDs, failing closed if unmapped."""

        engine = getattr(vehicle, "engine", None)
        manager = getattr(engine, "light_manager", None)
        scenario_to_object = getattr(manager, "_scenario_id_to_obj_id", None)
        spawned = getattr(manager, "spawned_objects", None)
        if not isinstance(scenario_to_object, Mapping) or not isinstance(spawned, Mapping):
            raise RuntimeError("OBS-V1.2 signals require live light-manager object mappings")
        result: dict[str, bool] = {}
        for physical_id in physical_ids:
            object_id = scenario_to_object.get(physical_id)
            light = spawned.get(object_id) if object_id is not None else None
            result[str(physical_id)] = light is not None and self._is_visible(vehicle, light)
        return result

    def _is_visible(self, vehicle: object, light: object) -> bool:
        origin = self._camera_origin(vehicle)
        anchor = self._light_head_anchor(light)
        dx, dy = anchor[0] - origin[0], anchor[1] - origin[1]
        planar_distance = hypot(dx, dy)
        if planar_distance > self.range_m or planar_distance <= self.ray_tolerance:
            return False
        heading = float(getattr(vehicle, "heading_theta"))
        alignment = (dx * cos(heading) + dy * sin(heading)) / planar_distance
        if acos(max(-1.0, min(1.0, alignment))) > self.horizontal_fov_rad / 2.0:
            return False
        return not self._has_blocker(vehicle, origin, anchor, ignored_object_id=_object_id(light))

    def _camera_origin(self, vehicle: object) -> tuple[float, float, float]:
        position = tuple(getattr(vehicle, "position", ()))
        if len(position) < 2:
            raise RuntimeError("OBS-V1.2 signal camera requires a 2D ego position")
        return float(position[0]), float(position[1]), self.camera_height_m

    @staticmethod
    def _light_head_anchor(light: object) -> tuple[float, float, float]:
        position = tuple(getattr(light, "position", ()))
        origin = getattr(light, "origin", None)
        get_z = getattr(origin, "getZ", None)
        if len(position) < 2 or not callable(get_z):
            raise RuntimeError("OBS-V1.2 signal requires a positioned physical light object")
        from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

        return (
            float(position[0]),
            float(position[1]),
            float(get_z()) + BaseTrafficLight.TRAFFIC_LIGHT_HEIGHT,
        )

    def _has_blocker(
        self,
        vehicle: object,
        origin: tuple[float, float, float],
        anchor: tuple[float, float, float],
        *,
        ignored_object_id: str | None,
    ) -> bool:
        engine = getattr(vehicle, "engine", None)
        physics = getattr(engine, "physics_world", None)
        if physics is None:
            raise RuntimeError("OBS-V1.2 signal ray requires MetaDrive physics worlds")
        from metadrive.constants import CollisionGroup
        from metadrive.utils.coordinates_shift import panda_vector
        from metadrive.utils.utils import get_object_from_node

        for world_name in ("dynamic_world", "static_world"):
            world = getattr(physics, world_name, None)
            ray_test_all = getattr(world, "rayTestAll", None)
            if not callable(ray_test_all):
                continue
            result = ray_test_all(
                panda_vector(origin), panda_vector(anchor), CollisionGroup.can_be_lidar_detected()
            )
            for hit in sorted(result.getHits(), key=lambda item: item.getHitFraction()):
                if float(hit.getHitFraction()) >= 1.0 - self.ray_tolerance:
                    continue
                candidate = get_object_from_node(hit.getNode())
                candidate_id = _object_id(candidate)
                if candidate_id == _object_id(vehicle) or candidate_id == ignored_object_id:
                    continue
                return True
        return False


def first_hit_lidar_sweep(vehicle: object) -> FirstHitLidarSweep:
    """Convenience entry point for one OBS-V1.2 LiDAR eligibility sweep."""

    return FirstHitLidarAdapter().sweep(vehicle)


def mapped_signal_visibility(
    vehicle: object,
    physical_ids: str | Iterable[str],
    *,
    range_m: float = 80.0,
    horizontal_fov_deg: float = 65.0,
    camera_height_m: float = 1.2,
) -> SignalVisibility | dict[str, bool]:
    """Convenience entry point for the OBS-V1.2 symbolic signal gate.

    A single physical ID returns its :class:`SignalVisibility`; an iterable
    returns the compact ID-to-boolean mapping used by batched control assembly.
    The keyword parameters let callers thread ``conf/obs/semantic_v3.yaml``
    through to this gate. Defaults reproduce the OBS-V1.2 SS6.2 baseline
    (80 m / 65 degrees / 1.2 m); ADR-045 approves overriding them via config
    for experiments that deliberately study a different sensor geometry.
    """

    adapter = SymbolicSignalVisibilityAdapter(
        range_m=range_m, horizontal_fov_deg=horizontal_fov_deg, camera_height_m=camera_height_m
    )
    if isinstance(physical_ids, str):
        return SignalVisibility(
            physical_ids, adapter.visible_ids(vehicle, (physical_ids,))[physical_ids]
        )
    return adapter.visible_ids(vehicle, physical_ids)
