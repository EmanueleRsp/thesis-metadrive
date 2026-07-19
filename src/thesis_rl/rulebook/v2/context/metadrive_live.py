"""Explicit MetaDrive-to-Rulebook live actor normalization.

This module contains only source-bound extraction.  It deliberately does not
construct a Rulebook evaluator or guess contact/signal state from diagnostics;
those providers remain explicit inputs to the live adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, hypot, isfinite
from typing import Any

from thesis_rl.rulebook.v2.context.live_adapter import actor_snapshot_from_payload
from thesis_rl.rulebook.v2.events import ContactOnsetBuffer
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


_LIVE_SIGNAL_STATE_MAP = {
    "TRAFFIC_LIGHT_GREEN": "GREEN",
    "TRAFFIC_LIGHT_YELLOW": "YELLOW",
    "TRAFFIC_LIGHT_RED": "RED",
    "LANE_STATE_GO": "GREEN",
    "LANE_STATE_ARROW_GO": "GREEN",
    "LANE_STATE_CAUTION": "YELLOW",
    "LANE_STATE_ARROW_CAUTION": "YELLOW",
    "LANE_STATE_FLASHING_CAUTION": "FLASHING_YELLOW",
    "LANE_STATE_STOP": "RED",
    "LANE_STATE_ARROW_STOP": "RED",
    "LANE_STATE_FLASHING_STOP": "RED",
    "TRAFFIC_LIGHT_UNKNOWN": "UNKNOWN",
    "LANE_STATE_UNKNOWN": "UNKNOWN",
}


def _sequence_pair(value: object, *, field_name: str) -> tuple[float, float]:
    """Read one finite 2D MetaDrive vector without relying on observations."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            raise ValueError(f"MetaDrive {field_name} must contain two values")
        return float(value[0]), float(value[1])
    try:
        return float(value[0]), float(value[1])  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"MetaDrive {field_name} must contain two values") from error


def _stable_actor_id(env: Any, actor: Any) -> str:
    actor_id = getattr(actor, "id", None)
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError("MetaDrive live actor must expose a non-empty id")
    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    mapping = getattr(traffic_manager, "obj_id_to_scenario_id", {})
    if isinstance(mapping, Mapping):
        scenario_id = mapping.get(actor_id)
        if isinstance(scenario_id, str) and scenario_id:
            return scenario_id
    return actor_id


def _actor_class(actor: Any) -> ActorClass:
    """Map only known MetaDrive object classes to the canonical taxonomy."""

    name = type(actor).__name__.lower()
    if "pedestrian" in name:
        return ActorClass.PEDESTRIAN
    if "cyclist" in name or "bicycl" in name:
        return ActorClass.CYCLIST
    if "vehicle" in name or name.endswith("car") or name.endswith("truck"):
        return ActorClass.VEHICLE
    if any(token in name for token in ("static", "building", "barrier", "cone")):
        return ActorClass.STATIC_COLLIDABLE
    raise ValueError(f"Unsupported MetaDrive live actor class: {type(actor).__name__!r}")


def _actor_speed_cap(actor: Any, actor_class: ActorClass) -> float | None:
    if actor_class is not ActorClass.VEHICLE:
        return None
    value = getattr(actor, "max_speed_m_s", None)
    if value is None:
        raise ValueError(
            f"MetaDrive vehicle {getattr(actor, 'id', '<unknown>')!r} has no speed cap"
        )
    return float(value)


def _actor_dimensions(actor: Any) -> tuple[float, float]:
    length = getattr(actor, "LENGTH", None)
    width = getattr(actor, "WIDTH", None)
    if length is None or width is None:
        raise ValueError(f"MetaDrive actor {getattr(actor, 'id', '<unknown>')!r} has no dimensions")
    return float(length), float(width)


def _live_lane_id(actor: Any) -> str | None:
    lane = getattr(actor, "lane", None)
    if lane is None:
        return None
    for name in ("index", "lane_id", "id"):
        value = getattr(lane, name, None)
        if value is not None:
            text = str(value)
            if text:
                return text
    return None


def actor_snapshot_from_metadrive(env: Any, actor: Any) -> ActorSnapshot:
    """Build one canonical actor snapshot from a live MetaDrive object.

    The function requires pose, velocity, dimensions, identity, and (for
    vehicles) a configured speed cap.  It never reads ScenarioDescription
    future tracks or policy observations.
    """

    actor_class = _actor_class(actor)
    position = _sequence_pair(getattr(actor, "position", None), field_name="position")
    velocity = _sequence_pair(getattr(actor, "velocity", None), field_name="velocity")
    get_z = getattr(actor, "get_z", None)
    if not callable(get_z):
        raise ValueError(f"MetaDrive actor {getattr(actor, 'id', '<unknown>')!r} has no get_z()")
    heading = getattr(actor, "heading_theta", None)
    if heading is None:
        heading_vector = _sequence_pair(getattr(actor, "heading", None), field_name="heading")
        heading = atan2(heading_vector[1], heading_vector[0])
    length, width = _actor_dimensions(actor)
    payload = {
        "actor_id": _stable_actor_id(env, actor),
        "actor_class": actor_class,
        "position_xy": position,
        "position_z": float(get_z()),
        "heading_rad": float(heading),
        "velocity_xy": velocity,
        "length_m": length,
        "width_m": width,
        "live_lane_id": _live_lane_id(actor),
        "configured_speed_cap_mps": _actor_speed_cap(actor, actor_class),
    }
    return actor_snapshot_from_payload(payload)


def live_vehicle_objects(env: Any) -> tuple[Any, ...]:
    """Return deterministic, de-duplicated live traffic objects."""

    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    vehicles = getattr(traffic_manager, "vehicles", None)
    if vehicles is None:
        raise ValueError("MetaDrive environment has no traffic_manager.vehicles collection")
    values = tuple(vehicles.values()) if isinstance(vehicles, Mapping) else tuple(vehicles)
    by_id: dict[str, Any] = {}
    for actor in values:
        actor_id = getattr(actor, "id", None)
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError("MetaDrive traffic object has no stable id")
        by_id[actor_id] = actor
    return tuple(by_id[key] for key in sorted(by_id))


def live_actor_objects(env: Any) -> tuple[Any, ...]:
    """Collect supported live actors from MetaDrive's public object registry.

    The traffic manager exposes vehicles, while ScenarioNet VRUs such as
    cyclists are exposed through ``engine.get_objects()``.  Both collections
    are merged deterministically by runtime ID.  Traffic-light and map
    objects are intentionally excluded because they are represented by the
    dedicated signal/static providers.
    """

    engine = getattr(env, "engine", None)
    candidates: list[Any] = list(live_vehicle_objects(env))
    get_objects = getattr(engine, "get_objects", None)
    if callable(get_objects):
        objects = get_objects()
        values = objects.values() if isinstance(objects, Mapping) else ()
        candidates.extend(values)
    by_id: dict[str, Any] = {}
    for actor in candidates:
        name = type(actor).__name__.lower()
        if "trafficlight" in name or "traffic_light" in name or "lane" in name:
            continue
        try:
            _actor_class(actor)
        except ValueError as error:
            raise ValueError(
                f"Unsupported MetaDrive live object in public registry: {type(actor).__name__!r}"
            ) from error
        actor_id = getattr(actor, "id", None)
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError("MetaDrive live actor has no stable runtime id")
        by_id[actor_id] = actor
    return tuple(by_id[key] for key in sorted(by_id))


def live_ego_snapshot(env: Any) -> ActorSnapshot:
    """Normalize the configured live ego vehicle from MetaDrive."""

    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    ego = getattr(traffic_manager, "ego_vehicle", None)
    if ego is None:
        raise ValueError("MetaDrive environment has no configured ego vehicle")
    return actor_snapshot_from_metadrive(env, ego)


def live_actor_snapshots(env: Any) -> tuple[ActorSnapshot, ...]:
    """Normalize live traffic actors, excluding the ego by runtime identity."""

    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    ego = getattr(traffic_manager, "ego_vehicle", None)
    if ego is None:
        raise ValueError("MetaDrive environment has no configured ego vehicle")
    ego_id = getattr(ego, "id", None)
    if not isinstance(ego_id, str) or not ego_id:
        raise ValueError("MetaDrive ego vehicle has no stable runtime id")
    return tuple(
        actor_snapshot_from_metadrive(env, actor)
        for actor in live_actor_objects(env)
        if getattr(actor, "id", None) != ego_id
    )


def live_signal_states_by_physical_id(env: Any) -> dict[str, str]:
    """Read current MetaDrive signal states without consulting future tracks.

    ScenarioNet's physical signal IDs are retained by MetaDrive's light manager
    in ``_scenario_id_to_obj_id``.  The returned values are the canonical
    Rulebook colors.  Unknown live states remain explicit ``UNKNOWN`` so the
    Rulebook evaluator can fail closed; no historical ``dynamic_map_states``
    sequence is used here.
    """

    manager = getattr(getattr(env, "engine", None), "light_manager", None)
    if manager is None:
        raise ValueError("MetaDrive environment has no light_manager")
    scenario_to_object = getattr(manager, "_scenario_id_to_obj_id", None)
    spawned = getattr(manager, "spawned_objects", None)
    if not isinstance(scenario_to_object, Mapping) or not isinstance(spawned, Mapping):
        raise ValueError("MetaDrive light_manager has no live scenario/object mapping")
    states: dict[str, str] = {}
    for physical_id, object_id in sorted(scenario_to_object.items(), key=lambda item: str(item[0])):
        light = spawned.get(object_id)
        if light is None:
            raise ValueError(f"Live traffic light object missing for physical id {physical_id!r}")
        get_state = getattr(light, "get_state", None)
        if not callable(get_state):
            raise ValueError(f"Live traffic light {physical_id!r} has no get_state()")
        payload = get_state()
        object_state = payload.get("object_state") if isinstance(payload, Mapping) else None
        if not isinstance(object_state, str):
            raise ValueError(f"Live traffic light {physical_id!r} returned no object_state")
        states[str(physical_id)] = _LIVE_SIGNAL_STATE_MAP.get(object_state, "UNKNOWN")
    return states


def _contact_method(contact: Any, *names: str) -> Any:
    for name in names:
        value = getattr(contact, name, None)
        if callable(value):
            return value()
        if value is not None:
            return value
    raise ValueError(f"MetaDrive contact has no supported accessor: {names!r}")


def _point3(value: object, *, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 3:
            raise ValueError(f"MetaDrive contact {field_name} must contain three values")
        point = (float(value[0]), float(value[1]), float(value[2]))
    else:
        try:
            point = (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"MetaDrive contact {field_name} must contain three values") from error
    if not all(isfinite(component) for component in point):
        raise ValueError(f"MetaDrive contact {field_name} must be finite")
    return point


def _node_object_id(obj: Any) -> str:
    value = getattr(obj, "id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("MetaDrive contact object must expose a non-empty id")
    return value


def contact_onset_from_metadrive_contact(
    contact: Any,
    env: Any,
    *,
    object_from_node: Any,
) -> ContactOnsetRecord:
    """Normalize one Bullet contact whose one body is the live ego vehicle.

    ``object_from_node`` is injected because MetaDrive's global object registry
    is an engine concern.  The helper never reads crash flags or ScenarioNet
    future tracks.  A contact without a resolvable ego/other object fails
    closed instead of manufacturing an actor identity.
    """

    if not callable(object_from_node):
        raise TypeError("object_from_node must be callable")
    node0 = _contact_method(contact, "getNode0", "get_node0", "node0")
    node1 = _contact_method(contact, "getNode1", "get_node1", "node1")
    object0 = object_from_node(node0)
    object1 = object_from_node(node1)
    if object0 is None or object1 is None:
        raise ValueError("MetaDrive contact objects could not be resolved")
    ego = getattr(getattr(env, "engine", None), "traffic_manager", None)
    ego = getattr(ego, "ego_vehicle", None)
    if ego is None:
        raise ValueError("MetaDrive contact environment has no ego vehicle")
    ego_id = _node_object_id(ego)
    id0 = _node_object_id(object0)
    id1 = _node_object_id(object1)
    if id0 == ego_id:
        ego_node, other_node, other = node0, node1, object1
        ego_is_node0 = True
    elif id1 == ego_id:
        ego_node, other_node, other = node1, node0, object0
        ego_is_node0 = False
    else:
        raise ValueError("MetaDrive contact does not involve the configured ego vehicle")
    del ego_node, other_node  # identity is established; geometry uses the manifold points.

    manifold = _contact_method(contact, "getManifoldPoint", "get_manifold_point", "manifold_point")
    point_a = _point3(
        _contact_method(
            manifold, "getPositionWorldOnA", "get_position_world_on_a", "position_world_on_a"
        ),
        field_name="position_world_on_a",
    )
    point_b = _point3(
        _contact_method(
            manifold, "getPositionWorldOnB", "get_position_world_on_b", "position_world_on_b"
        ),
        field_name="position_world_on_b",
    )
    ego_point, other_point = (point_a, point_b) if ego_is_node0 else (point_b, point_a)
    vector = (
        other_point[0] - ego_point[0],
        other_point[1] - ego_point[1],
    )
    norm = hypot(*vector)
    if norm <= 1.0e-12:
        normal = _point3(
            _contact_method(
                manifold, "getNormalWorldOnB", "get_normal_world_on_b", "normal_world_on_b"
            ),
            field_name="normal_world_on_b",
        )
        vector = (normal[0], normal[1])
        if not ego_is_node0:
            vector = (-vector[0], -vector[1])
        norm = hypot(*vector)
    if norm <= 1.0e-12 or not isfinite(norm):
        raise ValueError("MetaDrive contact normal cannot be normalized")
    return ContactOnsetRecord(
        actor_id=_stable_actor_id(env, other),
        actor_class=_actor_class(other),
        contact_point_xy=(
            (ego_point[0] + other_point[0]) / 2.0,
            (ego_point[1] + other_point[1]) / 2.0,
        ),
        normal_ego_to_other_xy=(vector[0] / norm, vector[1] / norm),
    )


class MetaDriveContactRecorder:
    """Control-step contact onset collector for the local Bullet hook."""

    def __init__(self, env: Any, *, object_from_node: Any) -> None:
        if not callable(object_from_node):
            raise TypeError("object_from_node must be callable")
        self.env = env
        self.object_from_node = object_from_node
        self.buffer = ContactOnsetBuffer()
        self._active_actor_ids: set[str] = set()
        self._last_snapshot_active_ids: frozenset[str] = frozenset()
        self.ignored_non_actor_contacts = 0
        self.deferred_manifold_contacts = 0

    def clear_control_step(self) -> None:
        self.buffer.clear_control_step()
        self._active_actor_ids.clear()

    def observe(self, contact: Any) -> None:
        try:
            record = contact_onset_from_metadrive_contact(
                contact,
                self.env,
                object_from_node=self.object_from_node,
            )
        except ValueError as error:
            # MetaDrive reports lane/boundary contacts through the same Bullet
            # callback, including contacts between two non-ego actors. Neither
            # is an ego contact onset; malformed resolved ego contacts still
            # propagate and fail closed.
            if str(error) not in {
                "MetaDrive contact objects could not be resolved",
                "MetaDrive contact does not involve the configured ego vehicle",
            }:
                if str(error) == (
                    "MetaDrive contact has no supported accessor: "
                    "('getManifoldPoint', 'get_manifold_point', 'manifold_point')"
                ):
                    # Bullet's ContactAdded callback carries node identities;
                    # manifold points are available from the world manifold
                    # query at the control-step snapshot boundary.
                    self.deferred_manifold_contacts += 1
                    return
                raise
            self.ignored_non_actor_contacts += 1
            return
        self.buffer.record_onset(record)
        self._active_actor_ids.add(record.actor_id)

    def _persistent_contact_records(self) -> tuple[ContactOnsetRecord, ...]:
        """Read current Bullet manifolds without changing simulator state."""

        world = getattr(getattr(self.env, "engine", None), "physics_world", None)
        world = getattr(world, "dynamic_world", world)
        if world is None:
            return ()
        getter = getattr(world, "get_manifolds", None) or getattr(world, "getManifolds", None)
        if not callable(getter):
            raise ValueError("MetaDrive Bullet world has no persistent-manifold accessor")
        records: list[ContactOnsetRecord] = []
        for manifold in getter():
            node0 = getattr(manifold, "getNode0", None) or getattr(manifold, "get_node0", None)
            node1 = getattr(manifold, "getNode1", None) or getattr(manifold, "get_node1", None)
            if not callable(node0) or not callable(node1):
                raise ValueError("MetaDrive persistent manifold has no node accessors")
            point_count = getattr(manifold, "getNumManifoldPoints", None) or getattr(
                manifold, "get_num_manifold_points", None
            )
            count = int(point_count()) if callable(point_count) else 1
            if count < 1:
                continue
            for index in range(count):
                point_getter = getattr(manifold, "getManifoldPoint", None) or getattr(
                    manifold, "get_manifold_point", None
                )
                if not callable(point_getter):
                    raise ValueError("MetaDrive persistent manifold has no point accessor")

                class _PersistentContact:
                    def getNode0(self):
                        return node0()

                    def getNode1(self):
                        return node1()

                    def getManifoldPoint(self):
                        try:
                            return point_getter(index)
                        except TypeError:
                            if count != 1:
                                raise
                            # Test doubles and older Bullet bindings expose a
                            # single-point accessor without an index.
                            return point_getter()

                try:
                    records.append(
                        contact_onset_from_metadrive_contact(
                            _PersistentContact(),
                            self.env,
                            object_from_node=self.object_from_node,
                        )
                    )
                except ValueError as error:
                    if str(error) not in {
                        "MetaDrive contact objects could not be resolved",
                        "MetaDrive contact does not involve the configured ego vehicle",
                    }:
                        raise
                    self.ignored_non_actor_contacts += 1
        return tuple(records)

    def snapshot_contact_state(self) -> tuple[tuple[ContactOnsetRecord, ...], frozenset[str]]:
        persistent = self._persistent_contact_records()
        persistent_ids = frozenset(record.actor_id for record in persistent)
        buffered = self.buffer.drain()
        buffered_ids = frozenset(record.actor_id for record in buffered)
        onset_records = list(buffered)
        seen_onsets = set(buffered_ids)
        for record in persistent:
            if (
                record.actor_id not in self._last_snapshot_active_ids
                and record.actor_id not in seen_onsets
            ):
                onset_records.append(record)
                seen_onsets.add(record.actor_id)
        current_ids = frozenset((*persistent_ids, *self._active_actor_ids))
        self._last_snapshot_active_ids = current_ids
        self._active_actor_ids.clear()
        return tuple(onset_records), current_ids


__all__ = [
    "MetaDriveContactRecorder",
    "actor_snapshot_from_metadrive",
    "contact_onset_from_metadrive_contact",
    "live_actor_objects",
    "live_actor_snapshots",
    "live_ego_snapshot",
    "live_signal_states_by_physical_id",
    "live_vehicle_objects",
]
