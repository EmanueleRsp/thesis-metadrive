"""Strict live-environment adapter hooks for Rulebook v2 snapshots.

Source-specific MetaDrive/ScenarioNet code supplies the callables in
``LiveSnapshotSources``.  The v2 core never guesses attribute names and never
falls back to observations or future trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, cast

from thesis_rl.rulebook.v2.context.snapshotter import capture_env_snapshot
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord, EnvSnapshot
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box


def _finite_pair(payload: object, *, field_name: str) -> tuple[float, float]:
    """Normalize one adapter-owned 2D value without source-specific fallbacks."""

    try:
        values = tuple(float(value) for value in cast(Any, payload))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Live actor {field_name} must be a finite 2D sequence") from error
    if len(values) != 2 or not all(isfinite(value) for value in values):
        raise ValueError(f"Live actor {field_name} must be a finite 2D sequence")
    return values


@dataclass(frozen=True, slots=True)
class LiveSnapshotSources:
    """Environment-owned providers for one canonical control-step snapshot."""

    scenario_id: Callable[[Any], str]
    step_index: Callable[[Any], int]
    sim_time_s: Callable[[Any], float]
    ego: Callable[[Any], ActorSnapshot]
    actors: Callable[[Any], tuple[ActorSnapshot, ...]]
    contact_onset_records: Callable[[Any], tuple[ContactOnsetRecord, ...]]
    active_contact_ids: Callable[[Any], frozenset[str]]
    signal_states_by_physical_id: Callable[[Any], Mapping[str, str]]

    def __post_init__(self) -> None:
        providers = (
            self.scenario_id, self.step_index, self.sim_time_s, self.ego,
            self.actors, self.contact_onset_records, self.active_contact_ids,
            self.signal_states_by_physical_id,
        )
        if any(not callable(provider) for provider in providers):
            raise TypeError("Every LiveSnapshotSources field must be callable")

    @classmethod
    def from_mapping(cls, providers: Mapping[str, Callable[[Any], object]]) -> "LiveSnapshotSources":
        """Construct sources from an adapter mapping, rejecting missing keys."""
        required = tuple(cls.__dataclass_fields__)
        missing = tuple(name for name in required if name not in providers)
        unknown = tuple(sorted(set(providers).difference(required)))
        if missing:
            raise ValueError(f"Missing live snapshot providers: {missing}")
        if unknown:
            raise ValueError(f"Unknown live snapshot providers: {unknown}")
        return cls(**cast(Any, {name: providers[name] for name in required}))


def actor_snapshot_from_payload(payload: Mapping[str, object]) -> ActorSnapshot:
    """Build one canonical actor from an adapter-owned live payload."""
    required = {"actor_id", "actor_class", "position_xy", "position_z", "heading_rad", "velocity_xy", "length_m", "width_m"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Live actor payload missing fields: {missing}")
    actor_class = payload["actor_class"]
    try:
        canonical_class = actor_class if isinstance(actor_class, ActorClass) else ActorClass(str(actor_class))
    except ValueError as error:
        raise ValueError(f"Unknown live actor class: {actor_class!r}") from error
    actor_id = payload["actor_id"]
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError("Live actor actor_id must be non-empty")
    position_xy = _finite_pair(payload["position_xy"], field_name="position_xy")
    velocity_xy = _finite_pair(payload["velocity_xy"], field_name="velocity_xy")
    try:
        position_z = float(cast(Any, payload["position_z"]))
        heading = float(cast(Any, payload["heading_rad"]))
        length = float(cast(Any, payload["length_m"]))
        width = float(cast(Any, payload["width_m"]))
    except (TypeError, ValueError) as error:
        raise ValueError("Live actor dimensions and pose must be numeric") from error
    if not all(isfinite(value) for value in (position_z, heading, length, width)):
        raise ValueError("Live actor pose/velocity must be finite 2D values")
    footprint = oriented_bounding_box(
        center_xy=position_xy,
        heading_rad=heading,
        length_m=length,
        width_m=width,
    )
    cap_value = payload.get("configured_speed_cap_mps")
    cap = None if cap_value is None else float(cast(Any, cap_value))
    live_lane_id = payload.get("live_lane_id")
    if live_lane_id is not None and (not isinstance(live_lane_id, str) or not live_lane_id):
        raise ValueError("Live actor live_lane_id must be a non-empty string when supplied")
    return ActorSnapshot(actor_id, canonical_class, position_xy, position_z, heading, velocity_xy, footprint, live_lane_id, cap)


def contact_onset_from_payload(payload: Mapping[str, object]) -> ContactOnsetRecord:
    """Normalize a contact-manifold record whose normal is ego-to-other."""

    required = {"actor_id", "actor_class", "contact_point_xy"}
    has_canonical_normal = "normal_ego_to_other_xy" in payload
    has_manifold_normal = "normal_xy" in payload and "normal_orientation" in payload
    if not has_canonical_normal and not has_manifold_normal:
        required.add("normal_ego_to_other_xy")
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Contact payload missing fields: {missing}")
    actor_id = payload["actor_id"]
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError("Contact actor_id must be non-empty")
    actor_class = payload["actor_class"]
    try:
        canonical_class = actor_class if isinstance(actor_class, ActorClass) else ActorClass(str(actor_class))
    except ValueError as error:
        raise ValueError(f"Unknown contact actor class: {actor_class!r}") from error
    point = _finite_pair(payload["contact_point_xy"], field_name="contact_point_xy")
    if has_canonical_normal:
        normal = _finite_pair(payload["normal_ego_to_other_xy"], field_name="normal_ego_to_other_xy")
    else:
        normal = _finite_pair(payload["normal_xy"], field_name="normal_xy")
        orientation = payload["normal_orientation"]
        if orientation == "other_to_ego":
            normal = (-normal[0], -normal[1])
        elif orientation != "ego_to_other":
            raise ValueError("Contact normal_orientation must be ego_to_other or other_to_ego")
    norm = (normal[0] ** 2 + normal[1] ** 2) ** 0.5
    if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-6:
        raise ValueError("Contact normal must be a non-zero unit vector oriented ego-to-other")
    return ContactOnsetRecord(actor_id, canonical_class, point, normal)


def wrap_collision_callback(
    original_callback: Callable[..., object],
    observer: Callable[..., object],
) -> Callable[..., object]:
    """Invoke the vendor callback unchanged, then publish the local hook event.

    The original callback owns simulator flags and its return value.  The
    observer is deliberately called only after it succeeds, so the adapter does
    not replace or reorder the vendored collision semantics.
    """

    if not callable(original_callback) or not callable(observer):
        raise TypeError("Collision callback and observer must be callable")

    def wrapped(*args: object, **kwargs: object) -> object:
        result = original_callback(*args, **kwargs)
        observer(*args, **kwargs)
        return result

    return wrapped


def install_collision_callback_hook(
    dynamic_world: object,
    *,
    original_callback: Callable[..., object],
    observer: Callable[..., object],
    callback_object_factory: Callable[[Callable[..., object]], object] | None = None,
) -> None:
    """Install the local hook through MetaDrive's existing callback boundary.

    ``callback_object_factory`` is injectable for tests; production adapters use
    Panda3D's ``PythonCallbackObject`` without importing it in the normative
    package at module import time.
    """

    setter = getattr(dynamic_world, "setContactAddedCallback", None)
    if not callable(setter):
        raise TypeError("dynamic_world must expose setContactAddedCallback(callable)")
    factory = callback_object_factory
    if factory is None:
        try:
            from panda3d.core import PythonCallbackObject
        except ImportError as error:
            raise RuntimeError("Panda3D is required to install the live collision hook") from error
        factory = PythonCallbackObject
    setter(factory(wrap_collision_callback(original_callback, observer)))


class LiveSnapshotAdapter:
    """Build immutable snapshots using only explicitly supplied live hooks."""

    def __init__(self, sources: LiveSnapshotSources) -> None:
        self.sources = sources

    def capture(self, env: Any) -> EnvSnapshot:
        sources = self.sources
        return capture_env_snapshot(
            scenario_id=sources.scenario_id(env),
            step_index=sources.step_index(env),
            sim_time_s=sources.sim_time_s(env),
            ego=sources.ego(env),
            actors=sources.actors(env),
            contact_onset_records=sources.contact_onset_records(env),
            active_contact_ids=sources.active_contact_ids(env),
            signal_states_by_physical_id=sources.signal_states_by_physical_id(env),
        )
