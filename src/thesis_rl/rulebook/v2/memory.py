"""Transactional memory/cache merge primitives for Rulebook v2."""

from __future__ import annotations

from dataclasses import fields, replace
from math import isfinite
from typing import Any, cast

from thesis_rl.rulebook.v2.geometry.canonical import canonical_geometry_wkb
from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorMotionHistory,
    ActorMotionSample,
    CacheDelta,
    EpisodeCache,
    MemoryDelta,
    RulebookMemory,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import EnvSnapshot


class EpisodeCacheOverlay:
    """Read-only view of committed cache plus same-step pending zones."""

    def __init__(self, cache: EpisodeCache, pending: CacheDelta) -> None:
        self._cache = cache
        self._pending = merge_cache_deltas((pending,))

    @property
    def pending_zone_ids(self) -> frozenset[str]:
        return frozenset(zone.zone_id for zone in self._pending.new_conflict_zones)

    @property
    def zone_ids(self) -> frozenset[str]:
        return frozenset(self._cache.conflict_zones) | self.pending_zone_ids

    def get_zone(self, zone_id: str):
        for zone in self._pending.new_conflict_zones:
            if zone.zone_id == zone_id:
                return zone
        return self._cache.conflict_zones.get(zone_id)


def initialize_rulebook_memory(
    *,
    reset_snapshot: EnvSnapshot,
    route: RoutePolyline,
    zone_polygons: dict[str, object],
) -> RulebookMemory:
    """Initialize reset-owned causal state without attributing reset events to policy."""

    projection = route.project(
        reset_snapshot.ego.position_xy,
        position_z=reset_snapshot.ego.position_z,
    )
    preexisting = frozenset(
        zone_id
        for zone_id, polygon in zone_polygons.items()
        if getattr(polygon, "is_valid", False)
        and not getattr(polygon, "is_empty", True)
        and reset_snapshot.ego.footprint.intersection(polygon).area > 0.0
    )
    return RulebookMemory(
        previous_contact_ids=reset_snapshot.active_contact_ids,
        preexisting_ego_occupancy_zone_ids=preexisting,
        previous_route_s_m=projection.s_m,
        actor_motion_histories=tuple(
            ActorMotionHistory(
                actor.actor_id,
                (
                    ActorMotionSample(
                        reset_snapshot.sim_time_s,
                        actor.position_xy,
                        actor.heading_rad,
                        actor.velocity_xy,
                    ),
                ),
            )
            for actor in (reset_snapshot.ego, *reset_snapshot.actors)
            if actor.actor_class is ActorClass.VEHICLE
        ),
        previous_sim_time_s=reset_snapshot.sim_time_s,
    )


def build_motion_history_preview(
    *,
    memory: RulebookMemory,
    post_state: EnvSnapshot,
    history_window_s: float,
) -> tuple[tuple[ActorMotionHistory, ...], MemoryDelta]:
    """Build the immutable causal preview and its sole transactional writer."""

    if not isfinite(history_window_s) or history_window_s <= 0.0:
        raise ValueError("history_window_s must be finite and positive")
    if not isfinite(post_state.sim_time_s):
        raise ValueError("post_state.sim_time_s must be finite")
    if (
        memory.previous_sim_time_s is not None
        and post_state.sim_time_s <= memory.previous_sim_time_s
    ):
        raise ValueError("sim_time_s must be strictly increasing")
    current_actors = (post_state.ego, *post_state.actors)
    live_vehicles = {
        actor.actor_id: actor for actor in current_actors if actor.actor_class is ActorClass.VEHICLE
    }
    if len(live_vehicles) != sum(
        actor.actor_class is ActorClass.VEHICLE for actor in current_actors
    ):
        raise ValueError("Conflicting live vehicle records for one actor ID")
    previous = {history.actor_id: history for history in memory.actor_motion_histories}
    histories: list[ActorMotionHistory] = []
    for actor_id in sorted(live_vehicles):
        actor = live_vehicles[actor_id]
        bounds = actor.footprint.bounds
        if (
            actor.footprint.is_empty
            or not actor.footprint.is_valid
            or not all(isfinite(value) for value in bounds)
        ):
            raise ValueError(f"Invalid finite footprint for actor {actor_id!r}")
        prior = previous.get(actor_id)
        if (
            prior is not None
            and prior.samples
            and prior.samples[-1].timestamp_s >= post_state.sim_time_s
        ):
            raise ValueError(f"Non-increasing timestamp for actor history {actor_id!r}")
        samples = () if prior is None else prior.samples
        samples = tuple(
            sample
            for sample in samples
            if sample.timestamp_s >= post_state.sim_time_s - history_window_s
        ) + (
            ActorMotionSample(
                post_state.sim_time_s,
                actor.position_xy,
                actor.heading_rad,
                actor.velocity_xy,
            ),
        )
        histories.append(ActorMotionHistory(actor_id, samples))
    result = tuple(histories)
    return result, MemoryDelta(
        writer="motion_history",
        writes=(
            ("actor_motion_histories", result),
            ("previous_sim_time_s", post_state.sim_time_s),
        ),
    )


def _memory_owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    for component in DEFAULT_RULEBOOK_V2_REGISTRY.components:
        for field_name in component.owned_memory_fields:
            owners[field_name] = component.name
    return owners


def merge_memory_deltas(memory: RulebookMemory, deltas: tuple[MemoryDelta, ...]) -> RulebookMemory:
    """Apply at most one complete field write per transition, without mutation."""

    known_fields = {field.name for field in fields(RulebookMemory)}
    owners = _memory_owners()
    updates: dict[str, object] = {}
    for delta in deltas:
        if not delta.writer:
            raise ValueError("MemoryDelta requires its normative writer")
        for field_name, value in delta.writes:
            if field_name not in known_fields:
                raise ValueError(f"Unknown RulebookMemory field: {field_name}")
            if owners.get(field_name) != delta.writer:
                raise ValueError(
                    f"Writer {delta.writer!r} does not own RulebookMemory field {field_name!r}"
                )
            if field_name in updates:
                raise ValueError(f"Duplicate MemoryDelta writer for field: {field_name}")
            updates[field_name] = value
    return replace(memory, **cast(Any, updates))


def merge_cache_deltas(deltas: tuple[CacheDelta, ...]) -> CacheDelta:
    """Merge pending zones after canonical byte-equivalence validation."""

    merged: dict[str, Any] = {}
    for delta in deltas:
        for zone in delta.new_conflict_zones:
            existing = merged.get(zone.zone_id)
            if existing is None:
                merged[zone.zone_id] = zone
                continue
            if not _conflict_zone_equivalent(existing, zone):
                raise ValueError(f"Conflicting geometries for conflict zone ID: {zone.zone_id}")
    return CacheDelta(new_conflict_zones=tuple(cast(Any, merged.values())))


def apply_cache_delta(cache: EpisodeCache, delta: CacheDelta) -> EpisodeCache:
    """Return a new cache, refusing to modify any committed zone geometry."""

    zones = dict(cache.conflict_zones)
    for zone in delta.new_conflict_zones:
        existing = zones.get(zone.zone_id)
        if existing is not None:
            if not _conflict_zone_equivalent(existing, zone):
                raise ValueError(f"Committed conflict zone cannot be modified: {zone.zone_id}")
            continue
        zones[zone.zone_id] = zone
    return replace(cache, conflict_zones=zones)


def _conflict_zone_equivalent(left: Any, right: Any) -> bool:
    """Compare conflict-zone records using the canonical geometry contract."""

    return canonical_geometry_wkb(left.polygon) == canonical_geometry_wkb(right.polygon)
