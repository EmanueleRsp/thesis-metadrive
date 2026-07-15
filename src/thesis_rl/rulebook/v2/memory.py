"""Transactional memory/cache merge primitives for Rulebook v2."""

from __future__ import annotations

from dataclasses import fields, replace

from thesis_rl.rulebook.v2.geometry.canonical import canonical_geometry_wkb
from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY
from thesis_rl.rulebook.v2.types import CacheDelta, EpisodeCache, MemoryDelta, RulebookMemory
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
    )


def _memory_owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    for component in DEFAULT_RULEBOOK_V2_REGISTRY.components:
        for field_name in component.owned_memory_fields:
            owners[field_name] = component.name
    return owners


def merge_memory_deltas(
    memory: RulebookMemory, deltas: tuple[MemoryDelta, ...]
) -> RulebookMemory:
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
    return replace(memory, **updates)


def merge_cache_deltas(deltas: tuple[CacheDelta, ...]) -> CacheDelta:
    """Merge pending zones after canonical byte-equivalence validation."""

    merged: dict[str, object] = {}
    for delta in deltas:
        for zone in delta.new_conflict_zones:
            existing = merged.get(zone.zone_id)
            if existing is None:
                merged[zone.zone_id] = zone
                continue
            if (
                canonical_geometry_wkb(existing.polygon)
                != canonical_geometry_wkb(zone.polygon)
                or existing != zone
            ):
                raise ValueError(f"Conflicting geometries for conflict zone ID: {zone.zone_id}")
    return CacheDelta(new_conflict_zones=tuple(merged.values()))


def apply_cache_delta(cache: EpisodeCache, delta: CacheDelta) -> EpisodeCache:
    """Return a new cache, refusing to modify any committed zone geometry."""

    zones = dict(cache.conflict_zones)
    for zone in delta.new_conflict_zones:
        existing = zones.get(zone.zone_id)
        if existing is not None and existing != zone:
            raise ValueError(f"Committed conflict zone cannot be modified: {zone.zone_id}")
        zones[zone.zone_id] = zone
    return replace(cache, conflict_zones=zones)
