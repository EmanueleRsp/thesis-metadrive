"""Transactional memory/cache merge primitives for Rulebook v2."""

from __future__ import annotations

from dataclasses import fields, replace

from thesis_rl.rulebook.v2.geometry.canonical import canonical_geometry_wkb
from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY
from thesis_rl.rulebook.v2.types import CacheDelta, EpisodeCache, MemoryDelta, RulebookMemory


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
