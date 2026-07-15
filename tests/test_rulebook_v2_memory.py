from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.memory import apply_cache_delta, merge_cache_deltas, merge_memory_deltas
from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ConflictZoneRecord,
    EpisodeCache,
    MemoryDelta,
    MovementKey,
    RulebookMemory,
    TaskRouteRecord,
)


def _zone(zone_id: str, x: float = 0.0) -> ConflictZoneRecord:
    return ConflictZoneRecord(
        zone_id=zone_id,
        polygon=Polygon(((x, 0.0), (x + 1.0, 0.0), (x + 1.0, 1.0), (x, 1.0))),
        ego_movement_key=MovementKey("a", "n", "e"),
        other_movement_key=MovementKey("b", "n", "f"),
        route_entry_s_m=x,
        route_exit_s_m=x + 1.0,
        elevation_m=0.0,
    )


def test_memory_merge_enforces_ownership_and_duplicate_writers() -> None:
    memory = RulebookMemory()
    updated = merge_memory_deltas(
        memory,
        (MemoryDelta("progress", (("previous_route_s_m", 3.0),)),),
    )
    assert updated.previous_route_s_m == 3.0
    assert memory.previous_route_s_m == 0.0
    with pytest.raises(ValueError, match="does not own"):
        merge_memory_deltas(
            memory, (MemoryDelta("progress", (("previous_contact_ids", frozenset()),)),)
        )
    with pytest.raises(ValueError, match="Duplicate"):
        merge_memory_deltas(
            memory,
            (
                MemoryDelta("progress", (("previous_route_s_m", 1.0),)),
                MemoryDelta("progress", (("previous_route_s_m", 2.0),)),
            ),
        )


def test_cache_merge_and_apply_reject_conflicting_geometry_without_mutation() -> None:
    route = TaskRouteRecord("scenario", ("lane",), "pg", "adapter", "hash")
    cache = EpisodeCache("scenario", route)
    delta = CacheDelta((_zone("zone-a"),))
    merged = merge_cache_deltas((delta, delta))
    applied = apply_cache_delta(cache, merged)
    assert set(applied.conflict_zones) == {"zone-a"}
    assert not cache.conflict_zones
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cache_deltas((delta, CacheDelta((_zone("zone-a", 2.0),))))
