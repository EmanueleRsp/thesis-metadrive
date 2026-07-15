from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.memory import (
    EpisodeCacheOverlay,
    apply_cache_delta,
    initialize_rulebook_memory,
    merge_cache_deltas,
    merge_memory_deltas,
)
from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ConflictZoneRecord,
    EpisodeCache,
    MemoryDelta,
    MovementKey,
    RulebookMemory,
    TaskRouteRecord,
    ActorClass,
    ActorSnapshot,
    EnvSnapshot,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


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


def test_cache_overlay_exposes_pending_zones_without_committing_them() -> None:
    cache = EpisodeCache("scenario", TaskRouteRecord("scenario", ("lane",), "pg", "adapter", "hash"))
    overlay = EpisodeCacheOverlay(cache, CacheDelta((_zone("pending"),)))
    assert overlay.zone_ids == frozenset({"pending"})
    assert overlay.get_zone("pending") is not None
    assert not cache.conflict_zones


def test_memory_initialization_records_reset_contacts_route_and_occupancy() -> None:
    ego = ActorSnapshot(
        "ego", ActorClass.VEHICLE, (2.0, 0.0), 0.0, 0.0, (0.0, 0.0),
        Polygon(((1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5))), "lane", 20.0
    )
    snapshot = EnvSnapshot(
        "scenario", 0, 0.0, ego, (), (), frozenset({"other"}), {}
    )
    memory = initialize_rulebook_memory(
        reset_snapshot=snapshot,
        route=RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        zone_polygons={"zone": Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)))},
    )
    assert memory.previous_contact_ids == frozenset({"other"})
    assert memory.previous_route_s_m == 2.0
    assert memory.preexisting_ego_occupancy_zone_ids == frozenset({"zone"})
