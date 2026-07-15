"""Pure conflict-zone occupancy lifecycle evaluator (DEC-002)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.types import MemoryDelta, RulebookMemory


@dataclass(frozen=True, slots=True)
class ZoneLifecycleView:
    pre_occupied_zone_ids: frozenset[str]
    post_occupied_zone_ids: frozenset[str]
    entered_zone_ids: frozenset[str]
    exited_zone_ids: frozenset[str]
    pending_preexisting_zone_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ZoneLifecycleEvaluation:
    view: ZoneLifecycleView
    memory_delta: MemoryDelta


class ZoneLifecycleEvaluator:
    """The only component allowed to write preexisting ego occupancy."""

    def evaluate(
        self,
        *,
        pre_ego_footprint: BaseGeometry,
        post_ego_footprint: BaseGeometry,
        zone_polygons: Mapping[str, BaseGeometry],
        pending_zone_ids: frozenset[str],
        memory: RulebookMemory,
    ) -> ZoneLifecycleEvaluation:
        if pre_ego_footprint.is_empty or post_ego_footprint.is_empty:
            raise ValueError("Zone lifecycle requires non-empty ego footprints")
        if not pre_ego_footprint.is_valid or not post_ego_footprint.is_valid:
            raise ValueError("Zone lifecycle requires valid ego footprints")
        pre_occupied = frozenset(
            zone_id
            for zone_id, polygon in zone_polygons.items()
            if polygon.is_valid and not polygon.is_empty and pre_ego_footprint.intersection(polygon).area > 0.0
        )
        post_occupied = frozenset(
            zone_id
            for zone_id, polygon in zone_polygons.items()
            if polygon.is_valid and not polygon.is_empty and post_ego_footprint.intersection(polygon).area > 0.0
        )
        pending_preexisting = frozenset(pending_zone_ids & pre_occupied)
        next_preexisting = frozenset(memory.preexisting_ego_occupancy_zone_ids | pending_preexisting)
        view = ZoneLifecycleView(
            pre_occupied_zone_ids=pre_occupied,
            post_occupied_zone_ids=post_occupied,
            entered_zone_ids=frozenset(post_occupied - pre_occupied),
            exited_zone_ids=frozenset(pre_occupied - post_occupied),
            pending_preexisting_zone_ids=pending_preexisting,
        )
        return ZoneLifecycleEvaluation(
            view=view,
            memory_delta=MemoryDelta(
                writer="zone_lifecycle",
                writes=(("preexisting_ego_occupancy_zone_ids", next_preexisting),),
            ),
        )
