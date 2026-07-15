from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.rulebook.v2 import ZoneLifecycleEvaluator
from thesis_rl.rulebook.v2.memory import merge_memory_deltas
from thesis_rl.rulebook.v2.types import RulebookMemory


def test_lazy_zone_occupied_in_pre_state_is_marked_preexisting_before_judgment() -> None:
    zone = Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)))
    evaluation = ZoneLifecycleEvaluator().evaluate(
        pre_ego_footprint=Polygon(((1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5))),
        post_ego_footprint=Polygon(((2.0, -0.5), (3.0, -0.5), (3.0, 0.5), (2.0, 0.5))),
        zone_polygons={"zone": zone},
        pending_zone_ids=frozenset({"zone"}),
        memory=RulebookMemory(),
    )
    assert evaluation.view.pending_preexisting_zone_ids == frozenset({"zone"})
    assert evaluation.view.entered_zone_ids == frozenset()
    committed = merge_memory_deltas(RulebookMemory(), (evaluation.memory_delta,))
    assert committed.preexisting_ego_occupancy_zone_ids == frozenset({"zone"})


def test_zone_lifecycle_reports_entry_and_exit_without_side_effects() -> None:
    zone = Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)))
    evaluation = ZoneLifecycleEvaluator().evaluate(
        pre_ego_footprint=Polygon(((-2.0, -0.5), (-1.0, -0.5), (-1.0, 0.5), (-2.0, 0.5))),
        post_ego_footprint=Polygon(((1.5, -0.5), (2.5, -0.5), (2.5, 0.5), (1.5, 0.5))),
        zone_polygons={"zone": zone},
        pending_zone_ids=frozenset(),
        memory=RulebookMemory(),
    )
    assert evaluation.view.entered_zone_ids == frozenset({"zone"})
    assert evaluation.view.exited_zone_ids == frozenset()
