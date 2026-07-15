from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot


def _actor(actor_id: str, actor_class: ActorClass, x: float) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id, actor_class, (x, 0.0), 0.0, 0.0, (0.0, 0.0),
        Polygon(((x, -0.5), (x + 1.0, -0.5), (x + 1.0, 0.5), (x, 0.5))), None, 20.0
    )


def test_clearance_iterates_live_candidates_and_uses_class_thresholds() -> None:
    result, memory_delta, cache_delta = evaluate_clearance(
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("vehicle", ActorClass.VEHICLE, 1.8), _actor("ped", ActorClass.PEDESTRIAN, 5.0)),
        vertically_compatible_actor_ids=frozenset({"vehicle", "ped"}),
    )
    assert result.raw["worst_actor_id"] == "vehicle"
    assert result.cost == pytest.approx(0.0)
    assert memory_delta.writes == ()
    assert cache_delta.new_conflict_zones == ()


def test_clearance_excludes_incompatible_and_non_collidable_actors() -> None:
    result, _, _ = evaluate_clearance(
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("vehicle", ActorClass.VEHICLE, 1.1), _actor("infra", ActorClass.INFRASTRUCTURE_NON_COLLIDABLE, 0.0)),
        vertically_compatible_actor_ids=frozenset({"infra"}),
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
