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
        actors=(
            _actor("cyclist", ActorClass.CYCLIST, 1.5),
            _actor("ped", ActorClass.PEDESTRIAN, 5.0),
        ),
        vertically_compatible_actor_ids=frozenset({"cyclist", "ped"}),
    )
    assert result.raw["worst_actor_id"] == "cyclist"
    assert result.cost == pytest.approx(0.5)
    assert memory_delta.writes == ()
    assert cache_delta.new_conflict_zones == ()


def test_clearance_excludes_vehicle_class_per_rulebook_v4_8() -> None:
    """REQ-R2-01: vehicle clearance is replaced by the scoped lateral-RSS
    metric; a VEHICLE actor must never contribute to the clearance cost,
    even when very close to ego."""
    result, _, _ = evaluate_clearance(
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("vehicle", ActorClass.VEHICLE, 1.05),),
        vertically_compatible_actor_ids=frozenset({"vehicle"}),
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
    assert result.raw["actors"] == ()


def test_clearance_static_distance_is_diagnostic_only_per_rulebook_v4_8() -> None:
    """REQ-R2-02/DEC-R2-04: a close static obstacle is logged in
    ``diagnostics`` as ``static_polygon_distance_m`` but never contributes
    to the clearance cost or its ``raw`` candidate set."""
    result, _, _ = evaluate_clearance(
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("cone", ActorClass.STATIC_COLLIDABLE, 1.05),),
        vertically_compatible_actor_ids=frozenset({"cone"}),
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
    assert result.raw["actors"] == ()
    assert result.diagnostics["static_polygon_distance_m"] == pytest.approx(0.05)


def test_clearance_excludes_incompatible_and_non_collidable_actors() -> None:
    result, _, _ = evaluate_clearance(
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("vehicle", ActorClass.VEHICLE, 1.1), _actor("infra", ActorClass.INFRASTRUCTURE_NON_COLLIDABLE, 0.0)),
        vertically_compatible_actor_ids=frozenset({"infra"}),
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
