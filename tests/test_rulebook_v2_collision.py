from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


def _actor(actor_id: str, actor_class: ActorClass, cap: float | None = 20.0) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id, actor_class, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0),
        Polygon(((0, 0), (1, 0), (1, 1), (0, 0))), None, cap
    )


def _onset(actor_id: str) -> ContactOnsetRecord:
    return ContactOnsetRecord(actor_id, ActorClass.VEHICLE, (0.0, 0.0), (1.0, 0.0))


def test_collision_cost_uses_pre_state_normal_speed_and_floor() -> None:
    result, memory_delta, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1, ego_velocity_xy=(5.0, 0.0),
        ego_configured_speed_cap_mps=10.0,
        pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, 10.0)},
        onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    assert result.cost == pytest.approx(0.0625)
    assert result.raw["new_collision"] is True
    assert memory_delta.writer == "collision"


def test_persistent_contact_is_not_a_new_collision_and_static_has_zero_velocity() -> None:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1, ego_velocity_xy=(0.0, 0.0),
        ego_configured_speed_cap_mps=10.0,
        pre_actors_by_id={"wall": _actor("wall", ActorClass.STATIC_COLLIDABLE, None)},
        onset_records=(_onset("wall"),), previous_contact_ids=frozenset({"wall"}),
        post_active_contact_ids=frozenset({"wall"}),
    )
    assert result.cost == 0.0
    assert result.status.value == "not_applicable"


def test_missing_dynamic_pre_state_and_invalid_cap_fail_fast() -> None:
    with pytest.raises(RulebookEvaluationError, match="no pre-state"):
        evaluate_collision_impact(
            scenario_id="scenario", step_index=1, ego_velocity_xy=(1.0, 0.0),
            ego_configured_speed_cap_mps=10.0, pre_actors_by_id={},
            onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
            post_active_contact_ids=frozenset({"other"}),
        )
    with pytest.raises(RulebookEvaluationError, match="speed normalization cap"):
        evaluate_collision_impact(
            scenario_id="scenario", step_index=1, ego_velocity_xy=(1.0, 0.0),
            ego_configured_speed_cap_mps=None, pre_actors_by_id={}, onset_records=(),
            previous_contact_ids=frozenset(), post_active_contact_ids=frozenset(),
        )
