from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


def _actor(
    actor_id: str, actor_class: ActorClass, cap: float | None = 20.0, *, x: float = 2.0, y: float = 0.0
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id, actor_class, (x, y), 0.0, 0.0, (0.0, 0.0),
        Polygon(((x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y - 0.5))),
        None,
        cap,
    )


def _onset(actor_id: str) -> ContactOnsetRecord:
    return ContactOnsetRecord(actor_id, ActorClass.VEHICLE)


def _ego(velocity_xy: tuple[float, float] = (0.0, 0.0)) -> ActorSnapshot:
    ego = _actor("ego", ActorClass.VEHICLE, 10.0, x=0.0)
    return ActorSnapshot(
        ego.actor_id,
        ego.actor_class,
        ego.position_xy,
        ego.position_z,
        ego.heading_rad,
        velocity_xy,
        ego.footprint,
        ego.live_lane_id,
        ego.configured_speed_cap_mps,
    )


def test_collision_cost_uses_pre_state_normal_speed_and_floor() -> None:
    result, memory_delta, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego((5.0, 0.0)),
        pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, 10.0)},
        onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    assert result.cost == pytest.approx(0.0625)
    assert result.raw["new_collision"] is True
    assert memory_delta.writer == "collision"


def test_persistent_contact_is_not_a_new_collision_and_static_has_zero_velocity() -> None:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego(),
        pre_actors_by_id={"wall": _actor("wall", ActorClass.STATIC_COLLIDABLE, None)},
        onset_records=(_onset("wall"),), previous_contact_ids=frozenset({"wall"}),
        post_active_contact_ids=frozenset({"wall"}),
    )
    assert result.cost == 0.0
    assert result.status.value == "not_applicable"


def test_missing_dynamic_pre_state_and_invalid_cap_fail_fast() -> None:
    with pytest.raises(RulebookEvaluationError, match="no pre-state"):
        evaluate_collision_impact(
            scenario_id="scenario", step_index=1,
            ego_configured_speed_cap_mps=10.0, pre_actors_by_id={},
            pre_ego=_ego((1.0, 0.0)),
            onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
            post_active_contact_ids=frozenset({"other"}),
        )
    with pytest.raises(RulebookEvaluationError, match="speed normalization cap"):
        evaluate_collision_impact(
            scenario_id="scenario", step_index=1,
            ego_configured_speed_cap_mps=None, pre_actors_by_id={}, onset_records=(),
            pre_ego=_ego((1.0, 0.0)),
            previous_contact_ids=frozenset(), post_active_contact_ids=frozenset(),
        )


@pytest.mark.parametrize(
    ("other_position", "ego_velocity", "other_velocity", "expected_normal"),
    [
        ((2.0, 0.0), (5.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        ((-2.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (-1.0, 0.0)),
        ((0.0, 2.0), (0.0, 5.0), (0.0, 0.0), (0.0, 1.0)),
    ],
)
def test_collision_uses_pre_state_centerline_for_front_rear_and_side_impacts(
    other_position: tuple[float, float],
    ego_velocity: tuple[float, float],
    other_velocity: tuple[float, float],
    expected_normal: tuple[float, float],
) -> None:
    other = _actor("other", ActorClass.VEHICLE, 10.0, x=other_position[0], y=other_position[1])
    other = ActorSnapshot(
        other.actor_id,
        other.actor_class,
        other.position_xy,
        other.position_z,
        other.heading_rad,
        other_velocity,
        other.footprint,
        other.live_lane_id,
        other.configured_speed_cap_mps,
    )
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego(ego_velocity),
        pre_actors_by_id={"other": other},
        onset_records=(_onset("other"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    assert result.cost == pytest.approx(0.0625)
    assert result.diagnostics["normal_source"] == "pre_state_canonical_footprint_centers"
    assert result.raw["actors"] == (
        {
            "actor_id": "other",
            "raw_speed_squared": 25.0,
            "cost": 0.0625,
            "normal_source": "pre_state_canonical_footprint_centers",
            "normal_ego_to_other_xy": expected_normal,
        },
    )


def test_collision_rejects_coincident_pre_state_centers() -> None:
    with pytest.raises(RulebookEvaluationError, match="footprint centers.*coincide"):
        evaluate_collision_impact(
            scenario_id="scenario",
            step_index=1,
            ego_configured_speed_cap_mps=10.0,
            pre_ego=_ego((5.0, 0.0)),
            pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, 10.0, x=0.0)},
            onset_records=(_onset("other"),),
            previous_contact_ids=frozenset(),
            post_active_contact_ids=frozenset({"other"}),
        )
