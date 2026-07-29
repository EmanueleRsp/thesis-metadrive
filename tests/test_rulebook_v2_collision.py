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


def test_collision_cost_uses_pre_state_normal_speed() -> None:
    """Rulebook v4.9: cost is the MAIS3+F risk at the pre-state normal speed.

    The v4.7 expectation here was ``(5/10)^2 = 0.0625``, i.e. the closing speed
    normalized by the configured speed cap.  ADR-027 replaced that mapping; the
    pre-state normal closing speed itself (5 m/s) is unchanged and is still
    what the cost is a function of.
    """

    result, memory_delta, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego((5.0, 0.0)),
        pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, 10.0)},
        onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    assert result.raw["worst_closing_speed_mps"] == pytest.approx(5.0)
    assert result.cost == pytest.approx(0.0038685, abs=1e-6)
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


def test_missing_dynamic_pre_state_is_ignored_for_first_frame() -> None:
    result, memory_delta, _ = evaluate_collision_impact(
        scenario_id="scenario", step_index=1,
        ego_configured_speed_cap_mps=10.0, pre_actors_by_id={},
        pre_ego=_ego((1.0, 0.0)),
        onset_records=(_onset("other"),), previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    assert result.status.value == "not_applicable"
    # REQ-EF-13: with no ``post_actor_ids`` supplied the actor was in neither
    # snapshot, which is the instrumentation-gap branch, not the "appeared this
    # step" one.  Key renamed from ``ignored_missing_pre_state_actor_ids``.
    assert result.diagnostics["unobserved_onset_actor_ids"] == ("other",)
    assert result.diagnostics["appeared_onset_actor_ids"] == ()
    assert memory_delta.writes[0] == ("previous_contact_ids", frozenset({"other"}))
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
    assert result.cost == pytest.approx(0.0038685, abs=1e-6)
    assert result.diagnostics["normal_source"] == "pre_state_canonical_footprint_centers"
    assert result.raw["actors"] == (
        {
            "actor_id": "other",
            "actor_class": "vehicle",
            "closing_speed_mps": pytest.approx(5.0),
            "injury_risk_curve": "car_driver_mais3f",
            "cost": pytest.approx(0.0038685, abs=1e-6),
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


def test_actor_that_appeared_this_step_is_separated_from_one_never_observed() -> None:
    """TEST-EF-20 / REQ-EF-13.

    A contact onset without a pre-state record has no computable closing speed
    (v4.9 §4.1 needs the pre-state normal). Two situations produce it and the
    post-state distinguishes them:

    * present in the post-state -> the actor genuinely appeared during this
      control step. At decision time nothing was there, so the ego had no
      alternative action and R1 = 0 is the correct causal attribution.
    * in neither snapshot -> the snapshot pipeline never observed an object the
      physics engine did. That is an instrumentation gap, and it must not read
      as a clean R1 = 0.
    """

    appeared, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego((1.0, 0.0)),
        pre_actors_by_id={},
        onset_records=(_onset("spawned"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"spawned"}),
        post_actor_ids=frozenset({"spawned"}),
    )
    assert appeared.cost == 0.0
    assert appeared.raw["appeared_onset_actor_ids"] == ("spawned",)
    assert appeared.raw["unobserved_onset_actor_ids"] == ()

    unobserved, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego((1.0, 0.0)),
        pre_actors_by_id={},
        onset_records=(_onset("ghost"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"ghost"}),
        post_actor_ids=frozenset({"someone-else"}),
    )
    assert unobserved.raw["appeared_onset_actor_ids"] == ()
    assert unobserved.raw["unobserved_onset_actor_ids"] == ("ghost",)
