"""`TEST-RB5.1-14` part 1: at-fault classification of contact onsets (ADR-071).

The classification decides two things at once and they must not be separable:
what R1 charges, and whether the episode terminates or truncates. Adopting the
first alone would be worse than the status quo -- the agent would still be
punished through the zero bootstrap and would additionally have learned that
provoking a not-at-fault contact is free.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.components.collision_fault import (
    STOPPED_SPEED_THRESHOLD_MPS,
    CollisionFault,
    classify_contact,
    is_at_fault,
)
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


def _actor(
    actor_id: str,
    position: tuple[float, float],
    velocity: tuple[float, float],
    *,
    actor_class: ActorClass = ActorClass.VEHICLE,
) -> ActorSnapshot:
    x, y = position
    return ActorSnapshot(
        actor_id,
        actor_class,
        position,
        0.0,
        0.0,
        velocity,
        Polygon(((x - 2.0, y - 0.9), (x + 2.0, y - 0.9), (x + 2.0, y + 0.9), (x - 2.0, y + 0.9))),
        None,
        20.0,
    )


def _ego(velocity: tuple[float, float]) -> ActorSnapshot:
    return _actor("ego", (0.0, 0.0), velocity)


def _onset(actor_id: str) -> ContactOnsetRecord:
    return ContactOnsetRecord(actor_id, ActorClass.VEHICLE)


def test_stopped_ego_struck_from_behind_is_not_at_fault() -> None:
    """The canonical RSS not-to-blame case: waiting at a red and being hit.

    Prevention is impossible, so charging it teaches the agent to avoid
    legitimate stops -- a new degeneracy, not a fix for the old one.
    """

    fault = classify_contact(
        pre_ego=_ego((0.0, 0.0)),
        actor=_actor("follower", (-4.0, 0.0), (5.0, 0.0)),
    )
    assert fault is CollisionFault.STOPPED_EGO
    assert is_at_fault(fault, ego_within_single_lane=True) is False


def test_moving_ego_struck_from_behind_is_not_at_fault_either() -> None:
    fault = classify_contact(
        pre_ego=_ego((5.0, 0.0)),
        actor=_actor("follower", (-4.0, 0.0), (9.0, 0.0)),
    )
    assert fault is CollisionFault.ACTIVE_REAR
    assert is_at_fault(fault, ego_within_single_lane=True) is False


def test_driving_into_a_stopped_track_is_at_fault() -> None:
    fault = classify_contact(
        pre_ego=_ego((5.0, 0.0)),
        actor=_actor("parked", (4.0, 0.0), (0.0, 0.0)),
    )
    assert fault is CollisionFault.STOPPED_TRACK
    assert is_at_fault(fault, ego_within_single_lane=True) is True


def test_front_impact_on_a_moving_agent_is_at_fault() -> None:
    fault = classify_contact(
        pre_ego=_ego((10.0, 0.0)),
        actor=_actor("leader", (5.0, 0.0), (2.0, 0.0)),
    )
    assert fault is CollisionFault.ACTIVE_FRONT
    assert is_at_fault(fault, ego_within_single_lane=True) is True


def test_lateral_impact_is_at_fault_only_when_the_ego_is_not_in_one_lane() -> None:
    """nuPlan's conditional branch: a side impact is the ego's fault when the
    ego was the one changing lanes or straddling."""

    fault = classify_contact(
        pre_ego=_ego((5.0, 0.0)),
        actor=_actor("neighbour", (1.0, 3.0), (5.0, 0.0)),
    )
    assert fault is CollisionFault.ACTIVE_LATERAL
    assert is_at_fault(fault, ego_within_single_lane=True) is False
    assert is_at_fault(fault, ego_within_single_lane=False) is True


def test_undeterminable_lane_containment_resolves_to_at_fault() -> None:
    """An input that cannot be resolved must never buy an exculpation.

    The failure would be silent, and it would reward exactly the states where
    the geometry is hardest to resolve.
    """

    assert is_at_fault(CollisionFault.ACTIVE_LATERAL, ego_within_single_lane=None) is True


def test_the_stopped_threshold_is_nuplans_published_value() -> None:
    assert STOPPED_SPEED_THRESHOLD_MPS == 5e-02
    just_moving = _ego((STOPPED_SPEED_THRESHOLD_MPS * 1.001, 0.0))
    assert (
        classify_contact(pre_ego=just_moving, actor=_actor("a", (-4.0, 0.0), (5.0, 0.0)))
        is CollisionFault.ACTIVE_REAR
    )


def test_r1_charges_nothing_for_a_not_at_fault_contact_but_reports_it() -> None:
    """ "Not charged" must never become "not observed"."""

    result, memory_delta, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=20.0,
        pre_ego=_ego((0.0, 0.0)),
        pre_actors_by_id={"follower": _actor("follower", (-4.0, 0.0), (9.0, 0.0))},
        onset_records=(_onset("follower"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"follower"}),
        post_actor_ids=frozenset({"follower"}),
    )

    assert result.cost == 0.0
    assert result.applicable is False
    assert result.diagnostics["at_fault_collision"] is False
    assert result.diagnostics["not_at_fault_contacts"] == (
        {"actor_id": "follower", "fault": "stopped_ego_collision"},
    )
    # The contact still enters the memory, so it is not re-charged next step.
    assert memory_delta.writes[0] == ("previous_contact_ids", frozenset({"follower"}))


def test_r1_still_charges_an_at_fault_contact_in_full() -> None:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=20.0,
        pre_ego=_ego((10.0, 0.0)),
        pre_actors_by_id={"parked": _actor("parked", (5.0, 0.0), (0.0, 0.0))},
        onset_records=(_onset("parked"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"parked"}),
        post_actor_ids=frozenset({"parked"}),
    )

    assert result.cost > 0.0
    assert result.diagnostics["at_fault_collision"] is True
    assert result.raw["worst_fault"] == "stopped_track_collision"
    assert result.diagnostics["not_at_fault_contacts"] == ()


def test_a_mixed_step_charges_only_the_at_fault_contact() -> None:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=20.0,
        pre_ego=_ego((10.0, 0.0)),
        pre_actors_by_id={
            "parked": _actor("parked", (5.0, 0.0), (0.0, 0.0)),
            "follower": _actor("follower", (-6.0, 0.0), (14.0, 0.0)),
        },
        onset_records=(_onset("parked"), _onset("follower")),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"parked", "follower"}),
        post_actor_ids=frozenset({"parked", "follower"}),
    )

    assert result.raw["worst_actor_id"] == "parked"
    assert [entry["actor_id"] for entry in result.raw["actors"]] == ["parked"]
    assert result.diagnostics["not_at_fault_contacts"] == (
        {"actor_id": "follower", "fault": "active_rear_collision"},
    )
    assert result.diagnostics["at_fault_collision"] is True


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (5.0, 0.0)])
def test_classification_never_returns_none(velocity) -> None:
    fault = classify_contact(pre_ego=_ego(velocity), actor=_actor("a", (3.0, 1.0), (4.0, 1.0)))
    assert isinstance(fault, CollisionFault)
