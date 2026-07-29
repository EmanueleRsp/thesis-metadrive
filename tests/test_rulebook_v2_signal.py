from shapely.geometry import LineString
import pytest
from thesis_rl.rulebook.v2.components.controls import evaluate_signal_transition
from thesis_rl.rulebook.v2.errors import (
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)
from thesis_rl.rulebook.v2.types import ApproachControl, MovementKey, TrafficControlRecord


def _signal():
    return TrafficControlRecord(
        "sig",
        ApproachControl.SIGNAL,
        ("lane",),
        MovementKey("a", "n", "e"),
        LineString(((10, -1), (10, 1))),
        10,
        0.0,
        ("p",),
    )


def test_signal_crossing_uses_pre_action_state():
    result, delta, _ = evaluate_signal_transition(
        control=_signal(),
        pre_state="GREEN",
        post_state="RED",
        pre_delta_m=0.1,
        post_delta_m=-0.1,
        speed_mps=5,
        route_tangent_xy=(1, 0),
        delta_t_s=0.1,
        previous_yellow_must_stop=False,
        previous_signal_delta_m=0.1,
        previous_group_id="sig",
        resolved_group_ids=frozenset(),
        crossing=True,
        ego_brake_mps2=4.0,
    )
    assert result.cost == 0.0
    assert "sig" in dict(delta.writes)["resolved_signal_group_ids"]


def test_signal_red_approach_is_continuous():
    result, _, _ = evaluate_signal_transition(
        control=_signal(),
        pre_state="RED",
        post_state="RED",
        pre_delta_m=3.0,
        post_delta_m=1.0,
        speed_mps=4,
        route_tangent_xy=(1, 0),
        delta_t_s=0.1,
        previous_yellow_must_stop=False,
        previous_signal_delta_m=3.0,
        previous_group_id="sig",
        resolved_group_ids=frozenset(),
        crossing=False,
        ego_brake_mps2=4.0,
    )
    assert 0.0 < result.cost < 1.0


def test_signal_unknown_is_typed_runtime_scenario_data_abort():
    with pytest.raises(RuntimeScenarioNotEvaluableError) as raised:
        evaluate_signal_transition(
            control=_signal(),
            pre_state="UNKNOWN",
            post_state="RED",
            pre_delta_m=1,
            post_delta_m=1,
            speed_mps=0,
            route_tangent_xy=(1, 0),
            delta_t_s=0.1,
            previous_yellow_must_stop=False,
            previous_signal_delta_m=1,
            previous_group_id="sig",
            resolved_group_ids=frozenset(),
            crossing=False,
            ego_brake_mps2=4.0,
        )
    assert raised.value.reason is RuntimeScenarioNotEvaluableReason.INVALID_SIGNAL_TRANSITION
    assert isinstance(raised.value.__cause__, ValueError)
    assert raised.value.diagnostics["active_signal_group_id"] == "sig"


def _signal_group(group_id: str, route_s: float = 10.0):
    return TrafficControlRecord(
        group_id,
        ApproachControl.SIGNAL,
        ("lane",),
        MovementKey("a", "n", "e"),
        LineString(((route_s, -1), (route_s, 1))),
        route_s,
        0.0,
        ("p",),
    )


def test_yellow_commitment_is_not_inherited_from_a_different_signal_group() -> None:
    """TEST-EF-17 / REQ-EF-10.

    Regression: ``previous_group_id`` was accepted and never used, so a newly
    encountered yellow inherited the previous intersection's frozen
    ``yellow_must_stop`` instead of deciding its own (v4.7 §7.6.3 freezes the
    value "per la stessa fase").
    """

    result, _, _ = evaluate_signal_transition(
        control=_signal_group("group-b"),
        pre_state="GREEN",
        post_state="YELLOW",
        pre_delta_m=30.0,
        post_delta_m=2.0,
        speed_mps=10.0,
        route_tangent_xy=(1.0, 0.0),
        delta_t_s=0.1,
        previous_yellow_must_stop=True,
        previous_signal_delta_m=40.0,
        previous_group_id="group-a",
        resolved_group_ids=frozenset(),
        crossing=False,
        ego_brake_mps2=4.0,
    )
    # d_req = 10*0.1 + 100/8 = 13.5 m; the ego is 2 m from the new line, so it
    # cannot stop and must not inherit group-a's commitment.
    assert result.raw["yellow_must_stop"] is False


def test_yellow_commitment_is_kept_within_the_same_signal_group() -> None:
    """TEST-EF-17 / REQ-EF-10: the reset is scoped to a group change only."""

    result, _, _ = evaluate_signal_transition(
        control=_signal_group("group-a"),
        pre_state="YELLOW",
        post_state="YELLOW",
        pre_delta_m=20.0,
        post_delta_m=19.0,
        speed_mps=10.0,
        route_tangent_xy=(1.0, 0.0),
        delta_t_s=0.1,
        previous_yellow_must_stop=True,
        previous_signal_delta_m=20.0,
        previous_group_id="group-a",
        resolved_group_ids=frozenset(),
        crossing=False,
        ego_brake_mps2=4.0,
    )
    assert result.raw["yellow_must_stop"] is True


def test_yellow_onset_decision_uses_one_coherent_snapshot() -> None:
    """TEST-EF-21 / REQ-EF-14.

    ``d_req`` is built from the post-state approach speed, so v4.7 §7.6.3-4
    requires the post-state signed distance on the other side of the
    comparison; the historical pairing with ``pre_delta_m`` misclassified
    cases near the threshold. With v=10 m/s, dt=0.1 s, b=4 m/s^2,
    d_req = 13.5 m.
    """

    def onset(post_delta_m: float) -> bool:
        result, _, _ = evaluate_signal_transition(
            control=_signal_group("group-a"),
            pre_state="GREEN",
            post_state="YELLOW",
            pre_delta_m=post_delta_m + 1.0,
            post_delta_m=post_delta_m,
            speed_mps=10.0,
            route_tangent_xy=(1.0, 0.0),
            delta_t_s=0.1,
            previous_yellow_must_stop=False,
            previous_signal_delta_m=None,
            previous_group_id="group-a",
            resolved_group_ids=frozenset(),
            crossing=False,
            ego_brake_mps2=4.0,
        )
        return bool(result.raw["yellow_must_stop"])

    assert onset(14.0) is True
    assert onset(13.0) is False
