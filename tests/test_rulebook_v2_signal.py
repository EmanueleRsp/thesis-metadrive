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
