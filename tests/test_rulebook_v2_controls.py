import pytest
from shapely.geometry import LineString

from thesis_rl.rulebook.v2.components.controls import evaluate_signal_state, select_active_signal_group, signal_group_state
from thesis_rl.rulebook.v2.types import ApproachControl, MovementKey, TrafficControlRecord


def _control(group, s, ids=("a",)):
    return TrafficControlRecord(group, ApproachControl.SIGNAL, ("lane",), MovementKey("a", "n", "e"), LineString(((s, -1), (s, 1))), s, 0.0, ids)


def test_signal_selection_is_route_ordered_and_skips_resolved():
    active = select_active_signal_group(controls=(_control("far", 20), _control("near", 10)), ego_front_s_m=5, resolved_group_ids=frozenset({"near"}))
    assert active is not None and active.control_group_id == "far"


def test_signal_group_requires_concordant_physical_states():
    control = _control("g", 10, ("a", "b"))
    assert signal_group_state(control=control, signal_states_by_physical_id={"a": "RED", "b": "RED"}) == "RED"
    with pytest.raises(ValueError):
        signal_group_state(control=control, signal_states_by_physical_id={"a": "RED", "b": "GREEN"})


def test_signal_component_not_applicable_without_group():
    result, delta, _ = evaluate_signal_state(control=None, ego_front_s_m=0, signal_states_by_physical_id={}, resolved_group_ids=frozenset())
    assert result.status.value == "not_applicable" and delta.writes == ()
