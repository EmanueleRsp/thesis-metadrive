from shapely.geometry import LineString
import pytest
from thesis_rl.rulebook.v2.components.controls import evaluate_stop, select_active_stop_group
from thesis_rl.rulebook.v2.types import ApproachControl, MovementKey, TrafficControlRecord

def _stop(group, s):
    return TrafficControlRecord(group, ApproachControl.STOP, ("lane",), MovementKey("a", "n", "e"), LineString(((s, -1), (s, 1))), s, 0.0, ())

def test_stop_selection_route_ordered():
    selected = select_active_stop_group(controls=(_stop("far", 20), _stop("near", 10)), ego_front_s_m=5, resolved_group_ids=frozenset())
    assert selected.control_group_id == "near"

def test_stop_crossing_uses_best_dwell_and_persists_resolution():
    result, delta, _ = evaluate_stop(control=_stop("s", 10), pre_delta_m=0.2, post_delta_m=-0.1, speed_mps=0.0,
                                     previous_continuous_s=0.6, previous_best_s=0.6, delta_t_s=0.4,
                                     previous_group_id="s", resolved_group_ids=frozenset(), crossing=True)
    assert result.cost == pytest.approx(0.4)
    assert "s" in dict(delta.writes)["resolved_stop_group_ids"]

def test_stop_rolling_crossing_is_violated():
    result, _, _ = evaluate_stop(control=_stop("s", 10), pre_delta_m=0.2, post_delta_m=-0.1, speed_mps=1.0,
                                 previous_continuous_s=0.0, previous_best_s=0.0, delta_t_s=0.1,
                                 previous_group_id="s", resolved_group_ids=frozenset(), crossing=True)
    assert result.cost == 1.0
