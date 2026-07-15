import pytest
from thesis_rl.rulebook.v2.components.controls import evaluate_crosswalk_yield
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval

def test_crosswalk_gap_and_commitment_are_continuous():
    result, delta, _ = evaluate_crosswalk_yield(zone_id="z", ego_interval=OccupancyInterval(1.0, 2.0), vru_intervals=(("ped", OccupancyInterval(2.2, 3.0)),), distance_to_entry_m=0.5, approach_speed_mps=4.0, delta_t_s=0.1, ego_occupied=False, ego_entered=False, preexisting_zone_ids=frozenset(), previous_illegal_entries=frozenset())
    assert 0.0 < result.cost < 1.0
    assert dict(delta.writes)["crosswalk_illegal_entries"] == frozenset()

def test_crosswalk_illegal_entry_persists_until_exit():
    result, delta, _ = evaluate_crosswalk_yield(zone_id="z", ego_interval=OccupancyInterval(0, None), vru_intervals=(("ped", OccupancyInterval(0.2, None)),), distance_to_entry_m=0.0, approach_speed_mps=0.0, delta_t_s=0.1, ego_occupied=True, ego_entered=True, preexisting_zone_ids=frozenset(), previous_illegal_entries=frozenset())
    assert result.cost == 1.0
    assert ("ped", "z") in dict(delta.writes)["crosswalk_illegal_entries"]
    result, delta, _ = evaluate_crosswalk_yield(zone_id="z", ego_interval=OccupancyInterval(2, 3), vru_intervals=(), distance_to_entry_m=-1, approach_speed_mps=0.0, delta_t_s=0.1, ego_occupied=False, ego_entered=False, preexisting_zone_ids=frozenset(), previous_illegal_entries=frozenset({("ped", "z")}))
    assert result.cost == 0.0 and dict(delta.writes)["crosswalk_illegal_entries"] == frozenset()

def test_crosswalk_missing_interval_fails_fast():
    with pytest.raises(ValueError):
        evaluate_crosswalk_yield(zone_id="z", ego_interval=None, vru_intervals=(), distance_to_entry_m=1, approach_speed_mps=0, delta_t_s=0.1, ego_occupied=False, ego_entered=False, preexisting_zone_ids=frozenset(), previous_illegal_entries=frozenset())
