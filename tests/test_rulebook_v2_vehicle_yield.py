from thesis_rl.rulebook.v2.components.controls import evaluate_vehicle_yield
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval

def test_vehicle_yield_approach_and_persistent_illegal_entry():
    result, delta, _ = evaluate_vehicle_yield(zone_id="z", ego_interval=OccupancyInterval(1, 2), prioritized_intervals=(("car", OccupancyInterval(2.2, 3)),), distance_to_entry_m=0.5, approach_speed_mps=4, delta_t_s=0.1, ego_occupied=False, entered_actor_ids=frozenset(), previous_illegal_entries=frozenset())
    assert 0 < result.cost < 1
    result, delta, _ = evaluate_vehicle_yield(zone_id="z", ego_interval=OccupancyInterval(1, None), prioritized_intervals=(("car", OccupancyInterval(1.1, None)),), distance_to_entry_m=0, approach_speed_mps=0, delta_t_s=0.1, ego_occupied=True, entered_actor_ids=frozenset({"car"}), previous_illegal_entries=frozenset())
    assert result.cost == 1 and ("car", "z") in dict(delta.writes)["vehicle_yield_illegal_entries"]


def test_vehicle_yield_freezes_movement_key_until_complete_exit():
    key = ("approach", "node", "exit")
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z", ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(("car", OccupancyInterval(2.2, 3)),),
        distance_to_entry_m=0.0, approach_speed_mps=0.0, delta_t_s=0.1,
        ego_occupied=True, entered_actor_ids=frozenset({"car"}),
        previous_illegal_entries=frozenset(), actor_movement_keys=(("car", key),),
    )
    assert dict(delta.writes)["frozen_actor_movement_keys"] == (("car", key),)
    _, exit_delta, _ = evaluate_vehicle_yield(
        zone_id="z", ego_interval=OccupancyInterval(2, 3), prioritized_intervals=(),
        distance_to_entry_m=-1.0, approach_speed_mps=0.0, delta_t_s=0.1,
        ego_occupied=False, entered_actor_ids=frozenset(), previous_illegal_entries=frozenset(),
        previous_frozen_movement_keys=(("car", key),), exited_actor_ids=frozenset({"car"}),
    )
    assert dict(exit_delta.writes)["frozen_actor_movement_keys"] == ()
