from thesis_rl.rulebook.v2.components.controls import evaluate_vehicle_yield
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval


def test_vehicle_yield_approach_and_persistent_illegal_entry():
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(("car", OccupancyInterval(2.2, 3)),),
        distance_to_entry_m=0.5,
        approach_speed_mps=4,
        delta_t_s=0.1,
        ego_occupied=False,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    assert 0 < result.cost < 1
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, None),
        prioritized_intervals=(("car", OccupancyInterval(1.1, None)),),
        distance_to_entry_m=0,
        approach_speed_mps=0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset({"car"}),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    # ADR-064: the illegal entry is still detected and recorded, but no longer
    # priced. What used to be a latched 1.0 is now a diagnostic; the approach
    # term above, which prices the decision while it is still being made, is what
    # remains in the reward.
    assert result.cost == 0.0
    assert result.diagnostics["latched_illegal_entry"] is True
    assert ("car", "z") in dict(delta.writes)["vehicle_yield_illegal_entries"]


def test_vehicle_yield_freezes_movement_key_until_complete_exit():
    key = ("approach", "node", "exit")
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(("car", OccupancyInterval(2.2, 3)),),
        distance_to_entry_m=0.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset({"car"}),
        previous_illegal_entries=frozenset(),
        actor_movement_keys=(("car", key),),
        ego_brake_mps2=4.0,
    )
    assert dict(delta.writes)["frozen_actor_movement_keys"] == (("car", key),)
    _, exit_delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(2, 3),
        prioritized_intervals=(),
        distance_to_entry_m=-1.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=False,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        previous_frozen_movement_keys=(("car", key),),
        exited_actor_ids=frozenset({"car"}),
    )
    assert dict(exit_delta.writes)["frozen_actor_movement_keys"] == ()


def test_vehicle_yield_pre_state_gap_creates_latch_with_empty_post_state_intervals():
    """DEC-005 Fase A/B (REQ-VY-01): an illegal entry judged from the pre-state
    gap is still recorded when the post-state has zero live prioritized actors
    (e.g. the actor left the conflict zone during the same control step).

    ADR-064 removed REQ-VY-04's half of this: the latch no longer sets the cost
    or forces applicability. The pre/post-state conformance the test exists for
    is unchanged and is what is asserted here."""
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(),
        distance_to_entry_m=0.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        pre_state_entered_actor_ids=frozenset({"car"}),
        pre_state_gap_violation=0.6,
    )
    assert result.cost == 0.0
    assert result.diagnostics["latched_illegal_entry"] is True
    assert ("car", "z") in dict(delta.writes)["vehicle_yield_illegal_entries"]


def test_vehicle_yield_pre_state_sufficient_gap_creates_no_latch():
    """REQ-VY-01: a sufficient pre-state gap (r_gap^- == 0) must not latch,
    even though the actor is present in the pre-state entered set."""
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(("car", OccupancyInterval(2.2, 3)),),
        distance_to_entry_m=0.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        pre_state_entered_actor_ids=frozenset({"car"}),
        pre_state_gap_violation=0.0,
        ego_brake_mps2=4.0,
    )
    assert ("car", "z") not in dict(delta.writes)["vehicle_yield_illegal_entries"]
    assert result.cost == 0.0


def test_vehicle_yield_active_latch_is_carried_but_no_longer_priced():
    """ADR-064 supersedes REQ-VY-04.

    The latch used to keep the component applicable and VIOLATED at cost 1.0 for
    as long as the ego occupied the zone. That is removed: inside the zone no
    action changes the cost, so the agent was charged repeatedly for a decision
    already taken with no gradient toward better behaviour, and the latch was a
    memory the agent could not see. It is still carried across steps and still
    reported, so an illegal entry remains auditable -- it is simply not in the
    reward."""
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, 2),
        prioritized_intervals=(),
        distance_to_entry_m=0.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset({("car", "z")}),
    )
    assert result.cost == 0.0
    assert result.applicable is False
    assert result.diagnostics["latched_illegal_entry"] is True
    assert ("car", "z") in dict(delta.writes)["vehicle_yield_illegal_entries"]


def test_vehicle_yield_illegal_entry_latch_clears_once_ego_stops_occupying_zone():
    """REQ-VY-01/04: a pre-existing latch for a zone must be released as
    soon as ego is no longer occupying it, regardless of live prioritized
    actors. (Whether the live transition keeps re-selecting the same
    zone_id long enough after ego's footprint fully clears it is a separate
    zone-selection-continuity question, outside the scope of this
    pre/post-state conformance fix — see the ExecPlan's Progress Log.)"""
    _, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(2, 3),
        prioritized_intervals=(),
        distance_to_entry_m=-1.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=False,
        entered_actor_ids=frozenset(),
        previous_illegal_entries=frozenset({("other", "z")}),
    )
    assert dict(delta.writes)["vehicle_yield_illegal_entries"] == frozenset()


def test_vehicle_yield_legacy_caller_without_pre_state_gap_falls_back_to_post_state_worst():
    """Backward compatibility: a caller that never separated pre/post
    snapshots (``pre_state_gap_violation`` omitted) keeps gating the latch
    on the post-state ``worst`` gap, exactly as before this change."""
    result, delta, _ = evaluate_vehicle_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1, None),
        prioritized_intervals=(("car", OccupancyInterval(1.1, None)),),
        distance_to_entry_m=0,
        approach_speed_mps=0,
        delta_t_s=0.1,
        ego_occupied=True,
        entered_actor_ids=frozenset({"car"}),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    assert result.diagnostics["latched_illegal_entry"] is True
    assert ("car", "z") in dict(delta.writes)["vehicle_yield_illegal_entries"]
