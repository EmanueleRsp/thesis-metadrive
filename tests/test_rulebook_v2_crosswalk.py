import pytest
from thesis_rl.rulebook.v2.aggregation import aggregate_max_component
from thesis_rl.rulebook.v2.components.controls import evaluate_crosswalk_yield
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval
from thesis_rl.rulebook.v2.types import ComponentStatus


def test_crosswalk_gap_and_commitment_are_continuous():
    result, delta, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(1.0, 2.0),
        vru_intervals=(("ped", OccupancyInterval(2.2, 3.0)),),
        distance_to_entry_m=0.5,
        approach_speed_mps=4.0,
        delta_t_s=0.1,
        ego_occupied=False,
        ego_entered=False,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    assert 0.0 < result.cost < 1.0
    assert dict(delta.writes)["crosswalk_illegal_entries"] == frozenset()


def test_crosswalk_illegal_entry_persists_until_exit():
    result, delta, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(0, None),
        vru_intervals=(("ped", OccupancyInterval(0.2, None)),),
        distance_to_entry_m=0.0,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=True,
        ego_entered=True,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    # ADR-064: detected and recorded, not priced. The second half of this test --
    # that the entry clears once the ego leaves the zone -- is the part that
    # still bears on behaviour, and it is unchanged.
    assert result.cost == 0.0
    assert result.diagnostics["latched_illegal_entry"] is True
    assert ("ped", "z") in dict(delta.writes)["crosswalk_illegal_entries"]
    result, delta, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(2, 3),
        vru_intervals=(),
        distance_to_entry_m=-1,
        approach_speed_mps=0.0,
        delta_t_s=0.1,
        ego_occupied=False,
        ego_entered=False,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset({("ped", "z")}),
        ego_brake_mps2=4.0,
    )
    assert result.cost == 0.0 and dict(delta.writes)["crosswalk_illegal_entries"] == frozenset()


def test_crosswalk_missing_interval_fails_fast():
    with pytest.raises(ValueError):
        evaluate_crosswalk_yield(
            zone_id="z",
            ego_interval=None,
            vru_intervals=(),
            distance_to_entry_m=1,
            approach_speed_mps=0,
            delta_t_s=0.1,
            ego_occupied=False,
            ego_entered=False,
            preexisting_zone_ids=frozenset(),
            previous_illegal_entries=frozenset(),
        )


def test_crosswalk_active_latch_no_longer_reaches_the_channel() -> None:
    """TEST-EF-15 / REQ-EF-08, superseded by ADR-064.

    REQ-EF-08 existed because a latched cost of 1.0 with ``applicable=False``
    was silently discarded by ``aggregate_max_component``. With the latch out of
    the cost the coupling has no subject: there is no latched cost left to
    discard, so the component is inapplicable when no VRU is predicted and the
    channel is 0 because nothing is being charged -- not because something is
    being dropped. The distinction is the whole point of the original
    regression, so it is asserted rather than deleted.
    """

    result, _, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(0.0, 1.0),
        vru_intervals=(),
        distance_to_entry_m=0.0,
        approach_speed_mps=3.0,
        delta_t_s=0.1,
        ego_occupied=True,
        ego_entered=False,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset({("ped", "z")}),
        ego_brake_mps2=4.0,
    )
    assert result.cost == 0.0
    assert result.applicable is False
    assert result.status is ComponentStatus.NOT_APPLICABLE
    # The entry is still visible, so "not charged" never becomes "not observed".
    assert result.diagnostics["latched_illegal_entry"] is True

    aggregated = aggregate_max_component(name="non_relaxable_compliance", components=(result,))
    assert aggregated.applicable is False
    assert aggregated.cost == 0.0


def test_crosswalk_stays_not_applicable_without_vru_and_without_latch() -> None:
    """TEST-EF-15 / REQ-EF-08: the applicability widening is latch-scoped only."""

    result, _, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(0.0, 1.0),
        vru_intervals=(),
        distance_to_entry_m=0.0,
        approach_speed_mps=3.0,
        delta_t_s=0.1,
        ego_occupied=True,
        ego_entered=False,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset(),
        ego_brake_mps2=4.0,
    )
    assert result.applicable is False
    assert result.cost == 0.0
    assert result.status is ComponentStatus.NOT_APPLICABLE


def test_crosswalk_latch_on_another_zone_does_not_make_this_zone_applicable() -> None:
    """TEST-EF-15 / REQ-EF-08: the latch must be matched by zone id."""

    result, _, _ = evaluate_crosswalk_yield(
        zone_id="z",
        ego_interval=OccupancyInterval(0.0, 1.0),
        vru_intervals=(),
        distance_to_entry_m=0.0,
        approach_speed_mps=3.0,
        delta_t_s=0.1,
        ego_occupied=True,
        ego_entered=False,
        preexisting_zone_ids=frozenset(),
        previous_illegal_entries=frozenset({("ped", "other-zone")}),
        ego_brake_mps2=4.0,
    )
    assert result.applicable is False
    assert result.cost == 0.0
