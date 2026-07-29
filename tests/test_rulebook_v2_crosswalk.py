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
    assert result.cost == 1.0
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


def test_crosswalk_active_latch_stays_applicable_and_reaches_r3() -> None:
    """TEST-EF-15 / REQ-EF-08.

    Regression: with the illegal-entry latch active and the ego still in the
    zone, the component reported ``cost=1.0`` but ``applicable=False`` as soon
    as the VRU left the prediction set. ``aggregate_max_component`` filters on
    ``applicable``, so the aggregated R3 collapsed to 0.0.
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
    assert result.cost == pytest.approx(1.0)
    assert result.applicable is True
    assert result.status is ComponentStatus.VIOLATED

    aggregated = aggregate_max_component(
        name="road_traffic_compliance", components=(result,)
    )
    assert aggregated.applicable is True
    assert aggregated.cost == pytest.approx(1.0)


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
