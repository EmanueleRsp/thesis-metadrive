from __future__ import annotations

from math import pi

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.rss_lateral import (
    LateralRSSCandidate,
    evaluate_rss_lateral,
    lateral_safe_distance_m,
)
from thesis_rl.rulebook.v2.context.live_adapter import actor_snapshot_from_payload
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.transition import _rss_lateral_candidates
from thesis_rl.rulebook.v2.types import ActorClass


def test_lateral_safe_distance_matches_reference_value_at_zero_speed() -> None:
    """AC-R2-03 (rulebook v4.8 §7, §12): two vehicles with zero inward
    lateral speed at rho_lat=0.5s, a_acc_max^lat=0.2, a_brake_min^lat=0.8,
    mu=0.10 give d_safe^lat ~= 0.1625 m."""
    safe = lateral_safe_distance_m(ego_inward_speed_mps=0.0, actor_inward_speed_mps=0.0)
    assert safe == pytest.approx(0.1625, abs=1.0e-6)


def test_evaluate_rss_lateral_not_applicable_without_candidates() -> None:
    result, _, _ = evaluate_rss_lateral(candidates=())
    assert result.applicable is False
    assert result.cost == 0.0
    assert result.status.value == "not_applicable"


def test_evaluate_rss_lateral_ignores_longitudinally_safe_candidates() -> None:
    """AC-R2-01/05: a candidate outside the longitudinal gate contributes
    zero cost even with a tiny lateral gap (stable-but-close, parallel
    lanes)."""
    result, _, _ = evaluate_rss_lateral(
        candidates=(
            LateralRSSCandidate(
                actor_id="parallel",
                lateral_gap_m=0.05,
                ego_inward_speed_mps=0.0,
                actor_inward_speed_mps=0.0,
                longitudinal_unsafe=False,
            ),
        ),
    )
    assert result.applicable is True
    assert result.cost == 0.0
    assert result.status.value == "satisfied"


def test_evaluate_rss_lateral_penalizes_a_gap_below_the_safe_distance() -> None:
    """AC-R2-02: a genuine lateral approach (gate true, gap below
    d_safe^lat) produces a positive bounded cost."""
    result, _, _ = evaluate_rss_lateral(
        candidates=(
            LateralRSSCandidate(
                actor_id="closing",
                lateral_gap_m=0.05,
                ego_inward_speed_mps=1.0,
                actor_inward_speed_mps=0.0,
                longitudinal_unsafe=True,
            ),
        ),
    )
    assert 0.0 < result.cost <= 1.0
    assert result.status.value == "violated"
    assert result.raw["worst_actor_id"] == "closing"


def test_evaluate_rss_lateral_worst_of_multiple_candidates() -> None:
    safe_candidate = LateralRSSCandidate("safe", 5.0, 0.0, 0.0, True)
    unsafe_candidate = LateralRSSCandidate("unsafe", 0.0, 1.0, 1.0, True)
    result, _, _ = evaluate_rss_lateral(candidates=(safe_candidate, unsafe_candidate))
    assert result.raw["worst_actor_id"] == "unsafe"
    assert result.cost == pytest.approx(1.0)


def test_evaluate_rss_lateral_rejects_negative_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        evaluate_rss_lateral(
            candidates=(LateralRSSCandidate("bad", -0.1, 0.0, 0.0, True),),
        )


# ---------------------------------------------------------------------------
# ADR-035 / REQ-EF-02, REQ-EF-03: candidate scoping.
#
# Before ADR-035 the scoping had no test coverage at all
# (``grep -rn "_rss_lateral_candidates" tests/`` returned nothing), which is
# why the suite passed while a receding rear vehicle, any same-lane leader,
# and a perpendicular crossing vehicle all produced spurious R2 violations.
# ---------------------------------------------------------------------------

EGO_BRAKE_MPS2 = 4.0
_EGO_LANE_ID = "EGO"
_ADJACENT_LANE_ID = "ADJ"


def _vehicle(actor_id, position_xy, velocity_xy, *, heading_rad=0.0):
    return actor_snapshot_from_payload(
        {
            "actor_id": actor_id,
            "actor_class": ActorClass.VEHICLE,
            "position_xy": position_xy,
            "position_z": 0.0,
            "heading_rad": heading_rad,
            "velocity_xy": velocity_xy,
            "length_m": 4.515,
            "width_m": 1.852,
            "live_lane_id": None,
            "configured_speed_cap_mps": 30.0,
        }
    )


def _straight_lane(lane_id, centre_y, *, reversed_direction=False):
    xs = range(200, -101, -2) if reversed_direction else range(-100, 201, 2)
    return RouteLaneRecord(
        lane_id,
        Polygon(
            [
                (-100.0, centre_y - 1.75),
                (200.0, centre_y - 1.75),
                (200.0, centre_y + 1.75),
                (-100.0, centre_y + 1.75),
            ]
        ),
        RoutePolyline(tuple((float(x), centre_y, 0.0) for x in xs)),
    )


def _crossing_lane():
    return RouteLaneRecord(
        "CROSS",
        Polygon([(28.25, -40.0), (31.75, -40.0), (31.75, 40.0), (28.25, 40.0)]),
        RoutePolyline(tuple((30.0, float(y), 0.0) for y in range(40, -41, -2))),
    )


def _lateral_cost(ego, actors, route_lanes):
    candidates = _rss_lateral_candidates(
        ego=ego,
        actors=actors,
        route=route_lanes[0].centerline,
        route_lanes=route_lanes,
        ego_brake_mps2=EGO_BRAKE_MPS2,
    )
    result, _, _ = evaluate_rss_lateral(candidates=candidates)
    return candidates, result


def test_rss_lateral_excludes_a_receding_rear_vehicle() -> None:
    """TEST-EF-04 / REQ-EF-02.

    Regression for the ego-always-rear gate: an ego at 20 m/s with a vehicle
    20 m behind at 5 m/s (i.e. falling further behind) used to be scored
    ``longitudinal_unsafe=True`` and ``q_RSS,lat = 1.0``.
    """

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0),)
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("rear", (-20.0, 0.0), (5.0, 0.0)),),
        lanes,
    )
    assert candidates == ()
    assert result.cost == 0.0
    assert result.applicable is False


def test_rss_lateral_excludes_a_closing_rear_vehicle_leaving_it_to_ttc() -> None:
    """TEST-EF-05 / REQ-EF-02.

    The mirror case of the same defect, previously a false negative
    (``longitudinal_unsafe=False``). Coverage is retained by TTC per v4.8 §8.
    """

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0),)
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (5.0, 0.0)),
        (_vehicle("rear", (-20.0, 0.0), (20.0, 0.0)),),
        lanes,
    )
    assert candidates == ()
    assert result.applicable is False


@pytest.mark.parametrize("gap_m", [15.0, 40.0, 60.0])
def test_rss_lateral_excludes_a_same_lane_leader(gap_m: float) -> None:
    """TEST-EF-06 / REQ-EF-02.

    Two vehicles in one lane have a lateral gap of exactly zero against
    ``d_safe^lat ~= 0.1625 m``, so every longitudinally unsafe same-lane pair
    used to cost ``1.0`` regardless of the gap, masking the graded
    ``q_RSS,long`` (0.840 / 0.460 / 0.156 at these three gaps).
    """

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0),)
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("lead", (gap_m, 0.0), (20.0, 0.0)),),
        lanes,
    )
    assert candidates == ()
    assert result.cost == 0.0


def test_rss_lateral_excludes_a_perpendicular_crossing_vehicle() -> None:
    """TEST-EF-07 / REQ-EF-02, REQ-EF-03 (v4.8 §8 requires NOT_APPLICABLE)."""

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0), _crossing_lane())
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("cross", (30.0, 6.0), (0.0, -8.0), heading_rad=-pi / 2.0),),
        lanes,
    )
    assert candidates == ()
    assert result.applicable is False


def test_rss_lateral_keeps_an_abreast_adjacent_lane_pair_without_convergence() -> None:
    """TEST-EF-08 / REQ-EF-02: the metric's proper domain is preserved."""

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0), _straight_lane(_ADJACENT_LANE_ID, 3.5))
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("side", (0.0, 3.5), (20.0, 0.0)),),
        lanes,
    )
    assert len(candidates) == 1
    assert candidates[0].longitudinal_unsafe is True
    assert result.applicable is True
    assert result.cost == 0.0


def test_rss_lateral_grades_a_converging_abreast_pair() -> None:
    """TEST-EF-09 / REQ-EF-02: the cost stays graded, not binary."""

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0), _straight_lane(_ADJACENT_LANE_ID, 3.5))
    _, mild = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("side", (0.0, 3.5), (20.0, -1.0)),),
        lanes,
    )
    _, strong = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("side", (0.0, 3.5), (20.0, -1.5)),),
        lanes,
    )
    assert mild.cost == 0.0
    assert 0.0 < strong.cost < 1.0
    assert strong.cost == pytest.approx(0.342, abs=1.0e-3)


def test_rss_lateral_excludes_an_abreast_vehicle_on_an_opposing_lane() -> None:
    """TEST-EF-10 / REQ-EF-03: bounded lane-tangent misalignment."""

    lanes = (
        _straight_lane(_EGO_LANE_ID, 0.0),
        _straight_lane("OPP", 3.5, reversed_direction=True),
    )
    candidates, result = _lateral_cost(
        _vehicle("ego", (0.0, 0.0), (20.0, 0.0)),
        (_vehicle("opp", (0.0, 3.5), (-20.0, 0.0), heading_rad=pi),),
        lanes,
    )
    assert candidates == ()
    assert result.applicable is False


def test_rss_lateral_candidate_order_is_independent_of_actor_order() -> None:
    """TEST-EF-23: determinism."""

    lanes = (_straight_lane(_EGO_LANE_ID, 0.0), _straight_lane(_ADJACENT_LANE_ID, 3.5))
    ego = _vehicle("ego", (0.0, 0.0), (20.0, 0.0))
    first = _vehicle("aaa", (0.0, 3.5), (20.0, -1.5))
    second = _vehicle("bbb", (1.0, 3.5), (20.0, -1.2))
    forward, _ = _lateral_cost(ego, (first, second), lanes)
    backward, _ = _lateral_cost(ego, (second, first), lanes)
    assert forward == backward
    assert [candidate.actor_id for candidate in forward] == ["aaa", "bbb"]
