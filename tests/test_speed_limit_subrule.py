"""`T-RB51-16`: the `speed_limit` sub-rule and its provenance gate (ADR-068).

Two things are asserted here and they are independent. The **cost** is nuPlan's
`speed_limit_compliance` form normalized by the limit. The **applicability** is
decided by provenance -- whether the source record carried a real-map posted
limit -- and not by whether the derived number looks plausible, because the
values PG writes under the same key are constructor defaults in the wrong unit.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.speed_limit import (
    SPEED_LIMIT_TOLERANCE_MPS,
    evaluate_speed_limit,
)
from thesis_rl.rulebook.v2.context.waymo_static_adapter import posted_speed_limit_mps
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ComponentStatus


_LIMIT_25_MPH_MPS = 40.23 / 3.6


def test_cost_is_zero_up_to_the_published_tolerance() -> None:
    for speed in (0.0, _LIMIT_25_MPH_MPS, _LIMIT_25_MPH_MPS + SPEED_LIMIT_TOLERANCE_MPS):
        result, _, _ = evaluate_speed_limit(
            ego_velocity_xy=(speed, 0.0),
            posted_speed_limit_mps=_LIMIT_25_MPH_MPS,
        )
        assert result.cost == 0.0
        assert result.applicable is True
        assert result.status is ComponentStatus.SATISFIED


def test_cost_is_the_relative_excess_over_the_limit() -> None:
    """Normalizing by the limit is what makes the cost comparable across roads.

    The same absolute excess is a worse violation on a slow street than on a
    fast road, and the reward has to say so with one number.
    """

    slow = 40.23 / 3.6  # 25 mph
    fast = 72.42 / 3.6  # 45 mph
    excess = 5.0

    on_slow, _, _ = evaluate_speed_limit(
        ego_velocity_xy=(slow + SPEED_LIMIT_TOLERANCE_MPS + excess, 0.0),
        posted_speed_limit_mps=slow,
    )
    on_fast, _, _ = evaluate_speed_limit(
        ego_velocity_xy=(fast + SPEED_LIMIT_TOLERANCE_MPS + excess, 0.0),
        posted_speed_limit_mps=fast,
    )

    assert on_slow.cost == pytest.approx(excess / slow)
    assert on_fast.cost == pytest.approx(excess / fast)
    assert on_slow.cost > on_fast.cost


def test_cost_saturates_at_one() -> None:
    result, _, _ = evaluate_speed_limit(
        ego_velocity_xy=(200.0, 0.0), posted_speed_limit_mps=_LIMIT_25_MPH_MPS
    )
    assert result.cost == 1.0


def test_speed_is_the_velocity_magnitude_not_one_axis() -> None:
    diagonal = (_LIMIT_25_MPH_MPS + SPEED_LIMIT_TOLERANCE_MPS + 5.0) / (2.0**0.5)
    result, _, _ = evaluate_speed_limit(
        ego_velocity_xy=(diagonal, diagonal), posted_speed_limit_mps=_LIMIT_25_MPH_MPS
    )
    assert result.cost == pytest.approx(5.0 / _LIMIT_25_MPH_MPS)


def test_absent_limit_is_inapplicable_never_defaulted() -> None:
    """The v1 extractor's `max_speed_km_h` fallback is specifically prohibited.

    That constant is the vehicle's own cap, not a legal limit; substituting it
    would make this rule a vehicle cap disguised as a norm, and would charge the
    ego on PG for exceeding a number no traffic law anywhere produced.
    """

    result, _, _ = evaluate_speed_limit(ego_velocity_xy=(30.0, 0.0), posted_speed_limit_mps=None)
    assert result.applicable is False
    assert result.status is ComponentStatus.NOT_APPLICABLE
    assert result.cost == 0.0
    assert result.raw["posted_limit_mps"] is None


@pytest.mark.parametrize(
    ("lane", "reason"),
    [
        ({"speed_limit_kmh": 20.0}, "PG's create_pg_block_utils default, no source datum"),
        ({"speed_limit_kmh": 1000.0}, "PG's AbstractLane default, no source datum"),
        ({"speed_limit_kmh": 40.23}, "a plausible value with no source datum"),
        ({"speed_limit_mph": 25.0}, "source datum with no derived value"),
        ({"speed_limit_mph": 25.0, "speed_limit_kmh": 0.0}, "unrecorded"),
        ({"speed_limit_mph": 25.0, "speed_limit_kmh": 1000.0}, "sentinel"),
        ({}, "neither"),
    ],
)
def test_provenance_gate_rejects_everything_without_a_source_datum(lane, reason) -> None:
    """The admission test is provenance, not plausibility.

    The third case is the one that matters: 40.23 km/h is exactly what a real
    25 mph limit converts to, and it is still rejected without the source datum,
    because a value-based test cannot tell a converted limit from a constructor
    default that happens to look reasonable.
    """

    assert posted_speed_limit_mps(lane) is None, reason


def test_provenance_gate_admits_a_real_map_limit_and_converts_it() -> None:
    limit = posted_speed_limit_mps({"speed_limit_mph": 25.0, "speed_limit_kmh": 40.23})
    assert limit == pytest.approx(40.23 / 3.6)


def test_lane_record_refuses_a_non_positive_limit() -> None:
    polyline = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    polygon = Polygon(((0.0, -1.75), (10.0, -1.75), (10.0, 1.75), (0.0, 1.75)))
    with pytest.raises(ValueError, match="posted speed limit"):
        RouteLaneRecord("lane", polygon, polyline, (), (), 0.0)
    # And accepts absence, which is the normal case on PG.
    assert RouteLaneRecord("lane", polygon, polyline).posted_speed_limit_mps is None
