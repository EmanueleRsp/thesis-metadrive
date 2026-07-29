from math import pi, sqrt

import pytest
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.components.road import (
    dashed_lateral_penetration,
    evaluate_dashed_line,
    evaluate_offroad,
    evaluate_solid_line,
    evaluate_wrongway,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot


def _ego(*, velocity=(0.0, 0.0), heading=0.0, cap=10.0):
    return ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (1.0, 0.0),
        0.0,
        heading,
        velocity,
        Polygon(((0.0, -0.5), (2.0, -0.5), (2.0, 0.5), (0.0, 0.5))),
        "lane",
        cap,
    )


def test_offroad_area_fraction_and_numeric_epsilon():
    footprint = Polygon(((0, 0), (2, 0), (2, 1), (0, 1)))
    result, _, _ = evaluate_offroad(
        ego_footprint=footprint, drivable_surface=Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))
    )
    assert result.cost == pytest.approx(0.5)
    tiny = Polygon(((-1e-5, 0), (2, 0), (2, 1), (-1e-5, 1)))
    result, _, _ = evaluate_offroad(ego_footprint=footprint, drivable_surface=tiny)
    assert result.cost == 0.0


def test_wrongway_uses_signed_route_velocity_and_not_heading_at_rest():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    result, _, _ = evaluate_wrongway(ego=_ego(velocity=(-5.0, 0.0), heading=pi), route=route)
    assert result.cost == pytest.approx(0.5)
    result, _, _ = evaluate_wrongway(ego=_ego(velocity=(0.0, 0.0), heading=pi), route=route)
    assert result.cost == 0.0


def test_wrongway_fails_without_speed_cap():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        evaluate_wrongway(ego=_ego(cap=None), route=route)


def test_solid_line_detects_occupancy_and_crossing():
    footprint = Polygon(((0, 0), (2, 0), (2, 1), (0, 1)))
    line = Polygon(((0.99, -1), (1.01, -1), (1.01, 2), (0.99, 2)))
    result, _, _ = evaluate_solid_line(ego_footprint=footprint, solid_boundaries=(line,))
    assert result.cost == 1.0
    clear = Polygon(((3, -1), (3.01, -1), (3.01, 2), (3, 2)))
    result, _, _ = evaluate_solid_line(ego_footprint=footprint, solid_boundaries=(clear,))
    assert result.cost == 0.0


def test_dashed_line_timer_is_continuous_and_resets_on_boundary_change():
    first = Polygon(((0.99, -1), (1.01, -1), (1.01, 2), (0.99, 2)))
    second = Polygon(((4.99, -1), (5.01, -1), (5.01, 2), (4.99, 2)))
    footprint = Polygon(((0, 0), (2, 0), (2, 1), (0, 1)))
    result, delta, _ = evaluate_dashed_line(
        ego_footprint=footprint,
        dashed_boundaries=(first, second),
        previous_boundary_id=None,
        previous_timer_s=0.0,
        delta_t_s=0.6,
    )
    assert result.cost == 0.0 and dict(delta.writes)["dashed_line_timer_s"] == pytest.approx(0.6)
    result, delta, _ = evaluate_dashed_line(
        ego_footprint=footprint,
        dashed_boundaries=(first, second),
        previous_boundary_id="0",
        previous_timer_s=0.6,
        delta_t_s=0.6,
    )
    assert (
        result.cost == pytest.approx(0.04)
        and dict(delta.writes)["active_dashed_boundary_id"] == "0"
    )
    far = Polygon(((9.99, -1), (10.01, -1), (10.01, 2), (9.99, 2)))
    result, delta, _ = evaluate_dashed_line(
        ego_footprint=footprint,
        dashed_boundaries=(far,),
        previous_boundary_id="0",
        previous_timer_s=1.5,
        delta_t_s=0.5,
    )
    assert result.cost == 0.0 and dict(delta.writes)["dashed_line_timer_s"] == 0.0


# 2 m x 1 m footprint, centroid (1.0, 0.5), so the lateral half-extent is 0.5 m.
_FOOTPRINT = Polygon(((0, 0), (2, 0), (2, 1), (0, 1)))


@pytest.mark.parametrize(
    ("marking_y", "expected_penetration"),
    [
        (0.5, 1.0),  # through the centroid: straddling the marking
        (0.75, 0.5),  # half way out
        (1.0, 0.0),  # tangent to the footprint edge
    ],
)
def test_dashed_penetration_is_one_on_the_centre_and_zero_at_the_footprint_edge(
    marking_y: float, expected_penetration: float
) -> None:
    """TEST-EF-26 / REQ-EF-17."""

    marking = LineString(((0.0, marking_y), (2.0, marking_y)))
    assert dashed_lateral_penetration(_FOOTPRINT, marking) == pytest.approx(expected_penetration)


def test_dashed_penetration_uses_the_directional_half_extent_not_half_the_width() -> None:
    """TEST-EF-27 / REQ-EF-17.

    During a lane change the ego is yawed relative to the marking, so its extent
    towards the marking exceeds half its width. Normalizing by half the width
    would leave a dead band where the marking is still inside the footprint but
    the penetration has already reached 0.

    A 2 m square yawed 45 degrees is a diamond with vertices at +-sqrt(2) on each
    axis. A marking 1.0 m from the centroid is well inside it, but a half-width
    normalizer (1.0 m) would score it exactly 0.
    """

    diamond = Polygon(((sqrt(2), 0), (0, sqrt(2)), (-sqrt(2), 0), (0, -sqrt(2))))
    marking = LineString(((-2.0, 1.0), (2.0, 1.0)))
    assert diamond.intersects(marking)
    penetration = dashed_lateral_penetration(diamond, marking)
    assert penetration == pytest.approx(1.0 - 1.0 / sqrt(2), abs=1e-6)
    assert penetration > 0.0


def test_dashed_cost_grades_in_space_as_well_as_in_time() -> None:
    """TEST-EF-28 / REQ-EF-17.

    Regression: activation is a spatial *step* -- any contact with the footprint,
    down to a bumper corner, counts -- and the cost was graded only in time. So an
    ego drifting with one wheel over the marking and an ego straddling it with its
    centre on it both saturated to 1.0 after 2 s. R3 could not tell them apart,
    and R3 outranks R4, so both were taught the same penalty.
    """

    def saturated_cost(marking_y: float) -> float:
        marking = LineString(((0.0, marking_y), (2.0, marking_y)))
        result, _, _ = evaluate_dashed_line(
            ego_footprint=_FOOTPRINT,
            dashed_boundaries=(marking,),
            previous_boundary_id="0",
            previous_timer_s=5.0,
            delta_t_s=0.1,
        )
        return result.cost

    # Time factor is saturated at 1.0 in all three cases, so the cost is the
    # penetration alone -- which the previous implementation collapsed to 1.0.
    assert saturated_cost(0.5) == pytest.approx(1.0)
    assert saturated_cost(0.75) == pytest.approx(0.5)
    assert saturated_cost(1.0) == pytest.approx(0.0)


def test_dashed_timer_and_penetration_are_independent_factors() -> None:
    """TEST-EF-29 / REQ-EF-17.

    The timer measures how long the ego has been engaged with this marking; the
    penetration measures how badly it is engaged *now*. Keeping them separate is
    what lets an ego that has lingered at the edge and then cuts inward be
    penalized immediately, without a fresh 1 s grace period.
    """

    edge = LineString(((0.0, 1.0), (2.0, 1.0)))
    lingering, delta, _ = evaluate_dashed_line(
        ego_footprint=_FOOTPRINT,
        dashed_boundaries=(edge,),
        previous_boundary_id="0",
        previous_timer_s=1.9,
        delta_t_s=0.1,
    )
    assert lingering.cost == pytest.approx(0.0)
    assert lingering.diagnostics["time_factor"] == pytest.approx(1.0)
    assert dict(delta.writes)["dashed_line_timer_s"] == pytest.approx(2.0)

    # Same marking id, ego now straddling it: the accumulated timer applies at
    # once instead of restarting.
    inward = LineString(((0.0, 0.5), (2.0, 0.5)))
    cutting_in, _, _ = evaluate_dashed_line(
        ego_footprint=_FOOTPRINT,
        dashed_boundaries=(inward,),
        previous_boundary_id="0",
        previous_timer_s=2.0,
        delta_t_s=0.1,
    )
    assert cutting_in.cost == pytest.approx(1.0)
