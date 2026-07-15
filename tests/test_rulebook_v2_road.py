from math import pi

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.road import evaluate_dashed_line, evaluate_offroad, evaluate_solid_line, evaluate_wrongway
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot


def _ego(*, velocity=(0.0, 0.0), heading=0.0, cap=10.0):
    return ActorSnapshot("ego", ActorClass.VEHICLE, (1.0, 0.0), 0.0, heading, velocity,
                         Polygon(((0.0, -0.5), (2.0, -0.5), (2.0, 0.5), (0.0, 0.5))),
                         "lane", cap)


def test_offroad_area_fraction_and_numeric_epsilon():
    footprint = Polygon(((0, 0), (2, 0), (2, 1), (0, 1)))
    result, _, _ = evaluate_offroad(ego_footprint=footprint, drivable_surface=Polygon(((0, 0), (1, 0), (1, 1), (0, 1))))
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
    result, delta, _ = evaluate_dashed_line(ego_footprint=footprint, dashed_boundaries=(first, second), previous_boundary_id=None, previous_timer_s=0.0, delta_t_s=0.6)
    assert result.cost == 0.0 and dict(delta.writes)["dashed_line_timer_s"] == pytest.approx(0.6)
    result, delta, _ = evaluate_dashed_line(ego_footprint=footprint, dashed_boundaries=(first, second), previous_boundary_id="0", previous_timer_s=0.6, delta_t_s=0.6)
    assert result.cost == pytest.approx(0.04) and dict(delta.writes)["active_dashed_boundary_id"] == "0"
    far = Polygon(((9.99, -1), (10.01, -1), (10.01, 2), (9.99, 2)))
    result, delta, _ = evaluate_dashed_line(ego_footprint=footprint, dashed_boundaries=(far,), previous_boundary_id="0", previous_timer_s=1.5, delta_t_s=0.5)
    assert result.cost == 0.0 and dict(delta.writes)["dashed_line_timer_s"] == 0.0
