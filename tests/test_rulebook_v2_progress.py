import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.components.progress import evaluate_progress, route_outside_fraction
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot

def _ego(x, cap=10):
    return ActorSnapshot("ego", ActorClass.VEHICLE, (x, 0), 0, 0, (1, 0), Polygon(((x-.5,-.5),(x+.5,-.5),(x+.5,.5),(x-.5,.5))), "lane", cap)

def test_progress_returns_raw_delta_and_normalized_margin():
    result, delta, _ = evaluate_progress(pre_ego=_ego(1), post_ego=_ego(3), route=RoutePolyline(((0,0,0),(10,0,0))), previous_route_s_m=1, delta_t_s=1)
    assert result.raw["route_delta_m"] == pytest.approx(2)
    assert result.cost == pytest.approx(.2)
    assert dict(delta.writes)["previous_route_s_m"] == pytest.approx(3)

def test_progress_rejects_discontinuous_memory():
    with pytest.raises(ValueError):
        evaluate_progress(pre_ego=_ego(1), post_ego=_ego(2), route=RoutePolyline(((0,0,0),(10,0,0))), previous_route_s_m=5, delta_t_s=1)


def test_route_outside_fraction_measures_corridor_adherence_without_changing_cost() -> None:
    """TEST-EF-22 / REQ-EF-15 (M6a, diagnostics only).

    R4 credits projected longitudinal progress with no lateral cut-off, and the
    R3 off-road surface is the union of *every* map lane, so advancing along a
    legal parallel road scores positive progress at zero off-road cost. This
    measures the frequency of that situation so the corridor gate of DEC-EF-06
    can be decided from data. It must not affect the margin.
    """

    corridor = Polygon([(0.0, -1.75), (100.0, -1.75), (100.0, 1.75), (0.0, 1.75)])
    inside = oriented_bounding_box(
        center_xy=(10.0, 0.0), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    outside = oriented_bounding_box(
        center_xy=(10.0, 20.0), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    straddling = oriented_bounding_box(
        center_xy=(10.0, 1.75), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    assert route_outside_fraction(inside, corridor) == pytest.approx(0.0)
    assert route_outside_fraction(outside, corridor) == pytest.approx(1.0)
    partial = route_outside_fraction(straddling, corridor)
    assert 0.0 < partial < 1.0
    assert route_outside_fraction(inside, None) is None
