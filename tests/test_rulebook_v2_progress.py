import pytest
from shapely.geometry import Polygon
from thesis_rl.rulebook.v2.components.progress import evaluate_progress
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
