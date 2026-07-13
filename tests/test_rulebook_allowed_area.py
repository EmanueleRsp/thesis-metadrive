from __future__ import annotations

from types import SimpleNamespace

import pytest

from thesis_rl.envs.wrappers import RuleRewardWrapper


def test_allowed_area_prefers_navigation_reference_lanes() -> None:
    shapely = pytest.importorskip("shapely.geometry")
    lane_a = SimpleNamespace(shapely_polygon=shapely.box(0.0, 0.0, 10.0, 3.0))
    lane_b = SimpleNamespace(shapely_polygon=shapely.box(10.0, 0.0, 20.0, 3.0))
    vehicle = SimpleNamespace(
        navigation=SimpleNamespace(current_ref_lanes=[lane_a], next_ref_lanes=[lane_b])
    )

    area, source = RuleRewardWrapper._extract_allowed_driving_area(vehicle, None, None)

    assert area is not None
    assert source == "navigation.current_ref_lanes + next_ref_lanes"
    assert area.area == pytest.approx(60.0)


def test_drivable_area_repairs_invalid_lane_polygons() -> None:
    shapely = pytest.importorskip("shapely.geometry")
    # A bow-tie polygon is invalid and used to make unary_union fail for the
    # entire Waymo map.
    invalid_lane = SimpleNamespace(
        shapely_polygon=shapely.Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    )
    valid_lane = SimpleNamespace(shapely_polygon=shapely.box(2.0, 0.0, 4.0, 2.0))
    env = SimpleNamespace(
        current_map=SimpleNamespace(
            road_network=SimpleNamespace(get_all_lanes=lambda: [invalid_lane, valid_lane])
        )
    )
    wrapper = object.__new__(RuleRewardWrapper)
    wrapper._cached_drivable_area = None

    area = wrapper._extract_drivable_area(env)

    assert area is not None
    assert area.is_valid
    assert area.area > 0.0


def test_route_progress_uses_scenario_env_step_info() -> None:
    progress, source = RuleRewardWrapper._extract_route_progress(
        SimpleNamespace(navigation=SimpleNamespace()),
        {"route_completion": 0.25, "track_length": 120.0},
    )

    assert progress == pytest.approx(30.0)
    assert source == "info.route_completion * info.track_length"


def test_route_progress_accepts_trajectory_navigation_coordinate() -> None:
    progress, source = RuleRewardWrapper._extract_route_progress(
        SimpleNamespace(navigation=SimpleNamespace(current_longitude=17.5)),
        {},
    )

    assert progress == pytest.approx(17.5)
    assert source == "ego_vehicle.navigation.current_longitude"


def test_lane_markings_prefer_scenario_map_features() -> None:
    shapely = pytest.importorskip("shapely.geometry")
    current_map = SimpleNamespace(
        get_boundary_line_vector=lambda *, interval: {
            "solid": {"type": "ROAD_LINE_SOLID_SINGLE_WHITE", "polyline": [[0.0, 0.0], [10.0, 0.0]]},
            "dashed": {"type": "ROAD_LINE_BROKEN_SINGLE_WHITE", "polyline": [[0.0, 3.0], [10.0, 3.0]]},
        }
    )
    env = SimpleNamespace(current_map=current_map)

    solid, dashed, source = RuleRewardWrapper._extract_lane_markings(env, lane=None)

    assert source == "current_map.get_boundary_line_vector"
    assert len(solid) == 1
    assert len(dashed) == 1
    assert solid[0].intersects(shapely.Point(1.0, 0.0))
