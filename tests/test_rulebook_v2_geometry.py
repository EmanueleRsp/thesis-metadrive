from __future__ import annotations

import pytest
import shapely
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.geometry.canonical import (
    CanonicalGeometryError,
    canonical_geometry_wkb,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.drivable import (
    DrivableLaneRecord,
    _union_selected_surfaces,
    drivable_surface_for_ego,
)
from thesis_rl.rulebook.v2.geometry.controls import derive_control_line
from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    deterministic_convex_decomposition,
    predict_occupancy_interval,
)
from thesis_rl.rulebook.v2.geometry.conflict_zones import (
    MovementCorridor,
    attach_route_intervals,
    build_crosswalk_conflict_zone_candidates,
    build_vehicle_conflict_zone_candidates,
    _vertical_overlap_compatible,
    select_first_ahead_or_occupied_zone,
)
from thesis_rl.rulebook.v2.geometry.footprint import (
    front_bumper_segment,
    oriented_bounding_box,
    swept_front_bumper,
)
from thesis_rl.rulebook.v2.geometry.lanes import (
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    derive_lane_movement_key,
    footprint_route_coordinates,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import vertically_compatible_at_xy
from thesis_rl.rulebook.v2.types import MovementKey


def test_lane_movement_key_uses_unique_successor_or_assigned_route() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    unique = RouteLaneRecord(
        "approach",
        Polygon(((-1.0, -2.0), (11.0, -2.0), (11.0, 2.0), (-1.0, 2.0))),
        route,
        ("exit",),
    )
    assert derive_lane_movement_key(unique) is not None
    assert derive_lane_movement_key(unique).exit_lane_id == "exit"

    branching = RouteLaneRecord(
        "approach",
        unique.polygon_xy,
        route,
        ("left", "right"),
    )
    assert derive_lane_movement_key(branching) is None
    resolved = derive_lane_movement_key(
        branching,
        assigned_route_lane_ids=("approach", "right"),
    )
    assert resolved is not None
    assert resolved.exit_lane_id == "right"


def test_canonical_wkb_and_synthetic_id_ignore_ring_orientation_and_small_noise() -> None:
    clockwise = Polygon(((0.0004, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0004, 0.0)))
    counterclockwise = Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
    assert canonical_geometry_wkb(clockwise) == canonical_geometry_wkb(counterclockwise)
    assert stable_geometry_id(
        scenario_id="s", namespace="map", feature_type="lane", geometry=clockwise
    ) == stable_geometry_id(
        scenario_id="s", namespace="map", feature_type="lane", geometry=counterclockwise
    )


def test_canonical_geometry_rejects_collapsed_or_invalid_input() -> None:
    with pytest.raises(CanonicalGeometryError, match="collapsed"):
        canonical_geometry_wkb(LineString(((0.0, 0.0), (0.0004, 0.0))))
    with pytest.raises(CanonicalGeometryError, match="invalid"):
        canonical_geometry_wkb(
            Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
        )


def test_vertical_compatibility_does_not_match_overpass_to_underpass() -> None:
    assert vertically_compatible_at_xy(
        first_elevation_at_xy=lambda _x, _y: 10.0,
        second_elevation_at_xy=lambda _x, _y: 12.9,
        x=0.0,
        y=0.0,
    )
    assert not vertically_compatible_at_xy(
        first_elevation_at_xy=lambda _x, _y: 10.0,
        second_elevation_at_xy=lambda _x, _y: 13.1,
        x=0.0,
        y=0.0,
    )


def test_invalid_elevation_is_fail_fast() -> None:
    with pytest.raises(ValueError, match="finite"):
        vertically_compatible_at_xy(
            first_elevation_at_xy=lambda _x, _y: float("nan"),
            second_elevation_at_xy=lambda _x, _y: 0.0,
            x=0.0,
            y=0.0,
        )


def test_oriented_bounding_box_uses_pose_heading_and_strict_dimensions() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0
    )
    assert footprint.bounds == pytest.approx((-2.0, -1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        oriented_bounding_box(center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.0, width_m=2.0)


def test_front_bumper_and_swept_region_are_canonical() -> None:
    pre = oriented_bounding_box(center_xy=(0.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    post = oriented_bounding_box(center_xy=(2.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    assert front_bumper_segment(pre, heading_rad=0.0).bounds == pytest.approx((2.0, -1.0, 2.0, 1.0))
    assert swept_front_bumper(
        pre, post, pre_heading_rad=0.0, post_heading_rad=0.0
    ).bounds == pytest.approx((2.0, -1.0, 4.0, 1.0))


def test_elevation_profile_interpolates_and_breaks_nearest_segment_ties_by_index() -> None:
    elevation = PolylineElevation(((0.0, 0.0, 0.0), (10.0, 0.0, 10.0), (10.0, 10.0, 30.0)))
    assert elevation(5.0, 1.0) == pytest.approx(5.0)
    assert elevation(10.0, 0.0) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="duplicate XY"):
        PolylineElevation(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))


def test_route_projection_uses_reset_and_previous_s_tie_breaks_at_self_intersection() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0)))
    at_reset = route.project((1.0, 1.0))
    near_final_crossing = route.project((1.0, 1.0), previous_s_m=route.length_m - 1.0)
    assert at_reset.segment_index == 0
    assert near_final_crossing.segment_index == 2


def test_route_consolidates_noisy_xy_points_with_median_elevation_and_rejects_large_spread() -> (
    None
):
    route = RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0005, 2.0), (10.0, 0.0, 4.0)))
    assert route.points_xyz[0] == pytest.approx((0.00025, 0.00025, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0, 3.1), (10.0, 0.0, 4.0)))


def test_route_projection_rejects_incompatible_vertical_level() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="vertically compatible"):
        route.project((5.0, 0.0), position_z=3.1)


def test_route_projection_diagnostics_report_nearest_and_vertical_feasibility() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 6.0)))
    diagnostics = route.projection_diagnostics((10.0, 0.0), position_z=2.9)
    assert diagnostics.nearest_planar_segment_index == 0
    assert diagnostics.nearest_planar_s_m == pytest.approx(10.0)
    assert diagnostics.nearest_planar_z_m == pytest.approx(6.0)
    assert diagnostics.nearest_planar_distance_m == pytest.approx(0.0)
    assert diagnostics.minimum_vertical_difference_m == pytest.approx(3.1)
    assert diagnostics.vertically_compatible_segment_count == 0
    assert diagnostics.route_min_z_m == pytest.approx(0.0)
    assert diagnostics.route_max_z_m == pytest.approx(6.0)


def test_lane_association_rejects_vertical_and_geometric_ties() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane-a", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route
    )
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane,)
        ).lane_id
        == "lane-a"
    )
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=3.1, heading_rad=0.0, route_lanes=(lane,)
        )
        is None
    )
    tied = RouteLaneRecord("lane-b", lane.polygon_xy, route)
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane, tied)
        )
        is None
    )


def test_bumper_gap_handles_separated_tangent_and_overlapping_footprints() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)))
    ego = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    other = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(12.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    assert bumper_to_bumper_gap(ego, other) == pytest.approx((3.0, True))
    tangent = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(9.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    assert bumper_to_bumper_gap(ego, tangent) == pytest.approx((0.0, True))


def test_drivable_surface_uses_current_vertical_layer_and_width_fallback() -> None:
    lower = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    upper = RoutePolyline(((0.0, 0.0, 10.0), (10.0, 0.0, 10.0)))
    ego = oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    lanes = (
        DrivableLaneRecord("lower", lower, None, 4.0),
        DrivableLaneRecord("upper", upper, None, 4.0),
    )
    surface = drivable_surface_for_ego(
        ego_footprint=ego,
        ego_position_xy=(5.0, 0.0),
        ego_position_z=0.0,
        lanes=lanes,
    )
    assert surface.bounds == pytest.approx((0.0, -2.0, 10.0, 2.0))
    with pytest.raises(ValueError, match="lane_width"):
        DrivableLaneRecord("broken", lower, None, None)


def test_drivable_surface_remains_available_when_ego_is_off_lane() -> None:
    lane = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    ego = oriented_bounding_box(center_xy=(50.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    surface = drivable_surface_for_ego(
        ego_footprint=ego,
        ego_position_xy=(50.0, 0.0),
        ego_position_z=0.0,
        lanes=(DrivableLaneRecord("lane", lane, None, 4.0),),
    )
    assert not surface.is_empty


def test_drivable_surface_reuses_exact_union_for_same_lane_layer() -> None:
    _union_selected_surfaces.cache_clear()
    lane = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    ego = oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    lanes = (DrivableLaneRecord("lane", lane, None, 4.0),)

    first = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.0, 0.0), ego_position_z=0.0, lanes=lanes
    )
    second = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.5, 0.0), ego_position_z=0.0, lanes=lanes
    )

    assert first.equals(second)
    assert _union_selected_surfaces.cache_info().hits == 1


def test_derived_control_line_is_orthogonal_and_signed_upstream_positive() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route
    )
    control = derive_control_line(
        control_point_xy=(5.0, 0.0), control_point_z=0.0, controlled_lane=lane
    )
    assert control.geometry.bounds == pytest.approx((5.0, -2.0, 5.0, 2.0))
    assert control.route_s_m == pytest.approx(5.0)
    assert control.signed_distance_m((4.0, 0.0)) == pytest.approx(1.0)
    assert control.signed_distance_m((6.0, 0.0)) == pytest.approx(-1.0)


def test_conflict_zone_candidates_are_wkb_ordered_stable_and_vertical_filtered() -> None:
    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    other = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        Polygon(((1.0, -1.0), (2.0, -1.0), (2.0, 3.0), (1.0, 3.0))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=other
    )
    repeated = build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=other
    )
    assert len(candidates) == 1
    assert candidates[0].zone_id == repeated[0].zone_id
    assert candidates[0].component_index == 0
    elevated_other = MovementCorridor(other.movement_key, other.polygon, lambda _x, _y: 3.1)
    assert not build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=elevated_other
    )


def test_vertical_overlap_accepts_repaired_multipolygon() -> None:
    geometry = shapely.MultiPolygon(
        (
            Polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            Polygon(((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))),
        )
    )
    assert _vertical_overlap_compatible(geometry, lambda _x, _y: 0.0, lambda _x, _y: 0.0)


def test_conflict_zone_selection_prefers_occupied_then_first_ahead_along_route() -> None:
    route = RoutePolyline(((0.0, 1.0, 0.0), (10.0, 1.0, 0.0)))
    ego_corridor = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    other_corridor = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        shapely.MultiPolygon(
            (
                Polygon(((1.0, -1.0), (2.0, -1.0), (2.0, 3.0), (1.0, 3.0))),
                Polygon(((5.0, -1.0), (6.0, -1.0), (6.0, 3.0), (5.0, 3.0))),
            )
        ),
        lambda _x, _y: 0.0,
    )
    attached = attach_route_intervals(
        route=route,
        candidates=build_vehicle_conflict_zone_candidates(
            scenario_id="scenario", ego_corridor=ego_corridor, other_corridor=other_corridor
        ),
    )
    ahead = select_first_ahead_or_occupied_zone(
        candidates=attached,
        ego_footprint=oriented_bounding_box(
            center_xy=(0.0, 1.0), heading_rad=0.0, length_m=0.5, width_m=0.5
        ),
        ego_front_s_m=0.25,
    )
    assert ahead is not None
    assert ahead.route_entry_s_m == pytest.approx(0.99)
    occupied = select_first_ahead_or_occupied_zone(
        candidates=attached,
        ego_footprint=oriented_bounding_box(
            center_xy=(5.5, 1.0), heading_rad=0.0, length_m=0.5, width_m=0.5
        ),
        ego_front_s_m=5.75,
    )
    assert occupied is not None
    assert occupied.route_entry_s_m == pytest.approx(4.99)


def test_crosswalk_zone_uses_its_own_namespace_and_no_other_movement_key() -> None:
    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    zones = build_crosswalk_conflict_zone_candidates(
        scenario_id="scenario",
        ego_corridor=ego,
        crosswalk_id="crosswalk-1",
        crosswalk_polygon=Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (1.0, 3.0))),
        crosswalk_elevation_at_xy=lambda _x, _y: 0.0,
    )
    assert len(zones) == 1
    assert zones[0].crosswalk_id == "crosswalk-1"
    assert not build_crosswalk_conflict_zone_candidates(
        scenario_id="scenario",
        ego_corridor=ego,
        crosswalk_id="crosswalk-1",
        crosswalk_polygon=zones[0].polygon,
        crosswalk_elevation_at_xy=lambda _x, _y: 3.1,
    )


def test_continuous_sat_returns_no_interval_finite_interval_and_open_end() -> None:
    zone = Polygon(((0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (0.0, 1.0)))
    crossing = Polygon(((-4.0, -0.5), (-2.0, -0.5), (-2.0, 0.5), (-4.0, 0.5)))
    finite = predict_occupancy_interval(
        actor_footprint=crossing, actor_velocity_xy=(4.0, 0.0), zone=zone, horizon_s=2.0
    )
    assert finite is not None
    assert finite.start_s == pytest.approx(0.5)
    assert finite.end_s == pytest.approx(1.5)
    assert (
        predict_occupancy_interval(
            actor_footprint=crossing, actor_velocity_xy=(0.0, 4.0), zone=zone, horizon_s=2.0
        )
        is None
    )
    inside = Polygon(((0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)))
    open_end = predict_occupancy_interval(
        actor_footprint=inside, actor_velocity_xy=(0.0, 0.0), zone=zone, horizon_s=3.0
    )
    assert open_end is not None
    assert open_end.start_s == 0.0
    assert open_end.is_open_end


def test_deterministic_decomposition_handles_concavity_and_hole() -> None:
    concave = Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)))
    holed = Polygon(
        ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)),
        holes=(((1.0, 1.0), (1.0, 4.0), (4.0, 4.0), (4.0, 1.0)),),
    )
    for polygon in (concave, holed):
        triangles = deterministic_convex_decomposition(polygon)
        assert all(polygon.covers(triangle) and triangle.area > 0.0 for triangle in triangles)
        assert shapely.union_all(triangles).area == pytest.approx(polygon.area)


def test_merge_corridors_produce_a_single_stable_conflict_component() -> None:
    ego = MovementCorridor(
        MovementKey("merge-ego", "merge-node", "main-out"),
        Polygon(((0, -1), (6, -1), (6, 1), (0, 1))),
        lambda _x, _y: 0.0,
    )
    merging = MovementCorridor(
        MovementKey("merge-other", "merge-node", "main-out"),
        Polygon(((3, -3), (5, -3), (5, 3), (3, 3))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="merge",
        ego_corridor=ego,
        other_corridor=merging,
    )
    assert len(candidates) == 1
    assert candidates[0].component_index == 0


def test_roundabout_corridor_intersection_remains_2_5d_filtered() -> None:
    circulating = MovementCorridor(
        MovementKey("roundabout", "rotary", "exit"),
        Polygon(((-4, -1), (4, -1), (4, 1), (-4, 1))),
        lambda _x, _y: 0.0,
    )
    entering = MovementCorridor(
        MovementKey("entry", "rotary", "roundabout"),
        Polygon(((1, -4), (3, -4), (3, 4), (1, 4))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="rotary",
        ego_corridor=circulating,
        other_corridor=entering,
    )
    assert len(candidates) == 1
    elevated_entry = MovementCorridor(entering.movement_key, entering.polygon, lambda _x, _y: 3.1)
    assert (
        build_vehicle_conflict_zone_candidates(
            scenario_id="rotary",
            ego_corridor=circulating,
            other_corridor=elevated_entry,
        )
        == ()
    )
