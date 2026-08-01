from __future__ import annotations

import glob
import pickle

import pytest

from thesis_rl.rulebook.v2.context.waymo_static_adapter import (
    _lane_record,
    build_waymo_static_adapter_result,
)


def _minimal_scenario(
    *, signal_lane_reachable: bool, signal_state: str = "LANE_STATE_UNKNOWN"
) -> dict:
    return {
        "id": "minimal",
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "lane-a": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "width": [3.5, 3.5],
                "exit_lanes": ["lane-b"] if signal_lane_reachable else [],
            },
            "lane-b": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "width": [3.5, 3.5],
                "exit_lanes": [],
            },
            "edge": {
                "type": "ROAD_EDGE_BOUNDARY",
                "polyline": [[3.0, -2.0, 0.0]],
            },
        },
        "dynamic_map_states": {
            "signal": {
                "type": "TRAFFIC_LIGHT",
                "lane": "lane-b",
                "stop_point": [11.0, 0.0, 0.0],
                "state": {"object_state": [signal_state, signal_state]},
            }
        },
    }


def test_bundled_waymo_fixture_converts_to_canonical_static_records():
    paths = sorted(glob.glob("third_party/metadrive/metadrive/assets/waymo/sd_*.pkl"))
    if not paths:
        pytest.skip("bundled Waymo fixture unavailable")
    with open(paths[0], "rb") as handle:
        scenario = pickle.load(handle)
    result = build_waymo_static_adapter_result(scenario, scenario_uid="waymo-fixture")
    assert result.task_route.lane_ids
    assert result.route_lanes
    assert result.scenario_uid == "waymo-fixture"
    assert not any(
        error.startswith("task_route_lane_missing") for error in result.validation_errors
    )
    assert result.movement_priority_records == ()

    # REQ-EF-05 / OPEN-EF-01: this bundled fixture's assigned route is not
    # contiguous (`assigned_route_invalid: ... 242->264`, reported by
    # `normalize_static_records`), so no canonical `RoutePolyline` exists and a
    # traffic control has no `route_s`. The scenario is already ineligible —
    # `build_episode_cache` rejects any result carrying validation errors — and
    # emitting a control with a lane-local coordinate instead is precisely the
    # defect REQ-EF-05 removes. The adapter therefore emits no controls and says
    # why. The "a stop control is produced with a correct canonical coordinate"
    # behaviour is covered by
    # `test_waymo_adapter_derives_a_canonical_stop_control_on_a_contiguous_route`.
    assert any(error.startswith("assigned_route_invalid") for error in result.validation_errors)
    assert "traffic_controls_skipped_unbuildable_assigned_route" in result.validation_errors
    assert result.traffic_controls == ()


def test_waymo_adapter_derives_a_canonical_stop_control_on_a_contiguous_route():
    """REQ-EF-05 / OPEN-EF-01 + OPEN-EF-04.

    Two 20 m lanes in series, with the stop sign placed 5 m into the *second*
    lane and offset 4 m to the roadside — which is where Waymo actually puts
    `STOP_SIGN.position` (measured over 991 sign/lane pairs: 99.4% fall outside
    the lane polygon, median lateral offset 4.32 m).

    Two regressions in one:
    - `route_s_m` must be the canonical route coordinate (25 m), not the
      lane-local one (5 m);
    - the control line must be derived at `q_c` on the centerline (§2.9.6 steps
      3 and 5), so a roadside sign position still yields a line across the lane
      rather than being offset or aborting with "multiple unresolved
      components".
    """

    def lane(lane_id, x0, x1, successors):
        return {
            "type": "LANE_SURFACE_STREET",
            "polyline": [[float(x), 0.0, 0.0] for x in range(x0, x1 + 1, 2)],
            "width": [[1.75, 1.75] for _ in range(x0, x1 + 1, 2)],
            "exit_lanes": list(successors),
        }

    scenario = {
        "map_features": {
            "lane-a": lane("lane-a", 0, 20, ("lane-b",)),
            "lane-b": lane("lane-b", 20, 40, ()),
            "stop-1": {
                "type": "STOP_SIGN",
                "position": [25.0, 4.0, 0.0],
                "lane": ["lane-b"],
            },
        },
        "metadata": {
            "assigned_route_lane_ids": ["lane-a", "lane-b"],
            "assigned_route_source": "test",
        },
    }
    result = build_waymo_static_adapter_result(scenario, scenario_uid="synthetic-stop")

    assert not result.validation_errors
    stops = [control for control in result.traffic_controls if control.control_type.value == "stop"]
    assert len(stops) == 1
    assert stops[0].route_s_m == pytest.approx(25.0, abs=0.05)
    # The line spans the lane across the centerline, not the sign's offset.
    min_x, min_y, max_x, max_y = stops[0].control_line.bounds
    assert min_x == pytest.approx(25.0, abs=0.05)
    assert max_x == pytest.approx(25.0, abs=0.05)
    assert min_y == pytest.approx(-1.75, abs=0.05)
    assert max_y == pytest.approx(1.75, abs=0.05)


def test_waymo_adapter_fails_fast_without_lane_geometry():
    with pytest.raises(ValueError, match="no lane geometry"):
        build_waymo_static_adapter_result({"map_features": {}, "metadata": {}}, scenario_uid="s")


def test_waymo_adapter_types_single_point_map_feature_instead_of_raising_geos():
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=False),
        scenario_uid="minimal",
    )
    assert "invalid_map_feature_geometry:edge" in result.validation_errors


def test_waymo_adapter_prefers_persisted_route_over_future_sdc_track() -> None:
    scenario = _minimal_scenario(signal_lane_reachable=False)
    scenario["metadata"]["assigned_route_lane_ids"] = ["lane-a", "lane-b"]
    scenario["metadata"]["assigned_route_source"] = "waymo_sdc_offline_task_annotation"
    scenario["tracks"]["ego"]["state"]["position"] = [[1000.0, 1000.0, 0.0]]
    scenario["tracks"]["ego"]["state"]["heading"] = [3.14]
    scenario["tracks"]["ego"]["state"]["valid"] = [True]

    result = build_waymo_static_adapter_result(scenario, scenario_uid="metadata-route")

    assert result.task_route.lane_ids == ("lane-a", "lane-b")
    assert result.task_route.route_assignment_source == "waymo_sdc_offline_task_annotation"


def test_waymo_adapter_preserves_lane_successors() -> None:
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=True), scenario_uid="topology"
    )
    lanes = {lane.lane_id: lane for lane in result.route_lanes}
    assert lanes["lane-a"].successor_lane_ids == ("lane-b",)
    assert lanes["lane-b"].successor_lane_ids == ()


def test_waymo_adapter_uses_per_side_widths_without_halving_them_again() -> None:
    lane = _lane_record(
        "asymmetric",
        {
            "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            "width": [[2.0, 1.0], [2.0, 1.0]],
        },
    )

    assert lane.polygon_xy.bounds == pytest.approx((0.0, -1.0, 10.0, 2.0))


def test_waymo_adapter_interpolates_missing_per_side_width_samples() -> None:
    lane = _lane_record(
        "partial-width",
        {
            "polyline": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            "width": [[2.0, 1.0], [0.0, 0.0], [2.0, 1.0]],
        },
    )

    assert lane.polygon_xy.bounds == pytest.approx((0.0, -1.0, 10.0, 2.0))


@pytest.mark.parametrize(
    ("marking_type", "expected_class_name"),
    [
        ("ROAD_LINE_SOLID_DOUBLE_WHITE", "LANE_MARKING_SOLID"),
        ("ROAD_LINE_BROKEN_SINGLE_YELLOW", "LANE_MARKING_DASHED"),
        ("ROAD_LINE_BROKEN_DOUBLE_YELLOW", "LANE_MARKING_DASHED"),
        ("ROAD_LINE_PASSING_DOUBLE_YELLOW", "LANE_MARKING_DASHED"),
        ("ROAD_EDGE_MEDIAN", "ROAD_BOUNDARY"),
    ],
)
def test_waymo_adapter_maps_newly_covered_marking_classes(
    marking_type: str, expected_class_name: str
) -> None:
    """TEST-RBCOST-010 / REQ-RBCOST-004/005: F2 regression, DEC-RBCOST-008/009."""
    from thesis_rl.rulebook.v2.types import MapFeatureClass

    scenario = _minimal_scenario(signal_lane_reachable=False)
    scenario["map_features"]["marking"] = {
        "type": marking_type,
        "polyline": [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    }
    result = build_waymo_static_adapter_result(scenario, scenario_uid="marking-coverage")
    matches = [
        feature
        for feature in result.map_features.values()
        if feature.feature_class is MapFeatureClass[expected_class_name]
    ]
    assert len(matches) == 1


def test_waymo_adapter_degenerate_new_marking_geometry_is_diagnostic_not_error() -> None:
    """DEC-RBCOST-006 regression.

    A read-only dry-run over the 1805 frozen Waymo records found 21 whose
    only *new* validation_errors entry, after the M3 marking-coverage change,
    was invalid_map_feature_geometry on a single-point instance of a newly
    covered class. Excluding them would shrink the frozen catalog, which the
    fallback option of DEC-RBCOST-006 forbids: for these classes only, a
    degenerate geometry becomes a diagnostic, not a validation error. Other
    classes (e.g. CROSSWALK) keep failing validation unchanged.
    """
    scenario = _minimal_scenario(signal_lane_reachable=False)
    scenario["map_features"]["degenerate_marking"] = {
        "type": "ROAD_LINE_PASSING_DOUBLE_YELLOW",
        "polyline": [[5.0, 0.0, 0.0]],
    }
    scenario["map_features"]["degenerate_crosswalk"] = {
        "type": "CROSSWALK",
        "polygon": [[5.0, 0.0, 0.0]],
    }
    result = build_waymo_static_adapter_result(scenario, scenario_uid="degenerate-marking")
    assert "degenerate_marking" not in result.map_features
    assert "degenerate_geometry:ROAD_LINE_PASSING_DOUBLE_YELLOW" in result.unmapped_feature_types
    assert not any(
        error.startswith("invalid_map_feature_geometry:degenerate_marking")
        for error in result.validation_errors
    )
    assert "invalid_map_feature_geometry:degenerate_crosswalk" in result.validation_errors


def test_waymo_adapter_records_unmapped_feature_type_without_a_validation_error() -> None:
    """TEST-RBCOST-011 / REQ-RBCOST-011."""
    scenario = _minimal_scenario(signal_lane_reachable=False)
    scenario["map_features"]["future"] = {
        "type": "ROAD_LINE_FUTURE_SCHEMA",
        "polyline": [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    }
    result = build_waymo_static_adapter_result(scenario, scenario_uid="unmapped-type")
    assert "future" not in result.map_features
    assert "ROAD_LINE_FUTURE_SCHEMA" in result.unmapped_feature_types
    assert not any("future" in error for error in result.validation_errors)


@pytest.mark.parametrize(
    ("reachable", "signal_state", "expected"),
    (
        (False, "LANE_STATE_UNKNOWN", False),
        (True, "LANE_STATE_UNKNOWN", True),
        (True, "TRAFFIC_LIGHT_UNKNOWN", True),
    ),
)
def test_waymo_adapter_validates_unknown_signal_only_when_topologically_relevant(
    reachable: bool,
    signal_state: str,
    expected: bool,
) -> None:
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=reachable, signal_state=signal_state),
        scenario_uid="minimal",
    )
    assert ("signal_state_unknown:signal" in result.validation_errors) is expected
