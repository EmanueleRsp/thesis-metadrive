from __future__ import annotations

from dataclasses import replace

import pytest

from thesis_rl.scenarios.arms import assign_primary_arm, derive_scenario_tags
from thesis_rl.scenarios.records import ScenarioFeatures


def _features(**overrides: object) -> ScenarioFeatures:
    values: dict[str, object] = {
        "scenario_id": "scenario",
        "source": "pg",
        "length": 10,
        "route_length_m": 9.0,
        "topology_tag": "simple",
        "has_intersection": False,
        "has_merge_or_roundabout": False,
        "has_route_traffic_light": False,
        "has_route_stop_sign": False,
        "has_route_crosswalk": False,
        "signal_reliability": "not_applicable",
        "has_vehicle": False,
        "has_pedestrian": False,
        "has_cyclist": False,
        "relevant_agents_q90": 0.0,
        "relevant_vehicles_q90": 0.0,
        "min_vehicle_distance_m": None,
        "min_vru_distance_to_route_m": None,
        "low_traffic": True,
        "dense_traffic": False,
        "vru_interaction": False,
    }
    values.update(overrides)
    return ScenarioFeatures(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_features(), "A0_simple_lane_follow"),
        (_features(has_vehicle=True, relevant_vehicles_q90=1.0), "A1_vehicle_interaction"),
        (
            _features(
                topology_tag="merge_or_roundabout", has_merge_or_roundabout=True
            ),
            "A2_merge_or_roundabout",
        ),
        (_features(topology_tag="intersection", has_intersection=True), "A3_intersection"),
        (_features(vru_interaction=True), "A4_vru_interaction"),
        (
            _features(
                topology_tag="mixed",
                has_intersection=True,
                has_merge_or_roundabout=True,
            ),
            "A5_complex_mixed",
        ),
        (
            _features(has_intersection=True, vru_interaction=True),
            "A5_complex_mixed",
        ),
    ],
)
def test_assign_primary_arm(features: ScenarioFeatures, expected: str) -> None:
    assert assign_primary_arm(features) == expected


def test_dense_vehicle_traffic_does_not_create_a5() -> None:
    features = _features(
        has_intersection=True,
        relevant_vehicles_q90=5.0,
        dense_traffic=True,
    )
    assert assign_primary_arm(features) == "A3_intersection"


def test_tags_keep_signal_uncertainty_independent() -> None:
    features = replace(
        _features(),
        has_intersection=True,
        has_route_traffic_light=None,
        signal_reliability="partial",
        dense_traffic=True,
    )
    tags = derive_scenario_tags(features, has_static_obstacle=True)

    assert "has_intersection" in tags
    assert "has_dense_traffic" in tags
    assert "has_unknown_signal" in tags
    assert "has_static_obstacle" in tags
    assert "has_signalized_intersection" not in tags
