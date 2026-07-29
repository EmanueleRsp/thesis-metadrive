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
        (_features(), "A0_simple_low_traffic"),
        (_features(has_vehicle=True, relevant_vehicles_q90=9.0), "A1_traffic"),
        (
            _features(
                topology_tag="intersection",
                has_intersection=True,
                relevant_agents_q90=8.0,
                vehicle_conflict_count=1,
            ),
            "A1_traffic",
        ),
        (
            _features(
                topology_tag="merge_or_roundabout",
                has_merge_or_roundabout=True,
                relevant_agents_q90=9.0,
            ),
            "A2_junction",
        ),
        (
            _features(
                topology_tag="intersection",
                has_intersection=True,
                relevant_agents_q90=25.0,
            ),
            "A3_complex_junction",
        ),
        (_features(vru_interaction=True), "A4_vru"),
        (
            _features(
                has_intersection=True,
                vru_interaction=True,
                vru_conflict_count=1,
            ),
            "A5_critical_mixed",
        ),
        (
            _features(
                topology_tag="mixed",
                has_intersection=True,
                has_merge_or_roundabout=True,
                vehicle_conflict_count=3,
            ),
            "A5_critical_mixed",
        ),
        (
            _features(
                has_intersection=True,
                relevant_agents_q90=30.0,
                vehicle_conflict_count=6,
            ),
            "A5_critical_mixed",
        ),
    ],
)
def test_assign_primary_arm(features: ScenarioFeatures, expected: str) -> None:
    assert assign_primary_arm(features) == expected


def test_complex_junction_below_critical_threshold_is_a3() -> None:
    features = _features(
        has_intersection=True,
        relevant_vehicles_q90=5.0,
        relevant_agents_q90=25.0,
        vehicle_conflict_count=3,
    )
    assert assign_primary_arm(features) == "A3_complex_junction"


def test_static_obstacle_excludes_an_otherwise_simple_scenario_from_a0() -> None:
    assert assign_primary_arm(_features(), has_static_obstacle=True) == "A1_traffic"


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_features(has_intersection=True, relevant_agents_q90=25.0), "A3_complex_junction"),
        (_features(vru_interaction=True), "A4_vru"),
        (_features(has_intersection=True, vru_conflict_count=1), "A5_critical_mixed"),
    ],
)
def test_static_obstacle_does_not_change_non_a0_arm_precedence(
    features: ScenarioFeatures, expected: str
) -> None:
    assert assign_primary_arm(features, has_static_obstacle=True) == expected


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
