from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from thesis_rl.scenarios.features import extract_scenario_features


def _track(object_type: str, positions: np.ndarray, valid: np.ndarray | None = None) -> dict:
    length = len(positions)
    return {
        "type": object_type,
        "state": {
            "position": positions,
            "heading": np.zeros(length),
            "valid": np.ones(length, dtype=bool) if valid is None else valid,
        },
        "metadata": {"type": object_type, "object_id": object_type.lower()},
    }


def _scenario(*, dynamic_lights: bool = False) -> dict:
    length = 10
    ego = np.column_stack((np.arange(length), np.zeros(length), np.zeros(length)))
    vehicle = ego + np.array([10.0, 0.0, 0.0])
    pedestrian = ego + np.array([0.0, 6.0, 0.0])
    return {
        "id": "synthetic",
        "version": "test",
        "length": length,
        "tracks": {
            "ego": _track("VEHICLE", ego),
            "vehicle": _track("VEHICLE", vehicle),
            "pedestrian": _track("PEDESTRIAN", pedestrian),
        },
        "map_features": {
            "crosswalk": {"type": "CROSSWALK", "polygon": np.zeros((4, 3))},
        },
        "dynamic_map_states": (
            {
                "light": {
                    "type": "TRAFFIC_LIGHT",
                    "state": {"object_state": np.asarray(["LANE_STATE_STOP"] * length)},
                    "metadata": {"type": "TRAFFIC_LIGHT", "object_id": "light"},
                }
            }
            if dynamic_lights
            else {}
        ),
        "metadata": {
            "sdc_id": "ego",
            "scenario_id": "synthetic",
            "metadrive_processed": True,
            "coordinate": "metadrive",
            "ts": np.arange(length) / 10,
        },
    }


def _waymo_topology_scenario() -> dict:
    scenario = _scenario()
    scenario["map_features"] = {
        "10": {
            "type": "LANE_SURFACE_STREET",
            "polyline": np.column_stack(
                (np.arange(10), np.zeros(10), np.zeros(10))
            ),
            "entry_lanes": ["8", "9"],
            "exit_lanes": ["11"],
        },
        "crosswalk": {
            "type": "CROSSWALK",
            "polygon": np.asarray(
                [[4.0, -3.0, 0.0], [4.0, 3.0, 0.0], [5.0, 3.0, 0.0]]
            ),
        },
    }
    return scenario


def test_feature_extraction_counts_relevant_agents_and_vru() -> None:
    features = extract_scenario_features(_scenario(), "pg")

    assert features.route_length_m == 9.0
    assert features.relevant_agents_q90 == 2.0
    assert features.relevant_vehicles_q90 == 1.0
    assert features.relevant_vrus_q90 == 1.0
    assert features.min_vehicle_distance_m == 10.0
    assert features.min_vru_distance_to_route_m == 6.0
    assert features.vru_interaction is True
    assert features.vehicle_conflict_count == 0
    assert features.vru_conflict_count == 0
    assert features.topology_tag == "unknown"


def test_realized_topology_can_be_mixed_but_profile_name_is_ignored() -> None:
    metadata = {
        "profile": "P0_simple",
        "realized_topology": {
            "has_intersection": True,
            "has_merge_or_roundabout": True,
        },
    }
    features = extract_scenario_features(_scenario(), "pg", metadata)

    assert features.topology_tag == "mixed"
    assert features.has_intersection is True
    assert features.has_merge_or_roundabout is True


def test_unresolved_light_route_relevance_is_partial() -> None:
    features = extract_scenario_features(_scenario(dynamic_lights=True), "waymo")

    assert features.has_route_traffic_light is None
    assert features.signal_reliability == "partial"
    assert features.has_unknown_signal is True


def test_complete_realized_route_controls() -> None:
    metadata = {
        "route_traffic_controls": {
            "has_traffic_light": True,
            "traffic_light_states_complete": True,
            "has_stop_sign": False,
            "has_crosswalk": True,
        }
    }
    features = extract_scenario_features(_scenario(dynamic_lights=True), "pg", metadata)

    assert features.signal_reliability == "complete"
    assert features.has_route_stop_sign is False
    assert features.has_route_crosswalk is True


def test_waymo_feature_extraction_uses_route_aware_topology() -> None:
    features = extract_scenario_features(_waymo_topology_scenario(), "waymo")

    assert features.has_intersection is True
    assert features.has_merge_or_roundabout is False
    assert features.topology_tag == "intersection"
    assert features.topology_confidence == "high"
    assert "route_crosswalk" in features.topology_evidence
    assert "convergence_at_controlled_junction" in features.topology_evidence


def test_route_light_with_only_unknown_states_is_missing() -> None:
    scenario = _waymo_topology_scenario()
    scenario["dynamic_map_states"] = {
        "light": {
            "type": "TRAFFIC_LIGHT",
            "lane": "10",
            "stop_point": np.asarray([4.0, 0.0, 0.0]),
            "state": {"object_state": ["LANE_STATE_UNKNOWN"] * scenario["length"]},
        }
    }

    features = extract_scenario_features(scenario, "waymo")

    assert features.has_route_traffic_light is True
    assert features.signal_reliability == "missing"


def test_waymo_topology_stays_unknown_without_route_lane_evidence() -> None:
    scenario = _scenario()
    scenario["map_features"] = {}
    features = extract_scenario_features(scenario, "waymo")

    assert features.topology_tag == "unknown"
    assert features.topology_confidence == "unknown"


def test_bundled_waymo_feature_extraction_is_route_aware() -> None:
    fixture_dir = Path("third_party/metadrive/metadrive/assets/waymo")
    scenario_path = sorted(fixture_dir.glob("sd_*.pkl"))[0]
    with scenario_path.open("rb") as handle:
        scenario = pickle.load(handle)

    features = extract_scenario_features(scenario, "waymo")

    assert features.length == scenario["length"]
    assert features.route_length_m > 0
    assert features.topology_tag in {
        "simple", "merge_or_roundabout", "intersection", "mixed", "unknown"
    }
    assert np.isfinite(features.relevant_agents_q90)
