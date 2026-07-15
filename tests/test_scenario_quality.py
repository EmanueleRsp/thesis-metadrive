from __future__ import annotations

from dataclasses import replace

from thesis_rl.scenarios.quality import apply_catalog_quality_policy
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _record() -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid="waymo:training_20s:scenario-1",
        scenario_id="scenario-1",
        source="waymo",
        relative_path="waymo/database/scenario-1.pkl",
        official_split="training_20s",
        source_log_id="log-1",
        source_scenario_id="scenario-1",
        dataset_version="training_20s",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=100,
        pg_profile=None,
        pg_seed=None,
        map_id=None,
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def _features(**overrides: object) -> ScenarioFeatures:
    values: dict[str, object] = {
        "scenario_id": "scenario-1",
        "source": "waymo",
        "length": 100,
        "route_length_m": 80.0,
        "topology_tag": "simple",
        "has_intersection": False,
        "has_merge_or_roundabout": False,
        "has_route_traffic_light": False,
        "has_route_stop_sign": False,
        "has_route_crosswalk": False,
        "signal_reliability": "not_applicable",
        "has_vehicle": True,
        "has_pedestrian": False,
        "has_cyclist": False,
        "relevant_agents_q90": 1.0,
        "relevant_vehicles_q90": 1.0,
        "min_vehicle_distance_m": 4.0,
        "min_vru_distance_to_route_m": None,
        "low_traffic": False,
        "dense_traffic": False,
        "vru_interaction": False,
        "sdc_valid_ratio": 1.0,
        "sdc_initial_valid": True,
        "sdc_route_z_range_m": 0.0,
        "map_feature_count": 4,
        "dynamic_object_count": 10,
    }
    values.update(overrides)
    return ScenarioFeatures(**values)  # type: ignore[arg-type]


def test_catalog_quality_policy_keeps_clean_record_valid() -> None:
    record = apply_catalog_quality_policy(_record(), _features())

    assert record.validation_status == "valid"
    assert record.validation_warnings == ()


def test_catalog_quality_policy_invalidates_hard_filter_failures() -> None:
    record = apply_catalog_quality_policy(
        _record(),
        _features(
            route_length_m=5.0,
            sdc_initial_valid=False,
            sdc_valid_ratio=0.6,
            sdc_route_z_range_m=5.0,
            map_feature_count=0,
            dynamic_object_count=121,
        ),
    )

    assert record.validation_status == "invalid"
    assert any("degenerate SDC route" in warning for warning in record.validation_warnings)
    assert any("invalid at t=0" in warning for warning in record.validation_warnings)
    assert any("low SDC valid ratio" in warning for warning in record.validation_warnings)
    assert any("overpass" in warning for warning in record.validation_warnings)
    assert any("no map features" in warning for warning in record.validation_warnings)
    assert any("too many dynamic objects" in warning for warning in record.validation_warnings)


def test_catalog_quality_policy_preserves_existing_warning_status() -> None:
    original = replace(
        _record(),
        validation_status="warning",
        validation_warnings=("existing warning",),
    )

    record = apply_catalog_quality_policy(original, _features())

    assert record.validation_status == "warning"
    assert record.validation_warnings == ("existing warning",)
