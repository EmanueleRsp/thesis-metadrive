from __future__ import annotations

import pytest

from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _record(**overrides: object) -> ScenarioRecord:
    values: dict[str, object] = {
        "scenario_uid": "pg:v1:scenario-1",
        "scenario_id": "scenario-1",
        "source": "pg",
        "relative_path": "pg/database/scenario-1.pkl",
        "official_split": None,
        "source_log_id": None,
        "source_scenario_id": None,
        "dataset_version": "v1",
        "converter_version": None,
        "split": "train",
        "runtime_index": 0,
        "length": 100,
        "pg_profile": "P0_simple",
        "pg_seed": 12,
        "map_id": "S",
        "primary_arm": "A0_simple_low_traffic",
        "tags": ("has_crosswalk",),
        "signal_reliability": "not_applicable",
        "validation_status": "valid",
        "validation_warnings": (),
    }
    values.update(overrides)
    return ScenarioRecord(**values)  # type: ignore[arg-type]


def _features(**overrides: object) -> ScenarioFeatures:
    values: dict[str, object] = {
        "scenario_id": "scenario-1",
        "source": "pg",
        "length": 100,
        "route_length_m": 80.0,
        "topology_tag": "unknown",
        "has_intersection": None,
        "has_merge_or_roundabout": None,
        "has_route_traffic_light": None,
        "has_route_stop_sign": None,
        "has_route_crosswalk": None,
        "signal_reliability": "partial",
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
    }
    values.update(overrides)
    return ScenarioFeatures(**values)  # type: ignore[arg-type]


def test_scenario_record_round_trip_preserves_tuples() -> None:
    record = _record()
    restored = ScenarioRecord.from_dict(record.to_dict())

    assert restored == record
    assert isinstance(restored.tags, tuple)


def test_scenario_record_round_trip_preserves_rulebook_eligibility() -> None:
    record = _record(
        rulebook_eligible=False,
        rulebook_validation_errors=("unresolved route signal",),
    )

    restored = ScenarioRecord.from_dict(record.to_dict())

    assert restored.rulebook_eligible is False
    assert restored.rulebook_validation_errors == ("unresolved route signal",)


def test_scenario_record_round_trip_preserves_assigned_route_metadata() -> None:
    record = _record(
        assigned_route_lane_ids=("lane-a", "lane-b"),
        assigned_route_source="pg_sdc_offline_task_annotation",
    )

    restored = ScenarioRecord.from_dict(record.to_dict())

    assert restored.assigned_route_lane_ids == ("lane-a", "lane-b")
    assert restored.assigned_route_source == "pg_sdc_offline_task_annotation"


def test_scenario_record_round_trip_preserves_driving_mission() -> None:
    from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection

    span = LaneSpan("lane-a", 0.0, 10.0)
    goal = DirectedGate("goal", (span,), "lane-a", 8.0)
    mission = DrivingMissionRecord(
        _record().scenario_uid,
        "mission-builder-v1",
        (MissionSection("section", span, (span,), goal),),
        goal,
    )
    restored = ScenarioRecord.from_dict(_record(driving_mission=mission.to_dict()).to_dict())

    assert restored.driving_mission is not None
    assert restored.driving_mission["mission_hash"] == mission.mission_hash


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute/scenario.pkl", "../escape.pkl", "pg/../escape.pkl", "pg\\scenario.pkl"],
)
def test_scenario_record_rejects_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(ValueError, match="relative_path"):
        _record(relative_path=relative_path)


def test_pg_record_requires_seed_and_waymo_forbids_it() -> None:
    with pytest.raises(ValueError, match="PG records require"):
        _record(pg_seed=None)
    with pytest.raises(ValueError, match="Waymo records cannot"):
        _record(source="waymo", pg_seed=12)


def test_scenario_features_support_unknown_topology_and_signal_state() -> None:
    features = _features()

    assert features.topology_tag == "unknown"
    assert features.has_intersection is None
    assert features.has_unknown_signal is True
    assert ScenarioFeatures.from_dict(features.to_dict()) == features


def test_signal_not_applicable_is_not_unknown() -> None:
    assert _features(signal_reliability="not_applicable").has_unknown_signal is False
