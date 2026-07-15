from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from math import inf

import pytest

from thesis_rl.rulebook.v2 import (
    RulebookEvaluationError,
    RulebookV2Config,
    TaskRouteRecord,
    load_rulebook_v2_config,
)
from thesis_rl.rulebook.v2.config import ExecutionConfig, RULEBOOK_V2_VERSION
from thesis_rl.rulebook.v2.context.task_route import build_task_route_record, validate_task_route, build_task_route_eligibility_index
from thesis_rl.rulebook.v2.context.map_matching import (
    OfflineTrackSample,
    map_match_sdc_track_to_task_route,
)
from thesis_rl.rulebook.v2.context.static_adapter import normalize_static_records
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from shapely.geometry import LineString, Polygon
from thesis_rl.rulebook.v2.errors import EvaluationFailure
from thesis_rl.rulebook.v2.registry import (
    DEFAULT_RULEBOOK_V2_REGISTRY,
    ComponentDefinition,
    RulebookV2Registry,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ComponentStatus,
    MacroRule,
    RuleComponentResult,
    RulebookResult,
    MapFeatureRecord,
    MapFeatureClass,
    TrafficControlRecord,
    ApproachControl,
    MovementKey,
)


def test_task_route_is_immutable_and_has_no_future_trajectory_fields() -> None:
    route = TaskRouteRecord("scenario-1", ("lane-a", "lane-b"), "waymo_offline", "v2", "abc")
    with pytest.raises(FrozenInstanceError):
        route.lane_ids = ()  # type: ignore[misc]
    assert "timestamp" not in route.__dataclass_fields__
    assert "future" not in " ".join(route.__dataclass_fields__)


def test_task_route_builder_and_eligibility_artifact_use_only_static_topology() -> None:
    route = build_task_route_record(
        scenario_uid="scenario-1",
        lane_ids=("lane-a", "lane-b"),
        provenance="waymo_offline_map_match",
        source_geometry_bytes=b"canonical-map",
    )
    eligible = validate_task_route(
        route,
        available_lane_ids={"lane-a": object(), "lane-b": object()},
        rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="geometry-hash",
        calibration_hash="calibration-hash",
    )
    assert eligible.rulebook_eligible
    assert eligible.validation_errors == ()
    assert "timestamp" not in eligible.__dataclass_fields__
    ineligible = validate_task_route(
        route,
        available_lane_ids={"lane-a": object()},
        rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="geometry-hash",
        calibration_hash="calibration-hash",
    )
    assert not ineligible.rulebook_eligible
    assert ineligible.validation_errors == ("missing_lane_ids:lane-b",)


def test_task_route_eligibility_index_is_deterministic_and_excludes_invalid_records():
    eligible_a = validate_task_route(
        build_task_route_record(scenario_uid="a", lane_ids=("lane",), provenance="pg", source_geometry_bytes=b"a"),
        available_lane_ids={"lane": object()}, rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="g", calibration_hash="c",
    )
    eligible_b = validate_task_route(
        build_task_route_record(scenario_uid="b", lane_ids=("lane",), provenance="pg", source_geometry_bytes=b"b"),
        available_lane_ids={}, rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="g", calibration_hash="c",
    )
    index = build_task_route_eligibility_index((eligible_b, eligible_a))
    assert tuple(index.by_scenario_uid) == ("a", "b")
    assert index.eligible("a").scenario_uid == "a"
    with pytest.raises(ValueError, match="not Rulebook v2 eligible"):
        index.eligible("b")


def test_task_route_validation_rejects_missing_identity_hashes():
    record = TaskRouteRecord("scenario", ("lane",), "pg", "adapter", "hash")
    result = validate_task_route(
        record, available_lane_ids={"lane": object()}, rulebook_version="",
        geometry_config_hash="", calibration_hash="",
    )
    assert not result.rulebook_eligible
    assert result.validation_errors == (
        "rulebook_version_missing", "geometry_config_hash_missing", "calibration_hash_missing",
    )


def test_offline_sdc_map_match_retains_lane_sequence_but_not_track_samples() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane-a", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route
    )
    record = map_match_sdc_track_to_task_route(
        scenario_uid="scenario-1",
        track=(
            OfflineTrackSample((1.0, 0.0), 0.0, 0.0),
            OfflineTrackSample((2.0, 0.0), 0.0, 0.0),
        ),
        route_lanes={"lane-a": lane},
        source_geometry_bytes=b"map",
        adapter_version="waymo-v1",
    )
    assert record.lane_ids == ("lane-a",)
    assert not hasattr(record, "track")


def test_static_adapter_normalizes_geometry_and_reports_missing_route_lanes() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    task_route = build_task_route_record(
        scenario_uid="scenario-1",
        lane_ids=("lane-a", "lane-missing"),
        provenance="pg_route",
        source_geometry_bytes=b"map",
    )
    result = normalize_static_records(
        scenario_uid="scenario-1",
        task_route=task_route,
        route_lanes=(
            RouteLaneRecord(
                "lane-a",
                Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))),
                route,
            ),
        ),
        map_features=(),
        traffic_controls=(),
    )
    assert result.validation_errors == ("task_route_lane_missing:lane-missing",)


def test_static_adapter_rejects_duplicate_controls_and_invalid_elevation():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    task_route = build_task_route_record(
        scenario_uid="scenario-1", lane_ids=("lane-a",), provenance="pg", source_geometry_bytes=b"map",
    )
    lane = RouteLaneRecord("lane-a", Polygon(((0, -2), (10, -2), (10, 2), (0, 2))), route)
    feature = MapFeatureRecord("feature", MapFeatureClass.ROAD_BOUNDARY, Polygon(((0, 0), (1, 0), (1, 1), (0, 1))), float("nan"))
    control = TrafficControlRecord("stop", ApproachControl.STOP, ("lane-a",), MovementKey("a", "n", "e"), LineString(((2, -2), (2, 2))), 2.0, 0.0, ())
    result = normalize_static_records(
        scenario_uid="scenario-1", task_route=task_route, route_lanes=(lane,),
        map_features=(feature,), traffic_controls=(control, control),
    )
    assert "invalid_map_feature_elevation:feature" in result.validation_errors
    assert "duplicate_control_group_id:stop" in result.validation_errors


def test_v2_config_rejects_nonconformant_execution_or_order() -> None:
    RulebookV2Config().validate()
    with pytest.raises(ValueError, match="fail-fast"):
        RulebookV2Config(execution=ExecutionConfig(silent_fallbacks=True)).validate()
    bad = RulebookV2Config(order=(MacroRule.ROUTE_PROGRESS,))
    with pytest.raises(ValueError, match="order"):
        bad.validate()
    with pytest.raises(ValueError, match="Unsupported"):
        load_rulebook_v2_config({"version": RULEBOOK_V2_VERSION, "unexpected": True})


def test_registry_rejects_memory_double_writer() -> None:
    duplicate = DEFAULT_RULEBOOK_V2_REGISTRY.components + (
        ComponentDefinition(
            "collision", MacroRule.COLLISION_IMPACT, None, frozenset({"previous_contact_ids"})
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        RulebookV2Registry(duplicate)


def test_registry_binds_all_normative_evaluators():
    normative = [component for component in DEFAULT_RULEBOOK_V2_REGISTRY.components if component.normative_output]
    assert all(component.evaluator is not None for component in normative)
    assert DEFAULT_RULEBOOK_V2_REGISTRY.components[10].evaluator is None
    with pytest.raises(ValueError, match="infrastructure"):
        DEFAULT_RULEBOOK_V2_REGISTRY.evaluate("zone_lifecycle")
    with pytest.raises(ValueError, match="Unknown"):
        DEFAULT_RULEBOOK_V2_REGISTRY.definition("missing")


def test_evaluation_error_keeps_typed_context() -> None:
    failure = EvaluationFailure("scenario-7", 12, "signal", "invalid state")
    error = RulebookEvaluationError(failure)
    assert error.failure == failure
    assert "scenario-7" in str(error)


def test_shapely_fixture_is_available_for_canonical_contracts() -> None:
    assert Polygon(((0, 0), (1, 0), (1, 1), (0, 0))).is_valid


def test_result_diagnostics_are_json_serializable() -> None:
    component = RuleComponentResult(
        "collision", 0.0, {"closing_speed_mps": 0.0}, True, True,
        ComponentStatus.SATISFIED, {"actors": ()},
    )
    result = RulebookResult(
        (0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 0.2, {"collision": component}, True
    )
    payload = result.to_dict()
    assert json.loads(json.dumps(payload))["components"]["collision"]["status"] == "satisfied"


def test_snapshot_and_result_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ActorSnapshot(
            "ego", ActorClass.VEHICLE, (inf, 0.0), 0.0, 0.0, (0.0, 0.0), Polygon(), None, 20.0
        )
    with pytest.raises(ValueError, match="finite"):
        RulebookResult((0.0, 0.0, 0.0, inf), (0.0, 0.0, 0.0), 0.0, {}, True)
