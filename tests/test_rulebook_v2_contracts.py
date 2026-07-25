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
from thesis_rl.rulebook.v2.context.task_route import (
    build_task_route_record,
    validate_task_route,
    build_task_route_eligibility_index,
    build_task_route_exclusion_report,
)
from thesis_rl.rulebook.v2.context.map_matching import (
    OfflineTrackSample,
    TaskRouteMapMatchError,
    map_match_sdc_track_to_task_route,
)
from thesis_rl.rulebook.v2.context.static_adapter import (
    normalize_static_records,
    validate_reset_contract,
)
from thesis_rl.rulebook.v2.context.static_sources import StaticRecordAdapter, StaticRecordSources
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, build_assigned_route_polyline
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
    MovementPriority,
    MovementPriorityRecord,
    RoundaboutPriorityRecord,
)


def test_map_feature_elevation_profile_interpolates_at_nearest_xy() -> None:
    feature = MapFeatureRecord(
        "boundary",
        MapFeatureClass.ROAD_BOUNDARY,
        LineString(((0.0, 0.0), (10.0, 0.0))),
        5.0,
        elevation_profile_xyz=((0.0, 0.0, 0.0), (10.0, 0.0, 10.0)),
    )

    assert feature.elevation_at_xy((2.5, 0.2)) == pytest.approx(2.5)


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
    assert eligible.assigned_route_lane_ids == ("lane-a", "lane-b")
    assert eligible.assigned_route_source == "waymo_sdc_offline_task_annotation"
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
        build_task_route_record(
            scenario_uid="a", lane_ids=("lane",), provenance="pg", source_geometry_bytes=b"a"
        ),
        available_lane_ids={"lane": object()},
        rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="g",
        calibration_hash="c",
    )
    eligible_b = validate_task_route(
        build_task_route_record(
            scenario_uid="b", lane_ids=("lane",), provenance="pg", source_geometry_bytes=b"b"
        ),
        available_lane_ids={},
        rulebook_version=RULEBOOK_V2_VERSION,
        geometry_config_hash="g",
        calibration_hash="c",
    )
    index = build_task_route_eligibility_index((eligible_b, eligible_a))
    assert tuple(index.by_scenario_uid) == ("a", "b")
    assert index.eligible_scenario_uids == frozenset({"a"})
    assert index.eligible("a").scenario_uid == "a"
    with pytest.raises(ValueError, match="not Rulebook v2 eligible"):
        index.eligible("b")
    report = build_task_route_exclusion_report(index)
    assert report.total_records == 2 and report.excluded_records == 1
    assert report.excluded_by_adapter == {"task-route-v1": 1}
    assert report.excluded_by_cause == {"missing_lane_ids:lane": 1}


def test_task_route_validation_rejects_missing_identity_hashes():
    record = TaskRouteRecord("scenario", ("lane",), "pg", "adapter", "hash")
    result = validate_task_route(
        record,
        available_lane_ids={"lane": object()},
        rulebook_version="",
        geometry_config_hash="",
        calibration_hash="",
    )
    assert not result.rulebook_eligible
    assert result.validation_errors == (
        "rulebook_version_missing",
        "geometry_config_hash_missing",
        "calibration_hash_missing",
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
    assert record.route_assignment_source == "offline_task_annotation"
    assert not hasattr(record, "track")


def test_assigned_route_polyline_uses_only_frozen_lane_ids_and_map_geometry() -> None:
    first = RouteLaneRecord(
        "lane-a",
        Polygon(((0.0, -2.0), (5.0, -2.0), (5.0, 2.0), (0.0, 2.0))),
        RoutePolyline(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )
    second = RouteLaneRecord(
        "lane-b",
        Polygon(((5.0, -2.0), (10.0, -2.0), (10.0, 2.0), (5.0, 2.0))),
        RoutePolyline(((5.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
    )
    route = build_assigned_route_polyline(("lane-a", "lane-b"), {"lane-a": first, "lane-b": second})
    assert route.points_xyz == ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0))


def test_assigned_route_polyline_fails_closed_for_missing_or_noncontiguous_lanes() -> None:
    lane_a = RouteLaneRecord(
        "lane-a",
        Polygon(((0.0, -2.0), (5.0, -2.0), (5.0, 2.0), (0.0, 2.0))),
        RoutePolyline(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )
    lane_c = RouteLaneRecord(
        "lane-c",
        Polygon(((8.0, -2.0), (13.0, -2.0), (13.0, 2.0), (8.0, 2.0))),
        RoutePolyline(((8.0, 0.0, 0.0), (13.0, 0.0, 0.0))),
    )
    with pytest.raises(ValueError, match="missing"):
        build_assigned_route_polyline(("lane-a", "lane-b"), {"lane-a": lane_a})
    with pytest.raises(ValueError, match="not contiguous"):
        build_assigned_route_polyline(("lane-a", "lane-c"), {"lane-a": lane_a, "lane-c": lane_c})


def test_offline_map_match_resolves_single_sample_on_canonical_lane_boundary() -> None:
    first = RouteLaneRecord(
        "lane-a",
        Polygon(((0.0, -2.0), (5.0, -2.0), (5.0, 2.0), (0.0, 2.0))),
        RoutePolyline(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )
    second = RouteLaneRecord(
        "lane-b",
        Polygon(((5.0, -2.0), (10.0, -2.0), (10.0, 2.0), (5.0, 2.0))),
        RoutePolyline(((5.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
    )
    record = map_match_sdc_track_to_task_route(
        scenario_uid="scenario-boundary",
        track=(
            OfflineTrackSample((4.0, 0.0), 0.0, 0.0),
            OfflineTrackSample((5.0, 0.0), 0.0, 0.0),
            OfflineTrackSample((6.0, 0.0), 0.0, 0.0),
        ),
        route_lanes={"lane-a": first, "lane-b": second},
        source_geometry_bytes=b"map",
        adapter_version="test-v1",
    )
    assert record.lane_ids == ("lane-a", "lane-b")


def test_offline_map_match_types_unavailable_lane_as_offline_exclusion() -> None:
    lane = RouteLaneRecord(
        "lane-a",
        Polygon(((0.0, -2.0), (5.0, -2.0), (5.0, 2.0), (0.0, 2.0))),
        RoutePolyline(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )
    with pytest.raises(TaskRouteMapMatchError) as caught:
        map_match_sdc_track_to_task_route(
            scenario_uid="scenario-missing",
            track=(OfflineTrackSample((100.0, 100.0), 0.0, 0.0),),
            route_lanes={"lane-a": lane},
            source_geometry_bytes=b"map",
            adapter_version="test-v1",
        )
    assert caught.value.validation_error == ("task_route_lane_association_ambiguous_or_unavailable")


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
        scenario_uid="scenario-1",
        lane_ids=("lane-a",),
        provenance="pg",
        source_geometry_bytes=b"map",
    )
    lane = RouteLaneRecord("lane-a", Polygon(((0, -2), (10, -2), (10, 2), (0, 2))), route)
    feature = MapFeatureRecord(
        "feature",
        MapFeatureClass.ROAD_BOUNDARY,
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
        float("nan"),
    )
    control = TrafficControlRecord(
        "stop",
        ApproachControl.STOP,
        ("lane-a",),
        MovementKey("a", "n", "e"),
        LineString(((2, -2), (2, 2))),
        2.0,
        0.0,
        (),
    )
    result = normalize_static_records(
        scenario_uid="scenario-1",
        task_route=task_route,
        route_lanes=(lane,),
        map_features=(feature,),
        traffic_controls=(control, control),
    )
    assert "invalid_map_feature_elevation:feature" in result.validation_errors
    assert "duplicate_control_group_id:stop" in result.validation_errors


def test_static_adapter_groups_physical_signal_heads_for_one_movement():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    task_route = build_task_route_record(
        scenario_uid="scenario-1",
        lane_ids=("lane-a",),
        provenance="pg",
        source_geometry_bytes=b"map",
    )
    lane = RouteLaneRecord("lane-a", Polygon(((0, -2), (10, -2), (10, 2), (0, 2))), route)
    movement = MovementKey("lane-a", "node", "lane-a")
    controls = tuple(
        TrafficControlRecord(
            physical_id,
            ApproachControl.SIGNAL,
            ("lane-a",),
            movement,
            LineString(((2, -2), (2, 2))),
            2.0,
            0.0,
            (physical_id,),
        )
        for physical_id in ("head-b", "head-a")
    )

    result = normalize_static_records(
        scenario_uid="scenario-1",
        task_route=task_route,
        route_lanes=(lane,),
        map_features=(),
        traffic_controls=controls,
    )

    assert result.validation_errors == ()
    assert len(result.traffic_controls) == 1
    assert result.traffic_controls[0].physical_control_ids == ("head-a", "head-b")


def test_static_adapter_rejects_vehicle_yield_metadata_for_missing_lanes():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    task_route = build_task_route_record(
        scenario_uid="scenario-1",
        lane_ids=("lane-a",),
        provenance="pg",
        source_geometry_bytes=b"map",
    )
    lane = RouteLaneRecord("lane-a", Polygon(((0, -2), (10, -2), (10, 2), (0, 2))), route)
    missing = MovementKey("lane-missing", "node", "lane-missing")
    result = normalize_static_records(
        scenario_uid="scenario-1",
        task_route=task_route,
        route_lanes=(lane,),
        map_features=(),
        traffic_controls=(),
        movement_priority_records=(
            MovementPriorityRecord(missing, missing, MovementPriority.OTHER_HAS_PRIORITY),
        ),
        roundabout_priority_records=(RoundaboutPriorityRecord("roundabout", "lane-a", "missing"),),
    )
    assert "movement_priority_lane_missing:lane-missing" in result.validation_errors
    assert "roundabout_priority_lane_missing:missing" in result.validation_errors


def test_static_record_sources_are_strict_and_normalize_source_neutrally():
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    task_route = build_task_route_record(
        scenario_uid="s", lane_ids=("lane",), provenance="pg", source_geometry_bytes=b"m"
    )
    lane = RouteLaneRecord("lane", Polygon(((0, -2), (10, -2), (10, 2), (0, 2))), route)
    sources = StaticRecordSources(
        lambda _record: task_route,
        lambda _record: (lane,),
        lambda _record: (),
        lambda _record: (),
    )
    result = StaticRecordAdapter(sources).normalize(object(), scenario_uid="s")
    assert result.validation_errors == ()
    with pytest.raises(ValueError, match="Missing"):
        StaticRecordSources.from_mapping({})


def test_reset_contract_rejects_caps_spawn_overlap_and_unknown_signal():
    from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box

    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (0, 0),
        0,
        0,
        (0, 0),
        oriented_bounding_box(center_xy=(0, 0), heading_rad=0, length_m=4, width_m=2),
        "lane",
        None,
    )
    other = ActorSnapshot(
        "other",
        ActorClass.VEHICLE,
        (0, 0),
        0,
        0,
        (0, 0),
        oriented_bounding_box(center_xy=(0, 0), heading_rad=0, length_m=2, width_m=1),
        "lane",
        None,
    )
    errors = validate_reset_contract(
        ego=ego, actors=(other,), signal_states_by_physical_id={"sig": "UNKNOWN"}
    )
    assert errors == (
        "ego_speed_cap_invalid",
        "vehicle_speed_cap_invalid:other",
        "spawn_overlap:other",
        "signal_state_unknown:sig",
    )


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
    normative = [
        component
        for component in DEFAULT_RULEBOOK_V2_REGISTRY.components
        if component.normative_output
    ]
    assert all(component.evaluator is not None for component in normative)
    assert DEFAULT_RULEBOOK_V2_REGISTRY.definition("zone_lifecycle").evaluator is None
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
        "collision",
        0.0,
        {"closing_speed_mps": 0.0},
        True,
        True,
        ComponentStatus.SATISFIED,
        {"actors": ()},
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
