from __future__ import annotations

from dataclasses import replace

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.records import ScenarioFeatures
from thesis_rl.scenarios.thresholds import ArmThresholds, apply_traffic_thresholds


ARMS = (
    "A0_simple_low_traffic",
    "A1_traffic",
    "A2_junction",
    "A3_complex_junction",
    "A4_vru",
    "A5_critical_mixed",
)

A0_MAX_RELEVANT_VEHICLES_Q90 = 8.0
A1_JUNCTION_MAX_RELEVANT_AGENTS_Q90 = 8.0
A1_JUNCTION_MAX_VEHICLE_CONFLICT_COUNT = 1
ARM_COMPLEX_RELEVANT_AGENTS_Q90 = 25.0
ARM_COMPLEX_VEHICLE_CONFLICT_COUNT = 4
ARM_MIXED_TOPOLOGY_CONFLICT_COUNT = 3
ARM_CRITICAL_RELEVANT_AGENTS_Q90 = 30.0
ARM_CRITICAL_VEHICLE_CONFLICT_COUNT = 6


def assign_primary_arm(
    features: ScenarioFeatures, *, has_static_obstacle: bool = False
) -> str:
    topology = (
        features.has_merge_or_roundabout is True
        or features.has_intersection is True
    )
    mixed_topology = (
        features.has_merge_or_roundabout is True
        and features.has_intersection is True
    )
    vru_context = features.vru_interaction or features.vru_conflict_count > 0
    complex_traffic = (
        features.relevant_agents_q90 >= ARM_COMPLEX_RELEVANT_AGENTS_Q90
        or features.vehicle_conflict_count >= ARM_COMPLEX_VEHICLE_CONFLICT_COUNT
    )
    vehicle_conflicts = features.vehicle_conflict_count
    critical = topology and (
        features.vru_conflict_count > 0
        or (
            mixed_topology
            and vehicle_conflicts >= ARM_MIXED_TOPOLOGY_CONFLICT_COUNT
        )
        or (
            features.relevant_agents_q90 >= ARM_CRITICAL_RELEVANT_AGENTS_Q90
            and vehicle_conflicts >= ARM_CRITICAL_VEHICLE_CONFLICT_COUNT
        )
    )

    if critical:
        return "A5_critical_mixed"
    if vru_context:
        return "A4_vru"
    if topology and complex_traffic:
        return "A3_complex_junction"
    if topology:
        if (
            features.relevant_agents_q90
            <= A1_JUNCTION_MAX_RELEVANT_AGENTS_Q90
            and vehicle_conflicts <= A1_JUNCTION_MAX_VEHICLE_CONFLICT_COUNT
        ):
            return "A1_traffic"
        return "A2_junction"
    known_simple = (
        features.has_merge_or_roundabout is False
        and features.has_intersection is False
    )
    if (
        known_simple
        and features.relevant_vehicles_q90 <= A0_MAX_RELEVANT_VEHICLES_Q90
        and not has_static_obstacle
    ):
        return "A0_simple_low_traffic"
    return "A1_traffic"


def derive_scenario_tags(
    features: ScenarioFeatures,
    *,
    has_static_obstacle: bool = False,
) -> tuple[str, ...]:
    tags: list[str] = []
    if features.dense_traffic:
        tags.append("has_dense_traffic")
    if features.has_merge_or_roundabout is True:
        tags.append("has_merge_or_roundabout")
    if features.has_intersection is True:
        tags.append("has_intersection")
        if features.has_route_traffic_light is True:
            tags.append("has_signalized_intersection")
        elif features.has_route_traffic_light is False:
            tags.append("has_unsignalized_intersection")
    if features.has_route_traffic_light is True:
        tags.append("has_traffic_light")
    if features.has_route_stop_sign is True:
        tags.append("has_stop_sign")
    if features.has_route_crosswalk is True:
        tags.append("has_crosswalk")
    if features.vru_interaction:
        tags.append("has_vru")
    if features.vehicle_conflict_count > 0:
        tags.append("has_vehicle_conflict")
    if features.vru_conflict_count > 0:
        tags.append("has_vru_conflict")
    if features.has_unknown_signal:
        tags.append("has_unknown_signal")
    if has_static_obstacle:
        tags.append("has_static_obstacle")
    return tuple(tags)


def classify_catalog_entry(
    entry: ScenarioCatalogEntry,
    thresholds: ArmThresholds,
    *,
    has_static_obstacle: bool | None = None,
) -> ScenarioCatalogEntry:
    thresholded = apply_traffic_thresholds(entry, thresholds)
    features = thresholded.features
    resolved_has_static_obstacle = (
        "has_static_obstacle" in thresholded.record.tags
        if has_static_obstacle is None
        else has_static_obstacle
    )
    return ScenarioCatalogEntry(
        record=replace(
            thresholded.record,
            primary_arm=assign_primary_arm(
                features, has_static_obstacle=resolved_has_static_obstacle
            ),
            tags=derive_scenario_tags(
                features, has_static_obstacle=resolved_has_static_obstacle
            ),
            signal_reliability=features.signal_reliability,
        ),
        features=features,
    )
