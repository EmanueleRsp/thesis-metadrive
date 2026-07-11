from __future__ import annotations

from dataclasses import replace

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.records import ScenarioFeatures
from thesis_rl.scenarios.thresholds import ArmThresholds, apply_traffic_thresholds


ARMS = (
    "A0_simple_lane_follow",
    "A1_vehicle_interaction",
    "A2_merge_or_roundabout",
    "A3_intersection",
    "A4_vru_interaction",
    "A5_complex_mixed",
)


def assign_primary_arm(features: ScenarioFeatures) -> str:
    merge_roundabout = features.has_merge_or_roundabout is True
    intersection = features.has_intersection is True
    vru_context = features.vru_interaction
    vehicle_interaction = features.relevant_vehicles_q90 > 0
    semantic_count = sum((merge_roundabout, intersection, vru_context))

    if semantic_count >= 2:
        return "A5_complex_mixed"
    if vru_context:
        return "A4_vru_interaction"
    if intersection:
        return "A3_intersection"
    if merge_roundabout:
        return "A2_merge_or_roundabout"
    if vehicle_interaction:
        return "A1_vehicle_interaction"
    return "A0_simple_lane_follow"


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
    if features.has_unknown_signal:
        tags.append("has_unknown_signal")
    if has_static_obstacle:
        tags.append("has_static_obstacle")
    return tuple(tags)


def classify_catalog_entry(
    entry: ScenarioCatalogEntry,
    thresholds: ArmThresholds,
    *,
    has_static_obstacle: bool = False,
) -> ScenarioCatalogEntry:
    thresholded = apply_traffic_thresholds(entry, thresholds)
    features = thresholded.features
    return ScenarioCatalogEntry(
        record=replace(
            thresholded.record,
            primary_arm=assign_primary_arm(features),
            tags=derive_scenario_tags(
                features, has_static_obstacle=has_static_obstacle
            ),
            signal_reliability=features.signal_reliability,
        ),
        features=features,
    )
