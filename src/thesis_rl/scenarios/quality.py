from __future__ import annotations

from dataclasses import replace

from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


MINIMUM_SDC_ROUTE_LENGTH_M = 10.0
MINIMUM_SDC_VALID_RATIO = 0.8
MAXIMUM_WAYMO_ROUTE_Z_RANGE_M = 4.0
MAXIMUM_DYNAMIC_OBJECTS = 120


def catalog_quality_warnings(features: ScenarioFeatures) -> tuple[str, ...]:
    """Return catalog-level reasons that make a scenario unsuitable for RL use."""

    warnings: list[str] = []
    if features.route_length_m < MINIMUM_SDC_ROUTE_LENGTH_M:
        warnings.append(
            f"degenerate SDC route: {features.route_length_m:.3f} m < "
            f"{MINIMUM_SDC_ROUTE_LENGTH_M:.1f} m"
        )
    if not features.sdc_initial_valid:
        warnings.append("SDC track is invalid at t=0")
    if features.sdc_valid_ratio < MINIMUM_SDC_VALID_RATIO:
        warnings.append(
            f"low SDC valid ratio: {features.sdc_valid_ratio:.3f} < "
            f"{MINIMUM_SDC_VALID_RATIO:.1f}"
        )
    if (
        features.source == "waymo"
        and features.sdc_route_z_range_m > MAXIMUM_WAYMO_ROUTE_Z_RANGE_M
    ):
        warnings.append(
            f"possible overpass/elevation artifact: SDC z range "
            f"{features.sdc_route_z_range_m:.3f} m > "
            f"{MAXIMUM_WAYMO_ROUTE_Z_RANGE_M:.1f} m"
        )
    if features.map_feature_count <= 0:
        warnings.append("scenario has no map features")
    if features.dynamic_object_count > MAXIMUM_DYNAMIC_OBJECTS:
        warnings.append(
            f"too many dynamic objects: {features.dynamic_object_count} > "
            f"{MAXIMUM_DYNAMIC_OBJECTS}"
        )
    return tuple(warnings)


def apply_catalog_quality_policy(
    record: ScenarioRecord,
    features: ScenarioFeatures,
) -> ScenarioRecord:
    """Mark records invalid when offline quality metrics fail hard filters."""

    quality_warnings = catalog_quality_warnings(features)
    warnings = tuple(dict.fromkeys(record.validation_warnings + quality_warnings))
    status = (
        "invalid"
        if record.validation_status == "invalid" or quality_warnings
        else record.validation_status
    )
    return replace(record, validation_status=status, validation_warnings=warnings)


__all__ = [
    "MAXIMUM_DYNAMIC_OBJECTS",
    "MAXIMUM_WAYMO_ROUTE_Z_RANGE_M",
    "MINIMUM_SDC_ROUTE_LENGTH_M",
    "MINIMUM_SDC_VALID_RATIO",
    "apply_catalog_quality_policy",
    "catalog_quality_warnings",
]
