from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry


def compute_feature_statistics(entries: Sequence[ScenarioCatalogEntry]) -> dict[str, Any]:
    if not entries:
        raise ValueError("feature statistics require at least one catalog entry")

    def numeric_summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        size = len(ordered)

        def quantile(fraction: float) -> float:
            if size == 1:
                return ordered[0]
            position = fraction * (size - 1)
            lower = int(position)
            upper = min(lower + 1, size - 1)
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        return {
            "min": ordered[0],
            "q50": quantile(0.50),
            "q90": quantile(0.90),
            "max": ordered[-1],
        }

    return {
        "total": len(entries),
        "topology": dict(
            sorted(Counter(entry.features.topology_tag for entry in entries).items())
        ),
        "signal_reliability": dict(
            sorted(Counter(entry.features.signal_reliability for entry in entries).items())
        ),
        "route_length_m": numeric_summary(
            [entry.features.route_length_m for entry in entries]
        ),
        "relevant_agents_q90": numeric_summary(
            [entry.features.relevant_agents_q90 for entry in entries]
        ),
        "relevant_vehicles_q90": numeric_summary(
            [entry.features.relevant_vehicles_q90 for entry in entries]
        ),
        "relevant_vrus_q90": numeric_summary(
            [entry.features.relevant_vrus_q90 for entry in entries]
        ),
        "sdc_valid_ratio": numeric_summary(
            [entry.features.sdc_valid_ratio for entry in entries]
        ),
        "sdc_route_z_range_m": numeric_summary(
            [entry.features.sdc_route_z_range_m for entry in entries]
        ),
        "dynamic_object_count": numeric_summary(
            [float(entry.features.dynamic_object_count) for entry in entries]
        ),
        "map_feature_count": numeric_summary(
            [float(entry.features.map_feature_count) for entry in entries]
        ),
        "invalid_records": sum(
            entry.record.validation_status == "invalid" for entry in entries
        ),
        "vehicle_conflict_count": numeric_summary(
            [float(entry.features.vehicle_conflict_count) for entry in entries]
        ),
        "vru_conflict_count": numeric_summary(
            [float(entry.features.vru_conflict_count) for entry in entries]
        ),
        "scenarios_with_vehicle_conflict": sum(
            entry.features.vehicle_conflict_count > 0 for entry in entries
        ),
        "scenarios_with_vru_conflict": sum(
            entry.features.vru_conflict_count > 0 for entry in entries
        ),
        "scenarios_with_vru_interaction": sum(
            entry.features.vru_interaction for entry in entries
        ),
    }


def compute_arm_distribution(entries: Sequence[ScenarioCatalogEntry]) -> dict[str, Any]:
    by_arm = Counter(entry.record.primary_arm for entry in entries)
    by_source: dict[str, Counter[str]] = {}
    for source in ("waymo", "pg"):
        by_source[source] = Counter(
            entry.record.primary_arm for entry in entries if entry.record.source == source
        )
    return {
        "total": len(entries),
        "by_arm": dict(sorted(by_arm.items())),
        "by_source": {
            source: dict(sorted(counts.items())) for source, counts in by_source.items()
        },
    }


def write_json_report(
    payload: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
