from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from thesis_rl.scenarios.arms import ARMS
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
        "topology": dict(sorted(Counter(entry.features.topology_tag for entry in entries).items())),
        "signal_reliability": dict(
            sorted(Counter(entry.features.signal_reliability for entry in entries).items())
        ),
        "route_length_m": numeric_summary([entry.features.route_length_m for entry in entries]),
        "relevant_agents_q90": numeric_summary(
            [entry.features.relevant_agents_q90 for entry in entries]
        ),
        "relevant_vehicles_q90": numeric_summary(
            [entry.features.relevant_vehicles_q90 for entry in entries]
        ),
        "relevant_vrus_q90": numeric_summary(
            [entry.features.relevant_vrus_q90 for entry in entries]
        ),
        "sdc_valid_ratio": numeric_summary([entry.features.sdc_valid_ratio for entry in entries]),
        "sdc_route_z_range_m": numeric_summary(
            [entry.features.sdc_route_z_range_m for entry in entries]
        ),
        "dynamic_object_count": numeric_summary(
            [float(entry.features.dynamic_object_count) for entry in entries]
        ),
        "map_feature_count": numeric_summary(
            [float(entry.features.map_feature_count) for entry in entries]
        ),
        "invalid_records": sum(entry.record.validation_status == "invalid" for entry in entries),
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
        "scenarios_with_vru_interaction": sum(entry.features.vru_interaction for entry in entries),
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
        "by_source": {source: dict(sorted(counts.items())) for source, counts in by_source.items()},
    }


def compute_pg_replenishment_report(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    targets: dict[str, dict[str, int]],
    allowed_signal_reliabilities: Sequence[str],
    selected_entries: Sequence[ScenarioCatalogEntry] = (),
    selection_error: str | None = None,
) -> dict[str, Any]:
    """Report whether the current PG pool needs additional offline seeds.

    This report intentionally distinguishes a hard PG-count shortage from a
    failed joint Waymo/PG grouped split.  The latter must not trigger blind PG
    generation, because the limiting constraint can be Waymo coverage or an
    indivisible group rather than the procedural candidate pool.
    """

    pg_entries = [entry for entry in entries if entry.record.source == "pg"]
    valid_entries = [
        entry for entry in pg_entries if entry.record.validation_status in {"valid", "warning"}
    ]
    rulebook_entries = [entry for entry in valid_entries if entry.record.rulebook_eligible is True]
    allowed = frozenset(allowed_signal_reliabilities)
    runtime_entries = [
        entry for entry in rulebook_entries if entry.features.signal_reliability in allowed
    ]
    all_runtime_entries = [
        entry
        for entry in entries
        if entry.record.validation_status in {"valid", "warning"}
        and entry.record.rulebook_eligible is True
        and entry.features.signal_reliability in allowed
    ]
    selected_pg = [entry for entry in selected_entries if entry.record.source == "pg"]
    requested_by_split = {split: int(targets["pg"][split]) for split in targets["pg"]}
    selected_by_split = {
        split: sum(entry.record.split == split for entry in selected_pg)
        for split in requested_by_split
    }
    requested_total = sum(requested_by_split.values())
    runtime_total = len(runtime_entries)
    return {
        "source": "pg",
        "requested_by_split": requested_by_split,
        "requested_total": requested_total,
        "population_counts": {
            "candidate": len(pg_entries),
            "valid_or_warning": len(valid_entries),
            "rulebook_eligible": len(rulebook_entries),
            "runtime_eligible": runtime_total,
        },
        "runtime_eligible_by_profile": dict(
            sorted(
                Counter(entry.record.pg_profile or "unknown" for entry in runtime_entries).items()
            )
        ),
        "runtime_eligible_by_arm": dict(
            sorted(Counter(entry.record.primary_arm for entry in runtime_entries).items())
        ),
        "runtime_eligible_by_source_arm": {
            source: dict(
                sorted(
                    Counter(
                        entry.record.primary_arm
                        for entry in all_runtime_entries
                        if entry.record.source == source
                    ).items()
                )
            )
            for source in ("pg", "waymo")
        },
        "selection_completed": selection_error is None,
        "selected_by_split": selected_by_split,
        "selected_by_split_arm": {
            split: {
                arm: sum(
                    entry.record.split == split and entry.record.primary_arm == arm
                    for entry in selected_pg
                )
                for arm in ARMS
            }
            for split in requested_by_split
        },
        "hard_count_shortfall": max(0, requested_total - runtime_total),
        "hard_count_surplus": max(0, runtime_total - requested_total),
        "selection_shortfall_by_split": (
            {
                split: max(0, requested_by_split[split] - selected_by_split[split])
                for split in requested_by_split
            }
            if selection_error is None
            else None
        ),
        "selection_error": selection_error,
        "replenishment_action": (
            "generate_additional_pg_seeds"
            if runtime_total < requested_total
            else "no_pg_count_replenishment_required"
        ),
        "minimum_additional_runtime_eligible_records": max(0, requested_total - runtime_total),
        "note": (
            "The generation count is a lower bound on successful, Rulebook-eligible "
            "PG records; retain fixed profiles, labels, and thresholds."
        ),
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
