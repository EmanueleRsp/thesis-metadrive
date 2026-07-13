"""Reusable orchestration helpers for the ScenarioNet dataset pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from thesis_rl.scenarios.arms import ARMS, classify_catalog_entry
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.records import ScenarioRecord
from thesis_rl.scenarios.records import SIGNAL_RELIABILITIES
from thesis_rl.scenarios.runtime_database import assign_runtime_indices
from thesis_rl.scenarios.splits import (
    assert_no_group_overlap,
    assert_pg_seed_disjoint,
    assert_waymo_training_20s,
    assign_grouped_splits,
)
from thesis_rl.scenarios.thresholds import ArmThresholds

SPLITS = ("train", "validation", "test")
SOURCES = ("waymo", "pg")


def group_id_for_entry(entry: ScenarioCatalogEntry) -> str:
    """Return the strongest available no-leakage grouping key."""

    record = entry.record
    if record.source == "waymo":
        return str(record.source_log_id or f"scenario:{record.scenario_uid}")
    if record.pg_seed is None:
        raise ValueError(f"PG record has no seed: {record.scenario_uid}")
    return f"pg-seed:{record.pg_seed}"


def group_ids_for_entries(entries: Sequence[ScenarioCatalogEntry]) -> dict[str, str]:
    return {entry.record.scenario_uid: group_id_for_entry(entry) for entry in entries}


def assign_source_splits(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    counts: Mapping[str, Mapping[str, int]],
    seed: int,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Assign deterministic train/validation/test splits independently per source."""

    if set(counts) != set(SOURCES):
        raise ValueError(f"counts must define exactly {SOURCES}")
    all_group_ids = group_ids_for_entries(entries)
    assigned_by_uid: dict[str, ScenarioRecord] = {}
    for source in SOURCES:
        source_entries = [entry for entry in entries if entry.record.source == source]
        source_counts = {split: int(counts[source][split]) for split in SPLITS}
        if sum(source_counts.values()) != len(source_entries):
            raise ValueError(
                f"{source} split counts sum to {sum(source_counts.values())}, "
                f"but catalog contains {len(source_entries)} records"
            )
        source_groups = {
            entry.record.scenario_uid: all_group_ids[entry.record.scenario_uid]
            for entry in source_entries
        }
        assigned_records = assign_grouped_splits(
            [entry.record for entry in source_entries],
            group_id_by_uid=source_groups,
            counts=source_counts,
            seed=int(seed) + (0 if source == "waymo" else 1),
        )
        assigned_by_uid.update(
            {record.scenario_uid: record for record in assigned_records}
        )

    result = tuple(
        ScenarioCatalogEntry(
            record=assigned_by_uid[entry.record.scenario_uid],
            features=entry.features,
        )
        for entry in entries
    )
    records = [entry.record for entry in result]
    assert_waymo_training_20s(records)
    assert_pg_seed_disjoint(records)
    assert_no_group_overlap(records, group_id_by_uid=all_group_ids)
    return result


def assign_source_splits_to_targets(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
    arm_minimums: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Select whole groups toward source totals and optional arm minimums.

    This is used by the one-command pipeline when the converted pool is larger
    than the thesis target. Unselected converted records remain on disk but do
    not enter the final catalog/runtime views. The resulting counts are written
    to the split manifest.
    """

    if set(targets) != set(SOURCES):
        raise ValueError(f"targets must define exactly {SOURCES}")
    normalized_minimums = _normalize_arm_minimums(arm_minimums)
    normalized_signal_policy = _normalize_signal_policy(
        allowed_signal_reliabilities
    )
    all_group_ids = group_ids_for_entries(entries)
    assigned_by_uid: dict[str, ScenarioRecord] = {}
    for source_index, source in enumerate(SOURCES):
        source_entries = [
            entry
            for entry in entries
            if entry.record.source == source
            and entry.features.signal_reliability in normalized_signal_policy[source]
        ]
        source_targets = {split: int(targets[source][split]) for split in SPLITS}
        if any(value < 0 for value in source_targets.values()):
            raise ValueError("split targets must be non-negative")
        for split in SPLITS:
            minimum_total = sum(normalized_minimums[source][split].values())
            if minimum_total > source_targets[split]:
                raise ValueError(
                    f"{source} {split} arm minimums sum to {minimum_total}, "
                    f"above source target {source_targets[split]}"
                )
        grouped: dict[str, list[ScenarioCatalogEntry]] = {}
        for entry in source_entries:
            grouped.setdefault(all_group_ids[entry.record.scenario_uid], []).append(entry)
        rng = np.random.default_rng([int(seed), source_index])
        group_ids = list(grouped)
        rng.shuffle(group_ids)
        current = {split: 0 for split in SPLITS}
        arm_counts = {
            split: {arm: 0 for arm in ARMS}
            for split in SPLITS
        }
        assignments: dict[str, str] = {}
        remaining = list(group_ids)
        split_order = sorted(SPLITS, key=lambda name: source_targets[name])
        for split in split_order:
            target = source_targets[split]
            while remaining and current[split] < target:
                capacity = target - current[split]
                fitting = [group for group in remaining if len(grouped[group]) <= capacity]
                candidates = fitting or remaining
                minimums = normalized_minimums[source][split]

                def candidate_key(group: str) -> tuple[int, int, int, int]:
                    group_arm_counts = {
                        arm: sum(
                            entry.record.primary_arm == arm for entry in grouped[group]
                        )
                        for arm in ARMS
                    }
                    deficit_reduction = sum(
                        min(
                            group_arm_counts[arm],
                            max(0, minimums[arm] - arm_counts[split][arm]),
                        )
                        for arm in ARMS
                    )
                    size = len(grouped[group])
                    return (
                        deficit_reduction,
                        int(size <= capacity),
                        -abs(current[split] + size - target),
                        size,
                    )

                selected = max(
                    candidates,
                    key=candidate_key,
                )
                remaining.remove(selected)
                assignments[selected] = split
                current[split] += len(grouped[selected])
                for entry in grouped[selected]:
                    arm_counts[split][entry.record.primary_arm] += 1
        for group, group_entries in grouped.items():
            assigned_split = assignments.get(group)
            if assigned_split is None:
                continue
            for entry in group_entries:
                assigned_by_uid[entry.record.scenario_uid] = replace(
                    entry.record,
                    split=assigned_split,  # type: ignore[arg-type]
                    runtime_index=None,
                )

    result = tuple(
        ScenarioCatalogEntry(
            record=assigned_by_uid[entry.record.scenario_uid],
            features=entry.features,
        )
        for entry in entries
        if entry.record.scenario_uid in assigned_by_uid
    )
    records = [entry.record for entry in result]
    assert_waymo_training_20s(records)
    assert_pg_seed_disjoint(records)
    assert_no_group_overlap(records, group_id_by_uid=all_group_ids)
    return result


def _normalize_signal_policy(
    allowed: Mapping[str, Sequence[str]] | None,
) -> dict[str, frozenset[str]]:
    result = {source: SIGNAL_RELIABILITIES for source in SOURCES}
    if allowed is None:
        return result
    unknown_sources = set(allowed) - set(SOURCES)
    if unknown_sources:
        raise ValueError(f"unknown signal-policy sources: {sorted(unknown_sources)}")
    for source, values in allowed.items():
        normalized = frozenset(str(value) for value in values)
        unknown_values = normalized - SIGNAL_RELIABILITIES
        if unknown_values:
            raise ValueError(
                f"unknown signal reliabilities for {source}: {sorted(unknown_values)}"
            )
        if not normalized:
            raise ValueError("allowed signal reliabilities must not be empty")
        result[source] = normalized
    return result


def _normalize_arm_minimums(
    arm_minimums: Mapping[str, Mapping[str, Mapping[str, int]]] | None,
) -> dict[str, dict[str, dict[str, int]]]:
    result = {
        source: {split: {arm: 0 for arm in ARMS} for split in SPLITS}
        for source in SOURCES
    }
    if arm_minimums is None:
        return result
    unknown_sources = set(arm_minimums) - set(SOURCES)
    if unknown_sources:
        raise ValueError(f"unknown arm-minimum sources: {sorted(unknown_sources)}")
    for source, source_payload in arm_minimums.items():
        unknown_splits = set(source_payload) - set(SPLITS)
        if unknown_splits:
            raise ValueError(f"unknown arm-minimum splits: {sorted(unknown_splits)}")
        for split, split_payload in source_payload.items():
            unknown_arms = set(split_payload) - set(ARMS)
            if unknown_arms:
                raise ValueError(f"unknown arms in minimums: {sorted(unknown_arms)}")
            for arm, value in split_payload.items():
                minimum = int(value)
                if minimum < 0:
                    raise ValueError("arm minimums must be non-negative")
                result[source][split][arm] = minimum
    return result


def arm_selection_diagnostics(
    entries: Sequence[ScenarioCatalogEntry],
    arm_minimums: Mapping[str, Mapping[str, Mapping[str, int]]] | None,
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Report selected counts and deficits for each source/split/arm."""

    minimums = _normalize_arm_minimums(arm_minimums)
    diagnostics: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for source in SOURCES:
        diagnostics[source] = {}
        for split in SPLITS:
            diagnostics[source][split] = {}
            for arm in ARMS:
                actual = sum(
                    entry.record.source == source
                    and entry.record.split == split
                    and entry.record.primary_arm == arm
                    for entry in entries
                )
                minimum = minimums[source][split][arm]
                diagnostics[source][split][arm] = {
                    "minimum": minimum,
                    "actual": actual,
                    "deficit": max(0, minimum - actual),
                }
    return diagnostics


def classify_entries(
    entries: Sequence[ScenarioCatalogEntry], thresholds: ArmThresholds
) -> tuple[ScenarioCatalogEntry, ...]:
    return tuple(classify_catalog_entry(entry, thresholds) for entry in entries)


def assign_catalog_runtime_indices(
    entries: Sequence[ScenarioCatalogEntry],
) -> tuple[ScenarioCatalogEntry, ...]:
    """Assign per-split runtime indices to valid/warning records only."""

    assigned_by_uid: dict[str, ScenarioRecord] = {}
    for split in SPLITS:
        split_entries = [
            entry
            for entry in entries
            if entry.record.split == split
            and entry.record.validation_status in {"valid", "warning"}
        ]
        if not split_entries:
            continue
        assigned_records = assign_runtime_indices([entry.record for entry in split_entries])
        assigned_by_uid.update(
            {record.scenario_uid: record for record in assigned_records}
        )

    result: list[ScenarioCatalogEntry] = []
    for entry in entries:
        record = assigned_by_uid.get(entry.record.scenario_uid)
        if record is None:
            record = replace(entry.record, runtime_index=None)
        result.append(ScenarioCatalogEntry(record=record, features=entry.features))
    return tuple(result)


__all__ = [
    "SPLITS",
    "SOURCES",
    "assign_catalog_runtime_indices",
    "arm_selection_diagnostics",
    "assign_source_splits",
    "assign_source_splits_to_targets",
    "classify_entries",
    "group_id_for_entry",
    "group_ids_for_entries",
]
