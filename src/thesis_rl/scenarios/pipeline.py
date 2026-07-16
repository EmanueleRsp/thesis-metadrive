"""Reusable orchestration helpers for the ScenarioNet dataset pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import ceil
from typing import Any, Mapping, Sequence

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


def eligible_entries(
    entries: Sequence[ScenarioCatalogEntry],
) -> tuple[ScenarioCatalogEntry, ...]:
    """Return records allowed to enter a frozen ScenarioNet dataset.

    The raw catalog remains an audit artifact and may contain rejected
    conversions. Splits, manifests, and runtime views instead share this one
    eligible population, so their counts cannot diverge.
    """

    return tuple(
        entry
        for entry in entries
        if entry.record.validation_status in {"valid", "warning"}
        and entry.record.rulebook_eligible is not False
    )


def group_id_for_entry(entry: ScenarioCatalogEntry) -> str:
    """Return the strongest available no-leakage grouping key."""

    record = entry.record
    if record.source == "waymo":
        source_log_id = str(record.source_log_id or "").strip()
        if source_log_id and not source_log_id.startswith("training_20s.tfrecord-"):
            return source_log_id
        return f"scenario:{record.scenario_uid}"
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
    entries = eligible_entries(entries)
    if not entries:
        raise ValueError("cannot assign splits: catalog has no eligible records")
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
        assigned_by_uid.update({record.scenario_uid: record for record in assigned_records})

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
    normalized_signal_policy = _normalize_signal_policy(allowed_signal_reliabilities)
    entries = eligible_entries(entries)
    if not entries:
        raise ValueError("cannot assign splits: catalog has no eligible records")
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
        arm_counts = {split: {arm: 0 for arm in ARMS} for split in SPLITS}
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
                        arm: sum(entry.record.primary_arm == arm for entry in grouped[group])
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


def assign_arm_balanced_splits_to_targets(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Select split x arm quotas with 50/50 source preference and fallback."""

    if set(targets) != set(SOURCES):
        raise ValueError(f"targets must define exactly {SOURCES}")
    normalized_signal_policy = _normalize_signal_policy(allowed_signal_reliabilities)
    entries = eligible_entries(entries)
    if not entries:
        raise ValueError("cannot assign splits: catalog has no eligible records")
    entries = tuple(
        entry
        for entry in entries
        if entry.features.signal_reliability in normalized_signal_policy[entry.record.source]
    )
    split_totals = _split_totals(targets)
    split_arm_targets = _split_arm_targets(split_totals, seed=seed)
    available_source_arm = _available_source_arm_counts(entries)
    split_source_arm_targets = _split_source_arm_targets(
        split_arm_targets,
        available_source_arm=available_source_arm,
    )
    all_group_ids = group_ids_for_entries(entries)
    grouped: dict[str, list[ScenarioCatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault(all_group_ids[entry.record.scenario_uid], []).append(entry)
    group_counts = {
        group: _group_source_arm_counts(group_entries) for group, group_entries in grouped.items()
    }

    rng = np.random.default_rng(int(seed))
    remaining = list(grouped)
    rng.shuffle(remaining)
    assignments: dict[str, str] = {}
    current_split = {split: 0 for split in SPLITS}
    current_source_split = {split: {source: 0 for source in SOURCES} for split in SPLITS}
    current_arm = {split: {arm: 0 for arm in ARMS} for split in SPLITS}
    current_source_arm = {
        split: {source: {arm: 0 for arm in ARMS} for source in SOURCES} for split in SPLITS
    }

    for split in SPLITS:
        while remaining and current_split[split] < split_totals[split]:
            capacity = split_totals[split] - current_split[split]
            fitting = [
                group
                for group in remaining
                if len(grouped[group]) <= capacity
                and all(
                    sum(group_counts[group][source][arm] for arm in ARMS)
                    <= int(targets[source][split]) - current_source_split[split][source]
                    for source in SOURCES
                )
                and all(
                    sum(group_counts[group][source][arm] for source in SOURCES)
                    <= split_arm_targets[split][arm] - current_arm[split][arm]
                    for arm in ARMS
                )
            ]
            if not fitting:
                break
            selected = max(
                fitting,
                key=lambda group: _arm_balanced_candidate_key(
                    group_counts[group],
                    split=split,
                    current_arm=current_arm,
                    current_source_arm=current_source_arm,
                    split_arm_targets=split_arm_targets,
                    split_source_arm_targets=split_source_arm_targets,
                    size=len(grouped[group]),
                ),
            )
            remaining.remove(selected)
            assignments[selected] = split
            current_split[split] += len(grouped[selected])
            for source in SOURCES:
                current_source_split[split][source] += sum(
                    group_counts[selected][source][arm] for arm in ARMS
                )
                for arm in ARMS:
                    count = group_counts[selected][source][arm]
                    current_arm[split][arm] += count
                    current_source_arm[split][source][arm] += count

    assigned_by_uid: dict[str, ScenarioRecord] = {}
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


def assert_runtime_split_contract(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    targets: Mapping[str, Mapping[str, int]],
    require_near_uniform_arms: bool,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
) -> None:
    """Validate the non-relaxable v1.1 runtime split constraints.

    Selection code may use a greedy heuristic because groups must remain
    indivisible.  That heuristic is never allowed to turn an infeasible pool
    into a silently smaller or scientifically different dataset: the caller
    must validate this contract before freezing a catalog or manifest.
    """

    if set(targets) != set(SOURCES):
        raise ValueError(f"targets must define exactly {SOURCES}")
    signal_policy = _normalize_signal_policy(allowed_signal_reliabilities)

    invalid_records = [
        entry.record.scenario_uid
        for entry in entries
        if entry.record.validation_status not in {"valid", "warning"}
    ]
    if invalid_records:
        raise ValueError(
            "runtime splits include records outside valid/warning status: "
            f"{sorted(invalid_records)[:5]}"
        )
    non_rulebook_records = [
        entry.record.scenario_uid for entry in entries if entry.record.rulebook_eligible is not True
    ]
    if non_rulebook_records:
        raise ValueError(
            "runtime splits require rulebook_eligible=True; offending records: "
            f"{sorted(non_rulebook_records)[:5]}"
        )
    disallowed_signal_records = [
        entry.record.scenario_uid
        for entry in entries
        if entry.features.signal_reliability not in signal_policy[entry.record.source]
    ]
    if disallowed_signal_records:
        raise ValueError(
            "runtime splits include records with disallowed signal reliability: "
            f"{sorted(disallowed_signal_records)[:5]}"
        )

    for source in SOURCES:
        if set(targets[source]) != set(SPLITS):
            raise ValueError(f"{source} targets must define exactly {SPLITS}")
        for split in SPLITS:
            target = int(targets[source][split])
            if target < 0:
                raise ValueError("split targets must be non-negative")
            actual = sum(
                entry.record.source == source and entry.record.split == split for entry in entries
            )
            if actual != target:
                raise ValueError(
                    f"runtime split target mismatch for {source}/{split}: "
                    f"requested {target}, selected {actual}"
                )

    if not require_near_uniform_arms:
        return
    expected_arm_counts = _split_arm_targets(_split_totals(targets), seed=seed)
    for split in SPLITS:
        arm_counts = {
            arm: sum(
                entry.record.split == split and entry.record.primary_arm == arm for entry in entries
            )
            for arm in ARMS
        }
        if arm_counts != expected_arm_counts[split]:
            raise ValueError(
                f"runtime split {split} does not match seed-derived near-uniform arm targets: "
                f"expected {expected_arm_counts[split]}, selected {arm_counts}"
            )


def _split_totals(
    targets: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    result = {split: sum(int(targets[source][split]) for source in SOURCES) for split in SPLITS}
    if any(value < 0 for value in result.values()):
        raise ValueError("split targets must be non-negative")
    return result


def _split_arm_targets(
    split_totals: Mapping[str, int], *, seed: int = 0
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_index, split in enumerate(SPLITS):
        total = int(split_totals[split])
        base, remainder = divmod(total, len(ARMS))
        ordered_arms = list(ARMS)
        np.random.default_rng([int(seed), split_index]).shuffle(ordered_arms)
        result[split] = {arm: base + (1 if arm in ordered_arms[:remainder] else 0) for arm in ARMS}
    return result


def _split_source_arm_targets(
    split_arm_targets: Mapping[str, Mapping[str, int]],
    *,
    available_source_arm: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    result = {split: {source: {arm: 0 for arm in ARMS} for source in SOURCES} for split in SPLITS}
    for arm in ARMS:
        total_target = sum(int(split_arm_targets[split][arm]) for split in SPLITS)
        ideal_pg = total_target // 2
        ideal_waymo = total_target - ideal_pg
        if available_source_arm is None:
            source_totals = {"pg": ideal_pg, "waymo": ideal_waymo}
        else:
            pg_available = int(available_source_arm["pg"][arm])
            waymo_available = int(available_source_arm["waymo"][arm])
            pg_total = min(ideal_pg, pg_available)
            waymo_total = min(ideal_waymo, waymo_available)
            pg_shortfall = ideal_pg - pg_total
            waymo_shortfall = ideal_waymo - waymo_total
            waymo_total += min(pg_shortfall, max(0, waymo_available - waymo_total))
            pg_total += min(waymo_shortfall, max(0, pg_available - pg_total))
            source_totals = {"pg": pg_total, "waymo": waymo_total}
        weights = {split: int(split_arm_targets[split][arm]) for split in SPLITS}
        for source in SOURCES:
            allocations = _distribute_quota(source_totals[source], weights)
            for split, value in allocations.items():
                result[split][source][arm] = value
    return result


def _available_source_arm_counts(
    entries: Sequence[ScenarioCatalogEntry],
) -> dict[str, dict[str, int]]:
    counts = {source: {arm: 0 for arm in ARMS} for source in SOURCES}
    for entry in entries:
        counts[entry.record.source][entry.record.primary_arm] += 1
    return counts


def _distribute_quota(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    if total <= 0:
        return {split: 0 for split in SPLITS}
    weight_total = sum(max(0, int(weights[split])) for split in SPLITS)
    if weight_total <= 0:
        return {split: 0 for split in SPLITS}
    raw = {split: total * max(0, int(weights[split])) / weight_total for split in SPLITS}
    result = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(result.values())
    for split in sorted(SPLITS, key=lambda name: raw[name] - result[name], reverse=True):
        if remainder <= 0:
            break
        result[split] += 1
        remainder -= 1
    return result


def _group_source_arm_counts(
    entries: Sequence[ScenarioCatalogEntry],
) -> dict[str, dict[str, int]]:
    counts = {source: {arm: 0 for arm in ARMS} for source in SOURCES}
    for entry in entries:
        counts[entry.record.source][entry.record.primary_arm] += 1
    return counts


def _arm_balanced_candidate_key(
    counts: Mapping[str, Mapping[str, int]],
    *,
    split: str,
    current_arm: Mapping[str, Mapping[str, int]],
    current_source_arm: Mapping[str, Mapping[str, Mapping[str, int]]],
    split_arm_targets: Mapping[str, Mapping[str, int]],
    split_source_arm_targets: Mapping[str, Mapping[str, Mapping[str, int]]],
    size: int,
) -> tuple[int, int, int, int, int]:
    weighted_benefit = 0.0
    arm_reduction = 0
    exact_source_reduction = 0
    arm_overfill = 0
    source_overfill = 0
    for source in SOURCES:
        for arm in ARMS:
            count = int(counts[source][arm])
            if count == 0:
                continue
            arm_deficit = max(
                0,
                int(split_arm_targets[split][arm]) - int(current_arm[split][arm]),
            )
            source_deficit = max(
                0,
                int(split_source_arm_targets[split][source][arm])
                - int(current_source_arm[split][source][arm]),
            )
            target = max(1, int(split_arm_targets[split][arm]))
            arm_weight = arm_deficit / target
            exact = min(count, source_deficit)
            exact_source_reduction += exact
            fallback = min(max(0, count - exact), max(0, arm_deficit - exact))
            arm_reduction += exact + fallback
            weighted_benefit += (exact * 1.2 + fallback) * arm_weight
            arm_overfill += max(
                0,
                int(current_arm[split][arm]) + count - int(split_arm_targets[split][arm]),
            )
            source_overfill += max(
                0,
                int(current_source_arm[split][source][arm])
                + count
                - int(split_source_arm_targets[split][source][arm]),
            )
    scaled_benefit = int(round(weighted_benefit * 1000))
    score_per_record = int(round(scaled_benefit / max(1, size)))
    score = score_per_record * 100 + scaled_benefit - arm_overfill * 25 - source_overfill * 5
    return (
        score,
        score_per_record,
        arm_reduction,
        exact_source_reduction,
        -arm_overfill,
    )


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
            raise ValueError(f"unknown signal reliabilities for {source}: {sorted(unknown_values)}")
        if not normalized:
            raise ValueError("allowed signal reliabilities must not be empty")
        result[source] = normalized
    return result


def _normalize_arm_minimums(
    arm_minimums: Mapping[str, Mapping[str, Mapping[str, int]]] | None,
) -> dict[str, dict[str, dict[str, int]]]:
    result = {source: {split: {arm: 0 for arm in ARMS} for split in SPLITS} for source in SOURCES}
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


def arm_source_balance_diagnostics(
    entries: Sequence[ScenarioCatalogEntry],
    targets: Mapping[str, Mapping[str, int]],
    *,
    available_entries: Sequence[ScenarioCatalogEntry] | None = None,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    split_totals = _split_totals(targets)
    split_arm_targets = _split_arm_targets(split_totals, seed=seed)
    available_source_arm = (
        _available_source_arm_counts(available_entries)
        if available_entries is not None
        else _available_source_arm_counts(entries)
    )
    split_source_arm_targets = _split_source_arm_targets(
        split_arm_targets,
        available_source_arm=available_source_arm,
    )
    diagnostics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_payload: dict[str, Any] = {
            "target_total": split_totals[split],
            "actual_total": sum(entry.record.split == split for entry in entries),
            "arms": {},
        }
        for arm in ARMS:
            source_payload: dict[str, Any] = {}
            actual_total = 0
            for source in SOURCES:
                actual = sum(
                    entry.record.split == split
                    and entry.record.primary_arm == arm
                    and entry.record.source == source
                    for entry in entries
                )
                target = split_source_arm_targets[split][source][arm]
                actual_total += actual
                source_payload[source] = {
                    "target": target,
                    "actual": actual,
                    "deficit": max(0, target - actual),
                }
            arm_target = split_arm_targets[split][arm]
            split_payload["arms"][arm] = {
                "target": arm_target,
                "actual": actual_total,
                "deficit": max(0, arm_target - actual_total),
                "sources": source_payload,
            }
        diagnostics[split] = split_payload
    return diagnostics


def classify_entries(
    entries: Sequence[ScenarioCatalogEntry], thresholds: ArmThresholds
) -> tuple[ScenarioCatalogEntry, ...]:
    return tuple(classify_catalog_entry(entry, thresholds) for entry in entries)


def balance_arm_distribution(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    target_total: int,
    seed: int,
    prefer_source: str = "waymo",
) -> tuple[tuple[ScenarioCatalogEntry, ...], dict[str, Any]]:
    """Trim over-represented arms while preserving real Waymo data first."""

    if target_total < 1:
        raise ValueError("target_total must be positive")
    if prefer_source not in SOURCES:
        raise ValueError(f"prefer_source must be one of {SOURCES}")
    target_per_arm = int(ceil(target_total / len(ARMS)))
    rng = np.random.default_rng(int(seed))
    kept: list[ScenarioCatalogEntry] = []
    removed: list[ScenarioCatalogEntry] = []
    source_order = {
        prefer_source: 0,
        **{source: 1 for source in SOURCES if source != prefer_source},
    }
    before_by_arm = Counter(entry.record.primary_arm for entry in entries)
    removed_by_arm_source: dict[str, dict[str, int]] = {
        arm: {source: 0 for source in SOURCES} for arm in ARMS
    }
    for arm in ARMS:
        arm_entries = [entry for entry in entries if entry.record.primary_arm == arm]
        indices = np.arange(len(arm_entries))
        rng.shuffle(indices)
        shuffled = [arm_entries[int(index)] for index in indices]
        ordered = sorted(
            shuffled,
            key=lambda entry: (
                source_order[entry.record.source],
                SPLITS.index(entry.record.split),
            ),
        )
        arm_kept = ordered[:target_per_arm]
        arm_removed = ordered[target_per_arm:]
        kept.extend(arm_kept)
        removed.extend(arm_removed)
        for entry in arm_removed:
            removed_by_arm_source[arm][entry.record.source] += 1

    kept_by_uid = {entry.record.scenario_uid: entry for entry in kept}
    balanced = tuple(
        kept_by_uid[entry.record.scenario_uid]
        for entry in entries
        if entry.record.scenario_uid in kept_by_uid
    )
    after_by_arm = Counter(entry.record.primary_arm for entry in balanced)
    diagnostics = {
        arm: {
            "target": target_per_arm,
            "before": int(before_by_arm[arm]),
            "after": int(after_by_arm[arm]),
            "removed": int(before_by_arm[arm] - after_by_arm[arm]),
            "deficit": max(0, target_per_arm - int(after_by_arm[arm])),
            "removed_by_source": removed_by_arm_source[arm],
        }
        for arm in ARMS
    }
    report = {
        "target_total": int(target_total),
        "target_per_arm": target_per_arm,
        "prefer_source": prefer_source,
        "input_records": len(entries),
        "selected_records": len(balanced),
        "removed_records": len(removed),
        "total_deficit": sum(values["deficit"] for values in diagnostics.values()),
        "diagnostics": diagnostics,
    }
    return balanced, report


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
        assigned_by_uid.update({record.scenario_uid: record for record in assigned_records})

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
    "assign_arm_balanced_splits_to_targets",
    "assert_runtime_split_contract",
    "arm_source_balance_diagnostics",
    "balance_arm_distribution",
    "classify_entries",
    "eligible_entries",
    "group_id_for_entry",
    "group_ids_for_entries",
]
