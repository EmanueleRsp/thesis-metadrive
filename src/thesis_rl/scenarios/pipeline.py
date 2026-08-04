"""Reusable orchestration helpers for the ScenarioNet dataset pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import replace
from itertools import permutations
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
    assert_no_pool_overlap,
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
    """Return the strongest available no-leakage grouping key.

    Waymo preference order (`SCENARIONET-INTEGRATION` v1.2 §3.2): a genuine
    shared log/segment identifier first; failing that, the map-identity
    fingerprint (`record.map_id`, populated from `map_features` at
    conversion time, see `waymo.py::waymo_map_fingerprint`), which catches
    co-located 20 s windows converted from different TFRecord shards; only
    then the per-scenario fallback.
    """

    record = entry.record
    if record.source == "waymo":
        source_log_id = str(record.source_log_id or "").strip()
        if source_log_id and not source_log_id.startswith("training_20s.tfrecord-"):
            return source_log_id
        map_id = str(record.map_id or "").strip()
        if map_id:
            return map_id
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
    arm_capacity_deficits = {
        arm: {
            "requested": sum(split_arm_targets[split][arm] for split in SPLITS),
            "available": sum(available_source_arm[source][arm] for source in SOURCES),
            "available_by_source": {
                source: available_source_arm[source][arm] for source in SOURCES
            },
        }
        for arm in ARMS
        if sum(available_source_arm[source][arm] for source in SOURCES)
        < sum(split_arm_targets[split][arm] for split in SPLITS)
    }
    if arm_capacity_deficits:
        details = "; ".join(
            f"{arm}: requested={payload['requested']}, "
            f"available={payload['available']} "
            f"(waymo={payload['available_by_source']['waymo']}, "
            f"pg={payload['available_by_source']['pg']}), "
            f"deficit={payload['requested'] - payload['available']}"
            for arm, payload in sorted(arm_capacity_deficits.items())
        )
        raise ValueError(f"arm/source coverage is infeasible before split assignment: {details}")
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
    if len(grouped) <= 18:
        exact_assignments = _solve_small_group_assignment(
            grouped,
            group_counts=group_counts,
            targets=targets,
            split_arm_targets=split_arm_targets,
            seed=seed,
        )
        if exact_assignments is not None:
            return _materialize_group_assignments(
                entries,
                grouped=grouped,
                assignments=exact_assignments,
                all_group_ids=all_group_ids,
            )

    scalable_assignments = _solve_scalable_singleton_group_assignment(
        grouped,
        targets=targets,
        split_arm_targets=split_arm_targets,
        seed=seed,
    )
    if scalable_assignments is not None:
        return _materialize_group_assignments(
            entries,
            grouped=grouped,
            assignments=scalable_assignments,
            all_group_ids=all_group_ids,
        )

    assignments: dict[str, str] = {}
    # The first pass is conventional train/validation/test order. If it gets
    # trapped by an indivisible group, retry every deterministic split order;
    # no retry may relax a hard source or arm quota.
    split_orders = (SPLITS,) + tuple(order for order in permutations(SPLITS) if order != SPLITS)
    for attempt, split_order in enumerate(split_orders):
        rng = np.random.default_rng([int(seed), attempt])
        remaining = list(grouped)
        rng.shuffle(remaining)
        candidate_assignments: dict[str, str] = {}
        current_split = {split: 0 for split in SPLITS}
        current_source_split = {split: {source: 0 for source in SOURCES} for split in SPLITS}
        current_arm = {split: {arm: 0 for arm in ARMS} for split in SPLITS}
        current_source_arm = {
            split: {source: {arm: 0 for arm in ARMS} for source in SOURCES} for split in SPLITS
        }

        for split in split_order:
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
                candidate_assignments[selected] = split
                current_split[split] += len(grouped[selected])
                for source in SOURCES:
                    current_source_split[split][source] += sum(
                        group_counts[selected][source][arm] for arm in ARMS
                    )
                    for arm in ARMS:
                        count = group_counts[selected][source][arm]
                        current_arm[split][arm] += count
                        current_source_arm[split][source][arm] += count

        assignments = candidate_assignments
        if (
            current_split == split_totals
            and current_source_split
            == {
                split: {source: int(targets[source][split]) for source in SOURCES}
                for split in SPLITS
            }
            and current_arm == split_arm_targets
        ):
            break

    return _materialize_group_assignments(
        entries,
        grouped=grouped,
        assignments=assignments,
        all_group_ids=all_group_ids,
    )


def _materialize_group_assignments(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    grouped: Mapping[str, Sequence[ScenarioCatalogEntry]],
    assignments: Mapping[str, str],
    all_group_ids: Mapping[str, str],
) -> tuple[ScenarioCatalogEntry, ...]:
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


def _solve_small_group_assignment(
    grouped: Mapping[str, Sequence[ScenarioCatalogEntry]],
    *,
    group_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    targets: Mapping[str, Mapping[str, int]],
    split_arm_targets: Mapping[str, Mapping[str, int]],
    seed: int,
) -> dict[str, str] | None:
    """Solve small grouped fixtures exactly without adding an optimizer dependency."""

    group_ids = list(grouped)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group: len(grouped[group]), reverse=True)
    suffix_sizes = [0] * (len(group_ids) + 1)
    for index in range(len(group_ids) - 1, -1, -1):
        suffix_sizes[index] = suffix_sizes[index + 1] + len(grouped[group_ids[index]])
    remaining_source = {
        split: {source: int(targets[source][split]) for source in SOURCES} for split in SPLITS
    }
    remaining_arm = {
        split: {arm: int(split_arm_targets[split][arm]) for arm in ARMS} for split in SPLITS
    }
    required_total = sum(sum(values.values()) for values in targets.values())
    assignments: dict[str, str] = {}

    def fits(group: str, split: str) -> bool:
        return all(
            sum(group_counts[group][source][arm] for arm in ARMS) <= remaining_source[split][source]
            for source in SOURCES
        ) and all(
            sum(group_counts[group][source][arm] for source in SOURCES) <= remaining_arm[split][arm]
            for arm in ARMS
        )

    def apply(group: str, split: str, direction: int) -> None:
        for source in SOURCES:
            remaining_source[split][source] -= direction * sum(
                group_counts[group][source][arm] for arm in ARMS
            )
        for arm in ARMS:
            remaining_arm[split][arm] -= direction * sum(
                group_counts[group][source][arm] for source in SOURCES
            )

    def search(index: int, selected_total: int) -> bool:
        if index == len(group_ids):
            return selected_total == required_total and all(
                value == 0
                for split_values in remaining_source.values()
                for value in split_values.values()
            )
        required_remaining = required_total - selected_total
        if required_remaining < 0 or suffix_sizes[index] < required_remaining:
            return False
        group = group_ids[index]
        for split in SPLITS:
            if not fits(group, split):
                continue
            apply(group, split, 1)
            assignments[group] = split
            if search(index + 1, selected_total + len(grouped[group])):
                return True
            del assignments[group]
            apply(group, split, -1)
        if suffix_sizes[index + 1] >= required_remaining:
            return search(index + 1, selected_total)
        return False

    return dict(assignments) if search(0, 0) else None


def _solve_scalable_singleton_group_assignment(
    grouped: Mapping[str, Sequence[ScenarioCatalogEntry]],
    *,
    targets: Mapping[str, Mapping[str, int]],
    split_arm_targets: Mapping[str, Mapping[str, int]],
    seed: int,
) -> dict[str, str] | None:
    """Construct an exact assignment for large singleton-group pools.

    The checked-out converter has no verified shared Waymo log/segment field,
    so normal Waymo groups are singleton scenarios; PG generation seeds are
    singleton too.  Solving that common case as a transportation problem keeps
    exact source and arm targets practical for pools far larger than the
    dependency-free exhaustive solver's fixture-oriented threshold.  Groups
    with multiple records deliberately fall through to the strict generic
    path below: they must never be split to make a quota fit.
    """

    if any(len(entries) != 1 for entries in grouped.values()):
        return None

    groups_by_source_arm = {source: {arm: [] for arm in ARMS} for source in SOURCES}
    for group, entries in grouped.items():
        entry = entries[0]
        groups_by_source_arm[entry.record.source][entry.record.primary_arm].append(group)

    available = {
        source: {arm: len(groups_by_source_arm[source][arm]) for arm in ARMS} for source in SOURCES
    }
    source_totals = {
        source: sum(int(targets[source][split]) for split in SPLITS) for source in SOURCES
    }
    arm_totals = {arm: sum(int(split_arm_targets[split][arm]) for split in SPLITS) for arm in ARMS}
    total_requested = sum(source_totals.values())
    if total_requested != sum(arm_totals.values()):
        raise ValueError("source and arm target totals disagree")
    if any(source_totals[source] > sum(available[source].values()) for source in SOURCES):
        return None

    # Choose the total Waymo allocation for each arm.  The lower bound ensures
    # that the available PG records can fill the rest of that arm; the upper
    # bound preserves its exact target.  Assigning one unit at a time makes the
    # result deterministic and as close to the approved 50/50 preference as
    # availability permits.
    waymo_by_arm = {arm: max(0, arm_totals[arm] - available["pg"][arm]) for arm in ARMS}
    upper_waymo_by_arm = {arm: min(arm_totals[arm], available["waymo"][arm]) for arm in ARMS}
    if any(waymo_by_arm[arm] > upper_waymo_by_arm[arm] for arm in ARMS):
        return None
    remaining_waymo = source_totals["waymo"] - sum(waymo_by_arm.values())
    if remaining_waymo < 0 or remaining_waymo > sum(
        upper_waymo_by_arm[arm] - waymo_by_arm[arm] for arm in ARMS
    ):
        return None
    ordered_arms = list(ARMS)
    np.random.default_rng([int(seed), 31]).shuffle(ordered_arms)
    arm_order = {arm: index for index, arm in enumerate(ordered_arms)}
    ideal_waymo_by_arm = {
        arm: arm_totals[arm] * source_totals["waymo"] / max(1, total_requested) for arm in ARMS
    }
    for _ in range(remaining_waymo):
        candidates = [arm for arm in ARMS if waymo_by_arm[arm] < upper_waymo_by_arm[arm]]
        if not candidates:
            return None
        arm = max(
            candidates,
            key=lambda name: (ideal_waymo_by_arm[name] - waymo_by_arm[name], -arm_order[name]),
        )
        waymo_by_arm[arm] += 1

    waymo_allocations = _transport_source_arm_quota(
        arm_totals=waymo_by_arm,
        split_arm_targets=split_arm_targets,
        split_source_targets={split: int(targets["waymo"][split]) for split in SPLITS},
        seed=seed,
    )
    if waymo_allocations is None:
        return None
    allocations = {
        "waymo": waymo_allocations,
        "pg": {
            split: {
                arm: int(split_arm_targets[split][arm]) - waymo_allocations[split][arm]
                for arm in ARMS
            }
            for split in SPLITS
        },
    }
    if any(
        sum(allocations[source][split][arm] for split in SPLITS) > available[source][arm]
        for source in SOURCES
        for arm in ARMS
    ):
        return None

    assignments: dict[str, str] = {}
    for source_index, source in enumerate(SOURCES):
        for arm_index, arm in enumerate(ARMS):
            # A population-size-independent key (see `_stable_permutation_key`):
            # sorting by this key, rather than shuffling a Python list with
            # `numpy`'s `Generator.shuffle`, keeps every existing candidate's
            # relative order fixed when upstream filtering (for example
            # driving-mission eligibility) removes other candidates from this
            # cell, so only the removed candidates -- not the whole cell --
            # change which split they land in.
            group_ids = sorted(
                groups_by_source_arm[source][arm],
                key=lambda name: _stable_permutation_key(
                    (int(seed), source_index, arm_index), name
                ),
            )
            offset = 0
            for split in SPLITS:
                count = allocations[source][split][arm]
                for group in group_ids[offset : offset + count]:
                    assignments[group] = split
                offset += count
    return assignments


def _transport_source_arm_quota(
    *,
    arm_totals: Mapping[str, int],
    split_arm_targets: Mapping[str, Mapping[str, int]],
    split_source_targets: Mapping[str, int],
    seed: int,
) -> dict[str, dict[str, int]] | None:
    """Allocate one source quota with exact totals and minimum split drift.

    The old largest-capacity-first transport was feasible but could concentrate
    an arm/source cell in a single split.  This min-cost flow retains exact
    arm and split quotas while minimizing the sum of absolute deviations from
    the arm's target split proportions.  Unit-cost edges encode the convex
    absolute-deviation objective without adding an optimizer dependency.
    """

    del seed  # Stable graph order is the deterministic tie-breaker.
    arm_total_sum = sum(int(arm_totals[arm]) for arm in ARMS)
    split_total_sum = sum(int(split_source_targets[split]) for split in SPLITS)
    if arm_total_sum != split_total_sum:
        return None

    source_node = 0
    arm_nodes = {arm: index + 1 for index, arm in enumerate(ARMS)}
    split_nodes = {split: len(ARMS) + index + 1 for index, split in enumerate(SPLITS)}
    sink_node = len(ARMS) + len(SPLITS) + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink_node + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: int) -> None:
        forward = [end, len(graph[end]), capacity, cost]
        reverse = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)

    for arm in ARMS:
        add_edge(source_node, arm_nodes[arm], int(arm_totals[arm]), 0)
    for split in SPLITS:
        add_edge(split_nodes[split], sink_node, int(split_source_targets[split]), 0)

    allocation_edges: dict[tuple[str, str], list[list[int]]] = {}
    for arm in ARMS:
        arm_total = int(arm_totals[arm])
        if arm_total < 0:
            return None
        arm_split_capacity = sum(int(split_arm_targets[split][arm]) for split in SPLITS)
        if arm_total > arm_split_capacity:
            return None
        for split in SPLITS:
            capacity = int(split_arm_targets[split][arm])
            if capacity < 0:
                return None
            target_numerator = arm_total * capacity
            edge_units: list[list[int]] = []
            for selected_count in range(1, capacity + 1):
                marginal_cost = abs(selected_count * arm_split_capacity - target_numerator) - abs(
                    (selected_count - 1) * arm_split_capacity - target_numerator
                )
                add_edge(arm_nodes[arm], split_nodes[split], 1, marginal_cost)
                edge_units.append(graph[arm_nodes[arm]][-1])
            allocation_edges[(split, arm)] = edge_units

    flow = 0
    while flow < arm_total_sum:
        distance: list[int | None] = [None] * len(graph)
        predecessor: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source_node] = 0
        queue = deque([source_node])
        queued = {source_node}
        while queue:
            start = queue.popleft()
            queued.remove(start)
            assert distance[start] is not None
            for edge_index, edge in enumerate(graph[start]):
                end, _, capacity, cost = edge
                candidate_distance = distance[start] + cost
                if capacity <= 0 or (
                    distance[end] is not None and candidate_distance >= distance[end]
                ):
                    continue
                distance[end] = candidate_distance
                predecessor[end] = (start, edge_index)
                if end not in queued:
                    queue.append(end)
                    queued.add(end)
        if predecessor[sink_node] is None:
            return None

        end = sink_node
        while end != source_node:
            start, edge_index = predecessor[end]  # type: ignore[misc]
            edge = graph[start][edge_index]
            edge[2] -= 1
            graph[end][edge[1]][2] += 1
            end = start
        flow += 1

    return {
        split: {arm: sum(edge[2] == 0 for edge in allocation_edges[(split, arm)]) for arm in ARMS}
        for split in SPLITS
    }


def _stable_permutation_key(seed: object, name: str) -> str:
    """Deterministic pseudo-random sort key, independent of population size.

    Unlike `numpy.random.Generator.shuffle` over a Python list -- whose
    resulting permutation depends on the *length* of the list being shuffled,
    so appending new groups silently reorders the existing ones -- this key
    depends only on `(seed, name)`. Appending a new group therefore only adds
    a new key; it never changes the relative order of existing groups. This
    is required by `SCENARIONET-INTEGRATION` v1.2 §6.5/`REQ-005`: a frozen
    empirical holdout must be provably unaffected by scenarios acquired
    afterward (`TEST-007`).
    """

    payload = f"{seed}:{name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reserve_empirical_holdouts(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    test_counts: Mapping[str, int],
    validation_counts: Mapping[str, int],
    seed: int,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[ScenarioCatalogEntry, ...], tuple[ScenarioCatalogEntry, ...]]:
    """Reserve label-blind empirical test/validation holdouts (SS3.5).

    `SCENARIONET-INTEGRATION` v1.2 §3.5/§6.5: test is reserved before
    validation, from a deterministic pseudo-random permutation of whole
    groups, independently per source. Neither reservation ever reads
    `primary_arm`, `tags`, `low_traffic`, or `dense_traffic` -- the walk order
    depends only on the seeded group permutation and group size, so
    relabelling the catalog's arm assignments cannot change which groups are
    reserved (`REQ-001`, `TEST-001`).

    Returns `(reserved, residual)`. `reserved` entries carry `split` set to
    `"test"`/`"validation"` and `holdout_pool="empirical"`; `residual` entries
    are unchanged and remain available for the stratified test pool and the
    training pool. Raises if either source cannot reach its exact counts
    without splitting a group.
    """

    if set(test_counts) != set(SOURCES) or set(validation_counts) != set(SOURCES):
        raise ValueError(f"counts must define exactly {SOURCES}")
    normalized_signal_policy = _normalize_signal_policy(allowed_signal_reliabilities)
    entries = eligible_entries(entries)
    entries = tuple(
        entry
        for entry in entries
        if entry.features.signal_reliability in normalized_signal_policy[entry.record.source]
    )
    all_group_ids = group_ids_for_entries(entries)

    reserved: list[ScenarioCatalogEntry] = []
    residual: list[ScenarioCatalogEntry] = []
    for source_index, source in enumerate(SOURCES):
        needed = {
            "test": int(test_counts[source]),
            "validation": int(validation_counts[source]),
        }
        if any(value < 0 for value in needed.values()):
            raise ValueError("empirical holdout counts must be non-negative")

        source_entries = [entry for entry in entries if entry.record.source == source]
        grouped: dict[str, list[ScenarioCatalogEntry]] = {}
        for entry in source_entries:
            grouped.setdefault(all_group_ids[entry.record.scenario_uid], []).append(entry)
        # A population-size-independent key (see `_stable_permutation_key`):
        # ordering by this key, rather than shuffling a Python list with
        # `numpy`'s `Generator.shuffle`, keeps every existing group's
        # position fixed when new groups are appended later.
        group_ids = sorted(
            grouped, key=lambda name: _stable_permutation_key((int(seed), source_index), name)
        )

        remaining = dict(needed)
        pool_order: tuple[str, ...] = ("test", "validation")
        assignments: dict[str, str] = {}
        for group_id in group_ids:
            size = len(grouped[group_id])
            for pool in pool_order:
                if remaining[pool] <= 0:
                    continue
                if size <= remaining[pool]:
                    assignments[group_id] = pool
                    remaining[pool] -= size
                    break
        if any(value != 0 for value in remaining.values()):
            raise ValueError(
                f"cannot reserve exact empirical holdout counts for {source}: "
                f"requested test={needed['test']}, validation={needed['validation']}, "
                f"unmet={remaining} out of {len(source_entries)} eligible records "
                f"in {len(grouped)} groups"
            )

        for group_id, group_entries in grouped.items():
            pool = assignments.get(group_id)
            if pool is None:
                residual.extend(group_entries)
                continue
            for entry in group_entries:
                reserved.append(
                    ScenarioCatalogEntry(
                        record=replace(
                            entry.record,
                            split=pool,  # type: ignore[arg-type]
                            holdout_pool="empirical",
                            runtime_index=None,
                        ),
                        features=entry.features,
                    )
                )

    reserved_records = [entry.record for entry in reserved]
    assert_waymo_training_20s(reserved_records)
    assert_pg_seed_disjoint(reserved_records)
    assert_no_pool_overlap(reserved_records, group_id_by_uid=all_group_ids)
    return tuple(reserved), tuple(residual)


def apply_frozen_empirical_holdouts(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    frozen_test_uids: Sequence[str],
    frozen_validation_uids: Sequence[str],
) -> tuple[tuple[ScenarioCatalogEntry, ...], tuple[ScenarioCatalogEntry, ...]]:
    """Reuse an already-frozen empirical holdout instead of recomputing it.

    `reserve_empirical_holdouts` is exact-count reservation from a
    deterministic permutation: appending new eligible scenarios to the pool
    it walks can shift *which* groups fill an exact count (a new group that
    sorts earlier can displace one that previously fit), so re-running it
    against a larger pool is not guaranteed to reproduce a prior selection
    unit-for-unit. `SCENARIONET-INTEGRATION` v1.2 §6.5's "immutable once
    frozen" (`REQ-005`) is therefore an operational guarantee, not a pure
    mathematical property of the allocator: once a holdout is frozen (its UID
    list persisted to the split manifest), every later pipeline run must call
    *this* function -- a pure membership partition against the recorded UIDs,
    with no randomness and no recomputation -- instead of calling
    `reserve_empirical_holdouts` again. Any scenario not in either frozen UID
    list falls to the residual, available only to the stratified pool and
    training (`TEST-007`).
    """

    test_uids = frozenset(frozen_test_uids)
    validation_uids = frozenset(frozen_validation_uids)
    overlap = test_uids & validation_uids
    if overlap:
        raise ValueError(f"frozen test/validation UID sets overlap: {sorted(overlap)[:5]}")

    reserved: list[ScenarioCatalogEntry] = []
    residual: list[ScenarioCatalogEntry] = []
    seen_test: set[str] = set()
    seen_validation: set[str] = set()
    for entry in entries:
        uid = entry.record.scenario_uid
        if uid in test_uids:
            seen_test.add(uid)
            reserved.append(
                ScenarioCatalogEntry(
                    record=replace(
                        entry.record, split="test", holdout_pool="empirical", runtime_index=None
                    ),
                    features=entry.features,
                )
            )
        elif uid in validation_uids:
            seen_validation.add(uid)
            reserved.append(
                ScenarioCatalogEntry(
                    record=replace(
                        entry.record,
                        split="validation",
                        holdout_pool="empirical",
                        runtime_index=None,
                    ),
                    features=entry.features,
                )
            )
        else:
            residual.append(entry)

    missing_test = test_uids - seen_test
    missing_validation = validation_uids - seen_validation
    if missing_test or missing_validation:
        raise ValueError(
            "frozen empirical holdout UIDs missing from the current catalog: "
            f"test={sorted(missing_test)[:5]}, validation={sorted(missing_validation)[:5]}"
        )
    return tuple(reserved), tuple(residual)


def reserve_stratified_pool(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    total: int,
    seed: int,
    protected_train_arm_minimums: Mapping[str, Mapping[str, int]] | None = None,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[ScenarioCatalogEntry, ...], tuple[ScenarioCatalogEntry, ...]]:
    """Select an arm-stratified, source-balanced test pool from the residual.

    Reuses the v1.1 `balanced_arm_source` policy (§6.6, unchanged) restricted
    to a single pool instead of three simultaneous splits: near-uniform
    per-arm target (max count difference 1) with a deterministically shuffled
    remainder, best-effort 50/50 Waymo/PG within each arm compensated by
    availability and exact group/eligibility constraints
    (`SCENARIONET-INTEGRATION` v1.2 §3.1/§3.5).

    ``protected_train_arm_minimums`` reserves scarce source/arm capacity for
    the subsequent train selection. It does not change the total per-arm test
    target, but can make the stratified source mix less balanced for the
    protected arm.

    A group whose members span more than one arm can never be fully consumed
    by a single per-(source, arm) cell and is therefore never selected here
    -- it remains in the residual, available to the training pool. This is a
    conservative, group-integrity-preserving simplification: it may leave a
    mixed-arm group out of the stratified pool even when it could otherwise
    contribute, but it never violates group indivisibility or an arm target.

    Returns `(selected, residual)`; `selected` entries carry `split="test"`
    and `holdout_pool="stratified"`. Raises loudly on infeasible per-arm
    capacity rather than silently relaxing a quota.
    """

    normalized_signal_policy = _normalize_signal_policy(allowed_signal_reliabilities)
    entries = eligible_entries(entries)
    entries = tuple(
        entry
        for entry in entries
        if entry.features.signal_reliability in normalized_signal_policy[entry.record.source]
    )
    if total < 0:
        raise ValueError("stratified pool total must be non-negative")

    base, remainder = divmod(int(total), len(ARMS))
    ordered_arms = list(ARMS)
    np.random.default_rng([int(seed), 0]).shuffle(ordered_arms)
    arm_targets = {arm: base + (1 if arm in ordered_arms[:remainder] else 0) for arm in ARMS}

    available_source_arm = _available_source_arm_counts(entries)
    protected_source_arm = {source: {arm: 0 for arm in ARMS} for source in SOURCES}
    if protected_train_arm_minimums is not None:
        unknown_sources = set(protected_train_arm_minimums) - set(SOURCES)
        if unknown_sources:
            raise ValueError(f"unknown protected-train sources: {sorted(unknown_sources)}")
        for source, arm_minimums in protected_train_arm_minimums.items():
            unknown_arms = set(arm_minimums) - set(ARMS)
            if unknown_arms:
                raise ValueError(f"unknown protected-train arms: {sorted(unknown_arms)}")
            for arm, value in arm_minimums.items():
                minimum = int(value)
                if minimum < 0:
                    raise ValueError("protected train arm minimums must be non-negative")
                if minimum > available_source_arm[source][arm]:
                    raise ValueError(
                        "protected train arm minimum exceeds residual capacity: "
                        f"{source}/{arm} requested={minimum}, "
                        f"available={available_source_arm[source][arm]}"
                    )
                protected_source_arm[source][arm] = minimum
    selectable_source_arm = {
        source: {
            arm: available_source_arm[source][arm] - protected_source_arm[source][arm]
            for arm in ARMS
        }
        for source in SOURCES
    }
    arm_capacity_deficits = {
        arm: {
            "requested": arm_targets[arm],
            "available": sum(selectable_source_arm[source][arm] for source in SOURCES),
            "available_by_source": {
                source: selectable_source_arm[source][arm] for source in SOURCES
            },
        }
        for arm in ARMS
        if sum(selectable_source_arm[source][arm] for source in SOURCES) < arm_targets[arm]
    }
    if arm_capacity_deficits:
        details = "; ".join(
            f"{arm}: requested={payload['requested']}, available={payload['available']} "
            f"(waymo={payload['available_by_source']['waymo']}, "
            f"pg={payload['available_by_source']['pg']}), "
            f"deficit={payload['requested'] - payload['available']}"
            for arm, payload in sorted(arm_capacity_deficits.items())
        )
        raise ValueError(f"stratified pool arm coverage is infeasible: {details}")

    source_arm_targets = {source: {arm: 0 for arm in ARMS} for source in SOURCES}
    for arm in ARMS:
        ideal_pg = arm_targets[arm] // 2
        ideal_waymo = arm_targets[arm] - ideal_pg
        pg_available = selectable_source_arm["pg"][arm]
        waymo_available = selectable_source_arm["waymo"][arm]
        pg_total = min(ideal_pg, pg_available)
        waymo_total = min(ideal_waymo, waymo_available)
        pg_shortfall = ideal_pg - pg_total
        waymo_shortfall = ideal_waymo - waymo_total
        waymo_total += min(pg_shortfall, max(0, waymo_available - waymo_total))
        pg_total += min(waymo_shortfall, max(0, pg_available - pg_total))
        source_arm_targets["pg"][arm] = pg_total
        source_arm_targets["waymo"][arm] = waymo_total

    all_group_ids = group_ids_for_entries(entries)
    grouped: dict[str, list[ScenarioCatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault(all_group_ids[entry.record.scenario_uid], []).append(entry)
    group_source_arm = {
        group_id: _group_source_arm_counts(group_entries)
        for group_id, group_entries in grouped.items()
    }

    selected_groups: set[str] = set()
    for source_index, source in enumerate(SOURCES):
        for arm_index, arm in enumerate(ARMS):
            target = source_arm_targets[source][arm]
            if target <= 0:
                continue
            homogeneous_candidates = [
                group_id
                for group_id, counts in group_source_arm.items()
                if group_id not in selected_groups
                and counts[source][arm] == len(grouped[group_id])
                and sum(
                    counts[other][a]
                    for other in SOURCES
                    for a in ARMS
                    if (other, a) != (source, arm)
                )
                == 0
            ]
            homogeneous_candidates.sort(
                key=lambda name: _stable_permutation_key((int(seed), source_index, arm_index), name)
            )
            filled = 0
            for group_id in homogeneous_candidates:
                if filled >= target:
                    break
                size = len(grouped[group_id])
                if filled + size > target:
                    continue
                selected_groups.add(group_id)
                filled += size
            if filled != target:
                raise ValueError(
                    f"cannot reach exact stratified target for {source}/{arm}: "
                    f"requested {target}, filled {filled} "
                    f"(some capacity may be trapped in mixed-arm groups)"
                )

    selected: list[ScenarioCatalogEntry] = []
    residual: list[ScenarioCatalogEntry] = []
    for group_id, group_entries in grouped.items():
        if group_id in selected_groups:
            for entry in group_entries:
                selected.append(
                    ScenarioCatalogEntry(
                        record=replace(
                            entry.record,
                            split="test",
                            holdout_pool="stratified",
                            runtime_index=None,
                        ),
                        features=entry.features,
                    )
                )
        else:
            residual.extend(group_entries)

    selected_records = [entry.record for entry in selected]
    assert_waymo_training_20s(selected_records)
    assert_pg_seed_disjoint(selected_records)
    assert_no_pool_overlap(selected_records, group_id_by_uid=all_group_ids)
    return tuple(selected), tuple(residual)


def assign_training_pool_from_residual(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    arm_minimums: Mapping[str, Mapping[str, int]] | None = None,
    source_counts: Mapping[str, int] | None = None,
    seed: int = 0,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Build the training pool from the residual and verify its arm minimums.

    Without ``source_counts``, every residual eligible entry enters training.
    With source counts, select an exact, near-uniform-by-arm train pool using
    the v1.1 ``balanced_arm_source`` allocator.  The allocator keeps the
    requested per-source train totals exact while compensating structural
    empty source/arm cells (notably ``A4_vru`` for PG) in the other cells.
    """

    normalized_minimums = {source: {arm: 0 for arm in ARMS} for source in SOURCES}
    if arm_minimums is not None:
        unknown_sources = set(arm_minimums) - set(SOURCES)
        if unknown_sources:
            raise ValueError(f"unknown arm-minimum sources: {sorted(unknown_sources)}")
        for source, payload in arm_minimums.items():
            unknown_arms = set(payload) - set(ARMS)
            if unknown_arms:
                raise ValueError(f"unknown arms in minimums: {sorted(unknown_arms)}")
            for arm, value in payload.items():
                minimum = int(value)
                if minimum < 0:
                    raise ValueError("arm minimums must be non-negative")
                normalized_minimums[source][arm] = minimum

    entries = eligible_entries(entries)
    if source_counts is None:
        result = tuple(
            ScenarioCatalogEntry(
                record=replace(
                    entry.record,
                    split="train",
                    holdout_pool=None,
                    runtime_index=None,
                ),
                features=entry.features,
            )
            for entry in entries
        )
    else:
        if set(source_counts) != set(SOURCES):
            raise ValueError(f"source_counts must define exactly {SOURCES}")
        if any(int(value) < 0 for value in source_counts.values()):
            raise ValueError("training source counts must be non-negative")
        targets = {
            source: {
                "train": int(source_counts[source]),
                "validation": 0,
                "test": 0,
            }
            for source in SOURCES
        }
        result = assign_arm_balanced_splits_to_targets(
            entries,
            targets=targets,
            seed=int(seed),
        )
        actual_source_counts = {
            source: sum(entry.record.source == source for entry in result) for source in SOURCES
        }
        expected_source_counts = {source: int(source_counts[source]) for source in SOURCES}
        if actual_source_counts != expected_source_counts:
            raise ValueError(
                "training pool could not satisfy exact source targets: "
                f"expected={expected_source_counts}, actual={actual_source_counts}"
            )
        arm_counts = {arm: sum(entry.record.primary_arm == arm for entry in result) for arm in ARMS}
        if arm_counts and max(arm_counts.values()) - min(arm_counts.values()) > 1:
            raise ValueError(f"training pool is not near-uniform across arms: actual={arm_counts}")
        result = tuple(
            ScenarioCatalogEntry(
                record=replace(entry.record, holdout_pool=None, runtime_index=None),
                features=entry.features,
            )
            for entry in result
        )
    actual = _available_source_arm_counts(result)
    deficits = {
        (source, arm): normalized_minimums[source][arm] - actual[source][arm]
        for source in SOURCES
        for arm in ARMS
        if actual[source][arm] < normalized_minimums[source][arm]
    }
    if deficits:
        details = "; ".join(
            f"{source}/{arm}: minimum={normalized_minimums[source][arm]}, "
            f"actual={actual[source][arm]}, "
            f"deficit={normalized_minimums[source][arm] - actual[source][arm]}"
            for source, arm in sorted(deficits)
        )
        raise ValueError(f"training pool does not meet configured per-arm minimums: {details}")
    return result


def assert_pg_holdout_profile_mixture(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    expected_fractions: Mapping[str, float],
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Verify the empirical PG holdout matches its declared frozen mixture.

    `SCENARIONET-INTEGRATION` v1.2 §3.3 (`DEC-003`): the PG holdout
    generation-profile mixture is declared and frozen before generation
    (equiprobable across the five profiles by default), independent of any
    observed arm deficit. This checks the *generated* empirical PG holdout
    (`source="pg"`, `holdout_pool="empirical"`) against that declared
    mixture within a tolerance, and returns the observed fractions for
    reporting (`TEST-009`). Raises if any profile's observed share deviates
    from its declared share by more than `tolerance`.
    """

    if abs(sum(expected_fractions.values()) - 1.0) > 1e-6:
        raise ValueError("expected_fractions must sum to 1.0")
    pg_holdout = [
        entry
        for entry in entries
        if entry.record.source == "pg" and entry.record.holdout_pool == "empirical"
    ]
    if not pg_holdout:
        raise ValueError("no empirical PG holdout records found")
    counts = Counter(entry.record.pg_profile for entry in pg_holdout)
    total = len(pg_holdout)
    observed = {profile: counts.get(profile, 0) / total for profile in expected_fractions}
    deviations = {
        profile: abs(observed[profile] - expected_fractions[profile])
        for profile in expected_fractions
        if abs(observed[profile] - expected_fractions[profile]) > tolerance
    }
    if deviations:
        details = "; ".join(
            f"{profile}: expected={expected_fractions[profile]:.3f}, "
            f"observed={observed[profile]:.3f}"
            for profile in sorted(deviations)
        )
        raise ValueError(
            f"PG holdout profile mixture deviates from the declared mixture: {details}"
        )
    return observed


def assert_seed_range_disjoint(
    candidate: tuple[int, int], *, used_ranges: Sequence[tuple[int, int]]
) -> None:
    """Verify a candidate inclusive seed range does not overlap any used range.

    `SCENARIONET-INTEGRATION` v1.2 §3.3: PG holdout seeds occupy a range
    disjoint from every other range used by the project (training, prior
    frozen datasets, development). `assert_pg_seed_disjoint`
    (`splits.py`) only checks disjointness *within* one assembled catalog;
    this checks a candidate range against ranges recorded from other,
    separately-generated builds, which cannot be inferred from a single
    catalog alone.
    """

    low, high = candidate
    if low > high:
        raise ValueError(f"invalid seed range: {candidate}")
    for other_low, other_high in used_ranges:
        if low <= other_high and other_low <= high:
            raise ValueError(
                f"candidate seed range {candidate} overlaps used range ({other_low}, {other_high})"
            )


def assign_holdout_first_splits(
    entries: Sequence[ScenarioCatalogEntry],
    *,
    test_empirical_counts: Mapping[str, int],
    validation_counts: Mapping[str, int],
    stratified_total: int,
    train_arm_minimums: Mapping[str, Mapping[str, int]] | None = None,
    train_source_counts: Mapping[str, int] | None = None,
    seed: int,
    allowed_signal_reliabilities: Mapping[str, Sequence[str]] | None = None,
    empirical_candidate_entries: Sequence[ScenarioCatalogEntry] | None = None,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Orchestrate the full v1.2 holdout-first allocation (§3.5).

    Freeze order: `test_empirical` -> `validation` (both label-blind) ->
    `test_stratified` (arm-balanced, from the residual) -> `train` (the
    configured exact source totals, or every remaining eligible record when
    no totals are configured, checked against `train_arm_minimums`). No
    later stage can move a record into an earlier one, because each stage
    only ever consumes the residual left by the previous one.

    When `empirical_candidate_entries` is supplied, it is the only population
    eligible for the empirical reservations. This supports the declared PG
    holdout seed range: pre-existing PG data remains in the full catalog for
    stratified/train use but cannot contaminate the frozen empirical mixture.
    """

    candidate_entries = tuple(empirical_candidate_entries or entries)
    all_entries_by_uid = {entry.record.scenario_uid: entry for entry in entries}
    unknown_candidate_uids = {entry.record.scenario_uid for entry in candidate_entries} - set(
        all_entries_by_uid
    )
    if unknown_candidate_uids:
        raise ValueError(
            "empirical candidate entries are not present in the full catalog: "
            f"{sorted(unknown_candidate_uids)[:5]}"
        )
    empirical_reserved, _ = reserve_empirical_holdouts(
        candidate_entries,
        test_counts=test_empirical_counts,
        validation_counts=validation_counts,
        seed=seed,
        allowed_signal_reliabilities=allowed_signal_reliabilities,
    )
    empirical_test_uids = [
        entry.record.scenario_uid for entry in empirical_reserved if entry.record.split == "test"
    ]
    empirical_validation_uids = [
        entry.record.scenario_uid
        for entry in empirical_reserved
        if entry.record.split == "validation"
    ]
    empirical, after_empirical = apply_frozen_empirical_holdouts(
        entries,
        frozen_test_uids=empirical_test_uids,
        frozen_validation_uids=empirical_validation_uids,
    )
    stratified, after_stratified = reserve_stratified_pool(
        after_empirical,
        total=stratified_total,
        seed=seed,
        protected_train_arm_minimums=train_arm_minimums,
        allowed_signal_reliabilities=allowed_signal_reliabilities,
    )
    train = assign_training_pool_from_residual(
        after_stratified,
        arm_minimums=train_arm_minimums,
        source_counts=train_source_counts,
        seed=seed,
    )

    combined = empirical + stratified + train
    combined_records = [entry.record for entry in combined]
    all_group_ids = group_ids_for_entries(eligible_entries(entries))
    assert_waymo_training_20s(combined_records)
    assert_pg_seed_disjoint(combined_records)
    assert_no_pool_overlap(combined_records, group_id_by_uid=all_group_ids)
    return combined


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
            ideal_per_source = arm_target / len(SOURCES)
            split_payload["arms"][arm] = {
                "target": arm_target,
                "actual": actual_total,
                "deficit": max(0, arm_target - actual_total),
                "sources": source_payload,
                "source_compensation_from_equal_share": {
                    source: source_payload[source]["actual"] - ideal_per_source
                    for source in SOURCES
                },
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
    "reserve_empirical_holdouts",
    "apply_frozen_empirical_holdouts",
    "reserve_stratified_pool",
    "assign_training_pool_from_residual",
    "assign_holdout_first_splits",
    "assert_pg_holdout_profile_mixture",
    "assert_seed_range_disjoint",
]
