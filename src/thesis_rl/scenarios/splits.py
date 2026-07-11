from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Mapping, Sequence, cast

import numpy as np

from thesis_rl.scenarios.records import ScenarioRecord, ScenarioSplit


SPLIT_ORDER = ("train", "validation", "test")


def assert_no_group_overlap(
    records: Sequence[ScenarioRecord],
    group_id_by_uid: Mapping[str, str],
) -> None:
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        try:
            group_id = group_id_by_uid[record.scenario_uid]
        except KeyError as exc:
            raise ValueError(f"missing group id for {record.scenario_uid}") from exc
        groups_by_split[record.split].add(str(group_id))
    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1 :]:
            overlap = groups_by_split[left].intersection(groups_by_split[right])
            if overlap:
                raise ValueError(f"source groups overlap between {left} and {right}: {sorted(overlap)}")


def assert_pg_seed_disjoint(records: Sequence[ScenarioRecord]) -> None:
    seed_split: dict[int, str] = {}
    for record in records:
        if record.source != "pg":
            continue
        assert record.pg_seed is not None
        previous = seed_split.setdefault(record.pg_seed, record.split)
        if previous != record.split:
            raise ValueError(
                f"PG seed {record.pg_seed} occurs in both {previous} and {record.split}"
            )


def assert_waymo_training_20s(records: Sequence[ScenarioRecord]) -> None:
    invalid = [
        record.scenario_uid
        for record in records
        if record.source == "waymo" and record.official_split != "training_20s"
    ]
    if invalid:
        raise ValueError(f"Waymo records outside training_20s: {invalid}")


def assign_grouped_splits(
    records: Sequence[ScenarioRecord],
    *,
    group_id_by_uid: Mapping[str, str],
    counts: Mapping[str, int],
    seed: int,
) -> tuple[ScenarioRecord, ...]:
    if set(counts) != set(SPLIT_ORDER):
        raise ValueError(f"counts must define exactly {SPLIT_ORDER}")
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("split counts must be non-negative integers")
    if sum(counts.values()) != len(records):
        raise ValueError("split counts must sum to the number of records")

    grouped: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        try:
            group_id = str(group_id_by_uid[record.scenario_uid])
        except KeyError as exc:
            raise ValueError(f"missing group id for {record.scenario_uid}") from exc
        grouped[group_id].append(record)

    rng = np.random.default_rng(int(seed))
    group_ids = sorted(grouped)
    rng.shuffle(group_ids)
    remaining = {split: int(counts[split]) for split in SPLIT_ORDER}
    assigned: list[ScenarioRecord] = []
    for group_id in group_ids:
        group = grouped[group_id]
        eligible = [split for split in SPLIT_ORDER if remaining[split] >= len(group)]
        if not eligible:
            raise ValueError(
                "grouped split cannot satisfy exact counts; "
                f"group {group_id!r} has {len(group)} records, remaining={remaining}"
            )
        # Fill the split with the largest remaining capacity; SPLIT_ORDER breaks ties.
        split = max(eligible, key=lambda item: (remaining[item], -SPLIT_ORDER.index(item)))
        assigned.extend(
            replace(record, split=cast(ScenarioSplit, split), runtime_index=None)
            for record in group
        )
        remaining[split] -= len(group)

    if any(remaining.values()):
        raise ValueError(f"grouped split did not fill requested counts: {remaining}")
    assigned_by_uid = {record.scenario_uid: record for record in assigned}
    result = tuple(assigned_by_uid[record.scenario_uid] for record in records)
    assert_no_group_overlap(result, group_id_by_uid)
    return result
