from __future__ import annotations

from dataclasses import replace

import pytest

from thesis_rl.scenarios.records import ScenarioRecord
from thesis_rl.scenarios.splits import (
    assert_no_group_overlap,
    assert_no_pool_overlap,
    assert_pg_seed_disjoint,
    assert_waymo_training_20s,
    assign_grouped_splits,
)


def _pg_record(index: int, *, split: str = "train", seed: int | None = None) -> ScenarioRecord:
    actual_seed = index if seed is None else seed
    return ScenarioRecord(
        scenario_uid=f"pg:v1:{index}",
        scenario_id=str(index),
        source="pg",
        relative_path=f"pg/database/{index}.pkl",
        official_split=None,
        source_log_id=None,
        source_scenario_id=None,
        dataset_version="v1",
        converter_version=None,
        split=split,  # type: ignore[arg-type]
        runtime_index=None,
        length=100,
        pg_profile="P0_simple",
        pg_seed=actual_seed,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def test_grouped_split_is_deterministic_and_disjoint() -> None:
    records = tuple(_pg_record(index) for index in range(10))
    groups = {record.scenario_uid: f"group-{index // 2}" for index, record in enumerate(records)}
    counts = {"train": 6, "validation": 2, "test": 2}

    first = assign_grouped_splits(records, group_id_by_uid=groups, counts=counts, seed=7)
    second = assign_grouped_splits(records, group_id_by_uid=groups, counts=counts, seed=7)

    assert [record.split for record in first] == [record.split for record in second]
    assert {split: sum(record.split == split for record in first) for split in counts} == counts
    assert_no_group_overlap(first, groups)


def test_grouped_split_fails_when_exact_counts_are_impossible() -> None:
    records = tuple(_pg_record(index) for index in range(4))
    groups = {
        record.scenario_uid: "large" if index < 3 else "small"
        for index, record in enumerate(records)
    }

    with pytest.raises(ValueError, match="cannot satisfy exact counts"):
        assign_grouped_splits(
            records,
            group_id_by_uid=groups,
            counts={"train": 2, "validation": 1, "test": 1},
            seed=0,
        )


def test_pg_seed_overlap_is_rejected() -> None:
    records = [_pg_record(1, split="train", seed=99), _pg_record(2, split="test", seed=99)]
    with pytest.raises(ValueError, match="PG seed 99"):
        assert_pg_seed_disjoint(records)


def test_waymo_origin_must_be_training_20s() -> None:
    base = _pg_record(1)
    waymo = replace(
        base,
        scenario_uid="waymo:v1:1",
        source="waymo",
        relative_path="waymo/database/1.pkl",
        official_split="validation",
        pg_seed=None,
        pg_profile=None,
    )
    with pytest.raises(ValueError, match="outside training_20s"):
        assert_waymo_training_20s([waymo])


def test_pool_overlap_distinguishes_empirical_from_stratified_test() -> None:
    # TEST-004/TEST-013 (SCENARIONET-INTEGRATION v1.2 REQ-013): the pool
    # check treats split="test" records as two disjoint pools according to
    # holdout_pool, in addition to the three canonical splits.
    empirical = replace(_pg_record(0, split="test", seed=100), holdout_pool="empirical")
    stratified = replace(_pg_record(1, split="test", seed=101), holdout_pool="stratified")
    validation = _pg_record(2, split="validation", seed=102)
    train = _pg_record(3, split="train", seed=103)
    records = (empirical, stratified, validation, train)
    group_ids = {record.scenario_uid: f"pg-seed:{record.pg_seed}" for record in records}

    assert_no_pool_overlap(records, group_ids)


def test_pool_overlap_rejects_shared_group_between_empirical_and_stratified() -> None:
    shared_seed = 200
    empirical = replace(_pg_record(0, split="test", seed=shared_seed), holdout_pool="empirical")
    stratified = replace(_pg_record(1, split="test", seed=shared_seed), holdout_pool="stratified")
    records = (empirical, stratified)
    group_ids = {record.scenario_uid: f"pg-seed:{record.pg_seed}" for record in records}

    with pytest.raises(ValueError, match="test_empirical and test_stratified"):
        assert_no_pool_overlap(records, group_ids)


def test_pool_overlap_degrades_to_three_way_check_without_holdout_pool() -> None:
    # A v1.1-produced test record with holdout_pool=None behaves exactly
    # like the existing three-way assert_no_group_overlap check.
    test_record = _pg_record(0, split="test", seed=300)
    validation_record = _pg_record(1, split="validation", seed=300)
    records = (test_record, validation_record)
    group_ids = {record.scenario_uid: f"pg-seed:{record.pg_seed}" for record in records}

    with pytest.raises(ValueError, match="train|validation|test"):
        assert_no_pool_overlap(records, group_ids)
