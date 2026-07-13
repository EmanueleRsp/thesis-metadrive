from __future__ import annotations

from dataclasses import replace

import pytest

from thesis_rl.scenarios.provider import FixedSequenceScenarioProvider, UniformScenarioProvider
from thesis_rl.scenarios.records import ScenarioRecord


def _record(index: int, source: str, *, split: str = "train") -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid=f"{source}:v1:{index}",
        scenario_id=str(index),
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{index}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=f"log-{index}" if source == "waymo" else None,
        source_scenario_id=str(index) if source == "waymo" else None,
        dataset_version="v1",
        converter_version="converter" if source == "waymo" else None,
        split=split,  # type: ignore[arg-type]
        runtime_index=index,
        length=100,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def test_uniform_provider_is_reproducible_per_worker() -> None:
    records = [_record(i, source) for source in ("waymo", "pg") for i in range(5)]
    first = UniformScenarioProvider(records, global_seed=42)
    second = UniformScenarioProvider(records, global_seed=42)

    sequence_one = [first.sample(split="train", worker_id=3).scenario_uid for _ in range(50)]
    sequence_two = [second.sample(split="train", worker_id=3).scenario_uid for _ in range(50)]

    assert sequence_one == sequence_two


def test_uniform_provider_source_balance_and_strict_failure() -> None:
    records = [_record(0, "waymo"), _record(0, "pg")]
    provider = UniformScenarioProvider(records, global_seed=0)

    for _ in range(10_000):
        provider.sample(split="train", worker_id=0)

    waymo_fraction = provider.reset_counts["waymo"] / 10_000
    assert 0.47 < waymo_fraction < 0.53
    with pytest.raises(LookupError, match="fallback is disabled"):
        provider.sample(split="test", worker_id=0, source="waymo")


def test_uniform_provider_rejects_fallback_mode() -> None:
    with pytest.raises(ValueError, match="strict=true"):
        UniformScenarioProvider([_record(0, "pg")], global_seed=0, allow_fallback=True)


def test_fixed_sequence_provider_order_and_exhaustion() -> None:
    records = [_record(0, "waymo", split="validation"), _record(1, "pg", split="validation")]
    provider = FixedSequenceScenarioProvider(records, repeat=False)

    assert provider.sample(split="validation", worker_id=0) == records[0]
    assert provider.sample(split="validation", worker_id=0) == records[1]
    with pytest.raises(LookupError, match="exhausted"):
        provider.sample(split="validation", worker_id=0)


def test_fixed_sequence_rejects_invalid_record() -> None:
    invalid = replace(_record(0, "pg"), validation_status="invalid")
    with pytest.raises(ValueError, match="invalid scenarios"):
        FixedSequenceScenarioProvider([invalid])
