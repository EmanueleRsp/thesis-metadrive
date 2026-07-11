from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from thesis_rl.scenarios.records import ScenarioRecord
from thesis_rl.scenarios.runtime_database import (
    assign_runtime_indices,
    build_runtime_database,
    verify_runtime_mapping,
)


def _record(relative_path: str, *, index: int = 0) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid=f"pg:v1:{index}",
        scenario_id=str(index),
        source="pg",
        relative_path=relative_path,
        official_split=None,
        source_log_id=None,
        source_scenario_id=None,
        dataset_version="v1",
        converter_version=None,
        split="train",
        runtime_index=index,
        length=10,
        pg_profile="P0_simple",
        pg_seed=index,
        map_id="S",
        primary_arm="A0_simple_lane_follow",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def _source_database(root: Path, name: str) -> tuple[Path, str]:
    database = root / "pg" / "database" / name
    database.mkdir(parents=True)
    filename = f"sd_{name}.pkl"
    (database / filename).write_bytes(b"scenario")
    with (database / "dataset_summary.pkl").open("wb") as handle:
        pickle.dump({filename: {"scenario_id": name}}, handle)
    with (database / "dataset_mapping.pkl").open("wb") as handle:
        pickle.dump({filename: ""}, handle)
    return database, filename


def test_runtime_database_uses_relative_mapping_without_copying_scenarios(tmp_path: Path) -> None:
    source, filename = _source_database(tmp_path, "one")
    record = _record(source.relative_to(tmp_path).as_posix())
    runtime = tmp_path / "runtime" / "train"

    build_runtime_database([record], data_root=tmp_path, runtime_directory=runtime)

    with (runtime / "dataset_mapping.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    assert mapping[filename] != ""
    assert (runtime / mapping[filename] / filename).resolve() == source / filename
    assert verify_runtime_mapping(runtime) == (filename,)
    assert not (runtime / filename).exists()


def test_runtime_database_rejects_missing_catalog_file(tmp_path: Path) -> None:
    record = _record("pg/database/missing.pkl")
    with pytest.raises(FileNotFoundError, match="catalog scenario file"):
        build_runtime_database([record], data_root=tmp_path, runtime_directory=tmp_path / "runtime")


def test_runtime_indices_are_deterministic_and_invalid_records_are_excluded() -> None:
    first = _record("pg/database/a.pkl", index=99)
    second = _record("pg/database/b.pkl", index=98)
    second = replace(second, scenario_uid="pg:v1:second", scenario_id="second")
    invalid = replace(
        first,
        scenario_uid="pg:v1:invalid",
        scenario_id="invalid",
        validation_status="invalid",
    )

    assigned = assign_runtime_indices([second, first])

    assert [record.runtime_index for record in assigned] == [0, 1]
    assert [record.scenario_uid for record in assigned] == ["pg:v1:0", "pg:v1:second"]
    assert invalid.validation_status == "invalid"
