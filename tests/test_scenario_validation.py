from __future__ import annotations

import pickle
from pathlib import Path

from thesis_rl.scenarios.records import ScenarioRecord
from thesis_rl.scenarios.validation import (
    validate_scenario_file,
    validate_records,
    validation_summary,
    write_validation_summary,
)


def _record(relative_path: str, length: int = 91) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid="waymo:v1:fixture",
        scenario_id="fixture",
        source="waymo",
        relative_path=relative_path,
        official_split="training_20s",
        source_log_id="source",
        source_scenario_id="fixture",
        dataset_version="v1",
        converter_version=None,
        split="train",
        runtime_index=0,
        length=length,
        pg_profile=None,
        pg_seed=None,
        map_id=None,
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def test_bundled_waymo_file_passes_application_validation() -> None:
    path = sorted(Path("third_party/metadrive/metadrive/assets/waymo").glob("sd_*.pkl"))[0]
    result = validate_scenario_file(path, _record(path.name))

    assert result.status in {"valid", "warning"}
    assert result.scenario_length == 91


def test_malformed_scenario_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "broken.pkl"
    with path.open("wb") as handle:
        pickle.dump({"id": "broken"}, handle)

    result = validate_scenario_file(path, _record(path.name))

    assert result.status == "invalid"
    assert result.warnings

    summary = validation_summary([result], catalog_hash="hash")
    assert summary["counts"]["invalid"] == 1
    path = write_validation_summary(
        [result], tmp_path / "validation_summary.json", catalog_hash="hash"
    )
    assert path.exists()


def test_validation_progress_callback_reports_each_record(tmp_path: Path) -> None:
    path = tmp_path / "broken.pkl"
    with path.open("wb") as handle:
        pickle.dump({"id": "broken"}, handle)

    events: list[tuple[int, int, str, str]] = []
    results = validate_records(
        [_record(path.name)],
        data_root=tmp_path,
        progress_callback=lambda index, total, record, result: events.append(
            (index, total, record.scenario_uid, result.status)
        ),
    )

    assert len(results) == 1
    assert events == [(1, 1, "waymo:v1:fixture", "invalid")]
