"""step_timing_instrumentation_v1 REQ-001/REQ-002/REQ-005 (docs/implementation/
step_timing_instrumentation_v1_exec_plan.md): per-chunk, per-component wall-
clock breakdown persisted to ``step_timing.csv`` via
``train_loop._write_step_timing_rows``."""

from __future__ import annotations

import csv
from pathlib import Path

from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.loops.train_loop import _write_step_timing_rows


def _read_rows(csv_dir: Path) -> list[dict[str, str]]:
    with (csv_dir / "step_timing.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_fields() -> dict[str, object]:
    return {"algorithm": "td3_sb3", "seed": 7, "run_id": "run-abc"}


def test_one_row_written_per_phase_component(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=3,
        global_step=1000,
        phase_seconds={"observation": 1.0, "env_step": 3.0, "learner_update": 6.0},
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    assert set(rows) == {"observation", "env_step", "learner_update"}
    assert rows["observation"]["run_id"] == "run-abc"
    assert rows["observation"]["algorithm"] == "td3_sb3"
    assert rows["observation"]["chunk_id"] == "3"
    assert rows["observation"]["global_step"] == "1000"


def test_percentages_and_averages_computed_correctly(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={"observation": 1.0, "env_step": 3.0, "learner_update": 6.0},
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    assert float(rows["observation"]["pct_of_elapsed"]) == 10.0
    assert float(rows["env_step"]["pct_of_elapsed"]) == 30.0
    assert float(rows["learner_update"]["pct_of_elapsed"]) == 60.0
    assert float(rows["observation"]["avg_seconds_per_step"]) == 0.01
    assert float(rows["env_step"]["avg_seconds_per_step"]) == 0.03


def test_top_level_percentages_sum_to_roughly_100_including_unattributed(
    tmp_path: Path,
) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={
            "observation": 1.0,
            "encoder_action": 0.5,
            "env_step": 3.0,
            "transition_collection": 0.5,
            "learner_update": 4.0,
            "unattributed": 1.0,
            # Nested diagnostic detail excluded from the "sums to elapsed"
            # invariant, matching how ``agent.py`` derives ``unattributed``.
            "worker_wrapped_env_step": 2.5,
            "rulebook_snapshot": 0.2,
        },
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    top_level = (
        "observation",
        "encoder_action",
        "env_step",
        "transition_collection",
        "learner_update",
        "unattributed",
    )
    total_pct = sum(float(rows[name]["pct_of_elapsed"]) for name in top_level)
    assert abs(total_pct - 100.0) < 0.5


def test_env_step_ipc_overhead_derived_when_both_inputs_present(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={"env_step": 5.0, "worker_wrapped_env_step": 3.0},
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    assert "env_step_ipc_overhead" in rows
    assert float(rows["env_step_ipc_overhead"]["seconds"]) == 2.0


def test_env_step_ipc_overhead_omitted_when_worker_timing_missing(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={"env_step": 5.0},
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    assert "env_step_ipc_overhead" not in rows
    assert "env_step" in rows


def test_empty_phase_seconds_writes_no_rows_and_does_not_raise(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={},
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds=None,
        elapsed_seconds=10.0,
        chunk_steps_actual=100,
    )

    assert not (tmp_path / "step_timing.csv").exists()


def test_zero_chunk_steps_actual_does_not_raise_and_guards_average(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    _write_step_timing_rows(
        recorder,
        base_csv_fields=_base_fields(),
        chunk_id=1,
        global_step=500,
        phase_seconds={"observation": 1.0},
        elapsed_seconds=10.0,
        chunk_steps_actual=0,
    )

    rows = {row["component"]: row for row in _read_rows(tmp_path)}
    assert float(rows["observation"]["avg_seconds_per_step"]) == 1.0
