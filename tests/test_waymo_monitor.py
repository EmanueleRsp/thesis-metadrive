from __future__ import annotations

from thesis_rl.scenarios.waymo_monitor import WaymoProgressState, parse_converter_line


def test_parse_converter_worker_events() -> None:
    assert parse_converter_line("INFO:scenarionet:Find 35 waymo files") == [
        {"kind": "total", "total_files": 35}
    ]
    assert parse_converter_line(
        "INFO:scenarionet:Worker 6 is reading raw file: /workspace/waymo_raw/training_20s.tfrecord-00034-of-01000"
    ) == [
        {
            "kind": "worker_read",
            "worker": 6,
            "file": "training_20s.tfrecord-00034-of-01000",
        }
    ]


def test_progress_state_aggregates_worker_file_completion() -> None:
    state = WaymoProgressState()
    state.apply({"kind": "started", "total_files": 2})
    state.apply({"kind": "worker_read", "worker": 0, "file": "first"})
    state.apply({"kind": "worker_read", "worker": 0, "file": "second"})
    assert state.completed_files == 1
    state.apply({"kind": "worker_finished", "worker": 0, "files": 2})
    assert state.completed_files == 2
    assert state.workers[0].status == "done"


def test_progress_state_records_failure() -> None:
    state = WaymoProgressState()
    state.apply({"kind": "finished", "exit_code": 1})
    assert state.finished is True
    assert state.exit_code == 1
    assert state.last_message == "Conversion failed"
