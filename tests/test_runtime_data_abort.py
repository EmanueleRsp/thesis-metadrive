from __future__ import annotations

import json

from thesis_rl.runtime.data_abort import RuntimeScenarioQuarantine, append_data_abort_record


def test_run_local_quarantine_round_trip_does_not_require_dataset_mutation(tmp_path) -> None:
    quarantine = RuntimeScenarioQuarantine()
    assert quarantine.add("waymo:fixture", "INVALID_SIGNAL_TRANSITION")
    assert not quarantine.add("waymo:fixture", "INVALID_SIGNAL_TRANSITION")
    path = tmp_path / "runtime_scenario_quarantine.json"
    quarantine.save(path)

    restored = RuntimeScenarioQuarantine.load(path)
    assert restored.scenario_uids == {"waymo:fixture"}
    assert restored.reason_counts == {"INVALID_SIGNAL_TRANSITION": 2}


def test_data_abort_jsonl_keeps_observation_out_of_text_record(tmp_path) -> None:
    path = tmp_path / "runtime_scenario_data_abort.jsonl"
    record = append_data_abort_record(
        path,
        {
            "run_id": "run-1",
            "scenario_uid": "pg:fixture",
            "reason_code": "INVALID_SIGNAL_TRANSITION",
            "full_traceback": "traceback",
            "final_observation": [1.0, 2.0],
        },
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert "final_observation" not in persisted
    assert persisted["last_valid_observation_sha256"] == record["last_valid_observation_sha256"]
