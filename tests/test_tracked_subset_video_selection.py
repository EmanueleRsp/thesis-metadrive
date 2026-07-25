"""EVAL-PROTOCOL v1.0 REQ-014/DEC-014 (amended 2026-07-25) tests: the fixed
tracked-subset GIF mechanism must select the same pre-declared scenario_uids
across every evaluation (periodic validation and final test alike), unlike
the post-hoc, score-based, single-eval_id ``select_video_episodes`` picks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from thesis_rl.analysis.videos.select_video_episodes import select_tracked_subset_episodes

FIELDNAMES = [
    "eval_id",
    "eval_type",
    "episode_id",
    "scenario_uid",
    "scenario_seed",
    "scenario_id",
    "stage",
    "global_step",
    "reward",
    "success",
    "collision",
    "out_of_road",
    "route_completion",
    "error_value",
    "violated_rules",
    "video_path",
    "video_authoritative_path",
    "video_manifest_path",
    "trajectory_log_path",
    "video_recorded_live",
]


def _write_eval_episodes(run_dir: Path, rows: list[dict[str, str]]) -> None:
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with (csv_dir / "eval_episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(eval_id: int, eval_type: str, scenario_uid: str) -> dict[str, str]:
    return {
        "eval_id": str(eval_id),
        "eval_type": eval_type,
        "episode_id": "1",
        "scenario_uid": scenario_uid,
        "scenario_seed": "1",
        "scenario_id": scenario_uid,
        "stage": "stage0",
        "global_step": str(eval_id * 1000),
        "reward": "1.0",
        "success": "1",
        "collision": "0",
        "out_of_road": "0",
        "route_completion": "0.9",
        "error_value": "0.0",
        "violated_rules": "",
        "video_path": "",
        "video_authoritative_path": "",
        "video_manifest_path": "",
        "trajectory_log_path": "",
        "video_recorded_live": "",
    }


def test_select_tracked_subset_episodes_matches_across_periodic_and_final_evals(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    tracked = ("waymo:v1:A0:1", "waymo:v1:A1:2")
    rows = [
        _row(1, "intermediate", "waymo:v1:A0:1"),
        _row(1, "intermediate", "waymo:v1:A2:99"),  # not tracked
        _row(2, "intermediate", "waymo:v1:A0:1"),
        _row(2, "intermediate", "waymo:v1:A1:2"),
        _row(3, "final", "waymo:v1:A0:1"),
        _row(3, "final", "waymo:v1:A1:2"),
        _row(3, "final", "waymo:v1:A3:5"),  # not tracked
    ]
    _write_eval_episodes(run_dir, rows)

    payload = select_tracked_subset_episodes(run_dir, tracked_scenario_uids=tracked)

    # 1 (eval 1) + 2 (eval 2) + 2 (eval 3/final) = 5 tracked occurrences.
    assert len(payload) == 5
    assert all(item["scenario_uid"] in tracked for item in payload)
    assert {item["eval_id"] for item in payload} == {"1", "2", "3"}
    assert any(item["eval_type"] == "final" for item in payload)
    assert any(item["eval_type"] == "intermediate" for item in payload)
    assert all(item["tag"] == "tracked_progression" for item in payload)

    json_path = run_dir / "videos" / "metadata" / "tracked_subset_selection.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == payload


def test_select_tracked_subset_episodes_empty_when_no_uid_matches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_eval_episodes(run_dir, [_row(1, "final", "waymo:v1:A0:1")])

    payload = select_tracked_subset_episodes(
        run_dir, tracked_scenario_uids=("waymo:v1:does_not_exist",)
    )
    assert payload == []
