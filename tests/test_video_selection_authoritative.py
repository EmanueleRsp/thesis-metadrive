from __future__ import annotations

import csv
import json
from pathlib import Path

from thesis_rl.analysis.videos.select_video_episodes import select_video_episodes


def test_select_video_episodes_carries_authoritative_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = csv_dir / "eval_episodes.csv"

    fieldnames = [
        "eval_id",
        "episode_id",
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
    rows = [
        {
            "eval_id": "1",
            "episode_id": "1",
            "scenario_seed": "101",
            "scenario_id": "seed_101",
            "stage": "baseline",
            "global_step": "100",
            "reward": "10.0",
            "success": "1",
            "collision": "0",
            "out_of_road": "0",
            "route_completion": "1.0",
            "error_value": "0.0",
            "violated_rules": "none",
            "video_path": "videos/offline.gif",
            "video_authoritative_path": "videos/final_eval/eval_0001/episode_0001.gif",
            "video_manifest_path": "videos/final_eval/eval_0001/episode_0001.manifest.json",
            "trajectory_log_path": "videos/final_eval/eval_0001/episode_0001.trajectory.jsonl",
            "video_recorded_live": "true",
        }
    ]
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    select_video_episodes(run_dir=run_dir, source="final", max_videos=1)

    payload = json.loads((run_dir / "videos" / "metadata" / "video_selection.json").read_text(encoding="utf-8"))
    assert payload[0]["video_path"] == "videos/final_eval/eval_0001/episode_0001.gif"
    assert payload[0]["video_authoritative_path"] == "videos/final_eval/eval_0001/episode_0001.gif"
    assert payload[0]["video_manifest_path"] == "videos/final_eval/eval_0001/episode_0001.manifest.json"
    assert payload[0]["trajectory_log_path"] == "videos/final_eval/eval_0001/episode_0001.trajectory.jsonl"
    assert payload[0]["has_authoritative_video"] is True
