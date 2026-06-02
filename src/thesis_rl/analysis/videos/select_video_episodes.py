from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_eval_episodes(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_best(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None

    def score(row: dict[str, str]) -> tuple[float, float]:
        route = _to_float(row.get("route_completion")) or 0.0
        err = _to_float(row.get("error_value")) or 0.0
        return (route, -err)

    return max(rows, key=score)


def _pick_median(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: _to_float(row.get("route_completion")) or 0.0)
    return ordered[len(ordered) // 2]


def _pick_worst_ev(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _to_float(row.get("error_value")) or -1.0)


def _pick_collision(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [r for r in rows if str(r.get("collision", "")).strip().lower() in {"1", "true", "yes"}]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row.get("error_value")) or -1.0)


def _pick_out_of_road(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [r for r in rows if str(r.get("out_of_road", "")).strip().lower() in {"1", "true", "yes"}]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row.get("error_value")) or -1.0)


def _dedupe_selected(items: list[tuple[str, dict[str, str]]]) -> list[tuple[str, dict[str, str]]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, dict[str, str]]] = []
    for tag, row in items:
        key = (
            str(row.get("eval_id", "")),
            str(row.get("episode_id", "")),
            str(row.get("scenario_seed", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((tag, row))
    return out


def select_video_episodes(
    run_dir: Path,
    *,
    source: str = "final",
    max_videos: int = 5,
) -> None:
    eval_csv = run_dir / "csv" / "eval_episodes.csv"
    rows = _load_eval_episodes(eval_csv)
    if not rows:
        raise ValueError(f"No rows in {eval_csv}")

    if source == "final":
        eval_ids = [int(_to_float(r.get("eval_id")) or -1) for r in rows]
        target_eval_id = max(eval_ids)
        rows = [r for r in rows if int(_to_float(r.get("eval_id")) or -1) == target_eval_id]
    else:
        target_eval_id = int(source)
        rows = [r for r in rows if int(_to_float(r.get("eval_id")) or -1) == target_eval_id]

    selected: list[tuple[str, dict[str, str]]] = []
    pickers = (
        ("best", _pick_best),
        ("median", _pick_median),
        ("worst_ev", _pick_worst_ev),
        ("collision", _pick_collision),
        ("out_of_road", _pick_out_of_road),
    )
    for tag, fn in pickers:
        episode = fn(rows)
        if episode is not None:
            selected.append((tag, episode))

    selected = _dedupe_selected(selected)[: max(1, int(max_videos))]

    metadata_dir = run_dir / "videos" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    json_path = metadata_dir / "video_selection.json"
    payload = []
    for tag, row in selected:
        payload.append(
            {
                "tag": tag,
                "eval_id": row.get("eval_id"),
                "episode_id": row.get("episode_id"),
                "scenario_seed": row.get("scenario_seed"),
                "scenario_id": row.get("scenario_id"),
                "stage": row.get("stage"),
                "global_step": row.get("global_step"),
                "reward": row.get("reward"),
                "success": row.get("success"),
                "collision": row.get("collision"),
                "out_of_road": row.get("out_of_road"),
                "route_completion": row.get("route_completion"),
                "error_value": row.get("error_value"),
                "violated_rules": row.get("violated_rules"),
            }
        )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index_path = metadata_dir / "video_index.csv"
    fieldnames = [
        "video_path",
        "tag",
        "eval_id",
        "episode_id",
        "stage",
        "global_step",
        "scenario_id",
        "scenario_seed",
        "reward",
        "success",
        "collision",
        "out_of_road",
        "route_completion",
        "error_value",
        "violated_rules",
        "original_reward",
        "replay_reward",
        "reward_abs_diff",
        "original_route_completion",
        "replay_route_completion",
        "route_completion_abs_diff",
        "original_error_value",
        "replay_error_value",
        "error_value_abs_diff",
        "original_success",
        "replay_success",
        "success_match",
        "original_collision",
        "replay_collision",
        "collision_match",
        "original_out_of_road",
        "replay_out_of_road",
        "out_of_road_match",
        "replay_match",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload:
            writer.writerow(
                {
                    "video_path": "",
                    "tag": item["tag"],
                    "eval_id": item["eval_id"],
                    "episode_id": item["episode_id"],
                    "stage": item["stage"],
                    "global_step": item["global_step"],
                    "scenario_id": item["scenario_id"],
                    "scenario_seed": item["scenario_seed"],
                    "reward": item["reward"],
                    "success": item["success"],
                    "collision": item["collision"],
                    "out_of_road": item["out_of_road"],
                    "route_completion": item["route_completion"],
                    "error_value": item["error_value"],
                    "violated_rules": item["violated_rules"],
                    "original_reward": item["reward"],
                    "replay_reward": "",
                    "reward_abs_diff": "",
                    "original_route_completion": item["route_completion"],
                    "replay_route_completion": "",
                    "route_completion_abs_diff": "",
                    "original_error_value": item["error_value"],
                    "replay_error_value": "",
                    "error_value_abs_diff": "",
                    "original_success": item["success"],
                    "replay_success": "",
                    "success_match": "",
                    "original_collision": item["collision"],
                    "replay_collision": "",
                    "collision_match": "",
                    "original_out_of_road": item["out_of_road"],
                    "replay_out_of_road": "",
                    "out_of_road_match": "",
                    "replay_match": "",
                }
            )

    print(f"Selected {len(payload)} episodes for run: {run_dir}")
    print(f"Wrote selection JSON -> {json_path}")
    print(f"Wrote video index CSV -> {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative episodes for video rendering.")
    parser.add_argument("--run-dir", required=True, help="Absolute or relative run directory (contains csv/).")
    parser.add_argument(
        "--source",
        default="final",
        help="`final` for latest eval_id, or explicit eval_id integer.",
    )
    parser.add_argument("--max-videos", type=int, default=5)
    args = parser.parse_args()

    select_video_episodes(
        run_dir=Path(args.run_dir),
        source=str(args.source),
        max_videos=int(args.max_videos),
    )


if __name__ == "__main__":
    main()
