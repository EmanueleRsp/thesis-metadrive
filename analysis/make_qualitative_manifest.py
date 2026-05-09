from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _episode_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("eval_id", "")).strip(),
        str(row.get("episode_id", "")).strip(),
        str(row.get("scenario_seed", "")).strip(),
    )


def _best_score(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    success = 1.0 if _as_bool(row.get("success")) else 0.0
    collision = 1.0 if _as_bool(row.get("collision")) else 0.0
    out_of_road = 1.0 if _as_bool(row.get("out_of_road")) else 0.0
    route_completion = _to_float(row.get("route_completion")) or 0.0
    top_rule = _to_float(row.get("top_rule_violation_rate")) or 0.0
    error = _to_float(row.get("error_value")) or 0.0
    return (success, -collision, -out_of_road, route_completion, -top_rule, -error)


def _worst_score(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    success = 1.0 if _as_bool(row.get("success")) else 0.0
    collision = 1.0 if _as_bool(row.get("collision")) else 0.0
    out_of_road = 1.0 if _as_bool(row.get("out_of_road")) else 0.0
    route_completion = _to_float(row.get("route_completion")) or 0.0
    top_rule = _to_float(row.get("top_rule_violation_rate")) or 0.0
    error = _to_float(row.get("error_value")) or 0.0
    return (collision, out_of_road, top_rule, error, -route_completion, -success)


def _pick_best(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=_best_score)


def _pick_worst(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=_worst_score)


def _pick_median(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            _to_float(row.get("route_completion")) or 0.0,
            -(_to_float(row.get("error_value")) or 0.0),
        ),
    )
    return ordered[len(ordered) // 2]


def _pick_rule_violation(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _to_float(row.get("top_rule_violation_rate")) or 0.0,
            _to_float(row.get("error_value")) or 0.0,
            1.0 if _as_bool(row.get("collision")) else 0.0,
        ),
    )


def _pick_transition_case(
    rows: list[dict[str, str]],
    promotions: list[dict[str, str]],
) -> tuple[dict[str, str] | None, str]:
    if not rows:
        return None, "no_eval_episodes"
    if not promotions:
        return None, "no_promotions"

    # Pick earliest promotion to show first curriculum transition.
    promo_row = min(promotions, key=lambda row: _to_float(row.get("global_step")) or float("inf"))
    promo_step = _to_float(promo_row.get("global_step"))
    promo_run_id = str(promo_row.get("run_id", "")).strip()
    if promo_step is None:
        return None, "promotion_step_missing"

    same_run_rows = [row for row in rows if str(row.get("run_id", "")).strip() == promo_run_id]
    pool = same_run_rows if same_run_rows else rows
    if not pool:
        return None, "no_rows_for_transition"

    candidate = min(
        pool,
        key=lambda row: (
            abs((_to_float(row.get("global_step")) or 0.0) - promo_step),
            0 if str(row.get("eval_type", "")).strip().lower() == "intermediate" else 1,
        ),
    )
    return candidate, "ok"


def _select_distinct(
    *,
    category: str,
    candidates: list[dict[str, str]],
    picker: Any,
    used_episode_keys: set[tuple[str, str, str]],
) -> tuple[dict[str, str] | None, str]:
    if not candidates:
        return None, "no_candidates"
    chosen = picker(candidates)
    if chosen is None:
        return None, "picker_returned_none"
    key = _episode_key(chosen)
    if key not in used_episode_keys:
        used_episode_keys.add(key)
        return chosen, "ok"

    remaining = [row for row in candidates if _episode_key(row) not in used_episode_keys]
    if not remaining:
        return chosen, f"duplicate_allowed_{category}"
    chosen2 = picker(remaining)
    if chosen2 is None:
        return chosen, f"duplicate_allowed_{category}"
    used_episode_keys.add(_episode_key(chosen2))
    return chosen2, "ok"


def build_qualitative_manifest(
    *,
    comparison_root: Path,
    max_per_category: int = 1,
) -> tuple[Path, Path]:
    if int(max_per_category) != 1:
        raise ValueError("Current implementation supports max_per_category=1 only.")

    aggregated_dir = comparison_root / "aggregated"
    eval_rows = _read_rows(aggregated_dir / "eval_episodes_all_runs.csv")
    promotions_path = aggregated_dir / "promotions_all_runs.csv"
    promotion_rows = _read_rows(promotions_path) if promotions_path.exists() else []

    required_eval = (
        "condition_id",
        "algorithm",
        "reward_type",
        "reward_behavior",
        "curriculum_name",
        "rulebook_config",
        "run_id",
        "run_dir",
        "seed",
        "eval_id",
        "episode_id",
        "scenario_seed",
        "global_step",
        "stage",
    )
    for row in eval_rows:
        missing = [c for c in required_eval if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required fields in eval_episodes row: {missing}")

    by_condition_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eval_rows:
        by_condition_rows[str(row["condition_id"]).strip()].append(row)

    promotions_by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in promotion_rows:
        condition_id = str(row.get("condition_id", "")).strip()
        if condition_id != "":
            promotions_by_condition[condition_id].append(row)

    qualitative_dir = comparison_root / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    csv_path = qualitative_dir / "video_manifest.csv"
    md_path = qualitative_dir / "video_manifest.md"

    fieldnames = [
        "comparison_dimension",
        "comparison_id",
        "condition_id",
        "algorithm",
        "reward_type",
        "reward_behavior",
        "curriculum",
        "rulebook_config",
        "category",
        "selection_status",
        "selection_reason",
        "run_dir",
        "run_id",
        "seed",
        "eval_type",
        "eval_id",
        "episode_id",
        "scenario_seed",
        "global_step",
        "stage",
        "success",
        "collision",
        "out_of_road",
        "route_completion",
        "top_rule_violation_rate",
        "error_value",
        "render_rel_path",
    ]

    comparison_dimension = comparison_root.parent.name
    comparison_id = comparison_root.name

    rows_out: list[dict[str, str]] = []
    for condition_id in sorted(by_condition_rows.keys()):
        rows = by_condition_rows[condition_id]
        template = rows[0]
        descriptors = {
            "comparison_dimension": comparison_dimension,
            "comparison_id": comparison_id,
            "condition_id": condition_id,
            "algorithm": str(template.get("algorithm", "")).strip(),
            "reward_type": str(template.get("reward_type", "")).strip(),
            "reward_behavior": str(template.get("reward_behavior", "")).strip(),
            "curriculum": str(template.get("curriculum_name", "")).strip(),
            "rulebook_config": str(template.get("rulebook_config", "")).strip(),
        }

        final_rows = [row for row in rows if str(row.get("eval_type", "")).strip().lower() == "final"]
        pool_rows = final_rows if final_rows else rows
        used_keys: set[tuple[str, str, str]] = set()

        category_specs = (
            ("best", lambda: _select_distinct(category="best", candidates=pool_rows, picker=_pick_best, used_episode_keys=used_keys)),
            ("median", lambda: _select_distinct(category="median", candidates=pool_rows, picker=_pick_median, used_episode_keys=used_keys)),
            ("worst", lambda: _select_distinct(category="worst", candidates=pool_rows, picker=_pick_worst, used_episode_keys=used_keys)),
            (
                "rule_violation_case",
                lambda: _select_distinct(
                    category="rule_violation_case",
                    candidates=pool_rows,
                    picker=_pick_rule_violation,
                    used_episode_keys=used_keys,
                ),
            ),
        )

        for category, resolver in category_specs:
            selected, reason = resolver()
            if selected is None:
                rows_out.append(
                    {
                        **descriptors,
                        "category": category,
                        "selection_status": "unavailable",
                        "selection_reason": reason,
                        "run_dir": "",
                        "run_id": "",
                        "seed": "",
                        "eval_type": "",
                        "eval_id": "",
                        "episode_id": "",
                        "scenario_seed": "",
                        "global_step": "",
                        "stage": "",
                        "success": "",
                        "collision": "",
                        "out_of_road": "",
                        "route_completion": "",
                        "top_rule_violation_rate": "",
                        "error_value": "",
                        "render_rel_path": "",
                    }
                )
                continue

            rows_out.append(
                {
                    **descriptors,
                    "category": category,
                    "selection_status": "selected",
                    "selection_reason": reason,
                    "run_dir": str(selected.get("run_dir", "")).strip(),
                    "run_id": str(selected.get("run_id", "")).strip(),
                    "seed": str(selected.get("seed", "")).strip(),
                    "eval_type": str(selected.get("eval_type", "")).strip(),
                    "eval_id": str(selected.get("eval_id", "")).strip(),
                    "episode_id": str(selected.get("episode_id", "")).strip(),
                    "scenario_seed": str(selected.get("scenario_seed", "")).strip(),
                    "global_step": str(selected.get("global_step", "")).strip(),
                    "stage": str(selected.get("stage", "")).strip(),
                    "success": str(selected.get("success", "")).strip(),
                    "collision": str(selected.get("collision", "")).strip(),
                    "out_of_road": str(selected.get("out_of_road", "")).strip(),
                    "route_completion": str(selected.get("route_completion", "")).strip(),
                    "top_rule_violation_rate": str(selected.get("top_rule_violation_rate", "")).strip(),
                    "error_value": str(selected.get("error_value", "")).strip(),
                    "render_rel_path": "",
                }
            )

        transition_selected, transition_reason = _pick_transition_case(
            rows=rows,
            promotions=promotions_by_condition.get(condition_id, []),
        )
        if transition_selected is None:
            rows_out.append(
                {
                    **descriptors,
                    "category": "curriculum_transition_case",
                    "selection_status": "unavailable",
                    "selection_reason": transition_reason,
                    "run_dir": "",
                    "run_id": "",
                    "seed": "",
                    "eval_type": "",
                    "eval_id": "",
                    "episode_id": "",
                    "scenario_seed": "",
                    "global_step": "",
                    "stage": "",
                    "success": "",
                    "collision": "",
                    "out_of_road": "",
                    "route_completion": "",
                    "top_rule_violation_rate": "",
                    "error_value": "",
                    "render_rel_path": "",
                }
            )
        else:
            rows_out.append(
                {
                    **descriptors,
                    "category": "curriculum_transition_case",
                    "selection_status": "selected",
                    "selection_reason": transition_reason,
                    "run_dir": str(transition_selected.get("run_dir", "")).strip(),
                    "run_id": str(transition_selected.get("run_id", "")).strip(),
                    "seed": str(transition_selected.get("seed", "")).strip(),
                    "eval_type": str(transition_selected.get("eval_type", "")).strip(),
                    "eval_id": str(transition_selected.get("eval_id", "")).strip(),
                    "episode_id": str(transition_selected.get("episode_id", "")).strip(),
                    "scenario_seed": str(transition_selected.get("scenario_seed", "")).strip(),
                    "global_step": str(transition_selected.get("global_step", "")).strip(),
                    "stage": str(transition_selected.get("stage", "")).strip(),
                    "success": str(transition_selected.get("success", "")).strip(),
                    "collision": str(transition_selected.get("collision", "")).strip(),
                    "out_of_road": str(transition_selected.get("out_of_road", "")).strip(),
                    "route_completion": str(transition_selected.get("route_completion", "")).strip(),
                    "top_rule_violation_rate": str(transition_selected.get("top_rule_violation_rate", "")).strip(),
                    "error_value": str(transition_selected.get("error_value", "")).strip(),
                    "render_rel_path": "",
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Condition | Category | Status | Seed | Eval | Episode | Route completion | Top-rule violation | Note |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows_out:
            handle.write(
                "| "
                + " | ".join(
                    [
                        row["condition_id"],
                        row["category"],
                        row["selection_status"],
                        row["seed"],
                        row["eval_id"],
                        row["episode_id"],
                        row["route_completion"],
                        row["top_rule_violation_rate"],
                        row["selection_reason"],
                    ]
                )
                + " |\n"
            )

    print(f"Wrote qualitative manifest CSV -> {csv_path}")
    print(f"Wrote qualitative manifest MD  -> {md_path}")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qualitative video manifest for a comparison view.")
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--max-per-category", type=int, default=1)
    args = parser.parse_args()

    build_qualitative_manifest(
        comparison_root=Path(args.comparison_root),
        max_per_category=int(args.max_per_category),
    )


if __name__ == "__main__":
    main()
