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


def _scenario_uid_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map ``scenario_uid`` -> first row with that UID (rows without a
    recorded ``scenario_uid`` are excluded). Used only by the
    `algorithm_disagreement` cross-condition selector (`REQ-014`), which
    requires directly comparing the identical `scenario_uid` across
    conditions.
    """
    by_uid: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = str(row.get("scenario_uid", "")).strip()
        if uid and uid not in by_uid:
            by_uid[uid] = row
    return by_uid


def _pick_disagreement(
    *,
    condition_id: str,
    rows: list[dict[str, str]],
    pools_by_condition: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, str] | None, str]:
    """REQ-014 `algorithm_disagreement`: pick the episode from ``rows``
    (this condition's pool) whose ``scenario_uid`` also appears in another
    condition's pool with a diverging `success` outcome, preferring the
    largest `route_completion` gap between the two conditions on that
    shared scenario. This is the minimal direct-comparison implementation
    described in `REQ-014`'s invariants (identical `scenario_uid` across
    compared conditions); it does not use any additional similarity or
    disagreement threshold beyond a differing boolean `success` outcome, so
    no `AWAITING_CONFIRMATION` threshold decision is needed.
    """
    by_uid = _scenario_uid_index(rows)
    if not by_uid:
        return None, "no_scenario_uid_recorded"

    other_condition_ids = sorted(cid for cid in pools_by_condition if cid != condition_id)
    if not other_condition_ids:
        return None, "no_other_conditions"

    best: dict[str, str] | None = None
    best_reason = "no_shared_scenario_uid_with_diverging_outcome"
    best_score = -1.0
    for other_condition_id in other_condition_ids:
        other_by_uid = _scenario_uid_index(pools_by_condition[other_condition_id])
        shared_uids = sorted(set(by_uid) & set(other_by_uid))
        for uid in shared_uids:
            this_row = by_uid[uid]
            other_row = other_by_uid[uid]
            this_success = _as_bool(this_row.get("success"))
            other_success = _as_bool(other_row.get("success"))
            if this_success == other_success:
                continue
            this_route = _to_float(this_row.get("route_completion")) or 0.0
            other_route = _to_float(other_row.get("route_completion")) or 0.0
            score = abs(this_route - other_route)
            if best is None or score > best_score:
                best = this_row
                best_score = score
                best_reason = f"diverges_from_condition_{other_condition_id}_on_scenario_uid_{uid}"

    if best is None:
        return None, best_reason
    return best, best_reason


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

    # REQ-014: the four approved categories only. `median` (no direct
    # REQ-014 equivalent) and `curriculum_transition_case` (a curriculum
    # diagnostic, not a shared-scenario_uid cross-condition comparison) are
    # intentionally dropped -- see docs/implementation/
    # evaluation_protocol_v1.0_exec_plan.md Milestone 9 (2026-07-25 entry)
    # for the rationale. `algorithm_disagreement` is a new cross-condition
    # selector (`_pick_disagreement`) comparing the identical `scenario_uid`
    # across conditions within the same comparison, per REQ-014's
    # invariants.
    selection_field_defaults = {
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

    def _selection_fields(selected: dict[str, str] | None) -> dict[str, str]:
        if selected is None:
            return dict(selection_field_defaults)
        return {
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

    pools_by_condition: dict[str, list[dict[str, str]]] = {}
    for condition_id, rows in by_condition_rows.items():
        final_rows = [row for row in rows if str(row.get("eval_type", "")).strip().lower() == "final"]
        pools_by_condition[condition_id] = final_rows if final_rows else rows

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

        pool_rows = pools_by_condition[condition_id]
        used_keys: set[tuple[str, str, str]] = set()

        category_specs = (
            (
                "representative_success",
                lambda: _select_distinct(
                    category="representative_success", candidates=pool_rows, picker=_pick_best, used_episode_keys=used_keys
                ),
            ),
            (
                "representative_failure",
                lambda: _select_distinct(
                    category="representative_failure", candidates=pool_rows, picker=_pick_worst, used_episode_keys=used_keys
                ),
            ),
            (
                "severe_rule_violation",
                lambda: _select_distinct(
                    category="severe_rule_violation",
                    candidates=pool_rows,
                    picker=_pick_rule_violation,
                    used_episode_keys=used_keys,
                ),
            ),
        )

        # `algorithm_disagreement` is resolved separately from the other
        # three categories because its `picker` needs to also surface a
        # disagreement-specific `selection_reason` (which other condition it
        # diverges from), unlike the generic single-row `picker(rows) -> row`
        # interface `_select_distinct` expects for the other categories.
        disagreement_reasons: dict[tuple[str, str, str], str] = {}

        def _disagreement_picker(candidates: list[dict[str, str]]) -> dict[str, str] | None:
            picked, picked_reason = _pick_disagreement(
                condition_id=condition_id,
                rows=candidates,
                pools_by_condition=pools_by_condition,
            )
            if picked is not None:
                disagreement_reasons[_episode_key(picked)] = picked_reason
            return picked

        category_specs = (
            *category_specs,
            (
                "algorithm_disagreement",
                lambda: _select_distinct(
                    category="algorithm_disagreement",
                    candidates=pool_rows,
                    picker=_disagreement_picker,
                    used_episode_keys=used_keys,
                ),
            ),
        )

        for category, resolver in category_specs:
            selected, reason = resolver()
            if category == "algorithm_disagreement" and selected is not None:
                reason = disagreement_reasons.get(_episode_key(selected), reason)
            rows_out.append(
                {
                    **descriptors,
                    "category": category,
                    "selection_status": "selected" if selected is not None else "unavailable",
                    "selection_reason": reason,
                    **_selection_fields(selected),
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
