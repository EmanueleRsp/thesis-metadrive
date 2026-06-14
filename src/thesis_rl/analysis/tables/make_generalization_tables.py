from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_rl.analysis.common_stats import mean_ci95, to_float
from thesis_rl.common.paths import default_analysis_root_str

METRICS = (
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "route_completion",
    "top_rule_violation_rate",
)

REQUIRED_EVAL_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "seed",
    "run_id",
    "eval_type",
    "global_step",
)

REQUIRED_FINAL_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "seed",
    "run_id",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _run_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("condition_id", "")).strip(),
        str(row.get("seed", "")).strip(),
        str(row.get("run_id", "")).strip(),
    )


def _descriptor(row: dict[str, str]) -> dict[str, str]:
    return {
        "condition_id": str(row.get("condition_id", "")).strip(),
        "algorithm": str(row.get("algorithm", "")).strip(),
        "reward_type": str(row.get("reward_type", "")).strip(),
        "reward_behavior": str(row.get("reward_behavior", "")).strip(),
        "curriculum": str(row.get("curriculum_name", "")).strip(),
        "rulebook_config": str(row.get("rulebook_config", "")).strip(),
    }


def build_generalization_tables(*, aggregated_dir: Path, tables_dir: Path) -> None:
    eval_rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    descriptors: dict[str, dict[str, str]] = {}
    train_by_run: dict[tuple[str, str, str], dict[str, float]] = {}
    train_step_by_run: dict[tuple[str, str, str], int] = {}
    eval_by_run: dict[tuple[str, str, str], dict[str, float]] = {}
    run_keys_by_condition: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

    for row in eval_rows:
        missing = [c for c in REQUIRED_EVAL_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in evals_all_runs.csv row: {missing}")

        if str(row.get("eval_type", "")).strip().lower() != "intermediate":
            continue

        key = _run_key(row)
        condition_id = key[0]
        run_keys_by_condition[condition_id].add(key)

        current_desc = _descriptor(row)
        prev_desc = descriptors.get(condition_id)
        if prev_desc is None:
            descriptors[condition_id] = current_desc
        elif prev_desc != current_desc:
            raise ValueError(f"Inconsistent descriptors for condition_id '{condition_id}'")

        step = to_float(row.get("global_step"))
        if step is None:
            continue
        step_int = int(step)
        prev_step = train_step_by_run.get(key)
        if prev_step is not None and step_int <= prev_step:
            continue

        values: dict[str, float] = {}
        for metric in METRICS:
            metric_value = to_float(row.get(metric))
            if metric_value is not None:
                values[metric] = metric_value
        if not values:
            continue

        train_step_by_run[key] = step_int
        train_by_run[key] = values

    for row in final_rows:
        missing = [c for c in REQUIRED_FINAL_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in final_eval_all_runs.csv row: {missing}")

        key = _run_key(row)
        condition_id = key[0]
        run_keys_by_condition[condition_id].add(key)

        current_desc = _descriptor(row)
        prev_desc = descriptors.get(condition_id)
        if prev_desc is None:
            descriptors[condition_id] = current_desc
        elif prev_desc != current_desc:
            raise ValueError(f"Inconsistent descriptors for condition_id '{condition_id}'")

        values: dict[str, float] = {}
        for metric in METRICS:
            metric_value = to_float(row.get(metric))
            if metric_value is not None:
                values[metric] = metric_value
        if values:
            eval_by_run[key] = values

    csv_path = tables_dir / "generalization_train_vs_eval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "n_runs_total",
            "n_runs_train",
            "n_runs_eval",
            "n_runs_gap",
        ]
        for metric in METRICS:
            fieldnames.extend(
                [
                    f"{metric}_train_mean",
                    f"{metric}_train_ci95",
                    f"{metric}_eval_mean",
                    f"{metric}_eval_ci95",
                    f"{metric}_gap_eval_minus_train_mean",
                    f"{metric}_gap_eval_minus_train_ci95",
                ]
            )

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for condition_id in sorted(run_keys_by_condition.keys()):
            run_keys = sorted(run_keys_by_condition[condition_id])
            row_out: dict[str, Any] = dict(descriptors[condition_id])
            row_out["n_runs_total"] = len(run_keys)
            row_out["n_runs_train"] = sum(1 for key in run_keys if key in train_by_run)
            row_out["n_runs_eval"] = sum(1 for key in run_keys if key in eval_by_run)
            row_out["n_runs_gap"] = sum(1 for key in run_keys if key in train_by_run and key in eval_by_run)

            for metric in METRICS:
                train_values = [train_by_run[key][metric] for key in run_keys if key in train_by_run and metric in train_by_run[key]]
                eval_values = [eval_by_run[key][metric] for key in run_keys if key in eval_by_run and metric in eval_by_run[key]]
                gap_values = [
                    eval_by_run[key][metric] - train_by_run[key][metric]
                    for key in run_keys
                    if key in train_by_run and key in eval_by_run and metric in train_by_run[key] and metric in eval_by_run[key]
                ]

                if train_values:
                    train_mean, train_ci = mean_ci95(train_values)
                    row_out[f"{metric}_train_mean"] = train_mean
                    row_out[f"{metric}_train_ci95"] = train_ci
                else:
                    row_out[f"{metric}_train_mean"] = ""
                    row_out[f"{metric}_train_ci95"] = ""

                if eval_values:
                    eval_mean, eval_ci = mean_ci95(eval_values)
                    row_out[f"{metric}_eval_mean"] = eval_mean
                    row_out[f"{metric}_eval_ci95"] = eval_ci
                else:
                    row_out[f"{metric}_eval_mean"] = ""
                    row_out[f"{metric}_eval_ci95"] = ""

                if gap_values:
                    gap_mean, gap_ci = mean_ci95(gap_values)
                    row_out[f"{metric}_gap_eval_minus_train_mean"] = gap_mean
                    row_out[f"{metric}_gap_eval_minus_train_ci95"] = gap_ci
                else:
                    row_out[f"{metric}_gap_eval_minus_train_mean"] = ""
                    row_out[f"{metric}_gap_eval_minus_train_ci95"] = ""

            writer.writerow(row_out)

    md_path = tables_dir / "generalization_train_vs_eval.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| Condition | n(train/eval/gap) | Success (train/eval/gap) | Collision (train/eval/gap) | Route completion (train/eval/gap) |\n"
        )
        handle.write("| --- | --- | --- | --- | --- |\n")
        for condition_id in sorted(run_keys_by_condition.keys()):
            run_keys = sorted(run_keys_by_condition[condition_id])
            n_train = sum(1 for key in run_keys if key in train_by_run)
            n_eval = sum(1 for key in run_keys if key in eval_by_run)
            n_gap = sum(1 for key in run_keys if key in train_by_run and key in eval_by_run)

            def _cell(metric: str) -> str:
                train_values = [train_by_run[key][metric] for key in run_keys if key in train_by_run and metric in train_by_run[key]]
                eval_values = [eval_by_run[key][metric] for key in run_keys if key in eval_by_run and metric in eval_by_run[key]]
                gap_values = [
                    eval_by_run[key][metric] - train_by_run[key][metric]
                    for key in run_keys
                    if key in train_by_run and key in eval_by_run and metric in train_by_run[key] and metric in eval_by_run[key]
                ]
                if not train_values or not eval_values or not gap_values:
                    return "-"
                train_mean, train_ci = mean_ci95(train_values)
                eval_mean, eval_ci = mean_ci95(eval_values)
                gap_mean, gap_ci = mean_ci95(gap_values)
                return (
                    f"{train_mean:.4f}±{train_ci:.4f} / "
                    f"{eval_mean:.4f}±{eval_ci:.4f} / "
                    f"{gap_mean:.4f}±{gap_ci:.4f}"
                )

            handle.write(
                "| "
                + " | ".join(
                    [
                        condition_id,
                        f"{n_train}/{n_eval}/{n_gap}",
                        _cell("success_rate"),
                        _cell("collision_rate"),
                        _cell("route_completion"),
                    ]
                )
                + " |\n"
            )

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-vs-eval generalization gap tables by condition.")
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_generalization_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()
