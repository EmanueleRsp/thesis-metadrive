from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_rl.analysis.common_stats import ci95
from thesis_rl.common.paths import default_analysis_root_str

FINAL_METRICS = (
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "top_rule_violation_rate",
    "route_completion",
    "mean_reward",
    "avg_error_value",
    "max_error_value",
    "counterexample_rate",
)

REQUIRED_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "eval_type",
    "scenario_set",
    "seed",
)


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    return math.sqrt(sum((x - mean_value) ** 2 for x in values) / (len(values) - 1))


def build_final_tables(
    aggregated_dir: Path, tables_dir: Path, *, include_ci: bool = False
) -> None:
    source = aggregated_dir / "final_eval_all_runs.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing aggregated file: {source}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_condition_seed: dict[str, set[str]] = defaultdict(set)
    descriptors: dict[str, dict[str, str]] = {}

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing required columns in {source}: {missing_cols}")

        for row in reader:
            condition_id = str(row.get("condition_id", "")).strip()
            if condition_id == "":
                raise ValueError(f"Empty condition_id in {source}")

            for col in REQUIRED_COLUMNS:
                if str(row.get(col, "")).strip() == "":
                    raise ValueError(f"Empty required field '{col}' for condition '{condition_id}' in {source}")

            seed = str(row.get("seed", "")).strip()
            by_condition_seed[condition_id].add(seed)

            desc = {
                "condition_id": condition_id,
                "algorithm": str(row["algorithm"]).strip(),
                "reward_type": str(row["reward_type"]).strip(),
                "reward_behavior": str(row["reward_behavior"]).strip(),
                "curriculum": str(row["curriculum_name"]).strip(),
                "rulebook_config": str(row["rulebook_config"]).strip(),
                "eval_type": str(row["eval_type"]).strip(),
                "scenario_set": str(row["scenario_set"]).strip(),
            }
            previous = descriptors.get(condition_id)
            if previous is None:
                descriptors[condition_id] = desc
            elif previous != desc:
                raise ValueError(f"Inconsistent descriptors for condition_id '{condition_id}' in {source}")

            for metric in FINAL_METRICS:
                value = _to_float(row.get(metric))
                if value is not None:
                    grouped[condition_id][metric].append(value)

    csv_path = tables_dir / "final_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "eval_type",
            "scenario_set",
            "n_seeds",
        ] + [f"{m}_mean" for m in FINAL_METRICS] + [f"{m}_std" for m in FINAL_METRICS] + [
            f"{m}_seed_values" for m in FINAL_METRICS
        ]
        if include_ci:
            fieldnames += [f"{m}_ci95" for m in FINAL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id in sorted(grouped.keys()):
            row: dict[str, Any] = dict(descriptors[condition_id])
            n_seeds = len(by_condition_seed.get(condition_id, set()))
            row["n_seeds"] = n_seeds
            for metric in FINAL_METRICS:
                values = grouped[condition_id].get(metric, [])
                if values:
                    m = _mean(values)
                    s = _sample_std(values, m)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_std"] = s
                    row[f"{metric}_seed_values"] = ";".join(str(v) for v in values)
                    if include_ci:
                        row[f"{metric}_ci95"] = ci95(s, n_seeds)
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_std"] = ""
                    row[f"{metric}_seed_values"] = ""
                    if include_ci:
                        row[f"{metric}_ci95"] = ""
            writer.writerow(row)

    md_path = tables_dir / "final_evaluation.md"
    with md_path.open("w", encoding="utf-8") as handle:
        headers = [
            "Condition",
            "Algorithm",
            "Reward Type",
            "Reward Behavior",
            "Curriculum",
            "Rulebook Config",
            "Eval Type",
            "Scenario Set",
            "n",
        ] + [metric for metric in FINAL_METRICS]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for condition_id in sorted(grouped.keys()):
            d = descriptors[condition_id]
            cells = [
                condition_id,
                d["algorithm"],
                d["reward_type"],
                d["reward_behavior"],
                d["curriculum"],
                d["rulebook_config"],
                d["eval_type"],
                d["scenario_set"],
                str(len(by_condition_seed.get(condition_id, set()))),
            ]
            for metric in FINAL_METRICS:
                values = grouped[condition_id].get(metric, [])
                if values:
                    m = _mean(values)
                    s = _sample_std(values, m)
                    cells.append(f"{m:.4f} (SD {s:.4f}, n={len(values)})")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final evaluation tables (raw seed values, mean, sample SD; "
        "no confidence interval, per EVAL-PROTOCOL v1.0 REQ-009) by condition."
    )
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Also emit an optional 1.96*sd/sqrt(n) CI95 column (off by default, REQ-009/DEC-003).",
    )
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_final_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
        include_ci=bool(args.include_ci),
    )


if __name__ == "__main__":
    main()
