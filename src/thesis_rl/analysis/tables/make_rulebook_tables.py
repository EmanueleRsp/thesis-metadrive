from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from thesis_rl.analysis.common_stats import mean_ci95, to_float

GLOBAL_METRICS = (
    "avg_error_value",
    "max_error_value",
    "counterexample_rate",
    "violated_rules_ratio",
    "unique_violation_patterns",
)

REQUIRED_FINAL_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "seed",
)

REQUIRED_RULE_COLUMNS = (
    "condition_id",
    "algorithm",
    "rule_name",
    "violation_rate",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def build_rulebook_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    rule_rows = _read_rows(aggregated_dir / "rule_metrics_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Global table from final_eval, grouped by condition.
    global_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_seeds: dict[str, set[str]] = defaultdict(set)
    descriptors: dict[str, dict[str, str]] = {}

    for row in final_rows:
        missing = [c for c in REQUIRED_FINAL_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in final_eval_all_runs.csv row: {missing}")

        condition_id = str(row["condition_id"]).strip()
        global_seeds[condition_id].add(str(row["seed"]).strip())

        desc = {
            "condition_id": condition_id,
            "algorithm": str(row["algorithm"]).strip(),
            "reward_type": str(row["reward_type"]).strip(),
            "reward_behavior": str(row["reward_behavior"]).strip(),
            "curriculum": str(row["curriculum_name"]).strip(),
            "rulebook_config": str(row["rulebook_config"]).strip(),
        }
        prev = descriptors.get(condition_id)
        if prev is None:
            descriptors[condition_id] = desc
        elif prev != desc:
            raise ValueError(f"Inconsistent descriptors for condition_id '{condition_id}'")

        for metric in GLOBAL_METRICS:
            value = to_float(row.get(metric))
            if value is not None:
                global_bucket[condition_id][metric].append(value)

    global_csv = tables_dir / "rulebook_compliance.csv"
    with global_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "n_seeds",
        ] + [f"{m}_mean" for m in GLOBAL_METRICS] + [f"{m}_ci95" for m in GLOBAL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id in sorted(global_bucket.keys()):
            row: dict[str, object] = dict(descriptors[condition_id])
            row["n_seeds"] = len(global_seeds.get(condition_id, set()))
            for metric in GLOBAL_METRICS:
                values = global_bucket[condition_id].get(metric, [])
                if values:
                    m, ci = mean_ci95(values)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_ci95"] = ci
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_ci95"] = ""
            writer.writerow(row)

    global_md = tables_dir / "rulebook_compliance.md"
    with global_md.open("w", encoding="utf-8") as handle:
        handle.write("| Condition | Algorithm | Reward Type | Reward Behavior | Curriculum | Rulebook Config | n | Avg EV | Max EV | CE ratio | Violated rules ratio | Unique patterns |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for condition_id in sorted(global_bucket.keys()):
            d = descriptors[condition_id]
            cells = [
                condition_id,
                d["algorithm"],
                d["reward_type"],
                d["reward_behavior"],
                d["curriculum"],
                d["rulebook_config"],
                str(len(global_seeds.get(condition_id, set()))),
            ]
            for metric in GLOBAL_METRICS:
                values = global_bucket[condition_id].get(metric, [])
                if values:
                    m, ci = mean_ci95(values)
                    cells.append(f"{m:.4f} ± {ci:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    # Per-rule table grouped by (condition_id, rule_name)
    per_rule_bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_rule_desc: dict[tuple[str, str], dict[str, str]] = {}

    for row in rule_rows:
        missing = [c for c in REQUIRED_RULE_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in rule_metrics_all_runs.csv row: {missing}")

        condition_id = str(row["condition_id"]).strip()
        rule = str(row["rule_name"]).strip()
        key = (condition_id, rule)

        per_rule_desc[key] = {
            "condition_id": condition_id,
            "rule_name": rule,
            "algorithm": str(row.get("algorithm", "")).strip(),
            "reward_type": str(row.get("reward_type", "")).strip(),
            "reward_behavior": str(row.get("reward_behavior", "")).strip(),
            "curriculum": str(row.get("curriculum_name", "")).strip(),
            "rulebook_config": str(row.get("rulebook_config", "")).strip(),
            "rule_priority": str(row.get("rule_priority", "")).strip(),
        }

        for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
            value = to_float(row.get(metric))
            if value is not None:
                per_rule_bucket[key][metric].append(value)

    per_rule_csv = tables_dir / "rule_violation_by_rule.csv"
    with per_rule_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "rule_name",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "rule_priority",
            "violation_rate_mean",
            "violation_rate_ci95",
            "mean_margin_mean",
            "mean_margin_ci95",
            "min_margin_mean",
            "min_margin_ci95",
            "max_margin_mean",
            "max_margin_ci95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(per_rule_bucket.keys()):
            row: dict[str, object] = dict(per_rule_desc[key])
            for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[key].get(metric, [])
                if values:
                    m, ci = mean_ci95(values)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_ci95"] = ci
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_ci95"] = ""
            writer.writerow(row)

    per_rule_md = tables_dir / "rule_violation_by_rule.md"
    with per_rule_md.open("w", encoding="utf-8") as handle:
        handle.write("| Condition | Rule | Violation rate | Mean margin | Min margin | Max margin |\n")
        handle.write("| --- | --- | --- | --- | --- | --- |\n")
        for key in sorted(per_rule_bucket.keys()):
            condition_id, rule = key
            cells = [condition_id, rule]
            for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[key].get(metric, [])
                if values:
                    m, ci = mean_ci95(values)
                    cells.append(f"{m:.4f} ± {ci:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {global_csv}")
    print(f"Wrote table MD  -> {global_md}")
    print(f"Wrote table CSV -> {per_rule_csv}")
    print(f"Wrote table MD  -> {per_rule_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rulebook compliance tables by condition.")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_rulebook_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()
