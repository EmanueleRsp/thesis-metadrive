from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analysis.common_stats import mean_ci95, to_float

GLOBAL_METRICS = (
    "avg_error_value",
    "max_error_value",
    "counterexample_rate",
    "violated_rules_ratio",
    "unique_violation_patterns",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_rulebook_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    rule_rows = _read_rows(aggregated_dir / "rule_metrics_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Global table from final_eval
    global_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_seeds: dict[str, set[str]] = defaultdict(set)
    for row in final_rows:
        algo = str(row.get("algorithm", "")).strip()
        seed = str(row.get("seed", "")).strip()
        if not algo:
            continue
        if seed:
            global_seeds[algo].add(seed)
        for metric in GLOBAL_METRICS:
            value = to_float(row.get(metric))
            if value is not None:
                global_bucket[algo][metric].append(value)

    global_csv = tables_dir / "rulebook_compliance.csv"
    with global_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["algorithm", "n_seeds"] + [f"{m}_mean" for m in GLOBAL_METRICS] + [f"{m}_ci95" for m in GLOBAL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algo in sorted(global_bucket.keys()):
            row: dict[str, object] = {"algorithm": algo, "n_seeds": len(global_seeds.get(algo, set()))}
            for metric in GLOBAL_METRICS:
                values = global_bucket[algo].get(metric, [])
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
        handle.write("| Algorithm | n | Avg EV | Max EV | CE ratio | Violated rules ratio | Unique patterns |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for algo in sorted(global_bucket.keys()):
            cells = [algo, str(len(global_seeds.get(algo, set())))]
            for metric in GLOBAL_METRICS:
                values = global_bucket[algo].get(metric, [])
                if values:
                    m, ci = mean_ci95(values)
                    cells.append(f"{m:.4f} ± {ci:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    # Per-rule table
    per_rule_bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rule_rows:
        algo = str(row.get("algorithm", "")).strip()
        rule = str(row.get("rule_name", "")).strip()
        if not algo or not rule:
            continue
        key = (rule, algo)
        for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
            value = to_float(row.get(metric))
            if value is not None:
                per_rule_bucket[key][metric].append(value)

    per_rule_csv = tables_dir / "rule_violation_by_rule.csv"
    with per_rule_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rule_name",
            "algorithm",
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
        for (rule, algo) in sorted(per_rule_bucket.keys()):
            row: dict[str, object] = {"rule_name": rule, "algorithm": algo}
            for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[(rule, algo)].get(metric, [])
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
        handle.write("| Rule | Algorithm | Violation rate | Mean margin | Min margin | Max margin |\n")
        handle.write("| --- | --- | --- | --- | --- | --- |\n")
        for (rule, algo) in sorted(per_rule_bucket.keys()):
            cells = [rule, algo]
            for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[(rule, algo)].get(metric, [])
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
    parser = argparse.ArgumentParser(description="Build rulebook compliance tables.")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_rulebook_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()

