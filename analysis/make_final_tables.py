from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    m = _mean(values)
    s = _sample_std(values, m)
    if len(values) <= 1:
        return m, 0.0
    ci = 1.96 * (s / math.sqrt(len(values)))
    return m, ci


def build_final_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    source = aggregated_dir / "final_eval_all_runs.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing aggregated file: {source}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_algo_n: dict[str, set[str]] = defaultdict(set)

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            algo = str(row.get("algorithm", "")).strip()
            seed = str(row.get("seed", "")).strip()
            if not algo:
                continue
            if seed:
                by_algo_n[algo].add(seed)
            for metric in FINAL_METRICS:
                value = _to_float(row.get(metric))
                if value is not None:
                    grouped[algo][metric].append(value)

    csv_path = tables_dir / "final_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["algorithm", "n_seeds"] + [f"{m}_mean" for m in FINAL_METRICS] + [f"{m}_ci95" for m in FINAL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algo in sorted(grouped.keys()):
            row: dict[str, Any] = {"algorithm": algo, "n_seeds": len(by_algo_n.get(algo, set()))}
            for metric in FINAL_METRICS:
                values = grouped[algo].get(metric, [])
                if values:
                    m, ci = _mean_ci95(values)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_ci95"] = ci
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_ci95"] = ""
            writer.writerow(row)

    md_path = tables_dir / "final_evaluation.md"
    with md_path.open("w", encoding="utf-8") as handle:
        headers = ["Algorithm", "n"] + [metric for metric in FINAL_METRICS]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for algo in sorted(grouped.keys()):
            cells = [algo, str(len(by_algo_n.get(algo, set())))]
            for metric in FINAL_METRICS:
                values = grouped[algo].get(metric, [])
                if values:
                    m, ci = _mean_ci95(values)
                    cells.append(f"{m:.4f} ± {ci:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final evaluation tables (mean ± 95% CI).")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_final_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()

