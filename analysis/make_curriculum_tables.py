from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis.common_stats import mean_ci95, to_float


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("algorithm", "")).strip(),
        str(row.get("seed", "")).strip(),
        str(row.get("run_id", "")).strip(),
    )


def build_curriculum_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    promotions_rows = _read_rows(aggregated_dir / "promotions_all_runs.csv")
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    eval_rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-run stats
    run_final_stage_reached: dict[tuple[str, str, str], float] = {}
    run_steps_to_final: dict[tuple[str, str, str], float] = {}
    run_failed_evals: dict[tuple[str, str, str], float] = defaultdict(float)

    for row in final_rows:
        key = _run_key(row)
        reached = str(row.get("final_stage_reached", "")).strip().lower() in {"true", "1", "yes"}
        run_final_stage_reached[key] = 1.0 if reached else 0.0
        total_steps = to_float(row.get("total_timesteps"))
        run_steps_to_final[key] = float(total_steps) if total_steps is not None else 0.0

    # If promoted rows exist, prefer first promotion to "final" stage as steps_to_final proxy.
    # Without explicit final stage marker in promotions, we keep total_timesteps proxy.
    _ = promotions_rows

    for row in eval_rows:
        key = _run_key(row)
        promoted = str(row.get("promoted", "")).strip().lower() in {"true", "1", "yes"}
        passed = str(row.get("passed_eval_gates", "")).strip().lower() in {"true", "1", "yes"}
        if (not promoted) and (not passed):
            run_failed_evals[key] += 1.0

    # Aggregate per algorithm
    by_algo_stage: dict[str, list[float]] = defaultdict(list)
    by_algo_steps: dict[str, list[float]] = defaultdict(list)
    by_algo_failed: dict[str, list[float]] = defaultdict(list)
    by_algo_seed: dict[str, set[str]] = defaultdict(set)

    for key, reached in run_final_stage_reached.items():
        algo, seed, _run_id = key
        by_algo_stage[algo].append(reached)
        by_algo_steps[algo].append(float(run_steps_to_final.get(key, 0.0)))
        by_algo_failed[algo].append(float(run_failed_evals.get(key, 0.0)))
        by_algo_seed[algo].add(seed)

    csv_path = tables_dir / "curriculum_efficiency.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "algorithm",
            "n_seeds",
            "final_stage_reached_rate_mean",
            "final_stage_reached_rate_ci95",
            "steps_to_final_stage_mean",
            "steps_to_final_stage_ci95",
            "failed_evals_before_promotion_mean",
            "failed_evals_before_promotion_ci95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algo in sorted(by_algo_stage.keys()):
            stage_m, stage_ci = mean_ci95(by_algo_stage[algo])
            steps_m, steps_ci = mean_ci95(by_algo_steps[algo])
            fail_m, fail_ci = mean_ci95(by_algo_failed[algo])
            writer.writerow(
                {
                    "algorithm": algo,
                    "n_seeds": len(by_algo_seed[algo]),
                    "final_stage_reached_rate_mean": stage_m,
                    "final_stage_reached_rate_ci95": stage_ci,
                    "steps_to_final_stage_mean": steps_m,
                    "steps_to_final_stage_ci95": steps_ci,
                    "failed_evals_before_promotion_mean": fail_m,
                    "failed_evals_before_promotion_ci95": fail_ci,
                }
            )

    md_path = tables_dir / "curriculum_efficiency.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Algorithm | n | Final stage reached | Steps to final stage | Failed evals before promotion |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for algo in sorted(by_algo_stage.keys()):
            stage_m, stage_ci = mean_ci95(by_algo_stage[algo])
            steps_m, steps_ci = mean_ci95(by_algo_steps[algo])
            fail_m, fail_ci = mean_ci95(by_algo_failed[algo])
            handle.write(
                "| "
                + " | ".join(
                    [
                        algo,
                        str(len(by_algo_seed[algo])),
                        f"{stage_m:.4f} ± {stage_ci:.4f}",
                        f"{steps_m:.1f} ± {steps_ci:.1f}",
                        f"{fail_m:.3f} ± {fail_ci:.3f}",
                    ]
                )
                + " |\n"
            )

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curriculum efficiency tables.")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_curriculum_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()

