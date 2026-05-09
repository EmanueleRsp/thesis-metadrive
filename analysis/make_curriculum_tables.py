from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analysis.common_stats import mean_ci95, to_float

REQUIRED_FINAL_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "seed",
    "run_id",
    "final_stage_reached",
    "steps_to_final_stage",
)

REQUIRED_EVAL_COLUMNS = (
    "condition_id",
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


def build_curriculum_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    eval_rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_final_stage_reached: dict[tuple[str, str, str], float] = {}
    run_steps_to_final: dict[tuple[str, str, str], float] = {}
    run_failed_evals: dict[tuple[str, str, str], float] = defaultdict(float)
    descriptors: dict[str, dict[str, str]] = {}
    by_condition_seed: dict[str, set[str]] = defaultdict(set)

    for row in final_rows:
        missing = [c for c in REQUIRED_FINAL_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in final_eval_all_runs.csv row: {missing}")

        key = _run_key(row)
        condition_id = key[0]

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

        by_condition_seed[condition_id].add(key[1])

        reached = str(row.get("final_stage_reached", "")).strip().lower() in {"true", "1", "yes"}
        run_final_stage_reached[key] = 1.0 if reached else 0.0

        steps_to_final = to_float(row.get("steps_to_final_stage"))
        if steps_to_final is None:
            raise ValueError(f"Invalid numeric steps_to_final_stage for run key {key}")
        run_steps_to_final[key] = float(steps_to_final)

    for row in eval_rows:
        missing = [c for c in REQUIRED_EVAL_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in evals_all_runs.csv row: {missing}")

        eval_type = str(row.get("eval_type", "")).strip().lower()
        if eval_type == "final":
            continue

        curriculum_enabled = str(row.get("curriculum_enabled", "")).strip().lower() in {"true", "1", "yes"}
        if not curriculum_enabled:
            continue

        if str(row.get("promoted", "")).strip() == "" or str(row.get("passed_eval_gates", "")).strip() == "":
            raise ValueError(
                "Missing required columns/values in evals_all_runs.csv row: ['promoted', 'passed_eval_gates']"
            )

        key = _run_key(row)
        promoted = str(row.get("promoted", "")).strip().lower() in {"true", "1", "yes"}
        passed = str(row.get("passed_eval_gates", "")).strip().lower() in {"true", "1", "yes"}
        if (not promoted) and (not passed):
            run_failed_evals[key] += 1.0

    by_condition_stage: dict[str, list[float]] = defaultdict(list)
    by_condition_steps: dict[str, list[float]] = defaultdict(list)
    by_condition_failed: dict[str, list[float]] = defaultdict(list)

    for key, reached in run_final_stage_reached.items():
        condition_id = key[0]
        by_condition_stage[condition_id].append(reached)
        by_condition_steps[condition_id].append(float(run_steps_to_final[key]))
        by_condition_failed[condition_id].append(float(run_failed_evals.get(key, 0.0)))

    csv_path = tables_dir / "curriculum_efficiency.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
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
        for condition_id in sorted(by_condition_stage.keys()):
            stage_m, stage_ci = mean_ci95(by_condition_stage[condition_id])
            steps_m, steps_ci = mean_ci95(by_condition_steps[condition_id])
            fail_m, fail_ci = mean_ci95(by_condition_failed[condition_id])
            writer.writerow(
                {
                    **descriptors[condition_id],
                    "n_seeds": len(by_condition_seed[condition_id]),
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
        handle.write("| Condition | n | Final stage reached | Steps to final stage | Failed evals before promotion |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for condition_id in sorted(by_condition_stage.keys()):
            stage_m, stage_ci = mean_ci95(by_condition_stage[condition_id])
            steps_m, steps_ci = mean_ci95(by_condition_steps[condition_id])
            fail_m, fail_ci = mean_ci95(by_condition_failed[condition_id])
            handle.write(
                "| "
                + " | ".join(
                    [
                        condition_id,
                        str(len(by_condition_seed[condition_id])),
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
    parser = argparse.ArgumentParser(description="Build curriculum efficiency tables by condition.")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_curriculum_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()
