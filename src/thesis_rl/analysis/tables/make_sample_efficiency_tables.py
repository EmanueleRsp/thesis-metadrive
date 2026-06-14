from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from thesis_rl.analysis.common_stats import mean_ci95, to_float
from thesis_rl.common.paths import default_analysis_root_str

REQUIRED_COLUMNS = (
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
    "success_rate",
    "collision_rate",
    "route_completion",
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


def build_sample_efficiency_tables(
    *,
    aggregated_dir: Path,
    tables_dir: Path,
    success_threshold: float,
    collision_threshold: float,
    route_completion_threshold: float,
) -> None:
    rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    descriptors: dict[str, dict[str, str]] = {}
    run_keys_by_condition: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    step_to_success: dict[tuple[str, str, str], int] = {}
    step_to_collision: dict[tuple[str, str, str], int] = {}
    step_to_route: dict[tuple[str, str, str], int] = {}

    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(f"Missing required columns/values in evals_all_runs.csv row: {missing}")

        # Sample efficiency should track learning-time evaluations.
        if str(row["eval_type"]).strip().lower() != "intermediate":
            continue

        key = _run_key(row)
        condition_id = key[0]
        run_keys_by_condition[condition_id].add(key)

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

        step = to_float(row.get("global_step"))
        success = to_float(row.get("success_rate"))
        collision = to_float(row.get("collision_rate"))
        route = to_float(row.get("route_completion"))
        if step is None or success is None or collision is None or route is None:
            continue
        step_int = int(step)

        if success >= success_threshold and key not in step_to_success:
            step_to_success[key] = step_int
        if collision <= collision_threshold and key not in step_to_collision:
            step_to_collision[key] = step_int
        if route >= route_completion_threshold and key not in step_to_route:
            step_to_route[key] = step_int

    csv_path = tables_dir / "sample_efficiency_thresholds.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "n_runs",
            "success_threshold",
            "collision_threshold",
            "route_completion_threshold",
            "success_met_runs",
            "success_met_rate",
            "steps_to_success_mean",
            "steps_to_success_ci95",
            "collision_met_runs",
            "collision_met_rate",
            "steps_to_collision_mean",
            "steps_to_collision_ci95",
            "route_met_runs",
            "route_met_rate",
            "steps_to_route_mean",
            "steps_to_route_ci95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for condition_id in sorted(run_keys_by_condition.keys()):
            run_keys = sorted(run_keys_by_condition[condition_id])
            n_runs = len(run_keys)

            success_steps = [float(step_to_success[key]) for key in run_keys if key in step_to_success]
            collision_steps = [float(step_to_collision[key]) for key in run_keys if key in step_to_collision]
            route_steps = [float(step_to_route[key]) for key in run_keys if key in step_to_route]

            row_out: dict[str, object] = {
                **descriptors[condition_id],
                "n_runs": n_runs,
                "success_threshold": success_threshold,
                "collision_threshold": collision_threshold,
                "route_completion_threshold": route_completion_threshold,
                "success_met_runs": len(success_steps),
                "success_met_rate": (len(success_steps) / n_runs) if n_runs > 0 else 0.0,
                "collision_met_runs": len(collision_steps),
                "collision_met_rate": (len(collision_steps) / n_runs) if n_runs > 0 else 0.0,
                "route_met_runs": len(route_steps),
                "route_met_rate": (len(route_steps) / n_runs) if n_runs > 0 else 0.0,
            }

            if success_steps:
                m, ci = mean_ci95(success_steps)
                row_out["steps_to_success_mean"] = m
                row_out["steps_to_success_ci95"] = ci
            else:
                row_out["steps_to_success_mean"] = ""
                row_out["steps_to_success_ci95"] = ""

            if collision_steps:
                m, ci = mean_ci95(collision_steps)
                row_out["steps_to_collision_mean"] = m
                row_out["steps_to_collision_ci95"] = ci
            else:
                row_out["steps_to_collision_mean"] = ""
                row_out["steps_to_collision_ci95"] = ""

            if route_steps:
                m, ci = mean_ci95(route_steps)
                row_out["steps_to_route_mean"] = m
                row_out["steps_to_route_ci95"] = ci
            else:
                row_out["steps_to_route_mean"] = ""
                row_out["steps_to_route_ci95"] = ""

            writer.writerow(row_out)

    md_path = tables_dir / "sample_efficiency_thresholds.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| Condition | n | Success met | Steps to success | Collision met | Steps to collision | Route met | Steps to route |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for condition_id in sorted(run_keys_by_condition.keys()):
            run_keys = sorted(run_keys_by_condition[condition_id])
            n_runs = len(run_keys)
            success_steps = [float(step_to_success[key]) for key in run_keys if key in step_to_success]
            collision_steps = [float(step_to_collision[key]) for key in run_keys if key in step_to_collision]
            route_steps = [float(step_to_route[key]) for key in run_keys if key in step_to_route]

            if success_steps:
                success_m, success_ci = mean_ci95(success_steps)
                success_cell = f"{success_m:.1f} ± {success_ci:.1f}"
            else:
                success_cell = "-"

            if collision_steps:
                collision_m, collision_ci = mean_ci95(collision_steps)
                collision_cell = f"{collision_m:.1f} ± {collision_ci:.1f}"
            else:
                collision_cell = "-"

            if route_steps:
                route_m, route_ci = mean_ci95(route_steps)
                route_cell = f"{route_m:.1f} ± {route_ci:.1f}"
            else:
                route_cell = "-"

            handle.write(
                "| "
                + " | ".join(
                    [
                        condition_id,
                        str(n_runs),
                        f"{len(success_steps)}/{n_runs}",
                        success_cell,
                        f"{len(collision_steps)}/{n_runs}",
                        collision_cell,
                        f"{len(route_steps)}/{n_runs}",
                        route_cell,
                    ]
                )
                + " |\n"
            )

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample-efficiency threshold tables by condition.")
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument("--success-threshold", type=float, default=0.70)
    parser.add_argument("--collision-threshold", type=float, default=0.20)
    parser.add_argument("--route-completion-threshold", type=float, default=0.80)
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_sample_efficiency_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
        success_threshold=float(args.success_threshold),
        collision_threshold=float(args.collision_threshold),
        route_completion_threshold=float(args.route_completion_threshold),
    )


if __name__ == "__main__":
    main()
