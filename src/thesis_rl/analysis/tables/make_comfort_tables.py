"""EP-COMFORT-DIAG: ride-comfort and jerk diagnostic tables.

Additive diagnostic reporting only (`DEC-CMF-004`). These tables are never a
primary comparison: `RULEBOOK-V5.1` §13 keeps comfort out of the rulebook and
the reward, and `EVAL-PROTOCOL` v1.3 keeps `route_completion` as the primary
metric, so the comfort columns are deliberately kept out of
`final_evaluation.*` and reported here instead. See
`docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md`.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_rl.analysis.common_stats import ci95, mean_sd, to_float
from thesis_rl.common.paths import default_analysis_root_str
from thesis_rl.runtime.comfort_diagnostics import (
    COMFORT_STATISTICS,
    NUPLAN_COMFORT_BOUNDS,
)

DIAGNOSTIC_LABEL = (
    "Diagnostic (EP-COMFORT-DIAG): ride comfort is excluded from the rulebook and the "
    "reward by RULEBOOK-V5.1 §13 and is not a primary comparison metric. Channels "
    "reproduce nuPlan's `ego_is_comfortable`: the published bounds, and the devkit's own "
    "Savitzky-Golay derivative parameters. Two adaptations are documented as DEV-CMF-001 "
    "-- acceleration is differentiated from the simulator's velocity, which MetaDrive "
    "publishes instead of an acceleration, and the series is split at unusable steps so "
    "no filter window spans a gap."
)

REQUIRED_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
)

# `comfort_rate` first: it is the one headline number, the share of episodes
# inside every nuPlan bound. The channel means say which bound is under strain.
COMFORT_METRICS: tuple[str, ...] = ("comfort_rate",) + tuple(
    f"mean_comfort_{name}" for name in COMFORT_STATISTICS
)

# Reported beside the metrics so a reader can see how much of the panel the
# means actually rest on (`REQ-CMF-07`).
COVERAGE_METRICS: tuple[str, ...] = (
    "comfort_episode_count",
    "comfort_excluded_episode_count",
)

_BOUND_BY_METRIC: dict[str, float] = {
    f"mean_comfort_{name}": float(getattr(NUPLAN_COMFORT_BOUNDS, name))
    for name in COMFORT_STATISTICS
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def build_comfort_tables(
    aggregated_dir: Path, tables_dir: Path, *, include_ci: bool = False
) -> None:
    """Read `final_eval_all_runs.csv` and emit the comfort diagnostic table.

    A run set recorded before this feature carries no comfort column at all
    (`DEC-CMF-002` scope, no backfill): that yields a header-only table rather
    than an error, mirroring `build_subrule_tables`.
    """

    rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds_by_condition: dict[str, set[str]] = defaultdict(set)
    descriptors: dict[str, dict[str, str]] = {}

    for row in rows:
        condition_id = str(row.get("condition_id", "")).strip()
        if condition_id == "":
            continue
        descriptors[condition_id] = {key: str(row.get(key, "")).strip() for key in REQUIRED_COLUMNS}
        seeds_by_condition[condition_id].add(str(row.get("seed", "")).strip())
        for metric in COMFORT_METRICS + COVERAGE_METRICS:
            value = to_float(row.get(metric))
            if value is not None:
                grouped[condition_id][metric].append(value)

    # A condition present only with empty comfort cells is dropped rather than
    # reported as a row of blanks, so the table's row count is the number of
    # conditions actually measured.
    measured = [
        condition_id
        for condition_id in sorted(descriptors)
        if any(grouped[condition_id].get(metric) for metric in COMFORT_METRICS)
    ]

    csv_path = tables_dir / "comfort_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            list(REQUIRED_COLUMNS)
            + ["n_seeds"]
            + [f"{m}_mean" for m in COMFORT_METRICS]
            + [f"{m}_sd" for m in COMFORT_METRICS]
            + [f"{m}_seed_values" for m in COMFORT_METRICS]
            + [f"{m}_mean" for m in COVERAGE_METRICS]
        )
        if include_ci:
            fieldnames += [f"{m}_ci95" for m in COMFORT_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id in measured:
            out: dict[str, Any] = dict(descriptors[condition_id])
            n_seeds = len(seeds_by_condition[condition_id])
            out["n_seeds"] = n_seeds
            for metric in COMFORT_METRICS:
                values = grouped[condition_id].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    out[f"{metric}_mean"] = m
                    out[f"{metric}_sd"] = s
                    out[f"{metric}_seed_values"] = ";".join(str(v) for v in values)
                    if include_ci:
                        out[f"{metric}_ci95"] = ci95(s, n_seeds)
                else:
                    out[f"{metric}_mean"] = ""
                    out[f"{metric}_sd"] = ""
                    out[f"{metric}_seed_values"] = ""
                    if include_ci:
                        out[f"{metric}_ci95"] = ""
            for metric in COVERAGE_METRICS:
                values = grouped[condition_id].get(metric, [])
                out[f"{metric}_mean"] = mean_sd(values)[0] if values else ""
            writer.writerow(out)

    md_path = tables_dir / "comfort_diagnostics.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"_{DIAGNOSTIC_LABEL}_\n\n")
        headers = ["Condition", "Algorithm", "Reward", "Curriculum", "n"] + [
            _md_header(metric) for metric in COMFORT_METRICS
        ]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for condition_id in measured:
            desc = descriptors[condition_id]
            cells = [
                condition_id,
                desc["algorithm"],
                f"{desc['reward_type']}/{desc['reward_behavior']}",
                desc["curriculum_name"],
                str(len(seeds_by_condition[condition_id])),
            ]
            for metric in COMFORT_METRICS:
                values = grouped[condition_id].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    cells.append(f"{m:.4f} ± {s:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {csv_path}")
    print(f"Wrote table MD  -> {md_path}")


def _md_header(metric: str) -> str:
    """Label a column with the bound it is measured against, where there is one."""
    if metric == "comfort_rate":
        return "Comfort rate"
    bound = _BOUND_BY_METRIC[metric]
    return f"{metric.removeprefix('mean_comfort_')} (bound {bound:g})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build ride-comfort diagnostic tables by condition. Diagnostic only: "
            "comfort is excluded from the rulebook and the reward (RULEBOOK-V5.1 §13)."
        )
    )
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Also emit an optional 1.96*sd/sqrt(n) CI95 column (off by default, REQ-009/DEC-003).",
    )
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_comfort_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
        include_ci=bool(args.include_ci),
    )


if __name__ == "__main__":
    main()
