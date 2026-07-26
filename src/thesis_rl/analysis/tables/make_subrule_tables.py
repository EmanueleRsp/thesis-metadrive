"""EP-SUBRULE-DIAG: R2/R3 sub-rule dominance and cost diagnostic tables.

Additive diagnostic reporting only (`DEC-SUB-001`); these tables never enter
a primary comparison and must not be read as replacing
`rulebook_compliance.*` / `rule_violation_by_rule.*`
(`docs/implementation/subrule_dominance_diagnostics_exec_plan.md`).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from thesis_rl.analysis.common_stats import mean_sd, to_float
from thesis_rl.common.paths import default_analysis_root_str

DIAGNOSTIC_LABEL = (
    "Diagnostic (EP-SUBRULE-DIAG): additive R2/R3 reporting, not a primary comparison metric."
)

REQUIRED_COLUMNS = (
    "condition_id",
    "algorithm",
    "scenario_source",
    "macro_rule",
    "subrule_name",
)

_SUBRULE_METRICS = (
    "applicability_rate",
    "violation_rate",
    "mean_cost",
    "max_cost",
    "applicable_episode_count",
    "excluded_episode_count",
)

_DOMINANCE_METRICS = (
    "dominance_share",
    "worst_component_count",
    "macro_violated_step_count",
    "multi_violation_share",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def build_subrule_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    """Read `subrule_metrics_all_runs.csv` and emit the two diagnostic tables.

    A run set without any `subrule_metrics.csv` (recorded before this
    feature, `DEC-SUB-003`) yields header-only tables, not an error.
    """

    rows = _read_rows(aggregated_dir / "subrule_metrics_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    key_fn = lambda row: (  # noqa: E731
        str(row.get("condition_id", "")).strip(),
        str(row.get("scenario_source", "")).strip(),
        str(row.get("macro_rule", "")).strip(),
        str(row.get("subrule_name", "")).strip(),
    )

    by_subrule_bucket: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    descriptors: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        missing = [c for c in REQUIRED_COLUMNS if str(row.get(c, "")).strip() == ""]
        if missing:
            raise ValueError(
                f"Missing required columns/values in subrule_metrics_all_runs.csv row: {missing}"
            )
        key = key_fn(row)
        descriptors[key] = {
            "condition_id": key[0],
            "algorithm": str(row.get("algorithm", "")).strip(),
            "scenario_source": key[1],
            "macro_rule": key[2],
            "subrule_name": key[3],
        }
        for metric in _SUBRULE_METRICS + _DOMINANCE_METRICS:
            value = to_float(row.get(metric))
            if value is not None:
                by_subrule_bucket[key][metric].append(value)

    liveness_csv = tables_dir / "subrule_metrics_by_subrule.csv"
    with liveness_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            list(REQUIRED_COLUMNS)
            + [f"{m}_mean" for m in _SUBRULE_METRICS]
            + [f"{m}_sd" for m in _SUBRULE_METRICS]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(by_subrule_bucket.keys()):
            out: dict[str, object] = dict(descriptors[key])
            for metric in _SUBRULE_METRICS:
                values = by_subrule_bucket[key].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    out[f"{metric}_mean"] = m
                    out[f"{metric}_sd"] = s
                else:
                    out[f"{metric}_mean"] = ""
                    out[f"{metric}_sd"] = ""
            writer.writerow(out)

    liveness_md = tables_dir / "subrule_metrics_by_subrule.md"
    with liveness_md.open("w", encoding="utf-8") as handle:
        handle.write(f"_{DIAGNOSTIC_LABEL}_\n\n")
        handle.write(
            "| Condition | Source | Macro rule | Sub-rule | Applicability | "
            "Violation rate | Mean cost | Max cost |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for key in sorted(by_subrule_bucket.keys()):
            condition_id, source, macro_rule, subrule_name = key
            cells = [condition_id, source, macro_rule, subrule_name]
            for metric in ("applicability_rate", "violation_rate", "mean_cost", "max_cost"):
                values = by_subrule_bucket[key].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    cells.append(f"{m:.4f} ± {s:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    dominance_csv = tables_dir / "subrule_dominance_within_macro.csv"
    with dominance_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            list(REQUIRED_COLUMNS)
            + [f"{m}_mean" for m in _DOMINANCE_METRICS]
            + [f"{m}_sd" for m in _DOMINANCE_METRICS]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(by_subrule_bucket.keys()):
            out: dict[str, object] = dict(descriptors[key])
            for metric in _DOMINANCE_METRICS:
                values = by_subrule_bucket[key].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    out[f"{metric}_mean"] = m
                    out[f"{metric}_sd"] = s
                else:
                    out[f"{metric}_mean"] = ""
                    out[f"{metric}_sd"] = ""
            writer.writerow(out)

    dominance_md = tables_dir / "subrule_dominance_within_macro.md"
    with dominance_md.open("w", encoding="utf-8") as handle:
        handle.write(f"_{DIAGNOSTIC_LABEL}_\n\n")
        handle.write(
            "| Condition | Source | Macro rule | Sub-rule | Dominance share | "
            "Macro-violated steps | Multi-violation share |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for key in sorted(by_subrule_bucket.keys()):
            condition_id, source, macro_rule, subrule_name = key
            cells = [condition_id, source, macro_rule, subrule_name]
            values = by_subrule_bucket[key].get("dominance_share", [])
            cells.append(f"{mean_sd(values)[0]:.4f} ± {mean_sd(values)[1]:.4f}" if values else "")
            counts = by_subrule_bucket[key].get("macro_violated_step_count", [])
            cells.append(f"{mean_sd(counts)[0]:.1f}" if counts else "")
            multi = by_subrule_bucket[key].get("multi_violation_share", [])
            cells.append(f"{mean_sd(multi)[0]:.4f} ± {mean_sd(multi)[1]:.4f}" if multi else "")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {liveness_csv}")
    print(f"Wrote table MD  -> {liveness_md}")
    print(f"Wrote table CSV -> {dominance_csv}")
    print(f"Wrote table MD  -> {dominance_md}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build R2/R3 sub-rule dominance and cost diagnostic tables (EP-SUBRULE-DIAG)."
    )
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_subrule_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()
