from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "route_completion",
    "top_rule_violation_rate",
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
    "curriculum_enabled",
    "rulebook_config",
    "seed",
    "eval_type",
    "scenario_set",
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


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    mean_v = sum(values) / len(values)
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((x - mean_v) ** 2 for x in values) / (len(values) - 1)
    ci = 1.96 * math.sqrt(var / len(values))
    return mean_v, ci


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        rows = list(reader)
    for row in rows:
        empty = [c for c in REQUIRED_COLUMNS if str(row.get(c, "")).strip() == ""]
        if empty:
            raise ValueError(f"Missing required values in row for {path}: {empty}")
    return rows


def _write_effect_table(
    rows: list[dict[str, object]],
    output_csv: Path,
    output_md: Path,
    group_cols: list[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_columns: list[str] = []
    for m in METRICS:
        metric_columns.extend([f"{m}_delta_mean", f"{m}_delta_ci95"])

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = group_cols + ["n_pairs"] + metric_columns
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_md.open("w", encoding="utf-8") as handle:
        headers = group_cols + ["n_pairs"] + [f"{m} (delta)" for m in METRICS]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            cells = [str(row[c]) for c in group_cols] + [str(row["n_pairs"])]
            for m in METRICS:
                mean_key = f"{m}_delta_mean"
                ci_key = f"{m}_delta_ci95"
                cells.append(f"{float(row[mean_key]):.4f} ± {float(row[ci_key]):.4f}")
            handle.write("| " + " | ".join(cells) + " |\n")


def _aggregate_pair_deltas(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    group_key_fn,
    group_cols: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for a_row, b_row in pairs:
        key = group_key_fn(a_row, b_row)
        counts[key] += 1
        for metric in METRICS:
            a_val = _to_float(a_row.get(metric))
            b_val = _to_float(b_row.get(metric))
            if a_val is None or b_val is None:
                continue
            grouped[key][metric].append(b_val - a_val)

    out: list[dict[str, object]] = []
    for key in sorted(grouped.keys()):
        row: dict[str, object] = {col: key[i] for i, col in enumerate(group_cols)}
        row["n_pairs"] = counts[key]
        for metric in METRICS:
            vals = grouped[key].get(metric, [])
            if not vals:
                row[f"{metric}_delta_mean"] = 0.0
                row[f"{metric}_delta_ci95"] = 0.0
                continue
            m, ci = _mean_ci95(vals)
            row[f"{metric}_delta_mean"] = m
            row[f"{metric}_delta_ci95"] = ci
        out.append(row)
    return out


def _pairs_curriculum_effect(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    idx: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if r["eval_type"] != "final":
            continue
        key = (
            r["algorithm"],
            r["reward_type"],
            r["reward_behavior"],
            r["rulebook_config"],
            r["scenario_set"],
            r["seed"],
            r["curriculum_name"],
        )
        idx[key][r["curriculum_enabled"].lower()] = r
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for rec in idx.values():
        if "false" in rec and "true" in rec:
            pairs.append((rec["false"], rec["true"]))
    return pairs


def _pairs_rulebook_reward_effect(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    idx: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if r["eval_type"] != "final":
            continue
        key = (r["algorithm"], r["curriculum_enabled"], r["rulebook_config"], r["scenario_set"], r["seed"])
        idx[key][r["reward_behavior"]] = r
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for rec in idx.values():
        if "monitor_only" in rec and "scalar_reward" in rec:
            pairs.append((rec["monitor_only"], rec["scalar_reward"]))
    return pairs


def _pairs_rulebook_variant_effect(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    idx: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["eval_type"] != "final":
            continue
        key = (
            r["algorithm"],
            r["curriculum_enabled"],
            r["reward_behavior"],
            r["reward_type"],
            r["scenario_set"],
            r["seed"],
        )
        idx[key].append(r)

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for arr in idx.values():
        arr_sorted = sorted(arr, key=lambda x: x["rulebook_config"])
        for i in range(len(arr_sorted) - 1):
            pairs.append((arr_sorted[i], arr_sorted[i + 1]))
    return pairs


def _pairs_algorithm_effect(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    idx: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["eval_type"] != "final":
            continue
        key = (r["reward_type"], r["reward_behavior"], r["curriculum_enabled"], r["rulebook_config"], r["seed"])
        idx[key].append(r)
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for arr in idx.values():
        arr_sorted = sorted(arr, key=lambda x: x["algorithm"])
        for i in range(len(arr_sorted) - 1):
            pairs.append((arr_sorted[i], arr_sorted[i + 1]))
    return pairs


def build_factor_effect_tables(aggregated_dir: Path, tables_dir: Path) -> None:
    rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)

    curr_pairs = _pairs_curriculum_effect(rows)
    curr_rows = _aggregate_pair_deltas(
        curr_pairs,
        group_key_fn=lambda a, b: (
            a["algorithm"],
            a["reward_type"],
            a["reward_behavior"],
            a["rulebook_config"],
            a["scenario_set"],
        ),
        group_cols=["algorithm", "reward_type", "reward_behavior", "rulebook_config", "scenario_set"],
    )
    _write_effect_table(
        curr_rows,
        tables_dir / "effects_curriculum.csv",
        tables_dir / "effects_curriculum.md",
        ["algorithm", "reward_type", "reward_behavior", "rulebook_config", "scenario_set"],
    )

    rb_reward_pairs = _pairs_rulebook_reward_effect(rows)
    rb_reward_rows = _aggregate_pair_deltas(
        rb_reward_pairs,
        group_key_fn=lambda a, b: (
            a["algorithm"],
            a["curriculum_enabled"],
            a["rulebook_config"],
            a["scenario_set"],
        ),
        group_cols=["algorithm", "curriculum_enabled", "rulebook_config", "scenario_set"],
    )
    _write_effect_table(
        rb_reward_rows,
        tables_dir / "effects_rulebook_reward.csv",
        tables_dir / "effects_rulebook_reward.md",
        ["algorithm", "curriculum_enabled", "rulebook_config", "scenario_set"],
    )

    rb_variant_pairs = _pairs_rulebook_variant_effect(rows)
    rb_variant_rows = _aggregate_pair_deltas(
        rb_variant_pairs,
        group_key_fn=lambda a, b: (
            a["algorithm"],
            a["curriculum_enabled"],
            a["reward_behavior"],
            a["reward_type"],
            a["rulebook_config"],
            b["rulebook_config"],
            a["scenario_set"],
        ),
        group_cols=[
            "algorithm",
            "curriculum_enabled",
            "reward_behavior",
            "reward_type",
            "rulebook_config_from",
            "rulebook_config_to",
            "scenario_set",
        ],
    )
    _write_effect_table(
        rb_variant_rows,
        tables_dir / "effects_rulebook_variant.csv",
        tables_dir / "effects_rulebook_variant.md",
        [
            "algorithm",
            "curriculum_enabled",
            "reward_behavior",
            "reward_type",
            "rulebook_config_from",
            "rulebook_config_to",
            "scenario_set",
        ],
    )

    algo_pairs = _pairs_algorithm_effect(rows)
    algo_rows = _aggregate_pair_deltas(
        algo_pairs,
        group_key_fn=lambda a, b: (
            a["reward_type"],
            a["reward_behavior"],
            a["curriculum_enabled"],
            a["rulebook_config"],
            a["algorithm"],
            b["algorithm"],
        ),
        group_cols=["reward_type", "reward_behavior", "curriculum_enabled", "rulebook_config", "algorithm_from", "algorithm_to"],
    )
    _write_effect_table(
        algo_rows,
        tables_dir / "effects_algorithm.csv",
        tables_dir / "effects_algorithm.md",
        ["reward_type", "reward_behavior", "curriculum_enabled", "rulebook_config", "algorithm_from", "algorithm_to"],
    )

    print(f"Wrote table CSV -> {tables_dir / 'effects_curriculum.csv'}")
    print(f"Wrote table CSV -> {tables_dir / 'effects_rulebook_reward.csv'}")
    print(f"Wrote table CSV -> {tables_dir / 'effects_rulebook_variant.csv'}")
    print(f"Wrote table CSV -> {tables_dir / 'effects_algorithm.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor-effect tables from final_eval aggregated rows.")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_factor_effect_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
    )


if __name__ == "__main__":
    main()
