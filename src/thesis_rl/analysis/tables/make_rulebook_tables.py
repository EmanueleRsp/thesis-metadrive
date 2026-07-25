from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from thesis_rl.analysis.common_stats import ci95, mean_sd, to_float
from thesis_rl.common.paths import default_analysis_root_str

# EVAL-PROTOCOL v1.0 REQ-007: R1--R3 (collision_impact, dynamic_interaction_safety,
# road_traffic_compliance) are cost-based constraint macro-rules with an
# applicable/violated/satisfied status. R4 (route_progress) is a task-progress
# margin in [-1, 1] with no applicability concept and is never reported using
# constraint-rule ("violation") terminology; see `MacroRule`/`MACRO_RULE_ORDER`
# in `src/thesis_rl/rulebook/v2/types.py`, produced by `Agent.evaluate()`'s
# `per_rule` rows and persisted as `rule_metrics.csv`/`rule_metrics_all_runs.csv`.
R4_PROGRESS_MARGIN_RULE_NAME = "route_progress"

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


def build_rulebook_tables(
    aggregated_dir: Path, tables_dir: Path, *, include_ci: bool = False
) -> None:
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
        ] + [f"{m}_mean" for m in GLOBAL_METRICS] + [f"{m}_sd" for m in GLOBAL_METRICS]
        if include_ci:
            fieldnames += [f"{m}_ci95" for m in GLOBAL_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id in sorted(global_bucket.keys()):
            row: dict[str, object] = dict(descriptors[condition_id])
            n_seeds = len(global_seeds.get(condition_id, set()))
            row["n_seeds"] = n_seeds
            for metric in GLOBAL_METRICS:
                values = global_bucket[condition_id].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_sd"] = s
                    if include_ci:
                        row[f"{metric}_ci95"] = ci95(s, n_seeds)
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_sd"] = ""
                    if include_ci:
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
                    m, ci = mean_sd(values)
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

        for metric in (
            "violation_rate",
            "mean_margin",
            "min_margin",
            "max_margin",
            # EVAL-PROTOCOL v1.0 REQ-008 (spec §7.2): seed-level episode
            # counts feeding `violation_rate`/`mean_margin` (applicable) and
            # excluded from them (zero applicable steps this seed's run).
            # R4/route_progress has no applicability concept, so these are
            # meaningless for it and are only surfaced in the R1--R3 table.
            "applicable_episode_count",
            "excluded_episode_count",
        ):
            value = to_float(row.get(metric))
            if value is not None:
                per_rule_bucket[key][metric].append(value)

    # EVAL-PROTOCOL v1.0 REQ-007: R1--R3 (cost-based constraint macro-rules)
    # and R4/route_progress (a progress-margin task-completion metric with no
    # applicability concept, range [-1, 1]) are structurally distinct and
    # must never share "violation rate"/"violated" terminology or the same
    # reporting table.
    r1_r3_keys = sorted(
        key for key in per_rule_bucket.keys() if key[1] != R4_PROGRESS_MARGIN_RULE_NAME
    )
    r4_keys = sorted(
        key for key in per_rule_bucket.keys() if key[1] == R4_PROGRESS_MARGIN_RULE_NAME
    )

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
            "violation_rate_sd",
            "mean_margin_mean",
            "mean_margin_sd",
            "min_margin_mean",
            "min_margin_sd",
            "max_margin_mean",
            "max_margin_sd",
            # EVAL-PROTOCOL v1.0 REQ-008 (spec §7.2): the seed-level
            # applicable/excluded episode counts feeding the metrics above,
            # reported across seeds (mean/sd) alongside the seed-level value.
            "applicable_episode_count_mean",
            "applicable_episode_count_sd",
            "excluded_episode_count_mean",
            "excluded_episode_count_sd",
        ]
        if include_ci:
            fieldnames += [
                "violation_rate_ci95",
                "mean_margin_ci95",
                "min_margin_ci95",
                "max_margin_ci95",
            ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in r1_r3_keys:
            row: dict[str, object] = dict(per_rule_desc[key])
            for metric in (
                "violation_rate",
                "mean_margin",
                "min_margin",
                "max_margin",
                "applicable_episode_count",
                "excluded_episode_count",
            ):
                values = per_rule_bucket[key].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    row[f"{metric}_mean"] = m
                    row[f"{metric}_sd"] = s
                    if include_ci and metric in (
                        "violation_rate",
                        "mean_margin",
                        "min_margin",
                        "max_margin",
                    ):
                        row[f"{metric}_ci95"] = ci95(s, len(values))
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_sd"] = ""
                    if include_ci and metric in (
                        "violation_rate",
                        "mean_margin",
                        "min_margin",
                        "max_margin",
                    ):
                        row[f"{metric}_ci95"] = ""
            writer.writerow(row)

    per_rule_md = tables_dir / "rule_violation_by_rule.md"
    with per_rule_md.open("w", encoding="utf-8") as handle:
        handle.write("| Condition | Rule | Violation rate | Mean margin | Min margin | Max margin |\n")
        handle.write("| --- | --- | --- | --- | --- | --- |\n")
        for key in r1_r3_keys:
            condition_id, rule = key
            cells = [condition_id, rule]
            for metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[key].get(metric, [])
                if values:
                    m, s = mean_sd(values)
                    cells.append(f"{m:.4f} ± {s:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    # R4 (route_progress): a distinct table using progress-margin language,
    # never "violation_rate"/"violated" (REQ-007). `negative_progress_rate`
    # reports the fraction of episodes whose min progress margin is negative
    # (net regression), a descriptive statistic, not a rule violation.
    r4_csv = tables_dir / "rulebook_r4_progress_margin.csv"
    with r4_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "rule_name",
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum",
            "rulebook_config",
            "negative_progress_rate_mean",
            "negative_progress_rate_sd",
            "mean_progress_margin_mean",
            "mean_progress_margin_sd",
            "min_progress_margin_mean",
            "min_progress_margin_sd",
            "max_progress_margin_mean",
            "max_progress_margin_sd",
        ]
        if include_ci:
            fieldnames += [
                "negative_progress_rate_ci95",
                "mean_progress_margin_ci95",
                "min_progress_margin_ci95",
                "max_progress_margin_ci95",
            ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        r4_metric_map = {
            "negative_progress_rate": "violation_rate",
            "mean_progress_margin": "mean_margin",
            "min_progress_margin": "min_margin",
            "max_progress_margin": "max_margin",
        }
        for key in r4_keys:
            desc = dict(per_rule_desc[key])
            desc.pop("rule_priority", None)
            row: dict[str, object] = desc
            for out_metric, src_metric in r4_metric_map.items():
                values = per_rule_bucket[key].get(src_metric, [])
                if values:
                    m, s = mean_sd(values)
                    row[f"{out_metric}_mean"] = m
                    row[f"{out_metric}_sd"] = s
                    if include_ci:
                        row[f"{out_metric}_ci95"] = ci95(s, len(values))
                else:
                    row[f"{out_metric}_mean"] = ""
                    row[f"{out_metric}_sd"] = ""
                    if include_ci:
                        row[f"{out_metric}_ci95"] = ""
            writer.writerow(row)

    r4_md = tables_dir / "rulebook_r4_progress_margin.md"
    with r4_md.open("w", encoding="utf-8") as handle:
        handle.write(
            "| Condition | Negative progress rate | Mean progress margin | "
            "Min progress margin | Max progress margin |\n"
        )
        handle.write("| --- | --- | --- | --- | --- |\n")
        for key in r4_keys:
            condition_id, _rule = key
            cells = [condition_id]
            for src_metric in ("violation_rate", "mean_margin", "min_margin", "max_margin"):
                values = per_rule_bucket[key].get(src_metric, [])
                if values:
                    m, s = mean_sd(values)
                    cells.append(f"{m:.4f} ± {s:.4f}")
                else:
                    cells.append("")
            handle.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote table CSV -> {global_csv}")
    print(f"Wrote table MD  -> {global_md}")
    print(f"Wrote table CSV -> {per_rule_csv}")
    print(f"Wrote table MD  -> {per_rule_md}")
    print(f"Wrote table CSV -> {r4_csv}")
    print(f"Wrote table MD  -> {r4_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rulebook compliance tables by condition.")
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Also emit an optional 1.96*sd/sqrt(n) CI95 column (off by default, REQ-009/DEC-003).",
    )
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    build_rulebook_tables(
        aggregated_dir=analysis_root / "aggregated",
        tables_dir=analysis_root / "tables",
        include_ci=bool(args.include_ci),
    )


if __name__ == "__main__":
    main()
