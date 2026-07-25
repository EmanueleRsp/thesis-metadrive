"""EVAL-PROTOCOL v1.0 REQ-009/DEC-003 (amended 2026-07-25) and REQ-007 tests.

REQ-009/DEC-003: an optional, off-by-default 95% CI column
(``1.96 * sd_a(x) / sqrt(n)``) may be added when the analyst opts in via
``include_ci=True``; the default (``include_ci=False``) must stay
byte-identical to the mandatory raw-values/mean/SD-only reporting.

REQ-007: R1--R3 (cost-based constraint macro-rules) and R4/route_progress (a
progress-margin task-completion metric with no applicability concept, range
[-1, 1]) must be reported in structurally distinct tables/columns, never
merged or labeled with shared "violation rate" terminology.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from thesis_rl.analysis.common_stats import ci95, mean_sd
from thesis_rl.analysis.tables.make_final_tables import build_final_tables
from thesis_rl.analysis.tables.make_rulebook_tables import build_rulebook_tables

FINAL_EVAL_HEADER = [
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "eval_type",
    "scenario_set",
    "seed",
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "top_rule_violation_rate",
    "route_completion",
    "mean_reward",
    "avg_error_value",
    "max_error_value",
    "counterexample_rate",
]

RULE_METRICS_HEADER = [
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "rule_name",
    "rule_priority",
    "violated",
    "violation_rate",
    "violation_count",
    "mean_margin",
    "min_margin",
    "max_margin",
]


def _write_final_eval_fixture(aggregated_dir: Path) -> None:
    rows = []
    for seed, success in zip([0, 1, 2], [0.8, 0.7, 0.9]):
        rows.append(
            {
                "condition_id": "ppo_sb3_thesis",
                "algorithm": "ppo_sb3",
                "reward_type": "rulebook",
                "reward_behavior": "scalar_reward",
                "curriculum_name": "scenario_acl_scenarionet",
                "rulebook_config": "v4.7",
                "eval_type": "final",
                "scenario_set": "test",
                "seed": str(seed),
                "success_rate": str(success),
                "collision_rate": "0.1",
                "out_of_road_rate": "0.05",
                "top_rule_violation_rate": "0.02",
                "route_completion": "0.9",
                "mean_reward": "10.0",
                "avg_error_value": "0.0",
                "max_error_value": "0.0",
                "counterexample_rate": "0.0",
            }
        )
    with (aggregated_dir / "final_eval_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_EVAL_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_ci95_helper_matches_historical_formula() -> None:
    values = [0.5, 0.6, 0.7]
    _m, s = mean_sd(values)
    expected = 1.96 * s / math.sqrt(len(values))
    assert math.isclose(ci95(s, len(values)), expected)


def test_ci95_helper_zero_n_returns_zero_without_division_error() -> None:
    assert ci95(1.0, 0) == 0.0


def test_build_final_tables_default_off_has_no_ci_column(tmp_path: Path) -> None:
    aggregated_dir = tmp_path / "aggregated"
    tables_dir = tmp_path / "tables"
    aggregated_dir.mkdir()
    _write_final_eval_fixture(aggregated_dir)

    build_final_tables(aggregated_dir=aggregated_dir, tables_dir=tables_dir)

    header = (tables_dir / "final_evaluation.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "ci95" not in header.lower()


def test_build_final_tables_include_ci_adds_column_without_removing_mean_sd(tmp_path: Path) -> None:
    aggregated_dir = tmp_path / "aggregated"
    tables_dir = tmp_path / "tables"
    aggregated_dir.mkdir()
    _write_final_eval_fixture(aggregated_dir)

    build_final_tables(aggregated_dir=aggregated_dir, tables_dir=tables_dir, include_ci=True)

    with (tables_dir / "final_evaluation.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        assert "success_rate_ci95" in header
        assert "success_rate_mean" in header
        assert "success_rate_std" in header
        assert "success_rate_seed_values" in header
        row = next(reader)

    values = [0.8, 0.7, 0.9]
    m, s = mean_sd(values)
    assert math.isclose(float(row["success_rate_mean"]), m)
    assert math.isclose(float(row["success_rate_std"]), s)
    assert math.isclose(float(row["success_rate_ci95"]), ci95(s, len(values)))


def _write_rule_metrics_fixture(aggregated_dir: Path) -> None:
    descriptor = {
        "condition_id": "ppo_sb3_thesis",
        "algorithm": "ppo_sb3",
        "reward_type": "rulebook",
        "reward_behavior": "scalar_reward",
        "curriculum_name": "scenario_acl_scenarionet",
        "rulebook_config": "v4.7",
    }
    final_rows = []
    for seed in (0, 1, 2):
        final_rows.append(
            {
                **descriptor,
                "seed": str(seed),
                "avg_error_value": "0.0",
                "max_error_value": "0.0",
                "counterexample_rate": "0.0",
                "violated_rules_ratio": "0.0",
                "unique_violation_patterns": "0",
            }
        )
    final_header = list(descriptor.keys()) + [
        "seed",
        "avg_error_value",
        "max_error_value",
        "counterexample_rate",
        "violated_rules_ratio",
        "unique_violation_patterns",
    ]
    with (aggregated_dir / "final_eval_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_header)
        writer.writeheader()
        writer.writerows(final_rows)

    rule_rows = []
    for seed, r1_margin, r4_margin in zip((0, 1, 2), (-0.1, 0.2, 0.1), (-0.3, 0.4, -0.2)):
        rule_rows.append(
            {
                **descriptor,
                "rule_name": "collision_impact",
                "rule_priority": "0",
                "violated": str(r1_margin < 0),
                "violation_rate": "0.1" if r1_margin < 0 else "0.0",
                "violation_count": "1" if r1_margin < 0 else "0",
                "mean_margin": str(r1_margin),
                "min_margin": str(r1_margin),
                "max_margin": str(r1_margin),
            }
        )
        rule_rows.append(
            {
                **descriptor,
                "rule_name": "route_progress",
                "rule_priority": "3",
                "violated": str(r4_margin < 0),
                "violation_rate": "0.2" if r4_margin < 0 else "0.0",
                "violation_count": "1" if r4_margin < 0 else "0",
                "mean_margin": str(r4_margin),
                "min_margin": str(r4_margin),
                "max_margin": str(r4_margin),
            }
        )
    with (aggregated_dir / "rule_metrics_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RULE_METRICS_HEADER)
        writer.writeheader()
        writer.writerows(rule_rows)


def test_r1_r3_and_r4_are_reported_in_distinct_tables(tmp_path: Path) -> None:
    aggregated_dir = tmp_path / "aggregated"
    tables_dir = tmp_path / "tables"
    aggregated_dir.mkdir()
    _write_rule_metrics_fixture(aggregated_dir)

    build_rulebook_tables(aggregated_dir=aggregated_dir, tables_dir=tables_dir)

    per_rule_csv = tables_dir / "rule_violation_by_rule.csv"
    r4_csv = tables_dir / "rulebook_r4_progress_margin.csv"
    assert per_rule_csv.exists()
    assert r4_csv.exists()

    with per_rule_csv.open(newline="", encoding="utf-8") as handle:
        r1_r3_rows = list(csv.DictReader(handle))
    assert all(row["rule_name"] != "route_progress" for row in r1_r3_rows)
    assert any(row["rule_name"] == "collision_impact" for row in r1_r3_rows)

    with r4_csv.open(newline="", encoding="utf-8") as handle:
        r4_reader = csv.DictReader(handle)
        r4_header = r4_reader.fieldnames or []
        r4_rows = list(r4_reader)

    # R4's table never uses constraint-rule ("violation"/"violated") terminology.
    for column in r4_header:
        assert "violat" not in column.lower(), f"R4 table column uses constraint terminology: {column}"
    assert "negative_progress_rate_mean" in r4_header
    assert "mean_progress_margin_mean" in r4_header
    assert len(r4_rows) == 1
    assert r4_rows[0]["rule_name"] == "route_progress"


def test_r4_table_supports_optional_ci_column(tmp_path: Path) -> None:
    aggregated_dir = tmp_path / "aggregated"
    tables_dir = tmp_path / "tables"
    aggregated_dir.mkdir()
    _write_rule_metrics_fixture(aggregated_dir)

    build_rulebook_tables(aggregated_dir=aggregated_dir, tables_dir=tables_dir, include_ci=True)

    with (tables_dir / "rulebook_r4_progress_margin.csv").open(newline="", encoding="utf-8") as handle:
        header = csv.DictReader(handle).fieldnames or []
    assert "mean_progress_margin_ci95" in header
