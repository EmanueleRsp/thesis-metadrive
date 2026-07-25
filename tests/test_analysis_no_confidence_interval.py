"""EVAL-PROTOCOL v1.0 REQ-009/DEC-003: no confidence interval, bootstrap
estimate, or significance test may be computed or presented by the core
analysis pipeline. Cross-seed reporting uses raw values, mean, and sample
standard deviation only."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from thesis_rl.analysis.common_stats import mean_sd
from thesis_rl.analysis.tables.make_final_tables import build_final_tables

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


def test_mean_sd_is_sample_standard_deviation_not_confidence_interval() -> None:
    values = [0.5, 0.6, 0.7]
    m, s = mean_sd(values)
    expected_mean = sum(values) / 3
    expected_sd = math.sqrt(sum((v - expected_mean) ** 2 for v in values) / 2)
    assert math.isclose(m, expected_mean)
    assert math.isclose(s, expected_sd)
    # The old formula would divide by sqrt(n) and multiply by 1.96; verify we
    # are NOT doing that (the two quantities differ for n=3).
    wrong_ci = 1.96 * expected_sd / math.sqrt(3)
    assert not math.isclose(s, wrong_ci)


def test_build_final_tables_emits_raw_values_mean_sd_no_ci(tmp_path: Path) -> None:
    aggregated_dir = tmp_path / "aggregated"
    tables_dir = tmp_path / "tables"
    aggregated_dir.mkdir()

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

    build_final_tables(aggregated_dir=aggregated_dir, tables_dir=tables_dir)

    csv_text = (tables_dir / "final_evaluation.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert "ci95" not in header.lower()
    assert "ci_95" not in header.lower()
    assert "success_rate_mean" in header
    assert "success_rate_std" in header
    assert "success_rate_seed_values" in header

    md_text = (tables_dir / "final_evaluation.md").read_text(encoding="utf-8")
    assert "±" not in md_text or "SD" in md_text
    assert "confidence" not in md_text.lower()
