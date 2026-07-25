"""EVAL-PROTOCOL v1.0 REQ-014 tests: qualitative-case manifest uses exactly
the four approved post-hoc categories (`representative_success`,
`representative_failure`, `severe_rule_violation`, `algorithm_disagreement`),
and `algorithm_disagreement` is a direct cross-condition comparison keyed by
shared `scenario_uid`."""

from __future__ import annotations

import csv
from pathlib import Path

from thesis_rl.analysis.videos.make_qualitative_manifest import build_qualitative_manifest

FIELDNAMES = [
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "rulebook_config",
    "run_id",
    "run_dir",
    "seed",
    "eval_id",
    "episode_id",
    "scenario_seed",
    "scenario_uid",
    "global_step",
    "stage",
    "eval_type",
    "success",
    "collision",
    "out_of_road",
    "route_completion",
    "top_rule_violation_rate",
    "error_value",
]


def _row(**overrides: object) -> dict[str, str]:
    base = {
        "condition_id": "cond_a",
        "algorithm": "ppo",
        "reward_type": "scalar",
        "reward_behavior": "shaped",
        "curriculum_name": "none",
        "rulebook_config": "v2",
        "run_id": "run_a_1",
        "run_dir": "/runs/run_a_1",
        "seed": "1",
        "eval_id": "1",
        "episode_id": "1",
        "scenario_seed": "100",
        "scenario_uid": "waymo:v1:s0",
        "global_step": "1500000",
        "stage": "final",
        "eval_type": "final",
        "success": "true",
        "collision": "false",
        "out_of_road": "false",
        "route_completion": "1.0",
        "top_rule_violation_rate": "0.0",
        "error_value": "0.0",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return base


def _write_eval_rows(comparison_root: Path, rows: list[dict[str, str]]) -> None:
    aggregated_dir = comparison_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    with (aggregated_dir / "eval_episodes_all_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_uses_exactly_the_four_approved_req014_categories(tmp_path: Path) -> None:
    comparison_root = tmp_path / "dimension" / "comparison_1"
    rows = [
        _row(condition_id="cond_a", episode_id="1", scenario_uid="s0", success="true"),
        _row(condition_id="cond_a", episode_id="2", scenario_uid="s1", success="false", collision="true", top_rule_violation_rate="0.9"),
    ]
    _write_eval_rows(comparison_root, rows)

    csv_path, _ = build_qualitative_manifest(comparison_root=comparison_root)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        out_rows = list(csv.DictReader(handle))

    categories = {row["category"] for row in out_rows}
    assert categories == {
        "representative_success",
        "representative_failure",
        "severe_rule_violation",
        "algorithm_disagreement",
    }
    # Exactly one row per (condition, category): a single condition here.
    assert len(out_rows) == 4


def test_manifest_no_longer_emits_median_or_curriculum_transition_case(tmp_path: Path) -> None:
    comparison_root = tmp_path / "dimension" / "comparison_1"
    rows = [
        _row(condition_id="cond_a", episode_id="1", scenario_uid="s0"),
    ]
    _write_eval_rows(comparison_root, rows)

    csv_path, _ = build_qualitative_manifest(comparison_root=comparison_root)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        out_rows = list(csv.DictReader(handle))
    categories = {row["category"] for row in out_rows}
    assert "median" not in categories
    assert "curriculum_transition_case" not in categories
    assert "best" not in categories
    assert "worst" not in categories
    assert "rule_violation_case" not in categories


def test_algorithm_disagreement_selects_shared_scenario_uid_with_diverging_success(tmp_path: Path) -> None:
    comparison_root = tmp_path / "dimension" / "comparison_1"
    rows = [
        # cond_a succeeds on s0; cond_b fails on s0 -> disagreement on s0.
        _row(condition_id="cond_a", run_id="run_a", episode_id="1", scenario_uid="s0", success="true", route_completion="1.0"),
        _row(condition_id="cond_b", run_id="run_b", episode_id="1", scenario_uid="s0", success="false", route_completion="0.2"),
        # Both agree (success) on s1: no disagreement signal there.
        _row(condition_id="cond_a", run_id="run_a", episode_id="2", scenario_uid="s1", success="true"),
        _row(condition_id="cond_b", run_id="run_b", episode_id="2", scenario_uid="s1", success="true"),
    ]
    _write_eval_rows(comparison_root, rows)

    csv_path, _ = build_qualitative_manifest(comparison_root=comparison_root)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        out_rows = list(csv.DictReader(handle))

    disagreement_rows = [row for row in out_rows if row["category"] == "algorithm_disagreement"]
    assert len(disagreement_rows) == 2  # one per condition (cond_a, cond_b)
    for row in disagreement_rows:
        assert row["selection_status"] == "selected"
        assert row["episode_id"] == "1"
        assert "diverges_from_condition_" in row["selection_reason"]


def test_algorithm_disagreement_unavailable_when_single_condition(tmp_path: Path) -> None:
    comparison_root = tmp_path / "dimension" / "comparison_1"
    rows = [_row(condition_id="cond_a", episode_id="1", scenario_uid="s0")]
    _write_eval_rows(comparison_root, rows)

    csv_path, _ = build_qualitative_manifest(comparison_root=comparison_root)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        out_rows = list(csv.DictReader(handle))
    row = next(r for r in out_rows if r["category"] == "algorithm_disagreement")
    assert row["selection_status"] == "unavailable"
    # `_select_distinct`'s generic wrapper reports its own reason when the
    # picker returns None; the picker-specific reason ("no_other_conditions")
    # is only surfaced on a successful pick (see `disagreement_reasons`).
    assert row["selection_reason"] == "picker_returned_none"
