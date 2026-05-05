from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_rl.reward.scale_tuning import suggest_scales


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def test_scale_tuning_strict_raises_on_insufficient_active_samples(tmp_path: Path) -> None:
    margin_log = tmp_path / "margins.jsonl"
    _write_jsonl(
        margin_log,
        [
            {"rule_components": {"vehicle_collision_energy": 0.0, "goal_progress": -10.0}},
            {"rule_components": {"vehicle_collision_energy": 0.0, "goal_progress": -11.0}},
            {"rule_components": {"vehicle_collision_energy": -1e-12, "goal_progress": -12.0}},
        ],
    )

    with pytest.raises(ValueError, match="Insufficient active samples"):
        suggest_scales(
            input_paths=[margin_log],
            percentile=90.0,
            min_scale=1e-6,
            min_active_margin=1e-9,
            min_samples=2,
            strict=True,
        )


def test_scale_tuning_returns_coverage_and_scales(tmp_path: Path) -> None:
    margin_log = tmp_path / "margins.jsonl"
    _write_jsonl(
        margin_log,
        [
            {"rule_components": {"vehicle_collision_energy": -0.2, "goal_progress": -10.0}},
            {"rule_components": {"vehicle_collision_energy": -0.3, "goal_progress": -11.0}},
            {"rule_components": {"vehicle_collision_energy": 0.0, "goal_progress": -12.0}},
        ],
    )

    scales, coverage = suggest_scales(
        input_paths=[margin_log],
        percentile=90.0,
        min_scale=1e-6,
        min_active_margin=1e-9,
        min_samples=2,
        strict=True,
    )

    assert set(scales) == {"vehicle_collision_energy", "goal_progress"}
    assert coverage["vehicle_collision_energy"]["active_samples"] == 2
    assert coverage["goal_progress"]["active_samples"] == 3
    assert coverage["vehicle_collision_energy"]["sufficient_samples"] is True
