from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.components.rss import (
    RSSCalibrationArtifact,
    RSSCandidate,
    evaluate_rss,
    safe_distance_m,
)
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError


def test_rss_safe_distance_and_continuous_deficit() -> None:
    calibration = RSSCalibrationArtifact("ego-hash", 3.0)
    safe = safe_distance_m(ego_speed_mps=5.0, front_speed_mps=5.0, ego_brake_mps2=3.0)
    result, _, _ = evaluate_rss(
        scenario_id="scenario", step_index=1,
        candidates=(RSSCandidate("front", safe, 5.0, 5.0),),
        calibration=calibration, expected_config_hash="ego-hash",
    )
    assert result.cost == pytest.approx(0.0)
    deficient, _, _ = evaluate_rss(
        scenario_id="scenario", step_index=1,
        candidates=(RSSCandidate("front", safe / 2.0, 5.0, 5.0),),
        calibration=calibration, expected_config_hash="ego-hash",
    )
    assert deficient.cost == pytest.approx(0.5)


def test_rss_missing_or_mismatched_calibration_fails_fast() -> None:
    candidate = (RSSCandidate("front", 1.0, 5.0, 5.0),)
    with pytest.raises(RulebookEvaluationError, match="missing"):
        evaluate_rss(
            scenario_id="scenario", step_index=1, candidates=candidate,
            calibration=None, expected_config_hash="ego-hash",
        )
    with pytest.raises(RulebookEvaluationError, match="hash"):
        evaluate_rss(
            scenario_id="scenario", step_index=1, candidates=candidate,
            calibration=RSSCalibrationArtifact("other", 3.0), expected_config_hash="ego-hash",
        )
