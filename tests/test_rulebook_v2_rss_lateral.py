from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.components.rss_lateral import (
    LateralRSSCandidate,
    evaluate_rss_lateral,
    lateral_safe_distance_m,
)


def test_lateral_safe_distance_matches_reference_value_at_zero_speed() -> None:
    """AC-R2-03 (rulebook v4.8 §7, §12): two vehicles with zero inward
    lateral speed at rho_lat=0.5s, a_acc_max^lat=0.2, a_brake_min^lat=0.8,
    mu=0.10 give d_safe^lat ~= 0.1625 m."""
    safe = lateral_safe_distance_m(ego_inward_speed_mps=0.0, actor_inward_speed_mps=0.0)
    assert safe == pytest.approx(0.1625, abs=1.0e-6)


def test_evaluate_rss_lateral_not_applicable_without_candidates() -> None:
    result, _, _ = evaluate_rss_lateral(candidates=())
    assert result.applicable is False
    assert result.cost == 0.0
    assert result.status.value == "not_applicable"


def test_evaluate_rss_lateral_ignores_longitudinally_safe_candidates() -> None:
    """AC-R2-01/05: a candidate outside the longitudinal gate contributes
    zero cost even with a tiny lateral gap (stable-but-close, parallel
    lanes)."""
    result, _, _ = evaluate_rss_lateral(
        candidates=(
            LateralRSSCandidate(
                actor_id="parallel",
                lateral_gap_m=0.05,
                ego_inward_speed_mps=0.0,
                actor_inward_speed_mps=0.0,
                longitudinal_unsafe=False,
            ),
        ),
    )
    assert result.applicable is True
    assert result.cost == 0.0
    assert result.status.value == "satisfied"


def test_evaluate_rss_lateral_penalizes_a_gap_below_the_safe_distance() -> None:
    """AC-R2-02: a genuine lateral approach (gate true, gap below
    d_safe^lat) produces a positive bounded cost."""
    result, _, _ = evaluate_rss_lateral(
        candidates=(
            LateralRSSCandidate(
                actor_id="closing",
                lateral_gap_m=0.05,
                ego_inward_speed_mps=1.0,
                actor_inward_speed_mps=0.0,
                longitudinal_unsafe=True,
            ),
        ),
    )
    assert 0.0 < result.cost <= 1.0
    assert result.status.value == "violated"
    assert result.raw["worst_actor_id"] == "closing"


def test_evaluate_rss_lateral_worst_of_multiple_candidates() -> None:
    safe_candidate = LateralRSSCandidate("safe", 5.0, 0.0, 0.0, True)
    unsafe_candidate = LateralRSSCandidate("unsafe", 0.0, 1.0, 1.0, True)
    result, _, _ = evaluate_rss_lateral(candidates=(safe_candidate, unsafe_candidate))
    assert result.raw["worst_actor_id"] == "unsafe"
    assert result.cost == pytest.approx(1.0)


def test_evaluate_rss_lateral_rejects_negative_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        evaluate_rss_lateral(
            candidates=(LateralRSSCandidate("bad", -0.1, 0.0, 0.0, True),),
        )
