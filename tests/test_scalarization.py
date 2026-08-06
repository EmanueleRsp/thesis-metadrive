"""Acceptance and regression tests for SCAL-V1.0."""

from __future__ import annotations

import math

import pytest

from thesis_rl.reward.scalarization import (
    ScalarizationConfig,
    ScalarizationConfigurationError,
    ScalarizationEvaluationError,
    scalarize_rulebook_margins,
)


def test_default_rank_neutral_vector_is_zero() -> None:
    # SCAL-V1.1/ADR-057: the dataclass default mode is now
    # bounded_priority_weighted_rank; bounded_satisfaction_rank's own
    # unchanged-formula assertions below pin it explicitly instead of relying
    # on the (now different) implicit default.
    result = scalarize_rulebook_margins(
        (0.0, 0.0, 0.0, 0.0),
        ScalarizationConfig(mode="bounded_satisfaction_rank", priority_base=2.01),
    )

    assert result.reward == 0.0
    assert result.satisfaction_pattern == (True, True, True)


@pytest.mark.parametrize(
    ("margins", "expected"),
    [
        ((-1.0, -1.0, -1.0, -1.0), -(2.01**3 + 2.01**2 + 2.01) - 1.0),
        ((0.0, 0.0, 0.0, 1.0), 0.25),
        ((0.0, -1.0, 0.0, 0.0), -(2.01**2) - 0.25),
    ],
)
def test_rank_reference_values(margins: tuple[float, ...], expected: float) -> None:
    result = scalarize_rulebook_margins(
        margins, ScalarizationConfig(mode="bounded_satisfaction_rank", priority_base=2.01)
    )
    assert result.reward == pytest.approx(expected)


def test_centered_sigmoid_is_zero_at_neutral_and_keeps_progress_continuous() -> None:
    cfg = ScalarizationConfig(mode="bounded_centered_sigmoid", priority_base=2.01)
    assert scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0), cfg).reward == pytest.approx(0.0)
    assert scalarize_rulebook_margins((0.0, 0.0, 0.0, 1.0), cfg).reward == pytest.approx(0.25)


def test_legacy_formula_supports_generic_n_and_matches_four_component_neutral_value() -> None:
    cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
        priority_base=2.01,
        vector_schema_id="legacy_v1_order",
        legacy_vector_schema_id="legacy_v1_order",
        legacy_rule_scales=(1.0, 2.0),
    )
    result = scalarize_rulebook_margins((0.0, 0.0), cfg)
    expected = 0.5 * (2.01**2 + 2.01)
    assert result.reward == pytest.approx(expected)
    assert result.satisfaction_pattern is None

    four_cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
        priority_base=2.01,
        vector_schema_id="legacy_v1_four",
        legacy_vector_schema_id="legacy_v1_four",
        legacy_rule_scales=(10000.0, 10.0, 1.0, 1.0),
    )
    assert scalarize_rulebook_margins((0.0,) * 4, four_cfg).reward == pytest.approx(15.246554505)


def test_legacy_requires_exactly_n_scales() -> None:
    cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
        priority_base=2.01,
        legacy_vector_schema_id="legacy",
        legacy_rule_scales=(1.0, 2.0),
    )
    with pytest.raises(ScalarizationEvaluationError, match="exactly one value"):
        scalarize_rulebook_margins((0.0, 0.0, 0.0), cfg)


@pytest.mark.parametrize("margins", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.1)])
def test_bounded_contract_requires_four_components_and_rejects_out_of_range(
    margins: tuple[float, ...],
) -> None:
    with pytest.raises(ScalarizationEvaluationError):
        scalarize_rulebook_margins(margins, ScalarizationConfig())


def test_bounded_canonicalization_only_accepts_tolerance_boundary_overshoot() -> None:
    result = scalarize_rulebook_margins(
        (-1.0 - 5.0e-9, 5.0e-9, -5.0e-9, 1.0 + 5.0e-9), ScalarizationConfig()
    )
    assert result.canonical_margins == (-1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ScalarizationEvaluationError):
        scalarize_rulebook_margins((-1.0 - 2.0e-8, 0.0, 0.0, 0.0), ScalarizationConfig())


def test_rank_priority_dominance_is_exact_for_all_satisfaction_patterns() -> None:
    cfg = ScalarizationConfig(mode="bounded_satisfaction_rank", priority_base=2.01)
    for bits in range(8):
        margins = tuple(0.0 if bits & (1 << index) else -1.0 for index in range(3)) + (0.0,)
        result = scalarize_rulebook_margins(margins, cfg)
        assert result.satisfaction_pattern == tuple(bool(bits & (1 << index)) for index in range(3))
    best_violation = scalarize_rulebook_margins((0.0, 0.0, -1.0, 1.0), cfg).reward
    worst_satisfied = scalarize_rulebook_margins((0.0, 0.0, 0.0, -1.0), cfg).reward
    assert worst_satisfied > best_violation


def test_sigmoid_is_stable_for_large_legacy_values() -> None:
    cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
        priority_base=2.01,
        legacy_vector_schema_id="legacy",
        legacy_rule_scales=(1.0,),
    )
    result = scalarize_rulebook_margins((1.0e12,), cfg)
    assert math.isfinite(result.reward)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"priority_base": 1.0},
        {"native_environment_reward_weight": 0.1},
        {"mode": "legacy_scaled_sigmoid"},
        {"mode": "bounded_satisfaction_rank", "legacy_rule_scales": (1.0,)},
        # SCAL-V1.1 REQ-SCAL11-003: each mode requires its own frozen base.
        {"mode": "bounded_satisfaction_rank", "priority_base": 3.0},
        {"mode": "bounded_priority_weighted_rank", "priority_base": 2.01},
        {"reward_compression_mode": "unknown"},
    ],
)
def test_invalid_configuration_fails_fast(kwargs: dict[str, object]) -> None:
    with pytest.raises(ScalarizationConfigurationError):
        ScalarizationConfig(**kwargs)


# --- SCAL-V1.1: bounded_priority_weighted_rank and reward_compression -----


def test_default_mode_is_priority_weighted_rank_with_base_three() -> None:
    """AC-SCAL11-001: DEC-SCAL11-001=A makes this the implicit default."""

    cfg = ScalarizationConfig()
    assert cfg.mode == "bounded_priority_weighted_rank"
    assert cfg.priority_base == 3.0
    assert cfg.reward_compression_mode == "none"


@pytest.mark.parametrize(
    ("margins", "expected"),
    [
        ((0.0, 0.0, 0.0, 0.0), 0.0),
        ((0.0, 0.0, 0.0, 1.0), 1.0),
        ((0.0, 0.0, 0.0, -1.0), -1.0),
        ((0.0, 0.0, -1.0, 1.0), -5.0),
        ((0.0, -1.0, 0.0, 1.0), -17.0),
        ((-1.0, 0.0, 0.0, 1.0), -53.0),
        ((-1.0, -1.0, -1.0, -1.0), -79.0),
    ],
)
def test_priority_weighted_rank_reference_values(
    margins: tuple[float, ...], expected: float
) -> None:
    """AC-SCAL11-002."""

    result = scalarize_rulebook_margins(margins, ScalarizationConfig())
    assert result.reward == pytest.approx(expected, abs=1.0e-6)
    assert result.raw_reward == pytest.approx(expected, abs=1.0e-6)


def test_priority_weighted_rank_dominance_is_exact_for_all_satisfaction_patterns() -> None:
    """AC-SCAL11-003: exhaustive pattern dominance at boundary/extreme margins."""

    cfg = ScalarizationConfig()

    def reward_for(pattern: tuple[bool, ...], m4: float) -> float:
        margins = tuple(0.0 if satisfied else -1.0 for satisfied in pattern) + (m4,)
        result = scalarize_rulebook_margins(margins, cfg)
        assert result.satisfaction_pattern == pattern
        return result.reward

    for bits_a in range(8):
        pattern_a = tuple(bool(bits_a & (1 << index)) for index in range(3))
        for bits_b in range(8):
            if bits_a == bits_b:
                continue
            pattern_b = tuple(bool(bits_b & (1 << index)) for index in range(3))
            first_differing = next(
                index for index in range(3) if pattern_a[index] != pattern_b[index]
            )
            if pattern_a[first_differing]:
                # A satisfies the first differing rule: worst case for A (m4=-1)
                # must still strictly beat B's best case (m4=1), independent of
                # every lower-priority indicator and margin value.
                assert reward_for(pattern_a, -1.0) > reward_for(pattern_b, 1.0)
            else:
                assert reward_for(pattern_b, -1.0) > reward_for(pattern_a, 1.0)


def test_priority_weighted_rank_dominance_algebraic_guard() -> None:
    """AC-SCAL11-004: symbolic bound check, not only the a'=3 instantiation."""

    base = ScalarizationConfig().priority_base
    assert base > 2.0
    assert base**2 > 2.0 * base + 2.0
    assert base**3 > 2.0 * base**2 + 2.0 * base + 2.0


def test_priority_weighted_rank_is_linear_with_no_saturation() -> None:
    """AC-SCAL11-005: fixed linear coefficient, no saturation across the domain."""

    cfg = ScalarizationConfig()
    coefficient = cfg.priority_base**3
    for low, high in [(-1.0, -0.5), (-0.9, -0.1), (-0.99, -0.01)]:
        low_reward = scalarize_rulebook_margins((low, 0.0, 0.0, 0.0), cfg).reward
        high_reward = scalarize_rulebook_margins((high, 0.0, 0.0, 0.0), cfg).reward
        assert high_reward - low_reward == pytest.approx(coefficient * (high - low))


def test_symlog_compression_reference_values() -> None:
    """AC-SCAL11-007: symlog matches the closed form; none leaves reward unchanged."""

    cfg_none = ScalarizationConfig()
    result_none = scalarize_rulebook_margins((-1.0, -1.0, -1.0, -1.0), cfg_none)
    assert result_none.reward == result_none.raw_reward
    assert result_none.reward_compression_mode == "none"

    cfg_symlog = ScalarizationConfig(reward_compression_mode="symlog")
    result = scalarize_rulebook_margins((-1.0, -1.0, -1.0, -1.0), cfg_symlog)
    assert result.raw_reward == pytest.approx(-79.0)
    assert result.reward == pytest.approx(-math.log(80.0), abs=1.0e-6)
    assert result.reward_compression_mode == "symlog"

    zero_result = scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0), cfg_symlog)
    assert zero_result.reward == 0.0


def test_symlog_compression_preserves_pairwise_order() -> None:
    """AC-SCAL11-006/REQ-SCAL11-005: h is strictly increasing."""

    cfg_none = ScalarizationConfig()
    cfg_symlog = ScalarizationConfig(reward_compression_mode="symlog")
    margins_a = (0.0, 0.0, 0.0, 1.0)
    margins_b = (0.0, 0.0, -1.0, 1.0)
    raw_a = scalarize_rulebook_margins(margins_a, cfg_none).reward
    raw_b = scalarize_rulebook_margins(margins_b, cfg_none).reward
    assert raw_a > raw_b

    compressed_a = scalarize_rulebook_margins(margins_a, cfg_symlog).reward
    compressed_b = scalarize_rulebook_margins(margins_b, cfg_symlog).reward
    assert compressed_a > compressed_b
