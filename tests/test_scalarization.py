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
    result = scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0), ScalarizationConfig())

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
    result = scalarize_rulebook_margins(margins, ScalarizationConfig())
    assert result.reward == pytest.approx(expected)


def test_centered_sigmoid_is_zero_at_neutral_and_keeps_progress_continuous() -> None:
    cfg = ScalarizationConfig(mode="bounded_centered_sigmoid")
    assert scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0), cfg).reward == pytest.approx(0.0)
    assert scalarize_rulebook_margins((0.0, 0.0, 0.0, 1.0), cfg).reward == pytest.approx(0.25)


def test_legacy_formula_supports_generic_n_and_matches_four_component_neutral_value() -> None:
    cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
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
        vector_schema_id="legacy_v1_four",
        legacy_vector_schema_id="legacy_v1_four",
        legacy_rule_scales=(10000.0, 10.0, 1.0, 1.0),
    )
    assert scalarize_rulebook_margins((0.0,) * 4, four_cfg).reward == pytest.approx(15.246554505)


def test_legacy_requires_exactly_n_scales() -> None:
    cfg = ScalarizationConfig(
        mode="legacy_scaled_sigmoid",
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
    cfg = ScalarizationConfig()
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
    ],
)
def test_invalid_configuration_fails_fast(kwargs: dict[str, object]) -> None:
    with pytest.raises(ScalarizationConfigurationError):
        ScalarizationConfig(**kwargs)
