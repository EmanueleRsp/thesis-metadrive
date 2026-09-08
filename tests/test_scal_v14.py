"""`RB51` M6: `SCAL-V1.4`, RULEBOOK-V5.1 §5.

`T-RB51-06` and `T-RB51-07`. The formula is asserted term by term against §5.1
rather than against a remembered number, and the rank-preservation condition of
§5.4 is asserted as a *constructor* gate: an inadmissible weight set must be
impossible to hold, not merely wrong when used.
"""

from __future__ import annotations

import pytest

from thesis_rl.reward.scalarization import (
    SIX_LEVEL_PRIORITY_BASE,
    SIX_LEVEL_VECTOR_SCHEMA_ID,
    ScalarizationConfig,
    ScalarizationConfigurationError,
    ScalarizationEvaluationError,
    scalarize_rulebook_margins,
)


A = SIX_LEVEL_PRIORITY_BASE
# ADR-081. Derived, not chosen: the reward discourages accelerating into a
# conflict only while `(w2*sigma + phi) * v_ref / (2*lambda4*tau)` exceeds the
# braking the conflict demands, and at `sigma = 0` that ceiling was 1.46 m/s^2
# against a vehicle's ~9. At `a = 2.5, sigma = 0.30` it is 12.43 m/s^2.
SIGMA = 0.30
PHI = 0.25
LAMBDA4 = 2.0
ETA = 1.0
LAMBDA6 = 0.2
DT_RATIO = 0.1


def config(**overrides) -> ScalarizationConfig:
    values = {
        "mode": "six_level_priority_weighted_rank",
        "priority_base": A,
        "vector_schema_id": SIX_LEVEL_VECTOR_SCHEMA_ID,
    }
    values.update(overrides)
    return ScalarizationConfig(**values)


def expected_reward(margins: tuple[float, ...]) -> float:
    """§5.1 written out independently of the implementation."""

    total = 0.0
    for weight, margin in zip((A**3, A**2, A), margins[:3]):
        satisfied = 1.0 if margin == 0.0 else 0.0
        total += weight * ((satisfied - 1.0) + SIGMA * margin)
        total += PHI * margin
    total += LAMBDA4 * margins[3]
    total += ETA * margins[4] * DT_RATIO
    total += LAMBDA6 * margins[5] * DT_RATIO
    return total


@pytest.mark.parametrize(
    "margins",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, 0.4, -0.2, -0.8),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        (0.0, -0.5, -0.3, -0.7, -0.9, -0.4),
        (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
    ],
)
def test_scal_v14_matches_the_specified_formula(margins: tuple[float, ...]) -> None:
    """`T-RB51-06` / `REQ-RB51-09`."""

    result = scalarize_rulebook_margins(margins, config())
    assert result.reward == pytest.approx(expected_reward(margins))
    assert result.mode == "six_level_priority_weighted_rank"


def test_a_satisfied_level_contributes_nothing_and_a_violated_one_pays_the_step() -> None:
    """`T-RB51-06`. The satisfaction indicator is what makes a level discrete.

    A satisfied level contributes exactly zero, and crossing the threshold costs
    the level's full weight however mild the violation is. That is §5.2's
    affordability constraint in one line: a violated L2 step costs `a^2` at once,
    so an L2 sub-rule has to fire *rarely* for the rulebook to be affordable --
    a constraint on frequency, not on severity.

    `sigma` does not soften that step and is not meant to (ADR-081); it adds a
    slope *inside* the violated set, which is what a deterministic policy
    gradient needs in order to know which way out is.
    """

    satisfied = scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), config()).reward
    assert satisfied == pytest.approx(0.0)

    barely = scalarize_rulebook_margins((0.0, -1e-4, 0.0, 0.0, 0.0, 0.0), config()).reward
    assert barely == pytest.approx(-(A**2) - (A**2 * SIGMA + PHI) * 1e-4)
    # The step dominates the slope by four orders of magnitude at this severity.
    assert barely == pytest.approx(-(A**2), rel=1e-3)


def test_severity_gives_the_violated_set_a_slope_a_policy_gradient_can_follow() -> None:
    """ADR-081. Without it the reward is flat inside a violation.

    A deterministic policy gradient moves along `grad_a Q`, so a level that
    charges the same amount at `cost = 0.01` and at `cost = 1.0` tells the actor
    that it is in trouble and nothing about which direction leaves.
    """

    mild = scalarize_rulebook_margins((0.0, -0.1, 0.0, 0.0, 0.0, 0.0), config()).reward
    severe = scalarize_rulebook_margins((0.0, -1.0, 0.0, 0.0, 0.0, 0.0), config()).reward

    assert severe < mild
    # The slope across the violated range is `w2*sigma + phi` per unit of cost.
    assert mild - severe == pytest.approx((A**2 * SIGMA + PHI) * 0.9)
    # It has to be a usable fraction of the level's own weight, not a rounding
    # error on it: at `sigma = 0` this ratio was 0.25/4.84 = 5.2 %.
    assert (A**2 * SIGMA + PHI) / A**2 > 0.25


def test_progress_and_the_two_levels_below_it_form_a_finite_exchange() -> None:
    """`T-RB51-06` / §5.2. No indicator on L4, so no reward for creeping.

    An indicator on progress would make the hierarchy uniform at the price of
    rewarding an ego inching forward at 0.01 m/s, which converts "stop for ever"
    into "creep for ever along the marking".
    """

    creep = scalarize_rulebook_margins((0.0, 0.0, 0.0, 1e-6, -1.0, -1.0), config()).reward
    still = scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0, 0.0, -1.0), config()).reward
    assert creep < still


def test_l6_reaches_only_the_last_term() -> None:
    """`T-RB51-06`. Varying `lambda6` may not move any other contribution."""

    margins = (0.0, -0.5, -0.3, 0.4, -0.2, -0.8)
    low = scalarize_rulebook_margins(margins, config(progress_rate_weight=0.0))
    high = scalarize_rulebook_margins(margins, config(progress_rate_weight=0.2))
    assert low.priority_contributions == high.priority_contributions
    assert high.reward - low.reward == pytest.approx(0.2 * margins[5] * DT_RATIO)


# ---------------------------------------------------------------------------
# T-RB51-07 -- the rank-preservation condition is a constructor gate
# ---------------------------------------------------------------------------


def test_selected_weights_are_admissible() -> None:
    """`T-RB51-07` / `REQ-RB51-10`. §5.5 as amended by ADR-081:
    `2.0 + 0.1*(1.0 + 0.2) = 2.12 < 2.5`, with `sigma = 0.30` inside the
    `sigma < 0.486` the base admits."""

    assert config().progress_weight == pytest.approx(LAMBDA4)
    assert config().severity == pytest.approx(SIGMA)
    assert config().priority_base == pytest.approx(A)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Each value sits just past the §5.4 bound at `a = 2.5`: the tail
        # `lambda4 + 0.1*(eta + lambda6)` must stay under `a`, and `sigma` under
        # 0.486 (binding at k=1, not k=2).
        ("progress_weight", 2.5),
        ("progress_weight", 3.0),
        ("relaxable_weight", 5.0),
        ("progress_rate_weight", 6.0),
        ("severity", 1.0),
    ],
)
def test_inadmissible_weights_cannot_be_constructed(field: str, value: float) -> None:
    """`T-RB51-07`. Refused at construction, so no such reward is ever emitted."""

    with pytest.raises(ScalarizationConfigurationError, match="rank-preservation"):
        config(**{field: value})


def test_progress_weight_cannot_overturn_a_non_relaxable_violation() -> None:
    """`T-RB51-07`. What the §5.4 bound actually buys, at the worst case.

    The worst case is a maximal progress step against the smallest possible L3
    violation, and it is why `lambda4` cannot be raised freely.

    "Smallest possible" is bounded below by `numerical_tolerance = 1e-8`:
    `_canonicalize_bounded` clamps anything at or under it to exactly zero, by
    design, so a margin below the tolerance is not a small violation but *no*
    violation, and the satisfaction indicator correctly does not fire. The
    fixture therefore sits above the tolerance; a value under it would assert
    nothing about the bound.
    """

    violating = scalarize_rulebook_margins((0.0, 0.0, -1e-6, 1.0, 0.0, 0.0), config()).reward
    compliant = scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.0, -1.0, -1.0), config()).reward
    assert compliant > violating

    # And the clamped case is asserted rather than left implicit, so the
    # distinction above cannot be silently undone by a future edit.
    below_tolerance = scalarize_rulebook_margins((0.0, 0.0, -1e-9, 1.0, 0.0, 0.0), config()).reward
    assert below_tolerance == pytest.approx(LAMBDA4)


# ---------------------------------------------------------------------------
# Arity: the six-level vector is a different schema, not a longer one
# ---------------------------------------------------------------------------


def test_six_level_mode_rejects_a_four_margin_vector() -> None:
    """A four-level vector under the six-level mode would silently drop L5/L6."""

    with pytest.raises(ScalarizationEvaluationError, match="requires 6 macro margins"):
        scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.1), config())


def test_legacy_modes_still_reject_a_six_margin_vector() -> None:
    """`DEC-RB51-003`: the legacy modes stay, and stay four-level.

    They are historical baselines and removing them would make earlier runs
    unreproducible; feeding them a six-level vector must fail rather than read
    the first four entries as if the preference order had not changed.
    """

    legacy = ScalarizationConfig(mode="bounded_priority_weighted_rank", priority_base=3.0)
    with pytest.raises(ScalarizationEvaluationError, match="requires 4 macro margins"):
        scalarize_rulebook_margins((0.0, 0.0, 0.0, 0.1, 0.0, 0.0), legacy)
