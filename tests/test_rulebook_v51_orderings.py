"""`TEST-RB5.1-01..06`: the six orderings RULEBOOK-V5.1 exists to guarantee.

Expert replay cannot reach these. It holds one logged trajectory per scenario and
therefore no counterfactual: there is no illegal shortcut in the data to compare
against the legal route the human actually drove. Every figure in RULEBOOK-V5.1
§5.5 establishes *admissibility* — that the reward does not condemn a competent
driver — and none of them establishes that the reward *ranks* trajectories the
way a competent driver would. These fixtures do, and they are the test class
RULEBOOK-V5.0 lacked entirely.

Each fixture is a pair of trajectories written directly as per-step channel
vectors ``(l1, l2, l3, delta_q, l5)``. That is the right altitude: whether a
given scene produces ``c_solid_line = 0.4`` is the sub-rule geometry's problem
and is covered by RULEBOOK-V5.0 §4's evidence, while these tests are about the
*hierarchy* — which channel a cost lands in, and what that placement implies for
the ordering.

Both comparison rules are checked wherever they differ, because they are two of
the four planned experimental arms and RULEBOOK-V5.1 `AC-RB5.1-02` requires the
strict-lexicographic failures to be reported rather than repaired:

* ``scalar`` — the summed ``SCAL-V1.3`` return of §5.1;
* ``strict lex`` — channel returns compared in order L1 ≻ L2 ≻ L3 ≻ L4 ≻ L5.

The two disagree on exactly one ordering, O1, and that disagreement is an
asserted property here rather than an accident.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

STEPS = 40
# One reference-speed step is `Delta q = 1` by construction (RULEBOOK-V5.1 §4.1),
# so a legal cruise well inside the limit advances a fraction of that.
CRUISE_DQ = 0.25
# The residual interaction cost any real trajectory accrues in traffic. RULEBOOK
# -V5.1 §5.5 measures the specified rulebook's L2 channel firing on 0.3978 % of
# expert steps, so this is not a hypothetical: it is why O1 fails under a strict
# ordering and why the thresholded arm exists.
TRAFFIC_L2 = 0.05


def load_measurement_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "measure_expert_rulebook_transition.py"
    spec = importlib.util.spec_from_file_location("measure_expert_rulebook_transition", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Loaded at import time so the plain helpers below can read `V51_TIME_COST`
# without taking the fixture. The fixture is kept because the tests name it.
MEASUREMENT = load_measurement_module()
# ADR-076. L4 carries the bare advance; the time preference lives at L6, whose
# cost the helpers derive from that same advance in exactly one place so no
# fixture can assert against a channel the instrument does not compute.
LAMBDA6 = 0.2


def l6_cost(advance: float) -> float:
    """`c_L6 = 1 - clip(Delta q, 0, 1)`, RULEBOOK-V5.1 §4.6.

    Summed over a completing trajectory this is `T - Q`, so L6 ranks by duration
    exactly while still giving a per-step gradient. Reverse motion clips to zero
    advance and therefore costs the maximum.
    """

    return 1.0 - max(0.0, min(advance, 1.0))


@pytest.fixture(scope="module")
def measurement() -> ModuleType:
    return MEASUREMENT


Step = tuple[float, float, float, float, float]


def scalar_return(module: ModuleType, trajectory: list[Step], *, lam: float, eta: float) -> float:
    """Summed ``SCAL-V1.3`` return, undiscounted.

    Undiscounted because §4.4 leaves the mission channel's discount an open
    decision and records that `gamma = 1` is what makes the L4 tie exact. These
    fixtures assert the ordering the specification claims, so they use the
    discount under which that claim is made.
    """

    return sum(
        module.v51_reward(
            l1=l1,
            l2=l2,
            l3=l3,
            delta_q=dq,
            l5=l5,
            l6=l6_cost(dq),
            lam=lam,
            eta=eta,
            lam6=LAMBDA6,
        )
        for l1, l2, l3, dq, l5 in trajectory
    )


def channel_returns(
    trajectory: list[Step],
) -> tuple[float, float, float, float, float, float]:
    """Per-channel episode totals, in the order the lexicographic arms compare.

    Six entries: the fixtures carry five, and L6 is derived from the advance
    because it is a function of it (§4.6) rather than an independent quantity.
    """

    totals = [sum(step[index] for step in trajectory) for index in range(5)]
    totals.append(sum(l6_cost(step[3]) for step in trajectory))
    return tuple(totals)  # type: ignore[return-value]


def strict_lex_prefers_first(first: list[Step], second: list[Step]) -> bool | None:
    """Compare by strict lexicographic order; ``None`` when every channel ties.

    L1, L2, L3 and L5 are costs (lower wins); L4 is progress (higher wins). The
    comparison stops at the first channel that differs, which is exactly the
    property that makes standing still unbeatable when it is 0 on the safety
    channels (§11.1).
    """

    first_channels = channel_returns(first)
    second_channels = channel_returns(second)
    for index, higher_is_better in enumerate((False, False, False, True, False, False)):
        a, b = first_channels[index], second_channels[index]
        if a == b:
            continue
        return a > b if higher_is_better else a < b
    return None


def standing_still() -> list[Step]:
    """The trajectory every ordering is defined against.

    Exactly zero on every channel: ADR-070's at-fault gate makes the interaction
    sub-rules inapplicable below 0.05 m/s, a stopped in-lane ego violates no
    position or control rule, and it makes no progress.
    """

    return [(0.0, 0.0, 0.0, 0.0, 0.0)] * STEPS


def legal_drive(*, l2: float = 0.0, l2_steps: int = 1, dq: float = CRUISE_DQ) -> list[Step]:
    """A legal cruise, optionally accruing residual interaction cost.

    ``l2`` is charged on ``l2_steps`` steps only, never on all of them, and the
    distinction is not cosmetic. Because L2 carries a satisfaction indicator, a
    violated step costs the full ``a² = 4.84`` regardless of severity, so a
    trajectory violating L2 on *every* step costs 193.6 against 20 of progress
    and genuinely loses to standing still even under the scalar comparison. That
    is RULEBOOK-V5.0 §6.6's design constraint, not a defect: an R2 sub-rule has
    to fire on a small fraction of steps for the rulebook to be affordable at
    all. The expert measures 0.3978 %; one step in forty here is 2.5 %, already
    six times worse than the human and still comfortably ranked.
    """

    steps = [(0.0, 0.0, 0.0, dq, 0.0)] * STEPS
    for index in range(min(l2_steps, STEPS)):
        steps[index] = (0.0, l2, 0.0, dq, 0.0)
    return steps


def with_relaxation(base: list[Step], *, first: int, count: int, severity: float) -> list[Step]:
    """Charge ``severity`` on L5 for ``count`` steps, leaving L4 untouched.

    Keeping L4 identical is the whole point of the relaxable channel: the detour
    must reach the same place, so the two trajectories tie through L4 and are
    separated only by how much they had to relax.
    """

    out = list(base)
    for index in range(first, first + count):
        l1, l2, l3, dq, _ = out[index]
        out[index] = (l1, l2, l3, dq, severity)
    return out


# ---------------------------------------------------------------------------
# O1 -- legal completion > standing still
# ---------------------------------------------------------------------------


def test_o1_legal_completion_beats_standing_still(measurement: ModuleType) -> None:
    """`TEST-RB5.1-01`. Holds under both comparisons when traffic is empty."""

    stop = standing_still()
    drive = legal_drive()

    assert scalar_return(measurement, drive, lam=2.0, eta=1.0) > scalar_return(
        measurement, stop, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(drive, stop) is True


def test_o1_fails_under_strict_lex_once_traffic_is_present(measurement: ModuleType) -> None:
    """`TEST-RB5.1-01`, the reported failure of `AC-RB5.1-02`.

    This is not a defect to repair. Standing still is exactly 0 on L1-L3 while
    any trajectory moving through traffic accrues some L2 cost, so a strict
    ordering stops at L2 and never reaches progress. It holds for **every**
    hierarchy in which stopping is safe, which is every admissible hierarchy, so
    no rulebook can fix it -- only a thresholded comparison can, by making "both
    within budget" a tie. Asserting the failure keeps the claim honest.
    """

    stop = standing_still()
    drive = legal_drive(l2=TRAFFIC_L2)

    # The scalar arm ranks it correctly: the exchange rate is finite.
    assert scalar_return(measurement, drive, lam=2.0, eta=1.0) > scalar_return(
        measurement, stop, lam=2.0, eta=1.0
    )
    # The strict-lexicographic arm does not, and stops at L2.
    assert strict_lex_prefers_first(drive, stop) is False
    assert channel_returns(stop)[1] < channel_returns(drive)[1]


def test_l2_indicator_makes_continuous_violation_lose_to_standing_still(
    measurement: ModuleType,
) -> None:
    """RULEBOOK-V5.0 §6.6's design constraint, asserted rather than assumed.

    The satisfaction indicator charges `a² = 4.84` for a violated L2 step however
    mild the violation, so a trajectory that is *never* clear of the interaction
    envelope loses to standing still under both comparisons — correctly, since it
    is not driving competently. This bounds how often an L2 sub-rule may fire,
    and it is why the measured 0.3978 % matters rather than merely being small.
    """

    stop = standing_still()
    always_violating = [(0.0, 0.05, 0.0, CRUISE_DQ, 0.0)] * STEPS

    assert scalar_return(measurement, always_violating, lam=2.0, eta=1.0) < scalar_return(
        measurement, stop, lam=2.0, eta=1.0
    )


# ---------------------------------------------------------------------------
# O2 -- completion needing a brief relaxation > standing still
# ---------------------------------------------------------------------------


def test_o2_brief_relaxation_to_complete_beats_standing_still(measurement: ModuleType) -> None:
    """`TEST-RB5.1-02`. The single ordering this restructure buys (§1.1).

    An obstacle forces two seconds of lane-marking contact. Under v5.1 the
    detour ties standing still on L1-L3 and wins on L4 before L5 is ever
    consulted; the relaxation is therefore free *relative to not moving*, which
    is the intended semantics.
    """

    stop = standing_still()
    detour = with_relaxation(legal_drive(), first=10, count=20, severity=0.5)

    assert scalar_return(measurement, detour, lam=2.0, eta=1.0) > scalar_return(
        measurement, stop, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(detour, stop) is True

    stop_l1, stop_l2, stop_l3 = channel_returns(stop)[:3]
    detour_l1, detour_l2, detour_l3, _, detour_l5 = channel_returns(detour)[:5]
    assert (detour_l1, detour_l2, detour_l3) == (stop_l1, stop_l2, stop_l3)
    assert detour_l5 > 0.0


def test_o2_is_what_v50_failed(measurement: ModuleType) -> None:
    """`TEST-RB5.1-02`, the counterfactual that justifies the restructure.

    Under RULEBOOK-V5.0 the same relaxable rules sat in R3, *above* progress. The
    identical detour is then separated from standing still at R3 rather than at
    R4, and standing still wins -- which is the motivating pathology. Modelling
    v5.0's placement is a one-line change here: move the L5 mass into L3.
    """

    stop = standing_still()
    detour_v51 = with_relaxation(legal_drive(), first=10, count=20, severity=0.5)
    detour_v50 = [(l1, l2, l3 + l5, dq, 0.0) for l1, l2, l3, dq, l5 in detour_v51]

    assert strict_lex_prefers_first(detour_v50, stop) is False
    assert strict_lex_prefers_first(detour_v51, stop) is True


# ---------------------------------------------------------------------------
# O3 -- legal route > illegal shortcut, both completing
# ---------------------------------------------------------------------------


def test_o3_legal_route_beats_illegal_shortcut(measurement: ModuleType) -> None:
    """`TEST-RB5.1-03`. Both reach the same place, so L5 decides.

    The two trajectories carry identical L4 totals because both complete the same
    mission, and RULEBOOK-V5.1 §4.2 is what guarantees that: undiscounted, the
    progress total depends on distance covered rather than on speed, so the
    faster shortcut cannot buy an advantage at L4.
    """

    legal = legal_drive()
    shortcut = with_relaxation(legal_drive(), first=5, count=15, severity=0.6)

    assert channel_returns(legal)[3] == pytest.approx(channel_returns(shortcut)[3])
    assert scalar_return(measurement, legal, lam=2.0, eta=1.0) > scalar_return(
        measurement, shortcut, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(legal, shortcut) is True


def test_o3_holds_because_progress_is_speed_independent(measurement: ModuleType) -> None:
    """`TEST-RB5.1-03`, the property the ordering rests on.

    Two legal trajectories covering the same distance at different speeds tie at
    L4 **exactly**. That tie is what lets L5 decide O3 instead of speed deciding
    it, and it is the reason ADR-076 put the time preference at L6 rather than
    folding it into L4: inside L4 it would have priced duration *above* lane
    compliance, so an illegal shortcut could buy its own violation with the time
    it saved.

    L6 still separates the two — the slower one takes more steps — but only
    after L5 has been consulted.
    """

    slow = [(0.0, 0.0, 0.0, 0.2, 0.0)] * 40
    fast = [(0.0, 0.0, 0.0, 0.8, 0.0)] * 10

    assert channel_returns(slow)[3] == pytest.approx(channel_returns(fast)[3])
    assert channel_returns(fast)[5] < channel_returns(slow)[5]


# ---------------------------------------------------------------------------
# O4 -- lane relaxation > collision
# ---------------------------------------------------------------------------


def test_o4_relaxation_beats_collision(measurement: ModuleType) -> None:
    """`TEST-RB5.1-04`. L1 separates, whatever the relaxation costs.

    The relaxation is made deliberately expensive -- maximum severity for half
    the episode -- so that passing this test is a statement about the hierarchy
    rather than about the magnitudes chosen.
    """

    relaxing = with_relaxation(legal_drive(), first=0, count=20, severity=1.0)
    colliding = list(legal_drive())
    colliding[12] = (0.4, 0.0, 0.0, CRUISE_DQ, 0.0)

    assert scalar_return(measurement, relaxing, lam=2.0, eta=1.0) > scalar_return(
        measurement, colliding, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(relaxing, colliding) is True


# ---------------------------------------------------------------------------
# O5 -- waiting at a red light > running it to finish
# ---------------------------------------------------------------------------


def test_o5_waiting_at_red_beats_running_it(measurement: ModuleType) -> None:
    """`TEST-RB5.1-05`. `signal` is non-relaxable, so it sits above progress.

    The waiting trajectory is given *less* progress than the running one, which
    is the honest construction: waiting really does cost mission progress within
    a bounded horizon. It must still win, because L3 is consulted first.
    """

    waiting = [(0.0, 0.0, 0.0, 0.0, 0.0)] * 15 + [(0.0, 0.0, 0.0, CRUISE_DQ, 0.0)] * 25
    running = [(0.0, 0.0, 0.8, CRUISE_DQ, 0.0)] * 5 + [(0.0, 0.0, 0.0, CRUISE_DQ, 0.0)] * 35

    assert channel_returns(running)[3] > channel_returns(waiting)[3]
    assert scalar_return(measurement, waiting, lam=2.0, eta=1.0) > scalar_return(
        measurement, running, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(waiting, running) is True


# ---------------------------------------------------------------------------
# O6 -- necessary relaxation > gratuitous relaxation
# ---------------------------------------------------------------------------


def test_o6_necessary_relaxation_beats_gratuitous(measurement: ModuleType) -> None:
    """`TEST-RB5.1-06`. Equal completion, so the smaller relaxation wins."""

    necessary = with_relaxation(legal_drive(), first=10, count=5, severity=0.5)
    gratuitous = with_relaxation(legal_drive(), first=10, count=25, severity=0.5)

    assert channel_returns(necessary)[3] == pytest.approx(channel_returns(gratuitous)[3])
    assert scalar_return(measurement, necessary, lam=2.0, eta=1.0) > scalar_return(
        measurement, gratuitous, lam=2.0, eta=1.0
    )
    assert strict_lex_prefers_first(necessary, gratuitous) is True


def test_o6_requires_l5_to_sum_rather_than_max(measurement: ModuleType) -> None:
    """`TEST-RB5.1-06`, the reason §3.3 aggregates L5 by a normalized sum.

    A detour that crosses a solid line **and** enters the opposing carriageway
    must cost more than one that only crosses the line. Under `max` the second
    concurrent violation is free and the two are indistinguishable, which breaks
    O6 in exactly the case it exists to catch.
    """

    solid_only = 0.6
    solid_and_opposing = (0.6, 0.5, 0.0)

    summed = sum(solid_and_opposing) / measurement.V51_L5_DENOMINATOR
    maxed = max(solid_and_opposing)

    assert summed > solid_only / measurement.V51_L5_DENOMINATOR
    assert maxed == pytest.approx(solid_only)


# ---------------------------------------------------------------------------
# Cross-cutting guarantees the six orderings depend on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0, 2.0])
def test_eta_reaches_only_l5(measurement: ModuleType, eta: float) -> None:
    """`AC-RB5.1-10` / `TEST-RB5.1-13`. `eta` must not move L1-L4.

    If it did, the exchange rate between progress and relaxable compliance would
    silently reprice the safety channels, and every figure measured at one `eta`
    would be incomparable with every figure measured at another.
    """

    trajectory = with_relaxation(legal_drive(l2=TRAFFIC_L2), first=4, count=8, severity=0.7)
    baseline = scalar_return(measurement, trajectory, lam=2.0, eta=0.0)
    scored = scalar_return(measurement, trajectory, lam=2.0, eta=eta)
    l5_mass = channel_returns(trajectory)[4]

    expected = baseline - eta * l5_mass * (measurement.V51_STEP_DT_S / measurement.V51_T_REF_S)
    assert scored == pytest.approx(expected)


def test_l4_is_invariant_to_repeated_traversal(measurement: ModuleType) -> None:
    """`TEST-RB5.1-07`. The property that replaced the running maximum.

    ADR-073 first used the monotone advance so that re-covering credited ground
    earned nothing. The signed form gets the same invariance by compensation:
    covering a stretch forward and back contributes `+x − x = 0`, so the return
    is invariant to how many times it is traversed — and, unlike the running
    maximum, it needs no state the observation does not carry.

    Asserting it here is what stops the memory being reintroduced by someone who
    reads only the first paragraph of that ADR.
    """

    straight = [(0.0, 0.0, 0.0, 0.4, 0.0)] * 10
    # Same net displacement, reached after three forward-and-back excursions.
    oscillating: list[Step] = []
    for _ in range(3):
        oscillating += [(0.0, 0.0, 0.0, 0.4, 0.0)] * 5
        oscillating += [(0.0, 0.0, 0.0, -0.4, 0.0)] * 5
    oscillating += [(0.0, 0.0, 0.0, 0.4, 0.0)] * 10

    # L4 is exactly invariant -- that is ADR-073's property and it survives
    # ADR-076 untouched, because the time cost was kept out of this channel.
    assert channel_returns(straight)[3] == pytest.approx(channel_returns(oscillating)[3])
    # The wasted steps are charged at L6 instead, which is where duration
    # belongs: below every compliance level.
    assert channel_returns(straight)[5] < channel_returns(oscillating)[5]
    assert scalar_return(measurement, straight, lam=2.0, eta=1.0) > scalar_return(
        measurement, oscillating, lam=2.0, eta=1.0
    )


def test_reverse_motion_is_penalised(measurement: ModuleType) -> None:
    """`TEST-RB5.1-07`. RULEBOOK-V5.0 §5.6 deleted `wrongway` partly because R4's
    margin "is negative for negative route advance". A non-negative L4 had
    quietly removed that half of the justification; the signed form restores it.
    """

    forward = [(0.0, 0.0, 0.0, 0.4, 0.0)] * 10
    backward = [(0.0, 0.0, 0.0, -0.4, 0.0)] * 10
    standing = standing_still()[:10]

    assert scalar_return(measurement, backward, lam=2.0, eta=1.0) < scalar_return(
        measurement, standing, lam=2.0, eta=1.0
    )
    assert scalar_return(measurement, standing, lam=2.0, eta=1.0) < scalar_return(
        measurement, forward, lam=2.0, eta=1.0
    )


def test_selected_weights_are_rank_preserving(measurement: ModuleType) -> None:
    """`AC-RB5.1-06`. The published pair must satisfy §5.4 at every level."""

    assert measurement.v51_is_rank_preserving(
        measurement.FINAL_PRIORITY_BASE,
        measurement.FINAL_SEVERITY,
        measurement.FINAL_FLAT_TIE_BREAKER,
        2.0,
        1.0,
    )
    # `lambda4 + 0.1 * eta < a` is the binding constraint at k = 3; 2.2 exactly
    # is the supremum of the violating case and must therefore be refused.
    assert not measurement.v51_is_rank_preserving(
        measurement.FINAL_PRIORITY_BASE,
        measurement.FINAL_SEVERITY,
        measurement.FINAL_FLAT_TIE_BREAKER,
        2.2,
        0.0,
    )


def test_progress_weight_cannot_overturn_a_non_relaxable_violation(
    measurement: ModuleType,
) -> None:
    """`TEST-RB5.1-10`. Per-step dominance of L3 over L4, at the worst case.

    The worst case is a maximal progress step, `delta_q = 1`, against the
    smallest possible L3 violation. This is precisely what the §5.4 bound buys,
    and it is why `lambda4` cannot be raised freely.
    """

    violating = measurement.v51_reward(
        l1=0.0, l2=0.0, l3=1e-6, delta_q=1.0, l5=0.0, lam=2.0, eta=1.0
    )
    compliant_no_progress = measurement.v51_reward(
        l1=0.0, l2=0.0, l3=0.0, delta_q=0.0, l5=1.0, lam=2.0, eta=1.0
    )
    assert compliant_no_progress > violating


# ---------------------------------------------------------------------------
# ADR-076 -- L6 `progress_rate`, the sixth level
# ---------------------------------------------------------------------------

# The reference shortcut of RULEBOOK-V5.1 §4.6, chosen as the *cheapest* illegal
# shortcut saving the *most* time, so a more aggressive one is easier for O3.
REFERENCE_EPISODE_STEPS = 200
REFERENCE_SHORTCUT_STEPS = 160
REFERENCE_L5_STEPS = 30
REFERENCE_L5_SEVERITY = 1.0 / 3.0
# The panel's mean *mission* span, 90.17 m, in units of D_REF = 2.2222 m. Not
# the assigned route (171.98 m): the mission runs from `s_start_m` to
# `s_goal_m`, which is 52 % of it.
REFERENCE_MISSION_Q = 90.17 / 2.2222


def completing_run(steps: int, *, l5_steps: int = 0, severity: float = 0.0) -> list[Step]:
    """A run covering the whole reference mission in ``steps`` steps."""

    advance = REFERENCE_MISSION_Q / steps
    assert advance <= 1.0, "fixture would be clipped, which would confound the test"
    return [
        (0.0, 0.0, 0.0, advance, severity if index < l5_steps else 0.0) for index in range(steps)
    ]


def test_l4_ties_exactly_between_two_completing_runs(measurement: ModuleType) -> None:
    """`TEST-RB5.1-15`. The theorem ADR-076 exists to protect.

    The whole point of putting duration at L6 rather than inside L4 is that this
    stays an *exact* tie. With a time cost folded into L4 the shortcut would
    arrive with a strictly larger mission channel, and O3 would hold only for
    shortcuts whose L5 exposure happened to exceed the time they saved — a
    calibration where there had been a proof.
    """

    legal = completing_run(REFERENCE_EPISODE_STEPS)
    shortcut = completing_run(REFERENCE_SHORTCUT_STEPS)

    assert channel_returns(legal)[3] == pytest.approx(channel_returns(shortcut)[3])


def test_o3_holds_against_the_reference_shortcut(measurement: ModuleType) -> None:
    """`TEST-RB5.1-16`. O3 where the shortcut actually arrives sooner.

    This is the case the equal-length fixtures cannot reach. Under strict
    lexicographic ordering it is decided at L5, before L6 is ever consulted,
    which is why the weight on L6 is unconstrained in that arm.
    """

    legal = completing_run(REFERENCE_EPISODE_STEPS)
    shortcut = completing_run(
        REFERENCE_SHORTCUT_STEPS,
        l5_steps=REFERENCE_L5_STEPS,
        severity=REFERENCE_L5_SEVERITY,
    )

    assert strict_lex_prefers_first(legal, shortcut) is True
    assert scalar_return(measurement, legal, lam=2.0, eta=1.0) > scalar_return(
        measurement, shortcut, lam=2.0, eta=1.0
    )


def test_lambda6_stays_below_its_o3_bound(measurement: ModuleType) -> None:
    """`TEST-RB5.1-17`. The scalar arm is the only one where L6 is bounded.

    Summing every channel re-couples what the ordering separates: the shortcut
    gains `lambda6` per step saved and pays `eta * c_L5 * (dt / T_REF)` per step
    relaxed. The lexicographic arms never evaluate this expression and are not
    bound by it.
    """

    eta = 1.0
    steps_saved = REFERENCE_EPISODE_STEPS - REFERENCE_SHORTCUT_STEPS
    dt_ratio = measurement.V51_STEP_DT_S / measurement.V51_T_REF_S
    exposure = eta * REFERENCE_L5_SEVERITY * dt_ratio * REFERENCE_L5_STEPS
    bound = exposure / (steps_saved * dt_ratio)

    assert bound == pytest.approx(0.25)
    assert 0.0 < LAMBDA6 < bound
    assert max(measurement.V51_LAMBDA6_GRID) <= bound


def test_l6_ranks_by_duration_and_leaves_l4_alone(measurement: ModuleType) -> None:
    """`TEST-RB5.1-18`. `sum c_L6 = T - Q` for any completing run.

    The identity is what makes L6 a duration ordering rather than a speed
    heuristic, and it holds for every completion time, which is why the O3 bound
    on `lambda6` does not depend on the mission's length.
    """

    for steps in (200, 160, 120, 80):
        run = completing_run(steps)
        assert channel_returns(run)[5] == pytest.approx(steps - REFERENCE_MISSION_Q)
        assert channel_returns(run)[3] == pytest.approx(REFERENCE_MISSION_Q)


def test_standing_still_and_reversing_cost_the_maximum_at_l6(
    measurement: ModuleType,
) -> None:
    """`TEST-RB5.1-19`. ADR-076's degenerate cases, asserted.

    A stopped ego advances nothing and a reversing one advances backwards; both
    clip to zero advance and pay `c_L6 = 1`. The below-standstill diagnostic must
    therefore compare against `-lambda6 * (dt/T_REF) * T` rather than against 0.
    """

    stop = standing_still()
    reverse = [(0.0, 0.0, 0.0, -0.4, 0.0)] * len(stop)

    assert channel_returns(stop)[5] == pytest.approx(float(len(stop)))
    assert channel_returns(reverse)[5] == pytest.approx(float(len(reverse)))

    label = next(
        f"v51_{name}"
        for name, lam, eta, lam6 in measurement.v51_weight_grid()
        if (lam, eta, lam6) == (2.0, 1.0, LAMBDA6)
    )
    expected = -LAMBDA6 * (measurement.V51_STEP_DT_S / measurement.V51_T_REF_S) * len(stop)
    assert measurement.v51_standstill_return(label, len(stop)) == pytest.approx(expected)
    assert measurement.v51_standstill_return("final", len(stop)) == 0.0


def test_l6_does_not_disturb_the_orderings_above_it(measurement: ModuleType) -> None:
    """`TEST-RB5.1-20`. Adding a level below L5 cannot break O1-O6.

    None of the six orderings is decided by a tie at the bottom, so a new last
    level can only refine ties that were previously unresolved. Checked on the
    two orderings that come closest to depending on it.
    """

    legal = legal_drive()
    gratuitous = with_relaxation(legal_drive(), first=5, count=10, severity=0.5)
    # O6: the gratuitous relaxation ties at L4 and L6 and loses at L5.
    assert channel_returns(legal)[3] == pytest.approx(channel_returns(gratuitous)[3])
    assert channel_returns(legal)[5] == pytest.approx(channel_returns(gratuitous)[5])
    assert strict_lex_prefers_first(legal, gratuitous) is True

    # O1 still fails at L2 under a strict ordering, before L6 is reached.
    stop = standing_still()
    traffic = legal_drive(l2=TRAFFIC_L2, l2_steps=1)
    assert strict_lex_prefers_first(traffic, stop) is False


def test_selected_weights_remain_rank_preserving_with_l6(measurement: ModuleType) -> None:
    """`TEST-RB5.1-21`. `lambda6` enters the §5.4 tail and must fit inside it."""

    assert measurement.v51_is_rank_preserving(
        measurement.FINAL_PRIORITY_BASE,
        measurement.FINAL_SEVERITY,
        measurement.FINAL_FLAT_TIE_BREAKER,
        2.0,
        1.0,
        LAMBDA6,
    )
    # 2.0 + 0.1 + 0.02 = 2.12 < a = 2.2; pushing lambda6 past the slack fails.
    assert not measurement.v51_is_rank_preserving(
        measurement.FINAL_PRIORITY_BASE,
        measurement.FINAL_SEVERITY,
        measurement.FINAL_FLAT_TIE_BREAKER,
        2.0,
        1.0,
        2.0,
    )
