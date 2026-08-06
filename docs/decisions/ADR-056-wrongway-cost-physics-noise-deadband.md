# ADR-056: Physics-noise deadband on `wrongway` cost, superseding ADR-050's scope

- Status: Approved
- Date: 2026-08-06
- Approval evidence: explicit user approval in this conversation ("sì
  procedi e correggi il problema grazie"), after the user explicitly
  instructed that previously approved decisions (`ADR-050`) are to be
  re-evaluated on their merits against new evidence, not treated as
  immutable, and after a full walkthrough of the mechanism was confirmed
  understood.
- Affected specification: `docs/specifications/rulebook_v4.12_specification.md`
  (amends `rulebook_v4.7_specification.md` §7.3.2).

## Context

`ADR-050` (2026-08-01) fixed a visible symptom: the `wrongway` diagnostic
`status` marker flickered `[ok]`/`[!]` on a stationary, correctly parked
ego, caused by physics-solver residual velocity noise (order
`1e-3`–`1e-1 m/s`) occasionally carrying a hairline reverse-longitudinal
component. That ADR added `WRONGWAY_STATUS_SPEED_EPSILON_MPS=0.1` to
`status` only, explicitly leaving `cost` untouched, reasoning: *"cost stays
the same continuous function... so the scalarizer's input... [is]
unaffected."*

While drafting `incoming/rulebook_scalarization_v1.1_specification_UNDER_REVIEW.md`
(a candidate new scalarization mode that embeds each margin's severity at
full priority weight, rather than diluting it through a shared tie-breaker),
this claim was re-examined rather than assumed. The scalarizer's own
numerical canonicalization tolerance is `1e-8` (`SCAL-V1.0` §3.3) — many
orders of magnitude below the `0.1 m/s`-scale noise `ADR-050` measured.
Concretely: the same residual noise that flickered the status marker
already produces a strictly positive `cost` today, which produces a
non-zero canonical macro margin `m_3`, which — under the *currently
approved* `bounded_satisfaction_rank` default, not only under the reviewed
new mode — flips the categorical indicator `I_3` and applies a full-weight
categorical reward penalty (`priority_base^1 * (pattern - 1) = -2.01`, not
diluted; only the continuous margin contribution is diluted by that
formula's shared tie-breaker) to a correctly stopped ego, on roughly half
of standstill steps given the noise's observed sign-alternating character.

`ADR-050`'s narrow claim was accurate about what that ADR itself changed.
It did not establish that `cost` was already correct, and left this
reward-affecting exposure unaudited. Rulebook `v4.7` §7.3.2's own prose
already promises `q_wrongway = 0` when "ego fermo" (ego at rest); the
implementation does not reliably deliver that under realistic physics
noise. This is therefore a correctness correction against the existing
authoritative contract, not a new behavioral policy choice, and reuses a
noise-floor constant already frozen twice elsewhere for the identical
physics phenomenon (`RSS_STANDSTILL_SPEED_MPS`, `ADR-048`;
`WRONGWAY_STATUS_SPEED_EPSILON_MPS`, `ADR-050`) rather than inventing a new
threshold.

## Decision

Give `cost` the same `0.1 m/s` deadband, as a continuous reparameterization
(not a discontinuous cutoff) so the existing `0` at rest / `1` at the speed
cap boundary behavior is preserved and no new jump is introduced at the
threshold:

\[
u=[-v_\parallel]_+,\quad \varepsilon_v=0.1\ \mathrm{m/s},
\]
\[
\mathrm{cost}=
\begin{cases}
0, & u\le\varepsilon_v\\
\operatorname{clip}\!\left(\dfrac{u-\varepsilon_v}{v_{\max,e}-\varepsilon_v},0,1\right), & u>\varepsilon_v
\end{cases}
\]

`status` is simplified to derive directly from the now-deadbanded `cost`
(`VIOLATED` iff `cost > 0.0`), removing the separate, now-redundant
`WRONGWAY_STATUS_SPEED_EPSILON_MPS` constant and its independent check —
one noise floor instead of two independently maintained ones.

This explicitly supersedes `ADR-050`'s scope conclusion ("cost is out of
scope"), while its diagnosis and measured noise magnitude remain the
empirical basis for this fix.

## Consequences

- `wrongway`'s macro margin `m_3` is now exactly `0` during standstill or
  near-standstill states, matching v4.7 §7.3.2's documented intent and
  removing a full-weight categorical reward penalty that was previously
  applied to a correctly stopped ego on a noise-dependent, roughly-random
  fraction of standstill steps.
- Episode returns from scenarios with stops (the large majority) computed
  under any run prior to this version are not directly comparable to runs
  after it.
- Genuine reverse motion (`u` well above `0.1 m/s`) is still detected;
  `status == VIOLATED` and `cost > 0.0` in that regime, unchanged in kind.
- Removes duplicated tolerance maintenance between `cost` and `status`.
- Does not depend on, and does not block on, the separate `SCAL-V1.1`
  scalarization amendment review; it corrects the currently deployed
  `SCAL-V1.0` default independently.

Regression tests: `test_wrongway_deadband_zeroes_cost_for_standstill_noise`,
`test_wrongway_cost_deadband_is_continuous_at_the_boundary`,
`test_wrongway_reaches_full_cost_at_speed_cap`,
`test_wrongway_status_derives_from_cost_not_a_separate_epsilon`
(`tests/test_rulebook_v2_road.py`); the pre-existing
`test_wrongway_status_has_a_deadband_against_standstill_physics_noise` is
updated because its old assertion `result.cost > 0.0` for a `0.01 m/s`
noisy-reverse ego encoded the exact defect this ADR corrects.
