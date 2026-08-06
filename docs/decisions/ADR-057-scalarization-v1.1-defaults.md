# ADR-057: `SCAL-V1.1` default mode, compression default, and algebraic-guard scope

- Status: Approved
- Date: 2026-08-06
- Approval evidence: explicit user approval in this conversation ("Top.
  Avevo letto le altre decisioni e approvo i suggerimenti, quindi procedi
  pure con l'implementazione automatica del piano"), approving the
  recommendation column of `docs/specifications/rulebook_scalarization_v1.1_specification.md`
  §15 for each open decision.
- Affected specification: `docs/specifications/rulebook_scalarization_v1.1_specification.md`
  (`SCAL-V1.1`, amends `SCAL-V1.0`).

## Context

`SCAL-V1.1` added a fourth scalarization mode,
`bounded_priority_weighted_rank`, and an optional post-hoc `symlog` reward
compression stage, both left as open decisions pending user review
(`DEC-SCAL11-001`, `002`, `004`; a fourth, `DEC-SCAL11-003`, was closed
independently by `ADR-056`/`rulebook_v4.12` before this ADR). Each open
decision carried a recommendation in the specification; the user reviewed
and approved all of them together rather than case by case.

## Decision

- **`DEC-SCAL11-001` = `A`.** `bounded_priority_weighted_rank`
  (`priority_base=3.0`) becomes the default scalarization mode for new
  conformant configuration, replacing `bounded_satisfaction_rank` as the
  default. Rationale: §1's diluted-signal finding (a shared,
  equally-weighted tie-breaker gives `m_1` and `m_4` the same continuous
  weight) is addressed only by the new mode; `DEC-SCAL11-003`'s
  precondition (the `wrongway` cost/status tolerance defect) is already
  closed. `bounded_satisfaction_rank` remains fully available and
  unchanged for any run that explicitly selects it.
- **`DEC-SCAL11-002` = `B`.** `reward_compression.mode` defaults to `none`.
  `symlog` remains available as an explicit opt-in. Rationale: §7.7's
  hierarchy-separation-vs-variance trade-off is unquantified in this
  project's exact setting (model-free scalar-head PPO/TD3/SAC, not
  DreamerV3's model-based twohot setting) and should be measured through an
  ablation before being made the default, not assumed.
- **`DEC-SCAL11-004` = `B`.** The algebraic-guard regression
  (`AC-SCAL11-004`) is scoped to `bounded_priority_weighted_rank` only. It
  is not retroactively extended to assert the analogous dominance
  inequalities for the three existing `SCAL-V1.0` modes. Rationale: out of
  scope for this amendment; recorded as a candidate follow-up for a
  separately scoped `SCAL-V1.0` test-coverage improvement, not a blocking
  gap of this document.

## Consequences

- Any configuration that does not explicitly set `scalarization.mode`
  now selects `bounded_priority_weighted_rank` with `priority_base=3.0`,
  not `bounded_satisfaction_rank` with `priority_base=2.01`. This is a
  reward-semantics change under `SCAL-V1.0`/`SCAL-V1.1` §11.3: a new run
  identity, empty rollout/replay state, and separate result labeling are
  required for any run started after this change relative to one started
  before it with an implicit default.
- Existing runs and configurations that explicitly set
  `scalarization.mode: bounded_satisfaction_rank` are unaffected.
- `reward_compression.mode` defaults to `none`; enabling `symlog` remains
  an explicit, logged opt-in per §10.3/REQ-SCAL11-006, never silently
  pooled with hierarchy-only results.
- `SCAL-V1.0`'s three existing modes keep their original acceptance
  coverage; no new regression is required for them by this decision.

Regression tests: `AC-SCAL11-001` (default mode selection),
`AC-SCAL11-004` (algebraic guard, scoped to the new mode only) in
`tests/test_scalarization.py`; checkpoint/resume identity coverage for the
new default in `tests/test_scalarization_wiring.py`.
