# ADR-028: Scenario ACL Generate-Arm Eligibility Renormalization

- Status: APPROVED
- Date: 2026-07-27
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-27
- Supersedes: nothing; amends `automatic_curriculum_learning_v1.1_specification.md` REQ-002/REQ-003 wording only
- Affected specifications: Automatic Curriculum Learning v1.1 (`ACL-SN-EMA-001`)
- Affected ExecPlan: `docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md`

## Context

FIND-001 (see the linked ExecPlan): the ACL v1.1 Generate arm draw excludes
catalog records already held by the scenario buffer. Once a semantic arm's
entire frozen pool has been absorbed into the buffer, every MAB draw of that
arm finds zero fresh candidates and silently falls back to Replay. Replay never
updates the MAB (REQ-001), so the arm's EMA score freezes at whatever value it
last held — observed live at `0.715461969165923` for `A4_vru`, frozen from MAB
update 1420 to 3118 while the arm kept the highest Generate probability
(~33.6%). The state is absorbing: a frozen high score cannot decay, so the arm
keeps being drawn, keeps degrading, and keeps not updating.

This is a deviation from, not a gap in, the approved specification: REQ-001
requires every Generate to reach the MAB, and REQ-003 requires an empirical
40/60 Generate/Replay split after warm-up. The live run's Generate share fell
from 0.423 to 0.279 once `A4_vru` exhausted.

Neither of the ACL v1.1 design's two inspirations hits this case. Peng et al.
(2024)'s MAB arms are procedurally generated (vehicle counts), so a draw always
returns a fresh scenario. Abouelazm et al. (2025)'s learning-potential buffer
scores individual scenarios, not semantic categories, so replay itself updates
the relevant score. Neither combines a MAB over semantic arms with a frozen,
finite per-arm catalog, which is this project's specific composition.

## Decision

Before drawing a Generate arm, compute per-arm eligibility: an arm is eligible
when at least one of its frozen catalog records is not already held by the
scenario buffer (and not already claimed elsewhere in the same vectorized
batch). `ScenarioArmBandit.probabilities()` and `sample_arm()` accept an
optional `eligible_mask`; an ineligible arm receives exactly probability zero,
and the `eta/K` exploration floor (REQ-002) is renormalized over the eligible
subset instead of over all `K` arms, so every eligible arm keeps a non-zero
floor. Omitting the mask reproduces the unrestricted REQ-002 distribution
unchanged.

`select_acl_slot_decision` (`src/thesis_rl/curriculum/scenario_acl/selection.py`)
computes the eligibility mask once per slot decision, before the arm draw, at
no extra RNG cost: the mask needs no random draw, so the RNG call
order — replay coin flip, then arm draw, then fresh-record draw — is
unchanged. A drawn arm is now always eligible, so it always finds a fresh
record; the previous "arm sampled, no candidate found" fallback becomes
unreachable in the normal case and is kept only as a defensive check.

The full-exhaustion corner case — every arm simultaneously out of fresh
records — is unaffected by this decision: it still degrades to Replay (or
raises, if the buffer is also empty), exactly as before, because no eligible
arm exists to renormalize over.

Diagnostics: `src/thesis_rl/curriculum/scenario_acl/driver.py` logs a
`scenario_acl_generate_pool_eligibility_changed` event exactly when the set of
ineligible arms changes (not once per episode, to avoid log spam over a long
exhaustion), and each `mab_history.jsonl` chunk snapshot carries a
`generate_ineligible_arms` field.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Neutral or penalizing EMA update on a degraded draw | Keeps every draw "counted" | Invents a learning-potential value for an episode that never ran, which ADR-014 and REQ-001 prohibit | Rejected |
| Raise a fatal error whenever an arm's pool exhausts | Faithful to the existing "missing frozen record is fatal" wording | Would abort every sufficiently long ACL run, since the smallest-pool arm exhausts as a matter of course, not as an exceptional failure | Rejected |
| Keep the silent Replay fallback, only log it | Minimal code change | Leaves REQ-003's 40/60 split violated indefinitely once any arm exhausts | Rejected |
| Renormalize over eligible arms only (selected) | Restores REQ-001 ("every Generate updates the MAB") and REQ-003 (40/60 split) without inventing feedback | Changes the Generate distribution and therefore the curriculum trajectory once an arm exhausts; not bit-comparable with pre-fix runs | Selected |

## Consequences

REQ-002's exploration floor now reads "every *eligible* arm has probability at
least `eta/K_eligible`" instead of "every arm has probability at least
`eta/K`". An arm's floor share grows as fewer arms remain eligible. This is an
approved, recorded deviation from the literal REQ-002 wording (see the ExecPlan
`DEV-EXH-001`), not a silent reinterpretation.

Runs before this fix and runs after it are not bit-comparable once any arm
exhausts its pool: the Generate arm distribution differs from that point
onward. Runs completed before this ADR remain usable for analyses that do not
depend on which semantic arm the curriculum favoured during Generate (see the
ExecPlan `DEC-EXH-003`); they carry an explicit caveat in
`docs/project_index.md`.

No change to the frozen ScenarioNet catalog, the six semantic arms, the LP
formulas, or the replay ranking contract (0.70/0.30, rank exponent 1.0,
staleness offset 1).

## Validation And Traceability

`TEST-EXH-001` through `TEST-EXH-007` in
`docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md`
§9, implemented in `tests/test_scenario_acl_arm_exhaustion.py` and
`tests/test_scenario_acl_mab.py`. Full repository suite passes
(1024 passed, 7 skipped as of 2026-07-27).

## Approval Record

- Approved by: user
- Approval evidence: explicit approval of `DEC-EXH-001` option A and
  `DEC-EXH-002` in the Claude Code conversation on 2026-07-27, following a
  request to compare the design against Peng et al. (2024) and
  Abouelazm et al. (2025)
- `DEC-EXH-003` (existing affected runs kept, with caveat) and `DEC-EXH-004`
  (no dataset rebuild for now) approved the same date; both are informational
  and require no code change
