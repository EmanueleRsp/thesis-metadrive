# Scenario ACL Exhausted-Arm MAB Starvation Bugfix ExecPlan

## 1. Metadata

- Feature/plan ID: `ACL-SN-EXH-001`
- Authoritative specification: `docs/specifications/automatic_curriculum_learning_v1.1_specification.md` (`ACL-SN-EMA-001`), `APPROVED`, `Authoritative: YES`, REQ-002/REQ-003 amended 2026-07-27 by ADR-028
- Status: `IMPLEMENTED`
- Created/updated: 2026-07-27
- Branch: `claude/exciting-rosalind-dc5ccc`
- Related ADRs: ADR-014 (LP-only usefulness), ADR-016 (vectorized execution), ADR-024 (runtime scenario data-abort), ADR-028 (this correction: Generate-arm eligibility renormalization)
- Related ExecPlans: `automatic_curriculum_learning_v1.1_exec_plan.md`, `scenario_acl_vectorized_execution_v1_exec_plan.md`, `runtime_scenario_data_abort_v1_exec_plan.md`

## 2. Objective And Scope

A semantic arm whose frozen Generate pool has been fully absorbed by the
scenario buffer stops producing Generate episodes, therefore stops producing MAB
feedback, therefore keeps its last EMA score forever. Because the score cannot
decay, the arm retains its Generate selection probability indefinitely and the
teacher spends that share of the Generate budget on draws that silently become
Replays. The objective is to make this state impossible to enter silently:
either the selector stops offering an unusable arm, or the run fails explicitly,
or the arm's score reflects its unusability.

In scope: the ACL Generate/Replay slot decision, the MAB feedback contract, ACL
diagnostics for the degraded path, a deterministic regression test, and the
approval record for the behavioral choice.

Out of scope: the frozen ScenarioNet catalog and split (their A4 pool size is a
dataset-curation matter, tracked separately under `live_training_reliability_v1`),
the replay ranking contract, the LP formulas, and ADR-024's data-abort path
(disproved as the cause, see §4).

Compatibility: any correction changes observable curriculum behavior and the
`mab_history.jsonl` trajectory, so runs before and after the fix are not
directly comparable for curriculum-sensitive claims.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-A` | For every Generate selection, run the frozen scenario and apply the EMA update immediately in the parent process; Replay does not update the MAB. | ACL v1.1 §7 "Algorithmic contract", REQ-001 |
| `REQ-B` | After warm-up, select Generate with probability 0.40 and Replay with probability 0.60. | ACL v1.1 REQ-003 |
| `REQ-C` | A missing frozen record is fatal. | ACL v1.1 §10 "Errors and diagnostics" |
| `REQ-D` | Each committed Generate records arm, pre/post score, normalized LP, probability, and update count; each episode records mode, scenario identity, and source. | ACL v1.1 §10 |
| `REQ-E` | An aborted (ADR-024) episode yields no LP, usefulness, MAB feedback, insert, or update. | ADR-024 "Decision" |

## 4. Current Repository Analysis

### 4.1 Defect mechanism (`VERIFIED`)

The vectorized slot decision now lives in
`src/thesis_rl/curriculum/scenario_acl/selection.py::select_acl_slot_decision`
(extracted verbatim from the closure previously inlined at
`src/thesis_rl/curriculum/scenario_acl/driver.py:586-672`; see §7.1). Its
Generate branch computes:

```
effective_excluded = batch-excluded UIDs | {record.scenario_id for record in buffer.records()}
candidates = [r for r in train_records if r.primary_arm == arm_name and r.scenario_uid not in effective_excluded]
```

When `candidates` is empty and the buffer is non-empty, the function returns a
**Replay** selection built from `buffer.sample_replay(...)`, carrying the
replayed record's arm — not the arm the MAB drew. The commit path at
`src/thesis_rl/curriculum/scenario_acl/driver.py:713-742` (`commit_event`)
calls `bandit.update(...)` only when `completion.selection.mode == "generate"`.
The MAB therefore never observes the draw.

Because Generate means "a record the buffer has never scored", an arm whose
entire frozen pool has been inserted into the buffer has zero fresh candidates
**permanently**: its buffer records are only released by capacity eviction, and
a high-LP arm's records are the least likely to be evicted. The state is
absorbing.

### 4.2 Live evidence (`VERIFIED`)

Run `EXP_thesis_RP_thesis_CUR_scenario_acl_scenarionet_REW_scalar_reward/td3_sb3/seed_0/20260726_055617`
(`/scratch/e.respino/thesis-metadrive/outputs/...`):

| Evidence | Value |
|---|---|
| `A4_vru` EMA score, MAB update 1420 → 3118 (chunks 5 → 13) | `0.715461969165923`, bit-identical |
| Other arms' final scores | `[0.156, 0.162, 0.136, 0.240, ..., 0.094]` |
| `A4_vru` Generate probability at final scores (`tau=0.5`, `eta=0.2`) | `0.3357` |
| Committed `A4` Generate episodes | 334, over 333 distinct scenario UIDs |
| Last `A4` Generate | collection tick 2433, `waymo:training_20s:b7201d26b2adc079` |
| `A4` Generate episodes after tick 2433 | 0 |
| `A4` Replay episodes after tick 2433 | 1554 (the largest arm share) |
| `A4_vru` records in the final scenario buffer | 333 |
| `A4_vru` frozen train pool (`artifacts/scenarionet/split_manifest.yaml`) | waymo 333, pg 0 (declared `structural_empty_cells`) |
| Generate share of committed episodes before tick 2433 | 0.423 |
| Generate share of committed episodes after tick 2433 | 0.279 |
| Runtime quarantine set at end of run | 4 Waymo UIDs |
| `scenario_acl_reset_timing` events vs committed episodes | 9501 vs 9496 |

The buffer holds exactly the 333 distinct UIDs the arm ever generated, i.e. the
whole pool. `arms.py:40-43` forces `A4_vru` to `source_probability={"waymo": 1.0,
"pg": 0.0}` because PG cannot produce VRU scenarios, so no PG backfill exists
and 333 is the hard ceiling.

### 4.3 Hypotheses evaluated

| Hypothesis | Verdict | Evidence |
|---|---|---|
| ADR-024 quarantine progressively removed A4's scenarios | `REJECTED` as the cause | Only 4 UIDs quarantined in the whole run; 9501 resets produced 9496 committed episodes, so aborts are rare and the arm was not being selected-then-aborted |
| A4 was still selected for Generate but its episodes were excluded | `REJECTED` | Reset/commit balance above; a systematic abort loop would create thousands of resets without commits |
| A4 stopped being offered a fresh record and silently fell back to Replay | `CONFIRMED` | Pool 333 = distinct Generate UIDs 333 = buffer A4 records 333; Generate share drops 0.423 → 0.279 at the cutoff; A4 becomes the dominant Replay arm |
| The Waymo-only source forcing in `arms.py:40-43` contributes | `CONFIRMED` as an amplifier, not the cause | It caps A4's pool at 333 with no cross-source backfill, so A4 exhausts first; the defect would eventually reach any arm |

### 4.4 Specification conformance

| Requirement | Status | Note |
|---|---|---|
| `REQ-A` | `VIOLATED` | A drawn Generate produced no scenario run and no EMA update |
| `REQ-B` | `VIOLATED` | Post-exhaustion Generate share 0.279 against the frozen 0.40 |
| `REQ-C` | `VIOLATED in spirit` | A missing frozen record is fatal only when the buffer is empty; otherwise it degrades silently |
| `REQ-D` | `VIOLATED` | No artifact, event, log, or counter distinguishes an exhaustion-driven Replay from a scheduled one |
| `REQ-E` | `RESPECTED` | ADR-024's contract is correct and is not implicated |

The defect is therefore a **specification deviation**, not a specification gap.

### 4.5 Blast radius (`VERIFIED`)

Every ACL ScenarioNet run reaches this state once any arm's pool is absorbed;
the arm that reaches it first is `A4_vru` because it has the smallest
single-source pool. Current occupancy of A4's 333-record pool:

| Run | A4 records in buffer | Status |
|---|---|---|
| `td3_sb3/seed_0/20260726_055617` | 333 / 333 | Exhausted; run ended 2026-07-26 16:34 on an unrelated async-eval state-dict error |
| `td3_sb3/seed_0/20260726_225902` | 332 / 333 | One record from exhaustion; interrupted (Ctrl+C) 2026-07-27 10:15 |
| `sac_sb3/seed_0/20260726_225902` | 224 / 333 | Interrupted (Ctrl+C) 2026-07-27 10:15 |
| `ppo_sb3/seed_0/20260727_024112` | 207 / 333 | Interrupted (Ctrl+C) 2026-07-27 10:15 |
| `EXP_sac-lite-cmp/.../20260727_101039` | 61 / 333 | Running as of 2026-07-27 17:15 |
| `EXP_sac-micro-cmp/.../20260727_101701` | 69 / 333 | Running as of 2026-07-27 16:48 |

The two live runs are not yet affected but are on the same trajectory.

## 5. Assumptions And Invariants

- Generate = a frozen catalog record the scenario buffer does not hold; Replay =
  a record it holds. Buffer membership is the only exclusion applied to a fresh
  draw besides the in-flight batch set (`VERIFIED`, `selection.py`).
- EMA scores and normalized LP are finite and in `[0, 1]`; only the arm of a
  committed Generate is updated (`SPECIFIED`, ACL v1.1 REQ-001).
- `p_i >= eta/K = 0.0333` for every arm, so no correction may drive an arm's
  probability to zero without deviating from REQ-002 (`SPECIFIED`).
- RNG call order inside a slot decision — replay coin flip, then arm draw, then
  fresh-record draw — is part of seeded reproducibility (`VERIFIED`); any fix
  must state its effect on it.
- The frozen catalog, split manifest, and `structural_empty_cells` are immutable
  within a run (`SPECIFIED`, ScenarioNet Integration v1.1).

## 6. Decisions And Approval Gates

All options below change observable curriculum behavior. Approval and rationale
are recorded in ADR-028.

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-EXH-001` | Specification clarification | What must happen when the MAB draws an arm with no fresh catalog record? | A: exclude the arm from the Generate distribution while it has no fresh record, renormalizing over the remaining arms; B: treat the exhausted draw as feedback and apply an explicit neutral/penalizing EMA update; C: raise an explicit fatal error, per REQ-C; D: keep the silent Replay but log and count it | **A**, with the exhaustion event logged and counted (see §7.2) | Restores REQ-B's 0.40 Generate share and REQ-A's "every Generate updates the MAB"; changes arm probabilities and therefore the curriculum trajectory | **APPROVED 2026-07-27** (option A) |
| `DEC-EXH-002` | Specification clarification | May `p_i` for an excluded arm fall below the `eta/K` exploration floor of REQ-002? | A: renormalize the floor over the eligible arms only; B: keep the floor over all six arms and accept wasted mass | A | REQ-002 wording needs an amendment for the eligible-arm subset | **APPROVED 2026-07-27** (option A) |
| `DEC-EXH-003` | Specification deviation | Are results from runs already affected still usable? | A: report them with an explicit caveat; B: rerun the affected configurations after the fix | User's call; the defect is confined to curriculum sampling and does not corrupt learner data | Affects thesis results reporting | **APPROVED 2026-07-27** (option A: keep, with caveat) |
| `DEC-EXH-004` | Blocking technical issue | Should `A4_vru`'s 333-record Waymo-only pool be enlarged by dataset curation? | A: fix the selector only; B: also rebuild the catalog | A now, B tracked separately under `live_training_reliability_v1` | Dataset rebuild is expensive and orthogonal to the selector defect | **APPROVED 2026-07-27** (option A: not now) |

Rationale for `DEC-EXH-001` option A: it is the only option that restores both
REQ-A and REQ-B without inventing feedback the specification forbids. Option B
invents an LP value for an episode that never ran, which ADR-014 and REQ-001
prohibit. Option C is faithful to REQ-C but would abort every long ACL run once
the smallest pool is absorbed, which is a normal, not exceptional, condition.
Option D leaves REQ-B violated. Neither of the ACL v1.1 design's two
inspirations (Peng et al. 2024; Abouelazm et al. 2025) hits this case: Peng's
arms are procedurally generated and never exhaust; Abouelazm's learning
potential lives on individual buffered scenarios, not semantic categories, so
replay itself keeps the relevant score alive. See ADR-028 for the full
comparison.

## 7. Proposed Design

### 7.1 Completed, behavior-neutral preparation

`select_acl_slot_decision` and `fresh_generate_candidates` were extracted into
`src/thesis_rl/curriculum/scenario_acl/selection.py` from the closure at
`driver.py:586-672`. The extraction preserves branch order, RNG call order, and
every emitted field; the driver keeps ownership of episode ids, generations, and
the Generate/Replay counters. The new `AclSlotDecision` wrapper exposes
`sampled_arm_index`, `sampled_arm_name`, and `generate_arm_exhausted`, which the
driver currently ignores. This makes the defect reachable from a deterministic
test without a simulator and gives the approved correction its insertion point.

### 7.2 Correction implemented under `DEC-EXH-001` option A

1. `eligible_generate_arm_mask` (`selection.py`) computes, per arm, whether
   `fresh_generate_candidates(...)` is non-empty, before any RNG call.
2. `ScenarioArmBandit.probabilities()`/`sample_arm()` (`mab.py`) accept an
   optional `eligible_mask`: an ineligible arm gets probability exactly zero,
   and the softmax and the `eta/K` floor are renormalized over the eligible
   subset (`DEC-EXH-002`). Omitting the mask reproduces the unrestricted
   REQ-002 distribution unchanged, so every other MAB caller is unaffected.
3. `select_acl_slot_decision` draws only from the eligible subset. A drawn arm
   is therefore always eligible and always finds a fresh record; the old
   "sampled, then found no candidate" branch is now defensively unreachable
   (kept as a `RuntimeError` guard, not silently removed).
4. If no arm is eligible at all (every arm's whole pool absorbed
   simultaneously), the existing Replay-or-fatal fallback is unchanged
   (`REQ-C`); `generate_arm_exhausted=True` flags this case on the decision.
5. Diagnostics (`REQ-D`): `driver.py` logs a
   `scenario_acl_generate_pool_eligibility_changed` event exactly when the set
   of ineligible arms changes (not once per episode), and each
   `mab_history.jsonl` chunk snapshot carries `generate_ineligible_arms`.
6. `selection_probability` for a Generate is the renormalized probability, and
   the EMA update uses it unchanged (REQ-001 applies no importance correction).

Effect on reproducibility: the arm draw consumes the same single RNG call in
the normal case, so seeded runs stay reproducible under the new code but are
not bit-comparable with pre-fix runs once any arm becomes ineligible (the drawn
arm can differ). The full-exhaustion corner case additionally consumes one
fewer RNG call than before (no arm draw is attempted); this is documented in
ADR-028 as an accepted, intentional divergence for an edge case with no
production occurrence to date.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-A` | `AC-EXH-001` | `scenario_acl/selection.py::select_acl_slot_decision` | `tests/test_scenario_acl_arm_exhaustion.py::test_exhausted_arm_no_longer_starves_the_mab` | Fixed |
| `REQ-B` | `AC-EXH-002` | `scenario_acl/selection.py::select_acl_slot_decision`, `mab.py::ScenarioArmBandit.probabilities` | `tests/test_scenario_acl_arm_exhaustion.py::test_exhausted_arm_generate_draw_is_reassigned_to_an_eligible_arm` | Fixed |
| `REQ-C` | `AC-EXH-003` | `scenario_acl/selection.py` (all-arms-ineligible fallback) | `tests/test_scenario_acl_arm_exhaustion.py::test_no_eligible_arm_with_an_empty_buffer_is_fatal`, `::test_all_arms_exhausted_falls_back_to_replay_and_flags_exhaustion` | Preserved |
| `REQ-D` | `AC-EXH-004` | `driver.py` (`scenario_acl_generate_pool_eligibility_changed` event, `generate_ineligible_arms` chunk field) | Covered by full-suite green run; no dedicated artifact-schema test (see §9) | Implemented |
| Exhaustion trigger | `AC-EXH-005` | `scenario_acl/selection.py::fresh_generate_candidates`, `::eligible_generate_arm_mask` | `tests/test_scenario_acl_arm_exhaustion.py::test_fresh_generate_candidates_empty_once_buffer_absorbed_the_arm_pool`, `::test_eligible_generate_arm_mask_excludes_only_the_exhausted_arm` | Covered |
| Non-regression of the healthy path | `AC-EXH-006` | `scenario_acl/selection.py` | `tests/test_scenario_acl_arm_exhaustion.py::test_generate_on_a_non_exhausted_arm_still_feeds_the_mab` | Covered |
| Eligible-subset floor renormalization | `AC-EXH-007` | `mab.py::ScenarioArmBandit.probabilities` | `tests/test_scenario_acl_mab.py::test_probabilities_zero_ineligible_arms_and_renormalize_the_floor`, `::test_probabilities_reduce_to_the_unrestricted_contract_without_a_mask`, `::test_probabilities_rejects_an_all_false_eligible_mask`, `::test_sample_arm_never_draws_an_ineligible_arm` | Covered |

## 9. Test Strategy Defined Before Implementation

Acceptance criteria:

- `AC-EXH-001`: an arm drawn for Generate either produces a Generate selection
  that feeds the MAB, or is not drawn at all; no draw may vanish silently.
- `AC-EXH-002`: after warm-up, the empirical Generate share stays consistent
  with 0.40 even when one arm's pool is exhausted.
- `AC-EXH-003`: an exhausted arm with an empty buffer remains fatal.
- `AC-EXH-004`: the exhausted state is visible in run artifacts.
- `AC-EXH-005`: buffer absorption of an arm's whole pool empties its fresh
  candidate list.
- `AC-EXH-006`: an arm with fresh records still produces Generate and updates
  its own score.

Mandatory matrix:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-EXH-001` | Unit | Exhaustion trigger | Six arms of six records; the whole `A4_vru` pool in the buffer | `fresh_generate_candidates` empty for `A4_vru`, non-empty for the others | `AC-EXH-005` |
| `TEST-EXH-001b` | Unit | Eligibility mask | Same fixture | `eligible_generate_arm_mask` is `False` only for `A4_vru` | `AC-EXH-005` |
| `TEST-EXH-002` | Unit | Reassigned draw | Bandit forced onto `A4_vru`, `exploit_probability=0` | Decision reports `eligible_arm_mask[A4]=False`, `mode="generate"` on a different arm, `sampled_arm_name != "A4_vru"`, `generate_arm_exhausted=False` | `AC-EXH-002` |
| `TEST-EXH-003` | Regression | Auto-block, corrected | 400 seeded decisions applying `driver.commit_event`'s update rule | `A4_vru` never sampled, zero Generate, score stays frozen but its selection probability under the live mask is exactly `0.0`, not growing | `AC-EXH-001` |
| `TEST-EXH-004` | Unit | Healthy control | Bandit forced onto `A0`, which has fresh records | `mode="generate"`, MAB score moves | `AC-EXH-006` |
| `TEST-EXH-005` | Unit | Fatal boundary | Every arm ineligible (only arm's records excluded), empty buffer | `RuntimeError: "No ACL arm has a fresh catalog record..."` | `AC-EXH-003` |
| `TEST-EXH-005b` | Unit | Full-exhaustion fallback | Every arm's pool absorbed, non-empty buffer | Falls back to Replay, `generate_arm_exhausted=True` | `AC-EXH-003` |
| `TEST-EXH-006` | Unit | Renormalized floor | `eligible_mask=[T,T,F,F,F,T]`, `eta=0.2` | Ineligible arms exactly `0.0`; eligible arms `>= eta/3`; sums to `1.0` | `AC-EXH-007` |
| `TEST-EXH-006b` | Unit | Contract preserved | No mask / all-`True` mask | Identical to the unrestricted REQ-002 distribution | `AC-EXH-007` |
| `TEST-EXH-006c` | Unit | Sampling respects mask | 200 draws, half the arms ineligible | Drawn arm always in the eligible subset | `AC-EXH-007` |
| `TEST-EXH-007` | Diagnostics | Exhaustion visibility | Full suite exercises `driver.py`'s new event/field code paths | Full suite green (indirect coverage; no dedicated fixture-level artifact test, see below) | `AC-EXH-004` |

All of `TEST-EXH-001` to `TEST-EXH-006c` are implemented and passing.
`TEST-EXH-003` used to assert the pre-fix **defect** and is now inverted to
assert the corrected behavior, per the mandatory-test policy in `PLANS.md`
(strengthened, not deleted — `git log` on this file preserves the original).
`TEST-EXH-007` is covered indirectly: the diagnostic code paths in `driver.py`
(`scenario_acl_generate_pool_eligibility_changed` event,
`generate_ineligible_arms` chunk field) execute during the full vectorized
training path exercised by the pre-existing `test_scenario_acl_vectorized_state.py`
and integration tests, all green; no dedicated unit test constructs a live
`mab_history.jsonl`/`events.jsonl` pair to assert the field/event schema
directly, which is a known gap left for `make smoke` (§10, M6).

Commands:

- Focused: `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_arm_exhaustion.py tests/test_scenario_acl_mab.py`
- ACL regression set: the same command over `tests/test_scenario_acl_*.py tests/test_curriculum_*.py tests/test_train_curriculum_helpers.py`
- Full suite: `make test` (executed as `docker compose run --rm dev uv run --no-sync python -m pytest -q`)
- Lint: `make lint PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl/selection.py src/thesis_rl/curriculum/scenario_acl/driver.py src/thesis_rl/curriculum/scenario_acl/mab.py tests/test_scenario_acl_arm_exhaustion.py tests/test_scenario_acl_mab.py"`
- Format check: `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl/selection.py src/thesis_rl/curriculum/scenario_acl/mab.py tests/test_scenario_acl_arm_exhaustion.py"`
- Smoke: `make smoke` (not run in this session; recommended before treating the correction as validated in a live setting)

## 10. Milestones

- [x] **M1 — Diagnosis.** Root cause confirmed from run artifacts; ADR-024 and
      the quarantine path excluded. Evidence in §4.2 and §4.3.
- [x] **M2 — Testability.** Behavior-neutral extraction of the slot decision
      into `selection.py`; ACL regression set (82 tests) green.
- [x] **M3 — Regression test.** `tests/test_scenario_acl_arm_exhaustion.py`,
      five tests, deterministic, no simulator.
- [x] **M4 — Approval.** `DEC-EXH-001` to `DEC-EXH-004` approved 2026-07-27 and
      recorded in ADR-028.
- [x] **M5 — Correction.** Implemented §7.2: `mab.py::probabilities`/`sample_arm`
      accept `eligible_mask`; `selection.py::eligible_generate_arm_mask` and the
      restricted draw; `driver.py` diagnostics; ACL v1.1 REQ-002/REQ-003 wording
      amended via ADR-028; `TEST-EXH-003` inverted; `TEST-EXH-006`/`TEST-EXH-007`
      added (the latter as indirect coverage, see §9).
- [ ] **M6 — Validation.** Full suite and focused lint/format done (§14).
      **Remaining**: `make smoke` and a short live ACL run confirming the
      Generate share holds after an arm exhausts have not been run in this
      session.

## 11. Progress And Findings Log

### 2026-07-27 — FIND-001 confirmed

Completed: analysis of `mab_history.jsonl`, `logs/events.jsonl`,
`scenario_buffer.json`, `scenario_acl_vector_state.json`, and
`artifacts/scenarionet/split_manifest.yaml` for the affected run and the five
other ACL runs on disk; behavior-neutral extraction; regression test.

Findings: the freeze is caused by frozen-pool exhaustion under the
buffer-membership exclusion, not by ADR-024 quarantine. Severity: high — it
diverts about a third of the Generate budget and violates ACL v1.1 REQ-A/REQ-B/
REQ-D, and the state is absorbing. It is systematic, not run-specific: the three
other completed runs sat at 332, 224, and 207 of A4's 333 records, and the run
at 332 was one record away.

Commands and results:

| Command | Result |
|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_arm_exhaustion.py` | 5 passed |
| `... pytest -q` over the ACL/curriculum set | 82 passed |
| `... ruff check --no-cache` on the three touched files | All checks passed |
| `... ruff format --no-cache --check` on the two new files | 2 files already formatted |
| `... pytest -q` (full suite) | 1018 passed, 7 skipped |

Decisions needed: `DEC-EXH-001` to `DEC-EXH-004`.

Next step: obtain approval, then implement §7.2.

Note: the analysis was reported as recorded in
`docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` §11
FIND-001. That file does not exist in the repository or in any branch's history;
this ExecPlan is the record.

### 2026-07-27 — Approval and correction implemented

Completed: user approved `DEC-EXH-001` option A and `DEC-EXH-002` after asking
(1) how the two inspiration papers (Peng et al. 2024, Abouelazm et al. 2025)
handle this case — neither does, since neither combines a semantic-arm MAB with
a frozen finite per-arm pool — and (2) for clarification on `DEC-EXH-002`'s
dependency on `DEC-EXH-001`. Confirmed `DEC-EXH-003` (keep affected runs, with
caveat) and `DEC-EXH-004` (no dataset rebuild now).

Implemented §7.2: `ScenarioArmBandit.probabilities()`/`sample_arm()` gained an
optional `eligible_mask` parameter (`mab.py`); `selection.py` gained
`eligible_generate_arm_mask()` and now draws only from eligible arms before any
candidate lookup; `driver.py` gained the
`scenario_acl_generate_pool_eligibility_changed` event and the
`generate_ineligible_arms` chunk field. `TEST-EXH-003` was inverted (same test
function, corrected assertions, docstring records the history); `TEST-EXH-001b`,
`TEST-EXH-002` (rewritten), `TEST-EXH-005b`, and `TEST-EXH-006`/`006b`/`006c`
were added. ACL v1.1 REQ-002/REQ-003 amended in place with inline "Amended
2026-07-27 by ADR-028" notes and a new `DEC-003` row; ADR-028 written and
approved.

Commands and results:

| Command | Result |
|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_mab.py tests/test_scenario_acl_arm_exhaustion.py` | 23 passed |
| `... pytest -q` over the ACL/curriculum set | 88 passed |
| `... pytest -q` (full suite) | 1024 passed, 7 skipped |
| `... ruff check --no-cache` on all 5 touched/added Python files | All checks passed |
| `... ruff format --no-cache` on `mab.py`, `selection.py`, `test_scenario_acl_arm_exhaustion.py` | 3 files reformatted (applied, then full suite re-run green) |

Not run: `make smoke`, and a short live ACL run confirming the Generate share
holds once an arm exhausts (M6, remaining).

Next step: `make smoke` and a live confirmation run, at the user's discretion;
otherwise this ExecPlan is `IMPLEMENTED`.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-EXH-001` | ACL v1.1 REQ-002: `p_i=(1-eta) softmax(q_i/tau)+eta/K` over all `K=6` arms | Renormalize over eligible arms only | An ineligible arm cannot honour a Generate draw | Approved 2026-07-27, `DEC-EXH-002`, ADR-028 | ACL v1.1 §6 REQ-002/REQ-003 (amended in place), `tests/test_scenario_acl_mab.py` |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/curriculum/scenario_acl/selection.py` | Added | Slot-decision extraction; `eligible_generate_arm_mask`; masked arm draw |
| `src/thesis_rl/curriculum/scenario_acl/mab.py` | Modified | `probabilities()`/`sample_arm()` accept `eligible_mask` with renormalized floor |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Selector delegates to `select_acl_slot_decision`; eligibility-transition event and chunk diagnostics field |
| `tests/test_scenario_acl_arm_exhaustion.py` | Added | `TEST-EXH-001` to `TEST-EXH-005b` |
| `tests/test_scenario_acl_mab.py` | Modified | `TEST-EXH-006`/`006b`/`006c` |
| `docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md` | Added | This plan |
| `docs/specifications/automatic_curriculum_learning_v1.1_specification.md` | Modified | REQ-002/REQ-003 amended in place; `DEC-003` added; ADR-028 cross-referenced |
| `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md` | Added | Approved correction record |
| `docs/project_index.md` | Modified | Bugfix register entry |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_mab.py tests/test_scenario_acl_arm_exhaustion.py` | 23 passed | 2026-07-27 | Post-correction focused suite |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q` over the ACL/curriculum set | 88 passed | 2026-07-27 | Confirms the correction is contained to the intended contract |
| `docker compose run --rm dev uv run --no-sync ruff check --no-cache <5 touched files>` | All checks passed | 2026-07-27 | `--no-cache` because the container cannot write `.ruff_cache` in this worktree |
| `docker compose run --rm dev uv run --no-sync ruff format --no-cache <3 files>` | 3 files reformatted | 2026-07-27 | Applied; full suite re-run green afterward |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q` (equivalent of `make test`) | 1024 passed, 7 skipped, 0 failed | 2026-07-27 | 1018 pre-correction plus 6 net new tests |
| `git diff --check` | Clean | 2026-07-27 | No whitespace defects |
| `make smoke` | Not run | — | Recommended before treating the correction as validated in a live setting |

Worktree environment note: this worktree initially lacked the `third_party`
submodules, MetaDrive's downloaded `metadrive/assets`, and `.env`, which produced
60 and then 1 environment-only failures unrelated to the change. All three were
provisioned locally (submodule init, asset copy from the primary checkout, `.env`
with the host UID/GID and worktree-local host directories) before the green run
above. None of them is tracked content.
