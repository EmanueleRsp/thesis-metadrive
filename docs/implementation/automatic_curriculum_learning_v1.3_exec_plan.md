# ExecPlan — ScenarioNet ACL v1.3: Generate/Catalog Decoupling and Coverage Cycles

## 1. Metadata

- Feature: decouple Generate eligibility from scenario-buffer membership; per-arm coverage cycles; selective replay buffer
- Plan ID: `ACL-SN-CAT-003`
- Authoritative specification: `docs/specifications/automatic_curriculum_learning_v1.3_specification.md` (`ACL-SN-EMA-001`, `v1.3`, `APPROVED`, `Authoritative: YES`, 2026-07-29)
- Status: `IMPLEMENTED_PENDING_RERUN` (design approved 2026-07-29; implementation completed 2026-07-29; two live observations pending the first full re-run, see §14)
- Created: 2026-07-29
- Last updated: 2026-07-29
- Branch: `scenarionet-implementation`
- Related ADRs: `ADR-032` (this change), `ADR-028` (scope narrowed by this change), `ADR-029`, `ADR-030`, `ADR-016`, `ADR-024`, `ADR-014`
- Supersedes in effect: `docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md` remains `IMPLEMENTED` and its code stays; this plan removes the condition that made it necessary
- Owner: thesis repository maintainer

## 2. Objective And Scope

### Observable capability

After this change, the ACL teacher's Generate path samples from an arm's full frozen catalog partition,
independently of what the scenario buffer currently holds. Consequently: (a) an arm can never leave the curriculum
by having its pool absorbed into the buffer (`FIND-001`); (b) the bandit's per-arm feedback is no longer drawn
from the low-learning-potential residual of each arm (`FIND-005`); (c) every admissible record of an arm is
visited exactly once per coverage cycle, so the curated catalog is covered systematically rather than by
coupon-collector chance; (d) the replay buffer holds a genuinely selective 12.5% active subset of the catalog.

### Success recognition

All acceptance criteria `AC-001..AC-012` of `v1.3` hold, with `AC-009..AC-012` being new; the full repository test
suite and the focused ACL suite pass; `make config`/`make config-gpu` resolve; a representative `make smoke` run
completes; ACL diagnostics show non-zero coverage-cycle counters and a `generate_on_buffered_record` buffer action
in a real run.

### In scope

- Generate candidate/eligibility semantics (`REQ-009`).
- Per-arm coverage-cycle state, its persistence, and its diagnostics (`REQ-009`, `REQ-005`).
- Buffer commit semantics for Generate episodes landing on buffered records (`REQ-010`).
- Staleness reference refreshed by Generate visits (`REQ-004`).
- `buffer_capacity` 1000 → 250 in `conf/curriculum/scenario_acl.yaml` and the config default (`DEC-010`).
- Checkpoint schema bump `acl_ema_v2` → `acl_ema_v3` and explicit rejection of older schemas (`REQ-005`).
- Alignment of the unreachable sequential ACL path (`DEC-011`).
- Documentation: `v1.3` specification, `ADR-032`, this plan, `docs/project_index.md`.

### Out of scope

- Any change to the frozen catalog, split assignment, or arm assignment of a record.
- Mutation/editor (`ADR-014`, prohibited).
- Intra-arm prioritization by learning potential (`RAT-010`).
- Ablation of `buffer_capacity` (`LIM-005`).
- Removal of the unreachable sequential path (`DEC-011`, tracked separately).
- The `train_loop.py:791` early-return wiring gaps already tracked in
  `docs/implementation/scenario_acl_path_wiring_gaps_exec_plan.md`.

### Compatibility constraints

- Checkpoints from `v1.1`/`v1.2` cannot resume; migration policy is restart (accepted, `ADR-032`).
- The observable RNG call order changes; `v1.1`/`v1.2` seeds do not reproduce their runs (accepted).
- Completed runs are retained for reporting with the `FIND-001`/`FIND-005` caveat already recorded.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-002` | Softmax/floor renormalized over Generate-eligible arms; under `REQ-009` all arms are normally eligible | `v1.3` §6 REQ-002 |
| `REQ-003` | 40/60 schedule; degradation to Replay only when no arm is eligible; fatal when the buffer is also empty | `v1.3` §6 REQ-003 |
| `REQ-004` | 70/30 replay mixture unchanged; `last_seen_step` refreshed by every committed episode, Generate included | `v1.3` §6 REQ-004 |
| `REQ-005` | Persist coverage state; schema `acl_ema_v3`; older schemas rejected explicitly | `v1.3` §6 REQ-005 |
| `REQ-009` | Generate admissibility excludes only quarantine and in-flight; per-arm coverage cycles; uniform draw within the cycle candidate set | `v1.3` §6 REQ-009 |
| `REQ-010` | Generate commit is update-or-insert; one buffer entry per `scenario_uid`; MAB updated regardless; eviction is reference-only | `v1.3` §6 REQ-010 |

Requirements `REQ-001`, `REQ-006`, `REQ-007`, `REQ-008` are carried unchanged from `v1.2` and are not re-derived
here; they must keep passing their existing acceptance criteria.

## 4. Current Repository Analysis

All statements below are `VERIFIED` by direct reading on 2026-07-29 unless marked otherwise.

### Call flow (vectorized production path)

- `src/thesis_rl/curriculum/scenario_acl/vectorized.py:175` `AclVectorSelectionCoordinator.select_batch` builds
  `excluded` from `state.active_selections` (in-flight UIDs) and passes it per slot; `validate_fresh_batch_unique`
  enforces batch uniqueness (`ADR-016`).
- `src/thesis_rl/curriculum/scenario_acl/driver.py:587` `select_batch`/`selector` closure calls
  `select_acl_slot_decision` and accounts modes and eligibility transitions.
- `src/thesis_rl/curriculum/scenario_acl/selection.py:118` `select_acl_slot_decision` — the decision itself.
  Line 59 (in `select_acl_slot_decision`) adds buffer members to `effective_excluded`: **this is the rule removed
  by `DEC-008`**. The module docstring declares the RNG call order (replay coin → arm draw → record draw) as an
  observable contract.
- `selection.py:56` `fresh_generate_candidates` filters `train_records` by `primary_arm` and the exclusion set —
  the catalog is re-filtered per draw, so arms are already non-destructive views; no `pop`/removal exists anywhere.
- `selection.py:71` `eligible_generate_arm_mask` → `ScenarioArmBandit.probabilities(eligible_mask=...)`
  (`mab.py:49`), the `ADR-028` renormalization.
- `driver.py:740` `commit_event`: computes `lp_scaled` (`REQ-007`), updates MAB for Generate only, then either
  `buffer.update` (replay branch, with the `skipped_evicted_before_commit` guard at `driver.py:794`) or
  `buffer.insert(_build_record_from_catalog_entry(...))` (generate branch, `driver.py:814`).
- `driver.py:867` on a typed data-abort calls `buffer.remove_scenario_id(...)` (`ADR-024`).

### Behavior to preserve

- Deterministic commit ordering and batch uniqueness (`ADR-016`).
- `skipped_evicted_before_commit` non-fatal handling (`driver.py:794`).
- Quarantine propagation to workers and across resume (`driver.py:576`, `driver.py:1066`).
- Reward-scale normalization applied to both modes, using the pre-update estimate (`REQ-007`).
- Rulebook exclusion from all ACL utility paths (`ADR-014`).

### Facts that shaped the design

- `ScenarioBuffer.contains_hash` dedups on `scenario_description_hash`, and
  `driver.py:386` sets `scenario_description_hash = sha256(scenario_uid)`. Dedup is therefore already equivalent
  to UID dedup; no key change is required by `REQ-010` (`VERIFIED`).
- `ScenarioRecord` already carries `num_seen` and `last_seen_step` and is already persisted with the buffer, so
  `REQ-010`'s in-place update needs no new record field (`VERIFIED`).
- Train catalog composition (`data/scenarionet/frozen/scenario_selection_index.json`, `VERIFIED` 2026-07-29):
  2,000 records; `A0/A1/A2` PG with 333/333/334, `A3/A4/A5` Waymo with 334/333/333. Each arm is single-source by
  construction of the frozen selection.
- `conf/curriculum/scenario_acl.yaml:5` `buffer_capacity: 1000`; `curriculum/config.py:107` default `1000`;
  `config.py:269` validates `warmup_buffer_size <= buffer_capacity` (100 ≤ 250 holds after `DEC-010`).

### Directly relevant debt

- The sequential path `driver.py:1646` (`choose_acl_episode`/`collect_catalog_episode`) duplicates the exclusion
  semantics and is unreachable in production (`train_loop.py:791` early return; recorded in
  `docs/project_index.md` row "ScenarioNet-ACL training-path wiring gaps"). Handled by `DEC-011`.
- `LIM-004`: `commit_event` is a private closure with no isolated-test harness.

## 5. Assumptions And Invariants

| Item | Value | How established | Violation handling |
|---|---|---|---|
| Arm partition of the catalog | immutable, per-record `primary_arm` | Frozen selection index (`VERIFIED`) | Fatal at provider level |
| Arm pool size | ~333 records per arm, minimum 333 | Measured (`VERIFIED`) | None; a `buffer_capacity` below the smallest arm pool is recommended, not enforced |
| Coverage state lifetime | run-local; not comparable across runs | Design (`DEC-009`) | Resume restores it; a fresh run starts at cycle 0 |
| `V_i` marking time | at selection, not at commit | Required for vectorized correctness | A record in flight cannot be redrawn within a cycle |
| Data-abort records | stay marked visited, enter `Q` | `ADR-024` + `DEC-009` | Cycle closure must not deadlock on quarantined records |
| RNG call order | replay coin → arm draw → record draw | Module contract, changed by this plan | Documented in `REQ-011`-adjacent notes; old seeds not reproducible |
| Checkpoint schema | `acl_ema_v3` | `REQ-005` | Older schemas rejected with explicit error |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-008` | Specification deviation (corrective) | Should Generate eligibility depend on buffer membership? | Keep + fallback tier / remove entirely / reduce capacity only | Remove entirely | Selection semantics, MAB feedback distribution, tests, RNG order | APPROVED 2026-07-29 (`ADR-032`) |
| `DEC-009` | Specification clarification | Coverage policy for a finite catalog once repeats are allowed | i.i.d. with replacement / per-arm coverage cycles / global cycle | Per-arm coverage cycles | New persisted state, schema bump, diagnostics | APPROVED 2026-07-29 (`ADR-032`) |
| `DEC-010` | Specification deviation | `buffer_capacity` value after `DEC-008` | keep 1000 / 250 / other | 250 | Replay selectivity, eviction traffic, config | APPROVED 2026-07-29 (`ADR-032`) |
| `DEC-011` | Implementation detail | Unreachable sequential ACL path | align / delete / leave divergent | Align | Prevents silent divergence; removal tracked separately | APPROVED 2026-07-29 (`ADR-032`) |

No unresolved gate remains. `ADR-028`'s renormalization is retained as a safety net for the residual
all-quarantined/all-in-flight case; this plan must not remove it.

## 7. Proposed Design

### 7.1 New module: `src/thesis_rl/curriculum/scenario_acl/catalog_state.py`

`ScenarioCatalogVisitState` — parent-owned, run-local:

- `visited: dict[str, bool]` (or a `set[str]` of visited UIDs) keyed by `scenario_uid`, scoped to the current cycle;
- `cycle_id: dict[str, int]` keyed by arm name;
- `num_generate_visits: dict[str, int]`, `last_generate_episode_id: dict[str, int]` (diagnostics only, not used for
  selection, per `RAT-010`);
- `mark_visited(scenario_uid, arm_name, episode_id)`;
- `close_cycle(arm_name)` → increments `cycle_id[arm]` and clears that arm's visited entries;
- `state_dict()` / `from_state_dict()` with schema tag `acl_coverage_v1`.

The state is keyed by UID only; it never holds scenario content and never mutates the catalog.

### 7.2 `selection.py`

- `fresh_generate_candidates(train_records, *, arm_name, excluded_scenario_uids, visit_state)` →
  records of the arm with `uid not in excluded_scenario_uids` (quarantine + in-flight only) and
  `not visit_state.is_visited(uid)`. Remove the buffer-membership union at the call site in
  `select_acl_slot_decision`.
- `eligible_generate_arm_mask(...)` → per arm, whether any record is admissible (quarantine + in-flight only),
  **ignoring** the coverage cycle, so cycle exhaustion never makes an arm ineligible.
- `select_acl_slot_decision(...)` gains a `visit_state` parameter. After the arm draw: build `F_i`; if empty, call
  `visit_state.close_cycle(arm)` and rebuild; then draw uniformly with `rng.integers` (unchanged call shape, so the
  RNG order remains replay coin → arm draw → record draw); then `visit_state.mark_visited(...)`.
- `AclSlotDecision` gains `coverage_cycle_closed: bool` and `coverage_cycle_id: int` for diagnostics.

### 7.3 `driver.py`

- Construct `ScenarioCatalogVisitState` next to the buffer/bandit; thread it into the `selector` closure.
- Generate branch of `commit_event` (`driver.py:814`) becomes update-or-insert: look up the UID among
  `buffer.records()`; if present, build the updated record through the existing `_update_replay_record` helper
  (renamed or given a mode-neutral alias) and call `buffer.update`, action `generate_on_buffered_record`;
  otherwise the current insert path, actions `inserted`/`rejected`.
- Persist/restore coverage state alongside `scenario_acl_state.json` and `scenario_buffer.json`
  (`_persist_buffer_state`, `_load_scenario_acl_resume_state` at `driver.py:433`/`driver.py:465`), file
  `scenario_coverage_state.json`.
- Log per-arm coverage-cycle transitions once per transition, mirroring the existing eligibility-transition logging
  at `driver.py:604`.
- Apply the same semantics to the sequential path at `driver.py:1646` (`DEC-011`).

### 7.4 `mab.py`, `buffer.py`, config

- `mab.py`: `state_schema = "acl_ema_v3"`; `from_state_dict` rejects `acl_ema_v2` and `acl_ema_v1` with an explicit
  message naming the required restart.
- `buffer.py`: no structural change; `insert`/`update` semantics are unchanged. The dedup key stays
  `scenario_description_hash` (already UID-equivalent, §4).
- `conf/curriculum/scenario_acl.yaml`: `buffer_capacity: 250`. `curriculum/config.py:107`: default `250`.

### 7.5 Errors, fallbacks, logging

Unchanged fatal conditions plus: a Generate draw whose arm was reported eligible but yields no admissible record
after cycle closure remains a `RuntimeError` (defensive, unreachable). Coverage-state persistence failure is fatal
on resume, consistent with existing buffer/state handling.

### 7.6 Alternatives rejected

Recorded in `ADR-032` "Alternatives Considered": fallback tier without decoupling; capacity-only fix; `DEC-008`
without coverage cycles; intra-arm prioritization; buffer coincident with the catalog.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-009` | `AC-009` | `selection.py: fresh_generate_candidates`, `select_acl_slot_decision` | `tests/test_scenario_acl_catalog_coverage.py::test_generate_candidates_ignore_buffer_membership` | RECONCILED |
| `REQ-009` | `AC-010` | `selection.py`, `mab.py` | `tests/test_scenario_acl_catalog_coverage.py::test_fully_buffered_arm_still_generates_and_updates_ema` | RECONCILED |
| `REQ-009` | `AC-011` | `catalog_state.py`, `selection.py` | `tests/test_scenario_acl_catalog_coverage.py::test_coverage_cycle_visits_each_record_once_then_restarts` | RECONCILED |
| `REQ-010` | `AC-012` | `driver.py: commit_event` | `tests/test_scenario_acl_buffer.py::test_generate_on_buffered_record_updates_in_place` | RECONCILED |
| `REQ-004` | `AC-012` | `driver.py: commit_event` | `tests/test_scenario_acl_buffer.py::test_generate_visit_refreshes_staleness_reference` | RECONCILED |
| `REQ-005` | `AC-005` | `catalog_state.py`, `mab.py`, `driver.py` | `tests/test_scenario_acl_catalog_coverage.py::test_coverage_state_round_trip`, `tests/test_scenario_acl_mab.py::test_rejects_acl_ema_v2_checkpoint` | RECONCILED |
| `REQ-002`/`REQ-003` | `AC-002`/`AC-003` | unchanged | existing `tests/test_scenario_acl_arm_exhaustion.py`, `tests/test_scenario_acl_mab.py` | PASSING |

## 9. Test Strategy Defined Before Implementation

Frozen mandatory matrix. `TEST-CAT-001..004` are the decision-critical ones.

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-CAT-001` | Unit | Generate candidates and the eligible-arm mask are invariant to buffer contents | one arm of N synthetic catalog records; buffer empty / half / full-arm | candidate sets and mask identical in all three cases | `REQ-009` / `AC-009` |
| `TEST-CAT-002` | Unit | An arm whose whole pool is in the buffer still yields Generate selections and EMA updates | buffer holding all of arm `i`, `q_i` highest | at least one `mode="generate"` on arm `i` in a seeded sequence; `q_i` changes after commit | `REQ-009` / `AC-010` |
| `TEST-CAT-003` | Unit | Coverage cycle completeness and restart | one arm of N records, N+2 consecutive Generate draws | first N draws are a permutation of the arm (no repeats); draw N+1 increments `cycle_id` and clears `V_i`; other arms' counters unchanged | `REQ-009` / `AC-011` |
| `TEST-CAT-004` | Unit | Generate on a buffered record updates in place | buffer holding record `s`; commit a Generate episode on `s` | exactly one entry for `s`; `LP_scaled`, `last_seen_step`, `num_seen` updated; action `generate_on_buffered_record`; MAB and reward-scale updated | `REQ-010` / `AC-012` |
| `TEST-CAT-005` | Unit | Quarantined and in-flight records are excluded without blocking cycle closure | arm with some records quarantined and some in flight | they never appear as candidates; the cycle closes on the remaining admissible set | `REQ-009` |
| `TEST-CAT-006` | Unit | Data-abort during a Generate episode keeps the record marked visited and quarantines it | abort event on a freshly drawn record | record in `Q`, still marked visited, removed from `Λ` if present, no MAB/buffer update | `REQ-009`, `ADR-024` |
| `TEST-CAT-007` | Unit | Coverage-state persistence round-trip | populated `ScenarioCatalogVisitState` | `from_state_dict(state_dict())` is identical, including per-arm cycle counters | `REQ-005` |
| `TEST-CAT-008` | Unit | Checkpoint incompatibility | `acl_ema_v2` and `acl_ema_v1` payloads | explicit `ValueError` naming the required restart | `REQ-005` / `AC-005` |
| `TEST-CAT-009` | Unit | No arm is eligible → Replay degradation; buffer empty → fatal | all records quarantined; buffer non-empty then empty | degraded Replay recorded; `RuntimeError` in the empty case | `REQ-003` |
| `TEST-CAT-010` | Unit | Config validation with the new default | `buffer_capacity=250`, `warmup_buffer_size=100` | resolves; `warmup > capacity` still rejected | `DEC-010` |
| `TEST-CAT-011` | Integration | Seeded selection sequence determinism | fixed seed, fixed catalog, two runs | identical mode/arm/record sequences and identical coverage counters | `REQ-005`, `ADR-016` |
| `TEST-CAT-012` | Regression | `ADR-028` renormalization still applies when it is genuinely needed | all records of one arm quarantined | that arm gets exactly zero probability; the floor is renormalized over the rest | `REQ-002` |

Existing mandatory tests that must keep passing: `tests/test_scenario_acl_mab.py`,
`tests/test_scenario_acl_buffer.py`, `tests/test_scenario_acl_usefulness.py`,
`tests/test_scenario_acl_vectorized_state.py`, `tests/test_scenario_acl_config.py` — all extended, none weakened.

`tests/test_scenario_acl_arm_exhaustion.py` was listed here as "unchanged", which was an error: four of its tests
construct arm unavailability through scenario-buffer absorption, the exact semantics `DEC-008` removes. They were
rewritten to construct it through the quarantine/in-flight exclusion set, preserving every assertion. See
`DEV-CAT-004` in §12.

### Commands (all verified as existing repository targets)

- Focused: `uv run --no-sync python -m pytest -q tests/test_scenario_acl_catalog_coverage.py tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py tests/test_scenario_acl_arm_exhaustion.py tests/test_scenario_acl_config.py tests/test_scenario_acl_vectorized_state.py`
- Full suite: `make test`
- Lint: `make lint`
- Format check (focused): `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl/catalog_state.py src/thesis_rl/curriculum/scenario_acl/selection.py src/thesis_rl/curriculum/scenario_acl/driver.py src/thesis_rl/curriculum/scenario_acl/mab.py tests/test_scenario_acl_catalog_coverage.py"`
- Config: `make config` and `make config-gpu`
- Smoke: `make smoke`
- Whitespace: `git diff --check`

No type-checking target is configured repository-wide; new public functions in `catalog_state.py` and the changed
signatures in `selection.py` carry annotations, but no global mypy gate exists (unavailable, not skipped).

## 10. Milestones

- [x] **M1 — Documentation and decision record.** `ADR-032`, specification `v1.3`, this plan, ADR-028 numbering
      collision resolved, `docs/project_index.md` updated. Evidence: files present, index rows added.
- [x] **M2 — Coverage state module.** `catalog_state.py` (`ScenarioCatalogVisitState`, schema `acl_coverage_v1`)
      plus `TEST-CAT-007` and the state-level half of `TEST-CAT-003`. See `DEV-CAT-004`.
- [x] **M3 — Selection semantics.** `selection.py` decoupling and coverage cycles plus `TEST-CAT-001`,
      `TEST-CAT-002`, `TEST-CAT-003`, `TEST-CAT-005`, `TEST-CAT-009`, `TEST-CAT-012`.
- [x] **M4 — Commit and persistence.** `driver.py` update-or-insert, coverage persistence/resume, cycle-closure
      and coverage diagnostics; `mab.py` schema bump; plus `TEST-CAT-004`, `TEST-CAT-006`, `TEST-CAT-008`,
      `TEST-CAT-011`.
- [x] **M5 — Configuration and sequential-path alignment.** `buffer_capacity: 250` in the YAML profile and the
      dataclass default, `DEC-011` alignment of the sequential path, `TEST-CAT-010`.
- [x] **M6 — Validation and reconciliation.** Full suite, lint, focused format check, compose validation, and two
      ACL-path smoke runs (sequential and vectorized). §14 filled with real results; §15 reconciled.

## 11. Progress And Findings Log

**2026-07-29 — plan created.** Design approved in-session (`ADR-032`). Findings recorded during analysis:

- `FIND-005` (new, analytical): the buffer-membership exclusion biases the bandit's per-arm feedback toward each
  arm's low-LP residual, because the buffer retains high-LP records by construction. Severity: invalidates the
  premise of `REQ-001`'s EMA; affects all arms, not only exhausted ones; active from early training. Not
  quantifiable retrospectively (`LIM-006`): the LP distribution of records outside the buffer was never logged.
- Confirmed by reading that arms were **never** consumed or emptied by selection: `fresh_generate_candidates`
  re-filters the immutable `train_records` on every draw. The widely assumed "records are removed from the arm"
  framing is incorrect; the defect is entirely in the definition of the exclusion set.
- Confirmed that eviction was already non-destructive: an evicted UID leaves `effective_excluded` on the next draw
  and becomes selectable again. The `FIND-001` absorbing state came from the *retention* of high-LP records, not
  from any loss of records.
- Corrected a catalog-size assumption used in earlier discussion: the training split holds **2,000** records
  (~333 per arm), not 1,000; `buffer_capacity=1000` was 50% of the catalog.
- Three files shared the number `ADR-028`; renumbered `acl-recorded-rationale-record` → `ADR-030` and
  `arm-difficulty-threshold-provenance` → `ADR-031`, keeping `ADR-028` for the code-referenced
  `scenario-acl-generate-eligibility-renormalization`. References updated in specifications, ExecPlans and the
  index.

**2026-07-29 — M2–M6 implemented.** Findings and decisions taken during implementation:

- `tests/test_scenario_acl_arm_exhaustion.py` was listed in §9 as an existing mandatory test that must keep
  passing *unchanged*. That was an error in the plan: four of its tests construct arm unavailability through
  scenario-buffer absorption, which is exactly the semantics `DEC-008` removes, so they pinned the behaviour the
  approved change deletes. They were rewritten to construct unavailability through the quarantine/in-flight
  exclusion set instead, preserving every protection they assert, and a new
  `test_buffer_absorption_alone_no_longer_makes_an_arm_ineligible` pins the removed root cause. Recorded as
  `DEV-CAT-004`; no assertion was weakened or skipped.
- `select_acl_slot_decision` takes `visit_state` as a **required** keyword argument rather than an optional one,
  so no call site can silently fall back to cycle-free sampling. `fresh_generate_candidates` keeps it optional,
  because the arm-eligibility mask must deliberately ignore coverage cycles (§7.2).
- `_update_replay_record` was renamed `_update_buffered_record`: under `REQ-010` it serves both modes. Only
  `driver.py` referenced it.
- Coverage state is **mandatory** on resume. A missing `scenario_coverage_state.json` raises explicitly instead
  of restarting every arm's cycle silently, which would let a resumed run re-draw records the interrupted run had
  already covered. This is stricter than §7.5 stated and is the safe direction.
- The two live observations named in §2 "success recognition" — a non-zero coverage-cycle counter and a
  `generate_on_buffered_record` buffer action — are **not** reachable in a 2,000-step smoke run: closing one arm's
  cycle needs ~333 Generate draws on that arm, and no record can repeat before then. Both behaviours are pinned
  deterministically by `TEST-CAT-003`/`TEST-CAT-006` and `TEST-CAT-004`; the live confirmation is deferred to the
  first full re-run (`LIM-007`, §14).

Status: implementation complete; the approved re-runs are the remaining external step.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-CAT-001` | ACL `v1.1`/`v1.2` §3: Generate selects a record the buffer does not hold | Generate selects any admissible catalog record; buffer membership is irrelevant | `FIND-001`, `FIND-005`; restores the reference methods' exploration semantics | `ADR-032` `DEC-008`, 2026-07-29 | `AC-009`/`AC-010`, `v1.3` §3/§6 |
| `DEV-CAT-002` | ACL `v1.2` §9: `buffer_capacity = 1000` (frozen) | `250` | Selectivity of the buffer admission rule (`RAT-011`) | `ADR-032` `DEC-010`, 2026-07-29 | `TEST-CAT-010`, `v1.3` §9 |
| `DEV-CAT-003` | `selection.py` module contract: RNG call order fixed since `v1.1` | Same order, different consumed values; old seeds do not reproduce | Unavoidable consequence of `DEC-008` | `ADR-032`, 2026-07-29 | `v1.3` §11 |
| `DEV-CAT-004` | This plan §9: `tests/test_scenario_acl_arm_exhaustion.py` keeps passing unchanged | Four of its tests rewritten to construct arm unavailability through quarantine/in-flight instead of buffer absorption; one test added | §9 was wrong: those tests pinned the exact semantics `DEC-008` removes, so they could not both pass and remain faithful. No assertion weakened, deleted, or skipped | Within `ADR-032` `DEC-008`, recorded 2026-07-29 | `tests/test_scenario_acl_arm_exhaustion.py` |
| `DEV-CAT-005` | §7.5: coverage-state persistence failure is fatal on resume | Also fatal when the coverage artifact is *absent* on resume | A missing file would silently restart every arm's cycle and re-draw already-covered records; failing explicitly is the safe direction | Implementation detail within `REQ-005`, 2026-07-29 | `driver.py: _load_scenario_acl_resume_state` |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/decisions/ADR-032-acl-generate-catalog-decoupling.md` | Added | `DEC-008..011` decision record |
| `docs/specifications/automatic_curriculum_learning_v1.3_specification.md` | Added | Authoritative `v1.3` contract |
| `docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md` | Added | This plan |
| `docs/decisions/ADR-030-acl-recorded-rationale-record.md` | Renamed from `ADR-028-*` | Numbering collision |
| `docs/decisions/ADR-031-arm-difficulty-threshold-provenance.md` | Renamed from `ADR-028-*` | Numbering collision |
| `docs/project_index.md` | Modified | Authority rows, ADR rows, renumbering |
| `docs/specifications/automatic_curriculum_learning_v1.2_specification.md` | Modified | ADR renumbering references only |
| `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` | Modified | ADR renumbering references only |
| `src/thesis_rl/curriculum/scenario_acl/catalog_state.py` | Added | `ScenarioCatalogVisitState`, schema `acl_coverage_v1` |
| `src/thesis_rl/curriculum/scenario_acl/selection.py` | Modified | `REQ-009`: buffer union removed, coverage cycles, `visit_state` required |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | `REQ-010` update-or-insert (both paths), coverage persistence/resume, cycle-closure events, `_update_buffered_record` rename, `DEC-011` |
| `src/thesis_rl/curriculum/scenario_acl/mab.py` | Modified | `acl_ema_v3`; `acl_ema_v1`/`acl_ema_v2` rejected explicitly |
| `src/thesis_rl/curriculum/config.py` | Modified | `buffer_capacity` default and payload default 1000 -> 250 |
| `conf/curriculum/scenario_acl.yaml` | Modified | `buffer_capacity: 250` |
| `tests/test_scenario_acl_catalog_coverage.py` | Added | `TEST-CAT-001`, `002`, `003`, `005`, `006`, `007`, `009`, `011`, `012` (13 tests) |
| `tests/test_scenario_acl_buffer.py` | Modified | `TEST-CAT-004` (update-in-place plus its rejected-insert contrast) |
| `tests/test_scenario_acl_mab.py` | Modified | `TEST-CAT-008`, parameterized over `acl_ema_v1`/`acl_ema_v2` |
| `tests/test_scenario_acl_config.py` | Modified | `TEST-CAT-010` (approved pair, warmup guard, code default) |
| `tests/test_scenario_acl_arm_exhaustion.py` | Modified | `DEV-CAT-004`: unavailability reconstructed through quarantine/in-flight; `FIND-001` root-cause regression added |

## 14. Validation Results

All commands were executed from the repository root through the Docker Compose environment on 2026-07-29.

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Focused ACL suite (`pytest -q tests/test_scenario_acl_catalog_coverage.py tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py tests/test_scenario_acl_arm_exhaustion.py tests/test_scenario_acl_config.py tests/test_scenario_acl_vectorized_state.py tests/test_scenario_acl_usefulness.py`) | PASS | 2026-07-29 | All `TEST-CAT-001..012` present and passing. |
| `make test` (full suite) | PASS | 2026-07-29 | Run three times, always green: 1055 passed after the implementation, 1055 after the focused formatting pass, 1094 in the final run. The count rises because several data-dependent tests collect against run artifacts, which the two ACL smoke runs produced in between; no test was added or removed between the second and third run. |
| `make lint` (`ruff check src tests scripts`) | PASS | 2026-07-29 | "All checks passed!" |
| `make format-check` (focused, the ten touched Python files) | PASS | 2026-07-29 | "10 files already formatted". `driver.py` had a pre-existing formatting defect in the eligibility-logging call adjacent to the edited region; it was formatted as part of this change because the file is materially modified. No mass formatting was performed. |
| `make config` | PASS | 2026-07-29 | Resolves quietly. |
| `make config-gpu` | PASS | 2026-07-29 | Resolves quietly. |
| `make smoke` (default preset, `curriculum=disabled`) | PASS | 2026-07-29 | Completes end to end, but **does not exercise the ACL path** — the preset disables the curriculum. Recorded so the result is not overread. |
| ACL smoke, sequential path (`presets/test/smoke_train curriculum=scenario_acl env=scenarionet`) | PASS | 2026-07-29 | 2,000 steps. `scenario_coverage_state.json` written with schema `acl_coverage_v1`; 29 Generate draws produced 29 *distinct* records (max lifetime visits per record = 1), confirming sampling without replacement inside the cycle; `scenario_acl_state.json` MAB schema `acl_ema_v3`; buffer capacity 250. |
| ACL smoke, vectorized production path (same, `env.vectorized.enabled=true env.vectorized.num_envs=4`) | PASS | 2026-07-29 | 2,000 steps, 4 workers. Same artifacts and schemas; 14 Generate draws, max lifetime visits per record = 1; deterministic commit ordering and batch uniqueness unaffected. |
| `git diff --check` | PASS | 2026-07-29 | No whitespace defects. |

**Not observed live (`LIM-007`).** The two run-level signals named in §2 — a non-zero `cycle_id` and a
`generate_on_buffered_record` buffer action — cannot occur in a 2,000-step smoke: closing one arm's cycle requires
~333 Generate draws on that arm, and no record can repeat before its cycle closes. Both are pinned deterministically
by `TEST-CAT-003`/`TEST-CAT-006` and `TEST-CAT-004`. Exact follow-up after the first full re-run: inspect
`artifacts/curriculum/scenario_coverage_state.json` for `cycle_id > 0` and grep
`artifacts/curriculum/scenario_buffer_events.jsonl` for `generate_on_buffered_record`.

No static type-checking target exists repository-wide (unavailable, not skipped); the new public interfaces in
`catalog_state.py` and the changed signatures in `selection.py`/`driver.py` carry annotations.

## 15. Final Reconciliation

| Requirement | Acceptance criterion | Implementation | Verifying test | Status |
|---|---|---|---|---|
| `REQ-009` | `AC-009` | `selection.py: fresh_generate_candidates`, `select_acl_slot_decision` | `test_generate_candidates_ignore_buffer_membership`, `test_buffer_absorption_alone_no_longer_makes_an_arm_ineligible` | RECONCILED |
| `REQ-009` | `AC-010` | `selection.py`, `mab.py` | `test_fully_buffered_arm_still_generates_and_updates_ema` | RECONCILED |
| `REQ-009` | `AC-011` | `catalog_state.py`, `selection.py` | `test_coverage_cycle_visits_each_record_once_then_restarts`, `test_close_cycle_affects_only_the_given_arm` | RECONCILED |
| `REQ-009` | — | `selection.py`, `catalog_state.py` | `test_quarantined_and_in_flight_records_are_excluded_without_blocking_closure`, `test_data_abort_keeps_the_record_visited_and_removes_only_the_buffer_reference` | RECONCILED |
| `REQ-010` | `AC-012` | `driver.py: commit_event`, `_update_buffered_record` | `test_generate_on_buffered_record_updates_in_place` | RECONCILED (closure untested, `LIM-004`) |
| `REQ-004` | `AC-012` | `driver.py: _update_buffered_record` | `test_generate_on_buffered_record_updates_in_place` (`last_seen_step`) | RECONCILED |
| `REQ-005` | `AC-005` | `catalog_state.py`, `mab.py`, `driver.py` | `test_coverage_state_round_trip`, `test_coverage_state_rejects_foreign_schema`, `test_legacy_checkpoint_schemas_are_rejected` | RECONCILED |
| `REQ-005` | — | `selection.py`, `driver.py` | `test_seeded_selection_sequence_is_deterministic` | RECONCILED |
| `REQ-002` | `AC-002` | `mab.py: probabilities` (unchanged) | `test_adr_028_renormalization_still_applies_to_a_genuinely_unavailable_arm` | RECONCILED |
| `REQ-003` | `AC-003` | `selection.py` (unchanged degradation path) | `test_no_admissible_record_degrades_to_replay_then_fails_when_buffer_is_empty` | RECONCILED |
| `DEC-010` | `v1.3` §9 | `config.py`, `conf/curriculum/scenario_acl.yaml` | `test_approved_v13_buffer_capacity_resolves_and_keeps_the_warmup_guard`, `test_scenario_acl_buffer_capacity_default_is_the_approved_value` | RECONCILED |
| `DEC-011` | — | `driver.py: choose_acl_episode`, `collect_catalog_episode` | Exercised end to end by the sequential ACL smoke run (§14); no unit harness exists for this path | RECONCILED, evidence limited |

`REQ-001`, `REQ-006`, `REQ-007`, `REQ-008` are carried from `v1.2` and keep passing their existing tests
(full suite, §14). No unapproved deviation remains; `DEV-CAT-001..005` are recorded in §12.

Known limitations: `LIM-002` (LP variance inflation, irreducible in this LP family), `LIM-003` (static arm
ordering, an empirical question for the re-runs), `LIM-004` (the `commit_event` closure has no isolated harness —
`REQ-010` is verified at the helper and buffer level plus end to end via the smoke runs), `LIM-005`
(`buffer_capacity = 250` is reasoned from `RAT-011`, not ablated), `LIM-006` (`FIND-005` is established
analytically and cannot be measured retrospectively), `LIM-007` (new: cycle closure and
`generate_on_buffered_record` are not observable in a smoke-length run; see §14 for the follow-up).

Deferred required work: none. Deferred optional work: removal of the unreachable sequential ACL path (`DEC-011`),
tracked separately. External step remaining: restarting the in-progress experiments, which cannot resume under
`acl_ema_v3` by design and whose acceptance the user gave explicitly.
