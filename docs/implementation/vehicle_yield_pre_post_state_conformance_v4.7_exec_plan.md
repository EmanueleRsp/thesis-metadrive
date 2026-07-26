# ExecPlan: Vehicle-Yield Pre/Post-State Conformance Fix (v4.7)

## 1. Metadata

- Feature: `vehicle_yield_pre_post_state_conformance`
- Plan ID: `VEHICLE-YIELD-PRE-POST-CONFORMANCE-V4.7`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`,
  version `4.7-final-implementation-complete`, status `approved`,
  `Authoritative: YES`
- Governing implementation decision: DEC-005 in
  `docs/implementation/rulebook_v2_implementation_plan.md:218-236` (already
  `ACCETTATA`/approved as part of the v4.7 implementation)
- Status: `VERIFIED`
- Created: 2026-07-24
- Last updated: 2026-07-25
- Branch: `scenarionet-implementation`
- Related ADRs: none new (conformance fix to an already-approved decision,
  not a new material decision)
- Owner: n/a (single session)

## 2. Objective And Scope

Restore conformance to DEC-005: the vehicle-yield illegal-entry latch must be
judged from the **pre-state** occupancy view of priority actors, independent
of whether those actors are still present/prioritized by the time the
**post-state** is observed; the continuous approach cost must keep using the
post-state view. The current production wiring in
`_vehicle_yield_inputs` (`src/thesis_rl/rulebook/v2/transition.py`) computed
the entire priority-actor set exclusively from `post.actors`, so an actor
that occupied the conflict zone at decision time but left it during the same
control step was invisible to the latch — exactly the retroactive-
legitimization failure mode DEC-005 was written to prevent.

In scope: the vehicle-yield component's pre/post-state wiring
(`_vehicle_yield_inputs`, `evaluate_vehicle_yield`), the entry-event geometric
predicate (swept front bumper per DEC-005, previously unimplemented for this
component), and the `applicable`/`cost` aggregation bug that discarded an
active latch whenever the post-state had zero live prioritized actors.

Out of scope: the four priority predicates themselves, `MovementKey`,
conflict-zone construction/IDs, `first_ahead_or_occupied` component
selection, CTRV interval prediction, the general shape of the yield cost
formula, and the zone-selection continuity question documented as a finding
below (§11) — none of these are redesigned here.

Compatibility: this changes the numeric value of `vehicle_yield`'s cost and
of `RulebookMemory.vehicle_yield_illegal_entries` in the specific scenario
where a priority actor's occupancy of the zone does not survive the control
step (previously silently non-violating; now correctly latched). No public
interface, configuration key, or checkpoint format changes.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-VY-01 | The illegal-entry latch is created from the pre-state occupancy view of the priority actor, independent of its presence/priority in the post-state | v4.7 §7.9.5; DEC-005 |
| REQ-VY-02 | The entry event geometric predicate uses the swept front bumper pre→post, not the plain post-state footprint | v4.7 §7.9.4-5; DEC-005 |
| REQ-VY-03 | The continuous approach cost, when no entry occurs, keeps using the post-state view (unchanged) | v4.7 §7.9.5; DEC-005 |
| REQ-VY-04 | An active latch remains reflected in the aggregated cost for as long as ego occupies the zone, even with zero live post-state prioritized actors | v4.7 §7.9.5 ("fino all'uscita completa"); DEC-005 |
| REQ-VY-05 | No false entry event for occupancy pre-existing at the moment a lazy zone is first created | v4.7 §7.9.5; DEC-005 |

## 4. Current Repository Analysis

- `_vehicle_yield_inputs` (`src/thesis_rl/rulebook/v2/transition.py`, was
  lines 514-769 before this change) is called once per transition from
  `evaluate_transition` with `pre=pre_state, post=post_state`. **VERIFIED**:
  before this fix, the priority-predicate loop (`for actor in post.actors:`)
  and the `occupied` sub-predicate (`actor.footprint.intersects(zone)`)
  read exclusively from `post.actors`, contradicting DEC-005's "il predicato
  'altro veicolo già nella zona' usa il pre_state". `entered_actor_ids` was
  derived directly from this post-based set.
- `evaluate_vehicle_yield` (`src/thesis_rl/rulebook/v2/components/controls.py`,
  now line 366) already had an unused `pre_state_entered_actor_ids`
  parameter with a docstring describing the intended DEC-005 semantics, and
  a consistency guard (`if entered_actor_ids and entered_actor_ids !=
  pre_state_entered_actor_ids: raise`) that was already permissive for the
  regression case but was never exercised because no caller ever supplied
  the parameter. **VERIFIED** via repository-wide grep before this change.
- **VERIFIED**: the function's early-return branch for
  `not prioritized_intervals` unconditionally set `applicable=False,
  cost=0.0`, never running the latch-creation loop. Combined with
  `aggregate_max_component` (`src/thesis_rl/rulebook/v2/aggregation.py:23-35`)
  discarding the cost of any non-`applicable` component, an active latch
  with zero live post-state prioritized actors was silently dropped from
  the aggregated `road_traffic_compliance` cost — a second, compounding
  defect discovered during this fix (in scope, same code path).
- `swept_front_bumper` (`src/thesis_rl/rulebook/v2/geometry/footprint.py`)
  already exists and is used for `solid_line` crossing detection
  (`transition.py`, `_control_distances`) but was never passed to
  vehicle-yield's entry-event check, which used a plain post-state footprint
  intersection instead.
- Existing tests: `tests/test_rulebook_v2_vehicle_yield.py` exercised only
  the pure `evaluate_vehicle_yield` function with hand-built intervals, never
  setting `pre_state_entered_actor_ids`. `tests/test_rulebook_v2_transition.py`'s
  `test_transition_vehicle_yield_uses_each_scoped_priority_predicate` kept the
  other actor's footprint identical between `pre` and `post` in every
  sub-case, so the existing suite could not distinguish pre-state occupancy
  from post-state occupancy and would not have caught this bug.

## 5. Assumptions And Invariants

- Zone selection remains a single pass from `post_state`
  (`select_first_ahead_or_occupied_zone`, unchanged); the pre-state pass
  re-evaluates priority predicates for the already-selected zone, it does
  not perform an independent selection (see DEC-VY-01).
- `actor_keys`/`actor_lanes` (movement-key identity) are frozen-for-the-
  transition context built once from `post.actors`, per DEC-005's "gli
  altri predicati statici/topologici usano il context congelato della
  transizione"; only the dynamic `occupied` sub-predicate and the occupancy-
  interval prediction differ between the pre-state and post-state passes.
  A direct consequence (documented, not a regression): an actor entirely
  absent from `post.actors` (not merely moved out of the zone footprint,
  but never appearing in the post-state actor list at all) cannot be
  discovered by the pre-state pass either, since it never gets an
  `actor_keys` entry. This matches the existing single-post-based-selection
  architecture and is unchanged by this fix.
- `worst_case_temporal_gap_violation` (extracted pure function) preserves
  the exact pre-existing numerics of the inline loop it replaces — verified
  by running the full pre-existing test suite unchanged before adding any
  new test.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| DEC-VY-01 | Implementation detail | How to resolve pre-state vs post-state zone selection | (A) reuse the single post-selected zone for the pre-state predicate re-evaluation; (B) independent pre-state selection pass with a reconciliation rule | A | No behavioral ambiguity, no extra geometry cost, stable latch key | Approved (implementation detail, no spec deviation) |
| DEC-VY-02 | Implementation detail | Whether `applicable`/`status` must reflect an active latch when `prioritized_intervals` is empty | (A) keep the old hardcoded `applicable=False`; (B) `applicable = bool(prioritized_intervals) or active_latch_for_zone` | B | Required for REQ-VY-04 to actually reach the aggregated cost | Approved (bug fix, regression-tested) |
| DEC-VY-03 | Implementation detail | Whether to fix the swept-front-bumper entry event in the same pass | (A) same milestone, separate REQ so it is independently verifiable; (B) defer | A | Already-approved DEC-005 requirement, primitive already used elsewhere, cheap | Approved |
| DEC-VY-04 | Implementation detail | Whether `entered_actor_ids` (post-based) should still be passed to `evaluate_vehicle_yield` alongside `pre_state_entered_actor_ids` | (A) pass `frozenset()` for `entered_actor_ids`, use `pre_state_entered_actor_ids` as the sole channel; (B) pass the post-based set too | A | Passing a genuinely different post-based set risked a spurious `ValueError` from the existing consistency guard when the two sets are disjoint but both non-empty (e.g. actor A exits pre→post while actor B enters post-only); (A) matches the guard's documented intent ("supplying both must agree") | Approved |

No decision in this table alters observable behavior beyond what DEC-005
already specifies; none required new user approval.

## 7. Proposed Design

- `src/thesis_rl/rulebook/v2/geometry/conflict_zones.py`: added
  `worst_case_temporal_gap_violation(*, ego_interval, other_intervals,
  gap_scale_s) -> float`, a pure extraction (byte-identical numerics) of the
  loop previously inline in `evaluate_vehicle_yield`.
- `src/thesis_rl/rulebook/v2/components/controls.py::evaluate_vehicle_yield`:
  added `pre_state_gap_violation: float | None = None`. The function body is
  now a single unified path (no more early-return branch for empty
  `prioritized_intervals`): the illegal-entry latch gates on
  `pre_state_gap_violation` when supplied (falling back to the post-state
  `worst` for legacy callers that never separated the two snapshots); the
  approach-cost terms (`commit`, `before`, `approach`) are computed only
  when `prioritized_intervals` is non-empty; `applicable = bool
  (prioritized_intervals) or active_latch_for_zone`.
- `src/thesis_rl/rulebook/v2/transition.py`: new private helper
  `_pre_state_priority_and_gap` (line 515) re-evaluates the four priority
  predicates against `pre.actors` for the zone already selected from
  `post_state`, then predicts pre-state occupancy intervals
  (`predict_conflict_zone_occupancy_intervals` with `ego=pre.ego`) and
  reduces them via `worst_case_temporal_gap_violation` to obtain `r_gap^-`.
  `_vehicle_yield_inputs` (line 588) now: (a) detects the entry event via
  `swept_front_bumper(pre.ego.footprint, post.ego.footprint, ...)`
  intersecting the zone, instead of the plain post footprint; (b) calls
  `_pre_state_priority_and_gap` only when an entry is detected this step
  (short-circuit, avoiding the extra prediction cost on every step); (c)
  computes `exited` over the union of post-based and pre-based prioritized
  ids, with a safe `post_by_id.get(...)` lookup instead of an unguarded
  `next(...)`; (d) passes `entered_actor_ids=frozenset()` and the real
  pre-state set via the dedicated `pre_state_entered_actor_ids`/
  `pre_state_gap_violation` keys (DEC-VY-04).

No new configuration, public interface, or memory field was introduced.
`RulebookMemory.vehicle_yield_illegal_entries` and
`frozen_actor_movement_keys` keep their existing shape and ownership.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| REQ-VY-01 | TEST-VY-01, TEST-VY-04, TEST-VY-06 | `transition.py::_pre_state_priority_and_gap`, `controls.py::evaluate_vehicle_yield` | `tests/test_rulebook_v2_vehicle_yield.py::test_vehicle_yield_pre_state_gap_creates_latch_with_empty_post_state_intervals`, `tests/test_rulebook_v2_transition.py::test_transition_vehicle_yield_latches_illegal_entry_from_pre_state_even_if_actor_exits_same_step` | Verified |
| REQ-VY-02 | TEST-VY-08 | `transition.py::_vehicle_yield_inputs` (swept bumper) | Exercised indirectly by the transition-level regression tests (ego crosses the zone within one step) | Verified |
| REQ-VY-03 | TEST-VY-07 | `transition.py::_vehicle_yield_inputs` (post-state approach fields, unchanged) | `tests/test_rulebook_v2_transition.py::test_transition_vehicle_yield_approach_cost_clears_once_actor_exits_before_entry` | Verified |
| REQ-VY-04 | TEST-VY-05, TEST-VY-09 | `controls.py::evaluate_vehicle_yield` (`applicable` fix) | `tests/test_rulebook_v2_vehicle_yield.py::test_vehicle_yield_active_latch_stays_applicable_with_no_live_prioritized_actors`, `test_vehicle_yield_illegal_entry_latch_clears_once_ego_stops_occupying_zone` | Verified |
| REQ-VY-05 | (no regression) | unchanged `preexisting` handling | Existing suite (`test_transition_invokes_complete_registry_and_keeps_vehicle_yield_not_applicable` and others) unaffected | Verified (no change) |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| TEST-VY-01 | Unit | `pre_state_gap_violation>0` creates a latch even with `prioritized_intervals=()` | `evaluate_vehicle_yield(prioritized_intervals=(), pre_state_entered_actor_ids={"car"}, pre_state_gap_violation=0.6, ego_occupied=True)` | `cost==1.0`, `applicable is True`, latch written | REQ-VY-01, REQ-VY-04 |
| TEST-VY-02 | Unit | `pre_state_gap_violation=0.0` (sufficient gap) creates no latch | as above with `pre_state_gap_violation=0.0` | no latch written, `cost==0.0` | REQ-VY-01 |
| TEST-VY-03 | Unit | Legacy caller without `pre_state_gap_violation` keeps the pre-existing post-state-only formula | existing direct-call tests, unchanged | unchanged assertions pass | Backward compatibility |
| TEST-VY-04 | Integration | Priority actor occupies zone in `pre`, exits before `post`, ego enters same step → latch created | `evaluate_transition`, actor footprint inside zone pre / outside post | `("other", zone_id)` in `next_memory.vehicle_yield_illegal_entries`, `cost==1.0` | REQ-VY-01 (the regression) |
| TEST-VY-05 | Integration | Priority actor's pre-state occupancy clears with a sufficient gap before ego's predicted entry | actor exits the zone with a large lateral speed | no latch, `cost==0.0` | REQ-VY-01 |
| TEST-VY-06 | Integration | Ego does not enter, actor exits → approach cost is zero | ego stays behind the zone both steps | `cost==0.0`, no latch | REQ-VY-03 |
| TEST-VY-07 | Unit | Active latch stays applicable/violated with zero live post-state prioritized actors | `evaluate_vehicle_yield(prioritized_intervals=(), previous_illegal_entries={("car","z")}, ego_occupied=True)` | `cost==1.0`, `applicable is True` | REQ-VY-04 |
| TEST-VY-08 | Unit | Latch clears once `ego_occupied=False` for that zone | `evaluate_vehicle_yield(ego_occupied=False, previous_illegal_entries={("other","z")})` | latch removed from the written memory delta | REQ-VY-01/04 |
| TEST-VY-09 | Regression suite | No behavior change for every pre-existing rulebook v2 test | `make rulebook-v2-check` | 216/216 pass, ruff clean | Backward compatibility |

Commands:

```bash
docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_vehicle_yield.py tests/test_rulebook_v2_transition.py
make rulebook-v2-check
```

## 10. Milestones

- [x] M1: Extract `worst_case_temporal_gap_violation`; refactor
  `evaluate_vehicle_yield`'s internal call site only (no signature/behavior
  change). Evidence: full pre-existing suite green before any new test.
- [x] M2: Add `pre_state_gap_violation`, unify `evaluate_vehicle_yield`'s two
  branches, fix `applicable` (REQ-VY-01, REQ-VY-04, DEC-VY-02). Evidence:
  TEST-VY-01/02/07/08.
- [x] M3: Add `_pre_state_priority_and_gap`, wire
  `pre_state_entered_actor_ids`/`pre_state_gap_violation` from
  `_vehicle_yield_inputs` (REQ-VY-01, DEC-VY-01, DEC-VY-04). Evidence:
  TEST-VY-04/05/06.
- [x] M4: Swept-front-bumper entry event + safe `exited` computation
  (REQ-VY-02, DEC-VY-03). Evidence: exercised by TEST-VY-04/05/06 (ego
  crosses the zone within one step in each).
- [x] M5: Full reconciliation — `make rulebook-v2-check` (216 tests + ruff),
  scoped `ruff format --diff` on modified files.

## 11. Progress And Findings Log

- 2026-07-24: Implemented M1-M5. Full `tests/test_rulebook_v2_*.py` suite:
  216 passed. `ruff check` on `src/thesis_rl/rulebook/v2` and the rulebook
  CLI entry points: clean. `git diff --check`: clean.
- 2026-07-24, finding (out of scope, documented, not fixed here): while
  writing an integration-level regression test for "latch clears once ego
  fully exits the zone", discovered that `select_first_ahead_or_occupied_zone`
  (`geometry/conflict_zones.py`) can drop a zone from candidacy (the
  `ahead` filter requires `route_exit_s_m >= ego_front_s_m -
  SIGNED_DISTANCE_EPSILON_M`, epsilon `5.0e-2` m) before ego's footprint has
  fully cleared it in geometric terms (a footprint longer than `2 *
  epsilon` clears the zone only after its front point has advanced roughly
  one footprint length past the exit boundary). Once the zone is no longer a
  selection candidate, `_vehicle_yield_inputs` returns the `empty` fallback
  and the latch-clearing branch inside `evaluate_vehicle_yield`
  (`if not ego_occupied: filter by zone_id`) is never invoked for that
  `zone_id` again. Severity/consequence: in the live transition pipeline,
  this specific clearing path may be effectively unreachable for a zone ego
  has already fully entered and exited; the pure-function-level tests
  (`test_vehicle_yield_freezes_movement_key_until_complete_exit`,
  `test_vehicle_yield_illegal_entry_latch_clears_once_ego_stops_occupying_zone`)
  confirm the formula is correct in isolation, but zone-selection continuity
  in the live pipeline was never covered end-to-end by the existing suite.
  This is a pre-existing property of `select_first_ahead_or_occupied_zone`,
  unrelated to and unchanged by the DEC-005 pre/post-state wiring fixed in
  this ExecPlan — out of scope here. Flagged to the user as a separate
  follow-up item (see final report); not an approval gate for this plan.

## 12. Deviations

No deviations identified. This plan restores conformance to the
already-approved DEC-005; it does not change any specification formula,
threshold, or acceptance behavior beyond fixing the two defects described in
§4 (pre/post-state wiring and the `applicable` aggregation bug), both of
which are corrections against the existing approved contract, not new
behavior.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/geometry/conflict_zones.py` | Modified | Added `worst_case_temporal_gap_violation` |
| `src/thesis_rl/rulebook/v2/components/controls.py` | Modified | `evaluate_vehicle_yield`: new parameter, unified body, `applicable` fix |
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | `_pre_state_priority_and_gap`, swept-bumper entry event, safe `exited`, new output keys |
| `tests/test_rulebook_v2_vehicle_yield.py` | Modified | TEST-VY-01/02/03/07/08 |
| `tests/test_rulebook_v2_transition.py` | Modified | TEST-VY-04/05/06 |
| `docs/implementation/vehicle_yield_pre_post_state_conformance_v4.7_exec_plan.md` | Created | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_vehicle_yield.py tests/test_rulebook_v2_transition.py` | PASS | 2026-07-24 | 19 passed |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -k rulebook` | PASS | 2026-07-24 | 295 passed, 596 deselected |
| `make rulebook-v2-check` | PASS | 2026-07-24 | 216 passed (scoped `test_rulebook_v2_*.py`); `ruff check` clean; `git diff --check` clean |
| `docker compose run --rm dev uv run --no-sync ruff format --diff <modified files>` | PASS (scoped) | 2026-07-24 | All newly written lines match canonical formatting; one pre-existing, untouched line in `transition.py` (`diagnostic_timing_seconds["vehicle_yield_actor_scan"]`, line 733) remains outside the repository-wide formatting baseline per AGENTS.md — intentionally not reformatted (unrelated to this change) |

## 15. Final Reconciliation

- REQ-VY-01: IMPLEMENTED, VERIFIED.
- REQ-VY-02: IMPLEMENTED, VERIFIED.
- REQ-VY-03: IMPLEMENTED, VERIFIED (no behavior change, confirmed by regression test).
- REQ-VY-04: IMPLEMENTED, VERIFIED.
- REQ-VY-05: VERIFIED (no change; confirmed by unaffected existing tests).

Known limitation (not required by this plan's scope, documented in §11):
zone-selection continuity in `select_first_ahead_or_occupied_zone` may
prevent the live transition pipeline from ever re-evaluating a zone with
`ego_occupied=False` once ego's footprint has fully cleared it, which would
make the latch-clearing branch unreachable end-to-end in that specific
scenario. The formula itself is correct and covered at the pure-function
level. No deferred required work remains within this plan's scope; this
limitation is reported to the user as an independent follow-up candidate.

Update 2026-07-25: resolved and merged into this working tree. The fix
(`_cleared_vehicle_yield_illegal_entries` in `transition.py`, documented in
`ADR-025-vehicle-yield-latch-cleanup-decoupled-from-zone-selection.md`, a
different ADR-025 than this repository's own
`ADR-025-r2-lateral-rss-clearance-replacement` — the numbering collided
because the two fixes were developed in isolated worktrees) was merged via
`git rebase` of `scenarionet-implementation` onto
`origin/scenarionet-implementation` (commit `e4be417`). It decouples
illegal-entry latch cleanup from the current step's zone selection: a latch
is now dropped as soon as the ego footprint stops intersecting the zone
polygon directly (via `cache.conflict_zones`), independent of whether that
zone is still an "ahead" selection candidate. This is complementary to, not
a replacement for, the DEC-005 pre/post-state fix in this plan — the two
compose in `_vehicle_yield_inputs`: `_cleared_vehicle_yield_illegal_entries`
governs when a latch is released, `_pre_state_priority_and_gap` governs when
one is created. Full `tests/test_rulebook_v2_*.py` suite re-run here after
the merge: 272 passed; `make rulebook-v2-check` clean (tests, `ruff check`,
`git diff --check`). The known limitation described above is resolved.

Behavior: vehicle-yield now judges illegal entries from the pre-state
occupancy view and keeps an active latch reflected in the aggregated cost
regardless of the post-state's live prioritized-actor set, per DEC-005.
Architecture: `_vehicle_yield_inputs` gained one new helper
(`_pre_state_priority_and_gap`) invoked only on a detected entry event (no
added cost on non-entry steps); `evaluate_vehicle_yield` is now a single
unified code path. Compatibility: no public interface or configuration
change; only the numeric cost/memory outcome changes for the specific
previously-mishandled scenario. Ready for experimental use.
