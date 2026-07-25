# ExecPlan: Vehicle-Yield Illegal-Entry Latch Cleanup Decoupled From Zone Selection

## 1. Metadata

- Feature / plan ID: `vehicle-yield-latch-cleanup-v1`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`,
  version 4.7-final-implementation-complete, `Authoritative: YES` (§2.8.2 is
  read but not changed by this plan)
- Status: `VERIFIED`
- Created: 2026-07-24
- Last updated: 2026-07-24
- Branch: `claude/eager-payne-40c302`
- Related ADR: `docs/decisions/ADR-025-vehicle-yield-latch-cleanup-decoupled-from-zone-selection.md`
- Owner: thesis repository maintainer (session-driven investigation)

## 2. Objective And Scope

**Objective**: once the ego footprint has genuinely and fully left a
vehicle-yield conflict zone, `memory.vehicle_yield_illegal_entries` must stop
reporting an active illegal entry for that zone, both for the component's own
cost formula and for the policy observation feature at
`src/thesis_rl/envs/observations/causal_semantic.py:1263-1266` that reads the
same memory field.

**Why**: a follow-up investigation (triggered by a note left in an unrelated
DEC-005 pre/post-state conformance fix) found that `§2.8.2` zone selection
(`select_first_ahead_or_occupied_zone`) drops a zone from candidacy — via its
"ahead" epsilon filter (`SIGNED_DISTANCE_EPSILON_M = 0.05` m) — well before
the ego footprint (vehicle length on the order of metres) actually stops
intersecting it. At the exact frame `ego_occupied` becomes `False` for a real
zone, that zone is already unselectable, so `_vehicle_yield_inputs` falls
back to the sentinel `empty` domain (`zone_id = "__no_vehicle_priority__"`),
and the pure evaluator's own cleanup (keyed on the *currently selected*
`zone_id`) never matches the real, stale entry. This is a live-pipeline bug,
not a synthetic-test artifact: it is geometrically certain for any vehicle
length greater than `2 * epsilon` (i.e. essentially always).

**In scope**:

- A wiring-layer fix inside `_vehicle_yield_inputs`
  (`src/thesis_rl/rulebook/v2/transition.py`) that clears stale
  `(actor_id, zone_id)` latch entries using cached zone geometry
  (`cache.conflict_zones`), independent of which zone is selected for the
  current step's cost evaluation.
- An end-to-end regression test driving `evaluate_transition` through and
  past a vehicle-yield conflict zone across multiple steps.

**Out of scope**:

- Any change to `select_first_ahead_or_occupied_zone` or §2.8.2 itself
  (rejected alternative, recorded in ADR-025).
- Any change to `evaluate_vehicle_yield`'s pure signature, contract, or
  existing unit tests.
- The unrelated DEC-005 pre/post-state conformance fix that surfaced this
  finding (already completed prior to this plan).

**Compatibility**: no public interface, configuration key, checkpoint
schema, or reward-vector shape change. Only the temporal accuracy of the
`vehicle_yield_illegal_entries` memory field (and the policy observation
feature derived from it) changes.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Once the ego footprint no longer intersects a vehicle-yield conflict zone's cached polygon, no `(actor_id, zone_id)` entry for that zone may remain in `memory.vehicle_yield_illegal_entries` after the transition step that observes the exit. | Derived from §2.8.2 (`select_first_ahead_or_occupied_zone`) and §7.9 (scoped vehicle-yield predicates); this is a memory-management correction, not a specification behavior change. |
| `REQ-002` | The currently selected zone's cost/approach computation for the step (§2.8.2, §7.9) must be byte-identical to the pre-fix behavior. | §2.8.2, §7.9 |

## 4. Current Repository Analysis

- `SPECIFIED`: §2.8.2 `first_ahead_or_occupied` selection criterion —
  `docs/specifications/rulebook_v4.7_specification.md:546-557`.
- `VERIFIED`: `select_first_ahead_or_occupied_zone` implements §2.8.2 exactly
  (occupied-max-area first, then "ahead" filtered by
  `SIGNED_DISTANCE_EPSILON_M`) — `src/thesis_rl/rulebook/v2/geometry/conflict_zones.py:345-379`.
- `VERIFIED`: `_vehicle_yield_inputs` falls back to a sentinel `empty` domain
  (`zone_id = "__no_vehicle_priority__"`) whenever `selected is None` —
  `src/thesis_rl/rulebook/v2/transition.py:547-566,670-671,722-723`.
- `VERIFIED`: `evaluate_vehicle_yield` clears `previous_illegal_entries` only
  for entries matching the *passed-in* `zone_id`, in both the
  no-prioritized-intervals branch and the main branch —
  `src/thesis_rl/rulebook/v2/components/controls.py:400-403,448-449`.
- `VERIFIED`: `cache.conflict_zones: Mapping[str, ConflictZoneRecord]` is
  keyed by `zone_id` and is populated for every zone ever selected via
  `CacheDelta.new_conflict_zones` (`transition.py:736-767`), merged into the
  next step's cache by `apply_cache_delta`
  (`src/thesis_rl/rulebook/v2/memory.py:216`,
  `src/thesis_rl/rulebook/v2/wrapper.py:185`). Consequently, by the time a
  `(actor_id, zone_id)` latch entry can exist in memory, `zone_id` is
  guaranteed present in `cache.conflict_zones` for the following step.
- `VERIFIED`: `ego_occupied` in the live pipeline is defined identically as
  `post.ego.footprint.intersects(zone)` with no vertical-compatibility gate
  (`transition.py:725`); the cleanup added by this plan reuses the same
  definition against the cached polygon, so it does not introduce a new
  occupancy semantics.
- `VERIFIED`: a second, independent consumer of
  `memory.vehicle_yield_illegal_entries` exists at
  `src/thesis_rl/envs/observations/causal_semantic.py:1263-1266` (policy
  observation feature `pair in vehicle_yield_illegal_entries`), confirming
  the bug's impact extends beyond the component's own cost formula.
- `VERIFIED` (mathematical, not requiring container execution): for any
  candidate zone with curvilinear exit `s_exit` and ego footprint length `L`,
  "ahead" becomes false at `front_s = s_exit + 0.05`, while `ego_occupied`
  becomes false only at `front_s = s_exit + L`. Since `L` (metres, real
  vehicle) is always `> 2 * 0.05` m, the "ahead" exclusion always precedes
  the occupancy exit, so `selected is None` at the exact frame `ego_occupied`
  transitions `True -> False` for that zone (absent another concurrently
  occupied/ahead candidate).
- Existing tests: `tests/test_rulebook_v2_vehicle_yield.py` (pure evaluator
  only), `tests/test_rulebook_v2_transition.py` (end-to-end transition
  wiring, including
  `test_transition_vehicle_yield_uses_each_scoped_priority_predicate` and
  `test_vehicle_conflict_pair_cache_reuses_complete_canonical_candidates`,
  which provide the fixture pattern reused for the new regression test).
- No existing end-to-end test drove ego fully through and past a
  vehicle-yield zone across multiple `evaluate_transition` steps with
  `apply_cache_delta` chaining before this plan.

## 5. Assumptions And Invariants

- Route curvilinear coordinate `s` increases monotonically along the task
  route; ego front/rear footprint bounds are derived consistently with
  `select_first_ahead_or_occupied_zone`'s existing usage.
- `cache.conflict_zones` polygons are canonical and stable for a given
  `zone_id` for the lifetime of an episode (already relied upon by the
  pre-existing cache-reuse test).
- The fix must not alter `evaluate_vehicle_yield`'s pure signature or
  observable per-call behavior for the currently selected zone (`REQ-002`);
  it only changes what `_vehicle_yield_inputs` supplies as
  `previous_illegal_entries`.
- Determinism: the cleanup is a pure function of `cache`, `memory`, and the
  post-state ego footprint already available in `_vehicle_yield_inputs`; no
  new randomness or ordering dependency is introduced (result built as a
  `frozenset`).

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification deviation vs. implementation detail | How to stop the stale-latch leak without changing §2.8.2 selection semantics | (A) decouple latch cleanup from selection in `_vehicle_yield_inputs`; (B) extend §2.8.2 "ahead" predicate to include latched zones; (C) document only, defer implementation | A | Memory-management correctness only, no observable selection/cost change for the current-step zone | Approved — user selected option A ("Pulizia latch disaccoppiata") on 2026-07-24 |

Recorded in `docs/decisions/ADR-025-vehicle-yield-latch-cleanup-decoupled-from-zone-selection.md`.

## 7. Proposed Design

Add a small pure helper in `src/thesis_rl/rulebook/v2/transition.py`,
private to the module (no new public interface):

```python
def _cleared_vehicle_yield_illegal_entries(
    *, cache: EpisodeCache, memory: RulebookMemory, ego_footprint: BaseGeometry
) -> frozenset[tuple[str, str]]:
    """Drop illegal-entry latches for zones the ego footprint has already left.

    §2.8.2 selection can exclude a zone from "ahead" candidacy (epsilon-scoped)
    long before the ego footprint stops intersecting it (vehicle-length
    scoped), so the currently selected zone_id is not a reliable signal for
    releasing latches on zones ego has fully exited (ADR-025).
    """
    active = set(memory.vehicle_yield_illegal_entries)
    for actor_id, zone_id in memory.vehicle_yield_illegal_entries:
        record = cache.conflict_zones.get(zone_id)
        if record is not None and not ego_footprint.intersects(record.polygon):
            active.discard((actor_id, zone_id))
    return frozenset(active)
```

`_vehicle_yield_inputs` computes this once, from `post.ego.footprint`, before
building the sentinel `empty` domain, and uses it (instead of the raw
`memory.vehicle_yield_illegal_entries`) for `previous_illegal_entries` in
every return path (the two early-return `empty` paths and the full domain
dict). `evaluate_vehicle_yield` itself is unchanged: its own per-current-zone
cleanup still runs and remains correct for the zone actually selected this
step, now operating on an already-cleared input set.

No new component, no new memory field, no new configuration, no fallback
policy change. The only altered runtime path is the value passed as
`previous_illegal_entries`.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `src/thesis_rl/rulebook/v2/transition.py::_cleared_vehicle_yield_illegal_entries`, `_vehicle_yield_inputs` | `tests/test_rulebook_v2_transition.py::test_transition_clears_vehicle_yield_illegal_entry_latch_after_ego_fully_exits_zone` | Verified |
| `REQ-002` | `AC-002` | (no change to `evaluate_vehicle_yield`) | `tests/test_rulebook_v2_vehicle_yield.py` (unmodified, still passing), `tests/test_rulebook_v2_transition.py::test_transition_vehicle_yield_uses_each_scoped_priority_predicate` (unmodified, still passing) | Verified |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Integration (end-to-end `evaluate_transition`, multi-step, `apply_cache_delta` chained) | Illegal-entry latch clears once ego footprint has fully and demonstrably left the zone, even though §2.8.2 "ahead" already excluded the zone earlier | Ego and a conflicting `other` vehicle drive through a lane-a/lane-b conflict zone; `other` occupies concurrently with ego to record an illegal entry; ego then advances several steps past the zone (footprint fully clear) | `next_memory.vehicle_yield_illegal_entries` is empty once the ego footprint no longer intersects the recorded zone polygon | `REQ-001` |
| `TEST-002` (regression guard) | Unit | Existing pure-evaluator tests remain valid and unmodified | `tests/test_rulebook_v2_vehicle_yield.py` (unchanged) | All pass unchanged | `REQ-002` |
| `TEST-003` (regression guard) | Integration | Existing scoped-priority-predicate transition test remains valid and unmodified | `tests/test_rulebook_v2_transition.py::test_transition_vehicle_yield_uses_each_scoped_priority_predicate` (unchanged) | Passes unchanged | `REQ-002` |

Commands:

- Focused test run inside the provisioned container:
  `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_vehicle_yield.py`
- Focused lint/format for touched files:
  `make lint PYTHON_QUALITY_PATHS="src/thesis_rl/rulebook/v2/transition.py tests/test_rulebook_v2_transition.py"`,
  `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/rulebook/v2/transition.py tests/test_rulebook_v2_transition.py"`

No global `make test` / `make format-check` run is claimed as a completion
gate per the repository-wide formatting baseline note in `AGENTS.md`.

## 10. Milestones

### M1 — Fix and regression test

- Objective: implement `_cleared_vehicle_yield_illegal_entries`, wire it into
  `_vehicle_yield_inputs`, add `TEST-001`.
- Status: Done and verified (Section 14).
- Expected files: `src/thesis_rl/rulebook/v2/transition.py`,
  `tests/test_rulebook_v2_transition.py`.
- Tests/commands: see Section 9.
- Completion evidence: see Section 14.
- Decision dependencies: `DEC-001` (approved).

## 11. Progress And Findings Log

- 2026-07-24 — Investigation opened as a follow-up to a note left in an
  unrelated DEC-005 pre/post-state conformance fix (that fix's own ExecPlan
  file was not found in this branch/worktree; only the finding description
  was available). Confirmed via code reading and pure geometric/inequality
  analysis (no container execution required) that the bug is real and
  affects the live training pipeline for every vehicle-yield zone crossing,
  not only synthetic tests with large position jumps. Found a second,
  previously unnoted impact: `causal_semantic.py:1263-1266` exposes the same
  stale memory field as a policy observation feature, so the bug corrupts a
  live policy input, not only an internal cost value. Presented findings and
  three remediation options to the user; user approved "decouple latch
  cleanup from zone selection" (Option A / `DEC-001`). Recorded in ADR-025.
  Proceeded to implement the wiring-layer fix and the end-to-end regression
  test in this same session.
- 2026-07-24 (continued) — Initialized the previously-uninitialized
  `third_party` git submodules to obtain a working `docker compose run --rm
  dev` environment, then ran the mandatory test matrix. The first draft of
  `TEST-001` failed for a test-construction reason unrelated to the fix: the
  `other` actor's position/heading caused `associate_route_lane` to
  ambiguously resolve to `lane-a` instead of `lane-b` (both lane polygons
  covered the chosen point), collapsing the conflict zone to the entire
  lane. Corrected by centring `other` outside lane-a's lateral extent (as
  the pre-existing `test_transition_vehicle_yield_uses_each_scoped_priority_predicate`
  does) while keeping its footprint overlapping the true conflict zone.
  After the correction, empirical tracing confirmed the predicted mechanism
  exactly: the zone stays selected via the "occupied" branch through
  `front_s = 12.5` (already past the "ahead" cutoff at `front_s <= 12.06`),
  and `select_first_ahead_or_occupied_zone` returns `None` only once
  `front_s = 14.5` (ego fully clear) -- at which point, without the fix, the
  latch entry survives, and with the fix it is already cleared. Verified the
  fix is load-bearing by reverting only `transition.py` via `git stash` and
  confirming `TEST-001` fails with the exact predicted stale entry, then
  restored it. Full validation evidence recorded in Section 14; plan status
  raised to `VERIFIED`.

## 12. Deviations

No deviations identified. `select_first_ahead_or_occupied_zone` and §2.8.2
are unchanged; `evaluate_vehicle_yield`'s pure contract is unchanged.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | Add `_cleared_vehicle_yield_illegal_entries` and use it for `previous_illegal_entries` in `_vehicle_yield_inputs` |
| `tests/test_rulebook_v2_transition.py` | Modified | Add `TEST-001` end-to-end regression test |
| `docs/decisions/ADR-025-vehicle-yield-latch-cleanup-decoupled-from-zone-selection.md` | Added | Record the approved decision |
| `docs/implementation/vehicle_yield_illegal_entry_latch_cleanup_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `git submodule update --init --recursive` (environment bootstrap, not part of the mandatory matrix) | PASS | 2026-07-24 | Required once because `third_party/{metadrive,scenarionet,stable-baselines3}` were uninitialized in this worktree, which made `docker compose run --rm dev` fail at `uv sync` (`metadrive-simulator` "does not appear to be a Python project"). |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_vehicle_yield.py` | PASS | 2026-07-24 | `12 passed, 1 warning`. Includes the new `TEST-001` (`test_transition_clears_vehicle_yield_illegal_entry_latch_after_ego_fully_exits_zone`) and the unmodified `TEST-002`/`TEST-003` guards. |
| Same command, fix temporarily reverted via `git stash` on `transition.py` only | FAIL (expected) | 2026-07-24 | `test_transition_clears_vehicle_yield_illegal_entry_latch_after_ego_fully_exits_zone` fails with `AssertionError: assert frozenset({('other', '01fc683d...')}) == frozenset()` — proves the new test actually exercises the bug and the fix is what makes it pass. Fix restored via `git stash pop` immediately after. |
| `docker compose run --rm dev uv run --no-sync ruff check --no-cache src/thesis_rl/rulebook/v2/transition.py tests/test_rulebook_v2_transition.py` | PASS | 2026-07-24 | `All checks passed!` (`--no-cache` used because `.ruff_cache` is not writable by the container user in this worktree). |
| `docker compose run --rm dev uv run --no-sync ruff format --no-cache --check src/thesis_rl/rulebook/v2/transition.py tests/test_rulebook_v2_transition.py` | PARTIAL | 2026-07-24 | `transition.py` "would be reformatted"; `ruff format --diff` shows the only hunk is at line ~681 (`vehicle_yield_actor_scan` timing assignment), which this plan did not touch — pre-existing repository formatting debt per `AGENTS.md`, not introduced by this change. `test_rulebook_v2_transition.py` is already formatted. |
| `git diff --check` | PASS | 2026-07-24 | No whitespace errors. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/ -k "rulebook_v2"` (broader regression sweep, not part of the mandatory matrix) | PASS (1 unrelated failure) | 2026-07-24 | `206 passed, 3 skipped`, plus 1 failure in `test_rulebook_v2_live_integration.py::test_collision_hook_preserves_waymo_dynamics_for_fixed_seed_and_actions` with `PermissionError: .../third_party/metadrive/metadrive/assets.lock` — a file-lock permission artifact from the just-initialized submodule directory owned by the host user versus the container user, unrelated to vehicle-yield and pre-existing to this change (not touched by this plan's diff). |

## 15. Final Reconciliation

- `REQ-001`: `IMPLEMENTED` and `VERIFIED` — `TEST-001` passes with the fix and
  was confirmed to fail without it (revert-and-rerun above), demonstrating
  the latch now clears at the exact step `ego_occupied` transitions to
  `False`, even though `select_first_ahead_or_occupied_zone` had already
  excluded the zone from "ahead" candidacy earlier.
- `REQ-002`: `IMPLEMENTED` and `VERIFIED` — `evaluate_vehicle_yield` was not
  modified; `TEST-002` (`tests/test_rulebook_v2_vehicle_yield.py`, unchanged)
  and `TEST-003`
  (`test_transition_vehicle_yield_uses_each_scoped_priority_predicate`,
  unchanged) both pass unmodified.

**Resulting behavior**: `memory.vehicle_yield_illegal_entries` no longer
retains an `(actor_id, zone_id)` entry once the ego footprint has genuinely
left that zone's cached polygon, regardless of whether the zone is still
selected for the current step's own cost/approach evaluation. The policy
observation feature at `causal_semantic.py:1263-1266` that reads the same
memory field is corrected as a consequence, without any change to its own
code or to the observation contract shape.

**Architecture/compatibility**: no public interface, configuration key,
checkpoint schema, or reward-vector shape changed. `select_first_ahead_or_occupied_zone`
and rulebook v4.7 §2.8.2 are unchanged.

**Executed checks**: see Section 14 (focused pytest, ruff lint, ruff
format-check, `git diff --check`, and a broader `rulebook_v2`-scoped pytest
sweep for additional regression confidence).

**Approved decisions**: `DEC-001` (ADR-025), user-approved on 2026-07-24.

**Deviations**: none (Section 12).

**Known limitations**:
- `make lint`/`make format-check` (the `Makefile` targets) were not invoked
  directly because their default `PYTHON_QUALITY_PATHS` is repo-wide and the
  repository-wide formatting baseline is not clean (per `AGENTS.md`); the
  equivalent focused `docker compose run --rm dev uv run --no-sync ruff ...`
  invocations were used instead, scoped to the two touched files.
- `make test` (the full suite) was not run; the focused command plus the
  broader `-k rulebook_v2` sweep were used as the practical equivalent for
  this change's blast radius. Follow-up: run `make test` in CI or a later
  session if full-suite confidence is required.
- The one failing test outside this change's scope
  (`test_collision_hook_preserves_waymo_dynamics_for_fixed_seed_and_actions`)
  is an environment artifact (submodule file-lock permissions after this
  session's `git submodule update --init`) and is unrelated to vehicle-yield;
  it was not investigated further as it is out of scope for this plan.

**Deferred optional work**: none identified.
