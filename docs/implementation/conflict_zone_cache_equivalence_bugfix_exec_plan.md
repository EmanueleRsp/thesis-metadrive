# Conflict-zone cache equivalence bugfix

## 1. Metadata

- Feature and plan ID: Rulebook v2 conflict-zone cache equivalence, `RBV2-CACHE-EQ-001`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete`, approved
- Status: `IMPLEMENTED`
- Created: 2026-07-22
- Branch: working tree
- Related ADRs: `docs/decisions/ADR-003-causal-ctrv-conflict-zone-prediction.md`

## 2. Objective And Scope

Allow a lazy conflict zone to be proposed again after it has been committed when
its canonical geometry is unchanged. Preserve rejection
of genuinely different geometries and atomic cache commits. No production
behavior outside cache equivalence is in scope.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-001 | Existing zones are accepted when their canonical geometry is byte-equivalent; the committed record remains authoritative. | §3.2 |
| REQ-002 | A different geometry for an existing zone is rejected without mutation. | §3.2, §3.3 |

## 4. Current Repository Analysis

- VERIFIED: `src/thesis_rl/rulebook/v2/memory.py` compares frozen records with
  dataclass equality; Shapely polygon instances recreated from identical
  coordinates are not equivalent under that comparison.
- VERIFIED: `transition.py` lazily reconstructs `ConflictZoneRecord` instances
  for vehicle-yield transitions.
- VERIFIED: existing tests cover conflicting geometry and atomicity, but not
  equivalent geometry represented by a distinct Shapely object.

## 5. Assumptions And Invariants

Canonical geometry uses `canonical_geometry_wkb` and the repository precision
grid. Existing cache records are retained on equivalent re-proposals; dynamic
metadata in a repeated lazy proposal cannot overwrite committed cache state.

## 6. Decisions And Approval Gates

No unresolved decision gates. The fix implements the already-authoritative
canonical-equivalence requirement.

## 7. Proposed Design

Add a private canonical-geometry equivalence helper in `memory.py`. Use it in both
`merge_cache_deltas` and `apply_cache_delta`, replacing dataclass equality while
retaining the existing error behavior and atomicity.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| REQ-001 | Equivalent geometry and changed proposal metadata are accepted | `memory.py` | `tests/test_rulebook_v2_memory.py::test_cache_accepts_equivalent_geometry_from_a_distinct_shapely_object` | VERIFIED |
| REQ-002 | Different geometry is rejected and cache is unchanged | `memory.py` | `tests/test_rulebook_v2_memory.py::test_cache_merge_and_apply_reject_conflicting_geometry_without_mutation` | VERIFIED |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Expected result | Requirement |
|---|---|---|---|---|
| TEST-001 | Unit regression | Same record fields, distinct equivalent polygon object | merge/apply succeeds and retains committed record | REQ-001 |
| TEST-002 | Unit regression | Same zone ID, different geometry | `ValueError`, no mutation | REQ-002 |

Commands: `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_memory.py`;
`git diff --check`; focused Ruff via `make lint` if available in the environment.

## 10. Milestones

- [x] Implement canonical structural equivalence and regression test.
- [x] Run focused tests and repository checks.
- [x] Reconcile plan and final diff.

## 11. Progress And Findings Log

### 2026-07-22

- Reproduced from the supplied log: worker slot 1 fails in
  `apply_cache_delta` with `Committed conflict zone cannot be modified`.
- Root cause identified as Shapely object equality inside frozen dataclass
  equality, not a TD3 or vectorization issue.
- Implemented canonical geometry plus structural-field comparison in both cache
  merge paths; equivalent re-proposals now retain the committed record.
- Relaxed the comparison to canonical geometry only after the follow-up training
  log showed state-derived metadata differences for the same stable zone ID.
- `pytest -q tests/test_rulebook_v2_memory.py`: 5 passed.
- `git diff --check`: passed.
- The wrapper test collection could not run because the system interpreter lacks
  the repository dependency `rich`; `uv` could not create its cache temporary in
  the sandbox's read-only home cache.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/memory.py` | Modify | Canonical conflict-zone equivalence |
| `tests/test_rulebook_v2_memory.py` | Modify | Regression coverage |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_memory.py` | NOT_RUN | 2026-07-22 | `uv` blocked by read-only `/home/e.respino/.cache/uv` |
| `pytest -q tests/test_rulebook_v2_memory.py` | PASS | 2026-07-22 | 5 passed |
| `pytest -q tests/test_rulebook_v2_memory.py tests/test_rulebook_v2_wrapper.py` | NOT_RUN | 2026-07-22 | Collection blocked by missing `rich` in system interpreter |
| `git diff --check` | PASS | 2026-07-22 | No whitespace errors |

## 15. Final Reconciliation

REQ-001 and REQ-002 are implemented and verified by focused unit tests. The
known limitation is environmental: the full wrapper regression was not
collectable without the provisioned project environment. No specification
deviation was introduced.
