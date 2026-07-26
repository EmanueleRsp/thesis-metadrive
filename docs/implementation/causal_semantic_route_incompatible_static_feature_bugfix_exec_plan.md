# ExecPlan: Causal Semantic Observation — Route-Incompatible Static Feature Crash

## 1. Metadata

- Feature / plan ID: `causal-semantic-route-incompatible-static-bugfix-v1`
- Authoritative specification: `docs/specifications/observation_v1.1_specification.md`
  (legacy `CausalSemanticBatchBuilder`, retained for reproducibility per
  ADR-022) and `docs/specifications/observation_v1.2_specification.md`
  (`PerceptionBoundedSemanticBatchBuilderV12`, unused/untested dead code
  path, fixed for consistency); no specification change
- Status: `VERIFIED`
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`
- Related decisions: none new — conformance fix restoring the same
  diagnostic-wrapping design already established and working in
  `_build_static_v12` (`PerceptionBoundedSemanticBatchBuilder`)
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: a static map feature (e.g. a road boundary line) whose
elevation is within `vertical_tolerance_m` of ego but does not correspond
to any vertically compatible route segment must be gracefully omitted from
the semantic observation (masked out, diagnostic counter incremented),
never crash the whole batch build with an unhandled `ValueError`.

**Why**: found while investigating 4 pre-existing test failures flagged in
`docs/project_index.md` (`tests/test_causal_semantic_batch.py`, all
`ValueError: Route projection has no vertically compatible segment`). The
base `CausalSemanticBatchBuilder._build_static` (legacy V1.1, still
actively tested) called `self.route.project(...)` for every candidate
static feature with no exception handling at all, while its sibling
`_build_static_v12` (in `PerceptionBoundedSemanticBatchBuilder`, the
actively used V1.2 observation builder) already had a `try/except`
distinguishing a genuine vertical-incompatibility skip from a structural
failure. This asymmetry meant the legacy builder crashed on exactly the
scenario its own tests were written to exercise (a feature elevated just
under `vertical_tolerance_m` from ego but with no compatible route
segment) — a real, load-bearing bug, not a stale test.

A second, independent bug was found while fixing the first: even after
adding the `try/except`, the incompatible-feature counter
(`self._route_incompatible_static_features`) never reached the public
`.diagnostics` property, because `_build_dynamic` (called before
`_build_static` in `build()`) unconditionally reconstructs
`self._last_diagnostics` with a hardcoded `route_incompatible_static_features=0`
default, and nothing re-synchronized it afterward.

**In scope**:

- Add the missing `try`/`except ValueError` around `self.route.project(...)`
  in `CausalSemanticBatchBuilder._build_static` (base/legacy class), mirroring
  `_build_static_v12`'s existing, already-reviewed pattern exactly: on a
  genuine vertical-incompatibility (`projection_diagnostics(...).vertically_compatible_segment_count == 0`),
  increment the counter and skip the feature; on any other `ValueError`,
  re-raise as `CausalSemanticObservationError` with full diagnostic context.
- Apply the identical fix to `PerceptionBoundedSemanticBatchBuilderV12._build_static`
  (a second override with the same gap), for consistency, even though this
  class is currently unused/untested elsewhere in the repository.
- Fix the diagnostics-propagation gap: after `_build_static` runs in both
  affected `build()` methods, refresh `self._last_diagnostics` with the
  current `route_incompatible_static_features` count via `dataclasses.replace`.
- Update `test_structural_static_projection_error_is_propagated` to expect
  the new `CausalSemanticObservationError` (a `ValueError` subclass) with
  diagnostic context, instead of the raw synthetic message — the old
  expectation reflected the base class's previously *missing* handling, not
  a deliberate contract; the new expectation matches the design already in
  production use via `_build_static_v12`.

**Out of scope**:

- `PerceptionBoundedSemanticBatchBuilder._build_static_v12` itself (already
  correct, untouched).
- Any change to the observation tensor shape, feature semantics, or
  `SemanticOverflowDiagnostics` schema.
- The broader "diagnostics reflects only the most recently run `_build_*`
  method" architecture — only the one field (`route_incompatible_static_features`)
  needed for this bug is patched; a full audit of every diagnostic field's
  propagation ordering was not performed.

**Compatibility**: no public interface, configuration key, or checkpoint
schema change. Observable behavior change: a static feature that used to
crash the whole observation build now instead is a no-op (masked out,
never contributed to the tensor) — a strict availability improvement, not
a semantic change to what a *valid* observation contains.

## 3. Authoritative Requirements

| ID | Requirement | Notes |
|---|---|---|
| `REQ-001` | A static feature whose route projection has zero vertically compatible segments must be skipped (masked, counted), not crash the build. | Matches the already-established `_build_static_v12` contract. |
| `REQ-002` | A static feature whose route projection fails for a genuinely structural reason (not vertical incompatibility) must still surface as an error, now with diagnostic context, not silently swallowed. | Same distinction already made by `_build_static_v12`. |
| `REQ-003` | `.diagnostics.route_incompatible_static_features` must reflect the actual count after `_build_static` runs in the same `build()` call. | New requirement — the counter existed but was never wired to the public property in the base class. |

## 4. Current Repository Analysis

- `VERIFIED`: `CausalSemanticBatchBuilder` (base class, `causal_semantic.py:160`)
  is the legacy V1.1 builder, superseded for new development by ADR-022 but
  still directly imported and unit-tested by `tests/test_causal_semantic_batch.py`
  — not dead code.
- `VERIFIED`: `_build_static_v12` (`PerceptionBoundedSemanticBatchBuilder`,
  the actively used V1.2 builder) already has the correct `try/except`
  pattern; used as the direct template for this fix.
- `VERIFIED`: `PerceptionBoundedSemanticBatchBuilderV12` (a third class,
  `causal_semantic.py:1868`) has its own `_build_static` override with the
  identical gap, but is not imported or referenced anywhere else in the
  repository (`grep` across `src/` and `tests/` found only its own
  definition) — fixed for consistency, not covered by any existing test.
- `VERIFIED`: `_build_dynamic` unconditionally reconstructs
  `self._last_diagnostics` with `SemanticOverflowDiagnostics(...)` using the
  dataclass default `route_incompatible_static_features=0`, and nothing in
  the base class's `build()` re-synchronizes it after `_build_static` runs
  — confirmed by direct inspection after the `try/except` fix alone did not
  make the failing diagnostics assertion pass.

## 5. Assumptions And Invariants

- `self._route_incompatible_static_features` is a persistent, per-episode
  cumulative counter (reset only in `reset()`, called on scenario change),
  not a per-step count — unchanged by this plan.
- `CausalSemanticObservationError` already inherits from `ValueError`
  (`causal_semantic.py:41`), so existing callers catching `ValueError`
  broadly are unaffected; only the exact message/type distinction changes
  for the one updated test.

## 6. Decisions And Approval Gates

No new material decision: this restores the base builder to the same
diagnostic-handling contract already implemented and accepted for
`_build_static_v12`. No specification change.

## 7. Proposed Design

Mirror `_build_static_v12`'s existing `try/except ValueError` block (with
`projection_diagnostics(...)` and the `CausalSemanticObservationError`
diagnostic message) into `CausalSemanticBatchBuilder._build_static` and
`PerceptionBoundedSemanticBatchBuilderV12._build_static`. Add, in both
affected `build()` methods, immediately after the `_build_static` call:

```python
self._last_diagnostics = replace(
    self._last_diagnostics,
    route_incompatible_static_features=self._route_incompatible_static_features,
)
```

## 8. Traceability

| Requirement | Implementation | Tests | Status |
|---|---|---|---|
| `REQ-001` | `causal_semantic.py::CausalSemanticBatchBuilder._build_static` | `tests/test_causal_semantic_batch.py::test_route_incompatible_static_feature_is_omitted_and_masked`, `test_mixed_static_features_keep_compatible_feature`, `test_all_incompatible_static_features_preserve_shape_and_empty_mask`, `test_waymo_feature_16_equivalent_geometry_is_omitted` | Verified |
| `REQ-002` | Same location | `tests/test_causal_semantic_batch.py::test_structural_static_projection_error_is_propagated` (updated) | Verified |
| `REQ-003` | `causal_semantic.py::CausalSemanticBatchBuilder.build` | Same 4 tests above (assert `builder.diagnostics.route_incompatible_static_features`) | Verified |

## 9. Test Strategy

All 5 requirement-relevant tests pre-existed in
`tests/test_causal_semantic_batch.py` (written before this fix, correctly
identifying the bug); one (`test_structural_static_projection_error_is_propagated`)
was updated to match the now-consistent diagnostic-wrapping contract. No
new test file created.

Commands:

- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py`
- `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"` (full non-integration suite)

## 10. Milestones

### M1 — Fix and verify

- Objective: add missing exception handling, fix diagnostics propagation,
  realign the one stale test expectation, verify no regression.
- Status: Done and verified (Section 14).

## 11. Progress And Findings Log

- 2026-07-26 — Investigated as part of a broader "fix pending bugs before
  real runs" request. Found the try/except gap first; fixing it alone left
  one test (`test_waymo_feature_16_equivalent_geometry_is_omitted`) still
  failing on the diagnostics assertion, which led to discovering the
  separate `_build_dynamic`-overwrites-`_last_diagnostics` ordering bug.
  Fixed both. Re-running `test_structural_static_projection_error_is_propagated`
  then failed for an expected reason: the fix intentionally changes
  behavior for genuinely structural errors (from raw passthrough to a
  wrapped, diagnostic-rich `CausalSemanticObservationError`), matching
  `_build_static_v12`'s already-accepted design — updated the test's
  expectation accordingly rather than reverting the fix. Also fixed the
  identical gap in the unused/untested `PerceptionBoundedSemanticBatchBuilderV12._build_static`
  for consistency. Full non-integration suite (972 tests) re-run clean
  after the fix, up from 968 passed / 4 failed before.

## 12. Deviations

No deviations identified. No specification, formula, or tensor-shape
change.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modified | Add `try/except` in both `_build_static` overrides; add `replace(...)` diagnostics refresh in both `build()` methods; `from dataclasses import dataclass, replace` |
| `tests/test_causal_semantic_batch.py` | Modified | Update `test_structural_static_projection_error_is_propagated` to expect `CausalSemanticObservationError` with diagnostic context |
| `docs/implementation/causal_semantic_route_incompatible_static_feature_bugfix_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py` | PASS | 2026-07-26 | `15 passed` (was `4 failed, 10 passed`, then `5 failed, 10 passed` mid-fix before the test realignment, then all green) |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"` (full non-integration suite) | PASS | 2026-07-26 | `972 passed, 5 deselected` (was `968 passed, 4 failed, 5 deselected` before this fix) |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/causal_semantic.py tests/test_causal_semantic_batch.py` | PASS | 2026-07-26 | `All checks passed!` |
| `docker compose run --rm dev uv run --no-sync ruff format --check` (same files) | PARTIAL | 2026-07-26 | `causal_semantic.py` "would be reformatted"; `ruff format --diff` shows only pre-existing, untouched lines (confirmed zero overlap with this diff via `git diff` cross-check) — pre-existing repository formatting debt per `AGENTS.md`, not introduced here. Test file already formatted. |

## 15. Final Reconciliation

- `REQ-001`, `REQ-002`, `REQ-003`: `IMPLEMENTED` and `VERIFIED` — all 4
  originally-failing tests plus the realigned structural-error test pass;
  full non-integration suite confirms no regression (972 passed).

**Resulting behavior**: the legacy V1.1 semantic observation builder no
longer crashes on a static feature that is geometrically near ego but has
no vertically compatible route segment; such features are now correctly
omitted and counted, matching the already-established V1.2 contract. A
genuinely structural projection failure still surfaces, now with richer
diagnostic context instead of a bare message.

**Architecture/compatibility**: no public interface, configuration key, or
checkpoint schema changed.

**Executed checks**: see Section 14.

**Approved decisions**: none required (bug fix restoring an already-accepted
design pattern to a sibling implementation).

**Deviations**: none (Section 12).

**Known limitations**: the broader ordering/propagation of other
`SemanticOverflowDiagnostics` fields across the various `_build_*` methods
was not audited beyond the one field needed for this fix; if another
diagnostic field is later found stale for the same structural reason, it
would need the same `replace(...)` treatment.

**Deferred optional work**: none identified.
