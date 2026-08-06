# ExecPlan: Rulebook Scalarization v1.1 — Priority-Weighted Rank Mode And Reward Compression

## 1. Metadata

- Feature: Fourth scalarization mode (`bounded_priority_weighted_rank`) and optional post-hoc `symlog` reward compression
- Plan ID: `SCAL-V1.1-PLAN`
- Specification path: `docs/specifications/rulebook_scalarization_v1.1_specification.md`
- Specification ID/version: `SCAL-V1.1`, `1.1` (amends `SCAL-V1.0`, `1.0`)
- Specification authority: `APPROVED`; `Authoritative: YES`
- Plan status: `IMPLEMENTED`
- Created: 2026-08-06
- Last updated: 2026-08-06
- Branch: `codex/route-coordinate-mission` (current worktree)
- Related ADRs: `docs/decisions/ADR-011-rulebook-scalarization-v1.md`, `docs/decisions/ADR-056-wrongway-cost-physics-noise-deadband.md`, `docs/decisions/ADR-057-scalarization-v1.1-defaults.md`
- Owner: thesis repository maintainer
- Approval evidence: explicit user approval in this conversation, 2026-08-06 ("Top. Avevo letto le altre decisioni e approvo i suggerimenti, quindi procedi pure con l'implementazione automatica del piano")

## 2. Objective And Scope

Add a fourth scalarization mode that embeds each rule's continuous margin
severity inside its own priority-weighted term (`priority_base=3.0`) instead
of a shared, equally-weighted tie-breaker, addressing the diluted-signal
finding in `SCAL-V1.1` §1. Add an independent, optional, stateless post-hoc
`symlog` reward-compression stage. Per `ADR-057`, the new mode becomes the
default scalarization mode; compression defaults to off.

In scope:

- `bounded_priority_weighted_rank` formula, per-mode frozen `priority_base`
  validation, and its dominance proof's algebraic-guard regression test;
- `reward_compression.mode` (`none`|`symlog`) config field and transform;
- `raw_scalar_reward`/`scalar_reward` diagnostic split;
- checkpoint/resume compatibility identity gains `reward_compression_mode`;
- default configuration flip (`bounded_priority_weighted_rank`,
  `priority_base=3.0`, `reward_compression.mode: none`).

Out of scope (per `SCAL-V1.1` §2.2/§2.3 and `ADR-057`):

- changing the rule hierarchy, macro-margin contract, or any rulebook cost;
- a compression scale parameter other than the parameter-free `τ=1` form;
- full training ablation of the two modes with/without compression;
- retroactive algebraic-guard coverage for the three `SCAL-V1.0` modes
  (`DEC-SCAL11-004`=`B`).

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-SCAL11-001` | Add `bounded_priority_weighted_rank` as a fourth mode; it is the default per `DEC-SCAL11-001`=`A`. | §6 |
| `REQ-SCAL11-002` | Define the priority-weighted rank formula (`I_k'`, per-margin weighted terms, unit-weight `m_4`, `priority_base=3.0`). | §6, §7.6 |
| `REQ-SCAL11-003` | Freeze `priority_base=3.0` for this mode; validator selects the required frozen value per mode. | §6 |
| `REQ-SCAL11-004` | Optional, stateless `reward_compression.mode` (`none`|`symlog`); part of resume-compatibility identity. | §6 |
| `REQ-SCAL11-005` | Compression preserves per-transition pairwise ordering (monotonicity of `h`). | §6 |
| `REQ-SCAL11-006` | Compression logged explicitly; never silently pooled with hierarchy-only results. | §6 |

## 4. Current Repository Analysis

| Status | Verified fact and path | Consequence |
|---|---|---|
| `VERIFIED` | `src/thesis_rl/reward/scalarization.py` implements `SCAL-V1.0`'s three modes; `ScalarizationConfig.__post_init__` only checks `priority_base > 1.0` generically, not a per-mode frozen value. | New per-mode validation (`REQ-SCAL11-003`) is a genuine behavior addition, not already present. |
| `VERIFIED` | `ScalarizationConfig.from_mapping` flattens nested `sigmoid`/`legacy` mappings from YAML into flat fields; `conf/scalarization/default.yaml` uses this nested style. | `reward_compression: {mode: ...}` must be flattened the same way for config-file compatibility. |
| `VERIFIED` | `ScalarizationResult` has no raw-vs-delivered reward split; `reward` is both the internal and the `scalar_reward` alias. | Needs a new `raw_reward` field plus a `raw_scalar_reward` property, mirroring the existing `scalar_reward` alias pattern. |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/wrapper.py:239-246` sets `info_dict["scalar_reward"]`/`info_dict["scalarization"]` from `scalarization_result`; `to_dict()` is already logged per transition via `_append_diagnostics`. | Adding `raw_reward`/`raw_scalar_reward` to `ScalarizationResult.to_dict()` satisfies most of §5's diagnostic requirement automatically; one explicit top-level `info_dict["raw_scalar_reward"]` line completes it per §10.4's named field. |
| `VERIFIED` | `src/thesis_rl/contracts/reward_semantics.py:build_reward_semantics_identity` builds an exact-match compatibility dict from `config["scalarization"]`, consumed by `assert_reward_semantics_compatible`'s exact dict comparison (fail-closed). | Adding `reward_compression_mode` to this identity dict is sufficient for `AC-SCAL11-008`; no change needed to `contracts/checkpoint_manifest.py`. |
| `VERIFIED` | `src/thesis_rl/contracts/checkpoint_manifest.py`'s `CHECKPOINT_MANIFEST_SHAPE_FIELDS` (used by the only production-wired manifest check, `assert_checkpoint_manifest_compatible_if_present`) deliberately excludes all `scalarization_*`/`rulebook_*` fields, per its own comment: "already exclusively covered by `assert_reward_semantics_compatible`". | Confirms `checkpoint_manifest.py` is out of scope for `AC-SCAL11-008`; touching it would be unrelated scope creep. |
| `VERIFIED` | `conf/scalarization/default.yaml` sets `mode: bounded_satisfaction_rank`, `priority_base: 2.01`, no `reward_compression` key. | Must be updated to the new default per `ADR-057`. |
| `VERIFIED` | `tests/test_scalarization.py` calls bare `ScalarizationConfig()` expecting `bounded_satisfaction_rank` behavior at several sites. | These must be updated to pass `mode="bounded_satisfaction_rank"` explicitly, since the dataclass default changes; this preserves each test's original assertion about the unchanged `SCAL-V1.0` formula while decoupling it from the (now-changed) implicit default, per the approved `DEC-SCAL11-001`. |
| `VERIFIED` | No production call site constructs `ScalarizationConfig()` with no arguments; `src/thesis_rl/runtime/wiring/builders.py:373-381` always calls `ScalarizationConfig.from_mapping(...)` against the Hydra-resolved `scalarization` config block. | The dataclass-default flip only affects test ergonomics and `from_mapping(None)`/`from_mapping({})`, not any hidden production path. |

## 5. Assumptions And Invariants

- `priority_base` equality is checked as exact float equality against the
  mode's required frozen constant (`2.01` or `3.0`), matching how these
  constants already appear as Python float literals in both the
  specification and `conf/scalarization/default.yaml` — no floating-point
  tolerance is needed for a literal-to-literal comparison.
- The compression transform `h` is a pure function of one already-computed
  scalar and the frozen `reward_compression_mode` string; it must not read
  any other state (`SCAL-V1.0` §8.2, `SCAL-V1.1` §8).
- `canonical_margins`, `priority_contributions`, and `satisfaction_pattern`
  keep their `SCAL-V1.0` meaning for all four modes; only `continuous_tie_breaker`'s
  numeric meaning differs for `bounded_priority_weighted_rank` (it holds the
  unit-weight `m_4` contribution, not the shared `T(m)/4` tie-breaker) —
  this is disclosed in the field's per-mode semantics, not a schema change.

## 6. Decisions And Approval Gates

All decisions carried by this document were resolved by `ADR-057` before
implementation began; none remain open.

| ID | Category | Issue | Resolution | Status |
|---|---|---|---|---|
| `DEC-SCAL11-001` | Specification clarification | New default mode? | `A`: `bounded_priority_weighted_rank` | Resolved, `ADR-057` |
| `DEC-SCAL11-002` | Specification clarification | `symlog` default? | `B`: off by default | Resolved, `ADR-057` |
| `DEC-SCAL11-003` | Blocking technical issue | `wrongway` cost/status tolerance defect | Closed independently, `rulebook_v4.12`/`ADR-056` | Resolved |
| `DEC-SCAL11-004` | Specification clarification | Retroactive algebraic guard? | `B`: out of scope | Resolved, `ADR-057` |

## 7. Proposed Design

- `src/thesis_rl/reward/scalarization.py`:
  - `SCALARIZATION_MODES` gains `"bounded_priority_weighted_rank"`.
  - `_REQUIRED_PRIORITY_BASE_BY_MODE` maps each mode to its frozen
    `priority_base` (`2.01` for the three `SCAL-V1.0` modes, `3.0` for the
    new one); `__post_init__` enforces exact equality.
  - `ScalarizationConfig` gains `reward_compression_mode: str = "none"`,
    validated against `{"none", "symlog"}`; dataclass defaults for `mode`
    and `priority_base` flip to `"bounded_priority_weighted_rank"`/`3.0`.
  - `from_mapping` flattens a nested `reward_compression: {mode: ...}`
    mapping the same way it already flattens `sigmoid`/`legacy`.
  - `scalarize_rulebook_margins` adds a `bounded_priority_weighted_rank`
    branch: `I_k' = is_satisfied - 1` (identical indicator computation to
    `bounded_satisfaction_rank`, reused), each priority term becomes
    `base_k * (I_k' + canonical_margin_k)`, and `continuous` becomes
    `canonical_margins[3]` (unit weight) instead of the four-way average.
    The raw sum is computed first (`raw_reward`); `reward_compression_mode`
    then optionally applies `h(r) = sign(r) * log1p(|r|)` to produce the
    delivered `reward`. Both are finite-checked independently.
  - `ScalarizationResult` gains `raw_reward: float` and
    `reward_compression_mode: str`; a `raw_scalar_reward` property mirrors
    the existing `scalar_reward` property; `to_dict()` exposes both.
- `src/thesis_rl/rulebook/v2/wrapper.py`: one added line,
  `info_dict["raw_scalar_reward"] = scalarization_result.raw_reward`,
  alongside the existing `info_dict["scalar_reward"]` assignment.
- `src/thesis_rl/contracts/reward_semantics.py`:
  `build_reward_semantics_identity` adds
  `"reward_compression_mode": scalarization.get("reward_compression", {}).get("mode")`
  (defaulting through the existing `_mapping` helper) to the `scalarization`
  sub-dict, so it participates in the existing exact-match resume check.
- `conf/scalarization/default.yaml`: `mode: bounded_priority_weighted_rank`,
  `priority_base: 3.0`, `specification_id: SCAL-V1.1`, `version: "1.1"`,
  new `reward_compression: {mode: none}` block.
- No changes to `contracts/checkpoint_manifest.py` or
  `runtime/wiring/checkpoint_identity.py` (out of scope; see §4's finding on
  `CHECKPOINT_MANIFEST_SHAPE_FIELDS`).

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SCAL11-001` | `AC-SCAL11-001` | `scalarization.py` (`SCALARIZATION_MODES`, dataclass defaults) | `test_scalarization.py::test_default_mode_is_priority_weighted_rank_with_base_three` | Implemented |
| `REQ-SCAL11-002` | `AC-SCAL11-002`, `003`, `005` | `scalarization.py` (`bounded_priority_weighted_rank` branch) | `test_scalarization.py::test_priority_weighted_rank_reference_values`, `::test_priority_weighted_rank_dominance_is_exact_for_all_satisfaction_patterns`, `::test_priority_weighted_rank_is_linear_with_no_saturation` | Implemented |
| `REQ-SCAL11-003` | `AC-SCAL11-001`, `AC-SCAL11-004` | `scalarization.py` (`_REQUIRED_PRIORITY_BASE_BY_MODE`) | `test_scalarization.py::test_priority_base_must_match_mode`, `::test_priority_weighted_rank_dominance_algebraic_guard` | Implemented |
| `REQ-SCAL11-004` | `AC-SCAL11-007`, `AC-SCAL11-008` | `scalarization.py` (`reward_compression_mode`), `reward_semantics.py` | `test_scalarization.py::test_symlog_compression_reference_values`, `test_scalarization_wiring.py::test_reward_compression_mode_is_part_of_resume_identity` | Implemented |
| `REQ-SCAL11-005` | `AC-SCAL11-006` | `scalarization.py` (`_symlog` monotonic transform) | `test_scalarization.py::test_symlog_compression_preserves_pairwise_order` | Implemented |
| `REQ-SCAL11-006` | `AC-SCAL11-007` | `ScalarizationResult.to_dict` (`reward_compression_mode` disclosed) | `test_scalarization.py::test_symlog_compression_reference_values` | Implemented |

## 9. Test Strategy

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-SCAL11-001` | Unit | Default mode/base | `ScalarizationConfig()` | mode=`bounded_priority_weighted_rank`, base=`3.0` | `REQ-SCAL11-001` |
| `TEST-SCAL11-002` | Unit | Reference values | `AC-SCAL11-002`'s 7 vectors | exact match within `1e-6` | `REQ-SCAL11-002` |
| `TEST-SCAL11-003` | Boundary | Exhaustive pattern dominance | all `2^3` patterns, extreme margins | first-differing-rule dominance holds | `REQ-SCAL11-002` |
| `TEST-SCAL11-004` | Regression | Algebraic guard | symbolic `a'>2`, `a'^2>2a'+2`, `a'^3>2a'^2+2a'+2` at `a'=3` | all hold | `REQ-SCAL11-002` |
| `TEST-SCAL11-005` | Invalid | Mode/base mismatch | `priority_base=2.01` with the new mode, and vice versa | `ScalarizationConfigurationError` | `REQ-SCAL11-003` |
| `TEST-SCAL11-006` | Unit | Compression reference values | known raw reward, `symlog` | `h(r)` within `1e-6`; `none` leaves value unchanged | `REQ-SCAL11-004` |
| `TEST-SCAL11-007` | Unit | Compression monotonicity | pair with `r(A) > r(B)` | `h(r(A)) > h(r(B))` | `REQ-SCAL11-005` |
| `TEST-SCAL11-008` | Integration | Resume identity | two configs differing only in `reward_compression.mode` | different `build_reward_semantics_identity` output | `REQ-SCAL11-004` |

Commands: `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scalarization.py tests/test_scalarization_wiring.py`;
scoped `ruff check`/`ruff format --check` on touched files (`make lint`
target does not apply cleanly repository-wide per `AGENTS.md`).

## 10. Milestones

### M1 — Scalarization core formula and configuration — `IMPLEMENTED`

- Files: `src/thesis_rl/reward/scalarization.py`
- Tasks: new mode branch, per-mode `priority_base` validation, dataclass
  default flip, `reward_compression_mode` field/transform, `raw_reward`
  split on `ScalarizationResult`.
- Tests: `TEST-SCAL11-001` through `007`.

### M2 — Diagnostics and resume-compatibility wiring — `IMPLEMENTED`

- Files: `src/thesis_rl/rulebook/v2/wrapper.py`, `src/thesis_rl/contracts/reward_semantics.py`
- Tasks: top-level `raw_scalar_reward` diagnostic; `reward_compression_mode`
  in the resume-compatibility identity dict.
- Tests: `TEST-SCAL11-008`; existing `tests/test_rulebook_v2_wrapper.py`
  and `tests/test_scalarization_wiring.py` regressions must keep passing.

### M3 — Default configuration flip — `IMPLEMENTED`

- Files: `conf/scalarization/default.yaml`
- Tasks: new default mode/base/specification identity, explicit
  `reward_compression.mode: none`.
- Tests: `tests/test_hydra_preset_test_configs.py` (existing Hydra
  config-loading regressions) must keep passing.

### M4 — Regression suite and reconciliation — `IMPLEMENTED`

- Update pre-existing `tests/test_scalarization.py` bare-`ScalarizationConfig()`
  call sites to explicit `mode="bounded_satisfaction_rank"` where they test
  that mode's unchanged formula.
- Run focused pytest, scoped ruff lint/format-check, and record results in
  §14.

## 11. Progress And Findings Log

- 2026-08-06: Plan drafted and implementation started immediately after
  `ADR-057` approval, per the user's instruction to proceed automatically.
  Confirmed via code reading that `contracts/checkpoint_manifest.py` is out
  of scope for `AC-SCAL11-008` (see §4), narrowing M2 to one file.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/reward/scalarization.py` | Modified | New mode, per-mode base validation, compression |
| `src/thesis_rl/rulebook/v2/wrapper.py` | Modified | `raw_scalar_reward` diagnostic |
| `src/thesis_rl/contracts/reward_semantics.py` | Modified | `reward_compression_mode` in resume identity |
| `conf/scalarization/default.yaml` | Modified | New default mode/base/compression |
| `tests/test_scalarization.py` | Modified | New-mode/compression tests; explicit mode on pre-existing default-dependent tests |
| `tests/test_scalarization_wiring.py` | Modified | Resume-identity coverage for compression mode |
| `docs/specifications/rulebook_scalarization_v1.1_specification.md` | Created (moved from `incoming/`) | Approved specification |
| `docs/decisions/ADR-057-scalarization-v1.1-defaults.md` | Created | Decision record |
| `docs/project_index.md` | Modified | Authority index entries |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scalarization.py tests/test_scalarization_wiring.py tests/test_rulebook_v2_wrapper.py` | `PASS` | 2026-08-06 | 42 passed |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/` | `PASS` (with documented pre-existing gap) | 2026-08-06 | 1297 passed, 47 failed; all 47 verified pre-existing and unrelated (route-coordinate-mission migration in progress on this branch — `ValueError: Progress requires pre/post mission context` in `components/progress.py:50`, plus unrelated `eval_artifacts`/`async_evaluation`/`hydra_preset`/`run_metadata` failures). Confirmed by direct re-run of a sample failure (`test_rulebook_synthetic_scenarios.py::...[wrong_way-...]`) showing the identical pre-existing stack trace. No scalarization-related test is among the 47. |
| Scoped `ruff check` on touched files | `PASS` | 2026-08-06 | `scalarization.py`, `wrapper.py`, `reward_semantics.py`, `test_scalarization.py`, `test_scalarization_wiring.py`, `test_rulebook_v2_wrapper.py` — all checks passed |
| Scoped `ruff format --check` on touched files | `PASS` | 2026-08-06 | One file (`test_scalarization_wiring.py`) needed `ruff format`; applied, then re-verified clean; re-ran the focused pytest subset after reformatting to confirm no behavioral change |
| `make smoke` | `NOT_RUN` | | Not executed in this session; the scalarization change is config/formula-level, exercised end-to-end by the full suite above and by `test_wrapper_uses_scalarizer_after_complete_rulebook_evaluation`; a full training smoke run is recommended before treating this ExecPlan as `VERIFIED` |

## 15. Final Reconciliation

| Requirement | Acceptance criteria | Status |
|---|---|---|
| `REQ-SCAL11-001` | `AC-SCAL11-001` | `IMPLEMENTED` |
| `REQ-SCAL11-002` | `AC-SCAL11-002`, `003`, `005` | `IMPLEMENTED` |
| `REQ-SCAL11-003` | `AC-SCAL11-001`, `AC-SCAL11-004` | `IMPLEMENTED` |
| `REQ-SCAL11-004` | `AC-SCAL11-007`, `AC-SCAL11-008` | `IMPLEMENTED` |
| `REQ-SCAL11-005` | `AC-SCAL11-006` | `IMPLEMENTED` |
| `REQ-SCAL11-006` | `AC-SCAL11-007` | `IMPLEMENTED` |

All in-scope requirements are implemented and covered by passing focused
tests; the full repository suite shows no new failures relative to this
branch's pre-existing, unrelated baseline gap. `make smoke` was not run in
this session — recorded as a known residual risk, not a passed check; the
plan status is `IMPLEMENTED` rather than `VERIFIED` pending that run. No
deviations from the approved specification were introduced during
implementation.
