# ExecPlan: R1 Injury-Risk Collision Cost (Rulebook v4.9)

## 1. Metadata

- Feature: `r1_injury_risk_collision_cost`
- Plan ID: `EP-R1-INJURY-RISK`
- Authoritative specification: `docs/specifications/rulebook_v4.9_specification.md`,
  ID `rulebook-v2-r1-injury-risk`, version `4.9`, status `APPROVED`,
  authoritative for the amended v4.7 §5.4/§5.6/§5.8 subset
- Base specification (unchanged parts): `docs/specifications/rulebook_v4.7_specification.md`,
  version `4.7-final-implementation-complete`
- Related ADR: `docs/decisions/ADR-027-r1-injury-risk-collision-cost.md` (`APPROVED`)
- Status: `IMPLEMENTED`
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`

## 2. Objective And Scope

### Objective

Make the `R1` collision cost independent of scenario configuration and give
it an explicit physical meaning: the probability that the impact produces
an at-least-serious injury (`MAIS3+F`), read from the peer-reviewed
injury-risk curves of Lubbe et al. (2022).

Success is recognized when, at equal normal closing speed and actor class,
`q_collision` is numerically identical in every scenario, and when the
v4.7 severity inversion (residential 4 m/s scoring above highway 15 m/s) is
reproduced as a failing-under-v4.7 regression test that passes under v4.9.

### In Scope

- `q_collision,i` in `src/thesis_rl/rulebook/v2/components/collision.py`.
- New injury-risk model module holding the frozen published coefficients.
- `R1` diagnostics fields (specification §6).
- Tests for the new contract plus regression for the demonstrated defect.

### Out Of Scope

- Collision onset detection, deduplication, contact memory, normal
  definition, pre-state velocity extraction, `eps_col`, error taxonomy of
  v4.7 §5.3/§5.5 — all preserved unchanged.
- `c_1(t) = max_i q_collision,i` aggregation — unchanged.
- `R2`/`R3`/`R4`, rule hierarchy, scalarization, ACL, observation.
- Scenario eligibility rules based on configured speed caps
  (`DEC-R1-06`: validation deliberately retained).
- `MOTORCYCLIST` actor class (`DEC-R1-07`: deferred).

### Compatibility constraints

Reward semantics change. Runs trained under v4.7 `R1` are not comparable
with v4.9 runs and must not be pooled (specification §9.3). The
scalarization contract, reward-vector shape `(4,)`, margin ranges
`m_1 in [-1,0]`, and the `RulebookResult` interface are all unchanged, so no
checkpoint schema, replay schema, or manifest field changes.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-R1-01` | Cost does not depend on scenario configuration | v4.9 §1.1, §4.1 |
| `REQ-R1-02` | Cost is the `MAIS3+F` probability of the source curve | v4.9 §4.1 |
| `REQ-R1-03` | Per-class vulnerability ordering follows the source | v4.9 §5 |
| `REQ-R1-04` | Coefficients reproduce the published anchors | v4.9 §4.3 |
| `REQ-R1-05` | `STATIC_COLLIDABLE` uses the car-driver curve | v4.9 §4.4 |
| `REQ-R1-06` | No fallback for unmapped actor classes | v4.9 §8 |
| `REQ-R1-07` | v4.7 §5.1-5.3/§5.5 invariants preserved | v4.9 §2 Out Of Scope |

## 4. Current Repository Analysis

All statements below are `VERIFIED` by reading the code at plan creation.

| Item | Path | Note |
|---|---|---|
| `R1` evaluator | `src/thesis_rl/rulebook/v2/components/collision.py` | `evaluate_collision_impact`; the cost line was `bounded_ratio = min(raw_speed, cap)/cap; cost = max(COLLISION_FLOOR, bounded_ratio**2)` |
| Cap selection | same file, `cap = ego_cap (+ actor_cap if VEHICLE)` | the scenario-dependent normalizer being removed |
| Actor taxonomy | `src/thesis_rl/rulebook/v2/types.py` | `ActorClass`: `VEHICLE`, `PEDESTRIAN`, `CYCLIST`, `STATIC_COLLIDABLE`, `INFRASTRUCTURE_NON_COLLIDABLE` |
| Macro aggregation | `src/thesis_rl/rulebook/v2/aggregation.py` | `aggregate_max_component` validates `cost in [0, 1+1e-8]`; unchanged and still satisfied |
| Existing tests | `tests/test_rulebook_v2_collision.py` | asserts `0.0625` (v4.7 value) in two tests |
| Contract tests | `tests/test_rulebook_v2_f6_contracts.py` | calls `evaluate_collision_impact`; asserts structure, not the numeric cost |
| Stable sigmoid precedent | `src/thesis_rl/reward/scalarization.py::_stable_sigmoid` | same overflow-safe pattern reused |
| Local source copy | `docs/papers/rulebook/Lubbe et Al. - 2022 - Safe speeds.pdf` | coefficients transcribed from Tables 2, 3, 5 |

Behavior to preserve: onset dedup by actor id, `previous_contact_ids`
memory delta emitted on every path (including the no-onset and
no-pre-state paths), `NOT_APPLICABLE` status when no evaluable onset,
fatal error on coincident pre-state centers, fatal error on invalid ego
cap, `max` over actors with `(cost, actor_id)` tie-break.

## 5. Assumptions And Invariants

- `u_i` is in `m/s`, non-negative, pre-state, normal component
  (v4.7 §5.3). The source's coefficients are per `km/h`, so the model
  multiplies by `3.6` internally; coefficients are stored in published
  units so they can be diffed against the paper. `VERIFIED` against the
  published 10%-risk anchors (see `AC-R1-05`).
- Reference age `65` years, dimensionless constant, identical for all
  classes (`DEC-R1-03`).
- Output domain `(0,1)`, strictly increasing in `u`; satisfies
  `aggregate_max_component`'s `[0, 1+1e-8]` precondition without clipping.
- `eps_col = 1e-6` retained but structurally inert, since
  `P_tau(0) >= 1.85e-3` for every mapped class (v4.9 §4.5).
- Stateless and deterministic: no seed, device, or episode dependence.

## 6. Decisions And Approval Gates

Specification-level decisions `DEC-R1-01`..`DEC-R1-07` are recorded in
v4.9 §12 and ADR-027. Implementation-level decisions:

| ID | Category | Issue | Alternatives | Decision | Status |
|---|---|---|---|---|---|
| `DEC-IMPL-001` | Mandatory-test change | `tests/test_rulebook_v2_collision.py` asserts the v4.7 value `0.0625` | (a) update expectations to the v4.9 values; (b) keep and mark xfail | (a) — the expectations encode the superseded formula; updating them is the direct consequence of the approved specification change, not adaptation of tests to code. The tests' structural assertions are strengthened, not weakened | `APPROVED` via ADR-027 |
| `DEC-IMPL-002` | Implementation detail | Where to put the risk model | (a) inline in `collision.py`; (b) separate module | (b) `injury_risk.py` — the model is a citable scientific constant set with its own tests, and keeping it separate makes the coefficient block diffable against the source | Decided internally |
| `DEC-IMPL-003` | Implementation detail | Retain `ego_configured_speed_cap_mps` parameter and its validation | (a) remove; (b) retain | (b) — `DEC-R1-06`; removing it would change scenario eligibility, which is out of scope. The parameter is validated and no longer used for cost; this is stated in the docstring so it does not read as dead code | Decided internally |

## 7. Proposed Design

New module `src/thesis_rl/rulebook/v2/components/injury_risk.py`:

- `InjuryRiskModel` frozen dataclass holding `(intercept, closing_speed_kmh,
  age_years)` exactly as published;
- `MAIS3F_MODEL_BY_ACTOR_CLASS` mapping per v4.9 §4.4;
- `injury_risk_cost(actor_class, normal_closing_speed_mps)` returning the
  logistic probability, raising `ValueError` for unmapped classes or
  non-finite input;
- `closing_speed_at_risk_kmh(...)` inverse, used only by the coefficient
  transcription test (`AC-R1-05`) and never on the runtime path.

`collision.py` changes: replace the two cost lines with a call to
`injury_risk_cost`, propagate `RulebookEvaluationError` for unmapped
classes, and update diagnostics per v4.9 §6.

Errors: unmapped actor class becomes a fatal `RulebookEvaluationError`
through the existing `_fail` helper; no fallback curve.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-R1-01` | `AC-R1-01`, `AC-R1-02` | `collision.py`, `injury_risk.py` | `tests/test_rulebook_v2_injury_risk.py::test_cost_is_independent_of_configured_speed_cap`, `::test_v47_severity_inversion_regression` | Done |
| `REQ-R1-02` | `AC-R1-04` | `injury_risk.py` | `::test_reference_cost_table_and_domain`, `::test_cost_is_strictly_increasing_in_closing_speed` | Done |
| `REQ-R1-03` | `AC-R1-03` | `injury_risk.py` | `::test_vulnerability_ordering_matches_source` | Done |
| `REQ-R1-04` | `AC-R1-05` | `injury_risk.py` | `::test_published_ten_percent_anchors_are_reproduced` | Done |
| `REQ-R1-05` | `AC-R1-06` | `injury_risk.py` | `::test_static_collidable_uses_car_driver_curve` | Done |
| `REQ-R1-06` | `AC-R1-07` | `collision.py`, `injury_risk.py` | `::test_unmapped_actor_class_is_fatal`, `::test_collision_rejects_unmapped_actor_class` | Done |
| `REQ-R1-07` | `AC-R1-08` | `collision.py` | `tests/test_rulebook_v2_collision.py` (whole module), `tests/test_rulebook_v2_f6_contracts.py` | Done |

## 9. Validation

Commands executed and results are recorded in §10. The mandatory matrix is:

| Category | Requirement | Covered by |
|---|---|---|
| nominal and boundary behavior | `REQUIRED` | `test_reference_cost_table_and_domain` |
| invalid inputs | `REQUIRED` | `test_unmapped_actor_class_is_fatal`, non-finite input test |
| numerical stability | `REQUIRED` | `test_cost_is_finite_at_extreme_closing_speed` |
| determinism | `REQUIRED` | pure function, no state; covered by fixed-value assertions |
| regression for the discovered defect | `REQUIRED` | `test_v47_severity_inversion_regression` |
| source fidelity | `REQUIRED` | `test_published_ten_percent_anchors_are_reproduced` |
| upstream/downstream integration | `REQUIRED` | full rulebook suite, `make rulebook-v2-check` |

## 10. Execution Record

### Commands executed 2026-07-26

| Command | Result |
|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_injury_risk.py` | `12 passed` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py` | `240 passed` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q` | `1008 passed in 91.27s` |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/rulebook/v2 tests/test_rulebook_v2_injury_risk.py tests/test_rulebook_v2_collision.py tests/test_rulebook_v2_f6_contracts.py` | `All checks passed!` |
| `docker compose run --rm dev uv run --no-sync ruff format` (new/materially changed files only) | applied, then re-verified clean |

`make rulebook-v2-check` was not invoked as a single target; its two
constituent commands (the `tests/test_rulebook_v2_*.py` suite and the scoped
`ruff check` over `src/thesis_rl/rulebook/v2`) were both run above with the
results shown. `make smoke` was **not** run: it requires a real training
environment and dataset mounts unavailable in this session. Residual risk is
low because the change is a pure function swap inside one component, with no
interface, configuration, or wiring change; the follow-up command is
`make smoke`.

### Findings during implementation

1. **`_fail` was annotated `-> None`** in `collision.py` while always raising
   (`rss.py` already used `NoReturn`). The new code path binds `risk` inside a
   `try/except` whose handler calls `_fail`, so the annotation had to be
   correct for the binding to be provably safe. Changed to `NoReturn`.
2. **The published anchor for the car-driver curve cannot be reproduced to
   1 km/h.** The computed value is 113.1 km/h against a published 112 km/h.
   The cause is coefficient rounding, not a transcription error: `per_kmh =
   0.041` carries only two significant digits, so the anchor is determined
   only to about ±2 km/h. `AC-R1-05` was therefore implemented as an interval
   test that propagates each coefficient's ±0.0005 rounding through the
   inverted curve and requires the published anchor to fall inside the
   resulting interval — a stronger check than an arbitrary tolerance. All
   three published anchors fall inside their intervals.
3. **`test_collision_saturates_when_pre_state_speed_exceeds_configured_cap`**
   tested a contract that v4.9 removes (exact saturation at the configured
   cap). It was renamed to
   `test_collision_cost_approaches_one_at_extreme_closing_speed` and now
   asserts the v4.9 contract (`0.999 < cost < 1.0`, never reaching 1), keeping
   the boundedness assertion it already had.
4. **`test_v47_severity_inversion_regression` also asserts the superseded
   values are not returned**, so a silent revert to the v4.7 formula fails the
   test rather than merely changing a number.

## 11. Milestones

- `M1` specification v4.9 + ADR-027 + this plan — done.
- `M2` `injury_risk.py` + its tests — done.
- `M3` `collision.py` wiring + diagnostics + updated existing tests — done.
- `M4` focused suite, rulebook suite, lint/format, index update — done.
