# ExecPlan — Geometry Abort And Loud Quarantine (`GEOM-ABORT`)

## 1. Metadata

| Field | Value |
|---|---|
| Feature | A third outcome for geometry failures in the live rulebook: neither fatal to the run nor silently absorbed, but quarantined with forensic evidence and a hard ceiling |
| Plan ID | `GEOM-ABORT` |
| Authoritative specifications | `RSA-V1` (`docs/implementation/runtime_scenario_data_abort_v1_exec_plan.md`, `IMPLEMENTED`) and ADR-024, which this plan **deviates from** by user approval; `RULEBOOK-V5.1` |
| Status | `IMPLEMENTED` for the evaluation path — both decisions approved (user, 2026-09-06); the training-loop path and resume persistence remain, see §9 |
| Created | 2026-09-06 |
| Branch | `feat/geometry-abort-quarantine`, stacked on `fix/constrained-decomposition-coverage` |
| Related | `open_items` `C9` (the defect that motivated it), `C7` (the vectorized-path limitation it inherits) |

**This plan must not be merged while the `AB-LEARN` seed-0 runs are in flight.** Unlike
`C9`, which was provably inert on every input the previous code could handle, this
change alters what happens on a geometry failure and is therefore observable behavior.
The repository working tree is bind-mounted live into the running containers
(`compose.yaml:21`) and evaluation workers re-import on spawn, so merging mid-run would
contaminate the experiment.

## 2. Objective And Scope

**Observable capability.** When the live rulebook fails on a geometry operation for an
enumerated, recognized reason, the episode ends at its last valid transition and the run
continues — while the failure is recorded loudly enough to be investigated and counted
against a ceiling that fails the run if the condition is systemic.

**Why it is needed.** `open_items` `C9` killed a 100 000-step screening run after seven
hours on a shortfall of 0.13 mm² over a 261 m² polygon. The existing quarantine
(`RSA-V1`) absorbed 5–11 scenarios per evaluation throughout the same runs, but by
`REQ-RSA-001` it recovers *only* the six enumerated signal/route data reasons and
`RuntimeScenarioNotEvaluableError` is documented as deliberately narrow: it "must not
wrap numerical, programming, serialization, worker, or learner failures". A geometry
`ValueError` is therefore fatal by design.

**That design was not wrong.** Had the quarantine swallowed `C9`, it would have hidden a
real dimensional defect behind a slightly higher discard rate that nobody inspects. The
cost of the crash bought the diagnosis. The purpose of this plan is **not** to make
geometry failures cheap; it is to stop them costing a multi-day run while keeping them
impossible to ignore.

**In scope.** A typed geometry error distinct from the data-abort type; its worker
transport; separate counters and forensic artifacts; a ceiling that fails the run;
enumerated raise sites in the geometry layer.

**Out of scope.** Widening `RuntimeScenarioNotEvaluableError` itself; recovery from
untyped or programming errors; the single-environment path (`C7`); dataset mutation.

## 3. Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-GA-001` | Geometry aborts use a type **distinct from** `RuntimeScenarioNotEvaluableError`, so `RSA-V1`'s narrow contract is unchanged and the two can never be confused in code or artifacts | Deviation from `REQ-RSA-001`, approved by user 2026-09-06 |
| `REQ-GA-002` | Only enumerated geometry reasons are recoverable; every other geometry failure stays fatal | Mirrors `REQ-RSA-001`'s discipline |
| `REQ-GA-003` | Every recoverable raise carries the **measured magnitude** of the violation in its diagnostics | `open_items` `C6`, `C9` |
| `REQ-GA-004` | Geometry-abort counters are reported separately and are never folded into `data_abort_*` | `REQ-RSA-007` |
| `REQ-GA-005` | Each occurrence writes a forensic record carrying enough geometry to reproduce it as a test fixture, and is logged at `ERROR` | `REQ-RSA-008` |
| `REQ-GA-006` | A configurable ceiling fails the run when geometry aborts become systemic | New; the safeguard that makes `REQ-GA-001` acceptable |
| `REQ-GA-007` | Episode truncation reuses the existing `RSA-V1` boundary machinery unchanged | `REQ-RSA-002`, `REQ-RSA-003` |

## 4. Design

### 4.1 A separate type, not a widened one

```
RuntimeGeometryNotEvaluableError(RuntimeError)
    reason: RuntimeGeometryNotEvaluableReason   # closed enum
    diagnostics: Mapping[str, Any]              # must carry the magnitude
    geometry_wkt: str | None                    # the offending polygon, for a fixture
```

It deliberately does **not** subclass `RuntimeScenarioNotEvaluableError`.
`RSA-V1`'s type keeps its documented narrowness; a reader of either type still knows
exactly what it means. The worker recognises both and replies with **different**
markers.

Initial closed reason set, one per condition actually observed or reachable today:

| reason | raised by | magnitude carried |
|---|---|---|
| `DECOMPOSITION_COVERAGE_SHORTFALL` | `_constrained_components_after_ear_exhaustion` | residual m², allowance m², ratio, polygon area, vertex and hole counts |
| `DECOMPOSITION_NO_VISIBLE_BRIDGE` | `_bridge_hole` | hole count, vertex counts |
| `DEGENERATE_RING` | `_normalized_ring` | signed area, vertex count |

### 4.2 Why a ceiling, and what it protects

Quarantine without a ceiling is silent absorption with extra steps: a systemic geometry
defect would present as a slowly rising discard rate on a metric nobody reads, which is
exactly the failure mode `C9`'s crash avoided. The ceiling restores the property that a
real defect eventually stops the run — just not on its first occurrence, and with
evidence in hand rather than a bare traceback.

Two counters, both run-local: total geometry aborts, and consecutive geometry aborts
within one evaluation batch. Exceeding either fails the run with a message naming the
dominant reason and the recorded artifact path.

### 4.3 Scenario quarantine is deliberately *not* automatic

`RSA-V1` quarantines the scenario UID on a data abort (`REQ-RSA-005`), which is right
for a data defect: the condition is a property of the record and will recur. A geometry
failure is **not** a property of the record alone — it depends on where the actors are,
which depends on the policy. Quarantining on first occurrence would silently shrink the
training pool because of one unlucky step, and the pool is a scientific quantity.

Proposed: truncate and record on every occurrence; quarantine the scenario UID only
after `N` occurrences on the *same* UID. See `DEC-GA-002`.

## 5. Decisions Requiring Approval

| ID | Issue | Alternatives | Recommendation | Consequence |
|---|---|---|---|---|
| `DEC-GA-001` **Approved** | What ceiling fails the run? | Absolute count per run / rate over evaluated episodes / consecutive-within-batch / a combination | **Both**: fail at `>= 3` consecutive aborts inside one evaluation batch, or when the run-total rate exceeds `1 %` of episodes attempted. The first catches a systemic break immediately; the second catches slow bleed | Too tight wastes runs on a benign rarity; too loose reinstates silent absorption. The observed `C9` rate was **one occurrence in ~8 evaluation batches**, so both defaults leave two orders of margin |
| `DEC-GA-002` **Approved** | Does a geometry abort quarantine the scenario? | Never / after N on the same UID / always, as `RSA-V1` does | **After `N = 3` on the same UID**, so a genuinely broken record leaves the pool while an unlucky step does not | Always-quarantine shrinks the training pool on policy-dependent events and biases the source mix; never-quarantine lets one broken record burn episodes for the whole run |

Both defaults are engineering judgement anchored on a single observed occurrence, not
calibrated constants. They are declared here so the choice is visible rather than buried.

## 6. Test Matrix

| ID | Level | Behavior | Expected | Requirement |
|---|---|---|---|---|
| `TEST-GA-001` | Unit | The four enumerated conditions raise the geometry type carrying a magnitude | Typed error; diagnostics non-empty and numeric | `REQ-GA-002`, `REQ-GA-003` |
| `TEST-GA-002` | Unit | An unenumerated geometry failure stays fatal | Plain exception propagates | `REQ-GA-002` |
| `TEST-GA-003` | Unit | The geometry type is **not** an instance of `RuntimeScenarioNotEvaluableError`, and vice versa | Both assertions hold | `REQ-GA-001` |
| `TEST-GA-004` | Integration | A worker hitting an enumerated geometry failure replies with the geometry marker, stays alive, and the slot resets | Worker alive; episode truncated at last valid transition | `REQ-GA-007` |
| `TEST-GA-005` | Integration | Counters land in `geometry_abort_*` and leave `data_abort_*` untouched | Both counter families correct and disjoint | `REQ-GA-004` |
| `TEST-GA-006` | Integration | The forensic record is written and carries geometry sufficient to rebuild the polygon | Record parses; WKT round-trips to a valid polygon | `REQ-GA-005` |
| `TEST-GA-007` | Integration | Exceeding either ceiling fails the run with the reason and artifact path in the message | Run terminates; message names both | `REQ-GA-006` |
| `TEST-GA-008` | Regression | `RSA-V1`'s existing data-abort behavior is unchanged | The existing data-abort suite passes untouched | `REQ-GA-001` |

## 7. Deviations

| ID | Contract | Change | Approval |
|---|---|---|---|
| `DEV-GA-001` | `RSA-V1` `REQ-RSA-001`: "Recover only the enumerated typed runtime scenario errors; all others are fatal", with scope excluding "numerical" errors | Adds a second, disjoint enumerated recovery class for geometry failures, bounded by a ceiling that restores fatality when the condition is systemic | User, 2026-09-06 |

## 8. Status

Design recorded; implementation not started pending `DEC-GA-001` and `DEC-GA-002`.
Merge is blocked on the `AB-LEARN` seed-0 runs completing regardless of those decisions.

## 9. Implementation Status And Findings

**2026-09-06.** Both decisions approved as recommended: ceiling at three consecutive
aborts within one evaluation batch **or** a run rate above 1 %, and scenario quarantine
only after three occurrences on the same UID.

**Finding, material: `DECOMPOSITION_OVERSHOOT` was dropped from the recoverable set.**
The drafted design listed it as a fourth reason. It is unreachable by construction: every
retained triangle has already passed `polygon.covers`, so the union of retained triangles
cannot escape the polygon. If it ever fires, a GEOS invariant is broken — a contradiction
rather than a tolerance question — so it stays a fatal `ValueError` and its reason code was
removed from the closed enum. The `C9` regression test that appeared to cover it in fact
exercised the *no-triangles* path; it now says so, and asserts the failure is **not** a
geometry abort.

**Enforced structurally rather than by convention.** `RuntimeGeometryNotEvaluableError`
refuses to be constructed with empty diagnostics, so `REQ-GA-003` cannot be forgotten at a
new raise site. The alternative — a convention — is what left `C9`'s original message
without a number.

### Implemented

| Piece | Location |
|---|---|
| Typed error and closed three-reason enum, disjoint from `RSA-V1`'s type; refuses construction without diagnostics | `rulebook/v2/errors.py` |
| Three enumerated raise sites, each carrying its magnitude | `rulebook/v2/geometry/continuous_sat.py` |
| Wrapper annotation: `scenario_uid`, `environment_step`, `final_observation` | `rulebook/v2/wrapper.py` |
| Worker transport under its own marker, worker kept alive, observation forwarded | `runtime/execution/deterministic_subproc_vec_env.py` |
| `GeometryAbortLedger`: counters, per-UID repeats, both ceilings; `append_geometry_abort_record` | `runtime/data_abort.py` |
| **Evaluation path**: ledger, `ERROR` log, forensic record with WKT, deferred quarantine, ceiling enforcement, `geometry_abort_coverage` reported separately from `data_abort_coverage` | `agent/agent.py::_evaluate_parallel` |
| **Training path**: same, sharing the `RSA-V1` boundary machinery through a combined `aborted_indices` set so `valid_mask`, the previous-transition close and the slot reset are written once for both kinds | `agent/agent.py::train_vectorized` |
| Forensic log path wired to `logs/runtime_geometry_abort.jsonl`, separate from the data-abort log | `runtime/loops/train_loop.py` |

**Ceiling ordering, deliberate.** In the training loop the ceiling is charged
*before* any episode that finished cleanly in the same iteration can reset the
consecutive counter, so several slots failing at once registers as one systemic
burst rather than being cancelled by a neighbouring success. Within an evaluation
batch the counter resets per batch, which is what `begin_batch` is for.

Tests: 14 in `tests/test_geometry_abort_quarantine.py` (`TEST-GA-001` to
`TEST-GA-007`, including a real subprocess vector environment for the transport and
the complement that an unenumerated geometry failure still kills the worker), plus 3
in `tests/test_train_vectorized_geometry_abort.py` covering the training loop: its own
forensic file with the data-abort log untouched, no quarantine on first sight, a burst
across slots tripping the ceiling, and a repeated UID eventually being quarantined.

### Remaining

1. **The two thresholds are module constants, not Hydra configuration.**
   `GEOMETRY_ABORT_MAX_CONSECUTIVE_IN_BATCH`, `GEOMETRY_ABORT_MAX_RUN_RATE` and
   `GEOMETRY_ABORT_QUARANTINE_AFTER_REPEATS` in `runtime/data_abort.py` carry the
   approved defaults and `GeometryAbortLedger` already accepts overrides, but nothing
   reads them from `conf/`. A run cannot currently loosen or tighten the ceiling
   without editing code.
2. **Resume persistence.** `RSA-V1` persists its quarantine alongside the checkpoint
   (`REQ-RSA-005`); `GeometryAbortLedger` serializes (`to_dict`/`from_dict`) but is not
   written to or restored from the checkpoint sidecar, so a resumed run restarts its
   counters at zero and forgets which UIDs had repeated.
3. **`C7` is inherited unchanged.** Like the data abort, this works only on the
   vectorized path; with `env.vectorized.enabled=false` a geometry abort is still fatal,
   because the conversion lives inside the worker boundary.

None of the three blocks the merge decision, which is gated on the `AB-LEARN` seed-0
runs finishing.
