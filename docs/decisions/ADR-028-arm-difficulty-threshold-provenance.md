# ADR-028: Arm Difficulty Threshold Provenance And Status

- Status: `Approved`
- Date: `2026-07-27`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-27`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`, §16.1
- Affected ExecPlans: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

`src/thesis_rl/scenarios/arms.py` (lines 19-26) defines eight numeric
constants used by `assign_primary_arm()` to classify each scenario into one
of the six ACL bandit arms (A0-A5):
`A0_MAX_RELEVANT_VEHICLES_Q90=8.0`, `A1_JUNCTION_MAX_RELEVANT_AGENTS_Q90=8.0`,
`A1_JUNCTION_MAX_VEHICLE_CONFLICT_COUNT=1`,
`ARM_COMPLEX_RELEVANT_AGENTS_Q90=25.0`, `ARM_COMPLEX_VEHICLE_CONFLICT_COUNT=4`,
`ARM_MIXED_TOPOLOGY_CONFLICT_COUNT=3`, `ARM_CRITICAL_RELEVANT_AGENTS_Q90=30.0`,
`ARM_CRITICAL_VEHICLE_CONFLICT_COUNT=6`.

These constants were introduced in commit `4632f58` ("Adjust dataset
geenration", 2026-07-13) with no explanatory commit message, no in-code
rationale, and no dedicated ADR. Specification §16.1 states only that "the
thresholds of the existing taxonomy are frozen," which elsewhere in the
document means "fixed by decision," not "empirically derived." This is a
materially different situation from the system's other two calibrated
thresholds, `tau_low`/`tau_dense` (`src/thesis_rl/scenarios/thresholds.py:
compute_arm_thresholds`), which have an explicit, documented statistical
procedure (train-only 0.40/0.75 quantiles over the balanced Waymo+PG train
set, recorded as `DEC-008`) — and which the specification explicitly states
do *not* determine the A0/A1 boundary.

A rationale for the eight arm-difficulty constants does exist, but only in a
historical, superseded document:
`docs/implementation/scenarionet_integration_implementation_plan.md`
(superseded 2026-07-16), section "Revisione v2: da tassonomia semantica a
scala curricolare," records decision `DEC-024`
(status `PROVVISORIAMENTE_CONFERMATA`, i.e. provisionally confirmed):

> Parametri difficoltà v2 [...] Sul catalogo selezionato A0≤8 recupera 2
> Waymo, A1-junction circa 70 e A3 743→circa 572; conferma finale subordinata
> ad audit visivo.

This shows the thresholds were chosen iteratively to rebalance arm coverage
across the converted candidate pool (v1's exclusive semantic taxonomy left
A3 with 1427 Waymo scenarios and A4 with only 12), not derived from an
independent statistical procedure on scenario features. The related issue
`ISS-020` ("Le label route-aware Waymo sono state validate quantitativamente
ma non ancora con ispezione visiva stratificata") remains
`ACCETTATO_NON_BLOCCANTE` and includes the explicit constraint: "Non cambiare
soglie per ottenere un istogramma desiderato." No visual audit closing
`ISS-020` has been recorded. The v1.1 ExecPlan (`DEC-SN-003`) carried the
"preserve existing six arms" decision forward but did not re-carry `DEC-024`
or its provisional status, so the coverage-driven origin was not visible from
the current authoritative specification chain.

A read-only empirical check against the frozen dataset
(`data/scenarionet/frozen/scenario_selection_index.json`, balanced train
subset, 1000 Waymo + 1000 PG) shows the eight constants sit at roughly
increasing empirical quantiles of their respective features (approximately
Q0.40/Q0.52/Q0.67/Q0.76/Q0.81/Q0.88/Q0.90), consistent with an internally
coherent difficulty ordering, but this check is circular: the frozen dataset
was itself selected using these same thresholds, so it cannot serve as
independent calibration evidence. No pre-selection candidate/eligible pool
survives in this worktree to support a non-circular recalibration.

No document literally named "tassonomia esistente" (as a standalone artifact)
was ever committed to this repository or found in `incoming/`; the phrase in
§16.1 refers to the six-arm A0-A5 label taxonomy retained since v1
(`DEC-SN-003`, ADR-001), not to a separate calibration document.

## Decision

1. Confirm that the eight arm-difficulty constants in `arms.py` are an
   intentional, coverage-driven heuristic choice, not a statistically
   calibrated boundary. They were selected so that, on the converted
   candidate pool available at the time, all six arms received a usable
   number of scenarios across both sources, given the qualitative difficulty
   ordering in §16.1 (simple → traffic → junction → complex junction → VRU →
   critical mixed).
2. Promote `DEC-024` from `PROVVISORIAMENTE_CONFERMATA` to `CONFERMATA`. The
   visual-audit condition originally attached to it (`ISS-020`) remains a
   separate, non-blocking, still-open item and is not resolved by this ADR.
3. Record explicitly, as a limitation, that these constants are not subject
   to the same quantile-based calibration procedure as `tau_low`/`tau_dense`
   (`DEC-008`) and are not intended to be: they define an ordinal curriculum
   difficulty scale over per-scenario features, not a dataset-relative
   density tag, and re-deriving them from quantiles of the current dataset
   would be circular (the dataset was itself selected using these
   thresholds).
4. Record the additional finding that, in the current frozen train split,
   arm and source are fully confounded (`A0-A2` are 100% PG, `A3-A5` are
   100% Waymo). This is a known consequence of `A4_vru × PG = 0` (ADR-012)
   propagating through the difficulty ordering, not a new defect, but it
   means any future threshold recalibration should be evaluated jointly with
   its effect on source/arm balance, not in isolation.
5. No change to `arms.py`, to the specification's numeric values, or to any
   dataset artifact is made by this ADR. A future statistical recalibration
   (quantile-based, analogous to `tau_low`/`tau_dense`) remains a distinct,
   separately approvable decision, to be considered only together with a
   full dataset rebuild (since it requires the pre-selection candidate pool
   for a non-circular calculation) and only after current training runs are
   complete.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Statistically recalibrate the eight constants now via quantiles of the current frozen dataset | Symmetry with `tau_low`/`tau_dense`; removes "magic numbers" | Circular (dataset was selected using these thresholds); requires rebuilding the dataset from the pre-selection candidate pool, invalidating in-progress training runs; requires rewriting mandatory tests in `tests/test_scenario_arms.py` | No non-circular data source available in this worktree; user has in-progress training runs that must not be invalidated |
| Leave the constants undocumented as before | No effort | Repeats the traceability gap this ADR exists to close; violates AGENTS.md's requirement that scientific behavioral choices be recorded | Rejected as the status quo problem |
| Recalibrate only `tau_low`/`tau_dense`-style thresholds and leave arm boundaries as a separate, permanently heuristic scale | Avoids circularity entirely | Does not address whether the current specific values (8.0, 25.0, 4, etc.) are the best achievable heuristic choice | Deferred; out of scope for this ADR, which addresses provenance and status, not re-optimization |

## Consequences

The arm-difficulty thresholds are now traceable to their origin (`DEC-024`,
2026-07-13) and their status is explicit: an intentional, coverage-driven
heuristic, confirmed by this ADR, not a statistically derived boundary. This
is disclosable as an explicit limitation in the thesis without requiring a
dataset rebuild. `ISS-020`'s visual-audit condition and the source/arm
confounding noted in Decision item 4 remain open, non-blocking items tracked
separately. No code, specification value, or dataset artifact changes; no
regression risk to in-progress training runs.

## Validation And Traceability

- No requirement IDs, acceptance criteria, or mandatory tests are affected;
  this ADR changes documentation and provenance status only.
- `tests/test_scenario_arms.py` (9 parametrized `assign_primary_arm` cases)
  remains valid and unmodified; it already exercises every arm boundary at
  the current constant values.
- Follow-up (not part of this ADR, requires separate approval): add a
  one-line provenance comment in `src/thesis_rl/scenarios/arms.py` above the
  constants, pointing to this ADR; add a short provenance note to
  specification §16.1.

## Approval Record

- Approved by: user
- Approval evidence: explicit user confirmation in this conversation ("In
  pratica sono stati calibrati per poter equilibrare il numero di scenari in
  ciascun arm, giusto?" / "Sì da va bene", 2026-07-27), confirming the
  coverage-driven, non-statistical rationale for the arm-difficulty
  constants and approving Alternative A (document as intentional heuristic)
  over statistical recalibration.
- Notes: this ADR documents provenance and status only. It does not approve
  or perform any change to `arms.py`, the specification's numeric values, or
  dataset artifacts; the user has explicitly deferred any such change until
  after current training runs complete and has not yet requested it.
