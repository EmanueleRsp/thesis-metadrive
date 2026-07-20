# Rulebook v4.7 Post-R1 Correctness Audit ExecPlan

## 1. Metadata

- Feature: Rulebook v4.7 correctness audit after the R1 live-contact repair
- Plan ID: `RULEBOOK-V4.7-POST-R1-AUDIT`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`, Rulebook v4.7, approved
- Status: `BLOCKED_PENDING_VEHICLE_PRIORITY_SOURCE_DECISION`
- Created: 2026-07-20
- Last update: 2026-07-20
- Related ADRs: `docs/decisions/ADR-015-r1-pre-state-centerline-normal.md`
- Owner: Codex

## 2. Objective And Scope

Audit every non-R1 Rulebook component and its required live/static input path
for defects that could silently invalidate experiment results. Repairs that
restore the approved specification are in scope; a new approval is required
only if a missing required source would force a new fallback or data policy.

In scope: R2 RSS/TTC/clearance; R3 road, markings, signal, stop, crosswalk,
and vehicle-yield; R4 progress; transition composition; snapshots; static
adapters; runtime logging; deterministic tests; and the supplied smoke run.
Out of scope: changing a formula, frozen parameter, fallback, data policy, or
experiment configuration without explicit approval.

## 3. Audit Requirements And Acceptance Criteria

| ID | Requirement | Evidence required |
|---|---|---|
| AUD-001 | Every component follows its authoritative formula, units, timing, and applicability policy. | Spec-to-code traceability and focused tests. |
| AUD-002 | Runtime inputs are causal and source-normalized. | Snapshot and adapter call-path inspection. |
| AUD-003 | Silent fallback, stale state, invalid geometry, and missing-data paths are identified. | Failure-path review and log inspection. |
| AUD-004 | Observed smoke-run component values are internally consistent. | JSONL/CSV evidence. |

## 4. Audit Method

1. Map the transition input construction to all registered evaluators.
2. Compare R2–R4 component formulas and state machines against the approved
   specification.
3. Trace live and static adapter data sources, coordinate frames, units, and
   validation gates.
4. Run the existing containerized Rulebook suite and inspect the supplied
   smoke-run logs for component activation, termination, and anomalous values.
5. Classify each finding as verified correct, limitation, suspected defect, or
   confirmed defect. Repair confirmed implementation defects while preserving
   the approved behavior; stop only for a new data-policy decision.

## 5. Initial Findings

| Date | Finding | Status |
|---|---|---|
| 2026-07-20 | The supplied run has six R1 onsets, all non-floor and sourced from pre-state canonical footprint centers. | Verified R1 regression resolved. |
| 2026-07-20 | The supplied run has frequent `time_limit` truncations with low route completion and often negative route-parallel speed. | Under audit; may be policy behavior rather than Rulebook defect. |
| 2026-07-20 | RSS evaluator formula matches §6.2, but transition candidate construction used same live lane ID and Euclidean speed magnitude. | Repaired: canonical 2.5D lane association, heading-concordance filtering, and tangent-projected speeds are used. |
| 2026-07-20 | Signal/crosswalk/vehicle-yield stopping-distance code used fixed `SIGNAL_BRAKE_MPS2=8.0`, rather than calibrated RSS `b_e`. | Repaired for signal/crosswalk and evaluator-ready vehicle yield: the transition supplies calibrated `b_e`; a relevant evaluated control now fails fast if calibration is absent. |
| 2026-07-20 | Transition selected signal and stop controls only at `route_s_m >= post_front_s`. A control crossed during the transition could therefore disappear before evaluation. | Repaired: pre/post selection retains the pre-state control through crossing evaluation; crossing now also requires the swept front bumper, control-line intersection, and the specified 0.05 m deadband. |
| 2026-07-20 | Static adapters created one traffic-control record per physical signal and transition read only the first physical state. | Repaired: equivalent heads are normalized to one deterministic group and every transition checks all group states for concordance. |
| 2026-07-20 | Crosswalk construction used `task_route.lane_ids[0]`, did not select occupied-else-first-ahead zones, and did not vertically filter VRUs before occupancy prediction. | Repaired: current canonical lane association, normative zone selection, vertical filtering, and configured causal prediction controls are used. |
| 2026-07-20 | Vehicle-yield transition input is a fixed `__no_vehicle_priority__` placeholder with no candidates or conflict-zone construction. | Confirmed defect: §7.9 is permanently `NOT_APPLICABLE` at runtime. |
| 2026-07-20 | Source-feasibility review: 500 sampled PG records expose only lane/marking topology and no traffic-control feature; the checked ScenarioNet/Waymo records and converter schema expose STOP_SIGN and dynamic traffic lights, but no yield sign, right-of-way, movement-priority, or roundabout-membership attribute. | Verified source limitation: no `NONE`, pairwise, or roundabout predicate can be established without adding an unverified source policy. |
| 2026-07-20 | Solid/dashed boundary inputs included every map feature of the matching class without applying the required vertical compatibility filter. | Repaired: only vertically compatible, elevation-complete markings are supplied; incomplete relevant static records are rejected. |
| 2026-07-20 | TTC, clearance, off-road, wrong-way, dashed timer, and progress have formula/code paths consistent with their primary requirements in the reviewed implementation. Their tests are primarily deterministic unit coverage; source-specific live coverage remains limited. | Verified within audit scope; no defect confirmed. |
| 2026-07-20 | In the supplied run, all of crosswalk, signal, stop, and vehicle-yield are `NOT_APPLICABLE` in all 2,654 logged transitions. R2/R3/R4 otherwise activate with finite bounded values; off-road and wrong-way detect extensive unsafe behavior despite native flags remaining false. | Runtime observation; it neither validates nor disproves the missing control/yield paths. |

## 6. Validation Matrix

| Command | Purpose | Status |
|---|---|---|
| `make rulebook-v2-check` | All Rulebook v2 tests, scoped Ruff, whitespace check | PASS: 185 passed, 1 skipped; Ruff and whitespace check passed on 2026-07-20 |
| Supplied smoke JSONL/CSV inspection | Runtime consistency and activation coverage | Completed: 2,654 transitions reviewed |
| `git diff --check` | Patch integrity if remediation is approved | PASS as part of `make rulebook-v2-check` |

## 7. Final Reconciliation

AUD-001 through AUD-004 are complete for the reviewed code and supplied smoke
run. The independent RSS and R3 control/marking repairs have deterministic
regressions and pass the complete Rulebook v2 suite. Vehicle yield remains
blocked: both static adapters always emit an empty `movement_priority_records`
tuple and only create STOP/SIGNAL controls, never validated `NONE`/roundabout
approach controls. The transition is consequently still a fixed
`__no_vehicle_priority__` NOT_APPLICABLE domain. Inferring priority from the
absence of a record or from geometry would violate §7.9.1; a source/data-policy
decision is required before implementing that remaining rule.

The source-feasibility review inspected the static adapter schemas, the
ScenarioNet Waymo converter/proto, 500 PG records, and bundled Waymo records.
All inputs inspected are static or reset-available, so no privileged future
trajectory is involved; however, the required priority facts are absent. In
particular, the Waymo `MapFeature` schema does not encode yield signs or
right-of-way, while PG export records do not encode controls at all. A lane
graph cycle is explicitly insufficient evidence of a roundabout. No further
vehicle-yield repair is therefore admissible under the approved specification.

User direction on 2026-07-20: repair implementation defects directly while
preserving the approved Rulebook v4.7 behavior; request a new decision only if
a required source input cannot be obtained without a fallback.
