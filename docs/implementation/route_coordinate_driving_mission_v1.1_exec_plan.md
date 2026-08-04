# ExecPlan: Driving Mission v1.1 Route Coordinate and v1.1.1 Amendment

## 1. Status and authorization

- Specification: `DRIVING-MISSION-V1.1`, `APPROVED`, `Authoritative: YES`
- ADR: `ADR-054`, `APPROVED`
- Plan status: `APPROVED`
- Amendment: `DRIVING-MISSION-V1.1.1`, `APPROVED`, `Authoritative: YES`
- Amendment ADR: `ADR-055`, `APPROVED`
- Production implementation authorization for the amendment: `YES`, explicit user approval recorded 2026-08-04
- Project index: v1.1.1 registration authorized
- Branch: `codex/route-coordinate-mission`
- Last updated: `2026-08-04`

This plan is the authorized implementation record for v1.1 and the approved
v1.1.1 amendment. It does not authorize scientific fallback, source-data
mutation, record exclusion/replacement, or route-planning alternatives.

## 2. Objective and scope

Prepare one immutable source-independent route mission for all 3,500 frozen scenarios. Preserve UID/source/split identity; orient distinct occurrences offline using the frozen assigned lane-ID sequence and the complete SDC trajectory only as an offline orientation witness; trim one mission-local canonical XY-arc-length route from reset to terminal goal; then apply the approved exact cursor, shared `MissionSnapshot`, pure R4, separate completion values, fixed observations, and one passive final gate. During dataset generation, build and validate this mission immediately after Rulebook filtering and before split selection. Records that fail mission construction are reported and removed from the candidate pool before split feasibility is evaluated, allowing the existing replenishment loop to acquire more source scenarios. Exclude no record from the already audited 3,500 frozen source selection under the approved migration.

Out of scope: source changes, future-SDC runtime access, native navigation authority, R1--R3 changes, new dependencies, online replanning, runtime lane/neighbor/boundary search, proximity fallback, component expansion, and implementation before approval.

## 3. Authoritative requirements and acceptance

| Requirement | Acceptance | Status |
|---|---|---|
| normalized route | immutable ordered oriented 3D occurrences and directly serialized mission-local route, finite and connected, no invented joins | implemented |
| reset/goal | explicit reset/goal endpoints, `s_start=0`, positive trimmed `s_goal` | implemented |
| exact cursor | contiguous search, exact unsaturated `s`, no clamp/freeze/recovery | planned |
| R4/completion | exact formulas and separate monotone maximum | planned |
| shared snapshot | semantic/LiDAR/reward/Rulebook/metrics consume one snapshot | planned |
| final gate | one offline geometric component segment covering the canonical anchor, passive runtime | audit: 3,500 unique |
| migration | preserve all 3,500 UID/path/split/source identities | implemented; pre-split eligibility integrated |

Stable acceptance IDs: `AC-RCM-001` canonical route round-trip and finite
geometry; `AC-RCM-002` first-occurrence reset normalization; `AC-RCM-003`
exact cursor without jump envelope; `AC-RCM-004` signed delta-s R4; `AC-RCM-005`
instantaneous/max completion; `AC-RCM-006` shared snapshot identity;
`AC-RCM-007` ten route samples; `AC-RCM-008` anchor gate and directed crossing;
`AC-RCM-009` termination/truncation; `AC-RCM-010` full migration identity.

Amendment acceptance IDs: `AC-RCM-011` occurrence orientation from temporal
source-station progression; `AC-RCM-012` occurrence connectivity without route
replacement or connectors; `AC-RCM-013` mission-local reset/goal trimming and
positive XY length; `AC-RCM-014` absence of trajectory in serialized/runtime
state; `AC-RCM-015` explicit reporting of mission/legal-direction conflicts.

## 4. Current repository analysis

The canonical index is `data/scenarionet/frozen/scenario_selection_index.json`; source files are under `/scratch/e.respino/thesis-metadrive/data/scenarionet`. Index and records are byte-identical after alignment. Read-only availability/content checks passed 3,500/3,500.

The geometric builder uses the canonical `0.01 m` geometry tolerance and a vertical compatibility check. It audits every static lane polygon at the goal cross-section, applies static tangent concordance, parameterizes the section as `x(u)=g+u*n` with `g=r(s_goal)`, forms intervals, merges overlaps and tolerance-adjacent intervals, and selects the unique component covering `u=0` with a closed tolerance-equivalent `covers` predicate. The final occurrence is only a consistency check. Results: PG 1,695 built; Waymo 1,805 built; 3,500 total; zero anchor-missing gates; zero anchor ambiguities.

- `waymo:training_20s:dffbfc6327b84335`
- `waymo:training_20s:b1128b1ac9f4e56a`
- `waymo:training_20s:dd1ea48c3239f622`
- `waymo:training_20s:fc680771abf4226f`

The four formerly ambiguous records now have one anchor-covering component each and are not excluded. More than one anchor-covering component after tolerance merging is a builder error; no tie-break is allowed. The earlier 210 Waymo metadata-unresolved result is retained only as historical diagnostic evidence.

The initial occurrence-orientation audit used a global polygon scan. Because
lane polygons overlap at junctions, that method could assign late poses back
to an earlier occurrence and invert its apparent station progression. The
builder now uses a sequence-constrained association that advances only through
the frozen occurrence order and never revisits a prior occurrence. A corrected
read-only sweep over all 3,500 records built 3,500 positive mission-local
routes: 1,695 PG and 1,805 Waymo, with 12,405 `FORWARD` and 1 `REVERSED`
occurrence classifications and no unresolved orientation.

For `waymo:training_20s:5e7bbc00b872c2ba`, all 199 valid poses are contained by
exactly one static lane candidate (`89`), the source station sequence has 187
negative, 0 positive, and 11 zero changes, and the occurrence is uniquely
`REVERSED`. Its projected reset is source station `59.823228 m`, its terminal
is `5.376686 m`, and the trimmed mission-local length is `54.368483 m`.

`VERIFIED`: v1 code in `src/thesis_rl/mission/` modeled ordered sections and
graph distance. `VERIFIED`: the v1.1 implementation adds the normalized route
fields, frozen `FinalGateSegment`, source builder, route-coordinate tracker,
passive runtime path, signed-delta progress, and explicit completion metrics.
The approved v1.1.1 contract represents source centerline orientation per frozen
route occurrence and trims the canonical route to reset and goal offline. The
implementation and artifact regeneration are authorized. The Parquet catalog
additionally requires the repository's optional `pyarrow` environment.

### Dataset-pipeline placement

The authoritative generation path is now:

`source generation/acquisition → raw catalog → Rulebook v2 filtering and driving-mission build → split feasibility and replenishment → thresholds → runtime views → freeze`

`filter_rulebook_v2_catalog` constructs a mission for every Rulebook-eligible
candidate and writes `driving_mission` into the filtered catalog. It also emits
`rulebook_v2/driving_mission_eligibility.json`, including deterministic builder
identity, per-record errors, and source counts. A candidate without a valid
v1.1.1 mission is not passed to split selection. The split CLIs accept
`--require-driving-mission` and reject a catalog that is not mission-ready; the
normal v1.1/v1.2 orchestration enables this flag. The freeze CLI always requires
the approved mission schema, so a post-freeze repair cannot silently produce an
official dataset.

The existing `materialize_driving_missions` command remains a migration and
diagnostic tool for already frozen catalogs. It is not the normal generation
path and cannot replace the pre-split eligibility stage.

## 5. Decisions and approval gates

| ID | Question | Status |
|---|---|---|
| `DEC-RCM-001` | approve complete specification and ADR | approved 2026-08-04 |
| `DEC-RCM-002` | enforce unique anchor-covering component and builder errors for zero/multiple components | approved by anchor-based contract; implementation test required |
| `DEC-RCM-003` | regenerate dependent artifacts without changing source records | approved policy; implementation pending |
| `DEC-RCM-004` | internal module/schema decomposition | deferred until contract approval |

The final gate schema is one frozen `final_gate_segment`, not a required complete source-semantic lane set and not a runtime envelope. It contains world geometry, static tangent, elevation, final occurrence identity, provenance/evidence, source geometry hash, and builder identity.

## 6. Planned milestones after approval

- [x] M0 — explicit approval of v1.1 specification and ADR (2026-08-04).
- [x] M1 — freeze v1.1 normalized-record schema and correction-first migration inputs.
- [x] M2 — implement the previously approved v1.1 route, cursor, and snapshot baseline with deterministic tests.
- [x] M3 — integrate the previously approved v1.1 R4, completion, observation, and consumer baseline.
- [x] M4 — complete the previously approved v1.1 anchor-gate audit.
- [x] M5 — approve and implement v1.1.1 occurrence orientation and mission-local trimming (2026-08-04).
- [x] M6 — implement sequence-constrained occurrence association and run the corrected 3,500-record amended audit (2026-08-04).
- [x] M7 — integrate offline mission eligibility before split selection; require the mission schema in split and freeze paths; add catalog/report and regression coverage (2026-08-04).

## 7. Mandatory validation matrix

Test finite/connected route geometry; first-occurrence reset association; goal containment and positive global station; self-intersection/roundabout/parallel/vertical cursor cases; exact forward/reverse/off-route R4 and speed-cap invariance; completion decrease and maximum monotonicity; shared snapshot identity; cross-section interval creation/merging, singleton and multi-lane contiguous components, non-guidable gaps, opposite/perpendicular/vertical exclusions, missing-final-occurrence detection, and ambiguity reporting; directed bumper crossing; termination/truncation; future-SDC causality; migration non-overwrite; and PG/Waymo smoke tests.

The focused implementation tests map as follows: `tests/test_driving_mission_v11.py`
(`AC-RCM-001`, `AC-RCM-003`, `AC-RCM-008`),
`tests/test_driving_mission_materialize.py` (`AC-RCM-010`), and
`tests/test_rulebook_v2_progress.py` (`AC-RCM-004`).

## 8. Validation commands

Executed commands include the focused mission/route/progress suite, the
ScenarioNet pipeline regression suite, the complete read-only
sequence-constrained builder sweep, focused Ruff checks, and shell syntax
validation. The real catalog regeneration remains a data operation and was not
run in this implementation pass.

## 9. Findings and reconciliation

The 13 reset and 25 terminal preliminary failures were uniquely repairable offline; `PGMap-6000153` has a positive global goal station after route concatenation. These approved migration decisions remain in scope. The geometric final-gate audit supersedes the former source-metadata completeness requirement. The first orientation audit falsely reported one invalid route because it globally reused overlapping lane polygons. The corrected sequence-constrained builder resolves all 3,500 records, including `waymo:training_20s:6042e6c648fca15d`, with no exclusion or replacement. The stale pre-normalization generated JSON was moved to `/tmp` and is not canonical.

## 10. Progress and findings log

- 2026-08-04: user approved specification, ADR, index promotion, and implementation.
- 2026-08-04: promoted documents and corrected the pre-existing bounded/clamp index entry.
- 2026-08-04: implemented v1.1 route record, anchor gate, route tracker, passive runtime, signed-delta progress, and completion fields.
- 2026-08-04: focused v1.1.1 mission/materialization/progress tests passed (`12 passed`); the target false conflict was traced to global polygon reassociation, fixed with sequence-constrained occurrence association, and the corrected 3,500-record sweep passed (`3,500/3,500`).
- 2026-08-04: Parquet catalog promotion not completed because `pyarrow` is unavailable; no dependency was added.
- 2026-08-04: integrated mission construction into the Rulebook pre-split filter; records failing mission construction are reported and excluded before split feasibility/replenishment, while split and freeze reject non-mission-ready catalogs. The focused pipeline and mission suite passed (`105 passed`).

## 11. Deviations

The v1.1.1 occurrence orientation and mission-local trimming amendment is now
approved and implemented in the mission builder/runtime path. The legacy v1
classes remain only for deserialization and compatibility tests; the v1.1.1
route path does not use intermediate gates, graph distance, envelope expansion,
native navigation, or a source-station offset as authority.

## 12. Files

| Path | Action |
|---|---|
| specification | approved authority |
| ADR-054 | approved decision |
| this ExecPlan | approved implementation record |
| audit findings | append geometric audit evidence |
| `data/scenarionet/frozen/scenario_selection_mission_v1_1_1.json` | pending | atomic promotion still requires the provisioned artifact-generation environment (`pyarrow`); builder audit is complete |
| `docs/project_index.md` | modified | register v1.1.1 authority and ADR-055 |
| `src/thesis_rl/scenarios/mission_eligibility.py` | added | deterministic pre-split mission builder/audit adapter |
| `scripts/prepare_scenarionet_dataset.sh` | modified | build mission before split and retry feasibility after mission filtering |

## 13. Validation results

| Command | Result | Notes |
|---|---|---|
| focused mission/progress tests | PASS | 12 tests including reversed occurrence and mission-local endpoint regression |
| 3,500-record sequence-constrained builder audit | PASS | 3,500/3,500 routes built; 3,500/3,500 positive `s_goal`; no exclusions |
| `git diff --check` | PASS | 2026-08-04 |
| Parquet catalog promotion | NOT_RUN | `pyarrow` missing; install/use provisioned environment before catalog build |
| full `make test` | NOT_REPEATED | Earlier baseline run: 1,266 passed and 51 failures outside this scoped pipeline integration; the current acceptance evidence is the focused 105-test suite below |
| `make lint`, format, config, smoke | NOT_RUN | final reconciliation pending |
| focused ScenarioNet/mission pipeline pytest suite | PASS | 105 tests in Docker, 2026-08-04 |
| focused Ruff check and format check | PASS | all modified Python files, 2026-08-04 |
| `bash -n scripts/prepare_scenarionet_dataset.sh` | PASS | 2026-08-04 |

## 14. Final reconciliation

`REQ-RCM-001` through `REQ-RCM-007` and amendment requirements
`AC-RCM-011` through `AC-RCM-015` are implemented for every record in the
corrected builder audit. The required 3,500/3,500 route-builder acceptance is
met: all routes are connected, positive, and have unique orientation/gate
construction, including `waymo:training_20s:6042e6c648fca15d`. The Parquet
catalog cannot be emitted in the current host without `pyarrow`; this is an
environmental artifact-generation limitation, not a mission-validity issue.
The normal pipeline now performs mission validation before split selection and
the official freeze path requires the validated mission schema. No source
pickle, planning route, or runtime fallback was introduced. The approved
3,500-record source selection remains unchanged; any future source generation
pass may replenish records only through the existing post-filter feasibility
loop.

The ChatGPT project source files changed during this task are
`docs/project_index.md`; replace that exact source in the project. The other
two synchronized sources, `docs/engineering_workflow.md` and
`docs/templates/specification_template.md`, were not changed.
