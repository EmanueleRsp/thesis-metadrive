# ExecPlan: Driving Mission v1.1 Route Coordinate

## 1. Status and authorization

- Specification: `DRIVING-MISSION-V1.1`, `APPROVED`, `Authoritative: YES`
- ADR: `ADR-054`, `APPROVED`
- Plan status: `APPROVED`
- Production implementation authorization: `YES`, explicit user approval recorded 2026-08-04
- Project index: not modified
- Branch: `codex/route-coordinate-mission`
- Last updated: `2026-08-04`

This plan is the authorized implementation record. It authorizes production
code and dependent artifact work only within the approved specification; it
does not authorize scientific changes, source-data mutation, record
exclusion/replacement, or unapproved fallbacks.

## 2. Objective and scope

Prepare one immutable source-independent route mission for all 3,500 frozen scenarios. Preserve UID/source/split identity; build the canonical XY-arc-length route; implement the exact sequential cursor, shared `MissionSnapshot`, pure R4, separate completion values, fixed observations, and one passive final gate. Exclude no record under this plan.

Out of scope: source changes, future-SDC runtime access, native navigation authority, R1--R3 changes, new dependencies, online replanning, runtime lane/neighbor/boundary search, proximity fallback, component expansion, and implementation before approval.

## 3. Authoritative requirements and acceptance

| Requirement | Acceptance | Status |
|---|---|---|
| normalized route | immutable ordered 3D occurrences, finite and connected, no invented joins | planned |
| reset/goal | offline correction and `s_start=0`; positive global goal station | planned |
| exact cursor | contiguous search, exact unsaturated `s`, no clamp/freeze/recovery | planned |
| R4/completion | exact formulas and separate monotone maximum | planned |
| shared snapshot | semantic/LiDAR/reward/Rulebook/metrics consume one snapshot | planned |
| final gate | one offline geometric component segment covering the canonical anchor, passive runtime | audit: 3,500 unique |
| migration | preserve all 3,500 UID/path/split/source identities | planned |

Stable acceptance IDs: `AC-RCM-001` canonical route round-trip and finite
geometry; `AC-RCM-002` first-occurrence reset normalization; `AC-RCM-003`
exact cursor without jump envelope; `AC-RCM-004` signed delta-s R4; `AC-RCM-005`
instantaneous/max completion; `AC-RCM-006` shared snapshot identity;
`AC-RCM-007` ten route samples; `AC-RCM-008` anchor gate and directed crossing;
`AC-RCM-009` termination/truncation; `AC-RCM-010` full migration identity.

## 4. Current repository analysis

The canonical index is `data/scenarionet/frozen/scenario_selection_index.json`; source files are under `/scratch/e.respino/thesis-metadrive/data/scenarionet`. Index and records are byte-identical after alignment. Read-only availability/content checks passed 3,500/3,500.

The geometric builder uses the canonical `0.01 m` geometry tolerance and a vertical compatibility check. It audits every static lane polygon at the goal cross-section, applies static tangent concordance, parameterizes the section as `x(u)=g+u*n` with `g=r(s_goal)`, forms intervals, merges overlaps and tolerance-adjacent intervals, and selects the unique component covering `u=0` with a closed tolerance-equivalent `covers` predicate. The final occurrence is only a consistency check. Results: PG 1,695 built; Waymo 1,805 built; 3,500 total; zero anchor-missing gates; zero anchor ambiguities.

- `waymo:training_20s:dffbfc6327b84335`
- `waymo:training_20s:b1128b1ac9f4e56a`
- `waymo:training_20s:dd1ea48c3239f622`
- `waymo:training_20s:fc680771abf4226f`

The four formerly ambiguous records now have one anchor-covering component each and are not excluded. More than one anchor-covering component after tolerance merging is a builder error; no tie-break is allowed. The earlier 210 Waymo metadata-unresolved result is retained only as historical diagnostic evidence.

`VERIFIED`: v1 code in `src/thesis_rl/mission/` modeled ordered sections and
graph distance. `VERIFIED`: the v1.1 implementation now adds the normalized
route fields, frozen `FinalGateSegment`, source builder, route-coordinate
tracker, passive runtime path, signed-delta progress, and explicit completion
metrics. `BLOCKED_FOR_MIGRATION`: one Waymo record (`waymo:training_20s:5e7bbc00b872c2ba`,
frozen-index position 248) has a reset projection at approximately 59.8 m and
its terminal projection at approximately 5.4 m on the assigned lane, yielding
a negative normalized goal station. The approved contract provides no
authorized correction for this source-geometry orientation conflict. The
Parquet catalog also requires the repository's optional `pyarrow` environment.

## 5. Decisions and approval gates

| ID | Question | Status |
|---|---|---|
| `DEC-RCM-001` | approve complete specification and ADR | approved 2026-08-04 |
| `DEC-RCM-002` | enforce unique anchor-covering component and builder errors for zero/multiple components | approved by anchor-based contract; implementation test required |
| `DEC-RCM-003` | regenerate dependent artifacts without changing source records | approved policy; implementation pending |
| `DEC-RCM-004` | internal module/schema decomposition | deferred until contract approval |

The final gate schema is one frozen `final_gate_segment`, not a required complete source-semantic lane set and not a runtime envelope. It contains world geometry, static tangent, elevation, final occurrence identity, provenance/evidence, source geometry hash, and builder identity.

## 6. Planned milestones after approval

- [x] M0 — explicit approval of specification and ADR (2026-08-04).
- [x] M1 — freeze normalized-record schema and correction-first migration inputs.
- [x] M2 — build canonical route, cursor, and snapshot with deterministic tests.
- [x] M3 — integrate R4, completion, observations, Rulebook relevance, metrics, and video consumers.
- [x] M4 — materialize offline geometric final gate and directed crossing termination.
- [ ] M5 — regenerate only dependent mission artifacts, preserving all identities; blocked by one source-geometry conflict.
- [ ] M6 — run full tests, 3,500-record audit, PG/Waymo smokes, and reconciliation.

## 7. Mandatory validation matrix

Test finite/connected route geometry; first-occurrence reset association; goal containment and positive global station; self-intersection/roundabout/parallel/vertical cursor cases; exact forward/reverse/off-route R4 and speed-cap invariance; completion decrease and maximum monotonicity; shared snapshot identity; cross-section interval creation/merging, singleton and multi-lane contiguous components, non-guidable gaps, opposite/perpendicular/vertical exclusions, missing-final-occurrence detection, and ambiguity reporting; directed bumper crossing; termination/truncation; future-SDC causality; migration non-overwrite; and PG/Waymo smoke tests.

The focused implementation tests map as follows: `tests/test_driving_mission_v11.py`
(`AC-RCM-001`, `AC-RCM-003`, `AC-RCM-008`),
`tests/test_driving_mission_materialize.py` (`AC-RCM-010`), and
`tests/test_rulebook_v2_progress.py` (`AC-RCM-004`).

## 8. Validation commands

Executed commands include the focused mission/route/progress suite and a
complete read-only builder sweep. `make test`, full lint/format, configuration
checks, catalog generation, and smoke remain to be run; catalog generation is
currently blocked by missing `pyarrow`, and the builder sweep is blocked by the
single Waymo record documented below.

## 9. Findings and reconciliation

The 13 reset and 25 terminal preliminary failures were uniquely repairable offline; `PGMap-6000153` has a positive global goal station after route concatenation. These approved migration decisions remain in scope. The geometric final-gate audit supersedes the former source-metadata completeness requirement. The current builder audit fails only the Waymo UID recorded above; no record was excluded or replaced. The stale pre-normalization generated JSON was moved to `/tmp` and is not canonical.

## 10. Progress and findings log

- 2026-08-04: user approved specification, ADR, index promotion, and implementation.
- 2026-08-04: promoted documents and corrected the pre-existing bounded/clamp index entry.
- 2026-08-04: implemented v1.1 route record, anchor gate, route tracker, passive runtime, signed-delta progress, and completion fields.
- 2026-08-04: focused v1.1 mission/materialization/progress tests passed (`11 passed`); a complete read-only builder sweep found one negative normalized-goal record at frozen-index position 248.
- 2026-08-04: Parquet catalog promotion not completed because `pyarrow` is unavailable; no dependency was added.

## 11. Deviations

No deviations from the approved specification identified. The legacy v1 classes
remain only for deserialization and compatibility tests; the v1.1 runtime path
does not use intermediate gates, graph distance, envelope expansion, or native
navigation as authority.

## 12. Files

| Path | Action |
|---|---|
| specification | approved authority |
| ADR-054 | approved decision |
| this ExecPlan | approved implementation record |
| audit findings | append geometric audit evidence |
| `data/scenarionet/frozen/scenario_selection_mission_v1_1.json` | pending | regeneration blocked by one source-geometry conflict |
| `docs/project_index.md` | modified | register v1.1 authority and v1.0 supersession |

## 13. Validation results

| Command | Result | Notes |
|---|---|---|
| focused mission/progress tests | PASS | 11 tests after the final `s_start=0` correction |
| 3,500-record builder sweep | BLOCKED | 3,499 records build; one Waymo record has terminal station before reset station on its assigned geometry |
| `git diff --check` | PASS | 2026-08-04 |
| Parquet catalog promotion | NOT_RUN | `pyarrow` missing; install/use provisioned environment before catalog build |
| full `make test` | NOT_RUN | final reconciliation pending |
| `make lint`, format, config, smoke | NOT_RUN | final reconciliation pending |

## 14. Final reconciliation

`REQ-RCM-001` through `REQ-RCM-007` are implemented in the owned mission path;
focused acceptance coverage is present, while full repository verification and
the 3,500-record artifact regeneration are pending. Known limitations: the
v1.1 Parquet catalog cannot be emitted in the current host without `pyarrow`,
and one Waymo record requires an explicitly authorized source-geometry
correction or a data-abort classification. No scientific fallback or
dependency decision was made. The approved specification, ADR, and index are
authoritative; the plan remains `APPROVED` while implementation reconciliation
is blocked by these repository/data constraints.

The ChatGPT project source files changed during this task are
`docs/project_index.md`; replace that exact source in the project. The other
two synchronized sources, `docs/engineering_workflow.md` and
`docs/templates/specification_template.md`, were not changed.
