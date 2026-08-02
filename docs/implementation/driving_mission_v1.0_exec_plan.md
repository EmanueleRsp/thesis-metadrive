# ExecPlan: Unified Driving Mission v1.0

## 1. Metadata

- Feature: unified driving mission, ordered gates, route completion, and R4
- Plan ID: `EXEC-DRIVING-MISSION-V1.0`
- Target specification:
  `docs/specifications/driving_mission_v1.0_specification_UNDER_REVIEW.md`
- Specification ID/version: `DRIVING-MISSION-V1.0`, `1.0-under-review`
- Specification approval status: `UNDER_REVIEW`, `Authoritative: NO`
- Plan status: `AWAITING_DECISIONS`
- Created: `2026-08-02`
- Last updated: `2026-08-02`
- Preparation branch: `scenarionet-implementation`
- Intended implementation branch: to be created by the user from the commit
  containing this documentation package
- Owner: thesis repository maintainer
- Related ADRs: ADR-004, ADR-022, ADR-033, ADR-036, ADR-043, ADR-049,
  ADR-051, and proposed ADR-052
- Preliminary audit:
  `docs/audits/driving_mission_feasibility_2026-08-02/findings.md`

This plan is not implementation-authoritative while its target specification
is `UNDER_REVIEW`. Production changes must not begin until M0 records explicit
approval and promotes the specification and ADR.

## 2. Objective And Scope

### Objective

Implement one frozen lane-level mission and one runtime mission tracker so R4,
observations, Rulebook route relevance, completion, success, metrics, and video
diagnostics use the same ordered task state. Legal alternate lanes remain valid
when the next mandatory gate is reachable. R4 becomes normalized remaining-
distance reduction, invariant to the ego configured speed cap.

Success is recognized when all approved requirements and acceptance criteria
are verified, the regenerated 3,500-record dataset passes the mission audit,
both PG and Waymo smoke paths pass, and no unapproved compatibility or
experimental deviation remains.

### In scope

- candidate-specification approval and ADR promotion;
- trusted read-only full-catalog preflight;
- mission records, source adapters, graph builder, gate builder, and tracker;
- environment integration and new termination/metric semantics;
- Rulebook/R4 migration;
- semantic and stacked-LiDAR observation schema successors;
- frozen dataset, manifests, checkpoint/replay identities, logs, and video;
- acceptance-first tests, deterministic regressions, smoke, and reconciliation.

### Out of scope

- algorithm, scalarizer, R1/R2 formula, or non-route R3 redesign;
- free destination replanning, temporary-road-closure planning, or parking;
- new external dependency;
- changing approved split membership without a separate explicit decision;
- production implementation in this preparation branch/session.

### Compatibility

This is intentionally incompatible with current route-related datasets,
checkpoints, replay buffers, normalizers, golden traces, and experiment results.
Tensor widths should remain `3009` for the semantic successor and `6489` for
the LiDAR successor unless preflight proves that an additional policy field is
necessary and the user approves it. A new schema ID is mandatory even if shape
is unchanged.

## 3. Authoritative Requirements

The following are target requirements, not yet authoritative.

| ID | Requirement | Candidate specification section |
|---|---|---|
| `REQ-MSN-001` | One immutable versioned mission per selected scenario | §6, REQ-MSN-001 |
| `REQ-MSN-002` | Ordered sections with preferred and allowed lane spans | §6, REQ-MSN-002 |
| `REQ-MSN-003` | Directed sequential gate crossing | §6, REQ-MSN-003 |
| `REQ-MSN-004` | Directed final goal from offline annotation | §6, REQ-MSN-004 |
| `REQ-MSN-005` | One environment-owned tracker and immutable snapshot | §6, REQ-MSN-005 |
| `REQ-MSN-006` | Deterministic legal recovery and remaining distance | §6, REQ-MSN-006 |
| `REQ-MSN-007` | Globally normalized remaining-distance R4 | §6-§7.2 |
| `REQ-MSN-008` | Mission-derived completion, success, and boundaries | §6, REQ-MSN-008 |
| `REQ-MSN-009` | Cross-arm observation consistency | §6, REQ-MSN-009 |
| `REQ-MSN-010` | Route-dependent Rulebook consistency | §6, REQ-MSN-010 |
| `REQ-MSN-011` | No future or privileged runtime information | §4, §6 |
| `REQ-MSN-012` | Full-catalog audit and controlled dataset migration | §6, REQ-MSN-012 |
| `REQ-MSN-013` | Mission diagnostics and evaluation metrics | §6, REQ-MSN-013 |
| `REQ-MSN-014` | Idempotent timing, reset, numerical, and seed behavior | §6, REQ-MSN-014 |
| `REQ-MSN-015` | Explicit dataset/checkpoint/replay incompatibility | §6, REQ-MSN-015 |

## 4. Current Repository Analysis

### Verified current call flow

1. `ScenarioRecord` in `src/thesis_rl/scenarios/records.py` loads the frozen
   `assigned_route_lane_ids` and source string.
2. `ThesisScenarioEnv._inject_assigned_route_metadata_into_scenario` injects
   that annotation after ScenarioDataManager reset.
3. PG/Waymo static adapters build `TaskRouteRecord`, all `RouteLaneRecord`
   objects, one preferred `RoutePolyline`, controls, zones, and episode cache.
4. `initial_memory_for_snapshot` initializes `previous_route_s_m` from a
   preferred-polyline projection.
5. each Rulebook transition evaluates `components/progress.py::evaluate_progress`
   with preferred-polyline pre/post projections and actor configured speed cap.
6. semantic and LiDAR builders independently project current ego and sample
   ten preferred-route waypoints.
7. `ThesisScenarioEnv._is_thesis_success` and `_attach_route_metrics` read
   MetaDrive native `TrajectoryNavigation`, creating a separate task authority.

### Exact current paths and symbols

| Classification | Path/symbol | Finding |
|---|---|---|
| `VERIFIED` | `src/thesis_rl/scenarios/records.py::ScenarioRecord` | No mission/gate/goal fields |
| `VERIFIED` | `src/thesis_rl/scenarios/catalog.py`, `frozen.py` | Catalog/frozen JSON serialize complete records and hashes |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/types.py::TaskRouteRecord` | Preferred lane list only |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/geometry/lanes.py::RouteLaneRecord` | Successors only; no normalized lateral relations/spans |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/geometry/route.py::RoutePolyline` | One concatenated preferred coordinate; lane starts diagnostic |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/components/progress.py::evaluate_progress` | `delta_s / (configured_cap * delta_t)` and `previous_route_s_m` |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/transition.py::route_reachable_control_lane_ids` | ADR-051 one-hop static workaround |
| `VERIFIED` | `src/thesis_rl/envs/thesis_scenario_env.py` | Native success/completion, frozen route adapter wiring |
| `VERIFIED` | `src/thesis_rl/envs/observations/causal_semantic.py` | ten 5 m preferred-route tokens; route-coordinate field |
| `VERIFIED` | `src/thesis_rl/envs/observations/assigned_route.py` | 22D navigation block from preferred polyline |
| `VERIFIED` | `src/thesis_rl/contracts/checkpoint_manifest.py` | observation identity participates in checkpoint compatibility |
| `VERIFIED` | `conf/env/scenarionet.yaml` | native success threshold `0.95`, minimum length `10 m` |
| `VERIFIED` | `conf/obs/semantic_v3.yaml`, `stacked_lidar_v2.yaml` | current route-source identities |
| `VERIFIED` | frozen JSON index | 3,500 records, all route-bearing and eligible; detailed audit findings recorded |
| `INFERRED` | source map topology | sufficient successor/lateral fields likely exist for mission building, but full coverage is not established |
| `AWAITING_CONFIRMATION` | source pickle trust | full audit requires explicit informed authorization to deserialize the exact frozen project artifacts |

### Relevant behavior to preserve

- ADR-004 immutable task annotation and strict runtime no-future boundary;
- canonical 3D/vertical compatibility and deterministic projection behavior;
- swept-front-bumper line-crossing primitive and current `0.05 m` tolerance;
- Rulebook lexicographic group ordering and existing R1-R3 cost formulas;
- observation causal/perception boundary and ADR-033 Rulebook-state exclusion;
- single-process and deterministic subprocess reset isolation;
- success/collision/out-of-road termination versus time-limit truncation;
- exact split/UID identity unless separately approved;
- diagnostic-only nature of video overlays.

### Directly relevant debt

- preferred route and native success use different coordinates;
- `previous_route_s_m` is Rulebook-owned although mission progress is an
  environment-level task state;
- static consumers each reconstruct route relevance;
- route projection can be discontinuous on self-intersections/roundabouts;
- ADR-051 covers one successor hop only;
- lane-start markers are named checkpoints without checkpoint semantics;
- current observation schemas have no mission schema identity.

### Dependency constraints

- no dependency addition is approved;
- use existing Shapely and repository graph/data structures or a bounded owned
  implementation;
- supported Python is `>=3.10,<3.11`;
- Docker Compose is the primary reproducible environment;
- source ScenarioDescription files are pickle artifacts and require trusted-
  provenance authorization before deserialization.

## 5. Assumptions And Invariants

| Item | Proposed invariant | Evidence/status | Failure behavior |
|---|---|---|---|
| Units | world/lane distance meters, time seconds, speed m/s, heading radians | Existing Rulebook contract; `VERIFIED` | Reject non-finite/wrong-domain input |
| Coordinates | canonical world XYZ, lane-local forward `s` | Existing adapters; `VERIFIED` | Typed static/runtime error |
| Mission time | frozen before reset; no runtime mutation of task order | ADR-004; `SPECIFIED` | Fail closed |
| Tracker update | exactly once per committed transition | Candidate REQ-MSN-005; `AWAITING_APPROVAL` | Fatal consistency error |
| Observation timing | post-commit snapshot only | Existing causal boundary; `VERIFIED` | Observation construction error |
| R4 range | finite `[-1,1]` | Candidate formula | Startup/runtime validation |
| `v_ref` | `22.2222222222 m/s`, global | `DEC-MSN-003`; `AWAITING_APPROVAL` | No scenario override |
| Gate epsilon | `0.05 m` | Existing control crossing; proposed reuse | Frozen config mismatch |
| Completion | max-so-far `[0,1]`; success sets `1` | `DEC-MSN-006`; `AWAITING_APPROVAL` | Fatal invariant error |
| Time limit | truncation only | Current environment; `VERIFIED` | Regression failure |
| Unreachable | task-failure termination | `DEC-MSN-001`; `AWAITING_APPROVAL` | Dependent work blocked |
| Semantic shape | `3009`, new schema | `DEC-MSN-004`; `AWAITING_APPROVAL` | Reject old identity |
| LiDAR shape | `6489`, new schema | `DEC-MSN-004`; `AWAITING_APPROVAL` | Reject old identity |
| Reset | no state across episodes/slots | Runtime contract; `VERIFIED` | Test failure/fatal |
| Split identity | UIDs and membership unchanged during migration | `DEC-MSN-008`; `AWAITING_APPROVAL` | Block materialization |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-MSN-001` | specification clarification | Boundary when pending gate is unreachable | terminate / continue / truncate | terminate `mission_unreachable` | done/replay/metrics | Awaiting approval |
| `DEC-MSN-002` | scientific behavior | Current final-goal source | terminal SDC projection / lane end / source-specific | terminal SDC projection for both sources | dataset/success | Awaiting approval |
| `DEC-MSN-003` | scientific parameter | R4 scale | 80 km/h global / empirical / scenario cap | global `22.2222222222 m/s` | reward comparability | Awaiting approval |
| `DEC-MSN-004` | compatibility | Observation migration | new same-shape IDs / wider / overwrite | new same-shape IDs | checkpoints/replays | Awaiting approval |
| `DEC-MSN-005` | task semantics | Alternate lane/recovery | exact path / final-only replanning / ordered recovery | fixed gates plus legal recovery | realism/R4 | Awaiting approval |
| `DEC-MSN-006` | success semantics | Goal condition | radius / directed crossing / crossing+stop | directed crossing, no stop | termination | Awaiting approval |
| `DEC-MSN-007` | annotation semantics | Intermediate gate placement | starts / every lane end / mandatory boundaries | section exits at mandatory movement boundaries | route density | Awaiting approval |
| `DEC-MSN-008` | data policy | Invalid mission after audit | replace / exclude/rebalance / block | block and request explicit policy | split validity | Awaiting approval |
| `DEC-MSN-009` | scientific claim | Relationship to PBRS | claim / exact PBRS / direct objective | direct bounded R4, no PBRS claim | thesis justification | Awaiting approval |
| `DEC-MSN-010` | blocking technical issue | Authorization to deserialize exact frozen pickle sources for preflight | authorize trusted exact paths / provide safe converted form | explicit informed authorization for pinned project artifacts | M1 audit | Awaiting user authorization |

No dependent production milestone starts while these gates are open. Private
class names and helper decomposition after approval are implementation details.

## 7. Proposed Design

### 7.1 Data contract

Proposed owned package:

```text
src/thesis_rl/mission/
├── types.py       # immutable annotations and snapshots
├── topology.py    # normalized legal lane graph and associations
├── builder.py     # deterministic offline mission annotation
├── gates.py       # directed gate construction/crossing
├── distance.py    # cached legal distance and stable routing
└── tracker.py     # episode-local state owner
```

Proposed frozen structures:

```text
DrivingMissionRecord
├── schema_version
├── builder_version
├── sections[]
│   ├── section_id
│   ├── preferred_lane_span
│   ├── allowed_lane_spans[]
│   └── exit_gate
├── final_goal
├── route_assignment_source
├── source_geometry_hash
└── mission_hash

MissionSnapshot
├── step_index
├── active_section_index / pending_gate_index
├── associated_lane_span
├── recovery_path_lane_ids
├── remaining_distance_m
├── route_progress_delta_m
├── instantaneous_completion / route_completion
├── reachable / success / reason
└── local_route_geometry
```

Geometry is reconstructed at reset from canonical source lanes. Records store
lane IDs and lane-local scalars, not Shapely objects. Dataclasses validate
identity, finiteness, ranges, ordering, uniqueness, and hashes.

### 7.2 Source annotation

Extend the offline route-map-matching/catalog pipeline to project the last valid
SDC pose and create route sections/gates from the preferred lane sequence plus
normalized topology. Consolidate ordinary unambiguous continuations into a
section; retain gates at mandatory movement boundaries and the final projected
goal. Build allowed lane spans by reverse reachability to the next gate under
traffic-rule-admissible successors and lane changes.

PG and Waymo adapters normalize source-specific successor, predecessor, and
lateral-neighbor fields. No source-specific runtime semantics are allowed.

### 7.3 Tracker and integration ownership

`ThesisScenarioEnv` creates the graph/tracker after source metadata injection
and before Rulebook/observation construction. It updates the tracker once at
the causal commit boundary. Extend `CausalSceneContext` with the immutable
mission snapshot, not mutable tracker state.

Rulebook memory drops `previous_route_s_m` after compatibility migration. The
progress component becomes a pure consumer of pre/post mission distances and
`delta_t`. The registry writer contract and golden traces change accordingly.

### 7.4 Legal routing and distance

Build a static directed graph at reset. Nodes represent lane spans split at
gates and relation boundaries. Edges represent forward traversal and legal lane
changes. Use non-negative metric lengths. Precompute downstream gate-to-goal
constants and reverse shortest-path distances to each gate. At a step, only
the current association/projection and cached lookup are required.

Tie-break order: lower metric distance, preferred-path agreement, fewer lane
changes, then stable lane/span ID. The final exact ordering is frozen in tests.

### 7.5 Gate crossing

Reuse the existing Rulebook swept-front-bumper construction and `0.05 m`
crossing tolerance. Gate objects include oriented world geometry, compatible
lane spans, vertical interval, and stable ID. Crossing is sequential and
idempotent; one transition may advance multiple zero-gap gates only if each
ordered swept intersection is independently satisfied, with a deterministic
upper bound preventing loops.

### 7.6 Observation migration

Create semantic and LiDAR successor schema IDs. Keep the existing tensor
group widths unless tests show an approved requirement cannot fit.

- replace preferred-polyline current coordinate with mission completion or
  normalized remaining distance according to the approved field revision;
- sample the active deterministic preferred/recovery path for local tokens;
- derive lane/heading errors from current mission association;
- rank controls/interactions using signed along-mission distance;
- keep masks, perception gates, causal history, and ADR-033 exclusion intact.

The two arms consume the same `MissionSnapshot.local_route_geometry`; they do
not independently re-route.

### 7.7 Rulebook migration

Migrate consumers explicitly rather than aliasing the old polyline:

- R4: pre/post remaining-distance reduction;
- signal/stop relevance: pending/reachable mission movement, replacing the
  one-hop ADR-051 predicate after equivalence tests;
- crosswalk/vehicle-yield zones: current recovery/preferred movement;
- RSS and lateral RSS: current compatible lane/planned local path;
- wrong-way: tangent of associated legal travel direction;
- wrong-carriageway: aligned/opposed surfaces relative to associated legal
  direction, not a far preferred projection;
- diagnostic route corridor: union of current allowed lane spans.

R1 and non-route R2/R3 formulas remain unchanged.

### 7.8 Errors, logging, and fallbacks

There is no native-navigation fallback. Static invalidity excludes a record
only under an approved policy. Runtime source mismatch is a typed data abort.
Agent-induced graph unreachability follows DEC-MSN-001. Programming/numerical
invariants are fatal. Every error carries scenario UID, source, mission hash,
gate, lane association, step, and stable reason.

### 7.9 Realistic alternatives

The rejected point-checkpoint and Euclidean-sum alternatives are simpler but
cannot encode road direction or legal reachability. Exact expert-path tracking
is realistic for imitation evaluation but overconstrains an RL planning task.
Free final-destination planning is realistic for a navigation stack but changes
the ordered benchmark. Directed gates plus legal recovery are the smallest
road-network abstraction that preserves the task while allowing valid lane use.

## 8. Traceability

| Requirement | Acceptance criteria | Planned implementation | Planned tests | Status |
|---|---|---|---|---|
| `REQ-MSN-001` | `AC-MSN-001`, `010` | `mission/types.py`, scenario records/frozen pipeline | `test_driving_mission_types.py`, `test_scenario_frozen.py` | Planned |
| `REQ-MSN-002` | `AC-MSN-002`, `003` | `mission/builder.py`, `topology.py` | `test_driving_mission_builder.py` | Planned |
| `REQ-MSN-003` | `AC-MSN-002`, `004` | `mission/gates.py`, `tracker.py` | `test_driving_mission_gates.py` | Planned |
| `REQ-MSN-004` | `AC-MSN-002`, `006` | source annotation, `mission/gates.py` | builder/gate/env tests | Planned |
| `REQ-MSN-005` | `AC-MSN-007`, `009` | `mission/tracker.py`, environment/context | tracker/env/vector tests | Planned |
| `REQ-MSN-006` | `AC-MSN-003`, `004` | `mission/topology.py`, `distance.py` | routing/distance tests | Planned |
| `REQ-MSN-007` | `AC-MSN-004`, `005` | Rulebook progress/transition/config | `test_rulebook_v2_progress.py` | Planned |
| `REQ-MSN-008` | `AC-MSN-006` | `thesis_scenario_env.py` | `test_thesis_scenario_env.py` | Planned |
| `REQ-MSN-009` | `AC-MSN-007`, `009` | observation builders/schemas/config | semantic/LiDAR tests | Planned |
| `REQ-MSN-010` | `AC-MSN-003`, `007` | Rulebook adapters/transition/geometry | Rulebook focused suite | Planned |
| `REQ-MSN-011` | `AC-MSN-008` | offline/runtime boundary | causality regression | Planned |
| `REQ-MSN-012` | `AC-MSN-001`, `010` | audit CLI/catalog/frozen pipeline | audit and frozen tests | Planned |
| `REQ-MSN-013` | `AC-MSN-006`, `007` | env info, runtime/eval/video artifacts | artifact/video tests | Planned |
| `REQ-MSN-014` | `AC-MSN-005`, `009` | tracker/env/vector runtime | numeric/reset/determinism tests | Planned |
| `REQ-MSN-015` | `AC-MSN-010` | checkpoint/replay/run identities | checkpoint/replay tests | Planned |

## 9. Test Strategy Defined Before Implementation

### Frozen mandatory matrix

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-MSN-001` | Unit | Mission record round trip/hash | valid multi-section record | exact identity and deterministic hash | 001, 014 |
| `TEST-MSN-002` | Unit | Invalid record rejection | empty IDs, bad spans/order, NaN/inf | reason-specific failure | 001, 014 |
| `TEST-MSN-003` | Unit | Source annotation determinism | same PG/Waymo static source twice | byte-identical record | 001, 012, 014 |
| `TEST-MSN-004` | Unit | Goal projection | terminal valid SDC pose mid-lane | goal at projected `s`, not lane end | 004 |
| `TEST-MSN-005` | Unit | Section consolidation | unambiguous chain then branch | gate only at mandatory boundary | 002, 007-decision |
| `TEST-MSN-006` | Unit | Allowed parallel lanes | reachable and non-reachable neighbors | only reachable spans allowed | 002, 006 |
| `TEST-MSN-007` | Unit | Forward gate crossing | swept bumper forward | advances exactly once | 003 |
| `TEST-MSN-008` | Unit | Invalid crossing variants | reverse/touch/spawn/out-of-order/wrong lane/level | no advance | 003, 004 |
| `TEST-MSN-009` | Unit | Multi-gate transition bound | two close gates | ordered finite deterministic advancement | 003, 014 |
| `TEST-MSN-010` | Unit | Shortest legal recovery | parallel lanes, merge, split | exact deterministic path/distance | 006 |
| `TEST-MSN-011` | Unit | Stable routing tie-break | equal-cost alternatives/permuted maps | bit-identical preferred result | 006, 014 |
| `TEST-MSN-012` | Unit | Cycles/roundabout | cyclic graph | finite correct distance, no loop | 006, 014 |
| `TEST-MSN-013` | Unit | Gate distance continuity | physical crossing at section boundary | no bookkeeping reward jump | 003, 007 |
| `TEST-MSN-014` | Unit | R4 nominal/reverse/clipping | signed distance changes | exact formula in `[-1,1]` | 007 |
| `TEST-MSN-015` | Regression | R4 cap invariance | same transition, caps 10/20 | bit-identical R4 | 007 |
| `TEST-MSN-016` | Unit | Completion monotonicity | advance then reverse | metric non-decreasing, R4 negative on reverse | 007, 008 |
| `TEST-MSN-017` | Integration | Success boundary | prior gates plus final forward crossing | success termination, completion 1 | 004, 008 |
| `TEST-MSN-018` | Integration | Unreachable boundary | ego state with no path | approved failure termination/reason | 006, 008 |
| `TEST-MSN-019` | Regression | Time limit | reachable unfinished mission at horizon | truncation only | 008 |
| `TEST-MSN-020` | Regression | Collision/out-of-road | existing physical fixtures | approved terminations preserved | 008 |
| `TEST-MSN-021` | Integration | Snapshot identity | all consumers in one step | same gate/path/distance/hash | 005, 009, 010, 013 |
| `TEST-MSN-022` | Unit | Semantic route tokens/masks | recovery path ends inside horizon | exact same shape, correct padding/mask | 009 |
| `TEST-MSN-023` | Unit | LiDAR navigation tokens | same mission snapshot | exact same path geometry and 6489 output | 009 |
| `TEST-MSN-024` | Regression | Rulebook latch exclusion | populated/empty Rulebook memory | policy route observation identical | 009, 011 |
| `TEST-MSN-025` | Integration | Rulebook route relevance | control/zone on pending vs foreign movement | only task-relevant item selected | 010 |
| `TEST-MSN-026` | Regression | ADR-051 coverage | route ends before one-hop signal | mission makes signal relevant | 010 |
| `TEST-MSN-027` | Regression | Alternate legal lane | non-preferred allowed lane | no preference-only R2/R3 cost | 006, 010 |
| `TEST-MSN-028` | Regression | Wrong-way/carriageway direction | associated parallel/opposing lanes | correct tangent/surface | 010 |
| `TEST-MSN-029` | Causality | Future SDC mutation | identical frozen mission/current, changed future | bit-identical runtime outputs | 011 |
| `TEST-MSN-030` | Integration | Reset and idempotence | alternating missions/repeated reads | no leak, no double advancement | 005, 014 |
| `TEST-MSN-031` | Integration | Subprocess vector isolation | multiple worker slots | deterministic isolated state | 014 |
| `TEST-MSN-032` | Compatibility | Old artifact rejection | old dataset/checkpoint/replay/schema | explicit incompatibility | 012, 015 |
| `TEST-MSN-033` | Catalog | Full trusted audit | exact 3,500 frozen source paths | complete report, zero silent mutations | 001, 012 |
| `TEST-MSN-034` | Integration | Frozen regeneration/replay | approved missions and same UIDs/splits | self-consistent hashes and replay | 012, 015 |
| `TEST-MSN-035` | Integration | Metrics/video | multi-gate episode | mission fields and true gate overlay | 013 |
| `TEST-MSN-036` | Performance | Step/reset timing | representative PG/Waymo panels | measured overhead recorded; no unapproved threshold | 013, 014 |
| `TEST-MSN-037` | Smoke | PG end to end | supported smoke profile | reset, rollout, learning, terminal artifact pass | all |
| `TEST-MSN-038` | Smoke | Waymo end to end | trusted representative scenario/panel | same end-to-end path passes | all |

Mandatory tests may be strengthened but not deleted, skipped, weakened, marked
expected-to-fail, or changed to match implementation without explicit approval.
Every discovered bug adds a regression test.

### Exact currently supported validation commands

Focused paths after tests exist:

```bash
uv run --no-sync python -m pytest -q tests/test_driving_mission_types.py tests/test_driving_mission_builder.py tests/test_driving_mission_gates.py tests/test_driving_mission_tracker.py
uv run --no-sync python -m pytest -q tests/test_rulebook_v2_progress.py tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_geometry.py tests/test_rulebook_v2_controls.py
uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py tests/test_assigned_route_observation.py tests/test_observation_v13_corrections.py tests/test_stacked_lidar_v2.py
uv run --no-sync python -m pytest -q tests/test_scenario_records.py tests/test_scenario_frozen.py tests/test_checkpoint_manifest_sidecar.py tests/test_transition_replay_persistence.py
```

Repository gates:

```bash
make rulebook-v2-check
make test
make lint
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/mission src/thesis_rl/scenarios src/thesis_rl/envs src/thesis_rl/rulebook tests"
make config
make config-gpu
make smoke
git diff --check
```

No global mypy command exists. Static checking is unavailable unless a bounded
module command is separately adopted; do not invent one.

The planned audit CLI does not exist yet. M1 must add and test it before using
the following planned invocation:

```bash
uv run --no-sync python -m thesis_rl.cli.scenarios.audit_driving_missions --data-root /scratch/e.respino/thesis-metadrive/data --frozen-index /scratch/e.respino/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json --output-dir docs/audits/driving_mission_full_catalog_<date>
```

The command must be read-only with respect to the data root and requires
explicit informed authorization because source files are pickle artifacts.

## 10. Milestones

### M0 — Approve scientific contract and promote documents

- Status: `AWAITING_DECISIONS`
- Dependencies: `DEC-MSN-001` through `DEC-MSN-009`.
- Tasks:
  - review all candidate requirements, formulas, defaults, and compatibility;
  - record explicit user decisions;
  - update proposed ADR-052 to Approved;
  - rename specification without `_UNDER_REVIEW`, set `APPROVED` and
    `Authoritative: YES`, record evidence/date;
  - update project index and links; set this plan `APPROVED`.
- Validation: link and metadata checks; `git diff --check`.
- Completion evidence: pending.

### M1 — Trusted full-catalog mission preflight

- Status: `NOT_STARTED`.
- Dependencies: M0 and `DEC-MSN-010` authorization.
- Expected files: audit CLI, focused tests, immutable JSON/CSV/Markdown report.
- Tasks:
  - implement a narrow read-only audit over exactly the frozen source paths;
  - normalize topology and attempt goal/section/gate/allowed-span construction;
  - report completeness and failures by source/split/arm/reason;
  - never overwrite source artifacts;
  - stop for user decision if any selected record fails.
- Tests: `TEST-MSN-001` through `006`, `011`, `012`, `033`.
- Completion evidence: pending.

### M2 — Mission records, topology, and offline builder

- Status: `NOT_STARTED`.
- Dependencies: successful/approved M1 result.
- Expected files: `src/thesis_rl/mission/{types,topology,builder,gates}.py`,
  source adapters, scenario records/catalog tests.
- Tasks: acceptance tests first; immutable schemas; source normalization;
  deterministic section/gate/goal construction; hashes and error codes.
- Tests: `TEST-MSN-001` through `009`, `029`.
- Completion evidence: pending.

### M3 — Runtime graph, distance, and tracker

- Status: `NOT_STARTED`.
- Dependencies: M2.
- Expected files: `mission/distance.py`, `mission/tracker.py`, context/env tests.
- Tasks: static graph, cached distances, association, legal recovery, gate
  state machine, snapshot, reset/idempotence/numeric checks.
- Tests: `TEST-MSN-007` through `013`, `016`, `030`, `031`.
- Completion evidence: pending.

### M4 — Environment success, completion, and boundaries

- Status: `NOT_STARTED`.
- Dependencies: M3 and DEC-MSN-001/006.
- Expected files: `thesis_scenario_env.py`, scene context, config, env tests.
- Tasks: install/update tracker at causal boundary; replace native authority;
  preserve native diagnostics temporarily; implement success/unreachable;
  retain time-limit truncation and physical terminations.
- Tests: `TEST-MSN-017` through `021`, `030`, `031`.
- Completion evidence: pending.

### M5 — R4 and Rulebook migration

- Status: `NOT_STARTED`.
- Dependencies: M3-M4.
- Expected files: progress, transition, types/memory/registry, route-dependent
  geometry/controls, golden traces, Rulebook tests.
- Tasks: acceptance tests first; replace `previous_route_s_m`; global config;
  migrate every listed route consumer; differential ADR-051 validation.
- Tests: `TEST-MSN-013` through `016`, `025` through `028` plus complete
  Rulebook regression suite.
- Completion evidence: pending.

### M6 — Observation and encoder compatibility migration

- Status: `NOT_STARTED`.
- Dependencies: M3-M5 and DEC-MSN-004.
- Expected files: observation schemas/builders/factory/config, checkpoint
  manifest, encoder compatibility declarations, tests.
- Tasks: new same-shape schema IDs; snapshot-based tokens/ranking/errors;
  checkpoint/replay rejection; preserve causality and Rulebook-latch exclusion.
- Tests: `TEST-MSN-021` through `024`, `029`, `032`.
- Completion evidence: pending.

### M7 — Dataset materialization, metrics, and diagnostics

- Status: `NOT_STARTED`.
- Dependencies: M1-M6 and explicit approval of any audit failure policy.
- Expected files: catalog/frozen/CLI, configs/manifests, runtime info/artifacts,
  video diagnostics, reports and tests.
- Tasks: regenerate annotations/index without changing UIDs/splits; embed hashes;
  add mission metrics; replace lane-start checkpoint markers with mission gates;
  validate frozen replay and panel identities.
- Tests: `TEST-MSN-001`, `003`, `004`, `033` through `036`.
- Completion evidence: pending.

### M8 — Full validation and reconciliation

- Status: `NOT_STARTED`.
- Dependencies: M1-M7.
- Tasks: focused tests, full suite, Rulebook gate, lint/format/config, PG and
  Waymo smoke, timing report, diff review, requirement/AC reconciliation,
  project index and final documentation.
- Tests: all `TEST-MSN-*`.
- Completion evidence: pending.

## 11. Progress And Findings Log

### 2026-08-02 — Preliminary repository and index analysis

- Read the selected ScenarioNet, Rulebook, semantic/LiDAR observation
  specifications, ADRs, templates, and ExecPlan rules.
- Verified current route/progress/observation/success call paths.
- Located the dataset at `/scratch/e.respino/thesis-metadrive/data/`.
- Read the frozen JSON index without source deserialization: 3,500 records,
  1,695 PG, 1,805 Waymo, no empty assigned routes, all eligible, 1-12 route
  lanes, 727 single-lane routes.
- Finding: implementation is feasible but must be a unified mission migration;
  R4-only work would preserve divergent task definitions.
- Finding: weighted sums of Euclidean pending-checkpoint distances are rejected
  in the candidate design in favor of legal graph distance through ordered
  gates.
- Blocking finding: the full topology/goal/gate preflight was not run because
  the exact source ScenarioDescriptions are pickle files. Loading requires
  explicit informed authorization; no workaround was attempted.
- Created candidate specification, proposed ADR, preliminary audit report, and
  this implementation-ready plan. No production code changed.
- Next step: user reviews/approves the nine scientific decisions and separately
  authorizes trusted-source deserialization; then start M0/M1 on the new branch.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-MSN-001` | ADR-004 stores one preferred lane sequence | Add sections, allowed spans, directed gates, and goal while preserving immutable/no-future semantics | Lane list cannot represent legal alternate lanes or unified success | Awaiting ADR-052 | all mission tests/docs |
| `DEV-MSN-002` | Rulebook v4.7 R4 uses preferred `delta_s` and actor cap | Remaining legal distance reduction and global reference speed | physical/scenario invariance and task consistency | Awaiting | progress tests/spec |
| `DEV-MSN-003` | ScenarioNet success uses native trajectory completion threshold | Ordered final mission gate | current task authority diverges from frozen route | Awaiting | env tests/spec |
| `DEV-MSN-004` | OBS-V1.3/OBS-LIDAR-V2.0 sample preferred polyline independently | Both consume one mission snapshot/recovery path | cross-consumer consistency | Awaiting | observation tests/spec |
| `DEV-MSN-005` | v4.11 uses one unambiguous terminal successor for controls | Pending mission reachability/movement | replace measured workaround with general task model | Awaiting; ADR-051 regression mandatory | Rulebook tests |
| `DEV-MSN-006` | ADR-043 checkpoint markers are lane starts | Draw true ordered mission gates | current label is diagnostic, not a milestone | Awaiting | video tests/ADR |

## 13. Files

### Files changed in this preparation task

| Path | Action | Purpose |
|---|---|---|
| `docs/audits/driving_mission_feasibility_2026-08-02/findings.md` | Added | Verified findings, data facts, risks, and blocked preflight |
| `docs/specifications/driving_mission_v1.0_specification_UNDER_REVIEW.md` | Added | Candidate scientific/behavioral contract |
| `docs/decisions/ADR-052-unified-driving-mission-contract.md` | Added | Proposed durable architecture/behavior decision |
| `docs/implementation/driving_mission_v1.0_exec_plan.md` | Added | Acceptance-first implementation plan |
| `docs/project_index.md` | Planned modification in this task | Register candidate/ADR/plan without granting authority |

### Planned production files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/mission/*.py` | Add | Mission schemas, topology, gates, distance, tracker |
| `src/thesis_rl/scenarios/records.py` | Modify | Persist mission identity/record |
| `src/thesis_rl/scenarios/{catalog,frozen,pipeline}.py` | Modify | Audit, serialize, freeze, and replay missions |
| `src/thesis_rl/cli/scenarios/` | Add/modify | Read-only audit and migration commands |
| `src/thesis_rl/scenarios/pg/`, Waymo adapters | Modify | Normalize topology and offline annotation |
| `src/thesis_rl/rulebook/v2/types.py` | Modify | Mission-aware static/runtime contracts |
| `src/thesis_rl/rulebook/v2/context/` | Modify | Build canonical mission graph/cache |
| `src/thesis_rl/rulebook/v2/components/progress.py` | Modify | New R4 formula |
| `src/thesis_rl/rulebook/v2/{memory,registry,transition}.py` | Modify | Remove duplicate progress state; migrate consumers |
| `src/thesis_rl/rulebook/v2/geometry/` | Modify | Mission lane/path/gate-aware geometry |
| `src/thesis_rl/contracts/causal_scene_context.py` | Modify | Publish immutable mission snapshot |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modify | Tracker owner, success, completion, boundaries |
| `src/thesis_rl/envs/observations/` | Modify | Snapshot-derived semantic/LiDAR route tokens |
| `src/thesis_rl/contracts/{observation_schema,checkpoint_manifest}.py` | Modify | New schema and compatibility identities |
| `conf/env/scenarionet.yaml`, `conf/obs/*.yaml` | Modify/add | Frozen mission and observation versions |
| runtime/evaluation/video artifact modules | Modify | Mission diagnostics, metrics, and true gate overlays |
| `tests/test_driving_mission_*.py` | Add | New mandatory core matrix |
| existing scenario/env/Rulebook/observation/checkpoint/replay/video tests | Modify/add | Integration and regressions |

The implementation session must refine this table before editing and keep it
aligned with the actual diff. Unrelated cleanup is prohibited.

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `jq` frozen-index metadata/statistics queries | `PASS` | 2026-08-02 | JSON-only, 3,500 records and statistics recorded in audit |
| Source ScenarioDescription full audit | `NOT_RUN` | 2026-08-02 | Pickle deserialization requires explicit informed authorization; follow-up is M1 planned audit command |
| Production focused tests | `NOT_RUN` | 2026-08-02 | No production code changed; execute per M2-M7 |
| `make rulebook-v2-check` | `NOT_RUN` | 2026-08-02 | Documentation-only preparation; mandatory after production migration |
| `make test` | `NOT_RUN` | 2026-08-02 | Documentation-only preparation; mandatory in M8 |
| `make smoke` | `NOT_RUN` | 2026-08-02 | Documentation-only preparation; PG and Waymo smoke mandatory in M8 |
| Documentation path/structure check | `PASS` | 2026-08-02 | All five changed/added paths exist; specification sections 1-18, ExecPlan sections 1-15, and unique REQ/AC/TEST registries were checked |
| `git diff --check` plus untracked-document whitespace scan | `PASS` | 2026-08-02 | No whitespace errors found after all document edits |

## 15. Final Reconciliation

### Current requirement status

All `REQ-MSN-001` through `REQ-MSN-015` and `AC-MSN-001` through
`AC-MSN-010` are `NOT_IMPLEMENTED`. Their contract and mandatory test mapping
are drafted, but the specification is not approved and no production code has
changed.

### Known limitations

- full source-data topology/goal/gate coverage is unknown until M1;
- the proposed global reference speed and boundary choices are not approved;
- exact normalized source lateral-neighbor semantics require the M1/M2 audit;
- no runtime or performance evidence exists for the proposed graph tracker;
- old artifacts are expected to be incompatible by design.

### Deferred required work

M0 through M8 are all required before experimental use. The source-data audit,
both source smoke paths, full tests, and final scientific reconciliation may not
be waived without explicit approval.

### Optional improvements

Future native PG mission persistence, exact discounted PBRS ablation, and
temporary-closure replanning remain out of scope.

### Readiness

The preliminary design and implementation plan are ready for user review and
branch handoff. They are not ready for implementation or experimental use
until the open decisions and trusted-pickle audit authorization are resolved.
