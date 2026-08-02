# Specification: Unified Driving Mission and Route Progress

## Metadata

- Feature: unified driving mission, ordered gates, route completion, and R4
- Specification ID: `DRIVING-MISSION-V1.0`
- Version: `1.0`
- Status: `APPROVED`
- Date: `2026-08-02`
- Supersedes: `NONE` until approved; if approved, it amends the route,
  progress, success, and route-observation subsets listed in Section 1.2
- Related specifications:
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`,
    amended by v1.2 and v1.3
  - `docs/specifications/rulebook_v4.7_specification.md`, amended by v4.8,
    v4.9, v4.10, and v4.11
  - `docs/specifications/observation_v1.3_specification.md`
  - `docs/specifications/observation_lidar_v2.0_specification.md`
- Related ADRs:
  - `docs/decisions/ADR-004-assigned-route-metadata-for-pg-and-waymo.md`
  - `docs/decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md`
  - `docs/decisions/ADR-051-traffic-control-route-successor-extension.md`
  - `docs/decisions/ADR-052-unified-driving-mission-contract.md` (approved)
- Related ExecPlan:
  `docs/implementation/driving_mission_v1.0_exec_plan.md`
- Authoritative: `YES`

## 1. Purpose And Context

This specification proposes one immutable driving mission and one canonical
runtime tracker for route progress, local route observation, Rulebook route
relevance, completion, and success. It replaces the current behavioral split
between frozen assigned-route lanes, a concatenated Rulebook polyline, local
observation waypoints, and MetaDrive native trajectory completion.

The scientific objective is to reward physically meaningful progress toward an
ordered driving task without requiring the ego to remain on one exact expert
centerline. A legal parallel lane or legal recovery path remains valid when it
can still reach the next mandatory movement. Illegal behavior remains governed
by the higher-priority Rulebook costs.

This is a project adaptation grounded in lane-level routing and route-completion
practice. No cited system directly defines this repository's record layout,
normalization constant, termination policy, or gate-crossing tolerances.

### 1.1 Verified defect being addressed

The current repository has no end-to-end intermediate checkpoint state. R4
measures `delta_s` on the preferred frozen polyline, observations sample that
same polyline locally, while success and reported completion use MetaDrive's
native reference trajectory. ADR-051 adds a one-hop route successor only for
traffic-control relevance. Those definitions can disagree for the same ego
state.

### 1.2 Proposed supersession scope after approval

Approval would supersede only the following behavioral subsets:

- ScenarioNet Integration v1.1 clauses that permit native ScenarioEnv
  navigation to define thesis progress, completion, or success;
- Rulebook v4.7 route-input, R4, route-projection-memory, route-relevance, and
  route-coordinate clauses, including the v4.11 one-hop control-membership
  amendment once equivalent mission tests pass;
- OBS-V1.3 fields whose semantics depend on the preferred `RoutePolyline`,
  including route samples, route-relative ranking, lane/heading error, and
  normalized current route coordinate;
- OBS-LIDAR-V2.0's 22-dimensional map-route navigation source semantics;
- ADR-004's lane-sequence representation, while preserving its immutable
  offline annotation and no-future-runtime principles;
- ADR-043's marker source, replacing diagnostic lane-start markers with true
  ordered mission-gate markers without changing the raster-only nature of the
  overlay.

Every unrelated clause remains unchanged.

## 2. Scope

### In Scope

- A versioned immutable mission record generated offline for PG and Waymo.
- Ordered route sections, allowed lane spans, directed intermediate gates, and
  one directed final goal.
- A deterministic legal lane-routing graph and recoverable alternate-lane
  semantics.
- A single environment-owned mission tracker and immutable per-transition
  snapshot.
- Remaining-distance progress, globally normalized bounded R4, completion,
  success, and mission-unreachable termination.
- Consistent semantic-v3-successor and stacked-LiDAR-successor route inputs.
- Migration of Rulebook route-dependent geometry and relevance consumers.
- Dataset, checkpoint, replay, manifest, diagnostics, and metric versioning.
- A read-only preflight over all 3,500 currently frozen scenarios before data
  migration.

### Out Of Scope

- Online destination inference from future SDC states.
- A global path planner that changes the ordered mission or chooses a new
  destination at runtime.
- Human-like lane preference, comfort, or strategic lane-change reward beyond
  Rulebook legality and mission reachability.
- An end-to-end perception system for lane topology or mission signs.
- Stop-at-destination parking semantics; the selected benchmark is a
  drive-through task.
- Changing R1, R2, or non-route-related R3 formulas.

### Optional Or Deferred

- Persisting native generator missions for a future PG dataset version.
- Legal overtaking exceptions for `wrong_carriageway` after ADR-049's
  diagnostic prerequisite is satisfied.
- Exact discounted potential-based shaping as a separate experimental reward
  variant.
- Dynamic replanning after a map closure or simulator-injected road blockage.

## 3. Terminology, Assumptions, And Preconditions

- **Preferred lane**: the offline annotated lane used to define the intended
  ordered movement, not the only legal ego lane.
- **Allowed lane span**: `(lane_id, start_s_m, end_s_m)` on which the ego may
  travel while retaining a legal route to the next ordered gate.
- **Route section**: an ordered task segment ending at one directed gate.
- **Directed gate**: an oriented cross-section of one or more compatible lane
  spans. Reaching it requires a forward swept-front-bumper crossing.
- **Pending gate**: the first gate not yet crossed in order.
- **Final goal**: the final directed gate, reached only after every prior gate.
- **Preferred path**: the deterministic lane sequence used for local navigation
  tokens and tie-breaking when multiple legal paths are equal.
- **Recovery path**: the deterministic shortest legal lane-graph path from the
  current associated lane/span to the pending gate.
- **Remaining distance `D`**: meters of legal graph travel to the final goal
  through every pending gate in order.
- **Completion**: a dimensionless `[0,1]` task metric derived from remaining
  distance and made non-decreasing for reporting.

Coordinates use canonical world XYZ meters, lane-local longitudinal `s` in
meters, headings in radians, speed in m/s, and the environment's committed
transition interval `delta_t` in seconds. Lane direction defines forward gate
orientation. Vertical compatibility uses the canonical Rulebook tolerance
unless an approved amendment changes it.

The source map is guaranteed upstream to contain the preferred annotated lane
IDs. Successor, predecessor, and lateral relations are runtime-independent
offline inputs that must pass the full preflight; absence is not silently
repaired from native navigation.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `DrivingMissionRecord` | Immutable task annotation | Ordered topology IDs and lane-local meters | Fixed before reset | Frozen dataset artifact; hash validated | Fail closed before control | Indirectly, through route tokens only |
| Canonical lane graph | Lane geometry and legal relations | World XYZ, lane-local `s` | Static per episode | PG/Waymo static adapter | Scenario not eligible | `NO`, except derived local tokens |
| Pre/post ego state | Committed physical transition | World pose, footprint, heading | `t`, `t+1` | Causal snapshot | Runtime data abort | Existing causal ego fields only |
| `delta_t` | Transition duration | seconds | finite and `>0` | Committed snapshots | Fatal transition error | `NO` |
| Mission snapshot | Canonical current mission state | Typed immutable record | Current committed step | `MissionTracker` | Fatal consistency error | Approved subset only |
| `v_ref` | Global progress reference speed | m/s | finite and `>0` | Frozen config | Startup failure | `NO` |

Prohibited at runtime:

- future SDC poses, velocities, lane associations, or replayed actions;
- MetaDrive native checkpoints, native reference-trajectory progress, or native
  destination success as a fallback authority;
- evaluator, curriculum, Rulebook result, violation latch, or future traffic
  information in the policy observation;
- online mutation of gate order, final destination, or mission provenance;
- silent map-only destination inference when an annotation is missing.

Complete SDC trajectories are permitted only during offline mission annotation,
as already established by ADR-004. Provenance and audit labels are never policy
features.

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `MissionSnapshot` | Canonical task state | Typed immutable record | One per committed step | Environment, Rulebook, observations | Idempotent within one step |
| `remaining_distance_m` | Legal distance through pending gates | finite meters, `>=0` while reachable | Current state | R4, metrics | Continuous across gate advancement within tolerance |
| `route_progress_delta_m` | `D_pre - D_post` | finite signed meters | Per transition | R4, diagnostics | Reverse/recovery regression may be negative |
| `progress_margin` | R4 | `[-1,1]` | Per transition | Rulebook scalarizer | Actor-cap invariant |
| `route_completion` | Maximum-so-far completion | `[0,1]` | Non-decreasing | Evaluation/logging | `1` on final success |
| `mission_success` | Ordered final gate reached | boolean | Terminal event | Environment | No point-radius shortcut |
| `mission_unreachable` | No legal path to pending gate | boolean + reason | Terminal event if approved | Environment | Never a time-limit truncation |
| Route tokens | Local preferred/recovery geometry | Existing tensor widths; versioned semantics | Forward ordered, masked | Semantic/LiDAR observations | No future dynamic state |

## 6. Functional Requirements

### REQ-MSN-001: Immutable Mission Record

Every selected scenario shall carry exactly one valid, versioned
`DrivingMissionRecord`, frozen before reset and included in dataset identity.
Missing, inconsistent, non-finite, or map-incompatible annotations fail closed.

### REQ-MSN-002: Ordered Sections And Allowed Lane Spans

The mission shall contain one or more ordered sections. Each section shall have
one preferred lane/span, one non-empty set of allowed lane spans, one directed
exit gate, and an explicit successor section or final goal. Allowed spans shall
include only traffic-rule-admissible lanes from which the next gate is reachable.

### REQ-MSN-003: Directed Sequential Gate Crossing

A gate is reached only when the ego's swept front bumper crosses it in the
forward direction, on a vertically compatible allowed lane/span, while that
gate is pending. Touching, spawning beyond, reverse crossing, crossing a future
gate out of order, or entering a point-radius neighborhood shall not advance
the mission.

### REQ-MSN-004: Directed Final Goal

The final goal shall use the same crossing primitive and shall be eligible only
after all intermediate gates. Success shall not require stopping or a speed
threshold. The current dataset's goal shall be the offline terminal valid SDC
pose projected to a compatible final lane/span, not the end of the whole lane.

### REQ-MSN-005: Single Mission State Owner

One environment-owned `MissionTracker` shall be the sole mutable owner of gate
index, remaining distance, completion history, and reachability. It shall
update once after the committed transition and publish one immutable snapshot
consumed by every downstream subsystem. Repeated reads within a step shall be
bit-identical and shall not mutate state.

### REQ-MSN-006: Legal Recovery And Remaining Distance

For the pending gate, the tracker shall associate the ego to a vertically and
directionally compatible current lane/span and compute a deterministic shortest
traffic-rule-admissible path. Legal lane changes and parallel lanes are allowed.
The ordered gate sequence and destination never change. Equal-cost paths use a
stable documented tie-break favoring the preferred path, then lane ID.

### REQ-MSN-007: R4 Route Progress

R4 shall be the clipped reduction in remaining legal distance divided by one
global frozen reference distance per step. Equal transitions shall produce the
same R4 regardless of ego configured speed cap, source, split, or scenario.
Gate-index advancement shall not create an artificial reward discontinuity.

### REQ-MSN-008: Completion, Success, And Boundaries

Reported `route_completion` shall be non-decreasing and mission-derived.
Success shall terminate the episode. Collision and physical out-of-road retain
their approved termination semantics. A time limit remains truncation.
`mission_unreachable`, if approved by `DEC-MSN-001`, is a task-failure
termination with a distinct reason and is never relabelled as truncation.

### REQ-MSN-009: Observation Consistency

Both selected observation arms shall receive local route geometry from the
same mission snapshot. Existing flat dimensions may remain unchanged, but
route-coordinate, lane-error, heading-error, control ranking, interaction
ranking, and route tokens shall have versioned mission semantics. No Rulebook
output or mission-evaluator-only flag shall become policy-visible.

### REQ-MSN-010: Rulebook Route Consistency

Every route-dependent Rulebook consumer shall use the mission graph/snapshot:
progress, relevant controls, conflict zones, vehicle-yield approach, crosswalk
relevance, RSS/lateral-RSS lane association, wrong-way tangent, and
wrong-carriageway aligned surfaces. Higher-priority legality costs shall remain
independent from a preference for one allowed lane.

### REQ-MSN-011: Causality

Runtime mission tracking shall depend only on frozen task metadata, static map
data, and current/past committed ego state. Tests shall prove that mutating
future SDC samples after annotation does not change any runtime mission output.

### REQ-MSN-012: Dataset Migration And Audit

Before materialization, a read-only audit shall process all 3,500 paths pinned
by the frozen index and report source-specific topology coverage, goal
projection, allowed-span construction, gate validity, graph reachability, and
reason-coded exclusions. Migration shall preserve scenario UIDs and split
membership. No invalid record may be silently replaced or moved across splits.

### REQ-MSN-013: Diagnostics And Metrics

Each episode shall report mission version/hash, active/final gate counts,
remaining distance, signed progress, instantaneous and maximum completion,
recovery-path use, lane association, reachability, success, native-navigation
differential diagnostics during migration, and reason-coded failures. These
values are diagnostic/evaluation metadata, not policy observation unless
explicitly listed by the observation specification.

### REQ-MSN-014: Timing, Reset, And Determinism

Reset shall clear all mission state and initialize the snapshot from the
causal reset ego state. Transition update order shall be deterministic under
vectorized execution and replay. State from one episode or worker slot shall
never leak into another. All public values shall reject NaN and infinity.

### REQ-MSN-015: Compatibility Identity

Mission schema, builder version, normalization constant, observation schema,
dataset selection hash, and relevant software versions shall participate in
run/checkpoint/replay compatibility. Existing checkpoints and replay buffers
shall be rejected rather than implicitly migrated.

## 7. Mathematical And Algorithmic Contract

### 7.1 Mission coordinate

For state `x` with pending gate index `k`, define:

```text
D(x, k) = d_legal(x, gate_k)
          + sum(j=k to K-1, d_fixed(gate_j, gate_(j+1)))
```

where `d_legal` is the shortest legal lane-graph distance from the ego's
associated lane/span to `gate_k`, and each downstream `d_fixed` is computed
once from the immutable mission graph. For the final gate, the sum is empty.
All edge weights are finite non-negative longitudinal meters. Legal lane-change
edges carry their deterministic geometric travel distance and never a negative
preference bonus.

The same downstream constants apply immediately before and after incrementing
`k`. At a crossing, the residual distance to the old gate and the initial
distance from that gate to the next one meet within the canonical geometric
tolerance; otherwise the mission is invalid.

### 7.2 Progress

For a committed transition:

```text
delta_D_t = D(x_t, k_t) - D(x_(t+1), k_(t+1))
R4_t = clip(delta_D_t / (v_ref * delta_t), -1, 1)
```

The proposed frozen default is:

```text
v_ref = 22.2222222222 m/s  # 80 km/h
```

`v_ref` is global across sources, scenarios, seeds, and actors. It is not the
local speed limit or current/configured ego speed. `delta_t` is the actual
finite positive committed interval. `R4` is not claimed to be Ng-style
policy-invariant potential shaping for a discounted return.

### 7.3 Completion

Let `D_0 > 0` be reset remaining distance:

```text
instantaneous_completion_t = clip(1 - D_t / D_0, 0, 1)
route_completion_t = max(route_completion_(t-1), instantaneous_completion_t)
```

The signed `delta_D_t` remains non-monotone so reversing or lengthening the
legal path can produce negative R4. The reported completion is monotone to
remain an interpretable coverage metric. On ordered final-goal success,
`D_t = 0` and `route_completion = 1`.

### 7.4 Gate crossing

For gate oriented with unit normal `n` pointing downstream, compute signed
front-bumper distances before and after the transition and the swept front
bumper geometry. A crossing requires:

```text
pre_signed >= -epsilon_gate
post_signed < -epsilon_gate
swept_front_bumper intersects gate_geometry
associated allowed lane/span is compatible
heading dot gate_forward_tangent > 0
```

The proposed `epsilon_gate` is `0.05 m`, matching the existing control-line
crossing tolerance. The exact footprint construction shall reuse the canonical
Rulebook swept-front-bumper primitive.

## 8. Applicability, State, And Timing

The mission is mandatory for every selected training, validation, test, smoke,
and evaluation scenario. No native-navigation fallback exists.

Reset order:

1. load and hash-check the frozen scenario and mission record;
2. build canonical lane geometry and legal graph;
3. validate the mission against that graph;
4. initialize the tracker from the reset ego snapshot;
5. publish the reset `MissionSnapshot`;
6. build Rulebook and observation adapters from that snapshot.

Step order:

1. capture the committed pre/post causal transition;
2. update gate crossing, lane association, recovery path, `D`, and completion
   exactly once;
3. publish the immutable post-transition mission snapshot;
4. evaluate Rulebook/R4 and done semantics from the same transition/snapshot;
5. rebuild the causal policy observation from the committed snapshot;
6. publish metrics and diagnostics.

Termination and truncation follow REQ-MSN-008. Tracker state is episode-local,
worker-slot-local, non-serializable into policy input, and fully reset.

## 9. Configuration

| Field | Type | Proposed default | Valid range | Meaning | Required | Frozen for experiments |
|---|---|---:|---|---|---|---|
| `mission.schema_version` | string | `driving_mission_v1` | exact supported value | Frozen record schema | `YES` | `YES` |
| `mission.builder_version` | string | `mission-builder-v1` | non-empty supported value | Offline derivation implementation | `YES` | `YES` |
| `mission.progress_reference_speed_mps` | float | `22.2222222222` | finite `>0` | R4 normalization | `YES` | `YES` |
| `mission.gate_crossing_epsilon_m` | float | `0.05` | finite `[0,0.25]` | Directed crossing tolerance | `YES` | `YES` |
| `mission.unreachable_policy` | enum | `terminate_failure` | approved values only | Boundary on lost reachability | `YES` | `YES` |
| `mission.native_navigation_diagnostics` | bool | `true` during migration | boolean | Differential metrics only | `NO` | `YES` for a run |

Invalid or unsupported values fail during startup. No per-scenario override of
`progress_reference_speed_mps` is allowed.

## 10. Errors, Logging, And Diagnostics

Offline invalidity uses stable reason codes including missing preferred lane,
invalid lane span, invalid terminal projection, missing legal topology,
non-contiguous gate order, unreachable gate, ambiguous unsupported movement,
vertical incompatibility, and non-finite geometry. Counts shall be reported by
source, split, arm, and reason.

Runtime static mission mismatch fails before control. A post-reset graph or
numeric inconsistency is a typed scenario-data abort with full forensic
diagnostics, not a silent fallback. `mission_unreachable` caused by the ego's
current task state follows the approved termination policy rather than becoming
a data abort. Programming invariants remain fatal.

Required per-step or per-episode diagnostics are listed in REQ-MSN-013. Native
route completion is diagnostic-only during migration and shall use an explicit
`native_` prefix.

## 11. Reproducibility And Compatibility

Mission construction is deterministic for fixed source bytes, builder version,
and configuration. Stable sorting and tie-breaks must not depend on mapping
iteration order. The mission record/hash is embedded in frozen dataset
artifacts and run metadata.

The migration changes dataset identity and route-related observation semantics.
It therefore requires new observation schema versions even when tensor shapes
remain `3009` and `6489`. Existing observation normalizers, checkpoints,
replay buffers, golden traces, evaluation results, and resumed runs are
incompatible. Historical modes may remain available only for explicit
reproducibility; no implicit conversion is permitted.

Scenario UIDs and approved split membership remain fixed. A record that fails
the new preflight is reported and blocks materialization until the user approves
an exclusion/replacement policy.

## 12. Acceptance Criteria

### AC-MSN-001: One Frozen Mission Per Scenario

- Given: the frozen 3,500-scenario index and trusted source files.
- When: the offline audit and mission builder run twice.
- Then: every accepted UID has one byte-identical valid mission; all failures
  have stable reason codes and no source file or split membership is modified.
- Related requirements: `REQ-MSN-001`, `REQ-MSN-012`, `REQ-MSN-014`.

### AC-MSN-002: Gate Semantics

- Given: forward, reverse, touch-only, spawn-beyond, wrong-lane, wrong-level,
  and out-of-order deterministic transitions.
- When: the tracker evaluates each transition.
- Then: only the forward compatible crossing of the pending gate advances the
  section exactly once.
- Related requirements: `REQ-MSN-002`, `REQ-MSN-003`, `REQ-MSN-004`.

### AC-MSN-003: Legal Alternate Lane

- Given: two legal parallel lanes that both reach the pending gate.
- When: the ego changes to the non-preferred lane and advances.
- Then: the mission remains reachable, remaining legal distance changes
  consistently, R4 reflects that change, and no preference-only Rulebook cost
  activates.
- Related requirements: `REQ-MSN-002`, `REQ-MSN-006`, `REQ-MSN-007`,
  `REQ-MSN-010`.

### AC-MSN-004: Ordered Distance Continuity

- Given: a multi-gate route and a transition crossing one gate.
- When: remaining distance is evaluated before and after the crossing.
- Then: the only R4 contribution is physical progress within tolerance; no
  checkpoint-removal jump occurs.
- Related requirements: `REQ-MSN-003`, `REQ-MSN-006`, `REQ-MSN-007`.

### AC-MSN-005: Normalization Invariance

- Given: two otherwise identical transitions with configured ego caps of
  10 m/s and 20 m/s.
- When: R4 is evaluated.
- Then: both margins are bit-identical and finite in `[-1,1]`.
- Related requirements: `REQ-MSN-007`, `REQ-MSN-014`.

### AC-MSN-006: Completion And Boundaries

- Given: progress, reversal, final crossing, collision, unreachable mission,
  and time-limit cases.
- When: environment boundaries and metrics are produced.
- Then: completion is non-decreasing, signed R4 may be negative, success is
  exactly final ordered crossing, collision/unreachable/success terminate, and
  only the time limit truncates.
- Related requirements: `REQ-MSN-008`, `REQ-MSN-013`.

### AC-MSN-007: Cross-Consumer Identity

- Given: one committed mission snapshot.
- When: R4, observation, Rulebook relevance, success, metrics, and video
  diagnostics consume it.
- Then: every consumer reports the same active gate, associated lane/path, and
  mission coordinate; native navigation cannot alter an output.
- Related requirements: `REQ-MSN-005`, `REQ-MSN-009`, `REQ-MSN-010`.

### AC-MSN-008: Causality

- Given: two source scenarios with identical frozen mission/static/current and
  past states but different future SDC trajectories.
- When: runtime outputs are produced.
- Then: mission snapshots, R4, observations, success, and Rulebook inputs are
  bit-identical.
- Related requirements: `REQ-MSN-011`.

### AC-MSN-009: Reset And Vectorization

- Given: alternating missions in single-process and deterministic subprocess
  environments.
- When: episodes reset and step.
- Then: no gate, distance, completion, path, or diagnostic state leaks across
  episodes or worker slots, and repeated seeded runs match.
- Related requirements: `REQ-MSN-005`, `REQ-MSN-014`.

### AC-MSN-010: Compatibility Rejection

- Given: an old observation checkpoint, replay buffer, dataset artifact, or
  mission record.
- When: it is loaded by the new path.
- Then: loading fails with the exact incompatible identity; no implicit
  migration occurs.
- Related requirements: `REQ-MSN-012`, `REQ-MSN-015`.

## 13. Required Validation Categories

- nominal and boundary behavior: required;
- invalid and incomplete inputs: required;
- masks, padding, state, reset, and update order: required;
- termination and truncation: required;
- deterministic seeds and reproducibility: required;
- numerical stability, NaN, and infinity: required;
- compatibility and migration: required;
- absence of future and privileged information: required;
- upstream, downstream, and end-to-end integration: required;
- regressions for current route-success, R4 projection, ADR-043, and ADR-051:
  required;
- read-only full-catalog preflight and representative PG/Waymo runtime smoke:
  required.

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or proposed decision |
|---|---|---|
| `REQ-MSN-001` | `AC-MSN-001`, `AC-MSN-010` | ADR-004; ADR-052 |
| `REQ-MSN-002` | `AC-MSN-002`, `AC-MSN-003` | Lanelet2 routing-graph concept; project adaptation |
| `REQ-MSN-003` | `AC-MSN-002`, `AC-MSN-004` | Existing canonical swept crossing; project decision |
| `REQ-MSN-004` | `AC-MSN-002`, `AC-MSN-006` | `DEC-MSN-002`, `DEC-MSN-006` |
| `REQ-MSN-005` | `AC-MSN-007`, `AC-MSN-009` | Proposed ADR-052 |
| `REQ-MSN-006` | `AC-MSN-003`, `AC-MSN-004` | Lanelet2; project tie-break adaptation |
| `REQ-MSN-007` | `AC-MSN-004`, `AC-MSN-005` | nuPlan/CARLA progress practice; `DEC-MSN-003` |
| `REQ-MSN-008` | `AC-MSN-006` | Gymnasium boundary contract; `DEC-MSN-001` |
| `REQ-MSN-009` | `AC-MSN-007`, `AC-MSN-009` | OBS-V1.3, OBS-LIDAR-V2.0; `DEC-MSN-004` |
| `REQ-MSN-010` | `AC-MSN-003`, `AC-MSN-007` | Rulebook v4.7-v4.11; ADR-052 |
| `REQ-MSN-011` | `AC-MSN-008` | ADR-004, ADR-022, ADR-033 |
| `REQ-MSN-012` | `AC-MSN-001`, `AC-MSN-010` | ScenarioNet v1.1-v1.3; `DEC-MSN-008` |
| `REQ-MSN-013` | `AC-MSN-006`, `AC-MSN-007` | EVAL-PROTOCOL; project decision |
| `REQ-MSN-014` | `AC-MSN-005`, `AC-MSN-009` | Existing deterministic runtime contract |
| `REQ-MSN-015` | `AC-MSN-010` | Existing checkpoint/replay identity contract |

## 15. Open Decisions And Limitations

| ID | Question | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|
| `DEC-MSN-001` | What happens when the next ordered gate becomes graph-unreachable? | Terminate task failure / continue with zero progress / truncate | Terminate with `mission_unreachable`; it is agent/task state, not an external time limit | Termination, metrics, replay boundaries | `APPROVED 2026-08-02` |
| `DEC-MSN-002` | How is the current dataset's final goal derived? | Terminal valid SDC projection / whole final-lane end / source-specific native goal | Terminal valid SDC projection for both current sources; preserves ADR-004's identical semantics and does not invent travel beyond the recorded task | Dataset annotation and success | `APPROVED 2026-08-02` |
| `DEC-MSN-003` | Which R4 normalization is frozen? | Global 80 km/h / empirical percentile / scenario cap | Global `22.2222222222 m/s`; current physical global cap, stable across scenarios | Reward scale and comparability | `APPROVED 2026-08-02` |
| `DEC-MSN-004` | Are unchanged tensor shapes allowed with new semantics? | New schema IDs with same shapes / widen tensors / overwrite existing schema | New schema IDs with same shapes where sufficient; reject old checkpoints/replays | Observation and experiment compatibility | `APPROVED 2026-08-02` |
| `DEC-MSN-005` | How are alternate lanes handled? | Exact preferred lane only / unordered destination routing / fixed gates plus legal recovery | Keep ordered gates; allow deterministic shortest legal recovery to the pending gate | Task realism and R4 | `APPROVED 2026-08-02` |
| `DEC-MSN-006` | What constitutes final success? | Point radius / directed crossing / crossing plus stop | Directed compatible crossing after prior gates; no stop condition | Success semantics | `APPROVED 2026-08-02` |
| `DEC-MSN-007` | Where are intermediate gates placed? | Lane starts / lane ends / only mandatory movement boundaries | At directed exits of route sections, consolidating non-decision continuation lanes and retaining mandatory movement boundaries | Mission density and annotation | `APPROVED 2026-08-02` |
| `DEC-MSN-008` | How are invalid missions migrated? | Silent replacement / exclude and rebalance / block and report | Preserve UIDs/splits and block materialization until a reason-coded audit is reviewed and an explicit exclusion policy is approved | Dataset identity and statistical validity | `APPROVED 2026-08-02` |
| `DEC-MSN-009` | Is R4 presented as potential-based shaping? | Claim PBRS / use exact discounted shaping / treat as bounded primary progress objective | Do not claim PBRS; retain the direct distance-reduction R4 as a Rulebook objective | Thesis justification | `APPROVED 2026-08-02` |

Known limitations if the recommendations are approved:

- route validity is defined by available map topology and traffic-rule
  metadata, not by unmodelled temporary closures;
- source maps may not encode every jurisdictional lane-use restriction;
- the fixed global reference speed is a normalization scale, not a claim that
  80 km/h is locally safe or legal;
- shortest legal distance does not model human strategic lane preference;
- the full 3,500-source-file preflight remains not run until trusted pickle
  deserialization is explicitly authorized.

## 16. References

- Poggenhans et al., *Lanelet2: A High-Definition Map Framework for the Future
  of Automated Driving*, 2018,
  <https://www.mrt.kit.edu/z/publ/download/2018/Poggenhans2018Lanelet2.pdf>:
  traffic-rule-dependent routing graphs, lane-change edges, and maneuverable
  adjacent lanelets.
- Motional, *nuPlan Metrics Description*,
  <https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md>:
  expert-route lane/lane-connector progress and integrated progress ratio.
- CARLA Leaderboard 2.0, *Evaluation Criteria*,
  <https://leaderboard.carla.org/evaluation_v2_0/>: route completion as
  percentage of route distance completed.
- Bansal et al., *ChauffeurNet: Learning to Drive by Imitating the Best and
  Synthesizing the Worst*, 2018, <https://arxiv.org/abs/1812.03079>:
  intended-route input and progress training signal.
- Jaeger et al., *CaRL: Learning Scalable Planning Policies with Simple
  Rewards*, 2025, <https://arxiv.org/abs/2504.17838>:
  route-completion-centered reward design in driving.
- Ng, Harada, and Russell, *Policy Invariance Under Reward Transformations*,
  1999,
  <http://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf>:
  exact discounted potential-based shaping condition.
- ADR-004: immutable offline assigned route and no future SDC at runtime.
- ADR-043: current lane-start markers are diagnostic only.
- ADR-051: measured one-hop route-end/control-relevance gap.

## 17. Implementation Handoff Checklist

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, units, frames, ranges, and masks.
- [x] Prohibited future, privileged, leaked, and diagnostic-only data is listed.
- [x] Formulas, algorithms, applicability, and fallbacks are unambiguous subject
  to the open decisions.
- [x] State, timing, reset, termination, and truncation behavior is defined.
- [x] Configuration fields and proposed scientifically frozen defaults are identified.
- [x] Errors, diagnostics, reproducibility, compatibility, and migration are covered.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected.
- [x] Scientific sources, project adaptations, and proposed decisions are distinct.
- [x] No material decision remains open.
- [x] Known limitations are explicit and do not hide missing requirements.

## 18. Approval Record

- Approved by: thesis repository maintainer
- Approval date: `2026-08-02`
- Approval evidence: explicit user approval in the Codex task dated `2026-08-02`, covering `DEC-MSN-001` through `DEC-MSN-009`
- Approval notes: M1 remains required; any invalid selected mission blocks materialization pending a separate explicit exclusion or replacement policy.
- Repository path after approval:
  `docs/specifications/driving_mission_v1.0_specification.md`
- Project index updated: `YES`, as a non-authoritative candidate only
