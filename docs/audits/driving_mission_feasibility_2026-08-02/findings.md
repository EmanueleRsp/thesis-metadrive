# Unified Driving Mission Feasibility Audit

## Metadata

- Audit ID: `DRIVING-MISSION-FEASIBILITY-2026-08-02`
- Date: `2026-08-02`
- Repository branch: `scenarionet-implementation`
- Scope: read-only preliminary analysis for `DRIVING-MISSION-V1.0`
- Dataset root: `/scratch/e.respino/thesis-metadrive/data/`
- Frozen index:
  `/scratch/e.respino/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json`
- Production code changed: `NO`

## Executive Finding

The implementation is feasible, but it is a cross-cutting mission-contract
migration rather than an isolated R4 formula change. The repository currently
uses at least four partially independent route concepts:

1. the frozen `assigned_route_lane_ids` task annotation;
2. the Rulebook `RoutePolyline` and `previous_route_s_m` progress memory;
3. semantic and LiDAR look-ahead points sampled from that polyline;
4. MetaDrive `TrajectoryNavigation.route_completion` for success and reported
   completion.

No end-to-end ordered mission checkpoint state exists. The points labelled as
planned checkpoints in evaluation videos are lane-start diagnostics only. A
single immutable driving mission and one environment-owned tracker can replace
these divergent semantics without requiring a new external dependency.

## Evidence Classification

- `VERIFIED`: inspected directly in repository code, documentation, or the
  JSON frozen index.
- `INFERRED`: a design consequence derived from verified interfaces, requiring
  deterministic tests or the full data audit before approval.
- `NOT_RUN`: validation could not safely be executed in this session.

## Verified Repository Findings

### Current dataset and task annotation

- `VERIFIED`: `ScenarioRecord` persists only
  `assigned_route_lane_ids` and `assigned_route_source` for task navigation
  (`src/thesis_rl/scenarios/records.py`).
- `VERIFIED`: both PG and Waymo routes in the current frozen population have
  identical source semantics: they are produced offline by SDC map matching,
  then consumed as immutable task metadata, as required by ADR-004.
- `VERIFIED`: `TaskRouteRecord` contains a preferred ordered lane sequence,
  provenance, adapter version, and source-geometry hash. It does not contain
  allowed parallel lanes, lane spans, ordered gates, or a final-goal region
  (`src/thesis_rl/rulebook/v2/types.py`,
  `src/thesis_rl/rulebook/v2/context/task_route.py`).
- `VERIFIED`: `RouteLaneRecord` carries polygon, centerline, and successor lane
  IDs, but no normalized lateral-adjacency or lane-change relation
  (`src/thesis_rl/rulebook/v2/geometry/lanes.py`).

### Current progress and success

- `VERIFIED`: R4 projects pre/post ego positions onto one concatenated
  preferred `RoutePolyline`, computes `delta_s`, and normalizes it by the
  current ego's configured speed cap times `delta_t`. Its only mission-related
  memory is `previous_route_s_m`
  (`src/thesis_rl/rulebook/v2/components/progress.py`).
- `VERIFIED`: the optional `route_outside_fraction` is diagnostic only. R4 can
  therefore credit projected progress while the ego follows a legal parallel
  lane outside the preferred corridor.
- `VERIFIED`: thesis success and the exported `route_completion` metric still
  depend on MetaDrive `TrajectoryNavigation.route_completion` and its native
  reference trajectory, not the frozen assigned route
  (`src/thesis_rl/envs/thesis_scenario_env.py::_is_thesis_success` and
  `_attach_route_metrics`).
- `VERIFIED`: time-limit exhaustion is a truncation; collision, physical
  out-of-road, and success are terminations. The proposed mission contract must
  preserve that distinction.

### Current observations and Rulebook consumers

- `VERIFIED`: semantic v3 exposes ten preferred-route samples at 5 m spacing,
  approximately 50 m of local look-ahead, and the normalized projection
  coordinate `route_s / route_length`. It exposes no gate index, pending-gate
  state, or remaining mission distance
  (`src/thesis_rl/envs/observations/causal_semantic.py`).
- `VERIFIED`: the LiDAR navigation block likewise exposes ten fixed-spacing
  points from `RoutePolyline`, plus lateral and heading error
  (`src/thesis_rl/envs/observations/assigned_route.py`).
- `VERIFIED`: traffic-control selection, interaction geometry, wrong-way
  tangents, wrong-carriageway surfaces, RSS associations, and route relevance
  are all derived directly or indirectly from the fixed route lane sequence or
  polyline.
- `VERIFIED`: ADR-051's one-hop control successor is a targeted workaround for
  routes ending before a relevant junction. A mission graph with ordered gates
  can subsume that workaround, but it must preserve its verified behavior until
  the replacement passes the same catalog checks.
- `VERIFIED`: ADR-043's `lane_start_points_xyz` are diagnostic markers only and
  are not mission checkpoints.

## Frozen Index Audit

The following facts were obtained with `jq` from JSON only; no source scenario
was deserialized.

| Measure | Result |
|---|---:|
| Frozen records | 3,500 |
| Frozen source paths | 3,500 |
| PG records | 1,695 |
| Waymo records | 1,805 |
| Rulebook-eligible records | 3,500 |
| Empty assigned routes | 0 |
| Minimum / maximum lane count | 1 / 12 |
| Mean lane count | 3.5443 |
| Single-lane routes | 727 |
| Minimum / maximum recorded route length | 10.0057 m / 593.0423 m |
| Mean recorded route length | 139.6730 m |
| Selection hash | `f5651c89183a216c78ffa434618f98ed33c861fd79a324bf1ce71be58568f7b1` |

The source split is 1,695 PG annotations labelled
`pg_sdc_offline_task_annotation` and 1,805 Waymo annotations labelled
`waymo_sdc_offline_task_annotation`. The selected split contains 2,200 train,
300 empirical validation, 700 empirical test, and 300 stratified-test records.

## Proposed Technical Direction

### Immutable annotation

Replace the lane-list-only contract with a versioned `DrivingMissionRecord`
that contains ordered route sections, preferred lanes, admissible lane spans,
directed exit gates, a directed final goal, provenance, builder version, and a
source-geometry hash. Geometry should be rebuilt deterministically at reset;
the frozen record should retain stable topology identifiers and scalar lane
coordinates rather than duplicate map polygons.

For the existing 3,500-record population, both sources should retain identical
task semantics: the preferred lane sequence and terminal goal are derived
offline from the complete SDC trajectory, then frozen. Runtime code must never
read future SDC states. Future use of a native PG generator destination is a
separate dataset-version decision, not an implicit fallback.

### One tracker and one snapshot

An environment-owned `MissionTracker` should update exactly once per committed
transition and publish an immutable `MissionSnapshot` containing:

- active ordered section and next pending gate;
- current associated lane/span and selected deterministic recovery path;
- signed remaining legal road distance to the final goal through pending gates;
- signed per-transition progress;
- instantaneous and maximum-so-far completion;
- reachability, final-goal crossing, and diagnostic reason codes;
- local future route tokens for both observation arms.

R4, success, termination, observations, Rulebook transition construction,
metrics, and video diagnostics should consume that same snapshot. MetaDrive's
native navigation completion should remain diagnostic during migration and be
removed as a thesis success authority after equivalence/non-equivalence has
been measured.

### Checkpoint and goal semantics

Intermediate milestones should be directed lane gates, not point-radius
targets. A gate is reached only by a forward swept-front-bumper crossing on a
compatible allowed lane/span and in order. This encodes position, direction,
lane compatibility, and crossing order without an arbitrary Euclidean radius.
The final goal should use the same directed-crossing primitive; no stop or
low-speed condition is proposed for this drive-through benchmark.

Allowed lane spans should include legal parallel lanes from which the next
ordered gate remains reachable. The ego receives neither an arbitrary penalty
nor a reward veto merely for using another legal lane. Instead, its remaining
distance is recomputed on the legal routing graph. Illegal occupancy or
movements remain the responsibility of R2/R3 costs.

### R4 candidate

Let `D_t` be the deterministic shortest traffic-rule-admissible road distance
from the current ego state to the final goal through all still-pending ordered
gates. The proposed progress is:

```text
delta_D_t = D_t - D_(t+1)
R4_t = clip(delta_D_t / (v_ref * delta_t), -1, 1)
```

with one global, experiment-frozen `v_ref`, recommended as `22.222222 m/s`
(80 km/h), rather than an actor/scenario speed cap. This makes equal physical
progress have equal value across actor configurations. The gate-to-gate
downstream distance must be included before and after gate advancement so the
coordinate is continuous and crossing a bookkeeping boundary creates no
artificial reward.

This is distance-to-go reduction, not the supervisor's tentative weighted sum
of Euclidean distances to `N` pending checkpoints. A Euclidean sum double
counts downstream geometry, changes discontinuously when a checkpoint is
removed, and can reward geometric shortcuts across non-drivable space. The
legal-graph distance preserves ordered task constraints and telescopes under
the undiscounted sum. It must not be described as policy-invariant potential
shaping unless the actual discounted shaping term satisfies the Ng et al.
`gamma * Phi(s') - Phi(s)` form; R4 is instead a bounded primary progress
objective in the project Rulebook.

## Feasibility Risks And Required Preflight

| Risk | Current evidence | Required resolution |
|---|---|---|
| Source topology coverage | Lane successors exist; source-specific lateral metadata is known to be present but is not normalized in `RouteLaneRecord` | Audit every source scenario and quantify missing, asymmetric, ambiguous, and invalid relations |
| Goal derivation | Preferred routes are frozen, but no terminal lane coordinate is persisted | Re-project terminal valid SDC pose offline and verify lane/span/gate validity |
| Gate construction | Lane centerlines and topology exist | Verify one and only one deterministic ordered gate sequence or exclude with an explicit reason |
| Recovery-path determinism | No routing graph abstraction exists | Define stable costs/tie-breaks and test cycles, parallel lanes, merges, splits, and roundabouts |
| Observation compatibility | Shapes can remain unchanged, semantics cannot | Introduce new observation schema IDs; reject old checkpoints/replays |
| Rulebook migration | Many components use preferred-route coordinates | Migrate consumers milestone-by-milestone and retain differential diagnostics |
| Runtime cost | Graph distance may be expensive if rebuilt per step | Build static graph once per reset and cache gate-to-go distances; benchmark step timing |

## Validation Not Run

`NOT_RUN`: the full 3,500-scenario topology/goal/gate audit. The source
ScenarioDescription files are pickle artifacts. Loading a pickle can execute
arbitrary code, so a generic deserialization command was not authorized in
this session. No workaround was used.

The implementation branch must first add a narrow, read-only audit command,
pin the exact 3,500 paths from the frozen index, record the trusted provenance
of those project artifacts, and execute it only after explicit informed user
authorization. The audit must write new reports outside the dataset source
directories and must never overwrite source artifacts.

## Literature And External Contract Alignment

- [Lanelet2](https://www.mrt.kit.edu/z/publ/download/2018/Poggenhans2018Lanelet2.pdf)
  represents road routing as traffic-rule-dependent routing graphs,
  including possible lane changes and maneuverable adjacent lanelets. This
  supports the proposed legal graph and allowed-lane-set abstraction, but not
  this project's exact record schema.
- [nuPlan](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md)
  measures ego progress along an expert route represented by lanes and lane
  connectors. [CARLA Leaderboard](https://leaderboard.carla.org/evaluation_v2_0/)
  reports percentage of route distance completed. These support route-coordinate
  completion rather than a sum of Euclidean checkpoint distances.
- [ChauffeurNet](https://arxiv.org/abs/1812.03079) provides the intended route
  as an input and includes progress as an auxiliary learning signal.
  [CaRL](https://arxiv.org/abs/2504.17838) reports strong driving results with a
  simple reward centered on route completion and infractions. These establish
  that route progress is a standard learning signal, while the precise
  normalization and mission-state integration remain project decisions.
- [Ng, Harada, and Russell](http://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
  establish the exact discounted potential-shaping
  form. The proposed R4 must not claim that theorem for an unadjusted
  `D_t - D_(t+1)` term under `gamma < 1`.

## Conclusion

The design is realistic and implementable with current map and route assets.
The main uncertainty is data coverage, not architectural plausibility. No
production implementation should begin until the candidate specification's
observable choices are explicitly approved and the full source-data preflight
has either passed or produced an approved exclusion policy.
