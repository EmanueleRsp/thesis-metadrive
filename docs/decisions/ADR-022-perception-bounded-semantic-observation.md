# ADR-022: Perception-Bounded Semantic Observation

- Status: Approved
- Date: 2026-07-21
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-21
- Supersedes: `docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md`
  only for the selected semantic-observation and encoder version
- Affected specifications:
  - `docs/specifications/observation_v1.2_specification.md`, ID `OBS-V1.2`
  - `docs/specifications/encoder_v1.1_specification.md`, ID `ENC-V1.1`
- Affected ExecPlan:
  `docs/implementation/perception_bounded_semantic_observation_v1.2_exec_plan.md`

## Context

OBS-V1.1 is temporally causal but assumes ideal global current-state
perception. It can expose actors and controls that are outside a plausible
sensor line of sight. It also publishes Rulebook-maintained temporal values to
the policy and has implementation findings that do not conform to its own
schema contract.

The thesis remains a map-assisted, post-perception planning study; it does not
expand into end-to-end RGB perception or learned multi-object tracking. The
replacement must preserve the no-future-data boundary, frozen assigned route,
canonical Rulebook geometry, bounded object-centric representation, and
deterministic reproducibility.

## Decision

1. Adopt OBS-V1.2 and ENC-V1.1 as a new incompatible semantic-observation and
   encoder pair. OBS-V1.1/ENC-V1.0 remain available only for historical
   reproducibility.
2. Make dynamic and live-static policy tokens perception-bounded by one
   240-beam, 50 m, planar, 360-degree Bullet LiDAR sweep. Only real first-hit
   detections may update the semantic track cache or enter token ranking.
3. Do not introduce RGB rendering or a learned detector/tracker. Conditional
   on a LiDAR detection, the semantic tracker is an explicitly idealized
   map-assisted tracker. It may be degraded only by the approved deterministic
   measurement-noise/dropout model after its parameters have been separately
   approved.
4. Model traffic-signal visibility with a symbolic forward semantic camera:
   FOV/range gate plus a 3D occlusion ray to a known light-head anchor. The
   light colour is available only after visibility succeeds and is otherwise
   `unknown`. If the required collider preflight fails, the implementation must
   stop at the approval gate; it must not silently switch to V2I/SPaT.
5. Replace the policy-visible Rulebook temporal vector with a 21-frame,
   perception-derived compliance history and a three-value ego-owned yellow
   onset memory. The Rulebook formula and its ground-truth onset evaluation do
   not change.
6. Keep one planar LiDAR layer. Multi-layer LiDAR is explicitly out of scope.
   The selected scenario domain is treated as single-level for this observation
   contract; 2.5D compatibility checks still prevent cross-level map/actor
   associations.
7. Correct the identified existing-contract defects in route-lane width,
   control frame transform, dynamic cache gaps and slot persistence. Apply the
   remaining source-taxonomy corrections only when the preflight proves that
   the required map labels/topology exist; otherwise use the declared
   `unknown` representation or fail closed.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Retain OBS-V1.1 global current-state oracle | No migration work | Overstates real ego knowledge and retains temporal shortcuts | Does not satisfy the approved perception boundary |
| RGB plus LiDAR fusion | Rich physical input | Rendering throughput, memory risk, new multimodal encoder, and learned perception scope | Disproportionate to the thesis planning scope |
| Three-plane LiDAR | Better vertical coverage | Additional sensor modelling, cost, and validation effort | Excluded by the approved single-level scope |
| Local V2I/SPaT as automatic signal fallback | Simple signal state access | Changes the information model when ray visibility fails | A fallback must be separately approved, never silent |
| Recompute yellow must-stop every step | Easier policy observability | Permits behaviour-dependent reinterpretation after yellow onset | Violates the approved Rulebook semantics |
| Full 21-frame stack of the complete semantic vector | General temporal context | Large incompatible input and unnecessary history for non-compliance groups | Selective compliance history is sufficient and auditable |

## Consequences

The new schema has flat dimension `3064` and LQ raw-token count `143`.
Checkpoints, replay data, observation normalizers, encoder instances, and
experiment results are incompatible across the v1.1/v1.0 and v1.2/v1.1
boundaries. No implicit migration is allowed.

Implementation begins only after the LiDAR and signal-collider preflights and
the mandatory test matrix recorded in the ExecPlan. A failed preflight is a
blocking technical finding, not authorization to weaken the contract.

## Validation And Traceability

OBS-V1.2 requirements `PB-OBS-001` through `PB-OBS-012` and ENC-V1.1
requirements `PB-ENC-001` through `PB-ENC-004` define the required behavior.
The mandatory preflights, deterministic tests, integration tests, and smoke
commands are frozen in the linked ExecPlan.

## Approval Record

- Approved by: user
- Approval evidence: explicit user instruction in this Codex conversation on
  2026-07-21: “D'accordo su tutto, procedi”, following the detailed review of
  the selected versioning, temporal-memory, planar-LiDAR, signal-visibility,
  no-RGB, and preflight decisions.
- Notes: noise magnitudes and any V2I/SPaT fallback are deliberately not
  approved by this ADR.
