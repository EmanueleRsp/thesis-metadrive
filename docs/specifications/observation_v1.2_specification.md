# Perception-Bounded Semantic Observation Specification

**Document ID:** OBS-V1.2  
**Version:** 1.2-perception-bounded  
**Status:** APPROVED  
**Authoritative:** YES  
**Date:** 2026-07-21  
**Supersedes:** OBS-V1.1 for the selected semantic-observation implementation path  
**Related decisions:** ADR-004, ADR-022, ADR-045  
**Related documents:** ENC-V1.1, RULEBOOK-V4.7, `docs/implementation/perception_bounded_semantic_observation_v1.2_exec_plan.md`

## 1. Purpose

This specification defines a causal semantic observation for the ego vehicle in
MetaDrive. It replaces global-registry access with a bounded, idealised semantic
perception model: physical visibility determines which actors and static
features may enter the observation; semantic attributes of admitted detections
remain idealised. The observation does not contain a future simulator state,
Rulebook output, or Rulebook-owned temporal latch.

The policy may use an assigned navigation route and a local HD-map model. These
are declared navigation and map priors, not sensor detections.

## 2. Scope and non-goals

In scope are the flat observation contract, the latent-query tokenisation
contract, perception eligibility, temporal memory, and implementation
corrections listed below.

Out of scope are RGB rendering, learned detection/classification/tracking,
multi-plane LiDAR, V2I/SPaT fallback, noise calibration, and changes to
Rulebook formulas, rewards, termination, scenarios, or the navigation-route
assignment policy.

## 3. Normative terms

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative. A feature is
*perceived* only when it passes the relevant eligibility rule for the current
control step. A semantic value is *unknown* when the source is not perceived or
cannot be resolved; it MUST be represented by its validity mask and zeroed
payload where this specification defines a mask.

## 4. Timing, frame, and committed-state invariant

The control period is `dt = 0.1 s`. At step `k`, the observation MUST be built
from the committed simulator state at `k`, plus only ego-owned memory written
at steps no later than `k`. It MUST NOT query a future state, another actor's
planned trajectory, or an uncommitted Rulebook decision.

All geometric vectors and headings exposed to the policy are in the ego frame
at step `k`: `x` forward, `y` left, angles wrapped to `[-pi, pi]`. Distances are
metres and speeds are metres per second unless stated otherwise.

## 5. Observation schema

The flat observation is `float32`, finite, and has exact dimension

```text
D = 3064
```

It retains every OBS-V1.1 group except `temporal(5)`, which is replaced by the
two groups in Section 8. The groups in flat order are:

| Group | Shape | Flat values |
|---|---:|---:|
| `ego_history` | `(5, 10)` | 50 |
| `ego_history_mask` | `(5,)` | 5 |
| `ego_current` | `(3,)` | 3 |
| `route` | `(10, 7)` | 70 |
| `route_mask` | `(10,)` | 10 |
| `dynamic` | `(16, 5, 22)` | 1,760 |
| `dynamic_mask` | `(16, 5)` | 80 |
| `static` | `(8, 13)` | 104 |
| `static_mask` | `(8,)` | 8 |
| `lane_road` | `(14,)` | 14 |
| `control` | `(8, 17)` | 136 |
| `control_mask` | `(8,)` | 8 |
| `interaction` | `(8, 35)` | 280 |
| `interaction_mask` | `(8,)` | 8 |
| `compliance_history` | `(21, 24)` | 504 |
| `compliance_history_mask` | `(21,)` | 21 |
| `yellow_onset_memory` | `(3,)` | 3 |

Thus `2541 - 5 + 504 + 21 + 3 = 3064`. The existing field definitions of the
retained OBS-V1.1 groups remain normative unless this document explicitly
overrides their source, frame, category, or eligibility semantics.

## 6. Perception model

### 6.1 Dynamic actors and live static obstacles

The environment MUST perform one 360-degree planar LiDAR sweep per control
step, with 240 equally spaced beams, maximum range 50 m, and sensor height
1.2 m. It MUST use Bullet's first physical hit for each beam, rather than
MetaDrive's broad-phase `detected_objects` list. Each hit MUST be resolved to a
stable internal actor identifier when possible.

Only an actor whose identifier is hit by at least one beam is perceived. Only
perceived actors may update the dynamic-history cache, occupy a dynamic token,
or participate in dynamic ranking. Once admitted by this rule, its position,
velocity, heading, dimensions, and source-confirmed class may be read as an
ideal semantic measurement for this baseline. A non-hit actor MUST be absent,
not merely down-ranked.

The model is intentionally planar. It is valid under the project domain
assumption of single-level road scenes. Existing range and vertical guards MUST
remain active; no multi-plane extension or multilevel-scene study is required.

Live static obstacles represented as simulator actors use the same first-hit
identifier rule as dynamic actors. In contrast, canonical road boundaries and
other immutable map features are local HD-map priors: they are eligible when
their nearest geometry lies within the static range and vertical guard, without
a LiDAR occlusion test. This deliberately models map knowledge, not a detector
for painted lines or mapped road edges. Long geometries MUST use their closest
point or segment to the ego for range/ranking; their arbitrary representative
point MUST NOT determine eligibility.

### 6.2 Signals and stop controls

Signals use a symbolic ideal forward semantic camera, not rendered RGB. A
candidate traffic light is observable only when all conditions hold:

1. its mapped physical light object resolves to a known world pose;
2. its virtual light-head anchor is within 80 m of the ego camera origin;
3. the anchor lies in the forward 65-degree horizontal field of view; and
4. a Bullet ray from the ego camera origin to that anchor has no eligible
   blocker before the anchor.

**Amendment (ADR-045, 2026-08-01):** the 80 m range and 65-degree field of
view above are the default, not a hard limit. `conf/obs/semantic_v3.yaml`'s
`signal_range_m` / `signal_fov_degrees` / `signal_camera_height_m` are the
effective source of truth; an experiment that does not override them
reproduces this baseline unchanged. A run using a non-default value departs
from this clause in the strict sense and MUST be identified as such via its
recorded config, not assumed to match the baseline.

The anchor MUST be derived from the light object's known pose plus the
MetaDrive traffic-light visual height. The implementation MUST NOT require a
collision ray to hit the traffic-light collider itself, since that collider is
not a reliable representation of the luminous head. If observable, the phase
may be read from the simulator as a perfect semantic classifier result. If not
observable or not resolvable, phase-dependent payload is zero and its validity
indicator is zero. The same visibility gate applies to signal-derived control
tokens and signal entries in `compliance_history`.

Stop signs and map stop lines are local HD-map/control priors. Their geometry
may be used when within the specified local control horizon; they are not
global knowledge of every scene control.

### 6.3 Map, route, and interaction information

The ego has an assigned route before reset, local HD-map topology, and ego-map
localisation. Route tokens MAY describe the next 50 m of that assigned route.
For each future route point, lane width MUST be taken from the lane containing
that route point, not from the current ego lane.

Adjacent-lane availability MUST be derived from verified local topology. If the
source cannot determine an adjacent lane, both availability fields are `0.0`.
For OBS-V1.2 this is the documented unknown/fail-closed representation: it
means “do not assume an adjacent lane is usable”, not “the map established that
no adjacent lane exists”. It MUST NOT silently report unavailable lanes as
known zeros.

Conflict zones and priorities are map-derived local semantic predictions. They
MUST use only perceived dynamic tracks and the assigned ego route. Unknown or
ambiguous route/topology/track information MUST propagate as invalid or masked,
not as a ground-truth prediction of another actor's intent.

## 7. Retained dynamic, static, control, and interaction semantics

The capacities remain `16` dynamic, `8` static, `8` control, and `8`
interaction slots. Ranking MUST be deterministic. Overflow MUST be instrumented
and acceptance tests MUST show that curated critical conflicts are not silently
dropped before a non-critical candidate.

Dynamic slot reuse MUST compare persistent actor identifiers to persistent actor
identifiers. It MUST NOT compare a snapshot object with an identifier. A track
that reappears inside the configured tracker-retention window SHOULD reuse its
previous slot; otherwise it is treated as a new detected track.

For every dynamic slot, the five history positions correspond to exact steps
`k-4` through `k`. The cache MUST store the source `step_index` of each sample.
If a sample is absent at an expected step, its entire payload is zero and its
mask is zero. Old samples MUST NOT be shifted next to a later reacquisition as
though they were consecutive measurements.

Static categories MUST distinguish only source-confirmed classes among cone,
barrier, wall, stationary vehicle, and generic obstacle. Unsupported source
classes MUST map to generic obstacle, not an invented fine class. Control
midpoints MUST be transformed into the ego frame. Interaction types MUST
distinguish intersection, merge, and roundabout only after the source taxonomy
preflight establishes a deterministic mapping; unclassified zones use the
documented generic/unknown encoding.

## 8. Ego-owned temporal state

`compliance_history` replaces the former policy-visible Rulebook timer/latch
vector. It is a right-aligned chronological trace over steps `k-20` through
`k`; unavailable leading positions are zero with mask zero. A valid row has 24
features in this exact order:

| Indices | Values |
|---|---|
| `0` | ego speed |
| `1:3` | left/right local clearance |
| `3:7` | left boundary type one-hot: solid, dashed, curb, other |
| `7:11` | right boundary type one-hot: solid, dashed, curb, other |
| `11` | ego intersects an active dashed boundary |
| `12` | the active dashed-boundary continuity indicator |
| `13` | a movement-relevant control is present |
| `14` | the active-control continuity indicator |
| `15:17` | active control type one-hot: signal, stop |
| `17:22` | observed signal state one-hot: red, yellow, green, off, unknown |
| `22` | signed front-bumper distance to the active control |
| `23` | the active control governs the ego movement |

The continuity indicators are local geometric/control associations, not global
map identifiers and not elapsed timers. They distinguish the same currently
relevant dashed boundary or control from a newly relevant one. This permits the
encoder to infer bounded-duration events without revealing a Rulebook result.

`yellow_onset_memory = [is_active, distance_at_onset, speed_at_onset]` is
ego-owned memory. At the first *observable* yellow state of a continuous active
signal, it MUST latch the current front-bumper distance and ego speed. It MUST
reset when that signal ceases to be active/observable or changes away from
yellow. It does not expose `yellow_must_stop` or any Rulebook decision. If a
signal first becomes observable while already yellow, the implementation MAY
latch that first observation; this deliberately gives a conservative policy
input but cannot reconstruct the physical onset known only to the simulator.

The 21-frame horizon is mandated by Rulebook V4.7's maximum two-second timer
with `dt=0.1 s`: `ceil(2.0 / 0.1) + 1 = 21`. The Rulebook itself retains its
internal timers and frozen-yellow calculation unchanged; these internal values
MUST NOT enter the policy observation.

## 9. Causality, idealisation, and uncertainty

The observation is temporally causal. The semantic tracker/classifier after
physical admission is intentionally near-perfect. This is an explicit
semantic-tracker baseline, not a claim of production sensor realism. No
independent per-field noise or dropout is part of this approved version.

Future trajectory, future signal phase, other actors' destination/intent, and
global actor registry membership without a current detection MUST NOT be
included. CPA and occupancy features MUST be calculated from current or cached
perceived measurements only. A later, separately approved extension may apply
temporally coherent track and ego-map noise, then recompute all derived
quantities from the perturbed state.

## 10. Latent-query token contract

This schema produces 143 raw tokens. The group order is identical to OBS-V1.1
through interaction tokens, followed by 21 compliance-history tokens and one
yellow-onset-memory token. The detailed projection, mask, type, time, and slot
embedding contract is ENC-V1.1.

## 11. Preconditions and strict failure policy

Before production integration, the tests in the linked ExecPlan MUST verify:

- first-hit LiDAR actor-ID resolution and required collider coverage in PG and
  Waymo fixtures, including vehicle occlusion, VRU/barrier coverage, partial
  beam hits, and cache reacquisition;
- light-head-anchor ray visibility for visible signals, vehicle/static
  occlusion, field-of-view rejection, and expected height geometry; and
- deterministic source support for adjacent-lane and interaction taxonomy.

If a precondition fails, implementation MUST stop at that unsupported feature
and report the failed source capability. It MUST NOT introduce V2I, RGB,
multiplane sensing, a broad-phase visibility proxy, or an unapproved default.

## 12. Acceptance criteria

- The schema has exactly 3,064 finite `float32` values and exactly 143 LQ raw
  tokens.
- Every policy-visible dynamic/live-static feature passed the first-hit
  visibility gate at its sample step.
- Occluded or missing dynamic samples produce a temporal gap, not an artificial
  consecutive history.
- A non-visible signal is unknown/masked; a visible signal is in range, FOV,
  and line of sight.
- No Rulebook timer/latch/result is policy-visible; the trace has 21 positions
  and the yellow memory contains only ego-owned onset measurements.
- The correction requirements in Sections 6--7 have deterministic regression
  tests.
- Existing legacy observation modes retain their historical contracts.

## 13. Compatibility

OBS-V1.2 is incompatible with OBS-V1.1 flat dimensions and encoder weights.
Runs, checkpoints, normalization statistics, and manifests MUST carry the
schema version and dimensions. There is no automatic migration or fallback to
OBS-V1.1. Historical experiments remain reproducible only by selecting their
original observation and encoder contracts.
