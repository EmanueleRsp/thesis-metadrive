# Perception-Bounded Semantic Observation V1.2 ExecPlan

**Plan ID:** PLAN-PB-OBS-V1.2  
**Status:** APPROVED — implementation not started  
**Specification:** OBS-V1.2 (`APPROVED`, authoritative)  
**Related specification:** ENC-V1.1 (`APPROVED`, authoritative)  
**Decision record:** ADR-022  
**Created:** 2026-07-21

## 1. Objective

Implement the `semantic_v3` observation and encoder path defined by OBS-V1.2
and ENC-V1.1. The result must be temporally causal, bounded by an idealised but
physical first-hit perception gate, and compatible with deterministic research
tests. Historical observation modes must retain their contracts.

This plan starts with source-capability preflights. It does not authorise a
fallback if those preflights fail; unsupported requirements remain blocked until
a new decision is approved.

## 2. Scope

Included:

- `semantic_v3` schema, builder, cache, encoder, configuration, and checkpoint
  metadata;
- one 240-beam, 50 m, 1.2 m planar first-hit LiDAR visibility adapter;
- symbolic 80 m, 65-degree forward signal visibility based on a light-head
  anchor and ray occlusion;
- the 21-row compliance trace and three-value yellow-onset ego memory;
- correction of route width, adjacent lane, control-frame, static geometry and
  categories, interaction categories, dynamic gap, and slot-persistence defects;
- deterministic unit, regression, integration, and smoke validation.

Excluded:

- RGB observations/rendering, learned perception, V2I/SPaT, multi-plane LiDAR,
  multilevel-scene experiments, and calibrated sensor noise;
- modifications to Rulebook V4.7 formulas or its internal compliance state;
- automatic migration of old checkpoints, datasets, statistics, or experiments.

## 3. Verified starting point

The following are repository facts verified on 2026-07-21:

| Area | Current location | Verified fact / implication |
|---|---|---|
| Observation schemas | `src/thesis_rl/contracts/observation_schema.py` | The existing semantic schema is OBS-V1.1 with 2,541 values; a separate schema is required. |
| Semantic builder | `src/thesis_rl/envs/observations/causal_semantic.py`, `semantic_state_v2.py` | Current live snapshots originate from simulator state and need an admission gate before cache/ranking. |
| Environment wiring | `src/thesis_rl/envs/factory.py`, `thesis_scenario_env.py` | A new explicit `semantic_v3` selector can coexist with historical `semantic_v2`. |
| Encoders | `src/thesis_rl/agent/planners/encoders/` and encoder tests | MLP and latent-query paths are schema-bound and require dimension/token tests. |
| Configuration | `conf/config.yaml`, `conf/obs/semantic_v2.yaml` | The default currently selects `semantic_v2`; migration must be explicit. |
| LiDAR source | `third_party/metadrive/metadrive/component/sensors/lidar.py`, `distance_detector.py` | MetaDrive exposes Bullet first-hit rays, but `detected_objects` is broad phase and cannot be used as visibility truth. |
| Signals | `third_party/metadrive/metadrive/component/traffic_light/base_traffic_light.py` | The traffic-light collider represents control/air-wall behaviour, not reliably the luminous head. |
| RGB cost | `third_party/metadrive/metadrive/component/sensors/base_camera.py` and local MetaDrive rendering tests | RGB requires rendering and is deliberately excluded for throughput/memory reasons. |

The exact public helper names, collision-group coverage, source-taxonomy keys,
and fixture setup are preflight subjects rather than assumptions.

## 4. Requirements and acceptance mapping

| Requirement | Source | Implementation target | Deterministic test |
|---|---|---|---|
| PB-OBS-001: exact 3,064-value causal schema | OBS-V1.2 §5 | schema and builder | schema dimension/dtype/finite test |
| PB-OBS-002: only first-hit perceived actors update dynamic state | OBS-V1.2 §6.1 | LiDAR adapter, snapshot admission, cache | visible/occluded/reacquired actor tests |
| PB-OBS-003: live-static visibility and local-map scope | OBS-V1.2 §6.1 | static eligibility/ranking | live-static hit and nearest-geometry tests |
| PB-OBS-004: symbolic signal visibility | OBS-V1.2 §6.2 | signal resolver and ray gate | range/FOV/vehicle/static/height tests |
| PB-OBS-005: map/route source corrections | OBS-V1.2 §6.3, §7 | route, topology, frame transforms | route-width, adjacent-lane, midpoint tests |
| PB-OBS-006: gap-correct dynamic history and stable slots | OBS-V1.2 §7 | cache and slot manager | absent-six-steps/reacquisition regression |
| PB-OBS-007: source-confirmed static/interaction types | OBS-V1.2 §7 | taxonomy adapters | category mapping/preflight tests |
| PB-OBS-008: ego-owned temporal state only | OBS-V1.2 §8 | trace/onset builder | 21-frame, reset, no-timer tests |
| PB-OBS-009: no implicit unsupported fallback | OBS-V1.2 §11 | preflight guards/config | failed-capability test/documented block |
| PB-OBS-010: capacity preserves critical candidates | OBS-V1.2 §7 | deterministic ranking/diagnostics | curated critical-overflow test |
| PB-OBS-011: causal information boundary | OBS-V1.2 §4, §9 | all builder sources | anti-future/anti-Rulebook-field audit |
| PB-OBS-012: legacy contracts remain unchanged | OBS-V1.2 §13 | versioned factory/config | existing legacy regression matrix |
| PB-ENC-001: MLP 3,064 input contract | ENC-V1.1 §3 | MLP factory | shape/parameter-count test |
| PB-ENC-002: 143-token LQ contract | ENC-V1.1 §4 | LQ tokeniser | order/mask/type/time tests |
| PB-ENC-003: explicit compatibility boundary | ENC-V1.1 §5 | config/checkpoint validation | mismatch rejection tests |
| PB-ENC-004: finite 256-dimensional encoder output | ENC-V1.1 §2, §6 | both encoder forward paths | batched finite-output test |

## 5. Invariants

- At control step `k`, policy-visible data is committed state at `k` plus
  ego-owned memory from `<= k`; no future trajectory, future phase, or Rulebook
  decision is exposed.
- An unobserved actor is absent from current ranking and does not write a cache
  sample. History gaps are zero payload plus zero mask at their actual steps.
- The first physical ray hit, not broad-phase overlap, determines LiDAR
  admission.
- A signal phase is readable only after range, FOV, target-anchor resolution,
  and blocker checks. Its state is otherwise unknown/masked.
- Local map and assigned route are allowed priors; another actor's intent is
  never inferred as ground truth.
- Rulebook timers, latches, and `yellow_must_stop` remain internal. The policy
  sees only the specified trace and onset measurements.
- No legacy mode changes its flat shape, token count, checkpoint contract, or
  default behaviour unless separately approved.

## 6. Approved decisions and non-decisions

The user approved ADR-022 on 2026-07-21. The implementation SHALL use one
planar LiDAR sweep only; it SHALL NOT perform a three-plane or multilevel-scene
study. RGB is not an alternative implementation path. Signal state is a perfect
semantic classification only after physical symbolic visibility, not a V2I
message. If the preflight cannot establish a required collider/source mapping,
work stops for that feature and requests a new decision.

The semantic tracker after detection remains an acknowledged idealisation. No
noise values are approved: temporally coherent track noise and ego-map noise
are deferred optional work, not placeholders to implement now.

## 7. Technical design

At each step, the builder performs this causal sequence:

```text
committed world k
  -> one Bullet first-hit LiDAR sweep
  -> resolve hit nodes to internal actor IDs
  -> admit detected dynamic/live-static candidates
  -> obtain ideal semantic measurements only for admitted IDs
  -> update timestamped track cache and deterministic slots
  -> derive dynamic/static/control/interaction tokens
  -> append compliance row and update yellow-onset ego memory
  -> flatten OBS-V1.2 / assemble ENC-V1.1 tokens
```

The LiDAR adapter must expose a small testable result object containing beam
index, hit fraction/distance, hit node, and resolved actor ID/class. Its caller
must not receive the broad-phase object list as evidence of visibility. The
adapter must ignore the ego body, preserve nearest-hit semantics, and apply the
existing range/vertical protections.

The signal adapter must resolve the scene light object through documented
manager/object mappings, construct a virtual head anchor from its known world
pose and visual height, then ray-test ego camera origin to anchor. A blocker is
an eligible ray hit strictly before the anchor distance with a small documented
numeric tolerance. The test implementation must define and exercise the
collision groups that block a signal. A target collider hit is not required.

The cache stores `(step_index, payload)` per actor ID and writes rows for exact
expected step indices. Reacquisition within the selected retention window
reuses the prior slot but preserves zero-mask gaps. Slot selection must compare
actor ID with actor ID only.

## 8. Milestones

### M0 — Documentation and contract lock (complete)

Create ADR-022, OBS-V1.2, ENC-V1.1, this ExecPlan, and index entries. Record
the approved choices, exact schema dimensions, and no-fallback policy.

Exit criterion: documents are internally consistent and `git diff --check`
passes.

### M1 — Capability preflight (required before production integration)

Add narrow test/harness code only as needed to verify, in both PG and Waymo
fixtures:

1. a first-hit ray resolves visible actor IDs;
2. an actor fully behind a vehicle is absent, while a partial beam hit is
   admitted;
3. required VRU and barrier classes have usable colliders and resolve correctly;
4. a physically visible, vehicle-occluded, static-occluded, out-of-FOV, and
   elevated light-head anchor behave as specified;
5. local topology can distinguish adjacent availability and source taxonomy can
   distinguish required interaction classes.

Also benchmark one sweep per control step against the existing semantic path at
the intended test configuration. Record exact commands, scenarios, timings,
and source limitations in this plan's Progress section.

Exit criterion: every required capability is demonstrated deterministically. A
failed item blocks the related milestone; no fallback is added.

### M2 — Schema and test-first contracts

Introduce OBS-V1.2 schema declarations, `semantic_v3` selection, group access,
zero/mask validation, and tests before builder integration. Introduce the
3,064-value flattening and 143-token assembly tests with fixed synthetic
fixtures. Preserve historical schema fixtures unchanged.

Exit criterion: schema contracts fail before implementation and pass after it;
old contracts still pass.

### M3 — Perception admission and temporal cache

Implement the first-hit adapter, actor/static admission, timestamped cache,
gap masks, deterministic actor-ID slot reuse, and overflow diagnostics. Ensure
all CPA/occupancy calculations consume admitted current/cached tracks only.

Exit criterion: visibility, occlusion, exact temporal-gap, slot-reuse, and
critical-overflow tests pass.

### M4 — Map, controls, and temporal corrections

Implement route-point lane widths, verified adjacent-lane flags, ego-frame
control midpoint, closest static geometry, source-confirmed static types,
interaction taxonomy, signal anchor visibility, compliance history, and yellow
onset ego memory. Remove every policy-visible OBS-V1.1 temporal Rulebook value.

Exit criterion: field-level regression tests and Rulebook causal-input audit
pass without changing Rulebook calculations.

### M5 — Encoder, configuration, and compatibility

Implement ENC-V1.1 projections/embeddings, `semantic_v3` MLP input, checkpoint
metadata validation, and explicit incompatibility errors. Update only configs
approved for `semantic_v3`; do not silently change experiment defaults.

Exit criterion: batch shape, finite-output, parameter-count, token-order,
masked-token, and manifest mismatch tests pass.

### M6 — Validation and reconciliation

Run focused tests after each milestone, then relevant Rulebook tests, focused
Ruff, `git diff --check`, and a representative `semantic_v3` training/reset-step
smoke. Reconcile every requirement in Section 4 with code and tests; update
this plan's Progress, Decisions, Files, and Remaining Work sections.

Exit criterion: all required checks pass and no unapproved deviation remains.

## 9. Mandatory test matrix

| Test ID | Level | Scenario / fixture | Expected result |
|---|---|---|---|
| T-PB-01 | preflight | PG visible/vehicle-occluded/partial actor | first-hit IDs and admission are exact |
| T-PB-02 | preflight | Waymo visible/vehicle-occluded/VRU/barrier | collider coverage and ID mapping are exact |
| T-PB-03 | preflight | PG and Waymo light states | anchor range/FOV/vehicle/static/height visibility is exact |
| T-PB-04 | unit | synthetic cache | six missing steps are zero/masked, not compressed |
| T-PB-05 | unit | synthetic slots | same actor ID reuses slot; object/ID mismatch cannot occur |
| T-PB-06 | unit | route/map synthetic fixture | route width, adjacency, midpoint, nearest local-map geometry correct |
| T-PB-07 | unit | taxonomy fixture | static and interaction mapping only emits supported categories |
| T-PB-08 | unit | temporal trace | 21 chronological rows, continuity/reset, yellow memory semantics |
| T-PB-09 | schema | synthetic complete observation | `(3064,)`, `float32`, finite, correct masks |
| T-PB-10 | encoder | synthetic batch | MLP/LQ `(B,256)`, 2,032,128 MLP parameters, 143 LQ tokens |
| T-PB-11 | encoder | masked tokens | masked dynamic/compliance payload does not influence output |
| T-PB-12 | compatibility | old manifest/checkpoint | schema mismatch is rejected explicitly |
| T-PB-13 | integration | curated conflict overflow fixture | no critical candidate is displaced by non-critical overflow |
| T-PB-14 | smoke | `semantic_v3` reset and short rollout | finite observations/actions/rewards, no rendering dependency |

## 10. Files expected to change

| File or area | Planned change |
|---|---|
| `src/thesis_rl/contracts/observation_schema.py` | Add OBS-V1.2 groups/dimensions and validation. |
| `src/thesis_rl/envs/observations/causal_semantic.py` | Add perception adapter use, cache/slot fixes, controls/map/temporal fields. |
| `src/thesis_rl/envs/observations/semantic_state_v2.py` or successor | Preserve historical path; introduce an explicitly versioned V1.2 builder. |
| `src/thesis_rl/envs/factory.py`, environment wiring | Add explicit `semantic_v3` selection. |
| `src/thesis_rl/agent/planners/encoders/` | Add ENC-V1.1 MLP/LQ contracts and metadata checks. |
| `conf/obs/` and checkpoint/config code | Add explicit V1.2 configuration; preserve V1.1 configs. |
| `tests/` | Add all tests in Section 9 and bug regressions. |
| `docs/` | Keep ADR, specifications, plan, and project index reconciled. |

## 11. Progress log

| Date | Milestone | Status | Evidence |
|---|---|---|---|
| 2026-07-21 | M0 | complete | ADR-022, OBS-V1.2, ENC-V1.1, and this plan created after explicit user approval. |
| 2026-07-21 | M1–M6 | not started | No production code or tests changed by this planning task. |

## 12. Risks, blockers, and remaining decisions

- **Collider/source capability:** M1 may show that a required scenario object
  lacks a usable collider or stable mapping. This blocks that feature by design;
  it is not permission to substitute broad-phase detection or V2I.
- **Signal anchor calibration:** source geometry must be confirmed in fixtures;
  a documented tolerance is required to avoid self/target numerical artefacts.
- **Topology/taxonomy:** adjacent-lane and merge/roundabout fields must remain
  unknown if source support is insufficient.
- **Performance:** one 240-beam sweep is expected to be materially cheaper than
  RGB, but M1 must measure it at the actual workload.
- **Ideal semantic tracking:** intentionally remains an idealised research
  assumption after physical detection. A calibrated stochastic tracker is
  deferred and requires a separate approved specification.

There are no unresolved design choices blocking M0. A preflight failure is an
evidence-based implementation blocker, not an ambiguity in the contract.

## 13. Validation record

No implementation validation is claimed yet. Documentation validation completed
on 2026-07-21:

```text
python -c '<schema/token/MLP parameter arithmetic assertions>'
git diff --check
```

The arithmetic assertions reported `flat=3064`, `tokens=143`, and
`mlp_parameters=2032128`; `git diff --check` passed. All M1–M6 commands and
results must be appended here when executed.

## 14. Final reconciliation checklist

- [x] User approval recorded in ADR-022.
- [x] OBS-V1.2 and ENC-V1.1 specify dimensions, causality, masks, and compatibility.
- [x] Preflight/no-fallback policy is explicit.
- [ ] M1 source capabilities verified.
- [ ] M2–M5 implementation and regression tests completed.
- [ ] M6 checks, smoke, and requirement-to-code-to-test reconciliation completed.
