# Semantic Observation And Encoder v1 ExecPlan

## 1. Metadata

- Plan ID: `PLAN-OBS-ENC-V1`
- Feature: semantic observation v1.1 and encoder v1.0 reconciliation
- Status: `IN_PROGRESS`
- Created: `2026-07-16`
- Last updated: `2026-07-17`
- Branch: `scenarionet-implementation` at `52e1794`
- Owner: thesis repository maintainer
- Authoritative specifications:
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`,
    version `1.1-final-implementation-complete`, `APPROVED`
  - `docs/specifications/encoder_v1.0_specification.md`, ID `ENC-V1.0`,
    version `1.0-final-implementation-complete`, `APPROVED`
- Related ADR: `docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md`

## 2. Objective And Scope

Implement the approved causal assigned-route observation contract and its MLP/LQ
encoder contract for the fork-backed TD3, SAC, and PPO backends. Success means
that each approved observation/encoder combination has the exact schema,
anti-leakage behavior, SB3 ownership, checkpoint compatibility checks, and
mandatory deterministic tests defined below.

In scope: semantic observation v1.1, stacked LiDAR baseline, schema and
flattening, causal context, encoder modules, SB3 bridge/configuration,
checkpoint manifest/publication, tests, manifests, and smoke coverage.

Out of scope: reward/rulebook scientific changes, ACL behavior, replay-buffer
extensions, new dependencies, perception robustness extensions, training
performance claims, implicit checkpoint migration, and legacy backend redesign.

Compatibility constraints: Python `>=3.10,<3.11`; checked-out MetaDrive commit
`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`; local SB3 `2.9.0` commit
`6a196a60c7df3550ac5832caad54ef8dce9a6f31`; no use of future scenario tracks
in online observation construction.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `OBS-REQ-001` | Build and freeze the assigned ego route from persisted lane IDs and map geometry at reset; reject invalid routes without native or online-future fallback. | OBS §2, §4, §14 |
| `OBS-REQ-002` | Provide a `308`-feature LiDAR frame and a deterministic `1540`-feature five-frame stack with the declared detector/noise contract. | OBS §5, §13.4, §14 |
| `OBS-REQ-003` | Produce semantic v1.1 structured groups, masks, flat order, `D=2541`, and `122` LQ raw tokens from one schema. | OBS §6--§7, §11, §13.5; ENC §5, §8 |
| `OBS-REQ-004` | Enforce causal selection, history, slot persistence, ambiguous-movement handling, normalization, and zeroed masked payloads. | OBS §2, §7--§10, §13.1--§13.3 |
| `OBS-REQ-005` | Supply an environment-owned causal context and commit its whitelisted memory in lockstep with transition evaluation without exposing rule results. | OBS §3.1, §12, §14 |
| `OBS-REQ-006` | Record reproducible configuration, upstream versions, diagnostics, and visual validation evidence. | OBS §5.6, §8.5, §13.4--§13.6, §14 |
| `ENC-REQ-001` | Implement rank/dtype/finite validation and the exact Flat MLP contract for LiDAR and semantic inputs. | ENC §4, §7, §18.1--§18.2 |
| `ENC-REQ-002` | Implement LQ v2 tokenization, embeddings, masks, four blocks, pooling, and output shape for semantic v1.1 only. | ENC §5, §8, §18.3--§18.5 |
| `ENC-REQ-003` | Integrate factory-built feature extractors with exact TD3, SAC, and PPO sharing, target, optimizer, and initialization semantics. | ENC §4.3, §9--§12, §16, §18.6--§18.7 |
| `ENC-REQ-004` | Implement fail-fast checkpoint compatibility and atomic generation publication. | ENC §15, §18.8 |
| `ENC-REQ-005` | Freeze supported configurations and complete the required smoke matrix without claiming performance. | ENC §3, §13--§14, §16, §18.9--§19 |

## 4. Current Repository Analysis

| Status | Verified fact or finding |
|---|---|
| `VERIFIED` | `src/thesis_rl/envs/observations/semantic_state.py::SemanticStateObservation` is the legacy implementation. Its current default layout is 2363 features and 107 LQ tokens, not the approved 2541/122 contract. |
| `VERIFIED` | `conf/obs/semantic_state.yaml` configures legacy route capacity `5`, legacy normalizations, and the `semantic_state` public selector. |
| `VERIFIED` | `src/thesis_rl/contracts/observation_spec.py::ObservationSpec` duplicates the legacy dimensions. `lq/unflatten.py` owns separate offset arithmetic. |
| `VERIFIED` | `src/thesis_rl/envs/factory.py::_configure_agent_observation` currently selects either native `lidar_state` or legacy `SemanticStateObservation`. |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/` provides route, lane, control, conflict-zone, occupancy, and memory primitives, but `ThesisScenarioEnv` does not currently expose the required causal-context adapter. |
| `VERIFIED` | `src/thesis_rl/sb3_extensions/features_extractors.py` silently flattens rank greater than two. Its extractor builds from config, which is the correct ownership direction but needs the approved interface and validation. |
| `VERIFIED` | Legacy encoder modules are under `src/thesis_rl/agent/planners/encoders/`; `MLPEncoder` applies final LayerNorm unconditionally and `LQEncoder` defaults residual gating to true. |
| `VERIFIED` | `src/thesis_rl/sb3_extensions/builders.py` is the Hydra-to-SB3 bridge. The local SB3 fork provides independent TD3/SAC actor/critic extractors for `share_features_extractor=false`, SAC has no target actor, and PPO supports shared extractors with `ortho_init=false`. |
| `VERIFIED` | Existing focused tests include `tests/test_semantic_state_observation.py`, `tests/test_sb3_extensions.py`, and `tests/test_sb3_direct_backends.py`; they test legacy behavior only. |
| `SPECIFIED` | Approved semantic checkpoints are incompatible with legacy 2363/107 checkpoints and must fail rather than migrate. |

## 5. Assumptions And Invariants

- All online policy features use only current/past observations, frozen task
  metadata, map data, and committed causal memory; future tracks, future signal
  phases, curriculum labels, rule costs/margins/statuses, IDs, and selection
  flags are forbidden.
- Positions/distances use metres, times seconds, headings radians, velocities
  m/s, accelerations m/s², and yaw rate rad/s before normalization.
- All history is re-expressed in the current ego frame. Semantic reset uses
  zero payload plus zero masks for unavailable past; LiDAR reset repeats the
  first valid frame.
- Semantic masks use `1=valid`, `0=padding`; masked payload is exactly zero.
  LQ applies masks to scene key/value attention and zeros masked embeddings.
- A transition commits next causal memory/context atomically before `obs[t+1]`.
  Termination and truncation remain distinct Gymnasium signals.
- LiDAR ray noise is owned exclusively by `RayNoiseWrapper`, drawn from
  `engine.np_random`; native sensor noise is zero.
- The schema fingerprint, observation version, architecture version, encoder
  type, feature dimension, SB3 version/commit, project commit, and seed are
  checkpoint compatibility inputs.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | implementation detail | Choose the internal `CausalSceneContext` representation. | mutable adapter / immutable typed context | Immutable typed context built from canonical Rulebook v2 primitives. | Internal API, deterministic tests | Approved by `OBS-V1.1` |
| `DEC-002` | implementation detail | Define unknown other-agent manoeuvre behavior. | hypothesize zones / emit only causal zones | Emit no token when zone geometry needs a future exit; retain known geometry with undefined priority. | Observation content and tests | Approved by `OBS-V1.1` |
| `DEC-003` | implementation detail | SB3 extractor ownership. | pass module instance / construct from config | Construct one encoder per extractor from plain config and schema. | Gradient routing and targets | Approved by `ENC-V1.0` |
| `DEC-004` | implementation detail | Checkpoint publication. | sidecar pairs / immutable generation plus pointer | Immutable generation directory and atomic verified `latest.json`. | Resume safety | Approved by `ENC-V1.0` |
| `DEC-005` | plan approval | Freeze this mandatory test matrix before production changes. | start coding / approve this plan first | Obtain explicit approval of this ExecPlan. | Test protection and milestone sequencing | Approved by explicit user approval on `2026-07-16` |
| `DEC-006` | specification clarification | Treat navigation route as immutable assigned task metadata for both sources. | map-only inferred destination / frozen assigned route / online SDC route | Persist PG generator route or Waymo offline SDC-map-matched route before reset; runtime consumes only lane IDs and map geometry. | Route semantics, dataset metadata, scenario validation, and observation content. | Approved by user on `2026-07-17`; ADR-004 |
| `DEC-007` | specification clarification | Select the offline source of assigned-route metadata for existing PG exports, which do not persist generator navigation. | (A) offline SDC map-match for PG, matching Waymo; (B) extend/regenerate PG exports with generator-assigned routes. | A for identical source behavior without changing scenario geometry; provenance remains non-policy metadata. | PG dataset annotation provenance and comparability. | Approved by user on `2026-07-17`; ADR-004 |

## 7. Proposed Design

1. Introduce `SemanticObservationSchemaV11` as the sole owner of group shapes,
   slices, flatten/unflatten, token order, masks, canonical serialization, and
   fingerprint. All observation and encoder paths depend on it.
2. Build a typed environment-owned causal context at reset and transition
   commit. It reuses Rulebook v2 primitives but exposes only explicitly
   permitted values to the observation builder.
3. Replace the legacy semantic builder with `SemanticStateObservationV2`; add
   a custom map-route navigation adapter and `StackedLidarStateObservation`.
   Both enforce runtime observation-space dimensions.
4. Replace legacy LQ unflatten/tokenization with schema-driven v1.1 paths.
   Retain legacy code only behind explicit legacy configuration if required for
   historical runs; it must not resume an approved v1.1 checkpoint.
5. Make `ThesisEncoderFeatureExtractor` reject non-flat input and build its
   encoder from serializable config/schema. Configure TD3/SAC/PPO policy kwargs
   explicitly and test module/optimizer ownership.
6. Add checkpoint generation publishing, validation, and manifest comparison
   before invoking SB3 load.

No new dependency, fallback, public data policy, reward change, or algorithmic
variant is introduced by this plan.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `OBS-REQ-001` | `AC-OBS-001` | `rulebook/v2/context/map_matching.py`, `rulebook/v2/context/pg_static_adapter.py`, `rulebook/v2/context/waymo_static_adapter.py`, `rulebook/v2/geometry/route.py`, `rulebook/v2/context/static_adapter.py`, route metadata in `scenarios/records.py`, `filter_rulebook_v2_catalog.py`, and `thesis_scenario_env.py` | `tests/test_rulebook_v2_contracts.py`, `tests/test_rulebook_v2_catalog_eligibility.py`, `tests/test_rulebook_v2_pg_adapter.py`, `tests/test_rulebook_v2_waymo_adapter.py`, `tests/test_scenario_records.py`, `tests/test_thesis_scenario_env.py` | Partial: offline assignment persistence, metadata-only adapter consumption, fail-closed polyline construction, and reset publication implemented; custom runtime navigation replacement remains pending. |
| `OBS-REQ-002` | `AC-OBS-002` | `envs/observations/assigned_route.py`, `envs/observations/ray_noise.py`, `envs/observations/causal_lidar.py`, `envs/observations/stacked_lidar.py`, `thesis_scenario_env.py`, map navigation | `tests/test_assigned_route_observation.py`, `tests/test_ray_noise.py`, `tests/test_causal_lidar.py`, `tests/test_stacked_lidar_observation.py`, `tests/test_thesis_scenario_env.py`, `TEST-OBS-003`--`TEST-OBS-005` | Partial: causal 22D navigation, single-owner ray-noise, exact 308D frame builder, strict five-frame stack, and pre-first-observation installation implemented; full live sensor smoke validation remains pending. |
| `OBS-REQ-003` | `AC-OBS-003` | `src/thesis_rl/contracts/observation_schema.py`, `src/thesis_rl/envs/observations/semantic_state_v2.py`, `src/thesis_rl/envs/factory.py` | `tests/test_observation_schema_v11.py`, `tests/test_semantic_state_v2.py` | Partial: schema-owned strict v2 flat adapter now emits/validates `2541`; environment-owned causal batch construction and live installation remain pending. |
| `OBS-REQ-004` | `AC-OBS-004` | selection/history/context utilities | `TEST-OBS-009`--`TEST-OBS-012` | Planned |
| `OBS-REQ-005` | `AC-OBS-005` | `src/thesis_rl/contracts/causal_scene_context.py`, `src/thesis_rl/rulebook/v2/wrapper.py` | `tests/test_causal_scene_context.py`, `tests/test_rulebook_v2_causal_context.py` | Partial: immutable boundary and post-commit Rulebook wrapper publication implemented; environment observation lifecycle wiring remains pending. |
| `OBS-REQ-006` | `AC-OBS-006` | route provenance in catalog records and scenario metadata | `tests/test_scenario_records.py`, `tests/test_rulebook_v2_catalog_eligibility.py` | Partial: route provenance is persisted/logged; full experiment manifest and visual evidence remain pending. |
| `ENC-REQ-001` | `AC-ENC-001` | `src/thesis_rl/agent/planners/encoders/base.py`, `mlp_encoder.py` | `tests/test_encoders_v10.py` | Implemented and focused-verified. |
| `ENC-REQ-002` | `AC-ENC-002` | `lq_encoder.py`, `contracts/observation_schema.py` | `tests/test_encoders_v10.py`, `tests/test_observation_schema_v11.py` | Implemented and focused-verified. |
| `ENC-REQ-003` | `AC-ENC-003` | `sb3_extensions/features_extractors.py`, `builders.py` | `tests/test_sb3_extensions.py`, `tests/test_sb3_direct_backends.py` | Partial: strict flat bridge and sharing flags verified; complete optimizer/target routing matrix remains pending. |
| `ENC-REQ-004` | `AC-ENC-004` | checkpointing | `TEST-ENC-012`--`TEST-ENC-014` | Planned |
| `ENC-REQ-005` | `AC-ENC-005` | presets and smoke driver | `TEST-ENC-015` | Planned |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-OBS-001` | Unit | Frozen assigned route is independent of runtime future SDC samples. | Same assignment/map/current task, altered future trajectory. | Identical route and observation. | `OBS-REQ-001` |
| `TEST-OBS-002` | Unit | Invalid assigned route fails closed. | Missing, invalid, or non-contiguous lane sequence. | Recorded exclusion/error before control. | `OBS-REQ-001` |
| `TEST-OBS-003` | Unit | LiDAR dimensions and stack order. | Deterministic mocked frames. | `308`, `1540`, oldest-to-current order. | `OBS-REQ-002` |
| `TEST-OBS-004` | Unit | Ray noise ownership and seed. | Fixed `engine.np_random` state. | One perturbation per declared ray block; reproducible values. | `OBS-REQ-002` |
| `TEST-OBS-005` | Integration | Upstream shape drift fails. | Simulated navigation/sensor dimension mismatch. | Explicit runtime failure. | `OBS-REQ-002` |
| `TEST-OBS-006` | Unit | Schema flatten/unflatten. | Canonical structured batch. | Exact group slices and `D=2541`. | `OBS-REQ-003` |
| `TEST-OBS-007` | Unit | LQ raw token count/order. | Structured batch with sentinel values. | Exactly 122 tokens in normative order. | `OBS-REQ-003` |
| `TEST-OBS-008` | Unit | Mask/padding contract. | Every variable group padded. | Zero payload and zero mask. | `OBS-REQ-003` |
| `TEST-OBS-009` | Unit | Future invariance. | Alter future tracks/signals/ACL/reward only. | Identical current observation. | `OBS-REQ-004` |
| `TEST-OBS-010` | Unit | Ambiguous other manoeuvre. | Multiple map exits with same past/current state. | No hypothetical interaction token; no future access. | `OBS-REQ-004` |
| `TEST-OBS-011` | Unit | History and slots. | Appearing, absent, promoted, and preempting actors. | Causal cache, deterministic slot rules, clean reset. | `OBS-REQ-004` |
| `TEST-OBS-012` | Unit | Frame/vertical invariance. | Global transforms and overpass fixtures. | Ego-relative equivalence; no cross-level tokens. | `OBS-REQ-004` |
| `TEST-OBS-013` | Integration | Transition timing. | One deterministic environment transition. | `obs[t+1]` contains committed timers/latches, not result fields. | `OBS-REQ-005` |
| `TEST-OBS-014` | Integration | Context access control. | Instrumented monitor result. | Observation cannot read costs/margins/statuses. | `OBS-REQ-005` |
| `TEST-OBS-015` | Unit | Config/version manifest. | Resolved Hydra config. | Frozen values and commits recorded. | `OBS-REQ-006` |
| `TEST-OBS-016` | Manual/integration | Visual diagnostic contract. | 20 Waymo and 20 PG selected scenarios. | Required overlays and overflow logs retained outside policy input. | `OBS-REQ-006` |
| `TEST-ENC-001` | Unit | MLP shapes/parameters. | `[B,1540]`, `[B,2541]`. | `[B,256]`, expected parameter counts, finite backward. | `ENC-REQ-001` |
| `TEST-ENC-002` | Unit | Encoder input validation. | Rank, shape, dtype, NaN/Inf variants. | Required fail-fast errors. | `ENC-REQ-001` |
| `TEST-ENC-003` | Unit | LayerNorm switch. | `layer_norm=false`. | No LayerNorm modules, including projection. | `ENC-REQ-001` |
| `TEST-ENC-004` | Unit | LQ schema/tokenization. | Sentinel structured tensor. | Group projections, type/time/slot embeddings, `[B,122,64]`. | `ENC-REQ-002` |
| `TEST-ENC-005` | Unit | LQ mask invariance. | Different masked payloads. | Equal output within deterministic tolerance. | `ENC-REQ-002` |
| `TEST-ENC-006` | Unit | LQ forward/backward. | Batch sizes one and greater than one. | `[B,256]`, finite gradients in all branches. | `ENC-REQ-002` |
| `TEST-ENC-007` | Unit | LQ restriction. | LiDAR + LQ builder request. | Explicit rejection. | `ENC-REQ-002` |
| `TEST-ENC-008` | Integration | TD3 ownership. | MLP and LQ policies. | Four separate encoder roles; correct optimizer/Polyak updates. | `ENC-REQ-003` |
| `TEST-ENC-009` | Integration | SAC ownership. | MLP and LQ policies. | Three roles, no target actor, correct optimizer/Polyak updates. | `ENC-REQ-003` |
| `TEST-ENC-010` | Integration | PPO ownership/init. | MLP and LQ policies. | One shared encoder, one optimizer registration, `ortho_init=false`. | `ENC-REQ-003` |
| `TEST-ENC-011` | Integration | Flat SB3 bridge. | Rank-three observation input. | Explicit error; no silent flatten. | `ENC-REQ-003` |
| `TEST-ENC-012` | Unit | Checkpoint round trip. | Each core configuration. | Equal deterministic output and complete manifest. | `ENC-REQ-004` |
| `TEST-ENC-013` | Unit | Compatibility rejection. | Legacy schema/type/sharing variants. | Informative `CheckpointCompatibilityError`. | `ENC-REQ-004` |
| `TEST-ENC-014` | Unit | Atomic publication recovery. | Interrupted temporary generation/stale pointer. | No mismatched generation resumes. | `ENC-REQ-004` |
| `TEST-ENC-015` | Smoke | Approved algorithm matrix. | Nine TD3/SAC/PPO × LiDAR/semantic MLP/LQ combinations. | Reset, rollout, update, save/load, inference; finite values. | `ENC-REQ-005` |

Acceptance criteria:

- `AC-OBS-001`: route construction is causal, deterministic, and fail-closed.
- `AC-OBS-002`: LiDAR layout, noise, and history exactly match the contract.
- `AC-OBS-003`: semantic schema is the unique source for `2541` flat values
  and `122` tokens.
- `AC-OBS-004`: selection/history/masks are causal, deterministic, and finite.
- `AC-OBS-005`: committed causal context is synchronized with `obs[t+1]`.
- `AC-OBS-006`: reproducibility metadata and non-policy diagnostics exist.
- `AC-ENC-001`: MLP contract validates shape, dtype, finiteness, and topology.
- `AC-ENC-002`: LQ token/mask behavior and gradients are exact and finite.
- `AC-ENC-003`: SB3 module ownership and gradient routing match each algorithm.
- `AC-ENC-004`: incompatible or incomplete checkpoints fail before load.
- `AC-ENC-005`: all nine smoke combinations complete without NaN/Inf.

Planned commands, to be run from repository root only after implementation:

```text
uv run --no-sync python -m pytest -q tests/test_semantic_state_observation.py tests/test_sb3_extensions.py tests/test_sb3_direct_backends.py
make format-check PYTHON_QUALITY_PATHS="<modified Python paths>"
make lint
make smoke
git diff --check
```

The focused test command will be extended with newly created test paths. No
repository-wide type-check command exists. Global format verification is not a
completion gate because its baseline is not clean.

## 10. Milestones

- [ ] **M1 — Schema and causal route/context.** The unique schema, immutable
  observation-safe context boundary, map-based route construction, and reset
  route publication are implemented and focused-tested. Environment-owned
  transition commit lifecycle remains pending.
- [ ] **M2 — Observation implementations.** Stacked LiDAR, route/noise wiring,
  and the strict semantic v1.1 flat adapter/config/tests are implemented.
  The environment-owned semantic batch builder and live semantic installation
  remain pending. Depends on M1.
- [ ] **M3 — Encoder contract.** The v1.0 MLP/LQ core, validation, schema-driven
  tokenization, and focused tests are implemented. Full acceptance coverage and
  end-to-end backend routing remain pending.
- [ ] **M4 — SB3 and checkpoints.** Integrate extractor ownership, policy
  kwargs, checkpoint generations, and integration tests. Depends on M3.
- [ ] **M5 — End-to-end verification.** Run mandatory regressions, quality
  checks, visual diagnostic set, and smoke matrix; reconcile all requirements.
  Depends on M2 and M4.

## 11. Progress And Findings Log

- `2026-07-16`: Joint review completed. The traffic-control dimension was
  reconciled to 17; semantic arithmetic confirms `D=2541` and 122 tokens.
- `2026-07-16`: User approved `OBS-V1.1` and `ENC-V1.0`. ADR-002 records the
  approved contract; no production code has changed.
- `2026-07-16`: Repository analysis confirmed legacy observation/encoder
  divergence and verified local MetaDrive/SB3 revisions.
- `2026-07-16`: User explicitly approved `PLAN-OBS-ENC-V1`, including its
  mandatory test strategy, milestones M1--M5, and recorded decisions. All
  approval gates are resolved; implementation started with acceptance tests.
- `2026-07-16`: Implemented the single `SemanticObservationSchemaV11` source
  of truth (2541 flat values, 122 tokens, deterministic fingerprint) and an
  immutable `CausalSceneContext` that exposes only canonical cache, snapshot,
  and causal memory. Runtime route construction and transition lifecycle are
  not yet wired.
- `2026-07-16`: Implemented the encoder v1.0 MLP/LQ core, strict rank/dtype/
  finite validation, LQ masked-token zeroing, and explicit SB3 sharing flags.
  The legacy semantic runtime observation and checkpoint generation contract
  remain unreconciled and prevent declaring M2/M4/M5 complete.
- `2026-07-16`: The Rulebook v2 wrapper now publishes `CausalSceneContext`
  only after a successful immutable memory/cache commit and never stores the
  `RulebookResult` in that context. Generic wrapper tests using non-canonical
  synthetic snapshots remain supported; runtime observation wiring remains
  pending.
- `2026-07-16`: Identified the route-source decision before M2 runtime
  implementation: `pg_static_adapter.py` and `waymo_static_adapter.py` call
  `map_match_sdc_track_to_task_route(...)`, which consumes SDC track samples.
  The pre-amendment OBS-V1.1 contract could not reuse this output as a route
  assignment.
- `2026-07-17`: User approved `DEC-006`. ADR-004 records the amended contract:
  PG and Waymo SDC-map-matched routes are frozen offline as immutable assigned
  task metadata. Runtime code may read only the persisted lane-ID sequence and
  canonical map geometry; future SDC access remains prohibited online.
- `2026-07-17`: Inspection of the checked PG fixture showed that its metadata
  has no assigned route, destination, or goal field; this established the
  provenance gap that `DEC-007` resolved.
- `2026-07-17`: User approved `DEC-007/A`: existing PG and Waymo adapters use
  the same offline SDC map-matching route annotation; runtime will consume only
  the persisted lane sequence and canonical map geometry.
- `2026-07-17`: Added `route_assignment_source`, `assigned_route_lane_ids`,
  catalog persistence, eligibility JSON propagation, and reset metadata
  publication. PG and Waymo now emit distinct provenance values from the same
  offline SDC map-matching function; native/custom navigation replacement is
  still pending.
- `2026-07-17`: Added fail-closed reset-time construction of
  `RoutePolyline` from the frozen lane-ID sequence and canonical lane
  centerlines. Missing or non-contiguous lane sequences are rejected and
  covered by `tests/test_rulebook_v2_contracts.py`; the custom semantic
  observation wiring still remains pending.
- `2026-07-17`: PG and Waymo static adapters now prefer persisted
  `assigned_route_lane_ids` metadata and do not inspect SDC tracks when it is
  present. Regression tests alter the future track to an incompatible pose and
  confirm that the frozen route is unchanged. The track-based branch remains
  explicitly limited to offline annotation of legacy inputs.
- `2026-07-17`: `ThesisScenarioEnv` now injects the frozen route metadata into
  the loaded scenario before downstream static adapters run, while leaving the
  track payload untouched. Missing route metadata remains a fail-closed reset
  error.
- `2026-07-17`: Added the causal `AssignedRouteWaypointAdapter`,
  `MapRouteNavigationObservation22`, and `RoutePolyline.point_at` primitive
  for fixed-spacing local waypoints from the frozen route. Runtime attachment
  to the full MetaDrive LiDAR wrapper and semantic token field mapping remain
  pending.
- `2026-07-17`: Added `RayNoiseWrapper` with seeded-RNG perturbation,
  normalized clipping, zero-dropout core defaults, and native-noise rejection
  for all three ray sensor groups.
- `2026-07-17`: Added strict `StackedLidarStateObservation` with a 308D frame
  contract, five-frame oldest-to-current stacking, deterministic first-frame
  fill, and explicit failure when causal frame wiring is absent. The factory
  now recognizes the `stacked_lidar_state` observation type; the production
  frame builder and sensor attachment remain pending.
- `2026-07-17`: Added `conf/obs/stacked_lidar_state.yaml` with the frozen
  navigation, detector, and ray-noise ownership settings from OBS-V1.1.
- `2026-07-17`: Added `CausalLidarFrameBuilder`, assembling the exact 6+22+12+
  12+16+240 frame and rejecting missing sensor blocks, native noise, shape
  drift, and out-of-range values. Attaching it to the live MetaDrive
  observation instance remains pending.
- `2026-07-17`: `ThesisScenarioEnv._get_reset_return` now installs the causal
  frame builder on observation instances that expose `set_frame_builder` before
  MetaDrive collects the first observation. The installation is metadata-only
  for route assignment and leaves the SDC track untouched.
- `2026-07-17`: The environment factory now freezes the stacked-LiDAR sensor
  dimensions and disables native noise for LiDAR, SideDetector, and
  LaneLineDetector before environment construction.
- `2026-07-17`: The smoke preset now selects the coherent TD3/LiDAR/identity
  baseline instead of combining TD3 with the semantic-only LQ encoder. The
  vector worker also returns `None` for optional methods absent from generic
  MetaDrive environments, preventing non-ScenarioNet smoke workers from
  terminating during runtime-stat collection.
- `2026-07-17`: Added `SemanticStateObservationV2`, a strict schema-owned
  adapter with the exact `(2541,)` space, structured
  `SemanticObservationBatch` input, binary-mask/zero-padding validation,
  finite bounded output checks, and fail-closed behavior when no causal batch
  builder is installed. Added an explicit `semantic_v2` factory selector and
  `conf/obs/semantic_v2.yaml`; the legacy `semantic_state` selector remains
  unchanged for compatibility until the complete environment-owned causal
  batch builder is available.

## Validation log

| Date | Command | Result |
|---|---|---|
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_contracts.py tests/test_rulebook_v2_catalog_eligibility.py tests/test_thesis_scenario_env.py tests/test_scenario_records.py` | PASS — 54 passed in 5.73s |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py tests/test_scenario_catalog.py tests/test_scenarionet_pipeline.py tests/test_rulebook_v2_causal_context.py` | PASS — 31 passed in 9.06s |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — 9 passed in 1.46s; persisted route metadata is preferred over altered SDC samples. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — 29 passed in 3.25s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync pytest -q tests/test_assigned_route_observation.py tests/test_rulebook_v2_contracts.py` | PASS — 22 passed in 1.97s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/rulebook/v2/geometry/route.py src/thesis_rl/envs/observations/assigned_route.py src/thesis_rl/envs/observations/__init__.py tests/test_assigned_route_observation.py` | PASS — all checks passed. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/rulebook/v2/geometry/route.py src/thesis_rl/envs/observations/assigned_route.py src/thesis_rl/envs/observations/__init__.py tests/test_assigned_route_observation.py` | PASS — all files formatted after focused formatting. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_assigned_route_observation.py tests/test_causal_lidar.py` | PASS — 5 passed in 2.83s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/causal_lidar.py src/thesis_rl/envs/observations/assigned_route.py tests/test_causal_lidar.py` | PASS — all checks passed. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/causal_lidar.py src/thesis_rl/envs/observations/assigned_route.py tests/test_causal_lidar.py` | PASS — all files formatted after focused formatting. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py` | PASS — 22 passed, Ruff clean, 2 files already formatted. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenarionet_causal.py tests/test_thesis_scenario_env.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/factory.py src/thesis_rl/envs/thesis_scenario_env.py tests/test_scenarionet_causal.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/factory.py src/thesis_rl/envs/thesis_scenario_env.py tests/test_scenarionet_causal.py` | PASS — 25 passed, Ruff clean, 3 files already formatted. |
| 2026-07-17 | `make smoke` | PASS — TD3 baseline completed 2,000 training steps, two intermediate evaluations, and final evaluation. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_sb3_extensions.py tests/test_deterministic_subproc_vec_env.py tests/test_thesis_scenario_env.py` | PASS — 35 passed in 3.04s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/sb3_extensions/builders.py src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_sb3_extensions.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/sb3_extensions/builders.py src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_sb3_extensions.py && git diff --check` | PASS — Ruff clean, 3 files formatted, no whitespace errors. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_lidar.py tests/test_stacked_lidar_observation.py tests/test_assigned_route_observation.py tests/test_ray_noise.py tests/test_scenarionet_causal.py tests/test_thesis_scenario_env.py && git diff --check` | PASS — 35 passed in 3.46s; no whitespace errors. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_lidar.py tests/test_stacked_lidar_observation.py tests/test_assigned_route_observation.py tests/test_ray_noise.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/causal_lidar.py src/thesis_rl/envs/observations/stacked_lidar.py src/thesis_rl/envs/observations/assigned_route.py tests/test_causal_lidar.py tests/test_stacked_lidar_observation.py tests/test_assigned_route_observation.py tests/test_ray_noise.py && git diff --check` | PASS — 10 passed in 3.33s; Ruff clean; no whitespace errors. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_assigned_route_observation.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/assigned_route.py src/thesis_rl/envs/observations/__init__.py tests/test_assigned_route_observation.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/assigned_route.py src/thesis_rl/envs/observations/__init__.py tests/test_assigned_route_observation.py` | PASS — 3 passed, Ruff clean, 3 files formatted. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_ray_noise.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/ray_noise.py src/thesis_rl/envs/observations/__init__.py tests/test_ray_noise.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/ray_noise.py src/thesis_rl/envs/observations/__init__.py tests/test_ray_noise.py` | PASS — 3 passed, Ruff clean, 3 files already formatted. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_stacked_lidar_observation.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/stacked_lidar.py src/thesis_rl/envs/observations/__init__.py src/thesis_rl/envs/factory.py tests/test_stacked_lidar_observation.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/stacked_lidar.py src/thesis_rl/envs/observations/__init__.py src/thesis_rl/envs/factory.py tests/test_stacked_lidar_observation.py` | PASS — 2 passed, Ruff clean, files formatted after focused formatting. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_stacked_lidar_observation.py tests/test_assigned_route_observation.py tests/test_ray_noise.py && git diff --check` | PASS — 8 passed in 2.55s; no whitespace errors. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_assigned_route_observation.py tests/test_ray_noise.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py tests/test_thesis_scenario_env.py && git diff --check` | PASS — 35 passed in 6.52s; no whitespace errors. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/rulebook/v2/context/pg_static_adapter.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — all checks passed. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/rulebook/v2/context/pg_static_adapter.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — all files formatted after focused formatting. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/rulebook/v2/context/pg_static_adapter.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — all checks passed. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/rulebook/v2/context/pg_static_adapter.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` | PASS — 6 files already formatted. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check <modified Python paths>` | PASS — all checks passed. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff format --check <modified Python paths>` | PASS — 19 files already formatted. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_scene_context.py tests/test_observation_schema_v11.py tests/test_encoders_v10.py tests/test_semantic_state_observation.py tests/test_sb3_extensions.py tests/test_sb3_direct_backends.py` | PASS — 29 passed in 4.36s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/rulebook/v2/geometry/route.py && docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_contracts.py` | PASS — Ruff clean; 20 passed in 0.13s. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py && docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py` | PASS — Ruff clean, 2 files formatted, 19 passed in 1.76s. |
| 2026-07-17 | `git diff --check` | BLOCKED by pre-existing unrelated trailing whitespace in `docs/implementation/rulebook_v2_implementation_plan.md:29`; no whitespace error was introduced by the route changes. |
| 2026-07-17 | `git diff --check` | PASS — no whitespace errors in the current worktree. |
| 2026-07-17 | `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_contracts.py tests/test_rulebook_v2_catalog_eligibility.py tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py tests/test_scenario_catalog.py tests/test_scenario_records.py tests/test_scenarionet_pipeline.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_causal_context.py tests/test_causal_scene_context.py tests/test_observation_schema_v11.py tests/test_encoders_v10.py tests/test_semantic_state_observation.py tests/test_sb3_extensions.py tests/test_sb3_direct_backends.py` | 117 passed, 1 failed: stale geometry hash assertion in `tests/test_rulebook_v2_catalog_eligibility.py` after unrelated Rulebook v4.7 changes; route/observation tests passed. |

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/contracts/observation_schema.py` | Planned addition | Single v1.1 schema and fingerprint |
| `src/thesis_rl/contracts/encoder_contract.py` | Planned addition | Shared encoder validation/version contract |
| `src/thesis_rl/contracts/checkpoint_manifest.py` | Planned addition | Compatibility and generation manifest types |
| `src/thesis_rl/envs/observations/semantic_state.py` or successor | Compatibility-preserved legacy implementation | Legacy semantic observation retained until v1.1 causal builder is complete |
| `src/thesis_rl/envs/observations/semantic_state_v2.py` | Added | Strict schema-owned semantic v1.1 flat adapter |
| `conf/obs/semantic_v2.yaml` | Added | Explicit v1.1 semantic configuration selector |
| `tests/test_semantic_state_v2.py` | Added | v1.1 adapter shape, mask, padding, and factory acceptance tests |
| `src/thesis_rl/envs/observations/stacked_lidar_state.py` | Planned addition | Map-route stacked LiDAR observation |
| `src/thesis_rl/envs/` context/environment wiring | Planned modification | Causal context lifecycle and route installation |
| `src/thesis_rl/agent/planners/encoders/` | Planned modification | MLP/LQ v1.0 implementation |
| `src/thesis_rl/sb3_extensions/features_extractors.py` | Planned modification | Strict flat bridge and config-built extractor |
| `src/thesis_rl/sb3_extensions/builders.py` | Planned modification | Algorithm ownership policy kwargs |
| `src/thesis_rl/sb3_extensions/checkpointing.py` | Planned addition | Atomic generation save/load |
| `conf/obs/` and `conf/agent/planner/` | Planned modification | Frozen observation/encoder configurations |
| `tests/test_semantic_state_observation.py` and new focused tests | Planned modification/addition | Mandatory observation/encoder regression matrix |
| `docs/implementation/semantic_observation_encoder_v1_exec_plan.md` | Current file | Living traceability and validation record |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Static arithmetic check for semantic flat/token dimensions | `PASS` | `2026-07-16` | `2541` flat dimensions and `122` raw tokens confirmed during review. |
| `git diff --check` | `PASS` | `2026-07-16` | Documentation promotion and planning changes had no whitespace errors. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_scene_context.py tests/test_observation_schema_v11.py tests/test_encoders_v10.py tests/test_semantic_state_observation.py tests/test_sb3_extensions.py tests/test_sb3_direct_backends.py` | `PASS` | `2026-07-16` | 29 passed in 16.68s. |
| `docker compose run --rm dev uv run --no-sync ruff check <modified Python paths>` | `PASS` | `2026-07-16` | All checks passed. |
| `docker compose run --rm dev uv run --no-sync ruff format --check <modified Python paths>` | `PASS` | `2026-07-16` | 18 files already formatted after focused Ruff formatting. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_causal_context.py tests/test_causal_scene_context.py tests/test_rulebook_v2_wrapper.py` | `PASS` | `2026-07-16` | 5 passed in 1.90s after the causal-context publication change. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_catalog_eligibility.py tests/test_scenario_records.py tests/test_scenario_catalog.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_contracts.py` | `PASS` | `2026-07-17` | 55 passed in 5.11s. |
| `git diff --check` | `PASS` | `2026-07-16` | Final focused diff has no whitespace errors. |
| `make smoke` | `PASS` | `2026-07-17` | TD3/LiDAR identity baseline completed 2,000 training steps, two intermediate evaluations, and final evaluation. The full nine-combination matrix remains pending until semantic runtime wiring is complete. |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/envs/observations/semantic_state_v2.py tests/test_semantic_state_v2.py && docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_semantic_state_v2.py tests/test_observation_schema_v11.py && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/semantic_state_v2.py src/thesis_rl/envs/observations/__init__.py src/thesis_rl/envs/factory.py tests/test_semantic_state_v2.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/semantic_state_v2.py src/thesis_rl/envs/observations/__init__.py src/thesis_rl/envs/factory.py tests/test_semantic_state_v2.py && git diff --check` | `PASS` | `2026-07-17` | 9 passed; Ruff clean; 4 files formatted; no whitespace errors. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_semantic_state_v2.py tests/test_observation_schema_v11.py tests/test_encoders_v10.py tests/test_semantic_state_observation.py tests/test_scenarionet_causal.py tests/test_thesis_scenario_env.py` | `PASS` | `2026-07-17` | 42 passed in 3.09s; new strict adapter and legacy compatibility regressions pass together. |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/factory.py src/thesis_rl/envs/observations/semantic_state_v2.py tests/test_semantic_state_v2.py && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/factory.py src/thesis_rl/envs/observations/semantic_state_v2.py tests/test_semantic_state_v2.py && git diff --check` | `PASS` | `2026-07-17` | Ruff clean, 3 files formatted, no whitespace errors. |

## 15. Final Reconciliation

`OBS-REQ-003`, `OBS-REQ-005`, and `ENC-REQ-003` are `PARTIAL`; `ENC-REQ-001`
and `ENC-REQ-002` are `IMPLEMENTED` and focused-`VERIFIED`. `OBS-REQ-003` now
has a strict schema-owned `(2541,)` runtime adapter and focused tests, but its
causal batch builder and live environment installation are still pending. All remaining
requirements remain `NOT_IMPLEMENTED` or `NOT_VERIFIED`. In particular,
full sensor smoke validation, the semantic runtime observation,
causal transition commit wiring, checkpoint generation publication/load, visual
diagnostics, and the nine-case smoke matrix remain required work. No
experimental use is authorized from this plan yet. The approved specifications
and ADR-002 remain authoritative; this ExecPlan records partial implementation
only and does not alter them.
