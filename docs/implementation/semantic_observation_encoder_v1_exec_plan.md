# Semantic Observation And Encoder v1 ExecPlan

## 1. Metadata

- Plan ID: `PLAN-OBS-ENC-V1`
- Feature: semantic observation v1.1 and encoder v1.0 reconciliation
- Status: `DRAFT`
- Created: `2026-07-16`
- Last updated: `2026-07-16`
- Branch: `scenarionet-implementation` at `52e1794`
- Owner: thesis repository maintainer
- Authoritative specifications:
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`,
    version `1.1-final-implementation-complete`, `APPROVED`
  - `docs/specifications/encoder_v1.0_specification.md`, ID `ENC-V1.0`,
    version `1.0-final-implementation-complete`, `APPROVED`
- Related ADR: `docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md`

## 2. Objective And Scope

Implement the approved causal map-based observation contract and its MLP/LQ
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
| `OBS-REQ-001` | Build and freeze a map-based ego route at reset; reject invalid routes without native/future fallback. | OBS §2, §4, §14 |
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
| `DEC-005` | plan approval | Freeze this mandatory test matrix before production changes. | start coding / approve this plan first | Obtain explicit approval of this ExecPlan. | Test protection and milestone sequencing | Awaiting approval |

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
| `OBS-REQ-001` | `AC-OBS-001` | route/context modules, environment factory | `TEST-OBS-001`, `TEST-OBS-002` | Planned |
| `OBS-REQ-002` | `AC-OBS-002` | stacked LiDAR observation, map navigation, noise wrapper | `TEST-OBS-003`--`TEST-OBS-005` | Planned |
| `OBS-REQ-003` | `AC-OBS-003` | schema, semantic builder | `TEST-OBS-006`--`TEST-OBS-008` | Planned |
| `OBS-REQ-004` | `AC-OBS-004` | selection/history/context utilities | `TEST-OBS-009`--`TEST-OBS-012` | Planned |
| `OBS-REQ-005` | `AC-OBS-005` | context provider and transition wiring | `TEST-OBS-013`, `TEST-OBS-014` | Planned |
| `OBS-REQ-006` | `AC-OBS-006` | Hydra configs, manifest/logging helpers | `TEST-OBS-015`, `TEST-OBS-016` | Planned |
| `ENC-REQ-001` | `AC-ENC-001` | base/MLP encoder | `TEST-ENC-001`--`TEST-ENC-003` | Planned |
| `ENC-REQ-002` | `AC-ENC-002` | LQ tokenizer, embeddings, blocks | `TEST-ENC-004`--`TEST-ENC-007` | Planned |
| `ENC-REQ-003` | `AC-ENC-003` | SB3 extractor/builders/configs | `TEST-ENC-008`--`TEST-ENC-011` | Planned |
| `ENC-REQ-004` | `AC-ENC-004` | checkpointing | `TEST-ENC-012`--`TEST-ENC-014` | Planned |
| `ENC-REQ-005` | `AC-ENC-005` | presets and smoke driver | `TEST-ENC-015` | Planned |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-OBS-001` | Unit | Map route is independent of future SDC samples. | Same map/current task, altered future trajectory. | Identical route and observation. | `OBS-REQ-001` |
| `TEST-OBS-002` | Unit | Invalid route fails closed. | Missing lane, destination, or path. | Recorded exclusion/error before control. | `OBS-REQ-001` |
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

- [ ] **M1 — Schema and causal route/context.** Create the schema, map route,
  context interface, and deterministic unit tests. Depends on `DEC-005`.
- [ ] **M2 — Observation implementations.** Implement semantic v1.1 and
  stacked LiDAR observations, config, masks, noise, and regression tests.
  Depends on M1.
- [ ] **M3 — Encoder contract.** Implement MLP/LQ v1.0 and schema-driven
  tokenization with unit tests. Depends on M1.
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
- Next step: obtain explicit approval of this `DRAFT` ExecPlan before changing
  production code or protected tests.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/contracts/observation_schema.py` | Planned addition | Single v1.1 schema and fingerprint |
| `src/thesis_rl/contracts/encoder_contract.py` | Planned addition | Shared encoder validation/version contract |
| `src/thesis_rl/contracts/checkpoint_manifest.py` | Planned addition | Compatibility and generation manifest types |
| `src/thesis_rl/envs/observations/semantic_state.py` or successor | Planned modification | Semantic v1.1 observation |
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
| Focused observation/encoder tests | `NOT_RUN` | `2026-07-16` | Production implementation and v1.1 tests do not exist yet. |
| `make smoke` | `NOT_RUN` | `2026-07-16` | Must run after M2/M4 implementation. |

## 15. Final Reconciliation

All requirements and acceptance criteria are `NOT_IMPLEMENTED` and
`NOT_VERIFIED` pending M1--M5. No experimental use is authorized from this
plan yet. The approved specifications and ADR-002 are authoritative; this
ExecPlan is a draft implementation record and does not alter them.
