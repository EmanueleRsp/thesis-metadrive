# ExecPlan: LiDAR Arm Temporal Alignment and LQ Tokenization

## 1. Metadata

- **Feature**: Rulebook-aligned temporal stacking and latent-query tokenization
  for the causal LiDAR observation arm.
- **Plan ID**: `OBS-LIDAR-LQ-001`
- **Authoritative specification**: none yet for the target contract. The current
  LiDAR contract lives in `docs/specifications/observation_v1.1_specification.md`
  (`OBS-V1.1`, `D_lidar=308`, `D_lidar,stack=1540`), which `docs/project_index.md`
  records as `SUPERSEDED` for new semantic development and retained for
  reproducibility. Two new specifications are required and are themselves
  deliverables of this plan (see `DEC-001`).
- **Status**: `IMPLEMENTED`
- **Created**: 2026-07-30
- **Last updated**: 2026-07-30 (implementation)
- **Branch**: `scenarionet-implementation`
- **Related ADRs**: `ADR-022` (perception-bounded path), `ADR-026` (slot identity
  removal precedent), `ADR-033` (Rulebook latch exclusion from the policy
  observation), `ADR-002` (observation/encoder contract). A new ADR recording the
  baseline-arm decision is a deliverable, after approval.
- **Owner**: thesis repository maintainer

## 2. Objective And Scope

### Observable capability

The LiDAR observation arm gains a temporal window sufficient to reconstruct every
*bounded* Rulebook timer, and a structured tokenization that lets it be encoded
by the existing latent-query encoder instead of a flat MLP. This makes the LiDAR
arm a fair, literature-grounded reference for the semantic arm rather than an
arm handicapped by a historical parameter choice.

### Why it is needed

The current five-frame stack covers 0.4 s. That value predates the introduction
of Rulebook v2 timers and was chosen before any temporal rule existed, so it has
no derivation. Meanwhile the Rulebook's longest bounded memory is 2.0 s
(`DASHED_TCAP_S`), and the semantic arm already covers exactly that window with
`compliance_history_length = 21`. The asymmetry is therefore an accident, not a
designed hypothesis, and a comparison built on it is not defensible.

Separately, the LiDAR arm currently has no latent-query encoder path, so any
comparison against `semantic_v3` + `lq_v3` confounds the observation with the
encoder.

### Success recognition

- `D_lidar,stack` covers 2.0 s of history, derived from `DASHED_TCAP_S`.
- The stacked observation is consumable by `LatentQueryEncoder` as a token
  sequence, with every input value consumed by exactly one token projection.
- `semantic_v3` and the legacy 1540-wide `stacked_lidar_state` remain
  bit-for-bit unchanged.
- The replay-buffer configuration is identical across off-policy algorithms.

### In scope

- A new causal LiDAR observation contract at 21 frames with a validity mask.
- A new LiDAR tokenizer producing 38 tokens with circular positional encoding
  over the LiDAR ring.
- A new encoder variant wiring that tokenizer to the existing
  `LatentQueryEncoder`.
- Replay-buffer unification and a `fast`-profile buffer override.
- Two new specifications and one ADR.

### Out of scope

- Any change to `semantic_v3` / `OBS-V1.3` or to `ENC-V1.3`. The semantic arm
  already covers the Rulebook window through `compliance_history`; no change is
  required or permitted here.
- Any change to Rulebook v2 timers, latches, or thresholds.
- Restart or migration of the three existing production runs.
- The native `obs=lidar_state` (161-wide MetaDrive default), retained unchanged
  as a pure-MetaDrive reference used by the calibration tool.
- Reward, curriculum, and algorithm changes other than `buffer_size`.

### Compatibility constraints

- The new observation is a **new mode name**, not a redefinition. The existing
  `stacked_lidar_state` contract (1540) and its `encoder_v1.0` MLP path stay
  intact for reproducibility, mirroring how `OBS-V1.3` §7 preserves
  `semantic_v2`.
- Observation dimension and encoder identity enter the checkpoint identity via
  `src/thesis_rl/runtime/wiring/checkpoint_identity.py:37`. No migration path
  from any existing checkpoint is offered, by design.

## 3. Authoritative Requirements

No approved specification covers the target contract; the requirements below are
derived from verified repository constants and from approved decisions, and
become normative only once the two new specifications are approved.

| ID | Requirement | Source |
|---|---|---|
| `REQ-001` | The stacked LiDAR observation MUST cover a history window of at least the longest bounded Rulebook timer, 2.0 s. | `DASHED_TCAP_S = 2.0` (`src/thesis_rl/rulebook/v2/components/road.py:23`) |
| `REQ-002` | At 10 Hz control (`decision_repeat=5 × physics_world_step_size=0.02`), the window MUST be realized as 21 samples. | `conf/env/scenarionet.yaml:39-40` |
| `REQ-003` | The observation MUST distinguish an episode-start warm-up frame from a genuine repeated observation. | `DEC-004` |
| `REQ-004` | The observation MUST NOT read any Rulebook-owned latch or timer, and MUST be bit-identical for a populated and an empty `RulebookMemory`. | `ADR-033` §Decision 1/4 |
| `REQ-005` | The per-frame contract MUST remain the verified 308-wide causal layout `ego(6) | navigation(22) | side(12) | lane(12) | nearby(16) | lidar(240)`. | `src/thesis_rl/envs/observations/causal_lidar.py:57` |
| `REQ-006` | The tokenizer MUST emit 38 tokens and MUST consume every input value exactly once. | `DEC-003` |
| `REQ-007` | LiDAR sector tokens MUST carry a circular positional encoding over the 360° ring. | STECA §II-C; `DEC-005` |
| `REQ-008` | Token features MUST be projected to `token_dim = 64` per group. | `src/thesis_rl/agent/planners/encoders/lq_encoder.py:109` |
| `REQ-009` | The legacy `stacked_lidar_state` (1540) and `semantic_v3` (3009) contracts MUST remain unchanged. | Compatibility constraint above |
| `REQ-010` | `buffer_size` MUST be identical across off-policy algorithms. | `DEC-006` |
| `REQ-011` | The observation MUST remain finite and within `[-1, 1]`. | `src/thesis_rl/envs/observations/causal_lidar.py:60-61` |

## 4. Current Repository Analysis

| Item | Path / symbol | Label |
|---|---|---|
| Causal 308D frame builder | `src/thesis_rl/envs/observations/causal_lidar.py:13` `CausalLidarFrameBuilder` | `VERIFIED` |
| Frame layout | `causal_lidar.py:57` — `ego, navigation, side, lane, nearby, lidar` | `VERIFIED` |
| Frozen ray counts | `causal_lidar.py:29-32` — 240/12/12 and 4 nearby, validated with raises | `VERIFIED` |
| Five-frame stacker | `src/thesis_rl/envs/observations/stacked_lidar.py:14` — `FRAME_DIM=308`, `HISTORY_LENGTH=5`, `STACKED_DIM=1540` | `VERIFIED` |
| Warm-up padding | `stacked_lidar.py:61-62` — replicates the current frame `HISTORY_LENGTH` times | `VERIFIED` |
| Route block shared with the semantic arm | `causal_lidar.py:16` uses `MapRouteNavigationObservation22`; semantic arm uses the same adapter (`src/thesis_rl/envs/observations/assigned_route.py:50`) | `VERIFIED` |
| Route provenance | `map_match_sdc_track_to_task_route` (`src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py:210`), provenance `waymo_sdc_offline_task_annotation` | `VERIFIED` |
| Frame builder installation | `src/thesis_rl/envs/thesis_scenario_env.py:346,360,377` | `VERIFIED` |
| Observation dispatch | `src/thesis_rl/envs/factory.py:44-140`; `stacked_lidar` branch sets 240/12/12 and `num_others=4` at `:48-70` | `VERIFIED` |
| LQ encoder frozen hyperparameters | `src/thesis_rl/agent/planners/encoders/lq_encoder.py:109,113` — raises unless `token_dim=64`, `num_latents=16`, `latent_dim=128`, `depth=4`, `num_heads=4`, `ff_dim=256`, `output_dim=256`, `pooling="mean"` | `VERIFIED` |
| Semantic-side token helpers | `src/thesis_rl/agent/planners/encoders/lq/unflatten.py`, `lq/masks.py` | `VERIFIED` |
| Semantic rule window | `compliance_history_length = 21` (`src/thesis_rl/envs/observations/causal_semantic.py:1322`), deque `maxlen=21`, indexed `range(step-20, step+1)` at `:2335` | `VERIFIED` |
| Semantic ego window | `ego_history (5,10)` with `ego_history_mask (5,)`; masked padding at `causal_semantic.py:502-530` | `VERIFIED` |
| Rulebook bounded timers | `DASHED_T0_S=1.0`, `DASHED_TCAP_S=2.0` (`components/road.py:22-23`), `STOP_MIN_DWELL_S=1.0`, `CROSSWALK_GAP_S=1.0` (`components/controls.py:26-27`), `history_window_s=0.5` (`v2/config.py:40`) | `VERIFIED` |
| Rulebook unbounded latch | `yellow_must_stop` (`v2/types.py:308`), set at the yellow onset, unbounded lookback | `VERIFIED` |
| `optimize_memory_usage` forbidden | `src/thesis_rl/sb3_extensions/replay/config.py:61-62` raises; `replay/prioritized.py:101-102` requires false for PER + n-step | `VERIFIED` |
| SB3 buffer allocation | `third_party/stable-baselines3/stable_baselines3/common/buffers.py:198,213,217` — `buffer_size // n_envs` then `(buffer_size, n_envs, obs_dim)`; `next_observations` allocated separately | `VERIFIED` |
| Configured buffers | `sac_sb3.yaml:8` = 1000000; `td3_sb3.yaml:8` = 300000; legacy `sac.yaml`/`td3.yaml` = 300000 | `VERIFIED` |
| Profile budgets | `thesis` 1500000; `fast` 120000 (`conf/run_profile/*.yaml:8`) | `VERIFIED` |
| Profile planner override mechanism | `_resolve_planner_cfg` (`src/thesis_rl/runtime/wiring/builders.py:259-277`) merges `cfg.planner.{td3,sac,ppo}` onto `cfg.agent.planner.algorithm` | `VERIFIED` |
| Replay persistence | `replay_persistence: false` in every profile except `smoke` | `VERIFIED` |
| Machine capacity | 573 GiB RAM (417 GiB available), 72× Neoverse-V2, GH200 480GB (96 GiB VRAM, ~47 GiB free), 254 GB free disk | `VERIFIED` 2026-07-30 |

### Behavior to preserve

- `semantic_v3` observation and `lq_v3` encoder, unchanged.
- `stacked_lidar_state` at 1540 and its `encoder_v1.0` MLP path, unchanged.
- `obs=lidar_state` native MetaDrive path, unchanged.
- The `[-1, 1]` and finiteness contracts, and the ray-noise validation that
  native MetaDrive sensor noise stays disabled.

### Relevant debt

- `run_profile.planner.ppo.*` (a nested key) is inert; only the top-level
  `planner:` block is consumed. The misleading comment sits at
  `conf/presets/test/smoke_train_ppo.yaml:17-20`. Not addressed by this plan.

## 5. Assumptions And Invariants

| Item | Value | How established | Violation handling |
|---|---|---|---|
| Control period | 0.1 s | `VERIFIED` from `conf/env/scenarionet.yaml:39-40` | A profile changing `decision_repeat` invalidates `REQ-002`; the frame count must then be re-derived, not kept |
| History window | 2.0 s = 21 samples | Derived from `DASHED_TCAP_S` | Raise at construction if the configured frame count and the Rulebook cap disagree |
| Per-frame dimension | 308 | `VERIFIED`, frozen with raises | Existing raise |
| Stacked dimension | 6489 = 21·308 + 21 | This plan, `DEC-004` | Hard shape assertion, as today |
| Value range | `[-1, 1]`, finite | `VERIFIED` | Existing raise |
| Frame order | oldest → current | `VERIFIED` from `stacked_lidar.py:65` | Preserved; asserted by test |
| Temporal gather stride | 308 across 21 frames | This plan | Asserted by a synthetic index-encoded fixture |
| Sector count | 30 = 240/8 | STECA §II-B | Raise if `num_lidar_rays % 8 != 0` |
| Coordinate frame | ego-relative, per-frame | `VERIFIED` — each frame is built in the ego frame at its own timestep | Documented: a sector index denotes a fixed *ego-relative* bearing, not a fixed world bearing; under large yaw rate the per-sector temporal series is not a world-fixed range series |
| Reset behavior | frame deque cleared on reset | `VERIFIED` `stacked_lidar.py:45-47` | Preserved; mask asserted zero-filled after reset |
| Rulebook independence | no `RulebookMemory` read | `VERIFIED` — `CausalLidarFrameBuilder` reads only sensors and the route adapter | Regression test per `REQ-004` |
| Seeds | unchanged | Existing execution seeding | n/a |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification clarification | The target contract has no specification. `OBS-V1.1` holds the current LiDAR contract but is `SUPERSEDED` and retained for reproducibility. | A: amend `OBS-V1.1` in place. B: new `observation_lidar_v2.0_specification.md` (`OBS-LIDAR-V2.0`) plus `encoder_v1.4_specification.md` (`ENC-V1.4`). | **B** | B keeps the historical record intact and follows the repository's established versioning pattern; A would mutate a document other plans cite for reproducibility | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-002` | Specification deviation | History depth 5 → 21. | A: keep 5. B: 10. C: 20. D: 21. | **D** | 5 has no derivation and predates Rulebook timers; 21 is `DASHED_TCAP_S/0.1 + 1` exactly; 10 and 20 are arbitrary and 10 truncates the dashed rule | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-003` | Implementation detail | How the temporal axis enters the tokens. | A: per-frame tokens (5·38 = 190 → 21·38 = 798). B: channel-stacking, 38 tokens with 21× feature width. | **B** | B keeps the token count and therefore the frozen LQ contract untouched, and makes the per-direction range-rate a layer-0 linear operation; A inflates attention cost and forces attention to do temporal matching | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-004` | Specification clarification | Warm-up padding. At 5 frames, replication affected 4 steps; at 21 frames it affects 20 steps (2.0 s), making "just started" indistinguishable from "static for 2 s" — the exact discrimination the Rulebook timers require. | A: keep replication (`D = 6468`). B: add a 21-wide validity mask, zero absent frames, mirror `ego_history_mask` (`D = 6489`). | **B** | B removes a genuine ambiguity introduced by `DEC-002` and restores parity with the semantic arm's padding convention; A silently encodes a fake saturated timer at every episode start | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-005` | Implementation detail | Which STECA components to adopt. | A: sector tokens + circular PE only. B: also the two-stage attention (Stage-I sector self-attention, Stage-II ego cross-attention). | **A** | The circular PE encodes a real ring topology at zero cost; the two-stage split is supported only by an unreplicated single-seed comparison (26.0% vs 33.0%), and V-Max shows the transformer family on a plateau (LQ 0.87 vs LQH/MTR/Wayformer 0.84), so the existing learned latents suffice | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-006` | Specification deviation | `buffer_size` 1000000 (SAC) vs 300000 (TD3). | A: keep. B: unify at 300000. C: unify at 1000000. | **B** | The ACL makes the data distribution non-stationary by design, so a buffer holding 67% of the run retains superseded curriculum stages; the current asymmetry also confounds any SAC/TD3 comparison. Deviates from V-Max's 1e6, which trains without a curriculum — recorded as a justified deviation | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-007` | Implementation detail | `buffer_size` on diagnostic profiles: at `fast` (120k steps) a 300k buffer never evicts, so the regime differs from `thesis`. | A: leave. B: profile override `buffer_size: 24000` on `fast` (20% ratio preserved). | **B** | B makes the diagnostic run representative of the production replay regime; memory is not the motive (417 GiB available) | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-008` | Specification clarification | Neighbor tokens are not identity-stable: `get_surrounding_vehicles_info` returns the *k* nearest per frame, so slot *j* may be a different vehicle across frames, while channel-stacking implies continuity. | A: declare as a limitation. B: per-frame tokens for the neighbor block only (38 → 54 tokens). | **A** | Follows the `ADR-026` precedent, which removed a slot identity embedding as spurious under first-fit assignment rather than engineering identity; the block is 84 of 6468 values, so B is disproportionate | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-009` | Specification clarification | `yellow_must_stop` has unbounded lookback and cannot be reconstructed from any finite window. | A: expose the Rulebook latch. B: declare the gap, scope it to the `signal` component, quantify its catalog incidence, and report a stratified robustness check. | **B** | A is forbidden by `ADR-033`: the Rulebook produces the reward, so exposing its latch is label leakage that makes the central thesis claim unfalsifiable. B converts an intrinsic representational difference into a measured result | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |
| `DEC-010` | Implementation detail | Naming of the new modes. | A: redefine `stacked_lidar_state` / `lq` in place. B: new `stacked_lidar_v2` observation and `lq_lidar` encoder, legacy modes untouched. | **B** | Mirrors `OBS-V1.3` §7 legacy-mode preservation; A breaks the 1540 contract other documents cite | Approved 2026-07-30 ("procedi ad implementare tutto il piano") |

No dependent production work starts while these gates are open.

## 7. Proposed Design

### Observation

New `StackedLidarObservationV2` alongside the existing five-frame class.
`CausalLidarFrameBuilder` is reused **unchanged** — the per-frame contract is
already correct and frozen; only the stack depth and the mask are new.

```
D = 21 · 308 + 21 = 6489
layout = [frame_0 (oldest) … frame_20 (current)] ++ frame_mask(21)
frame  = ego(6) | navigation(22) | side(12) | lane(12) | nearby(16) | lidar(240)
```

`frame_mask[k] = 1.0` when frame *k* was genuinely observed, `0.0` during
warm-up; absent frames are zero-filled rather than replicated (`DEC-004`).

### Tokenizer

New `LidarTokenizer` producing 38 tokens by gathering each group across frames at
stride 308, then projecting per group to `token_dim = 64`:

| Token group | Per frame | Stacked width | Count |
|---|---:|---:|---:|
| ego (+ mask appended) | 6 | 126 + 21 = 147 | 1 |
| navigation | 22 | 462 | 1 |
| side fan | 12 | 252 | 1 |
| lane fan | 12 | 252 | 1 |
| neighbor | 4 | 84 | 4 |
| LiDAR sector | 8 | 168 | 30 |
| **Total** | **308** | **6489** | **38** |

Partition check: `147 + 462 + 252 + 252 + 4·84 + 30·168 = 6489`. Every input
value is consumed exactly once (`REQ-006`).

A circular sine–cosine positional encoding over `θ_i = 2πi/30` is added to each
projected sector token (`REQ-007`). The mask additionally zeroes absent frame
slices inside every token's stacked input before projection.

### Encoder

New `lq_lidar` variant reusing `LatentQueryEncoder` untouched: 16 latents,
depth 4, `token_dim=64`, output 256. Only the group projections and the
tokenizer are new, mirroring how `ENC-V1.3` carries four token-projection widths
for the semantic path. A LiDAR counterpart to `lq/unflatten.py` and
`lq/masks.py` provides the strided gather and the mask application.

### Configuration

- `conf/obs/stacked_lidar_v2.yaml`, and a `factory.py` branch reusing the frozen
  240/12/12/4 sensor configuration.
- `conf/agent/planner/algorithm/sac_sb3.yaml`: `buffer_size` 1000000 → 300000.
- `conf/run_profile/fast.yaml`: `planner.{td3,sac}.buffer_size: 24000`.

### Errors and fallbacks

No new fallbacks. Shape, finiteness, and range violations raise, as today. A
configured frame count inconsistent with `DASHED_TCAP_S` and the control period
raises at construction rather than degrading silently.

### Rejected alternatives

Per-frame tokenization (`DEC-003` A); STECA's two-stage attention (`DEC-005` B);
strided or mixed-stride stacking — motivated only by a memory budget that the
verified machine capacity shows does not bind, and which would have required
declaring a timer-reconstruction error of ±0.2 s.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001`, `REQ-002` | `AC-001` | `envs/observations/stacked_lidar_v2.py` | `TEST-001`, `TEST-002` | Planned |
| `REQ-003` | `AC-002` | same | `TEST-003`, `TEST-004` | Planned |
| `REQ-004` | `AC-003` | `envs/observations/causal_lidar.py` (unchanged) | `TEST-005` | Planned |
| `REQ-005` | `AC-004` | same | `TEST-006` | Planned |
| `REQ-006` | `AC-005` | `agent/planners/encoders/lq_lidar/tokenizer.py` | `TEST-007`, `TEST-008` | Planned |
| `REQ-007` | `AC-006` | same | `TEST-009`, `TEST-010` | Planned |
| `REQ-008` | `AC-007` | `agent/planners/encoders/factory.py` | `TEST-011`, `TEST-012` | Planned |
| `REQ-009` | `AC-008` | none (regression guard) | `TEST-013`, `TEST-014` | Planned |
| `REQ-010` | `AC-009` | `conf/agent/planner/algorithm/sac_sb3.yaml`, `conf/run_profile/fast.yaml` | `TEST-015`, `TEST-016` | Planned |
| `REQ-011` | `AC-010` | `stacked_lidar_v2.py` | `TEST-017` | Planned |

### Acceptance criteria

- `AC-001` The observation has exactly 6489 finite `float32` values and covers
  21 samples at 0.1 s, i.e. 2.0 s.
- `AC-002` After reset, `frame_mask` is zero for every absent frame and absent
  frames are zero-filled; after 21 steps the mask is all ones.
- `AC-003` An observation built with a populated `RulebookMemory` is
  bit-identical to one built with an empty memory.
- `AC-004` The per-frame layout and the 240/12/12/4 counts are unchanged.
- `AC-005` The 38 token projections consume every one of the 6489 values exactly
  once.
- `AC-006` The circular encoding satisfies `PE(i) == PE(i mod 30)` and its
  distance is monotone in circular sector distance.
- `AC-007` `LatentQueryEncoder` accepts 38 tokens of width 64 and rejects any
  other token width.
- `AC-008` `semantic_v3` still emits 3009 values and `stacked_lidar_state` still
  emits 1540.
- `AC-009` Both off-policy algorithms resolve to `buffer_size = 300000`, and
  `run_profile=fast` resolves to 24000.
- `AC-010` Every value is finite and within `[-1, 1]`.

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Flat dimension | Constructed observation | `shape == (6489,)` | `REQ-001` |
| `TEST-002` | Unit | Window derivation | `DASHED_TCAP_S`, control period | Frame count 21; mismatch raises | `REQ-001`, `REQ-002` |
| `TEST-003` | Unit | Reset/mask | Fresh reset, 1 step | Mask `[0]*20 + [1]`; absent frames exactly zero | `REQ-003` |
| `TEST-004` | Unit | Mask saturation | 21 steps | Mask all ones; frame order oldest→current | `REQ-003` |
| `TEST-005` | Unit | Causality | Same context, populated vs empty `RulebookMemory` | Bit-identical arrays | `REQ-004` |
| `TEST-006` | Unit | Per-frame layout | Synthetic sensors | 308-wide frame, group offsets unchanged | `REQ-005` |
| `TEST-007` | Unit | Token count | Tokenizer on a 6489 vector | 38 tokens of width 64 | `REQ-006` |
| `TEST-008` | Unit | Exact partition | Index-encoded input (value = flat index) | Union of gathered indices == `range(6489)`, no duplicates; stride 308 verified | `REQ-006` |
| `TEST-009` | Unit | Circular PE wrap | Sector indices 0 and 30 | Identical encodings | `REQ-007` |
| `TEST-010` | Unit | Sector cyclic permutation | Scene rotated by 360/30° | Sector tokens permute cyclically | `REQ-007` |
| `TEST-011` | Unit | Encoder acceptance | 38×64 tokens | Forward returns `[B, 256]` | `REQ-008` |
| `TEST-012` | Unit | Encoder rejection | Token width ≠ 64 | Raises | `REQ-008` |
| `TEST-013` | Regression | Semantic unchanged | `semantic_v3` builder | 3009 values, existing assertions pass | `REQ-009` |
| `TEST-014` | Regression | Legacy LiDAR unchanged | `stacked_lidar_state` | 1540 values, replication padding preserved | `REQ-009` |
| `TEST-015` | Config | Buffer unification | Hydra compose, SAC and TD3 | Both 300000 | `REQ-010` |
| `TEST-016` | Config | Profile override | `run_profile=fast` | 24000 via `_resolve_planner_cfg` | `REQ-010` |
| `TEST-017` | Numerical | Range and finiteness | Randomized sensor returns | All finite, all in `[-1, 1]`; violation raises | `REQ-011` |
| `TEST-018` | Determinism | Repeatability | Same seed, twice | Identical observation sequences | `REQ-001` |
| `TEST-019` | Integration | Checkpoint identity | Build/save/load | Identity records `D=6489` and `lq_lidar`; mismatch refuses load | Compatibility |
| `TEST-020` | Smoke | End-to-end | `obs=stacked_lidar_v2`, `encoder=lq_lidar`, `run_profile=fast` | Training runs and evaluates without error | All |

Boundary, invalid, missing-data, mask/padding, reset, determinism, numerical,
compatibility, causality, integration, and regression categories are covered
above. Truncation/termination semantics are untouched by this plan and are
therefore not re-tested here.

### Commands

| Purpose | Command |
|---|---|
| Focused tests | `uv run --no-sync python -m pytest -q tests/test_stacked_lidar_v2.py tests/test_lq_lidar_tokenizer.py` |
| Regressions | `uv run --no-sync python -m pytest -q tests/test_semantic_state_v3.py tests/test_stacked_lidar_observation.py tests/test_encoders_v11.py` |
| Full suite | `make test` |
| Lint | `make lint` |
| Focused format check | `make format-check PYTHON_QUALITY_PATHS="<changed files>"` |
| Compose validation | `make config` and `make config-gpu` |
| Smoke | `make smoke` |
| Whitespace | `git diff --check` |

No mypy target exists repository-wide; new public interfaces carry annotations
but no global static-check gate is claimed.

## 10. Milestones

### M0 — Specifications and ADR (blocked by every gate)

- [x] Draft `observation_lidar_v2.0_specification.md` (`OBS-LIDAR-V2.0`) and
      `encoder_v1.4_specification.md` (`ENC-V1.4`).
- [x] Draft the ADR (`ADR-036`) recording the baseline-arm decision, STECA
      scoped to LiDAR tokenization only.
- [x] Flip to `APPROVED`/`Authoritative: YES` directly (the user's "procedi ad
      implementare tutto il piano" is the approval event; no separate
      `UNDER_REVIEW` round-trip was requested), update `docs/project_index.md`.
- Evidence: `docs/specifications/observation_lidar_v2.0_specification.md`,
  `docs/specifications/encoder_v1.4_specification.md`,
  `docs/decisions/ADR-036-lidar-arm-temporal-alignment-and-lq-tokenization.md`,
  `docs/project_index.md` rows.

### M1 — Observation at 21 frames with mask

- [x] `StackedLidarObservationV2`, `conf/obs/stacked_lidar_v2.yaml`,
      `factory.py` branch. `thesis_scenario_env.py` wiring was NOT needed:
      `_install_causal_observation_builder` already installs the frame builder
      generically on any observation exposing `set_frame_builder`, which
      `StackedLidarObservationV2` implements identically to the legacy class.
- [x] `TEST-001`–`TEST-006`, `TEST-017`, `TEST-018` (`tests/test_stacked_lidar_v2.py`).
- Depends on `DEC-001`, `DEC-002`, `DEC-004`, `DEC-010` — all approved.

### M2 — Tokenizer with circular positional encoding

- [x] `LidarTokenizer` (`agent/planners/encoders/lq_lidar/tokenizer.py`),
      strided gather, mask application, circular PE.
- [x] `TEST-007`–`TEST-010` (`tests/test_lq_lidar_tokenizer.py`).
- Depends on `DEC-003`, `DEC-005`, `DEC-008` — all approved.

### M3 — Encoder variant

- [x] `lq_lidar` (`LatentQueryEncoderLidar`) registered in the encoder factory
      with six group projections; `LatentQueryEncoder`'s core building blocks
      (`_LatentQueryBlock`) reused unchanged.
- [x] `TEST-011`, `TEST-012` (`tests/test_lq_lidar_tokenizer.py`), `TEST-019`
      (`tests/test_checkpoint_manifest_sidecar.py`, two new tests).
- Additional, not originally listed: the SB3 feature-extractor bridge
  allow-list (`src/thesis_rl/sb3_extensions/builders.py`) had to be extended
  with `lq_lidar`/`latent_query_lidar`, discovered only when attempting an
  end-to-end smoke run (M5). Without it, `td3_sb3`/`sac_sb3` refuse to build a
  policy with this encoder even though `build_encoder_for_env` already
  supports it generically.

### M4 — Replay configuration

- [x] `sac_sb3.yaml` buffer 300000; `fast.yaml` override 24000 (both `td3` and
      `sac` planner blocks).
- [x] `TEST-015`, `TEST-016` (`tests/test_replay_buffer_config_parity.py`),
      `make config`, `make config-gpu`.
- Depends on `DEC-006`, `DEC-007` — both approved.

### M5 — Regression and validation

- [x] `TEST-013`, `TEST-014` (existing `test_semantic_state_v3.py`,
      `test_stacked_lidar_observation.py` regressions, unchanged and passing);
      full suite (1174 tests after the fixes below), lint, focused
      format-check, `make config` / `make config-gpu`, `git diff --check`.
- [x] `TEST-020` (end-to-end smoke): now **passes**. Attempting
      `obs=stacked_lidar_v2 agent/planner/encoder=lq_lidar` against a real
      scenario surfaced three pre-existing defects in the declared-frozen
      `CausalLidarFrameBuilder`/`RayNoiseWrapper` (two used
      `isinstance(config, dict)` against a real MetaDrive `Config` object,
      which is never a `dict` instance; the third was `_lidar_blocks`
      assuming `sensor.perceive()` returns an object with `.cloud_points`/
      `.detected_objects` attributes, when MetaDrive's `Lidar.perceive`
      actually returns a plain `(cloud_points, detected_objects)` tuple —
      confirmed against `third_party/metadrive/metadrive/component/sensors/
      lidar.py`. All three fixed here, in scope, as bug fixes with
      regression tests, since they blocked the LiDAR arm's test matrix
      entirely and reproduced identically against the unmodified legacy
      `stacked_lidar_state`). End-to-end smoke confirmed passing (2000 steps,
      final evaluation completed with no error) for both `obs=stacked_lidar_v2
      agent/planner/encoder=lq_lidar` and, separately, the legacy
      `obs=stacked_lidar_state agent/planner/encoder=none`, under
      `presets/test/smoke_train`.
- Depends on M1–M4.

### M6 — Reconciliation

- [x] Requirement-by-requirement reconciliation (§15), `docs/project_index.md`
      updated, final diff reviewed.

## 11. Progress And Findings Log

**2026-07-30 — plan created.**

Findings established while scoping, all `VERIFIED` against the sources cited in
§4:

1. The five-frame depth has no derivation. It predates Rulebook v2 timers and was
   selected when no temporal rule existed. This is the finding that motivates the
   plan; the earlier framing of the LiDAR arm's short window as a *designed*
   hypothesis was wrong.
2. The Rulebook's longest bounded memory is 2.0 s (`DASHED_TCAP_S`), beyond which
   the dashed-line cost factor saturates, so longer history is provably
   irrelevant to that rule. 21 samples at 10 Hz follow.
3. `compliance_history_length = 21` is therefore not arbitrary either: the
   semantic arm already covers exactly this window. No semantic change is needed.
4. Both arms already share the identical route representation
   (`MapRouteNavigationObservation22`), and both derive it from the map-matched
   SDC track. Route-level privileged information is symmetric across arms and
   matches V-Max's convention (10 waypoints at 5 m spacing).
5. `optimize_memory_usage` is forbidden by construction for PER + n-step, so the
   2× observation cost in the replay buffer is structural.
6. Machine capacity removes the memory constraint entirely (417 GiB available
   against 15.6 GB for the buffer), which is what allows uniform 21-frame
   stacking instead of a strided compromise with a declared reconstruction error.
7. Raising the depth to 21 turns the existing replication padding into a real
   defect: 20 warm-up steps of a replicated frame are indistinguishable from a
   genuinely static 2.0 s history, which is exactly the discrimination the
   Rulebook timers require. Hence `DEC-004`.
8. `ADR-033` already settles the `yellow_must_stop` question: exposing a
   Rulebook-owned latch is label leakage. The gap is a consequence of an approved
   decision, not an open design question.

**2026-07-30 — implementation.**

9. User instruction "procedi ad implementare tutto il piano" approved all ten
   gates at the recommended option, including `DEC-008` after a clarifying
   explanation (k-nearest neighbor slots have no cross-frame identity; the
   `ADR-026` precedent — declare, don't engineer identity — applies directly).
10. While validating M5's end-to-end smoke acceptance criterion, discovered
    that `CausalLidarFrameBuilder.build` and
    `RayNoiseWrapper.validate_native_noise_disabled` reject every real
    MetaDrive vehicle: `vehicle.config` and per-sensor sub-configs are
    `metadrive.utils.config.Config` objects, never `dict` instances, so the
    existing `isinstance(config, dict)` checks always raised. This had never
    been caught because the existing unit tests for both modules only ever
    passed plain `dict` mocks. This silently meant the LiDAR observation arm
    (`stacked_lidar_state`, in production since before this plan, and the new
    `stacked_lidar_v2`) had never been exercised in a real training loop.
    Fixed by duck-typing on `.get` instead of the nominal type, with
    regression tests added to `tests/test_causal_lidar.py` and
    `tests/test_ray_noise.py`.
11. A further, third defect surfaced immediately after (a shape mismatch in
    `_lidar_blocks`'s handling of the real LiDAR sensor's `perceive()`
    return value: `getattr(result, "cloud_points", result)` silently fell
    back to `result` itself because `metadrive.component.sensors.lidar.
    Lidar.perceive` returns a plain `(cloud_points, detected_objects)`
    tuple, not the `detect_result` namedtuple `DistanceDetector.perceive`
    returns for `side_detector`/`lane_line_detector` — confirmed by reading
    `third_party/metadrive/metadrive/component/sensors/lidar.py`. This also
    meant `detected` was always `None`, so the nearby-vehicle block was
    silently computed with no detected vehicles. Fixed by unpacking the
    tuple positionally instead of via `getattr`; the existing test fixture
    in `tests/test_causal_lidar.py` was updated to distinguish the two real
    sensor return shapes rather than uniformly faking the namedtuple one.
    `TEST-020` (end-to-end smoke) now passes for both `stacked_lidar_v2` and
    the legacy `stacked_lidar_state`.
12. Attempting the smoke run also surfaced that
    `src/thesis_rl/sb3_extensions/builders.py`'s SB3 bridge encoder allow-list
    did not include `lq_lidar`, even though the underlying
    `build_encoder_for_env`/`ThesisEncoderFeatureExtractor` path already
    supported it generically. Added `lq_lidar`/`latent_query_lidar` to the
    allow-list (no new logic; matches how every other LQ variant is already
    listed).

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-001` | `OBS-V1.1` `D_lidar,stack = 1540` | New contract at 6489 under a new mode name; 1540 retained | `DEC-002`, `DEC-004` | Approved 2026-07-30 | `OBS-LIDAR-V2.0`, `TEST-014` |
| `DEV-002` | V-Max buffer size 1e6 | 300000 | ACL-induced non-stationarity; V-Max trains without a curriculum | Approved 2026-07-30 (`DEC-006`) | `ADR-036` |
| `DEV-003` | STECA full architecture | Sector tokens and circular PE only | Two-stage attention unsupported by replicated evidence | Approved 2026-07-30 (`DEC-005`) | `ADR-036` |
| `DEV-004` | `causal_lidar.py`/`ray_noise.py` treated as frozen, reused unchanged | Three fixes: two `isinstance(config, dict)` checks changed to duck-typed `.get` checks, plus `_lidar_blocks` changed to unpack the real `Lidar.perceive()` tuple positionally instead of `getattr(result, "cloud_points", result)` | All three defects rejected or silently mishandled every real MetaDrive vehicle/sensor call; discovered while validating M5, reproduce identically against the unmodified legacy `stacked_lidar_state`; pure bug fixes with no effect on any *correctly* produced observation value (the third fix also corrects a silent `detected_objects=None` that made the nearby-vehicle block always empty) | Implemented as an in-scope bug fix (AGENTS.md: every discovered bug requires a regression test) | `tests/test_causal_lidar.py`, `tests/test_ray_noise.py`, `ADR-036` §Consequences |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/specifications/observation_lidar_v2.0_specification.md` | Created | `OBS-LIDAR-V2.0` contract |
| `docs/specifications/encoder_v1.4_specification.md` | Created | `ENC-V1.4` contract |
| `docs/decisions/ADR-036-lidar-arm-temporal-alignment-and-lq-tokenization.md` | Created | Baseline-arm decision record |
| `src/thesis_rl/envs/observations/stacked_lidar_v2.py` | Created | 21-frame stack with mask (`StackedLidarObservationV2`) |
| `src/thesis_rl/envs/factory.py` | Modified | `stacked_lidar_v2` dispatch branch |
| `src/thesis_rl/agent/planners/encoders/lq_lidar/__init__.py` | Created | Package marker |
| `src/thesis_rl/agent/planners/encoders/lq_lidar/tokenizer.py` | Created | Strided gather, mask application, circular PE (38-token partition) |
| `src/thesis_rl/agent/planners/encoders/lq_lidar_encoder.py` | Created | `LatentQueryEncoderLidar` |
| `src/thesis_rl/agent/planners/encoders/factory.py` | Modified | Register `lq_lidar` / `latent_query_lidar` |
| `src/thesis_rl/sb3_extensions/builders.py` | Modified | Add `lq_lidar`/`latent_query_lidar` to the SB3 bridge allow-list (found necessary during M5) |
| `src/thesis_rl/envs/observations/causal_lidar.py` | Modified | Bug fixes: duck-type the vehicle-config mapping check instead of `isinstance(..., dict)`; unpack `Lidar.perceive()`'s real tuple return positionally instead of `getattr` (`DEV-004`) |
| `src/thesis_rl/envs/observations/ray_noise.py` | Modified | Bug fix: duck-type the per-sensor config mapping check instead of `isinstance(..., dict)` (`DEV-004`) |
| `conf/obs/stacked_lidar_v2.yaml` | Created | Observation selection |
| `conf/agent/planner/encoder/lq_lidar.yaml` | Created | Encoder selection |
| `conf/agent/planner/algorithm/sac_sb3.yaml` | Modified | `buffer_size` 1000000 → 300000 |
| `conf/run_profile/fast.yaml` | Modified | `planner.{td3,sac}.buffer_size: 24000` |
| `tests/test_stacked_lidar_v2.py` | Created | `TEST-001`–`TEST-006`, `TEST-017`, `TEST-018` |
| `tests/test_lq_lidar_tokenizer.py` | Created | `TEST-007`–`TEST-012` |
| `tests/test_replay_buffer_config_parity.py` | Created | `TEST-015`, `TEST-016` |
| `tests/test_checkpoint_manifest_sidecar.py` | Modified | Added `TEST-019` (two tests) |
| `tests/test_causal_lidar.py` | Modified | Regression tests for `DEV-004` |
| `tests/test_ray_noise.py` | Modified | Regression tests for `DEV-004` |
| `docs/project_index.md` | Modified | Index rows for `OBS-LIDAR-V2.0`, `ENC-V1.4`, `ADR-036` |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `uv run --no-sync python -m pytest -q tests/test_stacked_lidar_v2.py tests/test_lq_lidar_tokenizer.py` | `PASS` (18/18) | 2026-07-30 | `TEST-001`–`TEST-012`, `TEST-017`, `TEST-018` |
| `uv run --no-sync python -m pytest -q tests/test_replay_buffer_config_parity.py tests/test_hydra_preset_run_configs.py` | `PASS` (22/22) | 2026-07-30 | `TEST-015`, `TEST-016` plus existing profile-composition regressions |
| `uv run --no-sync python -m pytest -q tests/test_semantic_state_v3.py tests/test_stacked_lidar_observation.py tests/test_encoders_v11.py` | `PASS` (14/14) | 2026-07-30 | `TEST-013`, `TEST-014` — `semantic_v3`/legacy LiDAR unchanged |
| `uv run --no-sync python -m pytest -q tests/test_checkpoint_manifest_sidecar.py` | `PASS` (8/8) | 2026-07-30 | `TEST-019` |
| `uv run --no-sync python -m pytest -q tests/test_causal_lidar.py tests/test_ray_noise.py` | `PASS` (9/9) | 2026-07-30 | `DEV-004` regression tests |
| `uv run --no-sync python -m pytest -q` (full suite, re-run after the third `DEV-004` fix) | `PASS` (1174/1174, 1 pre-existing unrelated warning) | 2026-07-30 | No regression anywhere in the repository |
| `ruff check` / `ruff format --check` on `causal_lidar.py` and `test_causal_lidar.py` after the third fix | `PASS` | 2026-07-30 | Both files already formatted, no lint findings |
| `ruff check src tests scripts` | `PASS` | 2026-07-30 | `make lint` scope, clean |
| `ruff format --check` on every new/modified file in this plan | `PASS` after formatting the 4 newly created Python files (`lq_lidar/tokenizer.py`, `lq_lidar_encoder.py`, both new test files) | 2026-07-30 | `factory.py`'s pre-existing format debt (documented repository-wide baseline) is untouched by this plan's edits, confirmed via `ruff format --diff` showing only unrelated pre-existing lines |
| `make config` / `make config-gpu` | `PASS` | 2026-07-30 | Both compose files validate with the new config keys |
| `git diff --check` | `PASS` (no output) | 2026-07-30 | No whitespace errors |
| `uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train obs=stacked_lidar_v2 agent/planner/encoder=lq_lidar` | `PASS` | 2026-07-30 | `TEST-020`. Ran to completion (2000 steps, `Final Evaluation` printed) after all three `DEV-004` fixes; `CausalLidarFrameBuilder`/`RayNoiseWrapper` now work against the real MetaDrive vehicle/sensor API |
| `uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train obs=stacked_lidar_state` | `PASS` | 2026-07-30 | Legacy arm, confirming the `DEV-004` fixes unblock it identically and introduce no regression to the frozen 1540-wide contract |

## 15. Final Reconciliation

| Requirement | Status | Evidence |
|---|---|---|
| `REQ-001`, `REQ-002` | Met | `StackedLidarObservationV2.HISTORY_LENGTH = 21`; `TEST-002` asserts the `DASHED_TCAP_S`-derived count |
| `REQ-003` | Met | `frame_mask`, zero-fill on warm-up; `TEST-003`, `TEST-004` |
| `REQ-004` | Met | `StackedLidarObservationV2`/`CausalLidarFrameBuilder` never read `RulebookMemory`; `TEST-005` |
| `REQ-005` | Met | `CausalLidarFrameBuilder` reused unchanged; `TEST-006` |
| `REQ-006` | Met | `tokenizer.py` partition verified by index-encoded fixture; `TEST-008` |
| `REQ-007` | Met | Circular sin/cos PE, period exactly 30; `TEST-009`, `TEST-010` |
| `REQ-008` | Met | All group projections target `token_dim=64`; `TEST-011` |
| `REQ-009` | Met | `semantic_v3` (3009) and `stacked_lidar_state` (1540) regression-tested unchanged; `TEST-013`, `TEST-014` |
| `REQ-010` | Met | `buffer_size=300000` for both `td3_sb3`/`sac_sb3`, `24000` under `fast`; `TEST-015`, `TEST-016` |
| `REQ-011` | Met | Existing finite/`[-1,1]` raises preserved and tested; `TEST-017` |

Every requirement in this plan's authoritative table is met and tested,
including `TEST-020` (end-to-end smoke, both `stacked_lidar_v2` and the
legacy `stacked_lidar_state`). `M0`–`M6` are all complete. No acceptance item
remains outstanding.

### ChatGPT project source synchronization

`docs/project_index.md` changed in this session (two new rows plus one new
ADR registry row). `docs/engineering_workflow.md` and
`docs/templates/specification_template.md` were not touched.

### Known limitations carried by design

1. `yellow_must_stop` is not reconstructible from any finite window and is
   forbidden as a policy input by `ADR-033`. Scoped to the `signal` component,
   to be quantified on the catalog and reported as a stratified robustness
   check (`DEC-009`).
2. Neighbor slots are not identity-stable across frames (`DEC-008`).
3. A sector index denotes a fixed ego-relative bearing; under large yaw rate the
   per-sector temporal series is not a world-fixed range series.
4. Semantic tracking and classification remain ideal after physical admission,
   as carried forward from `OBS-V1.3` §10.
