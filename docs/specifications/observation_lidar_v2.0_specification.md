# Rulebook-Aligned Stacked LiDAR Observation Specification

**Document ID:** OBS-LIDAR-V2.0
**Version:** 2.0-rulebook-aligned
**Status:** APPROVED
**Authoritative:** YES
**Date:** 2026-07-30
**Approval evidence:** explicit user approval 2026-07-30 ("procedi ad
implementare tutto il piano"), following gate-by-gate review of
`docs/implementation/lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md`
(`DEC-001` through `DEC-010`)
**Supersedes:** none. `OBS-V1.1` (`D_lidar,stack = 1540`, `stacked_lidar_state`)
remains `SUPERSEDED` and retained unchanged for reproducibility; this document
introduces a new mode, not a redefinition.
**Related decisions:** `ADR-026`, `ADR-033`, `ADR-036`
**Related documents:** `OBS-V1.1`, `ENC-V1.4`,
`docs/implementation/lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md`

## 1. Purpose and scope

This document specifies `stacked_lidar_v2`, a 21-frame stacked causal LiDAR
observation with a per-frame validity mask. Its purpose is to give the LiDAR
observation arm a temporal window sufficient to reconstruct every *bounded*
Rulebook v2 timer, replacing the historical five-frame window that predates
Rulebook v2 and has no derivation.

The per-frame sensor contract (`CausalLidarFrameBuilder`, 308-wide) is frozen
and unchanged. Only the stack depth and the addition of an explicit validity
mask are new.

## 2. Window derivation

The Rulebook's longest bounded memory is `DASHED_TCAP_S = 2.0` s
(`src/thesis_rl/rulebook/v2/components/road.py`), beyond which the dashed-line
cost factor saturates, so additional history is provably irrelevant to that
rule. At the 10 Hz control period (`decision_repeat=5`,
`physics_world_step_size=0.02`, `conf/env/scenarionet.yaml`), this window is
realized as:

```text
window_frames = round(DASHED_TCAP_S / control_period_s) + 1 = 21
```

The `+1` includes the current frame alongside the 20 preceding samples. This
matches `compliance_history_length = 21` on the semantic arm
(`src/thesis_rl/envs/observations/causal_semantic.py`), so both arms cover
the identical Rulebook-derived window.

## 3. Flat contract

```text
D = 21 * 308 + 21 = 6489
layout = [frame_0 (oldest) ... frame_20 (current)] ++ frame_mask(21)
frame  = ego(6) | navigation(22) | side(12) | lane(12) | nearby(16) | lidar(240)
```

`frame_mask[k] = 1.0` when frame `k` was genuinely observed, `0.0` during
episode-start warm-up. Absent frames are zero-filled, not replicated. This
distinguishes a genuinely static 2.0 s history from a warm-up transient, which
the legacy five-frame replication convention (`stacked_lidar_state`) could not
do at this depth (see `ADR-036`).

## 4. Behavioral contract

- The frame builder (`CausalLidarFrameBuilder`) is reused unchanged; this
  document changes no per-frame semantics, ray count, or normalization.
- The observation MUST NOT read any Rulebook-owned latch or timer, and MUST be
  bit-identical for a populated and an empty `RulebookMemory`. The frame
  builder reads only sensors and the route adapter, never `RulebookMemory`.
- Reset clears the frame deque; the mask is all-zero immediately after reset
  and saturates to all-ones after 21 steps.
- Frame order is oldest to current, matching `stacked_lidar_state`.
- The observation remains finite and within `[-1, 1]`.

## 5. Known representational limitations (declared, not defects)

1. **`yellow_must_stop` is not reconstructible.** This Rulebook latch has an
   unbounded lookback from the yellow-signal onset. `ADR-033` forbids exposing
   Rulebook-owned latches to the policy observation because the Rulebook also
   produces the reward; exposing the latch would be label leakage. This is an
   intrinsic representational gap of the LiDAR arm relative to the semantic
   arm (which reconstructs an equivalent quantity causally from its own
   observed history, not from `RulebookMemory`), scoped to the `signal`
   Rulebook component.
2. **Neighbor slots are not identity-stable across frames.** The nearby-vehicle
   block returns the *k*-nearest vehicles per frame, ranked by distance; slot
   *j* may denote a different physical vehicle from one frame to the next.
   `ENC-V1.4` channel-stacks this block across time without a per-slot
   identity embedding, following the `ADR-026` precedent (removal of a
   dynamic-slot identity embedding under the same first-fit/ranking
   instability). This is declared as a limitation, not corrected, because the
   affected block is a small fraction of the total observation and the LiDAR
   rays already carry the primary traffic-geometry signal.
3. **A sector index denotes a fixed ego-relative bearing**, not a fixed world
   bearing. Under a large ego yaw rate the per-sector temporal series in
   `ENC-V1.4` is not a world-fixed range series. This is a property of any
   ego-relative ranging sensor and is not corrected here.

## 6. Compatibility

- `stacked_lidar_state` (1540) and its `encoder_v1.0` MLP path are unchanged.
- `semantic_v3` (`OBS-V1.3`, `D=3009`) is unchanged.
- Observation dimension and mode name enter the checkpoint identity
  (`src/thesis_rl/runtime/wiring/checkpoint_identity.py`). No migration path
  from any existing checkpoint is offered, by design.

## 7. Acceptance criteria

- `AC-001` The observation has exactly 6489 finite `float32` values and covers
  21 samples at 0.1 s, i.e. 2.0 s.
- `AC-002` After reset, `frame_mask` is zero for every absent frame and absent
  frames are zero-filled; after 21 steps the mask is all ones.
- `AC-003` An observation built with a populated `RulebookMemory` is
  bit-identical to one built with an empty memory.
- `AC-004` The per-frame layout and the 240/12/12/4 counts are unchanged.
- `AC-010` Every value is finite and within `[-1, 1]`.

## 8. Implementation

`src/thesis_rl/envs/observations/stacked_lidar_v2.py`
(`StackedLidarObservationV2`), dispatched via `obs=stacked_lidar_v2`
(`conf/obs/stacked_lidar_v2.yaml`, `src/thesis_rl/envs/factory.py`).
