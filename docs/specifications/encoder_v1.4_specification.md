# LiDAR Latent-Query Encoder Specification

**Document ID:** ENC-V1.4
**Version:** 1.4-lidar-tokenized
**Status:** APPROVED
**Authoritative:** YES
**Date:** 2026-07-30
**Approval evidence:** explicit user approval 2026-07-30 ("procedi ad
implementare tutto il piano"), following gate-by-gate review of
`docs/implementation/lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md`
(`DEC-001` through `DEC-010`)
**Supersedes:** none. `ENC-V1.3` (semantic arm) is unchanged.
**Related decisions:** `ADR-026`, `ADR-033`, `ADR-036`
**Related documents:** `OBS-LIDAR-V2.0`,
`docs/implementation/lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md`

## 1. Purpose and scope

This document specifies `lq_lidar`, a latent-query encoder for the
`OBS-LIDAR-V2.0` observation (`D = 6489`). It reuses the frozen latent-query
core architecture unchanged (16 latents, depth 4, `token_dim = 64`,
`latent_dim = 128`, `num_heads = 4`, `ff_dim = 256`, `output_dim = 256`,
`pooling = mean`; `src/thesis_rl/agent/planners/encoders/lq_encoder.py`). The
only normative content of this document is the tokenization scheme: how the
6489 flat values are partitioned into tokens and projected to `token_dim`.

## 2. Tokenization

38 tokens, gathered at stride 308 (the per-frame width) across the 21 stacked
frames, with every one of the 6489 input values consumed by exactly one token
projection:

| Token group | Per frame | Stacked width | Count |
|---|---:|---:|---:|
| ego (+ mask appended) | 6 | 126 + 21 = 147 | 1 |
| navigation | 22 | 462 | 1 |
| side fan | 12 | 252 | 1 |
| lane fan | 12 | 252 | 1 |
| neighbor | 4 | 84 | 4 |
| LiDAR sector | 8 | 168 | 30 |
| **Total** | **308** | **6489** | **38** |

Partition check: `147 + 462 + 252 + 252 + 4*84 + 30*168 = 6489`.

Sector count follows STECA's sector-tokenization convention (§II-B):
`num_sectors = num_lidar_rays / 8 = 30`. Only the sector-tokenization and
circular-positional-encoding components of STECA are adopted; STECA's
two-stage attention (Stage-I sector self-attention, Stage-II ego-centric
cross-attention) is NOT adopted (see `ADR-036` for the evidentiary
justification: an unreplicated single-seed comparison, and V-Max's own
transformer-family plateau across LQ/LQH/MTR/Wayformer).

The 21-wide validity mask (`OBS-LIDAR-V2.0` §3) additionally zeroes absent
frame slices inside every token's stacked input before projection, so a
warm-up frame contributes exactly zero to every token, not a replicated value.

## 3. Circular positional encoding

Each of the 30 projected sector tokens receives a fixed (non-learned) additive
positional encoding:

```text
theta_i = 2*pi*i / 30
pe(i)   = tile([sin(theta_i), cos(theta_i)], token_dim/2)
```

This is exactly periodic with period 30 (`pe(i) == pe(i mod 30)`), and the
pairwise Euclidean distance between any two sector encodings is a strictly
monotone function of the circular sector distance
`min(|i-j|, 30-|i-j|)`, since it uses a single fundamental frequency rather
than the multi-harmonic Transformer convention.

## 4. Slot identity

No per-slot identity embedding is added to the neighbor block (4 slots) or the
sector block (30 slots beyond the positional encoding above). For the
neighbor block this follows the `ADR-026` precedent: the *k*-nearest ranking
that fills these slots has no stable per-slot identity across frames, so a
learned per-slot embedding could only encode a spurious correlation. The
sector block's positional encoding is not a learned identity embedding; it
encodes the sensor's fixed physical ring topology, which the slot index does
genuinely, stably index.

## 5. Input contract

The encoder MUST reject any input whose flat dimension is not `6489`. There is
no fallback to the legacy 1540-wide `stacked_lidar_state` contract; that
contract keeps its existing `encoder_v1.0` MLP path unchanged.

## 6. Checkpoint compatibility

`lq_lidar` weights are incompatible with every other encoder variant. The
first projection layers have different shapes, so loading fails structurally.
No migration path is offered, by design, matching `ENC-V1.3` §5.

## 7. Acceptance criteria

- `AC-005` The 38 token projections consume every one of the 6489 values
  exactly once.
- `AC-006` The circular encoding satisfies `PE(i) == PE(i mod 30)` and its
  pairwise distance is monotone in circular sector distance.
- `AC-007` The encoder accepts 38 tokens of width 64 and rejects any other
  token width or input dimension.

## 8. Implementation

`src/thesis_rl/agent/planners/encoders/lq_lidar/tokenizer.py` (gather, mask
application, circular positional encoding),
`src/thesis_rl/agent/planners/encoders/lq_lidar_encoder.py`
(`LatentQueryEncoderLidar`), registered as `lq_lidar` in
`src/thesis_rl/agent/planners/encoders/factory.py`; SB3 bridge allow-list in
`src/thesis_rl/sb3_extensions/builders.py`.
