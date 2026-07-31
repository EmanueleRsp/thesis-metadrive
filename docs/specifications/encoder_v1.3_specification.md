# Perception-Bounded Encoder Specification

**Document ID:** ENC-V1.3
**Version:** 1.3-perception-bounded
**Status:** APPROVED
**Authoritative:** YES
**Date:** 2026-07-29
**Approval evidence:** explicit user approval 2026-07-29, together with OBS-V1.3
**Supersedes:** ENC-V1.1 and ENC-V1.2 for the OBS-V1.3 observation path
**Related decisions:** ADR-022, ADR-026, ADR-033
**Note:** two token groups are renamed by OBS-V1.3 `DEC-011`; the widths and the token order are unaffected.
**Related documents:** OBS-V1.3, `docs/implementation/semantic_observation_causal_correctness_v1.3_exec_plan.md`

## 1. Purpose and scope

ENC-V1.3 adapts the ENC-V1.2 encoder to the OBS-V1.3 field lists. It changes no
architecture, no attention topology, no training procedure and no
hyperparameter. The only normative change is the per-group input width of four
token projections and the flat input dimension of the MLP variant.

Everything ENC-V1.1 and ENC-V1.2 specify about the latent-query block
structure, mask handling, type embeddings, time embeddings and the removal of
the dynamic slot embedding (ADR-026) is carried over unchanged.

## 2. Input contract

The encoder consumes the OBS-V1.3 flat observation:

```text
D = 3009
```

The factory MUST reject any observation whose schema is not
`SemanticObservationSchemaV12` at `D = 3009`. There is no fallback to a
previous dimension.

## 3. Token projections

The latent-query variant projects each raw token group to `token_dim`. The
changed widths are:

| Projection | ENC-V1.2 | ENC-V1.3 |
|---|---:|---:|
| `lane_road_projection` | 14 | 12 |
| `controls_projection` | 17 | 15 |
| `interactions_projection` | 35 | 33 |
| `context_history_projection` (was `compliance_history_projection`) | 24 | 23 |

Every other projection is unchanged: `ego_history` 10, `ego_current` 3, `route`
7, `dynamic` 22, `static` 13, `signal_onset_state` (was `yellow_onset_memory`) 3.

The raw token count remains `143` and the token order remains the OBS-V1.3
group order. `LatentQueryEncoderV3Lite` and `LatentQueryEncoderV3Micro` inherit
these widths.

`LatentQueryEncoderV2`, which consumes OBS-V1.1, keeps its historical widths of
14, 17 and 35. It MUST NOT be changed by this document.

## 4. MLP variant

The flat MLP variant takes `3009` inputs instead of `3064`. With the approved
hidden sizes this changes the exact parameter count from `2,032,128` to
`2,003,968`, a difference of `55 × 512`. No other layer changes.

## 5. Checkpoint compatibility

ENC-V1.3 weights are incompatible with ENC-V1.1 and ENC-V1.2 checkpoints. The
first projection layer of four token groups and the first layer of the MLP
variant have different shapes, so loading fails structurally rather than
silently. There is no migration path by design.

Checkpoint manifests MUST record the observation schema version, the flat
dimension and the encoder version, so a mismatch is detected before a run
starts rather than during the first forward pass.

## 6. Acceptance criteria

- The encoder builds and runs a forward pass at `D = 3009` for the MLP variant
  and for all three latent-query variants.
- The raw token count is `143`.
- The encoder output is invariant to the payload of masked tokens.
- The dynamic-slot invariance property established by ADR-026 still holds.
- `LatentQueryEncoderV2` still accepts the OBS-V1.1 group widths.
