# Perception-Bounded Semantic Encoder Specification

**Document ID:** ENC-V1.1  
**Version:** 1.1-perception-bounded  
**Status:** APPROVED  
**Authoritative:** YES  
**Date:** 2026-07-21  
**Supersedes:** ENC-V1.0 for OBS-V1.2  
**Related specifications:** OBS-V1.2, RULEBOOK-V4.7  
**Related decision:** ADR-022

## 1. Purpose

This specification defines the encoder contract for OBS-V1.2. It preserves the
established MLP and latent-query architectural families while changing their
schema-bound input dimensions, token groups, masks, and checkpoint metadata.
It does not alter the legacy observation encoder contracts.

## 2. Common contract

Both encoder variants consume only the finite `float32` OBS-V1.2 observation.
They MUST preserve zero/mask semantics: an invalid payload is zero, and a mask
is the sole indication that it is invalid. They MUST NOT infer a detection from
a zero payload without its corresponding validity mask.

The encoder output remains a finite `float32` vector of dimension 256. Checkpoint
metadata MUST include:

```text
observation_type = semantic_v3
observation_schema = 1.2-perception-bounded
flat_observation_dim = 3064
encoder_schema = 1.1-perception-bounded
```

No OBS-V1.1/ENC-V1.0 checkpoint, normalisation statistic, or manifest may be
loaded as OBS-V1.2/ENC-V1.1 by implicit conversion.

## 3. MLP encoder

The semantic MLP encoder keeps the existing layer topology:

```text
Linear(3064, 512) -> LayerNorm(512) -> ReLU
Linear(512, 512)  -> LayerNorm(512) -> ReLU
Linear(512, 256)  -> LayerNorm(256) -> ReLU
Linear(256, 256)  -> LayerNorm(256) -> ReLU
```

The output dimension is 256 and the expected trainable parameter count is
2,032,128. This count is schema-specific and MUST have a regression test.

## 4. Latent-query encoder

### 4.1 Raw tokens

The raw token count is exactly 143. Token order is normative:

| Indices | Group | Count | Raw width |
|---:|---|---:|---:|
| `0:5` | ego history | 5 | 10 |
| `5` | ego current | 1 | 3 |
| `6:16` | route | 10 | 7 |
| `16:96` | dynamic history | 80 | 22 |
| `96:104` | static | 8 | 13 |
| `104` | lane/road | 1 | 14 |
| `105:113` | control | 8 | 17 |
| `113:121` | interaction | 8 | 35 |
| `121:142` | compliance history | 21 | 24 |
| `142` | yellow-onset memory | 1 | 3 |

The mask associated with a raw token is the group mask in OBS-V1.2. Dynamic
history uses its `(16, 5)` mask. Compliance history uses its 21-position mask.
Ego history and route retain their existing masks. The single-token groups
without an explicit mask are valid by construction.

### 4.2 Token projections and embeddings

Every group has its own learned `Linear -> ReLU -> LayerNorm` projection to
token dimension 64. Existing group projection widths remain unchanged. New
projections are:

```text
compliance_history: Linear(24, 64)
yellow_onset_memory: Linear(3, 64)
```

Type embeddings have dimension 64 and exactly ten type IDs:

| Type ID | Group |
|---:|---|
| 0 | ego history |
| 1 | ego current |
| 2 | route |
| 3 | dynamic history |
| 4 | static |
| 5 | lane/road |
| 6 | control |
| 7 | interaction |
| 8 | compliance history |
| 9 | yellow-onset memory |

`history_time_embedding` MUST have 21 entries of dimension 64. Compliance rows
receive time indices `0..20`, oldest to current. The five existing ego-history
and dynamic-history positions receive indices `16..20`, preserving their actual
placement in the common 2.1-second window. Dynamic-history tokens also retain
their 16-slot embedding. No slot embedding applies to compliance or yellow
memory tokens.

For each valid token, the input to the latent stack is its group projection plus
the applicable type, time, and slot embeddings. Invalid tokens MUST be masked
from cross-attention and self-attention exactly as in ENC-V1.0; their payload
must not influence any latent.

### 4.3 Latent stack

The latent-query architecture remains 16 learned latents, latent dimension 128,
four cross-attention/self-attention blocks, four attention heads, feed-forward
dimension 256, dropout zero, followed by the existing 256-dimensional output
projection. The only architectural changes are the raw-token assembly and
embedding cardinalities above.

## 5. Compatibility and configuration

`semantic_v3` is the configuration selector for this contract. `semantic_v2`
continues to denote the historical OBS-V1.1/ENC-V1.0 route. The implementation
MUST reject a mismatch among selected observation type, schema dimensions,
encoder implementation, checkpoint manifest, or normalisation statistics.

## 6. Acceptance criteria

- Both encoder variants accept a batch of OBS-V1.2 observations and return
  finite `(B, 256)` `float32` outputs.
- The MLP first projection has input width 3,064 and the exact parameter count
  in Section 3.
- The LQ encoder emits 143 raw tokens in the exact order in Section 4.1.
- Masked compliance and dynamic-history rows cannot affect an output under a
  controlled masked-token regression test.
- LQ uses ten type IDs and a 21-entry time embedding, with the prescribed
  `16..20` mapping for five-frame histories.
- Legacy semantic and non-semantic encoder tests remain unchanged and pass.
