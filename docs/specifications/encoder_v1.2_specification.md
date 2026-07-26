# Perception-Bounded Semantic Encoder Specification

**Document ID:** ENC-V1.2  
**Version:** 1.2-perception-bounded  
**Status:** APPROVED  
**Authoritative:** YES  
**Date:** 2026-07-26  
**Supersedes:** ENC-V1.1 for the latent-query (LQ) variant only; the MLP
variant in Section 3 is unchanged and normatively identical to ENC-V1.1 §3.  
**Related specifications:** OBS-V1.2, RULEBOOK-V4.7  
**Related decision:** ADR-026

## 1. Purpose

This specification amends ENC-V1.1 to remove the per-slot identity embedding
applied to dynamic-actor tokens in the latent-query (LQ) encoder
(`LatentQueryEncoderV3`). All other architectural properties of ENC-V1.1 are
preserved unchanged.

### 1.1 Rationale

Under OBS-V1.1 §7 (inherited unchanged by OBS-V1.2), the 16 dynamic-actor
slots are assigned sticky/first-fit: a track keeps the slot it first
acquired for as long as it remains live, purely to keep the per-slot 5-frame
history array temporally coherent. The physical slot index therefore carries
**no stable semantic meaning** — unlike route, static, control, and
interaction slots, which map to a fixed, meaningful role. ENC-V1.1 §4.2
nonetheless applied a learned `nn.Embedding(16, token_dim)` keyed by this
arbitrary buffer index to every dynamic-history token.

This is a recognized antipattern for encoding an unordered/arbitrarily
ordered set of elements with a fixed-capacity attention mechanism:

- **Set Transformer** (Lee, Lee, Kim, Kosiorek, Choi & Teh, *Set Transformer:
  A Framework for Attention-based Permutation-Invariant Neural Networks*,
  ICML 2019) induces permutation invariance via Pooling by Multihead
  Attention (PMA): learned seed vectors attend over a set of elements that
  carry **no** positional/identity embedding, by design, because the set has
  no canonical order and tagging elements by array position would encode
  noise, not signal. `LatentQueryEncoderV3`'s `latent_queries` attending into
  `scene_memory` is the same computational pattern.
- **DETR** (Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko,
  *End-to-End Object Detection with Transformers*, ECCV 2020) uses fixed,
  learned object queries (analogous to `latent_queries` here) precisely so
  that the model does not need to know in advance which query corresponds to
  which ground-truth object; invariance to the order of the target set is
  obtained via bipartite (Hungarian) matching, not by tagging inputs with an
  arbitrary index.
- Tracking-by-attention designs that do maintain a persistent per-track
  identity (MOTR, TrackFormer, TransTrack) do so through a continuously
  updated, content-derived hidden state carried across frames by the model
  itself — not through a static lookup table indexed by an arbitrary buffer
  slot. `LatentQueryEncoderV3` has no such cross-step state (each
  `forward()` call is independent; temporal history is already flattened
  into each token's own feature vector by the observation builder), so this
  precedent does not transfer and does not support keeping the embedding.

Removing `dynamic_slot_embedding` makes the encoder's treatment of the
dynamic-actor set invariant to which of the 16 physical slots a track
happens to occupy, eliminating a source of learnable-but-spurious
correlation without weakening the temporal-history contract in OBS-V1.1 §7,
which governs the observation builder and is unaffected by this
specification.

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
encoder_schema = 1.2-perception-bounded
```

No OBS-V1.1/ENC-V1.0 checkpoint, normalisation statistic, or manifest may be
loaded as OBS-V1.2/ENC-V1.2 by implicit conversion. An ENC-V1.1 checkpoint for
the LQ variant is also incompatible with ENC-V1.2 (the `dynamic_slot_embedding`
parameter tensor no longer exists) and MUST NOT be loaded by implicit
conversion; there is no migration path, consistent with ENC-V1.0 -> ENC-V1.1.

## 3. MLP encoder

Unchanged from ENC-V1.1. The semantic MLP encoder keeps the existing layer
topology:

```text
Linear(3064, 512) -> LayerNorm(512) -> ReLU
Linear(512, 512)  -> LayerNorm(512) -> ReLU
Linear(256, 256)  -> LayerNorm(256) -> ReLU
Linear(256, 256)  -> LayerNorm(256) -> ReLU
```

The output dimension is 256 and the expected trainable parameter count is
2,032,128. This count is schema-specific and MUST have a regression test.

## 4. Latent-query encoder

### 4.1 Raw tokens

Unchanged from ENC-V1.1. The raw token count is exactly 143. Token order is
normative:

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
token dimension 64. Existing group projection widths remain unchanged.

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
placement in the common 2.1-second window.

**Amended (ADR-026):** dynamic-history tokens receive their group projection,
type embedding, and `history_time_embedding` only. They do **not** receive a
per-slot embedding. The 16-slot dynamic buffer index is not semantically
stable under the sticky/first-fit persistent assignment of OBS-V1.1 §7 (see
§1.1 for the full rationale), so a learned per-index identity would encode
spurious correlation rather than signal. Route, static, control, and
interaction slot indices ARE semantically stable and unambiguously retain
their embeddings unchanged from ENC-V1.1.

For each valid token, the input to the latent stack is its group projection plus
the applicable type and time embeddings, plus the slot embedding for groups
that have one (route, static, control, interaction). Invalid tokens MUST be
masked from cross-attention and self-attention exactly as in ENC-V1.1; their
payload must not influence any latent.

### 4.3 Latent stack

Unchanged from ENC-V1.1. The latent-query architecture remains 16 learned
latents, latent dimension 128, four cross-attention/self-attention blocks,
four attention heads, feed-forward dimension 256, dropout zero, followed by
the existing 256-dimensional output projection. The only architectural
change relative to ENC-V1.1 is the removal of the dynamic-history slot
embedding in §4.2.

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
- The LQ encoder has no `dynamic_slot_embedding` parameter, and its output is
  identical (within floating-point tolerance) when identical dynamic-actor
  content is placed in a different one of the 16 physical slots, all else
  equal.
- Legacy semantic and non-semantic encoder tests remain unchanged and pass.
