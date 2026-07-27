# Notes: persistent actor identity and temporal memory in the encoder — considered alternatives

> **Citation status — verify before use.** This is a preparation note for an
> advisor meeting / slide deck, not an approved specification, ADR, or
> ExecPlan. The citations for Set Transformer and DETR are verified (already
> present in `docs/decisions/ADR-026-dynamic-slot-identity-removal-and-context-quota-ranking.md`).
> **All other citations below (MOTR, TrackFormer, TransTrack, R2D2, IMPALA,
> Transformer-XL, RecurrentPPO) were recalled from general knowledge and have
> NOT been verified against a primary source in this repository.** Confirm
> exact authors, venue, and year before citing any of them in the thesis or
> in any approved document.

Supporting notes for the encoder slides. They cover the ADR-026 decision
(removal of `dynamic_slot_embedding`) and a comparison with alternative
architectures that could have given dynamic actors a persistent identity over
time.

## 1. Problem

`LatentQueryEncoderV3` (`src/thesis_rl/agent/planners/encoders/lq_encoder.py`)
is a transformer-based encoder (Perceiver-style, latent queries attending
over a set of tokens). Every `forward()` call is **stateless**: it processes
a single observation frame (which already includes a 5-frame history buffer
for ego/dynamic/compliance, built by the observation builder) and carries
nothing over to the next call. The dynamic buffer (16 slots) assigns actors
with **sticky/first-fit** logic: an actor keeps the same physical slot while
tracked, but that slot can be reassigned to a different actor once the
previous one leaves the scene. The raw observation, by design, **does not
include a persistent actor ID or track age**
(`docs/specifications/observation_v1.1_specification.md:719-722`).

Decision taken (ADR-026): remove the per-slot embedding on dynamic tokens,
because the slot index has no stable meaning across the dataset. Slot
embeddings for route/static/control/interaction tokens were kept, since
those slot indices do have stable semantic meaning.

## 2. Current design — why attention (transformer) is correct for its own problem

Set attention / transformers solve the problem of *"which tokens matter in
this single frame and how do I combine them,"* independent of the physical
order in which they arrive. They do not solve, and are not meant to solve,
the problem of *"what to remember from one frame to the next"* — these are
orthogonal axes.

**Sources supporting the current design (verified, already cited in ADR-026):**
- Lee, Lee, Kim, Kosiorek, Choi, Teh — *Set Transformer: A Framework for
  Attention-based Permutation-Invariant Neural Networks*, ICML 2019. Pooling
  by Multihead Attention (PMA): input elements carry no identity/position tag,
  because the set has no canonical order — the same pattern as the
  `latent_queries` here.
- Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko — *End-to-End Object
  Detection with Transformers* (DETR), ECCV 2020. Fixed, learned queries
  achieve order invariance via bipartite matching in the loss, not by tagging
  inputs with an arbitrary index.

## 3. Alternative A — persistent track queries (MOTR/TrackFormer/TransTrack style)

**How it works:** instead of a fixed slot index, each track has a query
vector that the model itself updates and propagates from one frame to the
next (self-/cross-attention with the new frame's features). Identity comes
from a state the model builds and maintains, not from an input value. New
queries are born when new objects appear and are retired when an object
leaves the scene. This is still a transformer architecture (derived from
DETR) — it does not use recurrent cells such as LSTM/GRU.

**Models named in ADR-026 (generic names; exact bibliographic citations not
yet verified in this repository — check before citing in the thesis):**
- MOTR — *End-to-End Multiple-Object Tracking with Transformer*, ECCV 2022.
- TrackFormer — *Multi-Object Tracking with Transformers*, CVPR 2022.
- TransTrack — *Multiple Object Tracking with Transformer*, 2020/2021.

**Pros:**
- Real identity across frames, even far apart, not just within the current
  5-frame buffer window.
- Captures long-horizon predictive information (e.g., an actor's erratic
  behavior several seconds earlier) without growing the observation schema.
- Direct precedent in the object-detection/tracking literature.

**Cons:**
- Requires the encoder to carry state produced at frame *t* into frame
  *t+1* → the whole policy becomes stateful over time, not just the encoder.
- Needs a birth/death mechanism for queries as actors enter and leave the
  50 m radius (`dynamic_radius_m: 50.0`,
  `observation_v1.1_specification.md:587`).
- In the original literature, identity is learned under direct tracking
  supervision (matching loss against ground-truth tracks), which is absent
  in a pure RL setting.
- Large architectural change, explicitly out of scope in ADR-026 (alternatives
  table, last row: "Remove sticky/first-fit slot persistence entirely...
  Out of scope; would require its own ADR").

## 4. Alternative B — recurrent core on top of the per-frame encoding (classic RNN)

**How it works:** the transformer encoder stays unchanged and still processes
one frame at a time; its pooled output is fed into a recurrent cell
(LSTM/GRU) that accumulates state over time, or — an RNN-free variant — into
a transformer with persistent memory in the Transformer-XL style (a KV-cache
carried across steps). This is the standard pattern for giving an RL policy
long-term memory.

**References (high confidence, still verify before citing in the thesis —
not present in ADR-026):**
- Kapturowski, Ostrovski, Quan, Munos, Dabney — *Recurrent Experience Replay
  in Distributed Reinforcement Learning* (R2D2), ICLR 2019.
- Espeholt et al. — *IMPALA: Scalable Distributed Deep-RL with Importance
  Weighted Actor-Learner Architectures*, ICML 2018.
- Dai, Yang, Yang, Carbonell, Le, Salakhutdinov — *Transformer-XL: Attentive
  Language Models Beyond a Fixed-Length Context*, ACL 2019.
- RecurrentPPO — implementation in `sb3-contrib` (Stable-Baselines3), a
  recurrent variant of PPO commonly used in applied RL.

**Pros:**
- Does not require explicit per-actor identity: memory is scene/policy-level,
  conceptually simpler than per-object track queries.
- Well-established RL pattern with ready-made tooling (RecurrentPPO).

**Cons:**
- Recurrent RL policies are notoriously harder to train stably: requires
  handling state resets at episode boundaries and collecting rollouts as
  sequences instead of independent steps.
- Higher hyperparameter sensitivity, slower/noisier training.
- Does not directly solve the same problem (per-actor identity): it gives
  memory to the scene as a whole, not to a specific tracked actor.

## 5. Why this was not done now

Both alternatives solve a different, broader problem ("what to remember
across frames") than the specific defect fixed by ADR-026 ("a learned
embedding over a slot index with no stable meaning"). Introducing either
would require: (a) making the whole policy stateful over time, (b) changing
RL training (BPTT / sequential rollouts), and (c) possibly exposing a
persistent actor ID from the simulator (MetaDrive) that the observation
builder currently excludes by design. This is future work, not evaluated
experimentally in this repository, and would need its own ADR/specification.

## 6. Summary table for slides

| Option | Persistent actor identity | Required change | Training risk | Status |
|---|---|---|---|---|
| Current (LQ encoder, no slot embedding on dynamic tokens) | No (only within the 5-frame buffer) | None (already done) | No additional risk | Implemented, ADR-026 |
| Persistent track queries (MOTR/TrackFormer/TransTrack) | Yes, long horizon | Large: stateful policy, query birth/death, possibly a new ID field | High (tracking-style training untested in pure RL) | Not implemented, out of scope |
| Recurrent core (LSTM/GRU or Transformer-XL memory) | No (scene-level memory, not per-actor) | Large: stateful policy, sequential rollouts | High (known instability of recurrent policies) | Not implemented, out of scope |
