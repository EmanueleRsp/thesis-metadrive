# ADR-026: Dynamic Slot Identity Removal and Context-Quota Ranking by Distance

- Status: Approved
- Date: 2026-07-26
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-26
- Affected specifications:
  - `docs/specifications/encoder_v1.2_specification.md`, ID `ENC-V1.2`
    (supersedes `encoder_v1.1_specification.md` for the latent-query variant)
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`
    (SS8.1 Context quota, amended in place; inherited unchanged by OBS-V1.2)
- Affected ExecPlan:
  `docs/implementation/semantic_v3_dynamic_ranking_and_slot_embedding_removal_exec_plan.md`

## Context

A code review of the perception-bounded semantic observation (`semantic_v3`)
and its latent-query encoder (`lq_v3`) identified two related design
weaknesses in how the fixed-capacity dynamic-actor representation (16 slots)
is built and consumed:

1. **Encoder side.** `LatentQueryEncoderV3` adds a learned
   `dynamic_slot_embedding` (`nn.Embedding(16, token_dim)`) to every
   dynamic-history token, keyed by the physical buffer slot. OBS-V1.1 SS7
   assigns these slots sticky/first-fit — a track keeps whichever slot it
   first acquired, purely to keep its 5-frame history array temporally
   coherent — so the slot index carries no stable semantic meaning. A
   learned embedding over an index with no stable meaning can only encode
   spurious correlation between "which physical slot" and network behaviour.
2. **Observation side.** The non-conflict ("context quota") dynamic-actor
   ranking key (OBS-V1.1 SS8.1) ordered candidates by same-lane, then
   adjacent-lane, then route-ahead, then lateral distance, and only then by
   Euclidean distance. A much closer actor in a different lane could
   therefore lose a slot to a far same-lane actor.

The user's own project rationale for building a semantic observation at all
(rather than using MetaDrive's baseline `LidarStateObservation`) is that it
must be a genuinely better input for the policy and the encoder, not merely
an accepted trade-off — the thesis advisor did not require a semantic
observation specifically. Both weaknesses above were therefore examined
against object-centric/set-based representation-learning literature and
against literature on agent-selection heuristics for capacity-limited
driving-scene encoders, gathered in two rounds: an internal architectural
review of `lq_encoder.py`/`causal_semantic.py`, and an external literature
search requested by the user and reported back into this conversation.

## Decision

1. **Remove `dynamic_slot_embedding` from `LatentQueryEncoderV3`.**
   Dynamic-history tokens keep their group projection, type embedding, and
   `history_time_embedding` (which encodes recency, a stable quantity), but
   no longer receive a per-slot identity tag. Route, static, control, and
   interaction slot embeddings are unchanged — those slot indices ARE
   semantically stable. This is recorded as ENC-V1.2 SS4.2.
2. **Replace the OBS-V1.1 SS8.1 Context-quota ranking key** with plain
   Euclidean distance to ego (tie-broken by stable actor ID), removing the
   same-lane / adjacent-lane / route-ahead / lateral-distance precedence.
   The Conflict-quota key (CPA validity, `t_CPA`, `d_CPA`, lane relation,
   distance, actor ID) is explicitly kept unchanged.

## Literature Basis

**For decision 1 (encoder):**

- Lee, Lee, Kim, Kosiorek, Choi, Teh, *Set Transformer: A Framework for
  Attention-based Permutation-Invariant Neural Networks*, ICML 2019. Pooling
  by Multihead Attention (PMA) — learned seed vectors attending over a set of
  elements — is the same computational pattern as `latent_queries` attending
  over `scene_memory` here. PMA's input elements carry no positional/identity
  embedding, specifically because the set has no canonical order.
- Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko, *End-to-End Object
  Detection with Transformers*, ECCV 2020. Fixed, learned object queries
  achieve invariance to the order of the target set via bipartite matching in
  the loss, not by tagging inputs with an arbitrary index — supporting the
  same conclusion for `latent_queries`.
- Considered and rejected as a counter-argument: tracking-by-attention models
  with persistent track queries (MOTR, TrackFormer, TransTrack) do maintain a
  per-track identity across frames, which could appear to justify a
  slot-identity embedding. That identity is carried by a continuously
  updated, content-derived hidden state propagated by the model itself across
  frames. `LatentQueryEncoderV3` has no such cross-step state — each
  `forward()` call is independent, and temporal history is already flattened
  into each token's own feature vector by the observation builder — so this
  precedent does not transfer.

**For decision 2 (context-quota ranking):**

- Gao et al., *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized
  Representation*, CVPR 2020; GameFormer (Huang et al., ICCV 2023); Wayformer
  (Nayakanti et al., ICRA 2023). Dominant practice for agent selection under a
  fixed encoder capacity is k-nearest-by-distance.
  GameFormer specifically selects the 10 nearest background agents to the
  target agent; a related vectorized-representation approach selects up to
  10 surrounding agents within 30 m ordered by distance, zero-padded if fewer
  are available.
- The Waymo Open Motion Dataset's own agents-of-interest selection criterion
  (per a WOMD-replication paper) is not distance-based but a kinematic
  "interest score" from total heading variation, lateral deviation,
  acceleration, and progress — a proxy for "this agent is doing something
  non-trivial," used as the argument that CPA/TTC (constant-velocity
  extrapolation) is least reliable for exactly these agents.
- Sun, Zhao, Sadigh, Zhan, Anguelov, *Identifying Driver Interactions via
  Conditional Behavior Prediction*, ICRA 2021. Direct empirical comparison of
  a mutual-information-based interactivity metric against a plain-distance
  heuristic for agent selection: interactivity wins for small N (<= 4);
  distance overtakes it as N grows, partly because distance-only selection
  at small N includes topologically irrelevant near agents (opposite lanes)
  while excluding relevant farther ones. The paper's second experiment shows
  that demoting less-interactive agents to lower-fidelity scene context while
  keeping the top ~3 most interactive agents in the high-fidelity prediction
  set improves downstream prediction quality — directly supporting the
  existing two-tier design (conflict quota = curated high-fidelity CPA/TTC
  ranking; context quota = larger, coarser pool) rather than extending
  CPA/TTC to the whole pool.
- MAPLE (2026) and RAD-LAD (2026) as downstream-planning evidence: MAPLE
  shows driving-score/success-rate gains from including more
  KNN-selected reactive agents (with diminishing returns); RAD-LAD shows that
  raising raw dynamic-agent capacity from 32 to 128 does not help, because
  the additional distant agents add noise without signal — evidence that
  selection quality, not just capacity, matters once N grows, and that a
  cheap, robust distance filter is preferable to no filter at all.
- Acknowledged gap: no paper found directly compares Euclidean distance
  against CPA/TTC against lane-topology precedence on downstream *planning*
  metrics (collision rate, success rate) for an RL policy input specifically;
  the strongest transferable evidence is the interactivity-vs-distance
  crossover above, which uses trajectory-prediction accuracy as the outcome
  metric, not planning safety.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Keep `dynamic_slot_embedding`, add an explicit rank/priority feature to the token instead | No architecture change, no checkpoint break | Treats a symptom (missing priority signal) without removing the source of spurious slot-identity correlation; adds a schema dimension | Superseded by the root-cause fix once the architectural antipattern was confirmed against Set Transformer/DETR precedent |
| Extend CPA/TTC ranking to the whole context quota (initial proposal) | Reuses already-computed values, minimal diff | Contradicts the interactivity-vs-distance crossover for larger N; CPA is least reliable for accelerating/turning agents, which matter most in an uncurated pool | Rejected after the literature crossover was properly accounted for |
| Leave context-quota ranking as same-lane/ahead/lateral | No change | Not supported by any reviewed source; a closer, differently-laned actor can lose a slot to a farther same-lane one | Rejected in favour of plain distance |
| Remove sticky/first-fit slot persistence entirely (frame-stacking or recurrent policy instead) | Would eliminate the slot-identity problem at its root | Large architecture change (new encoder input shape, policy statefulness), well beyond this decision's scope, would require its own ADR | Out of scope; the per-slot history design is retained |

## Consequences

- `LatentQueryEncoderV3`'s parameter count decreases by `16 * token_dim`
  (the removed `dynamic_slot_embedding` weight matrix). Existing ENC-V1.1
  checkpoints for the LQ variant cannot be loaded under ENC-V1.2 (missing
  parameter); there is no migration path, consistent with the ENC-V1.0 ->
  ENC-V1.1 precedent. The MLP encoder variant is unaffected.
- The observation encoding itself (`causal_semantic.py` output tensor shape
  and dimensions) is unchanged by decision 1; only the encoder's internal use
  of the dynamic tokens changes.
- Decision 2 changes which actors occupy the context-quota slots when
  candidates exceed capacity; it does not change the flat observation
  dimension or the encoder's expected input shape.
- Active production training runs using `semantic_v3`/`lq_v3` train on a
  materially different effective input distribution after this change and
  must be restarted to reflect it; this restart decision is tracked as an
  open milestone in the affected ExecPlan, not resolved by this ADR.

## Validation And Traceability

Requirements and test IDs are defined in
`docs/implementation/semantic_v3_dynamic_ranking_and_slot_embedding_removal_exec_plan.md`.

## Approval Record

- Approved by: user
- Approval evidence: explicit user instruction in this conversation on
  2026-07-26, "sì procedi", following two rounds of review — an internal
  architectural analysis of `lq_encoder.py` and `causal_semantic.py`, and an
  external literature search the user ran themselves and reported back,
  after which the user explicitly rejected a minimal/conservative framing
  ("NON IMPUNTARTI SUL VOLER CAMBIARE IL MINIMO INDISPENSABILE") in favour of
  the best-supported design, revisable together with its governing
  specification.
- Notes: the sticky/first-fit slot-persistence mechanism itself (OBS-V1.1
  SS7) is explicitly out of scope and unaffected by this ADR.
