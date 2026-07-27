# ADR-028: Automatic Curriculum Learning v1.1 Retrospective Rationale Record

- Status: APPROVED
- Date: 2026-07-27
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-27
- Supersedes: none; retrospectively documents already-implemented, already-approved `v1.1` behavior
- Affected specifications: `automatic_curriculum_learning_v1.2_specification.md`
- Affected ExecPlan: `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` (`ACL-SN-INIT-002`)

## Context

`automatic_curriculum_learning_v1.1_specification.md` fixes the EMA arm score, the temperature-scaled softmax, the shared rank-normalization window, and the absence of an inverse-probability correction, but does not record *why* each choice was made relative to the reference paper (Peng et al., 2024, IROS) it is inspired by, nor why several paper mechanisms (cumulative accumulation, `1/p` correction, `N_MAB` resynchronization) were not carried over. This gap was identified during a `v1.2` planning session that started from an unrelated question (recreating an arm-classification table) and expanded into a full re-derivation of the MAB design against the paper and the current implementation.

## Decision

Record the following as the authoritative rationale for `v1.1`/`v1.2` behavior that is **not changing** in this ADR:

1. **Rank normalization of LP, not raw or min-max** — raw LP drifts down as the value function converges regardless of relative arm merit, has an algorithm-dependent unbounded scale, and is heavy-tailed; rank normalization is drift-free by construction, scale-free, and outlier-robust. The paper's Eq. 10 min-max-normalizes episode reward (externally bounded by reward design), not learning potential (unbounded), so it does not transfer.
2. **Shared rank window, not per-arm** — a per-arm window converges every arm to `~0.5` by construction (any sample sits near the middle of its own history), leaving the bandit with no cross-arm signal. The shared window is necessary for the mechanism to function.
3. **Bounded EMA, not the paper's cumulative accumulator (Eq. 11)** — the paper's feedback is signed, in `[-1,1]`; this project's rank-based feedback is always non-negative, in `[0,1]`. Cumulative accumulation with non-negative feedback is monotonically increasing for every arm, all arms eventually saturate at the clip, and the softmax then returns to uniform: the bandit stops discriminating. This was the structural condition of the pre-EMA ACL configuration. The move to EMA fixed a real degeneracy, not a stylistic preference.
4. **No inverse-probability (`1/p_i`) correction** — the paper's correction makes its estimator unbiased for a *sum* over rounds where unchosen arms contribute zero. This implementation updates only the sampled arm (no zero-contribution rounds to balance), so applying `1/p_i` would shift the EMA's fixed point from `E[U_i]` to `E[U_i]/p_i`, a bias rather than a correction. Quantitatively, because rank-based `U` is approximately uniform on `[0,1]`, `U >= eta/K` holds ~97% of the time, so a floor-probability arm would saturate near `1.0` on almost every update — not an edge case.
5. **Target-MAB path retained but disabled by default** — in the paper, periodic resynchronization *is* the update mechanism (layered on the cumulative accumulator). Layered on top of an already-smoothing EMA it adds delay and an untuned hyperparameter (`N_MAB`) with no identified problem to solve. Retained as a diagnostic option for future thrashing evidence, not enabled speculatively.
6. **Temperature term, absent from the paper** — the paper's unbounded weights allow arbitrary preference ratios via `e^w`; this project's bounded `q in [0,1]` caps the maximum ratio at `e^1` without a temperature, too flat to express meaningful preference. `tau=0.5` restores an expressive ratio (`e^2`).
7. **Full paper fidelity rejected on cost/benefit** — a fully faithful variant requires cumulative accumulation, `1/p` correction, `N_MAB` resync, re-centered `2U-1` feedback, and removal of the temperature, simultaneously. It is internally coherent but breaks the `acl_ema_v1`/`acl_ema_v2` checkpoint schema, introduces `N_MAB` as an untuned quantity, and has no ablation budget to justify the change. Rejected on cost, not on principle.
8. **Semantic-tier initialization question resolved as uniform** — applying the paper's `e^{-2i}` to the positional arm index would assert an ordinal difficulty ordering (`A1<A2<A3<A4`) the taxonomy does not guarantee (`A1_traffic` is also the fallback for any non-topological scenario; `A4_vru` is an orthogonal axis, not "one step harder" than `A3`). Only `A0` is structurally guaranteed easiest (the only arm defined by an upper complexity bound). Measured evidence (`MEAS-002`: `A0` is the lowest- or near-lowest-LP arm from the earliest recorded chunk onward, across three algorithms and four runs, and holds only 18 of 1000 retained buffer slots vs `A4`'s 333) additionally refutes the premise that an untrained agent extracts more LP from the simplest arm. `DEC-001` closes as "keep `q_i=0.50` uniform."
9. **Replay contamination of the shared rank window is measured, not fixed** — Replay episodes enter the same window as Generate episodes while only Generate updates the bandit. Measured on four runs (`MEAS-001`): a real level shift (2.64x median LP ratio, replay above generate) but no measurable loss of discriminative dispersion (`0.3031` vs `0.2972`, mixed vs Generate-only), because the shared window is common to all arms and the softmax is invariant to a constant offset. No code change follows from this finding.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Adopt full paper fidelity (cumulative + `1/p` + `N_MAB`) | Maximal traceability to the cited reference | Breaks checkpoint schema, adds untuned `N_MAB`, no ablation budget | `RAT-007` |
| Per-arm rank window | Removes non-stationarity from mixing sources | Converges every arm toward `0.5`, destroying the bandit's discriminative signal | `RAT-002` |
| Reinstate `1/p` correction with an invented cap | Superficially closer to EXP3 | The cap would be an unvalidated invented parameter compensating for a correction that does not apply to this estimator class | Considered and withdrawn within the same session (ExecPlan §11) |
| Positional-index exponential initialization | Matches the paper's literal formula | Asserts an ordinal difficulty axis this taxonomy does not guarantee, and is contradicted by measured LP | `RAT-008`; `MEAS-002` |

## Consequences

No code changes result from this ADR. `automatic_curriculum_learning_v1.2_specification.md` §15.1 carries `RAT-001..RAT-009` forward as the citable rationale. Future changes to any of the nine mechanisms above must engage with the specific structural argument recorded here, not merely restate a preference for closer paper fidelity.

## Validation And Traceability

No new tests: this ADR documents already-implemented, already-tested behavior. `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` §6.2 and §11 record the full derivations and the four-run measurement evidence (`MEAS-001`, `MEAS-002`) underlying `RAT-008` and `RAT-009`.

## Approval Record

- Approved by: user
- Approval evidence: iterative review and explicit confirmation across the 2026-07-27 session ("va bene ciò che mi hai suggerito, procedi pure alla verifica e poi anche all'implementazione"), following presentation of each rationale and its alternatives
