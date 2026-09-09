# Behavioural specification, stated without a hierarchy — DRAFT, not approved

**Status: DRAFT for user approval. Nothing here is authoritative and no repository
file has been changed.** Written 2026-09-09 as deliverable 2 of the rulebook
architecture review. It restates what the reward must prefer as orderings over
pairs of trajectories, using only quantities the RULEBOOK-V5.1 §3.4 atomic vector
already carries, and **without presupposing levels, an order, weights or
thresholds**. Every candidate architecture is scored against this, not against
§1.1's six orderings, because those name the level that decides them and so cannot
be used to compare architectures.

---

## 1. Objects and derived quantities

A trajectory `t` is one episode in one frozen scenario: a sequence of per-step
atomic cost vectors (§3.4) plus the mission the scenario declares.

Every quantity below is a function of that sequence. **None requires a new
observation**, and none uses the level structure.

| symbol | definition | units |
|---|---|---|
| `G(t)` | mission completed — the directed final-gate crossing of `mission/tracker.py` | {0,1} |
| `T(t)` | duration | control steps (0.1 s) |
| `Q(t)` | mission span `(s_goal − s_start) / D_REF`, a **scenario constant** | `D_REF` units |
| `D(t)` | route distance covered, `Σ_t Δq_t` (telescopes; undiscounted) | `D_REF` units |
| `X_imp(t)` | at-fault collision impact; non-zero on at most one step (ADR-071) | [0,1] |
| `X_int(t)` | interaction-risk exposure, `Σ_t max(c_ttc, c_clearance, c_rss_lateral)` | step·cost |
| `X_hard(t)` | non-negotiable-compliance exposure, `Σ_t max(c_offroad, c_signal, c_stop, c_crosswalk, c_vehicle_yield, c_speed_limit)` | step·cost |
| `X_soft(t)` | negotiable-lane exposure, `Σ_t (c_solid_line + c_wrong_carriageway + c_dashed_line)/3` | step·cost |

Two further objects are needed and both are **measured, not stipulated**:

* the **competent-driver envelope** `E`, the joint distribution of
  `(X_imp, X_int, X_hard, X_soft)` over the logged human expert on the same panel.
  The instrument already produces its per-step marginals
  (`docs/audits/reward_calibration_2026-09-07/`); the per-episode distribution is
  a small addition to the same accumulator;
* a **declared exchange rate** `R` between negotiable-lane exposure and arrival
  time, in seconds of delay per second of full-severity lane exposure. `P3`
  cannot be stated without one — see `I1`.

Write `t ≻ u` for "the reward must rank `t` above `u`". Unless a property says
otherwise, all unmentioned quantities are equal between the two trajectories, and
both run in the same scenario with the same mission.

---

## 2. The orderings

### Liveness

**P1 — completing beats not moving.** If `G(t)=1`, `D(u)=0`, and `t`'s exposures
lie inside `E`, then `t ≻ u`.

> The clause "inside `E`" is what makes this falsifiable rather than vacuous.
> "Legal completion" is unattainable: the logged human accrues interaction cost on
> **0.3978 %** of steps and non-negotiable-compliance cost on **0.6418 %**, so a
> property quantified over zero-exposure completions describes no trajectory the
> agent can produce. §1.1's O1 is that vacuous version; `P1` is the honest one.

**P2 — completion is worth bounded negotiable relaxation.** If `G(t)=1` with
`X_soft(t) ≤ S̄`, and `D(u)=0`, then `t ≻ u`, for the declared budget `S̄`.

### Priority

**P4 — an impact outranks any negotiable relaxation.** If `X_imp(t)=0` and
`X_imp(u)>0` then `t ≻ u`, for every `X_soft(t)` up to the whole episode.

**P5 — non-negotiable compliance outranks progress.** If `X_hard(t)=0`,
`X_hard(u)>0` and `D(t) < D(u)`, then `t ≻ u`, for `D(u) − D(t)` up to a whole
mission.

**P6 — minimum violation.** If `G(t)=G(u)`, `D(t)=D(u)`, `T(t)=T(u)` and `t`'s
exposure is no larger on every channel and strictly smaller on one, then `t ≻ u`.

### Trade

**P3 — negotiable relaxation may not be bought with time.** If `G(t)=G(u)=1`,
`X_soft(t) < X_soft(u)` and every other exposure is equal, then `t ≻ u` whenever
`T(t) − T(u) ≤ R · (X_soft(u) − X_soft(t)) · Δt`.

> `P3` is deliberately a **finite exchange rate and not a dominance**. `I1` proves
> no discounted architecture can make it a dominance, so stating it as one — which
> is what §1.1's O3 does — states something unobtainable. `R` is the parameter that
> makes it a contract; it has a physical reading and §4 derives its window.

**P9 — promptness.** If `G(t)=G(u)=1` with equal exposures and `T(t) < T(u)`,
then `t ≻ u`.

### Gradient and severity

**P7 — severity is monotone and its gradient points the right way.** On any one
channel, a deeper violation at equal duration is worse, and the reward's local
derivative with respect to the control must point toward reducing it in every
conflict a real vehicle could still resolve by braking. ADR-081's
`a_req^max = (w₂σ + φ)·v_ref / (2·λ₄·τ)` is this property for the interaction
channel; the property is general and currently stated for one channel only.

**P8 — mitigation.** If an impact is unavoidable, a lesser impact is preferred:
`X_imp(t) < X_imp(u)` with everything else equal implies `t ≻ u`.

> Nothing in §1.1 states `P8`, and nothing in the test matrix checks it. It is
> the property that decides whether the reward asks the agent to brake before an
> impact it cannot avoid.

### Integrity of the measurement

**P10 — no free exit.** No trajectory may improve its rank by ending the episode
early through a violation, for any pair of trajectories both inside `E`.

**P11 — no credit without motion.** A trajectory whose ego does not move earns no
progress credit, whatever the route projection reports.

**P12 — no credit past the goal.** No trajectory earns progress credit beyond the
mission goal.

**P14 — completing strictly beats stopping short.** If `G(t)=1` and `u` covers all
but a fraction `e` of the mission and then stops, then `t ≻ u`, by a margin that
survives the exposure the last manoeuvre costs.

> Neither the shipped architecture nor any candidate satisfies the second clause:
> the margin is 0.57–2.99 reward units against 8.375 for one fully-violated
> interaction step, because mission success is a **zero-value terminal**. The
> instrument for it is a terminal completion bonus; the criterion that decides
> whether one is needed is the fraction of evaluation episodes reaching ≥95 % route
> completion without a gate crossing, which nothing currently reports.

### Distribution

**P13 — tail aversion.** Between two policies with equal *mean* exposure on
`X_imp`, `X_int`, `X_hard` or `X_soft`, the one with the lighter upper tail is
preferred.

> `P13` is a property of policies, not of trajectory pairs, and no scalar sum of
> expectations can express it. It is the criterion the distributional component
> exists to serve, and it is the reason a channel whose return distribution is
> nearly deterministic given the others carries nothing for that component.

---

## 3. Impossibility results

These are proved, not measured, and they bound what any architecture can deliver.

**I1 — no discounted progress channel is duration-invariant.** Let a channel's
per-step value `x_t ≥ 0` be a Markov function of the observation with
`Σ_t x_t = Q` for every completing trajectory. Then for `γ < 1`, `Σ_t γ^t x_t` is
strictly larger for a trajectory delivering the same total earlier. Duration
invariance would require `x_t ∝ γ^{−t}`, which depends on `t`; `t` is not in the
observation and putting it there is forbidden (no new observation requirement,
and a time feature makes the policy non-stationary — ADR-081 rejects it
explicitly). **Therefore `P3` cannot be a dominance at `γ < 1` under any
hierarchy, with any number of levels, in any order.**

**I1b — an episodic budget has no per-state equivalent.** Tercan & Prabhu
(arXiv:2408.13493, Appendix D.3): for a per-episode budget "the corresponding
discounted threshold actually depends on the trajectory". A budget on accumulated
exposure therefore cannot be implemented as a threshold on Q-values, and the
converse failure is measured: Vamplew et al. (Machine Learning 84, 2011, §7.2)
report thresholded lexicographic Q-learning failing on a dense per-step constrained
objective "regardless of the value of the threshold", because its action selection
"considers only the expected future reward … ignoring any rewards received earlier
in the current episode". **Budgets stated over trajectories, as they are here, are
the right altitude for a requirement; the mechanism that implements them is a
separate and partly open question.**

**I2 — strict lexicographic comparison is degenerate.** If any channel is
non-zero with positive probability on every moving trajectory, the do-nothing
trajectory is optimal under strict lexicographic comparison. Measured: the
interaction channel fires on 0.3978 % of expert steps. Architecture-independent;
already declared (§11.1) and accepted (`AC-RB5.1-02`).

**I3 — progress must not be a thresholded channel.** `P3` and `P9` place opposite
requirements on a threshold placed on the progress channel, **at the same critical
value**, because under `γ < 1` the only thing separating the two comparisons at
that channel is duration. Measured on the §4.6 reference pair: `P3` requires
`τ_progress ≤ 27.97` and `P9` requires `τ_progress > 27.97`. The requirement is
empty. A threshold on progress additionally admits "reach the budget, then stop",
and must scale with a mission span the panel varies over 13–247 m (`D1`).

**I4 — no bounded scalar sum delivers `P5` at return level.** Over `L` steps it
needs `w_k > L·w_{k+1}`, i.e. `a ≳ L`. Already recorded by ADR-081 and settled by
restating the claim as per-step.

**I5 — `P2` and `P3` together require a finite exchange rate.** `P2` says a
negotiable exposure `r` is worth a whole mission; `P3` says an exposure `r'` is
worth more than the time a shortcut saves. Both hold simultaneously iff the
exchange rate lies in `((Q_fast − Q_slow)/r', Q/r]`, an interval that is non-empty
but bounded away from both 0 and ∞. A threshold supplies only 0 (inside budget) or
∞ (outside), so **no purely thresholded comparison satisfies `P2` and `P3`
together; a scalarization can.** This is a place where the scalar control is
structurally *better* than the ordered arms, and it should be reported as such.

---

## 4. What fixes each free quantity

Nothing here may be chosen by taste. Each row names the measurement or the
mechanism that fixes it.

| quantity | fixed by | status |
|---|---|---|
| `S̄` (negotiable budget) | the per-episode `X_soft` distribution of the logged expert; a budget that fails to admit the human is falsified | mean **0.1511** measured; per-episode distribution **not yet emitted** |
| `R` (exchange rate) | bounded below by `P3` against the §4.6 reference shortcut; bounded above by the manoeuvre it replaces — stopping from `v` and returning to it at a comfortable `a_c` costs `v/a_c` seconds, 5.0 s at 10 m/s and 2 m/s² | window derived, §5 of the review |
| interaction budget | the per-episode `X_int` distribution of the expert (limitation 1 already argues exactly this for `d₂`) | mean **0.4227** measured; distribution **not yet emitted** |
| non-negotiable budget | the same for `X_hard`. **Note: it cannot be zero** — the expert violates on 0.6418 % of steps, 92.4 % of it `offroad`, so a zero budget is falsified by the panel | mean **0.1507** measured |
| severity slope | `P7`, from the braking a real vehicle can produce (~9 m/s²) | ADR-081, `σ = 0.30` |
| the discount | `γ^L ≥ 1/a` with `L` the longest training episode | §1 of the review |

**One scoping note on "the last channel is unthresholded".** That is the documented
requirement of the *absolute thresholding* family (Vamplew et al. 2011 §3.2.3:
"objective n will be unconstrained, hence `C_n = +∞`"; Tercan & Prabhu §3). It is
**not** a property of the *slacking* family, where the loop runs to the last
objective and slacks it too (Li & Czarnecki, Algorithm 1; Skalse et al. IJCAI
2022). No source requires the last objective to be goal-reaching or terminating —
Vamplew et al.'s own example of an unconstrained final objective is "maximizing
factory production while maintaining a required safety level".

**Budget units.** Express every budget as exposure **per unit of mission span**,
`X / Q`. `Q` is a scenario constant the agent cannot influence, so this is
duration-invariant, immune to dilution by dawdling, and comparable across a panel
whose missions span 13–247 m. This is *not* §4.1.1's rejected route-length
normalization of the reward: the per-step vector is untouched, only the budget's
units change.

---

## 5. Mapping to §1.1's six orderings

| §1.1 | here | change |
|---|---|---|
| O1 | `P1` | quantified over the competent-driver envelope rather than over the empty set of zero-exposure completions |
| O2 | `P2` | budget made explicit |
| O3 | `P3` | restated as a finite exchange rate, because `I1` proves the dominance form unobtainable at `γ<1` |
| O4 | `P4` | unchanged |
| O5 | `P5` | unchanged |
| O6 | `P6` | unchanged |
| — | `P7`–`P13` | new; `P8`, `P10`, `P11`, `P12`, `P13` are stated nowhere in the repository and nothing tests them |
