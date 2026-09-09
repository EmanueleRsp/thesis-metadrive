# Rulebook architecture review — findings, candidates, recommendation

**Status: nothing recommended here is approved.** Written 2026-09-09 against
`main` at `8b9f6aa`, with no source, configuration or test change; the review and
its scripts were committed afterwards so the evidence would survive. No ADR, no
specification amendment and no ExecPlan exists for §4's recommendation.

Reproducible arithmetic: `g1_discount.py`, `g2_bench.py`, `g3_battery.py`,
`g4_hacking.py`, `g5_w5.py` and `g6_ratchet.py` in this directory. Standard
library only, except that `g1` reads the frozen selection index by path and
`g6` imports the repository — see the directory README for its command.

**§8 was added after the review** and supersedes two statements inside it; the
places affected point forward to it. It also records what a working continuation
should not redo.

> **SUPERSEDED IN FOUR PLACES, 2026-09-09 — read this before §6 or §8.**
> The instruments §6 asks for were then built and run, and they falsified four
> statements in this document. Every one is marked inline where it stands, and
> all four are carried with their evidence in
> **`docs/audits/progress_channel_integrity_2026-09-09/README.md`**, which is
> authoritative wherever it disagrees with this file.
>
> 1. **`ROUTE_CONTINUITY_JUMP_FACTOR` does not exist** in any Python file, so
>    "promoting it adds no level, weight or observation field, because the
>    constant is already there" is false. The bound was wired into
>    `evaluate_progress` on 2026-07-30 and `e63e0bf` removed it on 2026-08-03.
>    Affects §6's table and §8.2.
> 2. **The clip ratchet's "not armed" verdict is withdrawn.** The figure it
>    rested on is a monotone function of lateral reach, not a bound, and nothing
>    in the runtime bounds the reach. Affects §8.2.
> 3. **The `D14` backwards walk has been run**, and the headline drawn from it
>    was flattered by a share-of-route normalisation. Affects §8.3.
> 4. **`telescoping_max_error` is emitted signed**, with a per-episode
>    distribution. Affects §8.1 and §6's table.
>
> Also corrected: `F9`'s **596 / 27.09 %** is **591 / 26.86 %** — the original
> compared the scenario `length` with the break-even, and the criterion is over
> control steps, so an episode is `length − 1`. Conclusion unchanged.
>
> **A7 and `γ = 0.9982` were approved on 2026-09-09.** This document's own
> status line above predates that.

---

## 1. The discount — `F9` verified, and the decision

### 1.1 `F9` is a repository defect, and the chain is closed in code

| step | evidence |
|---|---|
| the frozen index's `length` is the scenario's own `SD.LENGTH` | `scripts/validate_frozen_scenarionet_content.py:52-56` cross-checks catalog against description; `scripts/build_dataset_funnel_report.py:75` labels it "scenario length (control steps)" |
| the runtime reads the same field | `third_party/metadrive/metadrive/manager/scenario_data_manager.py:106` — `current_scenario_length = current_scenario[SD.LENGTH]` |
| the episode runs it out | `src/thesis_rl/envs/thesis_scenario_env.py:47,1003` — truncation at `episode_steps >= scenario_length - 1 + extra`, with `extra_steps_after_scenario: 0` and `horizon: null` (`conf/env/scenarionet.yaml:16,69`) |
| so the longest training **episode** is 500 control steps | max `length` over `train` = 501 |
| at `a = 2.5`, `γ = 0.996` the break-even is 228.6 steps | `ln(2.5)/−ln(0.996)` |
| **591 of 2200 training records = 26.86 %** exceed it | train/pg p50 **241**, p95 **501**; train/waymo 0.0 % over. *Corrected*: this row read 596 / 27.09 %, which compared the scenario `length` with the break-even. The criterion is over control steps and an episode is `length − 1`, so the five records with `length = 229` — 228 steps, below the 228.6 break-even — do not belong in the count |
| the guard encodes the same error | `tests/test_hydra_agent_presets.py:257` hardcodes `horizon_steps = 199` under a docstring reading "The horizon is measured, not assumed" |

Classification per `AGENTS.md`: **a defect of the repository.** It holds for any
contributor on any machine, and the test that exists to catch it reproduces it.

### 1.2 The criterion, restated so the trade is visible

`Δ > L` with `Δ = ln(a)/−ln(γ)` is **exactly `γ^L ≥ 1/a`**. (In the table of
required discounts, `γ^L = 0.400 = 1/2.5` at every `L`; it is an identity, not a
coincidence.)

That puts ADR-081's two reasons on one axis. Reason 1 — no Bellman contraction at
`γ=1` while ~2/3 of episodes end in a bootstrapped truncation whose backup admits
`V + c` — wants `γ^L` **small**, because `γ^L` is precisely how much of a spurious
constant survives a whole episode. Reason 2 wants `γ^L ≥ 1/a`. **The criterion is
the most damping reason 1 can have without reason 2 failing.** Neither reason is
reopened; only `L` is corrected.

### 1.3 The exits, priced

| exit | what it requires | cost |
|---|---|---|
| **A. raise `γ`** | `γ ≥ 0.4^(1/500) = 0.998169` | effective horizon `1/(1−γ)` **250 → 556 steps (25.0 s → 55.6 s)**, ×2.2; whole-episode damping of a bootstrapped constant `0.135 → 0.406` |
| B. raise `a` at `γ = 0.996` | `a ≥ 7.419` | `a³` 15.6 → 408, **×26 on the critic's dynamic range**; drags `σ`, `λ₄` and the 217,189-transition grid, and ADR-081 already measured and rejected `a = 3.0` in both directions |
| C. cap the horizon | cap ≤ 229 | truncates **53.7 %** of PG train records; PG mean route 183.81 m is 405 steps at the expert's 4.535 m/s, so any cap below ~410 makes the median PG mission structurally incompletable |
| D. do nothing | — | for 26.86 % of training records the hierarchy inverts inside the episode, which is the property `γ` was chosen for; and the guard stays green |

**What exit A does *not* cost, and this is the load-bearing fact:** the
217,189-transition calibration is an **undiscounted** sum
(`scripts/measure_expert_rulebook_transition.py:2038`,
`episode_return += scalarized.reward`), and §5.4's rank-preservation inequality is
per-step and `γ`-free. Its own README says so: "What it does not measure: the
discount." So `a`, `σ`, `λ₄`, `η`, `λ₆`, `φ` and every calibration column are
untouched and **no 45-minute re-run is needed**.

### 1.4 The decision

> **`γ = 0.9982`** — the first four-decimal value satisfying `γ^L ≥ 1/a` at the
> true training horizon `L = 500` (`Δ = 508.6` steps, 1.7 % margin) — **at the cost
> of doubling the effective horizon from 250 to 556 steps, which weakens ADR-081's
> own contraction argument by the same factor 2.2 and leaves 40.6 % rather than
> 13.5 % of a bootstrapped constant alive at the end of the longest episode; it
> costs nothing in calibration, weights or measurement, because the grid is
> undiscounted.**

Side effects, all in the right direction and none of them a reason on their own:
the O3 margin improves from **−3.5789 to −1.9934** (still negative; only `γ=1`
restores it, and `I1` in the behavioural specification proves why), and the
discount's share of the time preference falls from 90.5 % to 79.7 %.

**Also required, and it is the durable half of the fix — done 2026-09-09:**
`tests/test_hydra_agent_presets.py` computes `L` from
`data/scenarionet/frozen/scenario_selection_index.json` instead of hardcoding
199, so the docstring's claim is now true and a future panel change fails the
guard instead of passing it. The fixture pins **the requirement rather than the
state** — `γ ≥ exp(−ln a / 500) = 0.998169`, a function of `a` and the measured
`L` alone — and deliberately asserts neither a discount nor the criterion's
verdict, both of which belong beside the approved value. Note the margin is a
property of the frozen index and not of the code: `γ = 0.9982` gives a break-even
of 508.6 against a longest training episode of 500, so **8.6 steps**, and one
510-step scenario in a regenerated index would put the criterion back in deficit.
`horizon_steps == 500` is the assertion that would say so.

---

## 2. The behavioural specification, level-free

`BEHAVIOURAL-SPEC-DRAFT.md` in this directory. Thirteen orderings `P1`–`P13` over
trajectory pairs, stated on quantities the §3.4 atomic vector already carries and
without naming a level; five impossibility results `I1`–`I5`; and a table naming,
for each free quantity, the measurement that fixes it.

The three results that change what a redesign has to achieve:

* **`I1`** — no Markov, bounded, per-step progress channel has a duration-invariant
  discounted return, so **O3 as a dominance is unobtainable at `γ<1` by any
  hierarchy with any number of levels in any order.** It must be restated as a
  finite exchange rate. This settles `D15` on mechanism rather than on preference.
* **`I3`** — O3 and promptness put opposite requirements on a threshold placed on
  the progress channel, **at the same critical value** (measured: `τ ≤ 27.97` and
  `τ > 27.97` on the §4.6 reference pair), because at `γ<1` the two comparisons
  *are* the same comparison at that channel. **Progress must not be a thresholded
  channel.** This is `F8`'s tension in its episodic form, and it is exact.
* **`I5`** — O2 and O3 together need a *finite* exchange rate, which a threshold
  cannot supply (it gives 0 inside budget and ∞ outside). **On this pair the scalar
  control is structurally better than any ordered arm**, and that is a result to
  report, not a defect to fix.

---

## 3. Candidates, scored

Seven architectures, ten orderings, three comparison rules, both discounts.
`P` = passes, `F` = fails. Full output: `g3_battery.py`, `g4_hacking.py`.

| | channels | weights | thresholds | scalar | strict lex | verdict |
|---|---:|---:|---:|---|---|---|
| **A0** status quo | 6 | 6 | 5 | fails O3 | fails O1, O2 (with traffic), O3 | baseline |
| **A1** progress last, negotiable as a priority level at weight 1 | 5 | 4 | 4 | **fails O2 (−5.07) and O4 (−13.08)** | ok | **falsified** |
| **A1c** progress last, negotiable in the continuous tail | 5 | 5 | 4 | fails O3 | passes O3 | survives, weak scalar |
| **A2a** progress last, collision merged with interaction | 4 | 5 | 3 | fails O3 | passes O3 | survives; but halves the impact gradient (P8 margin 2.82 → 1.22) and merges away two of the four tail-shaped channels |
| **A2b** progress last, compliance merged | 4 | 4 | 3 | **fails O2, O4 (−23.08) and O5 (−0.15)** | ok | **falsified** |
| **A6** six levels, negotiable by indicator (`F3`) | 6 | 6 | 5 | **passes all ten** | fails O3 | survives, no simplification |
| **A7** progress last, negotiable by bounded indicator, no L6 | **5** | **5** | **4** | **passes all ten** | **passes O3** | **recommended** |

### Two architectures deliberately not benched, and why

**A pure constrained MDP with no ordering** (budgets on every channel, maximise
progress) is not a separate candidate: it is what any thresholded architecture
*becomes* in the feasible regime. Adopting it as the rulebook would discard the
only thing the ordering supplies — graceful degradation when the budgets cannot all
be met — for no simplification of the reward. See objection 3 below.

**A two-tier architecture** (one hard-violation gate, one scalar objective
combining everything else) is rejected on the thesis's own terms rather than on
behaviour: with a single constrained channel there is no *order* among constraints,
so the lexicographic arm has nothing to compare against the scalarization and the
experiment loses its object. **This sets a floor on simplification**: the
architecture needs at least three constrained channels with genuinely different
priorities. A7 has four; `A2a` has three and pays for the third by merging the
collision outcome with the anticipatory interaction indicators, which §3.1 argues
against on semantics and §2.1's distributional criterion argues against on
measurement.

### Why the two falsified candidates fail

Putting the negotiable-lane channel into the priority block at weight `a⁰ = 1`
makes accumulated relaxation outrank a collision: 20 steps of full-severity
relaxation cost 26 against `a³ = 15.6`. This is `I4` (per-step dominance is not
per-return dominance) biting on a channel that fires often. Any candidate that
tries to buy the ordering "negotiable compliance above progress" as *scalar
dominance* dies here.

### Reward-hacking probes

**All figures at one parameterisation: `γ = 0.996` (the shipped discount) and
`w₅ = 0.15` (the recommended value).** An earlier revision of this table quoted the
creep row with A0 at `γ = 0.996` and A7 at `γ = 0.9982`, and the resulting "13×"
was an artefact of that mixture; it is not producible at any single setting.

| probe | outcome |
|---|---|
| sprint 40 % of the mission then idle | loses under every candidate (A0 −25.1, A7 −24.9) |
| complete then overshoot past the goal | **refuted as a hazard**: the tracker freezes on the directed gate crossing (`mission/tracker.py:260`) and the episode terminates, so no credit is collectable past the goal. (The residual is `D14`'s route/gate gap, explicitly out of scope.) |
| one clipped route-projection jump with no motion | **pays under every candidate**: +1.345 under A1, A1c, A2a, A2b and A7, and +1.358 under A0 and A6 — the two that keep L6, whose honest standing-still baseline is −2.757 rather than 0.000. `P11` is currently violated. **The reading of `telescoping_max_error = 62.294` that stood here — "the largest single number in the system", treated as a reward-hacking exposure — is withdrawn; see §8.1.** It is a *deficit*: the clip under-pays. The exposure is on the other side of the same clip (§8.2) |
| collide at fault to end a losing episode | **pays.** Break-even measured: **2.2–2.5 steps** of fully-violated interaction, or **5.3–6.2 steps** of non-negotiable violation, cost more than one at-fault collision at impact 0.6. Architecture-independent — it is `I4` again, and ADR-081 already settled the claim by restating it per-step. Belongs in the limitations, and in `P10` |
| creep along a marking for a whole episode | loses under every candidate, but by **−1.810 under A0** and **−19.988 under A7**: A7's deterrent is **11.0×** stronger. (The ratio is 19.4× at `w₅ = 0.25`, and it is 11.0× at both discounts — it does not depend on `γ`.) |
| relax for 30 steps with no benefit | **−0.944 under A0, −4.673 under A7** — 4.9× stronger |

---

## 4. The recommendation

> **Adopt A7 — five channels, progress last and unthresholded, `L6` deleted, the
> negotiable-lane channel charged by a bounded satisfaction indicator at
> `w₅ = 0.15`, and `η`, `λ₆` and `φ` removed, taking the reward from six free
> weights to four — at the cost of a projected below-standstill fraction of about
> 5.2 % against `AC-RB5.1-04`'s 7.45 % ceiling, the one column that needs the
> 45-minute grid to confirm, and of the braking criterion falling from 1.38× to
> 1.22× the physical limit. §5.4's thinnest margin *improves*, 1.1121 → 1.1390.**

```
K1  collision safety          (at-fault impact)              threshold 0
K2  interaction risk          (ttc, clearance, rss_lateral)  threshold from the panel
K3  non-negotiable compliance (offroad, signal, stop,        threshold from the panel
                               crosswalk, vehicle_yield,      — and it cannot be 0
                               speed_limit)
K4  negotiable lane compliance(solid_line, wrong_carriageway,threshold from the panel
                               dashed_line)
K5  mission progress          (signed route advance)         UNTHRESHOLDED, last,
                                                             open-ended
```

The atomic vector of §3.4 is **unchanged**: the same fourteen entries — thirteen
sub-rule costs plus `Δq` — and no new observation. §3.3's intra-level aggregation is unchanged too — `c_L5`
stays the normalized sum of three, so `F3`'s cost ("L5 stops measuring *how much*
relaxation") **applies only to the scalar adapter**: the ordered and distributional
arms read the continuous channel, and the atomic vector keeps all three sub-rules
regardless. What changes is the comparison order and the scalar adapter, which §3.4
already declares to be adapters rather than the contract.

The §3.2 invariants are all preserved: `DELTA_Q_MAX = 1` and its clip; no new
observation; memoryless Markov sub-rules; the termination/truncation split; one
per-step vector for every arm. Invariant 5 (the 217,189-transition calibration) is
touched only in that `a`, `σ` and `λ₄` **do not move** — the re-run is needed for
one column, not for the weights.

**What the thresholds are on, and what the literature actually supports.** This is
the part of A7 that carries real residual risk, and a literature check done for
this review changed what can honestly be claimed about it.

*The last channel being unthresholded is the documented requirement, and only
that.* Vamplew, Dazeley, Berry, Issabekov & Dekker (Machine Learning 84, 2011),
§3.2.3 p. 58: "note: objective n will be unconstrained, hence `C_n = +∞`", with
`CQ ← min(Q, C_j)`. Tercan & Prabhu §3 restate it. **No stronger requirement — a
terminating, goal-reaching final objective — appears in either paper**; the
handoff's §7 item 2 is right about that, and Vamplew's own worked example of an
unconstrained final objective is deliberately *not* goal-reaching ("maximizing
factory production while maintaining a required safety level"), which is exactly
A7's shape. **Caveat that must be carried:** the property holds for the *absolute
thresholding* family. Under *slacking* (Li & Czarnecki; Skalse et al. IJCAI 2022)
the loop runs to `i = k` and **the last objective is slacked too**, so "the last
channel is unthresholded" is a statement about the mechanism, not about the
architecture.

*Per-state Q-value thresholds are the wrong mechanism for this rulebook, and the
reason is measured rather than argued.* Vamplew et al. §7.2 pp. 75–76 report that
TLQ "performs extremely poorly when the time objective is thresholded", and name
the mechanism: "its action selection mechanism considers only the expected future
reward for each action from the current state, **ignoring any rewards received
earlier in the current episode**" — failing "regardless of the value of the
threshold". (Tercan & Prabhu attribute this to Vamplew, correctly; it is not their
own result.) Separately, Pineda, Wray & Zilberstein (AAAI-FS 2015) *measured* that
lexicographic value iteration with the per-state slack its own bound prescribes
"failed to make any significant change in costs, with respect to using no slack".
Both point the same way, and both reinforce `I3`.

*An episodic budget has no per-state equivalent, and multiple episodic budgets are
an open problem.* Tercan & Prabhu, Appendix D.3, treat exactly this case — a
per-episode budget, solved by augmenting the state with the amount already spent —
and state that "the corresponding discounted threshold **actually depends on the
trajectory**", i.e. there is no Q-value threshold that implements a budget. Their
optimality proof (D.3.2) is stated on **undiscounted** cumulative reward, so it
does not cover a discounted final objective as written. And D.3.3, on more than one
constrained channel: "extending the approach above to this setting is **not
straightforward** … we need to know which constraints can be satisfied together",
with the three strategies they sketch all suffering a halting problem and the third
"not concretized and … intended mostly as an idea for future research".

**What that means for A7, stated plainly.** The thresholded arm has two routes and
both carry a declared gap: the policy-gradient route (Tercan & Prabhu's
Lexicographic REINFORCE, which is what `F8` concluded the mechanism must be)
compares an accumulated episodic return against the threshold, but accumulates it
**undiscounted from a single sampled episode** while its own objective is
`J(θ) = V^{πθ}(s_init)` — a one-sample high-variance estimator the paper does not
comment on; and the state-augmentation route is proved for **one** constrained
channel and open for several, besides needing the accumulated exposure in the
policy's input, which is an `OBS-V1.3.x` amendment. Also on the ledger: Pineda et
al.'s Lemma 1 — finding an optimal *deterministic* policy for a lexicographic MDP
is NP-hard, though randomized policies stay polynomial.

**This is a cost of the thresholded arm, not of A7, and A7 reduces it.** A0 has
five constrained channels against A7's four, and A0's fifth is progress, the one
`I3` shows cannot be thresholded at all. Every constrained channel added is one
more constraint in a set the literature says it does not know how to satisfy
jointly. **That raises the value of removing channels above what §3 priced it at**
— and it is the one argument that could still favour `A2a`'s three, which I did not
weigh when scoring it and which is not enough on its own to overturn §3.1's
outcome-versus-indicator argument or §2.1's distributional one.

*Precedent for progress last.* Censi et al. Fig. 10 places "Passenger comfort / Own
progress towards goal" at the **bottom** of their rulebook, and Definition 17 adds
new rules at the bottom. Castro et al.'s minimum-violation formulation likewise
leaves an unconstrained **cost** as the last criterion — minimum-time, with the
authors noting the algorithm "applies to a much wider class of functions including
discounted cost". Neither is authority for A7, and Castro's structure differs in
one respect worth stating: it makes reaching the goal a *hard constraint above* the
violation ordering, which would contradict `P5` and O5 if transposed here. What
both support is the narrow point that an open-ended progress-or-cost objective
belongs last.

The behavioural specification states its budgets over trajectories, which is the
right altitude for a *requirement*; this section is the *mechanism*, and it is the
part with the most residual risk.

### What it buys

* **O3 holds again on the scalar arm at the shipped discount** (+0.581 at
  `γ=0.996`, +2.458 at `γ=0.9982`) — the ordering the whole v5.1 restructure was
  about, lost since ADR-081. And it holds on the ordered arms too, because the
  negotiable channel is consulted before progress.
* **`τ₄` ceases to exist.** `I3` proves a threshold on progress has an empty
  requirement; A7 has no such threshold, so `F8`'s 13× tension and `D1`'s 13–247 m
  span problem both dissolve rather than being calibrated around.
* **The degenerate thresholded policy is removed structurally.** The final
  objective is discounted progress, whose minimum is standing still. §7's objection
  to deleting L6 — that with L5 last the objective becomes "relax as little as
  possible", minimised by a legally stopped ego — does not apply, because L5 is not
  last; progress is. And `F6`'s challenge ("any redesign that deletes L6 must say
  what supplies that") is answered: nothing needs to, because no `τ₄` has blinded
  anything.
* **The unmeasurable parameter goes.** `η` is, by limitation 2's own admission, the
  one weight the expert panel cannot discriminate; A7 deletes it. `λ₆`, whose value
  `F10` shows is an open decision (`D15`), goes too. The `Δt/T_REF` normalization
  disappears entirely with them, so the reward stops carrying two different
  per-step scalings.
* **The O3 window becomes robust.** The shipped form can buy O3 back only inside
  **0.193 %** of the admissible `η` range (`F2`). A7's `w₅` window is
  **65.9 %** at `γ=0.996` and **72.1 %** at `γ=0.9982` — **~350× wider**.
* **`p5` goes positive again.** §5.5's headline claim is currently false at
  production (`−0.23`, `F14`); removing `λ₆` adds ≈ +3.87 to `p5` (`F7`) against
  the indicator's ≈ −0.27, so `p5 ≈ +3.4`.
* **The distributional criterion improves.** A7 keeps the four sparse, tail-shaped
  channels (interaction 0.3978 % of steps with `p75 = 0.988`; non-negotiable
  0.6418 % with `p99 = 1.0`; negotiable 0.8734 %; impact, a rare terminal event)
  and deletes the one that is nearly deterministic given the others — `c_L6 =
  1 − clip(Δq,0,1)` fires on **99.4024 %** of steps at `p50 = 0.884`, and is a
  pointwise function of the progress channel. It is the only channel whose return
  distribution carries nothing beyond its mean once progress is known.

### What it costs, stated once more without hedging

* `w₃/tail` 1.179 → 1.139, and §5.4's actually thinnest margin, at `k=2`,
  1.1121 → 1.0975. This partly spends what ADR-081 bought on the `w₃/tail` axis
  (1.04 → 1.18) and it is the honest price of pricing the negotiable channel at
  all. Note that ADR-081's "thinnest margin in the document" is `w₃/tail`; the
  binding §5.4 inequality is at `k=2` and is thinner in both configurations. It can be bought back by lowering `λ₄` to 1.9, which the calibration grid
  does **not** contain — the grid prices `λ₄ ∈ {0.5, 1.0, 1.5, 2.0, 2.15}` at
  `a=2.2, σ=0` and `{2.0, 2.5, 2.8}` at `a=3.0`, so `λ₄ = 1.9` at `a=2.5, σ=0.30`
  would need a run. Not recommended: the margin is small and `λ₄ = 2.0` is the
  approved, measured value.
* Below-standstill: the criterion moves from `R₀ < −0.81` to `R₀ < 0` because `λ₆`
  is gone, and `F7` bounds the resulting rise at **0.53 pp**; the indicator adds
  ≈ 0.27 reward units of expert cost. Projection **4.55 % → ≈5.2 %**, ceiling
  7.45 %. **This is the one column that cannot be derived and needs the run.**
* **The exact-lexicographic arm loses O2.** A7's strict-lex arm fails O2 where A0's
  passes it. But A0's pass is `F4`'s zero-traffic artefact: add one L2 step in forty
  at the residual `test_o1_fails_under_strict_lex_once_traffic_is_present` itself
  uses, and A0 fails O2 too. **Once the traffic residual is present, A7's strict-lex
  arm is no worse than A0's on every ordering and strictly better on O3.**
* One new parameter, `w₅`, with one new grid member in the instrument.

### Where `w₅ = 0.15` comes from

Criterion, stated before the value. Two physical bounds in the same currency,
seconds of arrival time:

* **lower** — the §4.6 reference shortcut must not pay. Direct measurement of the
  scalar margin gives `w₅ > 0.1313` at `γ=0.996` and `w₅ > 0.0736` at `γ=0.9982`;
* **upper** — passing must stay cheaper than waiting. Stopping from urban speed
  `v = 10 m/s` and returning to it at a comfortable `a_c = 2 m/s²` costs `v/a_c =
  5.0 s` of delay. One second of marking contact must cost less than that, giving
  `w₅ < 0.4477` at `γ=0.996` and `w₅ < 0.2641` at `γ=0.9982`; §5.4 caps it at
  0.3846 independently.

The binding choice is the **lower bound at the shipped discount**, so that the
architecture decision stands whether or not the discount decision is taken:
`w₅ > 0.1313`, rounded up to **0.15** for margin, because limitation 10 records the
reference shortcut as a stipulation rather than a measurement. Its cost is the
`a/tail` figure above. At `w₅ = 0.15` the scalar arm crosses a marking only if
doing so buys **1.83 m/s** of extra route advance, and one second of full lane
relaxation is worth **2.8 seconds** of arrival time.

### Does it preserve the asymmetry the thesis is testing?

Yes, and the gap is now locatable rather than assumed.

* **Where the thresholded arm can win.** The scalar control's tolerance for
  accumulated interaction risk **scales with what the mission is worth**: it will
  accept up to 9.7 fully-violated interaction steps to complete a mean Waymo
  mission and 19.8 to complete a mean PG one, against a measured expert exposure of
  0.42. A budget does not scale that way. And no scalar sum of expectations can
  express `P13` (tail aversion), which is the distributional component's whole
  purpose. Both gaps sit on the channels A7 keeps.
* **Where the scalar control wins, and it must be reported.** `I5`: O2 and O3
  together need a finite exchange rate, which only a scalarization supplies. A7's
  scalar arm satisfies both on one `w₅`; no threshold on the negotiable channel
  can.
* A7 makes the control **fairer** — it fixes the control's O3 failure — without
  touching either place the ordered arms have room. That is the trade the brief
  asks for.

### Four objections to A7, priced

**1. A7 reverts ADR-072, the entire v5.1 restructure.** It does — the negotiable
lane rules go back above progress, which is v5.0's placement. It does **not**
reintroduce v5.0's pathology, because `F3`'s reading applies in reverse: what
failed in v5.0 was the *price*, not the placement. v5.0 charged those rules at the
R3 priority weight `a = 2.2` per violated step, so standing still won past **37**
relaxed steps against a mean Waymo mission. A7 charges `w₅ = 0.15`, **14.7× less**,
so standing still wins only past **416** relaxed steps at full severity — twice the
Waymo episode, and 848 against a mean PG mission over a 500-step episode. The
pathology is out of reach.

**2. A7's exact-lexicographic arm loses O2.** True, and O2 is the one ordering the
v5.1 restructure bought. But `F4` shows A0's O2 pass in that arm is a zero-traffic
artefact: `TEST-RB5.1-02` builds the detour on `legal_drive()` with the default
`l2 = 0.0`. Add one L2 step in forty at the 0.05 residual that
`test_o1_fails_under_strict_lex_once_traffic_is_present` itself uses — reproduced
on the bench as `B2'` — and **A0 fails O2 too, at L2**. Once the residual any real
trajectory accrues is present, A7's exact-lex arm is no worse than A0's on every
ordering and strictly better on O3.

**3. With progress last and unthresholded, the thresholded arm is a constrained
MDP, and "lexicographic × distributional" risks collapsing into CVaR-constrained
RL, which is a populated field.** This is the sharpest objection and it is about
the thesis's novelty rather than about the reward. Three answers.

First, it is equally true of A0 — §2.5 of the handoff already states that in its
intended regime the thresholded arm is "constrained optimization of the last
objective, with L1–L5 as the constraints"; A7 makes that visible rather than
causing it.

Second, **a lexicographic ordering is provably not a scalarization.** Wray,
Zilberstein & Mouaddib (AAAI 2015), Proposition 6: "The optimal policy of an LMDP
π may not exist in the space of solutions captured by its corresponding scalarized
MOMDP's policy π_w", proved by counterexample. That is the citable form of the
claim. **It is not in Censi et al.** — an earlier draft of this document said it
was and that was wrong. Censi et al. never compare rulebooks against a
scalarization, never use the word, and Definition 16 explicitly *admits* linear
combinations with positive coefficients inside an equivalence class, using one in
their own experiment. What Censi et al. do claim is specifiability for regulation,
partial specification, and (Remark 12) that keeping safety rules at the top keeps
a learning system safe "not even with adversarial data".

Third, the graceful-degradation argument still holds but must be argued on
mechanism rather than on Censi: a constrained MDP is undefined when the budgets
cannot all be met, while a lexicographic ordering minimises the highest-priority
violation first and the *order* is what does that. Whether that case is frequent is
measurable and unmeasured — the expert violates the non-negotiable channel on
**0.6418 %** of steps, so a zero budget there is already infeasible for the human.
**The thesis's claim should be argued on the infeasible case, and that argument
needs a measurement nobody has run.**

**4. Deleting L6 halves the incentive to close the last stretch.** Measured:
completing versus covering 99 % of the mission and idling to the horizon is worth
**0.66–2.99** reward units under A0 and **0.57–0.77** under A7, so L6 supplies
13–74 % of it depending on when the agent finishes. Two things bound how much this
matters. Both figures are an order of magnitude below **8.375**, one fully-violated
interaction step, so *neither* architecture makes the last stretch worth a risky
manoeuvre — the real gap is that mission success is a zero-value terminal, and it
is shared. And A0's incentive is the larger one only because it is proportional to
`199 − T_c`, the *remaining length of the Waymo log*, which ADR-081 itself calls
"the length of the Waymo log, not a property of driving"; A7's is nearly constant
because it is proportional to the remaining mission distance, which is a property
of the task. **A7's completion incentive is smaller in magnitude and better in
kind.**

If measurement later shows the incentive insufficient, the instrument is a terminal
completion bonus `B` on the gate crossing, worth `γ^T·B`: it does not enter §5.4 at
all (a one-off terminal reward is not a per-step quantity), it prices lateness at
the outcome level, and it is a sparse terminal event of exactly the shape the
distributional component wants. It is **not** recommended now — it adds a parameter
against a brief whose object is simplification, and no measurement yet shows the
current incentive is too small. The criterion that would decide it: the fraction of
evaluation episodes that reach ≥95 % route completion without a gate crossing.

---

## 5. Realigning the scalarization

Only after the architecture is approved. The form:

```
r_t =  Σ_{k=1..3} a^(4−k) · [ (step(m_k) − 1) + σ·m_k ]
     +  w₅ · [ (step(m₅) − 1) + σ·m₅ ]
     +  λ₄ · Δq_t
```

with `m_k = −c_k ∈ [−1,0]`. **`η`, `λ₆`, `φ` and the `Δt/T_REF` factor are removed.**

`a = 2.5`, `σ = 0.30`, `λ₄ = 2.0` unchanged — approved and measured — and
`w₅ = 0.15`. Four free weights against today's six.

Rank preservation becomes, with the negotiable channel now inside the priority
block as a sub-unit level:

```
a^(4−k) > (1+σ)·Σ_{j>k, j≤3} a^(4−j) + w₅·(1+σ) + λ₄·ΔQ_MAX
```

Margins: `k=1` 1.1514, `k=2` 1.1478, `k=3` **1.1390** (thinnest), against A0's
1.1165 / **1.1121** / 1.1792. Every level passes, and **the binding margin is wider
than today's**.

### Why `φ` goes

Motivating `φ = 0.25` honestly ends in deleting it, on four grounds that compound.

1. **Its stated job is already done.** `φ` breaks the tie the satisfaction
   indicator creates between two margin vectors with the same discrete
   satisfaction pattern. At `σ = 0`, where `φ` was set, the priority term was
   constant inside the violated set and that tie was real. At `σ = 0.30` it is
   not: the slope `σ·a^(4−k)` is non-zero at every level. Mechanism, not
   preference.
2. **Its shape is the one ADR-081 itself criticises** — being absolute, the
   grading is inversely proportional to importance. `φ` is **5.1 %** of the
   severity slope at `k=1`, **11.8 %** at `k=2` and **25.0 %** at `k=3`.
3. **No document derives the value.** `F11` traces 0.25 to Veer et al.'s (ICRA
   2023, Theorem 1) averaged-robustness tie-breaker `1/N` with `N = 4`, their
   four-level schema, now summed over three margins. Where the only argument for a
   constant is the specification, that is a finding.
4. **It returns A7's only structural cost.** The thinnest §5.4 margin goes
   1.0975 → **1.1390**, above today's 1.1121.

**Falsified before recommending:** over the whole ten-ordering battery at both
discounts, **no ordering changes sign** and no reward-hacking probe changes
verdict; the largest movement is O5 (waiting at a red) from +9.075 to +8.083.

**The cost, in the same sentence:** `a_req^max` falls from 12.43 to
**10.96 m/s²**, i.e. from 1.38× to 1.22× the physical braking limit the criterion
has to clear, and the effect on the expert's mean return under the six-level
reward is **unmeasured** — `F11` prices it only on the four-level family whose rows
cannot be read across. **The falsifier:** one grid member at `φ = 0` in the same
45-minute run A7 needs anyway; if below-standstill breaches the 7.45 % ceiling,
revert to 0.25 and accept the 1.0975 margin. `ScalarizationConfig._validate_six_level_weights`
(`src/thesis_rl/reward/scalarization.py:147`) needs the corresponding change; it
currently requires `η` and `λ₆` to be finite and non-negative, and both go along
with `φ`.

Two things the scalar adapter deliberately does **not** do, and both are the
asymmetry rather than defects: it does not make the negotiable channel dominate
progress per step (that would need `w₅ > λ₄ = 2.0`, which `I4` forbids), and it
does not make one impact dominate a completed mission (that needs `a > 4.33`,
which the calibration rejects).

---

## 6. What must be measured or built before implementation

| item | why | cost |
|---|---|---|
| per-episode distributions of `X_int`, `X_hard`, `X_soft` on the expert panel | every threshold in A7 is calibrated from them, by limitation 1's own argument for `d₂`; the instrument currently emits only per-step marginals and the mean reward contribution | a small addition to the existing accumulator, one 45-minute run |
| one grid member at `w₅` under the A7 reward | `fraction_below_standstill` is the one column `F7`'s algebra cannot derive | same run |
| **the negative-clip ratchet** (§8.2, `C50`) | +72 reward units per closed lap at zero net displacement, unbounded, and in A7's thresholded regime L4 is the *only* gradient inside budget, so it is the entire signal | **RESOLVED AS A DECISION, 2026-09-09: change nothing in the reward.** The remedy named in the cell that stood here was wrong twice. `ROUTE_CONTINUITY_JUMP_FACTOR` **does not exist** in any Python file — the bound was in production and `e63e0bf` removed it on 2026-08-03 — so nothing can be "promoted"; and at ADR-035's factor of 2.0 it would not close the ratchet anyway, since the cursor could move 4.44 m in a step while `Δq` saturates at 2.2222 m, leaving about +18.8 per lap instead of +36. The bound that closes it is one clip width exactly, which `DRIVING-MISSION-V1.1` §8 withdrew as a "continuity/clamp" protocol. See the 2026-09-09 audit |
| ~~emit `telescoping_max_error` signed~~ (§8.1, `C51`) | published as `abs(...)`, so a surplus is indistinguishable from a deficit | **DONE 2026-09-09.** Emitted on both signs, plus the per-episode distribution with `concentration_in_worst_episode`, because two extremes cannot say whether the mass sits on one episode or fifty |
| ~~write the near-revisit re-audit script~~ (`V3`) | a standing constraint requires it on any regenerated index and it did not exist | **DONE 2026-09-09.** `scripts/audit_route_near_revisits.py` with five acceptance tests, reporting at one clip width and emitting a distribution. Reproduces the 2026-09-05 run exactly and carries the `g6` hairpin as a positive control |
| ~~the guard test reads `L` from the frozen index~~ | `F9` | **DONE 2026-09-09.** Reads 500 control steps from the committed index and pins the *requirement* (`γ ≥ 0.998169` at `a = 2.5`), not any discount and not the criterion's verdict, which land with the A7 change |
| update `tests/test_scal_v14.py` | it writes §5.1 out independently of the implementation with `ETA = 1.0`, `LAMBDA6 = 0.2` and `DT_RATIO = 0.1` hardcoded, and parametrises the inadmissible-weight cases on `relaxable_weight` and `progress_rate_weight`; all of those change | small, but it is a mandatory test and the change needs approval |
| argmax-within-level instrumentation | `F12`: under `max`, a sub-rule that is never the argmax contributes nothing, and `clearance`'s max cost 0.7094 against 1.0 for the other two may make it inert. `worst_named` already exists and is discarded at `scripts/measure_expert_rulebook_transition.py:2455-2456` | small, same run |

---

## 7. What I did not verify, and corrections to the handoff

**Not verified.** The claim that `A2a`'s merge of collision with interaction is
acceptable rests on a co-occurrence measurement that does not exist (`F12`); I
scored it on the fixtures only. The projected below-standstill of A7 is a
projection. One literature source could not be obtained: Issabekov & Vamplew
(AJCAI 2012), closed access with no open mirror — it matters only because gTLO
attributes to *that* paper the restriction "thresholded rewards zero except at the
last step" while Tercan & Prabhu attribute the path-objective failure to Vamplew
2011; the latter was verified directly in Vamplew 2011 §7.2, so the attribution
used here is sound.

**Corrections to the handoff's own §7, which corrected an earlier session and
introduced one error of its own.**

* §7 item 3 says "Li & Czarnecki **prove** the episode-level loss is bounded by
  `τ_i/(1−γ)`". **They do not prove it and it is not theirs.** Their paper contains
  no theorem, lemma or proposition at all; the bound appears as a single unproved
  sentence after Eq. 7. The result is **Wray, Zilberstein & Mouaddib, AAAI 2015,
  Proposition 1** ("if `η_i = (1−γ)δ_i` then `V^η_i(s) − V^π_i(s) ≤ δ_i`"), which
  Li & Czarnecki themselves cite, saying their Eq. 7 "essentially becomes the
  Q-learning version of lexicographic value iteration".
* The quotation "If the look-ahead horizon is long, so that `γ ≈ 1`, the margin is
  very small" is real but is attached to **a different quantity**: it closes their
  point (3) about the *min-operator's* noise margin `(1 − 1/γ)τ` under function
  approximation, not about the slack bound. The two move in **opposite**
  directions as `γ → 1` — the noise margin closes, the slack guarantee diverges —
  so joining them into one claim is not citable.
* What survives, and is stronger than either, is a *measurement*: Pineda et al.
  (AAAI-FS 2015) report that lexicographic value iteration using the `η` that
  proposition prescribes "failed to make any significant change in costs, with
  respect to using no slack". The per-state slack bound is operationally vacuous,
  and that is measured rather than argued.
* `Absolute Slacking` is attributed in the handoff to Li & Czarnecki. Tercan &
  Prabhu's own appendix attributes it to Wray et al. 2015 and Pineda et al. 2015
  ("see [35] and [21] for a definition based on slacks"), and Li & Czarnecki agree
  in their own §2. They are the DQN transposition, not the origin.
* An earlier draft of **this** document attributed to Censi et al. the argument
  that a rulebook beats a scalarization in infeasible situations. **They make no
  such comparison**, never use the word scalarization, and Definition 16 admits
  positive linear combinations inside an equivalence class — one is used in their
  own experiment. Corrected in objection 3; the citable form is Wray et al.
  Proposition 6.

**Verification debt found in passing, in addition to `F14`'s list.** The O1–O6
ordering fixtures call `measurement.v51_reward(...)` without `base` or `severity`,
so they inherit `FINAL_PRIORITY_BASE = 2.2` and `FINAL_SEVERITY = 0.0` — the
deliberately pinned v5.1 §5.5 baseline, documented at
`scripts/measure_expert_rulebook_transition.py:249-255`. **The six orderings are
therefore asserted at a weight pair production has not used since ADR-081.** The
pinning itself is correct and is there so the grid's baseline row keeps reproducing
the published figures; what is missing is a second assertion at the shipped
`a = 2.5, σ = 0.30`. I checked and every ordering still holds there (O3 excepted,
which is the known failure), so this is a verification gap and not a behavioural
one — but a change to `a` or `σ` would move production without moving the fixtures.
`tests/test_scal_v14.py` does guard the shipped pair for the scalarization
arithmetic and for §5.4 admissibility.

**Corrections to this document, found by an adversarial verification pass over the
artifact built from it.** Three claims in earlier revisions were wrong and are
fixed above:

* the creep-probe row paired **A0 at `γ = 0.996`** with **A7 at `γ = 0.9982`**, and
  the "13×" it reported is producible at no single setting. At one parameterisation
  the ratio is **11.0×** (`w₅ = 0.15`, either discount) or 19.4× (`w₅ = 0.25`);
* "pays +1.345 under every candidate" was true of five candidates out of seven.
  A0 and A6 keep L6, so standing still costs them −2.757 instead of 0 and their gap
  is **+1.358**;
* the gratuitous-relaxation row quoted **−7.8**, which is the figure at the
  instrument's default `w₅ = 0.25`, not at the recommended 0.15 where it is
  **−4.673**;
* "fourteen sub-rule costs, same `Δq`" over-counted the §3.4 vector by one: it has
  fourteen *entries*, thirteen of which are sub-rule costs.

None of them changes a verdict, a decision or a recommendation — every one is a
figure in the probe table or a count — but the first is the kind of mixed-frame
error `§7` of the handoff warns about, committed by this document.

**Corrections to repository findings.**

* `F12` says "L5's sum-of-three has never summed more than one term". The sub-rule
  table gives `dashed_line` 1145 violated steps and `solid_line` 758, totalling
  1903 against the channel's 1897, so **six steps do carry two sub-rules at once** —
  they are simply small enough that `max c_L5 = 0.3313` still comes from a single
  term. The conclusion (the sum is empirically almost unexercised) stands; the
  statement was too strong.
* The handoff's unverified candidate "success being a zero-value terminal while
  ~16 % of the polyline lies beyond the goal" is **refuted as an exploit**: the
  tracker freezes at the directed gate crossing and the episode terminates, so no
  progress credit is collectable past the goal.
* The handoff's unverified candidates "deliberate collision cheaper than ~2.5 steps
  of conflict" and "projection flips paying up to +2.0 for no motion" are both
  **confirmed**, at 2.2–2.5 steps and +1.345 respectively.
* The `v51_*` weight grid **does** price `λ₄` downward — 0.5, 1.0, 1.5, 2.0, 2.15,
  with below-standstill 8.8 / 6.1 / 4.5 / 3.4 / 3.2 % at `a=2.2, σ=0`. The handoff
  and §4 both say `λ₄` cannot be reduced without saying that the reduction is
  measured and what it costs. It is, and it costs about 1.1 pp per 0.5 of `λ₄`.

---

## 8. After the review — three questions asked of it, and what they found

Recorded here rather than left in a conversation, because two of the three close
without work and the third grew. All three concern the progress channel or the
degenerate policies §4 claims A7 removes. Working continuation:
`handoff-progress-channel-2026-09-09.md`.

### 8.1 The telescoping error is an under-payment — the earlier reading is withdrawn

§3's probe table used to call `telescoping_max_error = 62.294` "the largest single
number in the system" and treat it as a reward-hacking exposure. **That is wrong.**

Mechanism: a forward projection jump of `Δs = 3.609 m` gives `Δq_raw = 1.624`,
clipped to `1.0`, so the channel **loses** 0.624 units of credit. The clip can only
reduce the sum on the forward side. And the backward side does not compensate on
the expert panel: `result.rulebook_v51.behind_peak` reports `max_m = 0.053` and
`steps_beyond_1m = 0`, so the largest backward step over 217,189 transitions is
**5.3 cm** against a clip threshold of 2.2222 m — the negative clip never binds
there. All 62.294 units are deficit, over 1298 clipped steps (0.60 %). On the
expert panel this is an accuracy problem, not a hazard.

**But the statistic cannot say so**, because it is computed as
`abs(v51_episode_delta_q − v51_episode_delta_s / V51_REFERENCE_ADVANCE_M)`
(`scripts/measure_expert_rulebook_transition.py:2637-2639`) — the sign is
discarded, so a surplus and a deficit are indistinguishable. Emitting it signed is
one line and it is the detector for §8.2. Filed as `C51`, and **done on
2026-09-09**: emitted on both signs as `telescoping_max_surplus` and
`telescoping_max_deficit`, plus the per-episode distribution with
`concentration_in_worst_episode` — because two extremes say how large the worst
episode is and nothing about whether the mass sits on one episode or fifty, and a
bottom-tail statistic like `fraction_below_standstill` is measured exactly where
that distinction bites.

### 8.2 The negative clip is an unbounded ratchet — executed

The clip is symmetric; the magnitudes it bounds are not. Forward, vehicle physics
already caps `Δs` at `v_max·Δt = 2.2222 m`. Backward, a projection branch jump is
unbounded and is charged `−1`.

`g6_ratchet.py`, real `RoutePolyline`, hairpin of 84.03 m whose legs are one lane
apart, 1 m per step, projecting sequentially as the tracker does:

| trajectory | paid | telescoped | worst single `Δs` |
|---|---:|---:|---:|
| honest — the whole route once | 37.81 | 37.81 | 0.00 m |
| **closed loop ×1, net displacement zero** | **+36.00** | 0.00 | **−84.03 m** |
| closed loop ×2 | +72.00 | 0.00 | −84.03 m |
| closed loop ×3 | +108.00 | 0.00 | −84.03 m |

**+36 channel units = +72 reward units at `λ₄ = 2.0` per lap, linear in laps.**

What blocks it today is the geometry of the frozen panels, not the reward: the
2026-09-05 read-only audit over all 3,500 missions found zero routes with two
portions of `|Δs| > 15 m` closer than 6 m in plane. That is a property of the
population; its own standing constraint requires a re-audit on any regenerated
index; and the 15 m threshold does not cover revisits between one clip width and
15 m, which is all the ratchet needs. Filed as `C50`.

> **Two claims that stood here are withdrawn, 2026-09-09.**
>
> *"`ROUTE_CONTINUITY_JUMP_FACTOR = 2.0` already gives the projection a
> plausibility bound … promoting it adds no level, weight or observation field,
> because the constant is already there."* **The constant does not exist in any
> Python file.** It survives in ADR-035's prose and two docstrings citing it by
> name. The bound *was* wired into `evaluate_progress` on 2026-07-30 and
> **`e63e0bf` removed it on 2026-08-03** with the driving-mission v1.1 refactor:
> `mission/tracker.py::project` now documents itself as projecting "without a
> jump envelope or clamp", and no production call site passes `max_s_jump_m`.
> ADR-035 still asserts the bound in the present tense (`C52`). And at factor
> 2.0 it would not close the ratchet: the cursor could move 4.44 m in a step
> while `Δq` saturates at 2.2222 m, leaving about +18.8 per lap.
>
> *"The re-audit script does not exist (`V3`)."* It exists:
> `scripts/audit_route_near_revisits.py`. It reproduces the 2026-05 figures and
> reports additionally at **one clip width**, with the `g6` hairpin as a positive
> control.
>
> *And the verdict that first run produced — "the exposure is not armed" — is
> itself withdrawn.* It cited 0.247 m of fold excess, which is **a monotone
> function of lateral reach evaluated at an arbitrary radius, not a bound**: the
> same table gives 3.226 m at 5 m of reach, 13× more, and nothing in the runtime
> bounds the reach (`out_of_route_done: false`, `relax_out_of_road_done: false`,
> and `is_physically_out_of_road` degrades for route drift by ADR-053's design).
> The same committed run already carried the contradiction — unbounded band,
> on-route under-charge **max +128.751 with 1601 of 3,500 routes positive**, and
> a branch-switch of **180.484 channel units** at 58 m of lateral excursion,
> about five `g6` laps in one jump. The defensible claim is "not armed **within
> 5 m of lateral reach**".
>
> The recommendation survives on a narrower reason: nobody has shown a policy
> drives ~40 m off its route, and every algebraic closure spends a specification
> amendment. Full account, including the first per-record join of this audit with
> `D14`'s, in `docs/audits/progress_channel_integrity_2026-09-09/README.md`.

### 8.3 The parallel road is not excluded, and A7 makes it matter more

Measured 2026-09-08 over all 3,500 records: uncovered same-direction surface at the
**goal cross-section** in 1534/3500 (43.8 %), at a median gap of 16.55 m against a
median gate length of 10.26 m; 426 (12.2 %) within one ego width; residual real
exposure 289 records (8.3 %). The audit's own README has a section titled "What it
does not establish", and `D14` repeats it: the measurement covers the goal
cross-section only, so it does **not** show that an ego could drive such surface
while banking the `R4` budget, nor whether a trained policy goes there.

Three verified facts compose badly with progress last and unthresholded: no cost
opposes it (`offroad` is the union of every vertically compatible lane and
`wrong_carriageway` charges only opposing surface), no termination opposes it
(ADR-053 removed it by design, naming parallel carriageways), and no cost *below*
L4 can oppose it. The obvious remedy is `D14`'s exit (b), which
`DRIVING-MISSION-V1.1` §1 and §8 forbid, so it is a specification amendment and a
user decision.

> **The walk has been run, and two things here are corrected, 2026-09-09.**
>
> *"and under A7 there is nothing below L4"*, offered as closing the last remedy
> shape, **is the wrong conclusion.** The remedy wants to sit *above* progress,
> and A7 is the architecture that puts it there: in the shipped order an
> off-corridor rule's natural home is L5, *below* L4, so the off-route trajectory
> banks the larger L4 total and L5 is never consulted. Under A7 the same channel
> is `K4`, above `K5`, where an in-corridor trajectory has `K4 = 0` exactly and
> wins before progress is compared. A7 does not shrink the remedy space; it shows
> the space was already empty in the shipped order. What forecloses the remedy is
> the specification, and `driving_mission_v1.1_specification.md:104` forecloses
> **three different remedies in one sentence** — continuity/clamp protocols, off-route
> `R4` zeroing, and runtime authority of any final lateral envelope.
>
> *The measurement it says is missing now exists*, as that instrument's own
> `--walk-spacing-m` mode: 3,500 records, zero unusable stations. Reported in
> **metres**, because the share-of-route framing first drawn from it was flattered
> by short routes (median route length 29.1 m against the index's 113.6 m, with
> 123 of 130 below the index median): **6 records (0.17 %)** carry a corridor
> worth a whole mean mission, and **7 (0.20 %)** have a route even of median
> length. The mechanism is junction geometry — 199 of the 318 in the band are
> `topology=intersection`, over-represented 12.2 % against 3.4 % for `simple`.
> **Decision, 2026-09-09: no remedy, recorded as an observation.** The 318 records
> are committed per row in
> `docs/audits/progress_channel_integrity_2026-09-09/d14_corridor_records.csv`.

### 8.4 Collide-to-escape is a scalar-arm artefact — a result, not a defect

A trajectory enduring 200 steps of interaction violation at `c = 0.9` against one
colliding at fault on step 20, under A7 at `γ = 0.9982`, `w₅ = 0.15`, `φ = 0`:

| | scalar | ordered |
|---|---:|---|
| endure | −1265.96 | `K1 = 0.0000` |
| collide | −173.14 | `K1 = 0.5788` |
| verdict | **prefers colliding by 1092.8** | **the endurer wins at `K1`** |

The ordered arms compare the collision channel first and the non-colliding
trajectory has `K1 = 0` exactly, so the collider loses at any `τ₁ < 0.5788` — and
the recommended `τ₁` is 0, under strict lex and thresholded alike. **This is the
mirror image of `I5`:** there a scalarization is structurally better than any
ordered arm, here an ordered arm is structurally better than the control, and both
magnitudes are measured. `I4` says no bounded scalar sum can do better. No fix; it
belongs in the write-up as a result.

### 8.5 A terminal completion bonus should not be added

A bonus `B` on the gate crossing is worth `γ^T·B`, a time preference at the
*outcome* level, so it pushes where O3 and O5 pull against. At `γ = 0.9982`: O3
holds while `B < 47.2`, O5 while `B < 303.7`, and making the last stretch worth one
fully-violated interaction step (8.125) needs `B ≥ 9.7`. The window is non-empty —
**but the minimum value that fixes the stated problem is the value that puts
arrival before safety.** In the ordered arms `B` lives entirely inside `K5`, below
every safety channel, so it can cause no regression there and fixes nothing either.
Not recommended. The diagnostic that would reopen it: the fraction of evaluation
episodes reaching ≥95 % route completion without a gate crossing.
