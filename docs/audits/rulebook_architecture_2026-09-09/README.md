# Rulebook architecture review, 2026-09-09

Seven candidate rulebook architectures scored against fourteen behavioural
properties under three comparison rules, at two discounts; one verified defect of
the shipped discount; one executed reward-channel exploit; and a recommendation.

**Status.** The **measurements and the impossibility results are results**. The
**recommendation is a proposal and nothing in it is approved**: no ADR, no
specification amendment and no ExecPlan exists for it, and no repository
behaviour has changed. `REVIEW.md` §4 and `BEHAVIOURAL-SPEC-DRAFT.md` are kept
here because the evidence the eventual decision will rest on has to survive, and
because several findings hold whether or not the recommendation is ever adopted.

Kept in the repository rather than only under `outputs/` for the same reason
`reward_calibration_2026-09-07/README.md` gives: an analysis that lives on a
scratch filesystem is an analysis that will eventually be gone.

## Contents

| file | what it establishes | needs |
|---|---|---|
| `REVIEW.md` | the review: the discount decision, the candidate table, the recommendation, the scalarization realignment, the threshold mechanism against the TLO literature, §7 listing what was **not** verified, and **§8, added after the review and superseding two statements inside it** | — |
| `BEHAVIOURAL-SPEC-DRAFT.md` | **DRAFT, unapproved.** `P1`–`P14` stated over trajectory pairs without presupposing a hierarchy, the impossibility results `I1`–`I5`, and a table naming the measurement that fixes each free quantity. This is the artefact the candidates were scored against | — |
| `g1_discount.py` | re-verifies the training horizon against the frozen index and the truncation rule; restates the discount criterion as `γ^L ≥ 1/a`; prices every exit; shows the calibration grid is undiscounted | reads the frozen index; takes the repository root as `argv[1]` |
| `g2_bench.py` | the bench: seven architectures as (channels, scalar adapter) pairs, plus strict-lexicographic and thresholded comparison rules. Transcribes `v51_reward` faithfully | standard library |
| `g3_battery.py` | ten behavioural orderings over every candidate under all three rules, with the critical threshold per channel | standard library |
| `g4_hacking.py` | six reward-hacking probes and the summary table at both discounts | standard library |
| `g5_w5.py` | derives the admissible window for the proposed `w₅` from two physical bounds and sweeps its consequences | standard library |
| `g6_ratchet.py` | **the negative-clip ratchet**, executed against the real `RoutePolyline` | the repository: `PYTHONPATH=src uv run --no-sync python docs/audits/rulebook_architecture_2026-09-09/g6_ratchet.py` |

## The results that hold independently of the recommendation

**The shipped discount is evaluated at the wrong horizon.** ADR-081 requires
`Δ = ln(a)/−ln(γ) > L` and evaluated it at `L = 199`, the Waymo-only figure of
`RULEBOOK-V5.1` §4.6. The training mixture is half procedurally generated and the
longest training **episode** is **500** control steps: the frozen index's `length`
is the scenario's `SD.LENGTH`, the runtime reads the same field
(`third_party/metadrive/metadrive/manager/scenario_data_manager.py:106`), and the
episode truncates at `scenario_length − 1` with `horizon: null` and
`extra_steps_after_scenario: 0`. At `a = 2.5, γ = 0.996` the break-even is
**228.6** steps and **596 of 2200 training records (27.09 %)** exceed it.
`tests/test_hydra_agent_presets.py:257` hardcodes the same 199 under a docstring
reading "The horizon is measured, not assumed". Recorded as an open item.

**The criterion is an identity worth stating.** `Δ > L` is exactly
`γ^L ≥ 1/a`: whole-episode damping must not fall below the one-level priority
ratio. That puts ADR-081's two reasons on one axis — the contraction argument
wants `γ^L` small, the hierarchy argument wants it `≥ 1/a` — and makes the
criterion the most damping the first can have without the second failing.

**The calibration is discount-free.** `episode_return += scalarized.reward`
(`scripts/measure_expert_rulebook_transition.py:2038`) is an undiscounted sum, and
§5.4's rank-preservation inequality is per-step, so no discount change requires
the 45-minute grid to be re-run.

**Three impossibility results bound what any hierarchy can deliver.** `I1`: no
Markov, bounded, per-step progress channel has a duration-invariant discounted
return, so ordering O3 stated as a dominance is unobtainable at `γ < 1` by any
hierarchy in any order. `I3`: O3 and promptness place opposite requirements on a
threshold over the progress channel, at the same critical value, so progress
cannot be a thresholded channel. `I5`: O2 and O3 together require a *finite*
exchange rate, which a threshold cannot supply — so on that pair a scalarization
is structurally better than any ordered arm. Full statements in
`BEHAVIOURAL-SPEC-DRAFT.md` §3.

**Two quantified asymmetries with opposite signs.** Beside `I5`, the mirror
image: an at-fault collision terminates the episode, so a scalar sum prefers
colliding to enduring — break-even at **2.2–2.5** steps of fully-violated
interaction, or 5.3–6.2 steps of non-negotiable violation, against one at-fault
collision at impact 0.6 — while an ordered arm compares the collision channel
first and the non-colliding trajectory wins at any `τ₁ < 0.5788`. `I4` says no
bounded scalar sum can do better, and ADR-081 already settled that claim by
restating it per-step.

**The negative clip is an unbounded ratchet** (`g6_ratchet.py`). Executed: a
closed loop over a hairpin whose legs are one lane apart pays **+36 channel units
= +72 reward units per lap at zero net displacement**, linear in the number of
laps, because the −84.03 m return jump is charged −1. Recorded as an open item.

**Two questions closed without work** (`REVIEW.md` §8.4, §8.5). Collide-to-escape
is an artefact of the scalar arm alone — the ordered arms compare the collision
channel first and the non-colliding trajectory has `K1 = 0` exactly — so it is a
result to report rather than a defect to fix. And a terminal completion bonus
should not be added: the minimum value that would make the last stretch worth a
risky manoeuvre is the value that puts arrival before safety.

**The telescoping error is an under-payment, not an exploit.** `behind_peak`
reports `max_m = 0.053` against a clip threshold of 2.2222 m, so the negative clip
never binds on the expert panel and all of `telescoping_max_error = 62.294` is
deficit from the forward clip over 1298 steps (0.60 %). The statistic is
published as `abs(...)`, so it cannot distinguish a surplus from a deficit —
which is what would detect the ratchet. Recorded as an open item.

## Cross-validation

`g3` reproduces `TEST-RB5.1-16c`'s O3 margin of **−3.5789** at `γ = 0.996` and
**+0.2000** at `γ = 1`. `g5` reproduces the `+3.27` margin the previous session
measured for an L5 indicator at `w₅ = 0.25, λ₆ = 0.2`. `g1` reproduces the
27.09 % from the frozen index directly.

## What this review did not verify

Listed in `REVIEW.md` §7, with the corrections it made to earlier analyses —
including three of its own, found by an adversarial verification pass, and the
misattribution of the `τ_i/(1−γ)` slack bound, which is Wray, Zilberstein &
Mouaddib (AAAI 2015) Proposition 1 and is not proved by Li & Czarnecki.
