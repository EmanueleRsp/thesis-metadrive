# ADR-083: progress last and unthresholded, and the discount evaluated at the measured horizon

- Status: **Proposed**. The **architecture** and **`γ = 0.9982`** were approved by
  the user on **2026-09-09**, and every remaining gate of the `A7` ExecPlan §6 —
  `w₅ = 0.15`, `φ = 0`, the budget rule, the two mandatory-test changes, the
  specification form, the `AB-LEARN` amendment, the `RB51` disposition and the
  vocabulary break — on **2026-09-10**. **This document is new text and is not
  itself approved yet**, and neither is the specification it records
  (`RULEBOOK-V5.2`, `UNDER_REVIEW`). No production code may be written against
  either until both are approved.
- Date: 2026-09-11
- Approval evidence: the two dates above, recorded in the `A7` ExecPlan §6 and
  §11. Approval of *this* record and of `RULEBOOK-V5.2` is pending and is what
  moves the status line above to `Approved`.
- ExecPlan: `docs/implementation/rulebook_a7_five_channel_hierarchy_exec_plan.md`
  (`A7`), milestone `M2`. Unlike ADR-081, this decision **does** have an
  ExecPlan: it changes a specification contract, an experimental design and a
  public interface, and it spans nine milestones.
- Affected specifications: **`docs/specifications/rulebook_v5.2_UNDER_REVIEW_specification.md`**
  is the document this decision produces. It supersedes, on approval,
  `rulebook_v5.1_specification.md` §3, §4, §5, §9 and §10, and `SCAL-V1.4` in
  full. `reward_learnability_ab_screening_exec_plan.md` `REQ-AB-009` and
  `automatic_curriculum_learning_v2.0_specification.md` §3.2 are re-based by it.
- Affected configuration (on implementation, not now): the scalarization defaults
  — mode, vector schema id and weight set — and `gamma` plus
  `learning_potential_gamma` in the six algorithm configurations.
- Related: **amends ADR-075 and ADR-081** (the discount); **supersedes ADR-076**
  (`L6 progress_rate`, deleted); **partly reverts ADR-072** (the negotiable lane
  rules return above progress); supersedes ADR-069 and ADR-074 for the scalar
  *form* only; ADR-073 (signed progress) carries over unchanged as `K5`;
  ADR-063…ADR-071 (the sub-rules and the at-fault gate) are **untouched**.

## Context

`RULEBOOK-V5.1` put the three lane rules a competent driver may relax below
mission progress and left a sixth level below them to carry a time preference.
That hierarchy is implemented, measured and in production. Three of its
properties are defects of the *design*, each measured rather than argued, and no
two of them can be fixed independently.

**1. The shipped discount does not satisfy its own criterion.** The criterion
that keeps a future higher-priority violation from being damped below a present
lower-priority one is `ln(a)/−ln(γ) > L`. At `a = 2.5, γ = 0.996` the break-even
is **228.6** control steps, and **591 of 2200 training records (26.86 %)** run
longer. The criterion had been checked against `L = 199`, a Waymo-only figure;
the measured training horizon is **500** — the frozen index's `length` is the
scenario's own length, the runtime reads the same field, and the episode
truncates at `scenario_length − 1` with no horizon override. So inside a quarter
of the training population the hierarchy inverts, and the guard that was supposed
to prevent it asserted a horizon nobody had measured.

**2. Ordering O3 is lost at that discount, and no weight buys it back.** A
trajectory taking the legal route loses to an illegal shortcut on the scalar arm
by **−3.5789** where the undiscounted margin is **+0.2000**. The mechanism is not
a mis-set weight: the shortcut arrives sooner, so its discounted progress total
is strictly larger, and the comparison is decided **at progress** before the lane
channel is consulted. The window in which the lane weight could buy O3 back is
**0.193 %** of its own admissible range, which would be fitting a weight to a
single constructed fixture.

**3. Two of the six weights are pinned by nothing.** Across the lane weight's
whole admissible range `[0, 5]` the logged expert's mean episode return moves by
**less than 0.1 on ≈70**, because the logged human almost never relaxes a lane
rule — so no measurement discriminates it — and the sixth level's weight had an
open decision against its name. A parameter that neither a measurement nor an
ordering determines is a parameter that should not exist.

The three have one common cause. **Every road rule outranking progress is what
produces the standing-still attractor; every road rule ranking below it is what
makes an ordering depend on a discounted quantity that duration changes.** The
placement is the free variable, and it had never been examined for the negotiable
rules specifically.

## Decision

**Five channels, strictly ordered, with progress last and unthresholded.**

```
K1  collision safety            at-fault impact                         τ₁ = 0
K2  interaction risk            ttc, clearance, rss_lateral             τ₂ from the panel
K3  non-negotiable compliance   offroad, signal, stop, crosswalk,       τ₃ from the panel
                                vehicle_yield, speed_limit              — and it cannot be 0
K4  negotiable lane compliance  solid_line, wrong_carriageway,          τ₄ from the panel
                                dashed_line
K5  mission progress            signed route advance                    UNTHRESHOLDED, last
```

Four decisions, and they are one decision because none of them survives alone.

1. **Progress is last and unthresholded.** Mechanism from the literature: under
   absolute thresholding the last objective is the unconstrained one (Vamplew et
   al. §3.2.3, p. 58 — "objective n will be unconstrained, hence `C_n = +∞`"), and
   Censi et al. place own progress at the bottom of their rulebook (Fig. 10).
   Mechanism from this hierarchy's own algebra, and it is stronger than a
   convention: **a threshold on progress has an empty requirement.** On the
   reference pair the two properties such a threshold would have to deliver
   require `τ ≤ 27.97` and `τ > 27.97`, at the same critical value, because at
   `γ < 1` the two comparisons *are* the same comparison at that channel.
2. **The negotiable lane rules return above progress, charged by a bounded
   satisfaction indicator with a severity slope at weight `w₅ = 0.15`.** This is
   what makes O3 a property of the **order** — which discounting cannot move —
   instead of a property of an exact undiscounted tie, which §11.9 of the new
   specification shows is unobtainable for any bounded per-step progress channel.
3. **The sixth level is deleted**, with its weight and with the `Δt/T_REF`
   per-step scaling, so the reward stops carrying two different per-step
   normalizations.
4. **One shared discount `γ = 0.9982`** on every algorithm configuration, with
   the shaping discount equal to it, and **the criterion's verdict asserted beside
   the value**.

Weights: **`a = 2.5`, `σ = 0.30`, `λ₄ = 2.0`, `w₅ = 0.15`**; `η`, `λ₆` and `φ`
cease to exist. Six free weights become four.

## Why the reversal of ADR-072 is not a return to v5.0

This is the objection to answer first, because v5.0 had the lane rules above
progress and its own limitation 8 recorded the resulting pathology: standing
still beat completing the mission.

**v5.0's pathology was the price, not the placement.** v5.0 charged a relaxable
violation at the priority weight `a = 2.2` per violated step. A7 charges it at
`w₅·(1+σ) = 0.195`, i.e. **14.7× less** as a bare weight and 11.3× less including
the severity slope. The consequence is a crossover, and it is the whole argument:

```
standing still wins past   N* = λ₄·Q / (D_REF · w₅·(1+σ))   relaxed steps at full severity
```

with `Q` the mission span in metres. At a mean Waymo span of ≈90 m that is **416**
steps, and at a mean procedural-generation span of ≈184 m it is **848** — against
a 500-step episode, so between two and four times a whole episode. Under v5.0's
price the same formula gives **37** steps against a mean Waymo mission, which is
why standing still won there. Re-derived independently for this record.

So the reversal is admissible for a reason that is quantitative and checkable in
one line, and it does not depend on any judgement about placement being "more
natural".

## Why the indicator and the position are one decision

Charging `K4` by a bounded satisfaction indicator is not decoration, and neither
is its position; each is useless without the other.

- **With the indicator but below progress**, an ordered arm never reaches the
  channel on any pair that differs in progress — which is every pair of learned
  trajectories, since two of them have equal progress with probability zero.
- **Above progress but charged continuously**, the scalar arm's exchange rate
  against progress collapses as the violation becomes small: strict dominance
  would require `w₅·(1+σ) > λ₄·Δq` for every `Δq > 0` including `Δq → 0⁺`, which
  no finite weight delivers.

The scalar arm additionally needs a **finite** exchange rate, and no ordering
supplies one — which is the single respect in which the scalarization expresses
something the ordered arms cannot, and is worth recording because the rest of this
work is about the opposite asymmetry.

Two further consequences of the indicator, both measured. A satisfied `K4`
contributes **exactly zero**, so an in-corridor trajectory wins at that channel
before progress is compared. And the per-step price of a fully violated `K4` is
**0.195** against **8.125** for a fully violated `K2` step, so the negotiable
channel cannot outrank a safety channel by accumulation — which is exactly how
the two candidate architectures that charged it as a full priority level were
falsified (below).

## `w₅ = 0.15`: criterion first, then the value, then the cost

`w₅` is an exchange rate between negotiable lane exposure and arrival time, so
both bounds are stated in the same physical currency.

- **Lower bound**, the reference shortcut must not pay: `w₅ > 0.1313` at
  `γ = 0.996`, `w₅ > 0.0736` at `γ = 0.9982`.
- **Upper bound**, crossing a marking must stay cheaper than the manoeuvre it
  replaces: stopping from `v = 10 m/s` and returning to it at a comfortable
  `a_c = 2 m/s²` costs `v/a_c = 5.0 s` of delay, so one second of marking contact
  must cost less — `w₅ < 0.4477` at `γ = 0.996`, `w₅ < 0.2641` at `γ = 0.9982`.
- **Independent cap** from rank preservation: `w₅ < (a − λ₄)/(1+σ) = 0.384615`.

**The binding bound is the lower one at the superseded discount, deliberately**,
so that the architecture stands whether or not the discount decision is taken:
`0.1313` rounded up to **0.15** for margin, because the reference shortcut is a
stipulated construction rather than a measurement. Both windows contain it with
room — 65.9 % and 72.1 % of their own effective upper bounds.

**The cost, in the same breath.** A learner reads `w₅ = 0.15` as: cross a marking
only if doing so buys about **2.2 m/s** of extra route advance while you are on
it (`Δq = w₅(1+σ)/λ₄ = 0.0975`, i.e. 2.17 m/s). And **no admissible `w₅` opposes
a fast off-corridor drive**: the crossover at clip pace is 1.538462 against a cap
of 0.384615, a ratio of `λ₄/(a − λ₄) = ` **exactly 4** in which `σ` cancels, so no
re-tuning of the severity reaches it. That denominator *is* the rank-preservation
tail, so "no admissible `w₅` opposes a fast off-route drive by a factor of four"
is the same statement as "progress consumes four fifths of the per-step budget".
Declared as a limitation rather than fixed.

## `φ = 0`, and why removing a constant is the conservative move here

`φ` is a shared absolute tie-breaker of 0.25 added to every priority margin. It
is removed, on four grounds that compound.

Its stated job is already done: it breaks the tie the satisfaction indicator
creates between two vectors with the same satisfaction pattern, and at `σ = 0` —
where it was set — that tie was real, while at `σ = 0.30` the slope `σ·a^(4−k)`
is non-zero at every level. Its **shape** is the one ADR-081 criticised: being
absolute, its grading is inversely proportional to importance — 5.1 % of the
severity slope at `k = 1`, 25.0 % at `k = 3`, so the most important level is the
flattest. **No document derives its value**: 0.25 traces to Veer et al.'s
averaged-robustness tie-breaker `1/N` at `N = 4`, for their four-level schema,
here summed over three margins — so the only argument for it in this repository
is a specification of this repository, which by `AGENTS.md`'s Scientific Argument
Standards is a **finding** and points at the constant rather than at the code.
And removing it **returns** A7's only structural cost on the rank-preservation
axis: the thinnest margin goes 1.0975 → 1.1390, above the shipped
architecture's own 1.1121.

**The cost:** `a_req^max` falls from **12.43** to **10.96 m/s²**, i.e. from 1.38×
to 1.22× the ≈9 m/s² a real vehicle can produce. The criterion is pinned in its
**strict** form, `a_req^max > 9`, and that is a decision rather than a
convenience: 9 m/s² is already peak braking on dry asphalt, so margin above it
protects against nothing physical, only against uncertainty in the number itself.
If margin is later wanted back the lever is `σ`, which scales with each level's
own weight, and not `φ`, which does not.

**Falsified before recommending, and measured after.** The pre-registered
falsifier was one grid member: revert to 0.25 if `fraction_below_standstill`
breaches the 7.45 % ceiling. Measured on the full 1100-record panel under the A7
reward: **4.64 % at `φ = 0` against 4.73 % at `φ = 0.25`** — a movement of
+0.09 pp against 2.81 pp of remaining headroom. Not falsified.

## `γ = 0.9982`: the criterion is an identity, so the value is forced

`ln(a)/−ln(γ) > L` and `γ^L ≥ 1/a` are the same statement. So at the measured
`L = 500` and `a = 2.5`:

```
γ  ≥  exp(−ln a / 500)  =  0.998169
```

and `0.9982` is the first four-decimal value satisfying it — **the most damping
the contraction argument can have without the hierarchy argument failing.**

| quantity | `γ = 0.996` | **`γ = 0.9982`** |
|---|---:|---:|
| break-even against `L = 500` | 228.6 — fails | **508.6** — holds by 8.6 steps |
| `γ^L` against `1/a = 0.4` | 0.1348 | **0.4062** |
| effective horizon `1/(1−γ)` | 250 steps | **556 steps** |

**The cost, in one sentence:** the effective horizon doubles and 40.6 % rather
than 13.5 % of a spuriously bootstrapped constant survives to the end of the
longest episode, weakening ADR-081's own contraction argument by a factor of
about 2.2 — and it costs nothing in calibration, weights or measurement, because
the rank-preservation condition is per-step and `γ`-free and the panel
measurement is an undiscounted sum.

**The 8.6-step margin is a property of the current frozen index, not of the
code**, which is why the decision includes asserting the **verdict**
(`break_even_steps > horizon_steps`, with the horizon read from the committed
index) and not merely the value. One 510-step scenario in a regenerated index
would put the criterion back in deficit, silently, which is exactly what happened
to the superseded criterion: it *passed* while naming a horizon nobody had
measured.

`learning_potential_gamma` equals `γ` because potential-based shaping is
policy-invariant only when its discount is the MDP's (Ng, Harada & Russell).

## The budgets, and the one thing this decision refuses to decide

`τ₁`–`τ₄` are fixed by the logged expert's **per-episode** exposure distribution
on the frozen Waymo `train` panel, by a rule fixed before the numbers were seen:
**the maximum**, minus any episode attributable to a **declared** panel defect,
with exclusions listed per record, and `p95`/`p99` beside it as sensitivity.

The maximum rather than a quantile follows from the rule's own first requirement —
a budget must admit the logged competent driver, and a quantile excludes human
episodes while asserting that excluding one falsifies the budget. Measured
2026-09-10 on 1100 records and 217,189 transitions, with **no exclusion
declared**: `τ₁ = 0.000000`, `τ₂ = 110.379799`, `τ₃ = 21.287443`,
`τ₄ = 23.386514`. `τ₁ = 0` is **admissible as measured** — the expert records no
at-fault impact anywhere on the panel — and it is what makes the ordered arms
prefer 200 steps of interaction violation to colliding at fault.

**What this decision refuses.** Three candidate *objects* exist for a budget —
undiscounted realized, per-span, and discounted expected — and they do not
coincide; a realized budget would need the accumulated exposure in the policy's
input, which is an observation amendment. This decision **declares the
requirement on the undiscounted realized form** and records all three measured
values, and leaves **which object a threshold is enforced on** to `D1`. Choosing
a number's units here would decide a mechanism by stealth.

## Alternatives considered and rejected

Six architectures were scored against the same ten-ordering battery, and two of
the rejections are falsifications rather than preferences.

| candidate | result | why not |
|---|---|---|
| **A0**, the status quo | fails O3 on the scalar arm at the shipped discount; fails O1, O2 and O3 under strict lex | It is the baseline, and its discount does not satisfy its own criterion |
| **A1** progress last, negotiable as a full priority level at weight 1 | **falsified**: fails O2 (−5.07) and O4 (−13.08) | Accumulated relaxation outranks a collision — 20 steps of full-severity relaxation cost 26 against `a³ = 15.6`. **Any candidate that buys "negotiable compliance above progress" as scalar dominance dies here**, which is what forces the sub-unit weight |
| **A2b** progress last, the two compliance channels merged | **falsified**: fails O2, O4 (−23.08) and O5 (−0.15) | Same mechanism |
| **A2a** progress last, collision merged with interaction — four channels | survives | Halves the mitigation gradient (margin 2.82 → 1.22) by merging an **outcome** with an anticipatory **indicator**, and there is a floor on simplification: with fewer than three constrained channels of genuinely different priority the lexicographic arm has nothing to compare against the scalarization and the experiment loses its object. A7 has four constrained channels; A2a has three and pays for the third with that merge |
| **A6** six levels, negotiable by indicator | passes all ten orderings on the scalar arm, fails O3 under strict lex | **No simplification** — six channels, six weights. It is the candidate that isolates the two mechanisms: the *indicator* fixes the scalar arm, the *position* fixes the ordered arms |
| **A1c** progress last, negotiable in the continuous tail | survives, weak scalar arm | Fails O3 on the scalar arm, which is the ordering the whole restructure is about |
| a pure constrained MDP, no ordering | — | It is what any thresholded architecture *becomes* in the feasible regime; adopting it discards graceful degradation when budgets cannot all be met, for no simplification of the reward |
| a two-tier architecture | — | With one constrained channel there is no *order* among constraints, so the comparison this work exists to make has no object |
| an in-place amendment of `RULEBOOK-V5.1` instead of a new version | rejected, **measured** | v5.1 is cited by 85 files with 112 section citations across 52 of them, and **63 of those 112 (56 %) point at sections A7 rewrites**. An in-place amendment leaves all 63 silently meaning something else; a new version invalidates none, because this repository never re-points citations and `SUPERSEDED` is a first-class status in its own index |
| a terminal completion bonus, as compensation for deleting the sixth level | rejected | The minimum value that makes the last stretch worth one fully violated interaction step is `B ≥ 9.7`, and in the ordered arms `B` lives entirely inside `K5`, below every safety channel — so it can cause no regression there and fixes nothing either |

**One cost of the thresholded arm is this decision's to state, because this
decision reduces it.** Every constrained channel removed is one fewer constraint
in a set the literature does not know how to satisfy jointly: the
state-augmentation route is proved for one constrained channel and its authors
state that extending it to several is "not straightforward … we need to know
which constraints can be satisfied together", and finding an optimal
deterministic policy for a lexicographic MDP is NP-hard (Pineda, Wray &
Zilberstein, Lemma 1). **A0 has five constrained channels against A7's four, and
A0's fifth is progress — the one a threshold provably cannot be placed on.**

## Consequences

**Intentional breaks.**

- **Checkpoint compatibility breaks**, by three independent routes: the margin
  vector changes arity from 6 to 5, the vector schema id changes, and the weight
  set changes. All three reach the checkpoint reward-semantics identity.
- **The recorded channel vocabulary breaks.** The deleted level's name disappears
  and the order changes. This is *not* the same class of break as the 2026-08-20
  rename: that one moved every channel name, so a stale consumer failed visibly,
  whereas A7 keeps four of five names and a stale consumer produces a
  **populated and wrong** table. One guard is therefore added — the vector schema
  id joins the analysis condition identity — so that a pre-A7 and a post-A7 run
  are impossible to pool rather than merely unlikely to be.

**Two dependent contracts are re-based, and neither is rewritten here.**

- The A/B screening's pre-registered arm B is the reward this decision replaces.
  It is amended **explicitly and before any screening run**, because that plan's
  own freeze gate "is legitimate only because it was pre-registered". Moving it
  now costs nothing scientifically — both seed-0 runs died, neither has the
  checkpoint the official evaluation path requires, and the relaunch is unmade —
  and moving it later would cost everything. The joint-freeze decision
  **survives**: approving this specification freezes the *contract*, while the
  screening freezes the *question of optimizability*, and without the second
  freeze restated nothing stops an A8.
- The curriculum contract treats **six** level margins as a property that is
  fatal if absent — true today, false after implementation — so **the ACL as
  written cannot run on an A7 rulebook**, by its own design rather than by an
  oversight. Recorded as a re-base gate; no ACL revision is written here, because
  a revision drafted now would be drafted against a contract that is not yet
  authoritative. One of that plan's ordering decisions is **reopened without
  prejudice**: its stated reason for rejecting the A7-shaped order was a citation
  of ADR-072, which `AGENTS.md` does not admit as evidence.

**What this decision does not fix**, each with its figure, and all declared in
the new specification §11: no credit without motion (**+1.345** against a
**0.000** baseline, made *visible* by deleting the sixth level rather than caused
by it); the negative-clip ratchet (**+72 reward units per lap at zero net
displacement**, and it matters *more* under A7 because in the thresholded regime
progress is the only gradient inside budget); the legal parallel corridor (**318
of 3500 records**, and every remedy shape foreclosed by one approved
specification); collide-to-escape on the scalar arm (**prefers colliding by
1092.8**, where the ordered arms prefer not colliding at any `τ₁ < 0.5788`);
mission success as a zero-value terminal; and `K2 ≻ K3`, whose only surviving
argument is a specification of this repository and whose co-occurrence is
therefore measured and declared rather than resolved — **15 of 217,189 steps
(0.0069 %)** across **5 of 1100 episodes**, with a Pearson correlation of
**−0.0161** on those steps.

**Implementation is blocked** until `RULEBOOK-V5.2` is approved. The trap in the
change is already located and is not a design question: the scalar adapter
hard-codes the progress index as `3`, and under A7 progress is index **4** of
five, so the unmodified range check would treat a negated cost as signed and a
signed advance as a cost and **raise on every step with positive progress**. It
fails loudly rather than silently, which is the good case, and the frozen test
matrix pins both directions.
