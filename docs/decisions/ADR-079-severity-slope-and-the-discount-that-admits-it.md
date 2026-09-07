# ADR-079: a severity slope inside the violated set, and the discount that admits it

- Status: **DRAFT — awaiting the calibration measurement and user approval**
- Date: 2026-09-07
- Approval evidence: _pending_. User asked on 2026-09-07 for `σ` to be explained
  and derived rather than asserted ("basta poi emendare e giustificare in modo
  dettagliato come ci siamo arrivati a quel valore (se c'è un calcolo dietro…)").
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §4.4 (the discount), §5.4 and §5.5 (the weights).
- Affected configuration: `conf/scalarization/default.yaml`
  (`priority_base`, `severity`), `conf/agent/planner/algorithm/*.yaml` (`gamma`).
- Related: **amends ADR-075** (`γ = 1`), builds on ADR-073 (L4), ADR-076 (L6).

## Context

Two parameters of `SCAL-V1.4` were never derived from anything.

`σ = 0` is inherited verbatim from `SCAL-V1.2` (§5.5 says so). It is **not** a
measured value: `scripts/measure_expert_rulebook_transition.py` sweeps `σ` only
inside the four-level counterfactual family, where `λ = 1` is a placeholder and
there is no L5/L6 tail, and its six-level grid (`v51_weight_grid`) holds
`FINAL_PRIORITY_BASE` and `FINAL_SEVERITY` fixed while sweeping `λ₄`, `η` and
`λ₆`. So `σ` has never been priced in the hierarchy that is actually in
production.

`a = 2.2` is a **lower** bound, not an optimum. §5.4 derives the binding
constraint at `k = 3`, where the lower-level sum is empty:

```
a  >  λ₄ · ΔQ_MAX  +  (η + λ₆) · (Δt / T_REF)   =   2.0 + 0.12   =   2.12
```

and 2.2 is the first round value above it. Nothing in the document places an
upper bound on `a`.

## What `σ` is, and why zero is a choice with consequences

For a normative level `k` the per-step term is, verbatim from
`reward/scalarization.py`:

```
term_k = w_k · [ 1_sat(m_k) − 1 + σ·m_k ] + φ·m_k        w_k = a^(4−k),  m_k = −cost_k
```

so a satisfied level contributes exactly 0 and a violated one contributes
`−w_k − (w_k·σ + φ)·cost_k`. Three parameters with three distinct jobs:

- `w_k` is the price of **crossing** from compliance to violation, and it is
  what encodes the hierarchy;
- `σ` is how much worse a violation gets as it deepens, as a **fraction of that
  level's own weight**;
- `φ` is a shared **absolute** slope, identical at every level.

With `σ = 0` the only grading left is `φ = 0.25`, and because `φ` is absolute the
grading is inversely proportional to importance — the most important level is the
flattest:

| level | weight | range inside a violation | share of the weight |
|---|---:|---:|---:|
| L1 | 10.648 | 0.25 | 2.3 % |
| L2 | 4.84 | 0.25 | 5.2 % |
| L3 | 2.2 | 0.25 | 11.4 % |

Concretely, for the TTC sub-rule (`cost = 1 − ttc/0.95`, `ttc.py`):

| TTC | L2 contribution |
|---|---:|
| 0.9501 s | 0 |
| 0.9499 s | −4.840 |
| 0.10 s | −5.063 |
| 0.00 s | −5.090 |

The component computes a perfectly good continuous risk and the scalarization
discards all but 5 % of it.

## Why that specifically damages a deterministic-policy-gradient learner

TD3's actor update is `∇_θ J = E[ ∇_a Q(s,a)|_{a=π(s)} · ∇_θ π(s) ]`: the only
thing that moves the policy is the local slope of `Q` in action space. Inside the
violated set that slope carries almost no information about which direction
reduces the violation, so the agent learns that violating costs `w_k` and learns
nothing about how to stop violating. The discrete step of `w_k` only pays if the
agent can cross back over the threshold **within one control step**, which at
0.3–0.5 s of TTC it cannot.

## The derivation

Ask when the reward discourages accelerating into a conflict. Leader at distance
`d`, closing speed `u = v − v_lead`, so `ttc = d/u`; progress is `dq = v/v_ref`
with `v_ref = 22.222 m/s` (`progress.py`), and `τ = 0.95 s`:

```
∂dq/∂v   = 1/v_ref
∂ttc/∂v  = −d/u² = −ttc/u
∂cost/∂v = ttc/(τ·u)

∂r/∂v = λ₄/v_ref − (w₂σ + φ) · ttc/(τ·u)
```

`∂r/∂v < 0` — the reward prefers slowing — exactly when

```
(w₂σ + φ) · v_ref / (λ₄ · τ)  >  u/ttc  =  u²/d  =  2 · a_req
```

where `a_req = u²/2d` is the constant deceleration the conflict demands. So the
criterion has a physical reading:

```
a_req^max  =  (w₂σ + φ) · v_ref / (2 · λ₄ · τ)
```

= the hardest conflict, measured by the braking it demands, in which the reward's
local gradient still points the right way. A vehicle brakes at about 9 m/s².

At the shipped weights (`w₂ = 4.84`, `φ = 0.25`, `λ₄ = 2.0`) the factor is
5.848 m/s² per unit:

| `σ` | `a_req^max` | |
|---|---:|---|
| **0 (shipped)** | **1.46 m/s²** | the gradient points the wrong way in any real conflict |
| 0.05 | 2.88 | |
| 0.1227 | 4.94 | the largest `σ` §5.4 admits at `a = 2.2` |

**This is the result that forces `a` upward.** The admissible `σ` window at
`a = 2.2` contains no value that makes the reward physically sensible, because
§5.4 caps `σ` at 0.1227 there and 0.1227 covers barely half the braking limit.

At `a = 3.0` (`λ₄` unchanged at 2.0) §5.4 re-derives to `σ < 1.0317`, and:

| `σ` | `a_req^max` | |
|---|---:|---|
| 0.143 | 9.0 m/s² | the minimum that covers the physical braking limit |
| 0.30 | 17.3 | ~1.9× the limit |
| 0.50 | 27.8 | ~3× the limit |

Raising `a` also repairs the thinnest margin in the document: `w₃/tail` goes from
`2.2/2.12 = 1.04` — one step of maximal progress repays 96 % of running a red
light — to `3/2.12 = 1.42`.

## The discount follows from the same inequality

ADR-075's break-even table compares a **future collision** (`a³`) against a
**present L3 violation** (`a¹`), a ratio of `a² = 4.84`. The binding comparison
is one level apart, not two — L1 against L2 ("violate the interaction-risk rule
now, or collide in Δ steps?"), and identically L2 against L3 — with ratio `a`.
Dominance survives while

```
Δ  <  ln(a) / (−ln γ)
```

which is **half** ADR-075's figure at every `γ`. Its conclusion that `γ = 0.99`
breaks inside the episode is therefore right for a stronger reason than stated
(78 steps, not 157), and its "safe" mark on `γ = 0.995` does not hold at
`a = 2.2` over the measured 199-step horizon (§4.6: p5/p50/p95 = 197/199/200).

Requiring `Δ > L = 199`:

| | `a = 2.2` | `a = 3.0` |
|---|---|---|
| `γ = 0.995` | Δ = 157 ✗ | Δ = 219 ✓ |
| `γ = 0.997` | Δ = 262 ✓ | Δ = 366 ✓ |
| `γ = 0.999` | Δ = 788 ✓ | Δ = 1097 ✓ |

so `γ ≥ 0.9961` is required at `a = 2.2` and `γ ≥ 0.9945` at `a = 3.0`. The two
decisions are coupled by one inequality.

`γ < 1` is needed independently of the hierarchy, and this is the half ADR-075
did not weigh. With `γ = 1` the Bellman operator is non-expansive rather than
contracting, and it has a unique fixed point only if every trajectory ends in a
true terminal at value zero. Roughly two thirds of episodes end in a
**bootstrapped truncation** at the horizon instead, where the backup `Q ← r + Q(s')`
admits `V + c` for any constant `c`; only the terminating episodes pin `c`, and
they are the minority. Any approximation bias then walks the constant, which is
the mechanism behind the observed critic-loss growth. The damping that reaches
`t = 0` over a 199-step episode is `γ^199`: **0.37** at 0.995, **0.55** at 0.997,
and **0.82** at 0.999 — which is why the fallback ADR-075 declared restores the
contraction on paper and damps almost nothing inside an episode.

C21 removes the aggravating factor separately: the state that bootstrap reads was
an emptied world, because upstream despawns every replayed actor on the step the
horizon used to fire.

## Options considered and rejected

- **Horizon as a true terminal, with remaining time in the observation.** The
  horizon is the length of the Waymo log, not a property of driving; the task
  does not end at 20 s. Terminating there teaches the agent that the world ends,
  and adding a time feature makes the policy non-stationary — it would learn to
  drive differently near the end of a log — besides changing `D` and
  invalidating every encoder and checkpoint. Pardo et al. (ICML 2018) prescribe
  bootstrapping exactly when the time limit is not part of the task; the
  repository is already on the right side of that.
- **`σ = 0.10` with `a = 2.2` unchanged.** Strictly better than the status quo
  (the range inside a violation triples) and it changes one specification value
  instead of two, but `a_req^max` reaches only 4.4 m/s², so the reward still
  points the wrong way in a hard-braking conflict. Kept as the fallback if the
  calibration below rejects `a = 3.0`.
- **Normalizing L2 and L3 by duration, like L5 and L6.** It would make 20 steps
  of L2 cost less than one collision, which is arithmetically the return-level
  inversion this document does not fix — but at the price of telling the agent
  that two seconds at 0.1 s of TTC are cheap. Rejected: it cures the symptom by
  corrupting the semantics.
- **Raising `λ₆` to price standing still.** Inadmissible: `λ₆ < 0.25` is pinned
  by the episodic O3 condition (40 stalled steps must not beat 30 steps riding a
  marking), independently of §5.4.

## What this decision does *not* fix

Per-step dominance does not imply per-return dominance, and no summed scalar
reward of reasonable dynamic range can deliver it: over `L` steps it would need
`w_k > L · w_{k+1}`, i.e. `a ≳ 199`, giving a top weight of ~8·10⁶. The user
decided on 2026-09-07 (option A) to **restate the claim** — the scalarization
implements a per-step, reactive rule hierarchy, and the thesis must not claim
trajectory-level lexicographic optimality. §11.1 already documents the related
standing-still property; this is its companion and belongs beside it.

## The measurement

`v51_weight_grid` holds `a` and `σ` fixed, and the four-level counterfactual
family varies them under a placeholder `λ = 1` with no L5/L6 tail, so neither
could answer this. A third grid, `v51_calibration_grid`, was added to
`scripts/measure_expert_rulebook_transition.py` and prices `(a, σ, λ₄)` jointly
**under the six-level reward**, refusing non-rank-preserving members exactly as
the other two do. Run on the identical scope §5.5 used — 1100 Waymo `train`
records, 217 189 transitions, 0 skipped — into
`outputs/rb51_calibration_a_sigma_l4.json`.

The instrument is validated by its own baseline: the shipped point reproduces
§5.5's row exactly (mean **70.70**, p1 **−61.81**, below standstill **3.36 %**).

| package | `a_req^max` | `w₃`/tail | below standstill | mean | p1 |
|---|---:|---:|---:|---:|---:|
| `a=2.2 σ=0 λ₄=2.0` (shipped) | 1.46 | 1.04 | **3.36 %** | 70.70 | −61.8 |
| **`a=2.5 σ=0.30 λ₄=2.0`** | **12.43** | **1.18** | **4.55 %** | 68.31 | −99.4 |
| `a=3.0 σ=0.30 λ₄=2.8` | 12.32 | 1.03 | 4.55 % | 98.66 | −130.7 |
| `a=3.0 σ=0.15 λ₄=2.0` | 9.36 | 1.42 | 5.73 % | 65.78 | −154.2 |
| `a=3.0 σ=0.30 λ₄=2.0` | 17.25 | 1.42 | 5.82 % | 65.14 | −160.0 |

## Decision

**`a = 2.5`, `σ = 0.30`, `γ = 0.996`.** `λ₄ = 2.0`, `η = 1.0`, `λ₆ = 0.2` and
`φ = 0.25` are unchanged, so §5.5's measured calibration of the utility tail
stands untouched.

It is the only package that improves on the shipped one along **two** axes at
once — the reward's local gradient goes from physically wrong (correct only in
conflicts demanding under 1.5 m/s² of braking) to 1.4× the physical braking
limit, and the thinnest margin in the document goes from 1.04 to 1.18 — for
1.19 pp of below-standstill.

`a = 3.0` was measured and rejected in both directions: at `λ₄ = 2.8` it buys the
below-standstill back but crushes `w₃`/tail to **1.03**, i.e. one step of maximal
progress would repay 97 % of running a red light, which is *worse* than the
status quo; at `λ₄ = 2.0` it keeps the margin but costs 2.4 pp.

`γ = 0.996` follows from `a = 2.5` by the inequality above: `Δ = 229` steps
against a 199-step horizon, a 15 % margin, and 457 steps two levels apart.
`γ = 0.995` would not do — `Δ = 183 < 199`.

**The cost, stated plainly.** The p1 of the expert return goes from −61.8 to
−99.4, so the critic has to fit a dynamic range about 1.6× wider, and 1.19 pp
more of the expert panel now scores below standing still. Both are the price of
a reward whose gradient points the right way inside a violation, and neither was
measurable before this grid existed.

**Not yet confirmed by the user.** The `σ` value was delegated on 2026-09-07
conditional on a derivation ("se dici che il valore che mi indichi ora è valido
va bene, basta poi… giustificare in modo dettagliato come ci siamo arrivati");
raising `a` is inseparable from it but was not itself put to the user, and
`γ = 0.996` differs from the 0.995 discussed in that exchange because the
horizon is 199 steps, not the 91 assumed there. Status stays DRAFT until that is
confirmed.

## Consequences

- Checkpoints written under the previous reward or discount are not comparable
  and not resumable; this compounds the breaks already accepted in `D4`.
- `RULEBOOK-V5.1` §4.4, §5.4 and §5.5 need amending, and ADR-075 needs a
  superseding note pointing here.
- The `a_req^max` criterion is a new derived quantity, not a specification
  contract: it is a lower bound on `σ` from vehicle dynamics, and it should be
  restated if `τ`, `v_ref` or the TTC cost shape change.
