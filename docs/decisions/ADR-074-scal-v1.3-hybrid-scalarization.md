# ADR-074: `SCAL-V1.3`, per-step dominance over L1–L3 and a finite L4/L5 exchange

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-14
- Approval evidence: explicit user approval of `RULEBOOK-V5.1` on 2026-08-14
  ("approvo la v5.1"), which carries this decision.
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §5. Supersedes `SCAL-V1.2` (ADR-069), which it contains unchanged for L1–L3.
- Related: ADR-069 (`SCAL-V1.2`), ADR-072 (the five levels), ADR-073 (L4).

## Context

`SCAL-V1.2` scalarizes four channels with Veer et al.'s rank-preserving
construction: a satisfaction indicator per level, geometric priority weights, and
progress as a weighted tail. ADR-072 adds a fifth channel *below* progress, and
that channel cannot be scalarized the same way.

`SCAL`'s per-step dominance works because each level carries a **satisfaction
indicator** `step(m_k)`, which makes the level discrete — satisfied or not.
Progress is continuous. Strict dominance of L4 over L5 would require
`λ₄·Δq > λ₅·c_L5` for **every** `Δq > 0`, including `Δq → 0⁺`. **No finite `λ₄`
satisfies that.**

## Decision

```
r_t  =  Σ_{k=1..3} a^(4−k) · [ (step(m_k) − 1) + σ · m_k ]
        + φ · Σ_{k=1..3} m_k
        + λ₄ · Δq_t
        − λ₅ · c_L5,t · (Δt / T_REF)
```

with `m_k = −c_Lk ∈ [−1, 0]`, `Δq_t ∈ [0, 1]` (ADR-073), `c_L5,t ∈ [0, 1]`,
`T_REF = 1 s`.

**L1–L3 are `SCAL-V1.2` verbatim**, so their per-step dominance is inherited
rather than re-argued. **L4 and L5 form a finite exchange.**

**Selected: `λ₄ = 2.0`, `η = λ₅ = 1.0`**, with `a = 2.2`, `σ = 0`, `φ = 0.25`
inherited.

### Rank-preservation condition

```
a^(4−k)  >  (1 + σ) · Σ_{j>k, j≤3} a^(4−j)  +  φ · (3 − k)
            +  λ₄ · ΔQ_MAX  +  λ₅ · (Δt / T_REF)
```

Same derivation as `SCAL-V1.2`; only the utility tail differs. It reduces to
that document's condition whenever the tail equals its `2λ`. The binding
constraint is `k = 3`, where the lower-level sum is empty:

```
λ₄ + 0.1 · λ₅  <  a = 2.2
```

Selected pair: `2.0 + 0.1 = 2.1 < 2.2`. Inadmissible pairs are never priced, in
the same way `family_grid` refuses non-rank-preserving members.

`λ₄ = 2.0` is preferred to the marginally better `λ₄ = 2.15` (+79.91 against
+73.85) because at 2.15 only `η ≤ 0.5` remains admissible, while at 2.0 the
budget reaches `η = 2`. Six points of mean return buy room for L5, which is the
channel this whole restructure exists to create.

## Two ways to force uniform dominance, both rejected

1. **Indicator on L4** (`1[Δq > 0]`) makes `SCAL` uniform across all five
   levels — and rewards infinitesimal creep. An ego inching forward at 0.01 m/s
   satisfies the indicator, gains dominance over L5, and rides a solid line for
   free. This converts "stop for ever" into "creep for ever along the marking",
   which is a **worse** degeneracy than the one being removed.
2. **Indicator on a progress threshold** (`1[Δq > δ]`) removes the creep but
   introduces an unmotivated calibrated constant and a cliff the policy will sit
   on. That is the class of construction this project has rejected throughout.

The finite exchange is therefore not a concession. It is the only construction
that claims exactly what it can prove.

## The withdrawn claim

`RULEBOOK-V5.0` §11.8 asserted that its scalarization "is the exact lexicographic
order written as a scalar". That contradicted its own §6.5 and is **withdrawn**,
in v5.0 itself as well as here. Veer et al.'s Theorem 1 ranks whole trajectories
and does not establish the episodic property; summing a per-step score does not
preserve an order over trajectories.

**Nothing is lost by this correction.** `SCAL-V1.2` never had episodic
lexicographic order either. What changes is that this decision stops claiming it.
The proved property is retained exactly where it was demonstrated: **strict
per-step dominance over L1–L3**.

## Measured outcome

1100 Waymo `train` records, 217,189 transitions, 0 skipped, 0 errors, varying
only the weights on identical atomic costs:

| rulebook | mean | p1 | p5 | p50 | below standstill |
|---|---:|---:|---:|---:|---:|
| v5.0 specified (`SCAL-V1.2`) | +31.14 | −150.85 | −15.86 | +25.72 | 7.45 % |
| `λ₄ = 1.0` | +33.57 | −91.62 | −6.37 | +26.15 | 6.27 % |
| `λ₄ = 1.5` | +53.71 | −80.84 | +2.44 | +40.61 | 4.64 % |
| **`λ₄ = 2.0`, `η = 1`** | **+73.85** | **−59.67** | **+9.16** | **+55.23** | **3.36 %** |
| `λ₄ = 2.15` | +79.91 | −48.01 | +9.89 | +59.42 | 3.27 % |

Dominates `SCAL-V1.2` on every column. The measured slope is **40.3 per unit of
`λ₄`**, which equals v5.0's own mean R4 episode return — a consistency check that
the two formulations measure the same quantity.

`η` reaches only L5: L1–L4 channel values are bit-identical across `η ∈ [0, 5]`,
asserted by `test_eta_reaches_only_l5`.

## Risks

1. **`η` is the one weight not pinned by measurement.** The condition leaves it
   nearly free, and the expert panel cannot discriminate: across `η ∈ [0, 5]` the
   mean return moves by under 0.1, because the expert almost never relaxes a lane
   rule. `η = 1.0` rests on a two-sided argument — a *gratuitous* relaxation
   unlocks no progress so it loses for any `η > 0`, while a *necessary* one must
   still win — and on admissibility headroom. It must be validated by
   `TEST-RB5.1-06`, not by replay. **Stated rather than hidden.**
2. **The exposure of risk 1 is bounded to one arm of four.** The lexicographic
   and distributional arms consume the atomic cost vector directly and never
   evaluate this expression. An imperfect `η` degrades a baseline, not the
   experiment.
3. **Per-step dominance is not episodic dominance.** Two L3 violations and one L2
   violation of equal magnitude exchange at the ratio `a` over an episode. The
   correct phrasing remains "geometric priority weights, strict per-step
   dominance, exchange rate `a` at episode level".
4. **The satisfaction indicator charges the full weight regardless of severity.**
   A violated L2 step costs `a² = 4.84` however mild it is, so an L2 sub-rule has
   to fire on a small fraction of steps for the rulebook to be affordable at all.
   A trajectory violating L2 on *every* step loses to standing still even under
   this scalarization — correctly, but it means the measured 0.3978 % is a
   constraint being satisfied, not merely a small number. Asserted by
   `test_l2_indicator_makes_continuous_violation_lose_to_standing_still`.
5. **`λ₄` is capped by the hierarchy, not by taste.** `λ₄ < a − 0.1·η` is what
   guarantees no single step of progress overturns a non-relaxable violation.
   Raising `a` to buy more progress weight would steepen every penalty in
   proportion and is not a free lever.
