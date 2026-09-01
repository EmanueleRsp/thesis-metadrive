# ADR-069: `SCAL-V1.2` - rank-preserving per-step scalarization at a = 2.2, severity outside the priority weight

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`. The user
  rejected the argument from precedent explicitly - that the current formula
  should be kept because it was chosen first - and asked for a construction
  justified on the merits.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §6; supersedes `docs/specifications/rulebook_scalarization_v1.1_specification.md`
  §7.6.

## Context

`SCAL-V1.1` (`bounded_priority_weighted_rank`, base 3) charges each channel

```
base^e * ((step(m_k) - 1) + m_k)
```

so a violating R2 step costs between 9 and 18 reward units. At the expert's mean
progress margin of 0.205 per step, one R2 violation costs about 59 steps -
nearly six seconds - of normal driving.

Veer, Leung, Cosner, Chen and Pavone (ICRA 2023), Theorem 1, define the
rank-preserving reward as

```
R(rho) = sum_i ( a^(N-i+1) * step(rho_i) + (1/N) * rho_i ),  a > 2,  rho_i in [-a/2, a/2]
```

with `tanh` normalisation and, in Remark 1, a sigmoid replacing `step` for
differentiability. Two things follow that had not been stated in this
repository. The satisfaction indicator is **part of the theorem**, and rank
preservation fails without it. And severity sits **outside** the priority
weight, as a shared `1/N` tie-breaker; `SCAL-V1.1` §7.6 deliberately moved it
inside, which is what forces base 3.

## Decision

Adopt the family

```
r = sum_{k=1..3} a^(4-k) * [ (step(m_k) - 1) + sigma * m_k ] + phi * sum_{k=1..3} m_k + lambda * m_4
```

with the member **a = 2.2, sigma = 0, phi = 0.25, lambda = 1**, admissible only
if, for every level k,

```
a^(4-k) > (1 + sigma) * sum_{j>k} a^(4-j) + phi * (3 - k) + 2 * lambda
```

## Rationale

**The two per-step adaptations are forced and are now stated.** The step term
must be shifted so a satisfied step scores 0 rather than `+a^e`; unshifted, a
stopped ego collects the whole priority stack every step forever. And progress
cannot remain a `1/N` tie-breaker, because for this work progress is the task,
not a tie-break among equally compliant trajectories.

**The condition reproduces both published anchors without being told them.** At
`sigma = phi = 0, lambda = 1` it reduces to `a > 2`, Veer et al.'s own
condition. At `sigma = 1, lambda = 1` it forces `a >= 2.92`, which is exactly
why `SCAL-V1.1` had to re-derive base 3. `lambda` enters as `2 * lambda`, which
is where the asymmetry between the bounded cost channels and the progress
channel is resolved: the hierarchy itself states how much progress weight it can
tolerate.

**Measured, on the same rulebook and the same per-step costs:**

| member | p1 | p5 | p10 | p50 | mean | below standstill |
|---|---:|---:|---:|---:|---:|---:|
| a=2.01, sigma=0, phi=0 | -128.26 | -7.07 | +5.28 | +26.87 | +32.87 | 6.27 % |
| **a=2.2, sigma=0, phi=0.25** | -155.38 | -13.38 | +5.08 | +26.36 | **+31.50** | **7.18 %** |
| a=3, sigma=0, phi=0 | -276.79 | -39.04 | +4.62 | +26.09 | +26.08 | 8.73 % |
| a=3, sigma=1, phi=0 (`SCAL-V1.1`) | -434.44 | -58.34 | +3.87 | +26.00 | **+19.90** | 9.27 % |
| no indicator (not rank-preserving) | -118.27 | +2.14 | +5.84 | +28.15 | +34.11 | 4.91 % |

Dropping the indicator yields +34.11 but converts the reward into a weighted
sum with no hierarchy guarantee. Staying inside rank preservation captures 82 %
of the achievable gain at a = 2.2 and 91 % at a = 2.01: the theoretical property
costs about one point in fourteen. There was no trade-off, only a badly chosen
base.

**Why a = 2.2 and not the marginally better a = 2.01.** The level-3 constraint
is `a > 2 * lambda`; at lambda = 1 the margin at a = 2.01 is 0.01, so any later
change breaks it, while at a = 2.2 the margin is 0.2. More importantly, a = 2.01
**cannot afford Veer et al.'s own tie-breaker** - `phi = 0.25` at that base is
rejected by the condition - so "a = 2.01, phi = 0" is the published formula
stripped of the term that supplies a gradient toward compliance inside a
violation. `a = 2.2, sigma = 0, phi = 0.25` is the published structure intact
with the single forced adaptation `lambda: 1/N -> 1`, and the flat tie-breaker
costs 0.23 points of mean return.

## Verification

The admissibility condition is implemented as a decision procedure and checked
against an **exhaustive counterexample search** (`TEST-RSEC-019`); the family is
pinned to reproduce `SCAL-V1.1` exactly at `(3, 1, 0, 1)` (`TEST-RSEC-018`). The
predicate is deliberately conservative - it compares against the supremum of the
violating case, which is never attained - so it rejects knife-edge members that
enumeration cannot break.

## Scope

This scalarization produces the scalar baseline. The planned lexicographic and
distributional algorithms consume the rulebook cost vector directly and bypass
it entirely; the choice of `(a, sigma, phi, lambda)` does not constrain them.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
