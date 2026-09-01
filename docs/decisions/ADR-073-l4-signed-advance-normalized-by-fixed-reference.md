# ADR-073: L4 is the signed route advance, normalized by a fixed reference

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14;
  **amended 2026-08-14** to replace the monotone advance with the signed one
- Date: 2026-08-14
- Approval evidence: explicit user approval of `RULEBOOK-V5.1` on 2026-08-14
  ("approvo la v5.1"), which carries this decision. The amendment below carries
  its own explicit approval, same date, after the user observed that a signed
  advance is invariant to repeated traversal without needing memory.
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §4. Supersedes v4.7's R4 progress definition, which `RULEBOOK-V5.0` left
  authoritative and `RULEBOOK-V5.0` §5.7 nevertheless criticized.
- Related: ADR-072 (which creates L5 and therefore the need for L4 to be
  well-behaved), ADR-068 (`speed_limit`, which is what constrains speed once
  progress stops doing so), ADR-059 (progress ratio as reported metric).

## Context

v4.7's R4 margin is `clip(Δs / (22.22 · Δt), −1, 1)` — described in v5.0 §5.7 as
a speed ratio that "directly incentivizes driving at up to 80 km/h, with nothing
opposing it". Two defects, one of which turned out not to be a defect:

- **Real:** it is not monotone. Reversing earns negative margin and re-advancing
  earns it back, so the per-step credit assignment rewards and punishes the same
  metre twice.
- **Not real:** the "incentivizes speed" reading. Summed over an episode at fixed
  `Δt`, `Σ Δs/(v_ref·Δt) = (s_T − s_0)/(v_ref·Δt)` — the total depends on
  **distance covered**, not on speed. The degeneracy v5.0 complained about is
  real only through the episode horizon, and it is closed by `speed_limit` (ADR-068)
  now sitting in L3, above progress.

ADR-072 places relaxable lane rules below progress. That makes L4's definition
load-bearing in a way it was not before: if two policies that both complete the
mission never tie at L4, L5 is never consulted and the restructure buys nothing.

## Decision

```
Δq_t  = clip( (s_{t+1} − s_t) / D_REF , −1, +1 ),   D_REF = v_ref · Δt = 2.2222 m
```

with `v_ref = MISSION_PROGRESS_REFERENCE_SPEED_MPS = 22.2222 m/s`. `Δq_t = 1`
reads as "advanced one reference-speed step". `s_t` is what R4 reads today; no
new perception, map datum or observation.

**Route completion leaves the reward** and is reported as an episodic metric,
which is where the evaluation protocol wants it.

### `D_REF` is a unit, not a calibration

The episode ceiling works out to `a · (distance covered) / (longest single
step)`, in which `D_REF` cancels exactly. Changing it only rescales `λ₄`
inversely, so it carries no free parameter. It is fixed at `v_ref · Δt` so `λ₄`
is directly comparable with the priority weights `a³, a², a`.

### The clip is the vehicle's physical bound, not a design choice

MetaDrive fixes `max_speed_km_h = 80` for every vehicle type
(`third_party/metadrive/metadrive/component/pg_space.py:233`) and enforces it by
cutting engine force above that speed (`base_vehicle.py:499`), with no override
in this repository. 80 km/h **is** `v_ref`. The agent therefore cannot produce a
step above `D_REF` and the clip never binds on any policy this reward trains. It
exists so the rank-preservation condition has a declared bound.

### Why signed rather than monotone — the amendment

This decision first specified the **monotone** advance,
`Δq = clip((σ_{t+1} − σ_t)/D_REF, 0, 1)` with `σ` the running maximum of `s`, on
the argument that the running maximum is what prevents oscillation being a
progress strategy: re-covering credited ground earns nothing.

That argument does not survive scrutiny, for three independent reasons.

**1. The signed form has the same invariance, without memory.** Undiscounted,
`Σ Δq = (s_T − s_0)/D_REF`. Covering a stretch forward and back contributes
`+x − x = 0`, exactly as not covering it: the return is invariant to how many
times a stretch is traversed. The monotone form obtains that invariance by
*memory*, the signed form by *compensation*, and only the first carries hidden
state.

**2. The exploit the memory guarded against cannot be executed.** MetaDrive sets
`enable_reverse=False` by default (`metadrive/envs/base_env.py:115`) and applies
negative engine force only `if self.enable_reverse`
(`base_vehicle.py:504`), with no override in this repository. **The agent cannot
reverse under power at all.** Negative `Δs` can only come from lateral drift
moving the projection, or from being pushed — events already priced by L1 and L2.

**3. The monotone form silently rectifies projection noise.** Measured over
217,189 expert steps, `σ − s` is non-zero on **20.99 %** of them and in 702 of
1100 episodes, but never exceeds **0.053 m** and never once exceeds 1 m. That is
not reversing; it is the arc-length projection jittering by centimetres as the
ego moves laterally within its lane. The running maximum absorbs each upward
excursion and never returns it, introducing a small but **directional** bias. The
signed form compensates it, because the noise is zero-mean.

Two further properties the amendment gains:

- it credits the **final position** rather than the peak, so an episode ending
  behind its high-water mark scores the distance actually completed;
- it **penalizes reverse motion**, restoring the property `RULEBOOK-V5.0` §5.6
  relied on when it deleted `wrongway` — that reverse route advance is covered
  "by R4, whose margin is negative for negative route advance". A non-negative L4
  had quietly removed the half of that justification, and this restores it.

**Measured consequence: none.** Re-run on the same 1100 records, the expert mean
moves from **+73.85 to +73.84** at `λ₄ = 2.0, η = 1`, with p1, p5, p50 and the
below-standstill fraction identical. The two forms are empirically
indistinguishable, so the choice rests on the properties above.

**`REQ-RB5.1-OBS-01` is withdrawn, not discharged.** It asked whether `σ` is
reconstructible from the observation window. With no `σ`, there is no hidden
state to observe: L4 depends only on the current step's advance, which the
observation already supports through speed and heading. The requirement had no
subject left.

### What the definition does not buy

Ordering O3, contrary to an earlier draft. Two policies covering the same route
arc length tie at L4 under either form, because the undiscounted total is
distance. O3 holds in v5.0 as well.

## Two alternatives, both measured and both rejected

Kept on the record because both are intuitively attractive and neither failure is
predictable from the definitions.

**1. Normalize by each record's own route length** (`Δq = Δs / L_route`), so an
episode's progress return is the completion fraction in `[0, 1]`. Rejected:

- Every mission is then worth at most `λ₄` **regardless of length**, while the
  penalty channels stay per-step and grow with episode length.
- The rank-preservation bound is set by the **shortest route in the panel** —
  measured **19.96 m** against a 171.98 m mean. On a route that short one step
  buys a large fraction of the mission, so one degenerate record dictates `λ₄`
  for all 1100. Measured ceiling **32.7**, below v5.0's own 40.3, and the expert
  measured **−8.49** at `λ₄ = 1`: the mission was worth less than the violations
  incurred completing it.

**2. Cap the advance at the reference speed while still dividing by `L_route`.**
The argument was sound — progress made by speeding is already priced by L3 and
must not be paid twice — but it addressed the wrong cause. Measured: max `Δq` was
**0.033371 with and without it**, identical to six figures, because that maximum
came from a short route (`2.2222 / 0.033371 = 66.6 m`), not from a fast step. It
also broke the telescoping identity on 1298 steps. No benefit, a certain cost.

**A third correction, not an alternative:** the panel's ego starts **31.6 %**
into its own assigned route, because the lane sequence includes the whole first
lane while the ego enters part-way through it. That is a real defect worth fixing
for the *reported* metric. It changes the reward ceiling by **nothing**, because
`L_route` cancels there.

## Consequences

- Episode returns are no longer comparable with the evaluation protocol's
  `route_completion`. That comparability moves to the reported metric, which is
  the standard separation between a training signal and a score.
- A longer mission earns proportionally more, which is intended: a longer mission
  plausibly needs more relaxation, and it faces more per-step penalty exposure.
- The converse: a short mission has a small progress budget against per-step
  penalties of unchanged size, so short scenarios are intrinsically harsher. The
  panel's shortest route is 19.96 m. **No record was excluded** to improve any
  figure in this decision, and excluding them was explicitly declined.
- The Test A figures are **conservative**. The logged Waymo expert is a real car
  under no 80 km/h cap: its step advance reaches 3.609 m (130 km/h), and the clip
  binds on 1298 of 217,189 steps (0.6 %). Those are the expert being charged for
  speed the policy could never attain, so the measured expert return is a lower
  bound, and the telescoping identity is exact for every agent trajectory.

## Risks

1. **The discount is not free** (`REQ-RB5.1-GAMMA`). `Σ Δq = (s_T − s_0)/D_REF`
   holds only undiscounted. Under `γ = 0.99` at 10 Hz over ~200 steps, early
   progress weighs substantially more, so a faster illegal shortcut beats a slower
   legal route *at L4* and L5 is never reached — for the lexicographic arms. For
   the scalar arm it is immaterial. The resolution must be recorded in the
   algorithm specification before any lexicographic arm is trained.
2. **`v_ref` now carries two unrelated roles.** It is the progress normalizer here
   and the reference speed v5.0 criticized as a desired-speed degeneracy. The
   coincidence with MetaDrive's `max_speed_km_h` is what makes the clip inert, and
   that coincidence is a property of the simulator configuration, not a law. If
   `max_speed_km_h` were ever raised, the clip would start binding on agent
   trajectories and this decision would need re-deriving.
