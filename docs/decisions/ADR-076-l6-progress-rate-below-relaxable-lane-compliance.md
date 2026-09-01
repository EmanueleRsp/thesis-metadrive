# ADR-076: `L6 progress_rate`, a sixth level below relaxable lane compliance

- Status: **Approved**
- Date: 2026-08-20
- Approval evidence: explicit user approval on 2026-08-20 ("sì, procedi con L6"),
  after a first proposal placing the same quantity inside L4 was rejected by the
  user on the grounds that it reintroduced the very failure `γ = 1` had removed.
  That objection was correct and is recorded below as the rejected alternative.
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §3 (the hierarchy becomes six levels), §4.6 (the new level) and §5
  (`SCAL-V1.4`).
- Related: ADR-075, which makes this decision necessary and is not severable
  from it; ADR-072 (the five levels this extends); ADR-073 (L4, left untouched);
  ADR-074 (`SCAL-V1.3`, superseded in form by `SCAL-V1.4`).

## Context

ADR-075 sets `γ = 1`. That fixes the hierarchy and makes O3 exact, and it
removes something at the same time: **all time pressure**. Under `γ = 1` the
mission channel telescopes, so two trajectories that reach the same place score
identically at L4 however long they take.

The obvious objection is that reaching the goal is itself the pressure — L4 is
maximal on arrival, so an agent must be fast enough to arrive. **Measured, that
bound is far too loose to serve.** On the 1100 Waymo `train` records:

| | p5 | p50 | p95 |
|---|---:|---:|---:|
| mission span `s_goal − s_start` | 13.0 m | **67.5 m** | 247.2 m |
| horizon | 197 steps | 199 steps | 200 steps |
| **mean speed required to arrive** | 0.65 m/s | **3.40 m/s** | 12.42 m/s |

The agent is capped at 22.22 m/s, so at the median it may travel **6.5× faster
than the minimum that still arrives**. Inside that band L4 is constant and
slower driving reduces exposure to the L2 interaction sub-rules, which all scale
with speed. Without a sixth level the optimum is to crawl at 12 km/h for twenty
seconds — a policy with a full mission score and no defensible behaviour.

The failure mode is documented. CaRL (Jaeger et al., CoRL 2025), on a reward
carrying collision and off-road penalties but no progress term: *«a policy that
always stays stationary would be optimal in reactive traffic ... such a reward is
not reasonable for pure RL»*. Their own remedy is the opposite branch of this
decision — *«Our reward does not encode that getting to the goal faster is
better. This is because most RL algorithms naturally encode a notion of urgency
via a discount factor»* (γ = 0.99), plus a hard `Blocked` infraction. **That
branch is closed to this project**, because ADR-075 measured that discounting
erodes the geometric priority weights and breaks the hierarchy inside a 20 s
episode. CaRL can rely on the discount because their reward is a soft-constrained
sum with no priority ordering to erode.

## Decision

A **sixth level `progress_rate`, below L5**, with a single sub-rule
`advance_shortfall`:

```
c_L6,t  =  1 − clip( Δq_t , 0 , 1 )   ∈ [0, 1]
```

`Δq_t` is ADR-073's advance, unchanged. **L4 carries the bare advance again**:
nothing is subtracted from it.

The sub-rule is named `advance_shortfall` rather than after its level. Naming it
`progress_rate` was implemented first and reverted: the aggregated level result
then shadowed the atomic one in the component map, silently breaking §3.4's
contract that every sub-rule cost stays exposed. Nothing failed, because for a
single-sub-rule level the two values coincide. Level/sub-rule name collisions
are now rejected rather than merged.

Selected weight for the scalar arm: **`λ₆ = 0.2`**.

### Why below L5, and why that is the whole point

An illegal shortcut carries `c_L5 > 0`, so under any ordering it **loses at L5
before L6 is ever consulted**. The weight on L6 is therefore *unconstrained by
O3 in the lexicographic and distributional arms*: how strongly the reward
prefers arriving sooner is decoupled from whether arriving sooner can pay for a
lane violation.

That decoupling is exactly what the rejected alternative could not provide.

### Why `1 − clip(Δq, 0, 1)` rather than a flat time counter

Summed over any completing trajectory the two are the same quantity:

```
Σ_t c_L6,t  =  T − Σ_t Δq_t  =  T − (s_goal − s_start)/D_REF
```

so L6 **ranks by duration exactly**, and the constant `Q` cancels from every
comparison. The difference between two completing runs is exactly their step
difference, independent of mission length — which is why the `λ₆` bound below
does not depend on `Q`.

What the advance-based form adds is a **per-step gradient**: at each step,
advancing more lowers the cost now. A flat counter gives the same ranking with
no local signal, which matters for value-based learning and which CaRL
explicitly had to work around by choosing Monte-Carlo returns.

Standing still and reversing both clip to zero advance and cost the maximum,
consistent with L4's own signed treatment of reverse motion.

### The bound, which binds on the scalar arm only

Reference shortcut of §4.6, the *cheapest* illegal shortcut saving the *most*
time: over a 200-step episode it saves **40 steps** while riding **one** marking
for **30 steps** at `c_L5 = 1/3`.

```
gain = λ₆ · 40 · (Δt/T_REF)      cost = η · (1/3) · (Δt/T_REF) · 30 = 1.0
=>  λ₆ < 0.25       (η = 1.0)
```

`λ₆ = 0.2` keeps a 20 % margin. Rank preservation (§5.4) gains `λ₆·(Δt/T_REF)`
in its tail: `2.0 + 0.1 + 0.02 = 2.12 < a = 2.2`.

### Thresholds for the thresholded-lexicographic arm

- **`τ₄` shrinks but does not vanish.** Two completing trajectories now tie at
  L4 to exact arithmetic, where a time cost inside L4 would have needed 0.400
  and `γ = 0.99` needed 1.975 — a width that also conflated completing the route
  with abandoning a fifth of it. That collapse is what this decision buys.

  **Corrected 2026-08-20:** an earlier revision of this ADR read that collapse
  as `τ₄ = 0`. That is the *fixture* value and it is wrong for training. The
  fixtures tie exactly because both trajectories are constructed over the same
  mission; two learned trajectories have equal L4 with probability zero, so at
  `τ₄ = 0` the comparison stops at L4 on essentially every pair and **L5 and L6
  decide nothing.**

  What replaces it is **not** a value. A value cannot be fixed here, because the
  object a threshold applies to differs across the candidate algorithms:
  *Absolute Thresholding* (Gábor et al. 1998) thresholds **Q-values**,
  *Absolute Slacking* (Li & Czarnecki 2019) thresholds a slack from the **state's
  own optimum** — already relative, so it sidesteps the mission-length problem —
  and a policy-gradient construction compares **expected returns**. A rule
  phrased on the completion fraction presumes the third reading.

  This ADR records only what holds under all three: `τ₄ > 0`, it must express
  "the progress difference is too small to justify a lane relaxation", and a
  single absolute constant cannot serve missions spanning 13 m to 247 m.
  **Sequenced after the rulebook is frozen.** Tracked as `D1` in
  `docs/open_items.md`.

  The one thing settled here is a property of the reward, not of the comparison:
  **the channel is not renormalized.** `Σ Δq` stays absolute, so §4.1.1's measured
  rejection stands and the scalar arm's calibration is untouched.
- **`τ₆`: none.** L6 is the last objective, and thresholded lexicographic
  ordering leaves the last objective unthresholded (Gábor et al. 1998; Vamplew
  et al. 2011).

This makes the arm structurally identical to the setting in which TLO was
originally validated: a thresholded prefix, an unthresholded final objective
that minimizes time to the terminal state, and `γ = 1`. Tercan & Prabhu's remark
that Vamplew's undiscounted results held *«due to their objectives ... a
terminating objective which pushes the agent to actually reach a goal state»*
applies to this structure directly, which it did not to the earlier proposal.

## The rejected alternative: the same quantity inside L4

The first proposal was `L4_t = Δq_t − κ`, `κ = 0.01`. It was implemented and
measured (expert mean +69.90 against +73.85, exactly the predicted `λ₄·κ·T`)
and then **rejected**, for a reason the user identified before the measurement
was read:

**inside L4, duration is priced above lane compliance.** Every unit of
preference for arriving sooner is also a unit the shortcut can spend on its own
violation, so O3 stops being a theorem — an exact tie at L4 decided by L5 — and
becomes a calibration, `80κ < η·c_L5·Δt·30`. That is precisely the property
`γ = 1` was bought to obtain, sold back at a smaller price.

It was also **too weak to do its job**, and necessarily so: the O3 bound caps it
at `κ < 0.0125`, hence at most 5 reward units over a 200-step episode, while a
single violated L2 step costs `a² = 4.84`. A cost bounded below the thing it has
to outweigh cannot outweigh it.

Both properties are recorded rather than deleted, because the bound and the
frontier remain the correct analysis for the scalar arm — where, as below, the
coupling genuinely is unavoidable.

## Risks

1. **The scalar arm keeps the coupling, and is measurably weaker for it.**
   Summing every channel re-couples what an ordering separates: `λ₆ < 0.25`
   buys 3.18 reward units between a crawl and a full-speed completion, against
   `a² = 4.84` for one violated L2 step. The scalar arm therefore prefers speed
   only while crawling would save **fewer than 0.66 L2 steps**. This is not a
   defect to engineer around — it is a scalarization failing to express what an
   ordering expresses, which is the comparison this thesis exists to make, and it
   must be **reported as a result**.
2. **L6 charges a legitimately stopped ego.** An ego correctly waiting at a red
   light pays `c_L6 = 1` per step. Gating L6 on whether an L3 rule is active was
   considered and rejected: it would couple two levels and add a gate to
   calibrate, while the hierarchy already supplies the justification — a
   trajectory that runs the red differs at L3, above, and two trajectories both
   waiting tie at L6 for as long as they both wait.
3. **`λ₆` is calibrated against a stipulated shortcut.** The expert never takes
   an illegal shortcut (it relaxes a lane rule on 0.8734 % of steps), so there is
   no empirical distribution to fit. The profile is declared in §4.6 and enforced
   by fixture; a more aggressive shortcut is *easier* for O3, not harder.
4. **A sixth level is more structure.** It adds one weight to the scalar arm and
   one channel to the vector the other arms consume. It adds no observation, no
   perception and no new quantity: `c_L6` is a function of `Δq`, which L4 already
   computes.

## Sources

- Jaeger, B. et al., *CaRL: Learning Scalable Planning Policies with Simple
  Rewards*, CoRL 2025 — the stationary-optimum failure, and the discount-supplies-
  urgency branch this project cannot take.
- Tercan, A. and Prabhu, V. S., *Thresholded Lexicographic Ordered Multiobjective
  Reinforcement Learning*, ECAI 2024. arXiv:2408.13493.
- Vamplew, P. et al., *Empirical evaluation methods for multiobjective
  reinforcement learning algorithms*, Machine Learning 84, 2011.
- Gábor, Z., Kalmár, Z. and Szepesvári, C., *Multi-criteria reinforcement
  learning*, ICML 1998 — thresholded lexicographic ordering, last objective
  unthresholded.
