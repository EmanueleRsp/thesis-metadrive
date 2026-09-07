# ADR-075: `γ = 1` for every arm, because discounting erodes the rule hierarchy

> **Amended by ADR-079 (2026-09-07, DRAFT pending user confirmation):
> `γ = 0.996`.** The erosion argument is upheld; two things are corrected. The
> break-even table below is computed one level too far apart (a future collision
> `a³` against a present **L3** violation `a¹`, ratio `a²`), and the binding
> comparison is L1 against L2 with ratio `a`, so every figure here is 2× too
> generous — 0.99 breaks at 78 steps, not 157. And the undiscounted case carries
> a cost this document does not weigh: roughly two thirds of episodes end in a
> bootstrapped truncation rather than a true terminal, so at `γ = 1` the value
> level is pinned only by the terminating minority and approximation bias walks
> it, which is the mechanism behind the observed critic-loss growth. The
> declared 0.999 fallback would have restored the contraction on paper while
> damping 18 % of that drift over a 199-step episode, against 55 % at 0.996.

- Status: **Approved, amended by ADR-079**
- Date: 2026-08-20
- Approval evidence: explicit user approval on 2026-08-20 ("approvo entrambe,
  procedi con ADR-075 e ADR-076"), after the three measurements below and the
  literature review were reported.
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §4.4. **Resolves `REQ-RB5.1-GAMMA`**, the last open requirement of that
  document.
- Affected configuration: `conf/agent/planner/algorithm/{ppo,ppo_sb3,sac,
  sac_sb3,td3,td3_sb3}.yaml` (`gamma`), and `learning_potential_gamma` in the
  two PPO configs.
- Related: ADR-073 (L4), ADR-074 (`SCAL-V1.3`), ADR-076 (`L6 progress_rate`, the
  time preference this decision makes necessary).

## Context

`RULEBOOK-V5.1` §4.4 left `γ` open and listed three options without selecting
one. It also made two claims that later measurement falsified, both corrected in
that document by this decision:

1. that the choice is "immaterial" for the **scalar** arm;
2. that at `γ = 0.99` the O3 ordering "holds only approximately".

Every figure below comes from `SCAL-V1.3` at the selected `λ₄ = 2.0`, `η = 1.0`
over the panel's mean **mission span** (`s_goal − s_start`, 90.17 m,
`Q = 40.58` normalized units) with a 200-step reference episode.

**Corrected 2026-08-20.** An earlier revision of this document used 171.98 m,
which is the *assigned route*; the mission covers only 52 % of it
(`q_start 0.316 → q_end 0.837`). Every L4 quantity below was therefore
overstated by 1.9×, and the tables are restated at the correct span. **No
conclusion changes**: the decisive measurement is the break-even step, which does
not depend on the span at all, and O3 still fails at every `γ < 1`.

## The decisive measurement, which is not about O3

The hierarchy prices a collision at `a³ = 10.648` and a non-relaxable violation
at `a¹ = 2.2`. Discounting erodes those two at different rates, so past some
horizon a **future collision costs less than one L3 violation now**:

| `γ` | break-even step | seconds | vs a 200-step episode |
|---|---:|---:|---|
| **0.99** | **157** | **15.7 s** | **hierarchy breaks inside the episode** |
| 0.995 | 315 | 31.5 s | safe |
| 0.999 | 1576 | 157.6 s | safe |
| 1.0 | never | — | safe |

At `γ = 0.99` on ~20 s episodes, a policy choosing between "run this red light
now" and "collide in 16 seconds" prefers the collision. This is independent of
L4, of the scalarization and of O3: it is exponential discounting against
geometric priority weights. **`γ = 0.99` was already incompatible with the
hierarchy this project built**, before any question about mission progress.

This consequence was derived here and no source stating it was found; it is
claimed as this repository's own result, not attributed.

## O3 under discounting, measured rather than argued

O3 — "legal route ≻ illegal shortcut, both completing". Scalar arm, margin
`A − B` where positive means O3 holds. `B` is a shortcut riding one lane marking
for 30 steps (`c_L5 = 1/3`):

| `γ` | scalar O3 margin at `T_B = 160` |
|---|---:|
| **1.0** | **+1.000** |
| 0.999 | −0.442 |
| 0.995 | −3.579 |
| 0.99 | −4.423 |

Only `γ = 1` gives O3 exactly, and it does so by construction: it is the unique
value for which `Σ_t Δq_t = (s_T − s_0)/D_REF` telescopes, i.e. for which two
trajectories reaching the same place tie at L4 whatever their duration.

`γ = 0.99` fails by −4.4 on the reference shortcut. **"Holds only approximately"
was an understatement and is withdrawn.**

## Three remedies falsified before selecting

1. **A threshold on L4** (the thresholded-lex arm). The `τ₄` wide enough to tie
   the legal route with the shortcut also ties it with a run that abandons the
   route:

   | `γ` | `τ₄` needed | also ties abandoning at |
   |---|---:|---|
   | 0.99 | 1.975 | **80 %** |
   | 0.995 | 1.690 | 90 % |
   | 0.999 | 0.539 | 99 % |
   | 1.0 | 0 | nothing |

   At `γ = 0.99` the discount-induced gap between two *completing* trajectories
   is larger than the gap between completing and abandoning a fifth of the
   route. No threshold separates them. An earlier claim that the thresholded arm
   is "immune" to this is **withdrawn**: it rested on an analogy with O1, where
   the tolerated residual really is small (L2 fires on 0.3978 % of expert
   steps), and the analogy does not survive contact with these magnitudes.

2. **Potential-based shaping** `γΦ(s′) − Φ(s)` with `Φ = s/D_REF`. Telescopes to
   `γ^T Φ_T − Φ_0`, and `done_function` terminates on arrival
   (`src/thesis_rl/envs/thesis_scenario_env.py:777`), so `T` differs:
   `0.99²⁰⁰·40.58 = 5.44` against `0.99¹⁶⁰·40.58 = 8.13`. The shortcut still
   wins by 2.69 — the same gap the plain form produces. Discounting **is** the preference for arriving
   sooner; one cannot discount and then ask the reward not to prefer it.

3. **Raising `η`.** Bounded by `λ₄ + 0.1·η < a`, so `η < 2` at `λ₄ = 2.0`.
   Offsetting the `γ = 0.99` gap would need a marking ridden continuously at full
   severity, which is not the shortcut worth protecting against.

## What the literature actually does

**It uses 0.99 and does not justify it.** This holds for MetaDrive itself and
for the closest comparable work, **V-Max** (RL framework for autonomous driving
on Waymax, 2025), whose appendix lists `γ = 0.99` across every algorithm with no
discussion, under a weighted-sum reward and no rule priority ordering. This
repository's six algorithm configs inherit the same default.

Tercan & Prabhu (ECAI 2024) state the situation directly:

> «the γ parameter is assumed to be an environment constant and traditionally
> set to values close to 1. Moreover, there is no real way to find the correct γ
> value apart from computing the action-value function, the very thing we are
> trying to compute.»

**There is therefore no justified practice to copy — only an inherited default.**

The same paper proves that discounting is structurally incompatible with
*thresholded lexicographic* ordering in value-based methods: their maze yields
`δ₁ < R(1−γ)` and `δ₁ ≥ Rγ(1−γ²)`, hence `0 > γ² + γ − 1`, i.e. `γ < 0.62`.

**Scope of that citation, stated honestly.** Their counterexample requires a
primary objective rewarded only at terminal states; all five channels here are
per-step, so the number 0.62 does not transfer. What transfers is that the
conflict between discounting and thresholded priority is a published result
rather than an artefact of this design.

Their remark on the undiscounted precedent is what settles the direction:

> «We believe the reason why Vamplew et al. (2011) has not observed this issue in
> their experiments with undiscounted (γ = 1) Deep Sea Treasure (DST) is due to
> their objectives. The secondary objective of DST, minimizing the time steps
> before terminal state, is a terminating objective which pushes the agent to
> actually reach a goal state.»

`γ = 1` is thus the setting in which thresholded lexicographic ordering was
originally validated — **conditional on a terminating objective supplying the
pressure the discount used to supply.** That condition is the subject of
ADR-076 and is not optional.

For finite-horizon episodic goal-reaching the undiscounted formulation is
standard: the return is finite because the horizon is, and it avoids the
systematic short-term bias of exponential discounting. Episodes here terminate
at the logged horizon or earlier (ADR-058 removed the tail), so every policy
terminates and the task is proper.

## Decision

**`γ = 1` on every channel and every arm.**

Justification, in the order that decides it:

1. discounting erodes the geometric priority weights and breaks the hierarchy
   **inside** a 20 s episode at `γ = 0.99` (measured, above);
2. the hierarchy's exchange rates are defined **per-step**, and any `γ < 1`
   makes them depend on when in the episode a violation occurs;
3. the conflict between discounting and thresholded lexicographic ordering is a
   published result, and the original TLO validation was undiscounted;
4. the task is episodic with a guaranteed finite horizon, where `γ = 1` is the
   standard formulation and the return is finite.

**One `γ` for all four arms.** Different discounts across arms would make any
difference in results non-attributable to the preference structure under test,
which is the comparison this thesis exists to make.

**`learning_potential_gamma` must track `γ`.** Ng et al.'s policy-invariance
theorem for potential-based shaping holds only when the shaping discount equals
the MDP discount; leaving it at 0.99 under `γ = 1` would forfeit the guarantee
(`src/thesis_rl/agent/planners/algorithms/ppo.py:563`, `ppo_sb3.py:464`).

### Declared fallback

If the value-based arms (SAC, TD3) fail to stabilize at `γ = 1`, drop to
**`γ = 0.999` for every arm together**, never per-arm. At 0.999 the hierarchy
stays safe by a factor of eight (break-even 157.6 s against 20 s episodes) and
O3 degrades by −0.4 to −1.8 on the reference shortcut, a degradation that must
then be **reported in the results**, not absorbed silently.

This fallback is declared in advance precisely so that taking it is a recorded
event rather than a tuning decision made under pressure.

## Risks

1. **Off-policy value stability.** With `γ = 1` the Bellman operator is not a
   sup-norm contraction. The task is proper — the logged horizon guarantees
   termination of every policy — so the formulation stays well-posed, but
   bootstrapped Q-learning at `γ = 1` is practically less stable than at 0.99.
   PPO at `γ = 1` is ordinary. This is the risk the declared fallback exists for.
2. **Value-target scale.** Undiscounted returns on the measured panel span
   roughly −150 to +80; at `γ = 0.99` the L4 contribution alone shrinks from
   40.58 to 17.57 on the mean mission. Existing configurations were tuned at 0.99 and their
   normalization has to be revisited.
3. **`γ = 1` removes all time pressure by itself.** Two trajectories arriving at
   10 s and at 20 s tie exactly, and nothing else in the rulebook prefers the
   faster one — `speed_limit` is an upper bound only. Arrival is not a
   sufficient bound either: the median mission needs 3.40 m/s to arrive inside
   its horizon while the agent is capped at 22.22 m/s, so the indifference band
   is **6.5× wide** and crawling stays available. **ADR-076 is the mitigation
   and is not severable from this decision.**
4. **Tercan & Prabhu's own remedy is not a `γ` choice** but abandoning
   value-based methods for thresholded ordering in favour of policy gradient.
   That is a design constraint on how the thresholded arm gets implemented —
   PPO is the safe carrier, SAC/TD3 the flagged combination — and is recorded
   here rather than decided.

## Sources

- Tercan, A. and Prabhu, V. S., *Thresholded Lexicographic Ordered Multiobjective
  Reinforcement Learning*, ECAI 2024. arXiv:2408.13493.
- Vamplew, P. et al., *Empirical evaluation methods for multiobjective
  reinforcement learning algorithms*, Machine Learning 84, 2011 — the
  undiscounted TLO experiments referenced above.
- Skalse, J. et al., *Lexicographic Multi-Objective Reinforcement Learning*,
  IJCAI 2022. arXiv:2212.13769.
- V-Max: *A Reinforcement Learning Framework for Autonomous Driving*, 2025.
  arXiv:2503.08388 — `γ = 0.99`, unjustified, on Waymax.
- Li, Q. et al., *MetaDrive: Composing Diverse Driving Scenarios for
  Generalizable Reinforcement Learning* — `γ = 0.99`.
- Ng, A. Y., Harada, D. and Russell, S., *Policy invariance under reward
  transformations*, ICML 1999 — the shaping-discount constraint.
