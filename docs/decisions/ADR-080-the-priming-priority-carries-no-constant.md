# ADR-080: the priming priority is the buffer's own maximum, and carries no constant

- Status: **Approved**
- Date: 2026-09-07
- Approval evidence: explicit user approval on 2026-09-07. The user first
  rejected the reasoning that had been used to add the floor — "non
  giustifichiamo le scelte con 'la specifica dice così'. La motivazione dietro
  dev'essere valida e corretta, altrimenti è una roba circolare (le specifiche
  le abbiamo comunque scritte noi)" — then chose, after the investigation below
  was reported, to remove the floor and amend the specification, and approved
  the amendment on being told it was the better rule.
- **No ExecPlan.** The code change is a two-line revert to behaviour that had
  been in place since `TRANSITION-REPLAY` was implemented; the specification
  amendment requires no implementation. `docs/project_index.md` gains a row for
  the amendment because it is a specification, not because there is a plan.
- Affected specifications:
  `docs/specifications/transition_replay_v1_specification.md` `REQ-013` and
  `AC-016`, restated by
  `docs/specifications/transition_replay_v1.0.1_amendment.md`
  (`TRANSITION-REPLAY-V1.0.1`).
- Affected code: `src/thesis_rl/sb3_extensions/replay/prioritized.py`
  (`_insertion_priority`).
- Related: `C37` and `C38` in `docs/open_items.md`; `D13`, withdrawn the same
  day; ADR-079, which moved the reward scale this constant was expressed in.

## Context

`REQ-013` stated two different rules and called them one. Its prose: "new
transitions receive the current maximum raw priority; the initial maximum is
`1.0`." Its invariant: \(p_{new} = \max(1, p_{\max,current})\). In the prose,
`1.0` is where the maximum *starts*; in the formula it is a permanent floor. The
implementation had always followed the prose.

On 2026-09-07 the divergence was read as a code defect and the code was changed
to the formula, justified by the formula. Two things then had to be settled: was
the change right on its merits, and which half of a self-contradictory
requirement should survive.

## Decision

Keep the prose rule. An unseen transition is primed at the buffer's exact
current maximum raw priority; `1.0` applies only while nothing is priced.
Restate `REQ-013` and `AC-016` so that prose and invariant agree, and record
`1.0` as an initialiser rather than an invariant.

## Why, without appeal to the specification

**The floor was the only reward-scale-dependent term in an exactly
scale-equivariant sampler.** Leaves hold \(p^{\alpha}\), an address's
probability is its leaf over the total, and importance weights are
max-normalised inside each batch, so multiplying every priority by a positive
constant changes no address and no weight. Verified bit-exactly: under a
\(|TD|\times 100\) rescaling of one random stream the address sequence is
**100.000 %** identical without the floor (largest weight difference 4.4e-16)
and **0.641 %** — chance — with it. Priorities are absolute
\(0.5\sum_i|\delta_i|\) in reward units, and ADR-079 moved those units on the
same day (`a` 2.2 → 2.5, so \(w_1\) 10.648 → 15.625); `reward_compression:
symlog` can compress them further, which would turn the floor into an
uncalibrated 4–16x multiplier in the 0.01–0.1 band. A constant in reward units
has to be re-derived whenever the reward is recalibrated, and the mechanism
supplies no criterion for deriving it.

**It bought nothing measurable.** The share of draws reaching rows that have
never been priced is fixed by arrivals over draws,
`n_envs / (gradient_steps · batch_size)` ≈ 0.49 % at the shipped geometry,
*independently of the value those rows carry*; measured 0.39 % with and without
the floor. Priming therefore sets only the latency to a first sample: 0–1.7
steps against a row residence near 10 000, with no transition evicted unsampled
in any of 12 configurations examined. Priming at the current maximum already
places an unseen transition at the top of the distribution, which is the whole
purpose.

**Where it binds it disables the mechanism it sits in.** Once the floor is what
gets written at insertion, `_current_max_raw_priority` collapses onto that
constant and stops measuring the critic, so the exact-maximum bookkeeping the
rest of `REQ-013` depends on goes inert.

**The evidence that motivated the floor was wrong.** The claim was that a run
whose residuals sit below 1 has a buffer maximum below 1. That is an invalid
mean→maximum step: `critic_loss` is an importance-weighted mean of *squared*
residuals over one batch, while priming reads a maximum over up to 300 000
accumulated and stale rows. The supporting figure ("critic losses 0.6–0.9") has
no artifact — the values recorded that day are 0.0018–0.396 — and the run whose
`critic_loss` was 0.396 has a measured buffer maximum of **8.42774**, so the
inference was off by 13.4x on its own example. Read out of all six PER buffers
persisted on disk, the maxima are 2.97, 3.91, 5.19, 5.35, 8.43, 10.83; even the
degenerate run whose rewards all lie in `[0, 0.0905]`, where the reward term
cannot produce a residual above 1, measures 2.97.

## Alternatives rejected

- **A monotone "maximum ever seen"**, as OpenAI Baselines and its descendants
  implement. It is a ratchet on a non-stationary quantity: this repository has
  already recorded critic losses going 22 → 506 inside one run, and one such
  excursion pins the priming value for hundreds of thousands of steps of a run
  whose purpose is for residuals to shrink. Measured on a decaying stream, it
  fossilised at 8.47 (light tail) and 151.16 (heavy tail) while the live p99
  fell to 0.078–0.080, and never decayed.
- **A scale-free floor \(c \cdot p_{\max,current}\) with \(c > 1\).** It is
  self-referential — each insertion re-seeds the maximum it just read,
  multiplied by \(c\) — and therefore a multiplicative ratchet: measured at
  \(c = 1.5\) and 512 draws per insertion, the priming value diverged to
  9.8e+58 in 4000 steps, identical at every residual scale, i.e. fully
  decoupled from the data. The criterion that would make \(c\) derivable
  returns \(c = 1\) at this configuration, which is the decision above.

## Consequences

- No behaviour change relative to every completed run: the reverted code is the
  code they ran. The floor existed for about one hour, on one branch, and never
  in a run.
- `new_transition_priority: current_max` in `td3_sb3.yaml` and `sac_sb3.yaml` is
  accurate again. It has no reader; that is `C38`, deferred to its own change
  together with the four other dead `per:` keys and the resolver's missing
  unknown-key validation.
- The regression test that asserted the floor was removed with explicit
  approval and replaced by one asserting the property the decision rests on —
  that priming tracks the live maximum and rescales exactly with it. Coverage is
  redirected, not reduced.
- **Open, and deliberately so.** The buffer maximum has never been logged during
  a run. The regime in which a floor would bind — converged, light-tailed
  residuals, where it binds on 27–98 % of insertions at \(\sigma \le 0.2\) — has
  been simulated but not observed. Instrumenting `_current_max_raw_priority` in
  the existing `training_progress` event would settle it empirically, and this
  decision is reopenable on that measurement.
