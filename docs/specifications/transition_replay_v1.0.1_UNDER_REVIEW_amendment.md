# Specification amendment: the priming priority is the current maximum, and 1.0 is its initialiser

## Metadata

- Feature: `transition_replay_new_transition_priority`
- Specification ID: `TRANSITION-REPLAY-V1.0.1`
- Version: `1.0.1`
- Status: `UNDER_REVIEW`
- Date: `2026-09-07`
- Amends: `docs/specifications/transition_replay_v1_specification.md`, `REQ-013`
  and its acceptance criterion `AC-016` only. `REQ-014`'s proportional
  stratified sampling, the importance-weight contract, the *n*-step target
  construction, the persistence contract and every other requirement are
  unchanged.
- Approval evidence: the substance was chosen by the user on 2026-09-07 —
  "remove the floor and amend the specification" — after a five-angle
  investigation refuted the reasoning that had been used to add it. **This
  document itself has not yet been stamped `APPROVED`**, and per `AGENTS.md` an
  `UNDER_REVIEW` specification is not an authoritative implementation contract.
- Authoritative: `NO` until approved.

## 1. What is wrong with `REQ-013`

`REQ-013` states two different rules and calls them one.

Its prose:

> new transitions receive the current maximum raw priority; the initial maximum
> is `1.0`.

Its invariant:

\[
p_{\mathrm{new}} = \max(1, p_{\max,\mathrm{current}}).
\]

The second is not a formalisation of the first. The prose describes a rule with
no constant in it, in which `1.0` is the value the maximum *starts at* before
anything has been priced. The formula describes a rule in which `1.0` is a
permanent floor, binding whenever the buffer's maximum falls below it.

`AC-016` inherits the formula: *"its raw priority equals `max(1.0,
current_max_raw_priority)`"*.

The implementation has always followed the prose. On 2026-09-07 the divergence
was read as a code defect and the code was changed to the formula; the change
was reverted the same day once the reasoning behind it was measured and found
wrong. The register records that episode as `C37`.

## 2. Why the prose is the rule worth keeping

None of the following is an argument from this repository's own documents.

**The sampler is exactly equivariant to a rescaling of the priorities, and a
constant floor is its only reward-scale-dependent term.** Leaves hold
\(p^{\alpha}\), an address's probability is its leaf divided by the total, and
the importance weights are max-normalised inside each batch. Multiplying every
priority by a constant therefore changes no sampled address and no weight.
Verified bit-exactly: under a \(|TD| \times 100\) rescaling of the same random
stream, the address sequence is **100.000 %** identical without the floor
(max weight difference 4.4e-16) and **0.641 %** — chance level — with it.

Raw priorities are absolute \(0.5\sum_i |\delta_i|\) over the twin critics
against an *n*-step target, in reward units. Those units are set by
`conf/scalarization/default.yaml`, which ADR-079 changed on the same day the
floor was added (`a` from 2.2 to 2.5, so \(w_1\) from 10.648 to 15.625), and
which `reward_compression: symlog` — approved but off — can compress. In the
0.01–0.1 band a floor at 1.0 acts as an uncalibrated 4–16x multiplier. A
constant expressed in reward units has to be re-derived every time the reward is
recalibrated, and nothing in the mechanism supplies a criterion for deriving it.

**The deficit a floor would close measures zero.** The share of draws reaching
rows that have never been priced is pinned by arrivals over draws —
`n_envs / (gradient_steps · batch_size)`, about 0.49 % at the shipped geometry —
*independently of the value those rows are primed at*; measured 0.39 % with and
without the floor. The priming level therefore sets only the latency to a first
sample, measured at 0–1.7 steps against a row residence of about 10 000, and no
transition was evicted without having been replayed in any of 12 configurations
examined. Priming at the current maximum already places an unseen transition at
the **top** of the distribution, which is what the mechanism exists to
guarantee.

**Where a floor binds it destroys the machinery it sits inside.** Once the floor
is the value written at insertion, `_current_max_raw_priority` collapses onto
that constant and stops measuring the critic at all, so the exact-maximum
bookkeeping the rest of `REQ-013` depends on becomes inert.

## 3. What `1.0` actually is

It is Schaul et al.'s \(p_1\), the priority assigned to the first transition so
that the recursion has somewhere to start. In the reference implementations that
popularised it — OpenAI Baselines and its descendants — it doubles as a
monotone running maximum initialised to 1.0, under Atari rewards **clipped to
[-1, 1]**, where a priority above 1 is by construction rare and the constant is
commensurate with the scale. This project clips no rewards and its residuals
live on the return scale: read directly out of all six PER buffers persisted on
disk, the maxima are 2.97, 3.91, 5.19, 5.35, 8.43 and 10.83. The constant is not
commensurate with anything here.

## 4. The amendment

`REQ-013` is restated so that prose and invariant say the same thing:

- Required observable behavior:
  - a transition that has never been priced receives the **current maximum raw
    priority** of the buffer;
  - while no transition is priced, that maximum reads `1.0`.
- Invariant:

\[
p_{\mathrm{new}} =
\begin{cases}
p_{\max,\mathrm{current}} & \text{if any row is priced,} \\
1 & \text{otherwise.}
\end{cases}
\]

- Rationale: a transition can be sampled before a TD error has been computed for
  it, and priming at the current maximum places it at the top of the
  distribution without introducing a reward-scale-dependent constant into an
  otherwise scale-equivariant sampler.

`AC-016` is restated to assert the same two cases, and to add the property that
carries the reasoning: rescaling every raw priority by a positive constant must
leave the sampled addresses and the importance weights unchanged.

## 5. Consequences

- **No code change is required by this amendment.** The implementation matches
  the amended requirement, and matched the original prose before 2026-09-07.
- `tests/test_transition_replay_per.py::test_per_priming_is_scale_equivariant_and_tracks_the_live_maximum`
  covers the amended `AC-016`, including the rescaling property. The floor test
  written on 2026-09-07 was removed with the user's explicit approval; it
  asserted a hand-built four-leaf state which, measured at production geometry,
  ordinary insertion cannot reach.
- `new_transition_priority: current_max` in `td3_sb3.yaml` and `sac_sb3.yaml`
  becomes an accurate description again. It has no reader; that is `C38`.
- **Not settled by this amendment**: the buffer maximum has never been logged
  during a run, so the regime in which a floor would bind — a converged,
  light-tailed residual distribution, where it binds on 27–98 % of insertions at
  \(\sigma \le 0.2\) — has been simulated but not observed. Instrumenting
  `_current_max_raw_priority` in the existing `training_progress` event would
  settle it empirically and is deferred to its own change.
