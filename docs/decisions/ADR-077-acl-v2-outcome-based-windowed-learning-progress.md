# ADR-077: ACL v2.0, outcome-based windowed learning progress

- Status: **Approved**
- Date: 2026-09-01
- Approval evidence: explicit user approval on 2026-09-01 of `DEC-201`…`DEC-206`
  as a block, after the assistant restated each decision with its question, its
  resolution, its evidentiary basis and the cost of choosing wrongly; plus the
  user's selection of the simpler alternative on `DEC-207`, with the explicit
  instruction not to add complexity that is not carrying its weight. The
  five-dimension order of `DEC-206` was selected by the user on the same date
  from three stated alternatives.
- Affected specifications:
  `docs/specifications/automatic_curriculum_learning_v2.0_specification.md`
  (`ACL-SN-EMA-001`, `v2.0`), which supersedes
  `automatic_curriculum_learning_v1.3_specification.md` in full.
- Related: ADR-029 (which named this fix and placed it out of scope), ADR-030,
  ADR-032 and ADR-028 (carried forward unchanged), ADR-016 (commit ordering),
  ADR-024 (data abort), and the `RULEBOOK-V5.1` decisions this contract now
  consumes — ADR-063, ADR-066, ADR-068, ADR-070, ADR-071, ADR-072, ADR-076.

## Context

Every ACL version from `v1` to `v1.3` derived the teacher's per-arm feedback
from a prediction-error learning potential: `mean(max(GAE, 0))` for PPO and
`mean(max(δ, 0))` for TD3/SAC. ADR-029 established by measurement that this
signal is proportional to the reward magnitude of the arm that produced it
(Spearman correlation between median LP and median `|reward|`, per arm: `+0.771`,
`+0.886`, `+0.829`, `+0.943` across four runs) and added per-arm reward-scale
normalization to remove the *scale* component.

The residual defect was recorded, not fixed, as `LIM-002`: at critic convergence
`E[δ] = 0`, so `mean(max(δ, 0))` is proportional to `σ(δ)`. A structurally
noisier arm keeps a systematically higher feedback with no learning in progress.
ADR-029 named the principled fix explicitly and deferred it: *"a learning-progress
signal (slope of return per arm over repeated visits)"*.

Two further facts shaped this decision.

**The defect was observed, not only predicted.** In a user diagnostic run,
`A4_vru`'s reward-scale EMA grew `48.9 → 104.7` while its bandit score stayed
the highest of the six (`FIND-006`). Normalization had not removed the
preference for the noisiest arm.

**The teacher was not the binding constraint.** Replaying the proposed teacher
over the real committed Generate sequences of three completed runs measured
`SNR ≈ 0.01` on every arm and every reconstructible dimension — the stationary
row (`FIND-013`). The cause was that the policy did not improve on any driving
outcome while the scalar reward did. That was a reward-specification defect, and
it was addressed separately by `RULEBOOK-V5.1`: under the specification measured
then the human expert scored `−203.35` with `46.55 %` of episodes below
standstill; under `RULEBOOK-V5.1` the same 1100-record Waymo panel scores the
expert at `+70.70` with `3.36 %`. `FIND-013` is therefore obsolete rather than
falsified, and no `SNR` measurement exists under the reward now in production.

A second, independent motivation is methodological. Under `v1.3` PPO's feedback
came from GAE advantages and TD3/SAC's from TD residuals — different quantities,
different units, different sensitivity to critic quality. The curriculum was not
the same curriculum across the algorithm arm of the experimental design. A
teacher consuming only environment and Rulebook outcomes is identical across
planners and across the four reward settings, which is what makes the reward-
setting × curriculum comparison of `EVAL-PROTOCOL` interpretable.

## Decision

Replace the prediction-error family with a **windowed, ordinal, outcome-based**
learning-progress signal, attenuated by an explicit learnability gate. Per arm
and per dimension the teacher compares the `H` most recent Generate episodes
against the `H` preceding ones with the Vargha–Delaney common-language effect
size `G`, whose expectation is exactly `0.5` under exchangeability regardless of
the distribution's variance. That property is what removes `LIM-002` at its
root, and it was verified by measurement over a `250×` range of noise scale
(`FIND-010`), not asserted analytically.

The seven decisions approved as a set:

| ID | Decision | Rests on |
|---|---|---|
| `DEC-201` | L1's episodic cost is the `max` over **all** steps and L1 is always observed; `applicable=False` means *satisfied*, not *unobserved* | code verification (`FIND-008`) |
| `DEC-202` | The gate consumes at-fault collision incidence on L1, the violated-applicable-step fraction `F̄` on L2/L3/L5, and route completion on `T`; the comparison `G` keeps the fine-grained cost `C` | code verification (`FIND-009`, `FIND-011`) |
| `DEC-203` | Exact conditional permutation band at `0.05` **per dimension, uncorrected**; framed as a calibrated deadband, never as a significance test | reproducible synthetic measurement (`FIND-010`) |
| `DEC-204` | The `v1.3` prediction-error channel is retained and continues to run, strictly **inert**, as a diagnostic | design; enables within-run comparison of the two signals |
| `DEC-205` | `H = 20`; `H = 30` is the indicated revision if the deferred `SNR` measurement returns below `0.5` | measured power sweep (`FIND-012`) |
| `DEC-206` | Five teacher dimensions in the order `L1 → L2 → L3 → T → L5` | `RULEBOOK-V5.1` §3 hierarchy; ADR-072, ADR-076 |
| `DEC-207` | An episode truncated by a not-at-fault contact receives **no special handling** and updates every window, `T` included | user instruction; no measurement establishes the alternative's premise |

The scenario buffer becomes an arm-balanced recency memory: admission and
eviction no longer depend on any usefulness value, `P_progress = p_i / n_i`, and
staleness is the only per-record term. Generate eligibility, per-arm coverage
cycles, quarantine and the data-abort contract are carried forward unchanged
from `v1.3`.

## Rejected alternatives

**Holm's correction, retained.** Rejected on measurement, and explicitly *not*
on the argument that it blocked the curriculum — that argument was tested and is
false: peak arm probability during an arm's learning phase was `0.24–0.27`
uncorrected against `0.23–0.26` corrected, and at low signal neither departed
from uniform. What Holm suppresses is a doubling of the false-fire rate, and
those false fires are symmetric in sign, so they cost variance and not bias
(mean feedback `0.498–0.500` either way). Removing it buys a factor two of
sensitivity at a measured-negligible bias cost. Any derived document claiming
that multiplicity correction was preventing the curriculum from working is
falsifiable with the recorded script.

**An episode-level violation indicator on L2/L3/L5.** Rejected because it
saturates: L2's sub-rules are applicable whenever a relevant actor exists and
are violated at *some* step in essentially every episode with traffic, so
`v̄ → 1` and the gate would close permanently even while the violated-step
fraction fell from `70 %` to `10 %`.

**Averaging L1's cost over applicable steps.** Rejected because L1 is applicable
only on a new at-fault contact, so the window would contain only collision
episodes and would measure severity given a crash instead of collision
behaviour.

**A per-record curricular score.** Rejected (`RAT-206`): a single episode cannot
separate an adequately difficult scenario from a lucky success, an unlucky
failure, or transient policy variation. Learnability is estimated at arm level.

**Excluding not-at-fault-truncated episodes from the `T` window.** Proposed by
the assistant on the argument that the truncation caps `route_completion` for a
reason the policy is not charged for and that its frequency grows with traffic
and therefore with the arm. Rejected by the user: the distortion is hypothetical
while the special case is certain. `ADR-071` already requires the not-at-fault
rate as a diagnostic, so the question can be reopened with a number. Recorded as
`LIM-210`.

**Four dimensions with L5 dropped, and `L1 → L2 → L3 → L5 → T`.** Both rejected
under `DEC-206`: the first discards a measurable dimension that priority
ordering already consults last, and the second would place lane relaxation above
mission progress, inverting ADR-072.

## Consequences

**Breaking.** Checkpoint schema `acl_ema_v3` → `acl_progress_v1`, with no
migration path; `ScenarioRecord` gains `last_generate_step` and loses the
ranking meaning of `usefulness`; the RNG call order changes, so `v1.3` seeds do
not reproduce their runs. Runs in progress must be restarted — already required
independently by `γ = 1`, `speed_limit` and the observation amendments.

**Pre-registration.** §11 requires that approval precede any comparison run
under this version. This ADR and §18 of the specification are that evidence, and
they are dated before the learnability runs.

**What may not be claimed.** That `H = 20` is validated against real data (it is
validated against a synthetic power sweep, and the only real measurement was
taken under a superseded reward); that the arms attain a signal level at which
the teacher becomes active; that `v2.0` will produce a non-uniform curriculum;
or that the `v1.3` curriculum harmed policy performance (`FIND-007`: one seed,
40 episodes, weakened further by `FIND-013`).

**Interpretation caveat.** `FIND-015` measures that on PG records four of L3's
six sub-rules never apply and a fifth is rare, while on Waymo records L3 is a
six-sub-rule level. Since the arms are correlated with the source, a fired `L3`
means a different statement depending on which arm fired it. The estimator is
unaffected — `G` never forms a cross-arm comparison — but curriculum analyses
must report the arm's source composition. Recorded as `LIM-209`.
