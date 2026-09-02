# Ride-Comfort Diagnostics — Instrument Validation And Expert Admissibility

- Date: 2026-09-01
- Status: `EVIDENCE`; not a specification and not authoritative
- Subject: `docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md`
  (`EP-COMFORT-DIAG`), under `docs/specifications/rulebook_v5.1_specification.md`
  §13
- Purpose: settle two questions empirically rather than by argument — (1) does an
  unfiltered kinematic derivative in this simulator measure ride comfort at all,
  and (2) do nuPlan's comfort bounds survive the admissibility test this
  repository applies to every rulebook sub-rule?
- Artifacts: `comfort_test_a_waymo_train.json` (the Test A result, extracted from
  the full measurement written to
  `outputs/comfort_expert_test_a_waymo_train.json`).

Reproduce Test A with:

```bash
docker compose run --rm dev uv run --no-sync python scripts/measure_expert_rulebook_transition.py --data-root /workspace/data/scenarionet --frozen-index /workspace/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json --split train --source waymo --workers 24 --output /workspace/outputs/comfort_expert_test_a_waymo_train.json
```

Roughly 20 minutes on 24 workers. No learner and no training: this is offline
replay of logged expert trajectories.

---

## 0. What is measured, and what is therefore *not* established

Both findings below concern the **instrument**, not any policy. §2 replays
logged Waymo expert trajectories; §1 measures the same instrument against itself
under two derivative conventions on live simulator rollouts of an untrained
policy.

Nothing here establishes that any trained agent is comfortable or uncomfortable.
It establishes what the comfort channels are capable of measuring, and whether a
competent human satisfies their thresholds in this simulator's state
representation.

## 1. Why ride comfort is measured at all

`RULEBOOK-V5.1` §13 excludes comfort and jerk from the rulebook and from the
reward, on the supervisor's guidance that ride comfort belongs downstream of the
policy — in a filter or damper acting on the produced control commands — rather
than in a normative rulebook. §13 explicitly permits logging jerk as a
diagnostic.

That exclusion creates a measurement obligation rather than removing one. A
policy under no pressure to produce smooth control may satisfy every rulebook
level with a bang-bang throttle, and without a diagnostic channel the work
cannot say whether that happened. The claim "comfort was deliberately left out
of the reward" is only defensible alongside evidence of what the resulting
policies actually do, and comfort is a dimension every published planning
benchmark reports, so its absence would also block comparison.

The channel is therefore diagnostic by construction: it is computed at
evaluation time only, it is reported outside the primary comparison tables, and
no rule, margin, reward, or termination decision reads it.

## 2. The instrument

nuPlan's `ego_is_comfortable`: seven kinematic channels, each with a threshold
the devkit describes as "determined empirically from examination of a dataset of
expert trajectories", and the boolean their conjunction implies. Both the
thresholds and the Savitzky-Golay derivative parameters are transcribed from
the devkit rather than chosen here, and are pinned by a test so they cannot
drift.

Two adaptations are forced by the environment and are the only departures:

1. nuPlan reads the simulator's own `center_acceleration_2d` and smooths it
   (`window_length=8`, `poly_order=2`). MetaDrive's authoritative snapshot
   publishes velocity, not acceleration, so acceleration here is the
   Savitzky-Golay **first derivative** of velocity at the same window and
   polynomial order. Same filter, same window, one differentiation earlier.
2. nuPlan filters one contiguous trajectory. An evaluation episode may contain
   steps whose kinematics are unusable, so the series is split at those gaps and
   each contiguous segment is filtered independently. No filter window spans a
   gap.

One devkit detail is worth recording because it is invisible from the function
signatures: `extract_ego_yaw_rate` accepts a `window_length` but never forwards
it to `approximate_derivatives`, which therefore applies its own default of 5.
The yaw channels really run at window 5, not the 15 the signature suggests.
Reproducing the devkit's behaviour rather than its documentation is what makes
these numbers comparable to published nuPlan figures.

The offline instrument of §2 and the online evaluation path call the **same**
function (`ego_kinematics_payload`) on the same `EnvSnapshot`, so the human
reference and the agent measurements are produced by one definition rather than
two.

## 3. Finding 1 — an unfiltered derivative measures the integrator, not the driving

The first implementation used raw backward finite differences, on the reasoning
that filtering would require a dependency and that unfiltered maxima are merely
*conservative*: noisier, biased upward, failing more readily than nuPlan's would
on the same trajectory. That reasoning was wrong in a way worth recording,
because "conservative" implies the measurement is still usable.

Measured on one identical live rollout, filtered against unfiltered:

| Channel | Unfiltered | Filtered (nuPlan) | Bound |
|---|---|---|---|
| `max_abs_lon_jerk` | 63.29 | **2.35** | 4.13 |
| `max_abs_mag_jerk` | 30.85 | **1.32** | 8.37 |

A factor of roughly 27. At that scale the unfiltered channel is not a
pessimistic estimate of ride comfort; it is dominated by the physics
integrator's step-to-step response and by controller chatter, quantities that
have no relationship to how a passenger would experience the ride.

The practical consequence is visible in the end-to-end smoke run. Unfiltered,
`comfort_rate` was **0.0 on every evaluation panel**, with mean per-episode jerk
between 46 and 174 m/s^3 against a bound of 8.37: a saturated constant, useless
as a comparison between arms. Filtered, on the same configuration, the same
metric reads **0.21 on one panel and 1.00 on another**, with 53 of 87 episodes
comfortable and a median `max_abs_lon_jerk` of 0.83. The filter did not rescale
the numbers; it converted a degenerate constant into a measurement that
discriminates, on a policy that had barely trained.

**Generalisable point.** A kinematic derivative taken raw from a physics
simulator at policy rate measures the simulator. Any jerk, yaw-acceleration, or
comfort figure computed this way — in this work or elsewhere — should state its
derivative convention, because the convention changes the result by more than an
order of magnitude and can silently saturate the metric.

## 4. Finding 2 — the comfort bounds are admissible for this expert (Test A)

`RULEBOOK-V5.0` §2.1 admits a rule only if a competent driver can satisfy it:

> A rule that carries a satisfaction indicator must be satisfiable by a
> competent driver. [...] A rule the expert violates on a large fraction of
> steps is charging the agent for driving, not for driving badly.

Every rulebook sub-rule was falsified this way. The comfort bounds had entered
on nuPlan's published authority alone, which is a methodological inconsistency
in a work whose whole argument is that thresholds must be falsified rather than
asserted. It also left a live risk unquantified: thresholds calibrated on real
vehicles need not transfer to MetaDrive's state representation, and if the
expert breached them routinely the metric would be uninformative.

The bounds were therefore put through the same instrument, on the same panel, by
the same script that produced `RULEBOOK-V5.1` §5.5.

**Scope: 1100 Waymo `train` records, 217,189 transitions, 0 skipped** — the
identical scope the v5.1 rulebook was calibrated on. A verdict was defined on
all 1100 episodes.

| Channel | Expert violation rate | p50 | p95 | p99 | Bound |
|---|---|---|---|---|---|
| `max_abs_lon_jerk` | 5.73 % | 1.145 | 4.860 | 26.14 | 4.13 |
| `max_lon_accel` | 5.00 % | 1.104 | 2.399 | 2.667 | 2.40 |
| `min_lon_accel` | 3.55 % | -1.287 | -0.004 | -0.002 | -4.05 |
| `max_abs_mag_jerk` | 3.09 % | 1.040 | 3.751 | 26.08 | 8.37 |
| `max_abs_lat_accel` | 0.00 % | 0.145 | 2.574 | 2.991 | 4.89 |
| `max_abs_yaw_rate` | 0.00 % | 0.016 | 0.508 | 0.558 | 0.95 |
| `max_abs_yaw_accel` | 0.00 % | 0.066 | 0.397 | 0.634 | 1.93 |

**Expert comfort rate: 0.8973.** The logged human satisfies all seven bounds
simultaneously on 89.7 % of records, and no single bound is violated on more
than 5.73 %.

Read against this repository's own precedents the result is unambiguous. Under
the same test, `rss` longitudinal was **rejected** at 18.29 % of applicable
expert steps and demoted to a reported diagnostic; `speed_limit` was **adopted**
at 0.000 %. Every comfort channel sits far closer to the adopted end than to the
rejected one. The bounds are admissible in MetaDrive's state representation, not
merely in nuPlan's.

## 5. What this licenses, and what it does not

Licensed:

- Reporting `comfort_rate` **relative to a human reference**. An arm scoring
  0.40 is not simply "uncomfortable"; it is less comfortable than the 0.90 a
  logged human scores on the identical instrument, on the identical panel.
- Claiming the exclusion of comfort from the reward was measured rather than
  assumed, since the resulting behaviour is now observable.
- Comparing against published nuPlan figures, with the two §2 adaptations named.

Not licensed:

- Any claim about a trained policy. Nothing here evaluates one.
- Treating comfort as a rulebook channel or a reward term. `RULEBOOK-V5.1` §13
  excludes it, and this evidence does not reopen that.
- Reading 0.8973 as "the human comfort rate". It is this expert, on this panel,
  under this instrument and these two adaptations.

## 6. Guidance on thesis use

Recorded explicitly because not everything in `EP-COMFORT-DIAG` deserves thesis
space, and padding a methods chapter with engineering history weakens it.

**Worth including.**

- §1, the argument that excluding comfort from the reward creates a measurement
  obligation rather than removing one. It justifies the diagnostic's existence
  in two or three sentences and pre-empts the obvious examiner question about
  why a driving-quality dimension is absent from the reward.
- §4, the Test A result. This is the substantive contribution: it extends the
  falsification methodology the work already applies to the rulebook to an
  evaluation metric, closing a methodological inconsistency, and it produces a
  human reference number that makes every later comfort figure interpretable.
  The comparison against the `rss` rejection and the `speed_limit` adoption is
  what makes 5.73 % legible as "clearly admissible" rather than an isolated
  percentage.
- §3, as a methods caution, probably a short paragraph or a footnote. The
  factor-27 result is genuinely useful to a reader computing kinematic
  derivatives in a simulator, and it explains why the derivative convention is
  stated at all.

**Not worth including.**

- The CSV schema, the writer call sites, the analysis-table wiring: engineering,
  fully recorded in the ExecPlan.
- The two defects found during implementation (a comfort verdict that
  short-circuited before checking definedness, and a first version reading an
  info key the live evaluation stack never publishes). They are development
  history. The second has a mild methodological moral — a metric can pass every
  unit test and still measure nothing, because unit tests pin a contract to
  itself — but it is not a thesis result and should not be dressed as one.

## 7. Limitations

1. **Two adaptations remain** (§2): velocity-derived acceleration, and
   segmentation at unusable steps. Neither changes the filter, the window, or
   the polynomial order, but both should be named when quoting against
   published nuPlan numbers.
2. **Heavy right tail in the jerk channels.** `max_abs_lon_jerk` has p95 4.86
   against p99 26.14 and a worst of 62.55; `max_abs_mag_jerk` behaves the same
   way. A tail of that shape is far more consistent with logged-track
   discontinuity in a small number of Waymo records than with human driving.
   The reported violation rates are therefore a slight **over**-estimate, which
   is the conservative direction for an admissibility test and does not
   threaten the pass. Confirming it would mean inspecting the worst handful of
   records; not done.
3. **Waymo only.** As with every other calibration in this work, PG records are
   excluded: their logged ego is `IDMPolicy`, so replay establishes nothing
   about what a competent human would do.
4. **`train` split only**, consistent with the rulebook calibration policy of
   never falsifying against validation or test data.
