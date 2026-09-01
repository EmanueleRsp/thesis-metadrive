# ADR-063: RSS longitudinal is demoted from a reward sub-rule to a reported diagnostic

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending. Presented with the measurement below; the user
  approved the direction of the redesign but the specification carrying it is
  `UNDER_REVIEW`.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.2, §8 (amends `rulebook_v4.7_specification.md` §7.1).
- Withdraws: `DEC-RSEC-001` (the R2 cost deadband), which was an attempt to
  rescale this sub-rule rather than to question it.

## Context

`components/rss.py` charges the ego whenever the bumper-to-bumper gap to a
same-stream leader falls below the RSS longitudinal safe distance
(Shalev-Shwartz et al., 2017) with rho = 1.0 s, a_max = 3.5 m/s^2, b = 8.0 m/s^2.

Replaying 1100 logged Waymo `train` records (217,189 transitions) through
production's own `evaluate_transition`:

| quantity | value |
|---|---:|
| `rss` violated, % of all steps | **8.400 %** |
| `rss` violated, % of applicable steps | **18.290 %** |
| mean cost on violated steps | 0.337 |
| expert mean episode return, production rulebook | **-203.35** |
| expert mean episode return with `rss` removed | **-5.29** |
| expert episodes below standstill, production | 46.55 % |
| expert episodes below standstill with `rss` removed | 19.64 % |

Removing this single sub-rule accounts for roughly 97 % of the gap between the
expert and a policy that never moves.

## Decision

`rss` leaves the reward. It is computed every step and reported as a diagnostic
KPI at unchanged published parameters.

## Rationale

**Controlled invariance.** A region is admissible as a per-step penalty only if
from every state inside it an action exists that keeps the ego inside the
admissible set. The RSS envelope of other vehicles is not such a region: a
cut-in places the ego inside it instantaneously and no action available this
step restores the gap. A planner meeting an infeasible constraint reports
infeasibility and executes a fallback; a per-step reward simply keeps charging.
Unavoidable states are expressed in this repository by termination, which
already exists.

**Recalibrating rho was rejected on the merits.** At urban speeds RSS with
rho = 1 s is algebraically a headway rule of ~1.6-1.9 s plus a constant, so
lowering rho is the same knob as lowering a time-headway threshold. rho is a
physical claim about the ego's reaction time; selecting it so that the expert
passes is precisely the trial-and-error reward design that Knox et al. (2023),
sanity check 7, identify as the field's dominant methodological error.

**The guarantee was never implemented.** RSS's safety guarantee requires the
proper response, which this repository has never implemented and which carries
an unbounded per-actor latch that is not Markovian. A rule named RSS implies a
guarantee it does not provide.

**Two alternatives were built and measured before concluding.** A time-headway
rule (six thresholds x four minimum-speed gates) and a controlled-invariant
"responsive RSS" (charge only when the gap is unsafe *and* the ego is not
decelerating) both fail Test A at every published parameter; the responsive form
removes 61 % of violations at rho = 1 s but still stands at 7.17 % of applicable
steps. Both results are recorded in the specification §4.6 as negative results.

## Alternatives rejected

1. **Lower rho to 0.3-0.5 s** - fitted physical parameter, see above.
2. **Add a deadband to the cost** (`DEC-RSEC-001`) - rescales the symptom and
   leaves the non-invariance intact.
3. **Replace with time headway** - measured, fails Test A at every anchored
   threshold, and its rate *rises* when low-speed queueing is excluded.
4. **Keep it and accept the cost** - contradicts the premise that a competent
   driver must outscore standing still.

## Consequences

The reward's anticipatory longitudinal coverage becomes `ttc` (relative
velocity) plus R1. An agent tailgating at matched speed is unpenalised until the
leader brakes; nuPlan's closed-loop score has no following-distance metric
either and relies on TTC plus collisions. The limitation is declared in the
specification §11.2 and quantified by the retained diagnostic.
