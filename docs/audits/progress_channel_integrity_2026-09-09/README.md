# Progress-channel integrity, 2026-09-09

Continues `docs/audits/rulebook_architecture_2026-09-09/`, whose §8 opened the
three defects of the mission-progress channel. This directory holds what closed
them and the decision brief for the one that did not.

**Status.** `V3` and `C51` are closed. `C49`'s measured-horizon half is closed
here; its discount was **approved on 2026-09-09 at `γ = 0.9982`** together with
the A7 architecture, and lands with the A7 ExecPlan change rather than with this
one. `C50` is a decision and `D14` is now measured. No approved behaviour changed
here: the reward, the projection and the mission station are exactly what they
were, and neither `γ` nor any weight is touched by this change.

| file | what it holds |
|---|---|
| `README.md` | this: the `V3` result, the `C50` decision brief with its three candidates priced, and `D14`'s backwards route walk |
| `run_2026-09-09.txt` | the `V3` audit's output over the current frozen index, verbatim |
| `d14_walk_2026-09-09.json` | the walk's summary at both station spacings, without the per-record rows |

The instruments are `scripts/audit_route_near_revisits.py`, with acceptance tests
in `tests/test_route_near_revisit_audit.py`, and the `--walk-spacing-m` mode
added to `scripts/measure_final_gate_carriageway_coverage.py`. Both write
per-record JSON that is megabytes wide and regenerable with `--output`, so only
the summaries are committed.

## What the near-revisit audit found

Read-only over `data/scenarionet/frozen/scenario_selection_index.json`, all 3,500
missions, no simulation. Two readings, because one would overstate safety.

**With the ego on the route**, one route of 3,500 can present the clip with an
arc-length change it cannot express, and the amount is **+0.028 channel units** —
6.2 cm of arc length past one clip width. No pair anywhere in the index exceeds
**twice** the clip width at one step of travel.

**With the ego off the route**, the exposure is bounded by how far it may go. An
ego next to the medial axis between two route portions switches branch in one
step whatever their separation, so the figure that matters is the *fold excess*
`a - d`: how much longer the route is between two portions than the straight line
between them. It is zero on a straight route, small under curvature, and the
whole loop on a fold.

| lateral reach | worst fold excess | worst arc separation | branch-switch under-charge |
|---|---:|---:|---:|
| on the route (0 m) | 0.061 m | 2.284 m | 0.028 |
| one lane (1.75 m) | 0.247 m | 3.726 m | 0.677 |
| 3.00 m | 1.017 m | 6.824 m | 2.071 |
| 5.00 m | 3.226 m | 13.209 m | 4.944 |

The median route has a fold excess of **0.000 m** at every reach up to 10 m.

`g6_ratchet.py` executes the ratchet on a hairpin whose legs are one lane apart
and whose fold excess is therefore about **80 m**. The worst comparable geometry
in the frozen index is **0.247 m**. A quarter of a metre cannot close a loop, so
what the index admits at that reach is a one-off of 0.677 channel units, not a
gain that repeats per lap.

**The audit is cross-validated in both directions.** It reproduces the 2026-09-05
run exactly — zero routes with two portions more than 15 m apart in arc length
within 3.5 m or within 6 m, closest approach 10.581 m — and its positive control
is the `g6` hairpin, on which it recovers the arc separation that script's return
jump charges `-1`.

**Why not the original threshold.** Arc length is never shorter than its chord,
so "two portions more than one clip width apart in arc length within one lane
width" fires on every route in the index while describing nothing. Substituting
2.2222 m for 15 m in the 2026-09-05 criterion would have been degenerate; the arc
separation has to be paired with the planar one. The instrument's docstring
derives the pairing.

**One thing the audit prices that it was not built for.** ADR-035 sized its
continuity bound at a *factor* of `2.0` rather than `1.0` because cutting the
inside of a bend advances the centerline coordinate faster than the ego moves.
That factor was chosen without measurement. The measurement is now here: the
largest station advance any route in the index offers for one step of travel is
**2.284 m = 1.028 clip widths**. The legitimate requirement is 1.03, not 2.

## `C50`: the decision, and what each exit costs

Three facts were verified against the code first, because two claims the working
notes carried into this were wrong.

**The continuity bound does not exist in production, and has not since
2026-08-03.** `ROUTE_CONTINUITY_JUMP_FACTOR` is in no Python file; it survives in
ADR-035's prose and two comments that cite it by name. The bound was wired into
`evaluate_progress` on 2026-07-30 (`b30763a`, `be7c694`) and **`e63e0bf` removed
it** with the driving-mission v1.1 refactor, which moved `R4` onto the mission
tracker's `delta_s`. `mission/tracker.py::project` documents itself as projecting
"without a jump envelope or clamp", and no production call site passes
`max_s_jump_m`. So there is nothing to promote from preference to gate: the fix
is to *re-introduce* a bound, and the factor would become a real constant it is
not today. Filed as `C52`.

**Removing the negative clip is not a one-line change.** `_canonicalize_bounded`
(`reward/scalarization.py`) range-checks the progress margin to `[-1, +1]` and
raises `ScalarizationEvaluationError` outside it; the vector schemas are named
`rulebook_v2_macro_v4` and `rulebook_v5_1_six_level_v1`, and boundedness is the
property SCAL-V1.1 through V1.4 rest their admissibility arguments on. An
unbounded negative `Δq` does not merely make the reward unbounded below — it
violates the margin contract at the scalarization boundary and would abort the
step.

**§5.4 is untouched by a negative `Δq`, as the working notes said.** Verified in
`_validate_six_level_weights`: the condition is
`a^(4-k) > (1+σ)·Σ(lower) + flat·|lower| + tail` with `tail` built from
`DELTA_Q_MAX = 1`, the maximum *positive* progress. A negative increment of any
magnitude cannot enter it.

### A — re-introduce a continuity bound at the mission tracker

`MissionTracker.project` supplies `max_s_jump_m = k · v_max · Δt`.

Keeps every contract: `Δq` stays in `[-1, +1]`, no schema change, no §5.4 change,
no rulebook level, weight or observation field.

Costs a new code constant, one call-site change, and the semantics for "no
candidate is plausible", which is the hard part and has only three exits. Failing
closed is what ADR-035 rejected, with a measured reason: it broke three transition
tests that legitimately move the ego further than the bound in one synthetic step,
and it converts a rare geometry into a lost episode. Clamping the station inside
the same call corrupts the geometry that call also returns — `tangent_xy` and
`lateral_distance_m` would describe a point the ego is not at, which `wrongway`
and the corridor diagnostics read. Abandoning the bound is today's behaviour.

**And at `k = 2` it does not close the ratchet.** The cursor could then move
`4.44 m` in a step while `Δq` saturates at `2.2222 m`, so a retreating step banks
one channel unit and keeps banking it: the g6 loop still nets about `+18.8` per
lap instead of `+36`. Halving an unbounded gain is not closing it.

### A′ — bound the station cursor at exactly one clip width

The same change with `k = 1`, which is a different proposition, because then
`|Δs| ≤ D_REF` and `clip(Δs/D_REF, -1, +1)` **is the identity**. The clip stops
being a mechanism at all: `Σ Δq = (s_T - s_0)/D_REF` exactly, the telescoping
identity `AC-RB5.1-07` asserts becomes true by construction rather than
approximately, `C51`'s deficit and `C50`'s surplus both vanish, and no closed
trajectory can bank progress — algebraically, for any geometry, not as a property
of a population. It **removes** a mechanism rather than adding one, which is the
direction this work is supposed to go.

ADR-035 rejected `k = 1` as one that "would deprioritize legitimate motion". That
was an argument without a measurement, and the measurement above contradicts it
at the scale that matters: the largest one-step station advance the index offers
an on-route ego is `1.028 · D_REF`, so `k = 1` truncates one route's one step by
6.2 cm and nothing else.

The cost is real and it is elsewhere. The station would no longer be the
arc-length of the nearest point but of a bounded cursor, so on a curve-cutting
trajectory it lags — `route_completion`, `remaining_distance_m` and the
observation's station anchor all move with it. That is a change to what the
mission station *means*, so it is a `DRIVING-MISSION-V1.1` amendment and a user
decision, not an implementation choice. The lag is zero for a policy that stays
near the centerline and grows with lateral offset; the table above bounds it.

**And the approved specification already rejected this shape.**
`DRIVING-MISSION-V1.1` §8 lists
**"continuity/clamp/freeze/recovery/accumulated-travel/HMM protocols"** among the
obsolete elements to remove, and a station cursor bounded per step is a
continuity protocol by any reading. So `A′` does not merely redefine the station:
it re-introduces a mechanism that document withdrew. That is worth knowing before
"the clip becomes the identity" decides anything — the argument for `A′` is strong
on mechanism, and it is arguing against an approved removal rather than into a
gap.

### B — remove the negative clip

Closes the ratchet algebraically too — `Σ Δq ≤ (s_T - s_0)/D_REF` with equality
lost only on the forward side, so a closed loop can never pay positively.
**Rejected on contract grounds**: it breaks the bounded-margin range check above,
which is a SCAL-V1.4 amendment reaching every scalarization mode, and it leaves
the forward deficit in place. It buys less than `A′` for a larger amendment.

### C — leave the reward alone and make the standing constraint enforceable

The audit that the exec plan §11.1 has required since 2026-09-05 now exists and
comes back clean. Wiring it into the frozen-index validation path is what turns
`C50` from a latent unbounded exposure into a population property with an
executable guard — which is what the standing constraint asked for and never had.

### Recommendation

**`C` now, `A′` if a future index fails the audit, `B` rejected.**

The exposure is not armed on the panels the thesis trains and evaluates on, and
that is measured rather than argued: 0.247 m of fold excess at the reach the
exploit needs. Every candidate that closes it algebraically spends a
specification amendment — SCAL-V1.4's bounded margin for `B`, the station's
meaning in DRIVING-MISSION-V1.1 for `A′`. Spending one to close an exposure the
measurement says is not armed is the wrong trade while the constraint that would
detect a future one is now executable.

If `C50` is to be closed algebraically rather than by population property, `A′`
is the candidate: it is the only one that removes a mechanism instead of adding
one, it makes an acceptance criterion exact instead of approximate, and its
central objection — that `k = 1` deprioritizes legitimate motion — is the one
this audit measured and found to be worth 6.2 cm. But it is a **re-introduction
of something `DRIVING-MISSION-V1.1` §8 removed**, not a gap-filling change, and
that is the cost to weigh against the elegance rather than after it.

## `D14`: the backwards route walk, executed

`D14` has recorded since 2026-09-08 that the carriageway-coverage measurement
"covers the goal cross-section only, so it does **not** show that an ego could
drive such surface *while banking the `R4` budget* — that needs the backwards
route walk, which is designed but not run". It has now been run.

**It is the same instrument, not a new one.** `--walk-spacing-m` calls the
builder's own cross-section decomposition at stations along the route instead of
only at the goal, so the walk inherits the reconciliation against the frozen gate
that instrument already performs — a record whose host component disagrees with
the frozen segment is refused before any of it is measured. 3,500 records walked,
**zero unusable stations**.

**What it looks for.** Starting at the goal cross-section, a *corridor* is
same-direction drivable surface outside the route's own carriageway, at least one
ego width wide, containing a position from which the ego misses the finite final
gate. The walk then follows it backwards while consecutive cross-sections still
offer an overlapping component. `offroad` is zero on it by construction, because
the drivable surface is the union of every vertically compatible lane, and
`wrong_carriageway` is zero because it charges only opposing surface. What the
corridor costs to *enter* is reported separately, as the gap of non-drivable
surface between it and the route's own carriageway.

**Reported in entry-gap bands rather than against a threshold**, for the reason
the near-revisit audit reports proximity bands: a cut-off here would be a choice
of how much `offroad` an ego may pay to enter, which is a parameter this work is
not allowed to add. Figures below are the 1 m walk; the 5 m walk is the
spacing control.

| entry gap | records | share of index | corridor p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| ≤ 0.05 m — map representation only | 94 | 2.7 % | 2.0 m | 56 m | 93 m |
| ≤ 0.50 m | 165 | 4.7 % | 12.0 m | 68 m | 93 m |
| **≤ one ego width (1.852 m)** | **318** | **9.1 %** | **15.0 m** | **56 m** | **158.6 m** |
| any | 1266 | 36.2 % | 28.0 m | 143 m | 535 m |

The `any` row is the one to distrust: its median corridor sits at 55 m of lateral
offset across a median 43 m of non-drivable surface, so it is mostly other
streets that happen to run parallel somewhere inside the ±100 m cross-section.
That is the same remoteness the 2026-09-08 run reported at the goal.

**The band that matters is the third, and it establishes what `D14` said was
not established.** In it the median corridor offset is 10.70 m — about three
lanes — and exactly **1 of 318** corridors is part of the assigned route, so
these are not the ego's own carriageway seen twice.

* **130 records, 3.7 % of the index, have such a corridor covering at least half
  the mission.** All 130 are Waymo; by split, 67 train, 43 test, 14 validation,
  so it is present in evaluation and not only in training.
* **Two records have one covering the whole route**: 158.6 m at an offset of
  −26.0 m and 139.7 m at +22.0 m, entered across gaps of 0.888 m and 0.872 m.
* In channel units the corridor is worth `length / D_REF`: median **6.75**,
  maximum **71.36**, against a mean mission of **40.58**. On the worst record an
  ego can bank more than a whole average mission's progress on a legal parallel
  street and arrive outside the gate.

**Spacing control.** A coarse walk could bridge a gap shorter than its spacing
and over-report. Refining 5 m to 1 m moves the figures the *other* way: the band
grows from 303 to 318 records and the at-least-half subset from 124 to 130, with
120 of the 124 kept, 4 lost and 10 gained, a median length ratio of 1.000 and an
unchanged maximum of 158.6 m. The 5 m figure was therefore a mild under-report,
not an artefact of bridging.

**A preliminary reading of this measurement was wrong and is withdrawn.** On the
first 60 records every corridor was `LANE_SURFACE_UNSTRUCTURE` — the interior of
a PG junction — and a station-by-station trace of one of them showed the corridor
ending because the surface ended, 15 m back, where the route leaves the junction
and becomes a 6.4 m street. The conclusion drawn from that, that the parallel
surface is junction interior with nothing to bank on, holds for **PG and not for
Waymo**: over the full index 1,137 of 1,266 corridors are `LANE_SURFACE_STREET`
and the at-least-half subset is entirely Waymo. The first 60 records are sorted
by `scenario_uid` and were PG-heavy, and a sample chosen by sort order is not a
sample.

**What this does not show.** Geometry, not behaviour: whether a trained policy
goes there is what `route_fully_outside_max_run` answers, and no
`eval_episodes.csv` exists on disk. The continuity test is a lower bound in two
ways — a component must fit an ego width at every sampled station, and
consecutive components must overlap — so the true exposure is at least this.

**The remedy remains a user decision and a specification amendment, and the
hierarchy is not what blocks it.** An earlier version of this paragraph argued
that a sub-rule below progress provably cannot oppose it — which `D14`
establishes by weight arithmetic and is true — and then that A7 "leaves nothing
below L4 at all", so the only remedy shape left was the one that does not work.
**That second step is wrong and the A7 ExecPlan session was right to dispute
it.** The remedy does not want to sit below progress; it wants to sit above it,
and A7 is the architecture that puts it there. Under the shipped six-level order
the natural home for an off-corridor rule is L5, relaxable lane compliance, which
is *below* L4: the off-route trajectory banks the larger L4 total, wins there, and
L5 is never consulted. Under A7 the same channel is `K4`, *above* `K5`, so in the
ordered arms an in-corridor trajectory has `K4 = 0` exactly and wins before
progress is compared at all — the same mechanism as the collide-to-escape result.
A7 does not shrink the remedy space; it makes visible that the space was already
empty under the shipped order.

**What blocks the remedy is the specification, in two clauses rather than one.**
`D14`'s own exit (b) is attenuating positive `Δq` outside the route corridor, and
`DRIVING-MISSION-V1.1` §1 says the mission is "not … a way to put legality,
heading, or lateral offset into `R4`". A `K4` rule avoids that clause by not
touching `R4` — but any such rule needs a lateral envelope at runtime, and §8
lists **"runtime authority of any final lateral envelope"** among the elements to
remove, beside off-route `R4` zeroing, approved after `DEC-EF-06` took the
measure-then-decide option. So both shapes are foreclosed by the same approved
document, and unforeclosing either is a user decision.

**All three remedy shapes are foreclosed by one sentence, which is the finding.**
`driving_mission_v1.1_specification.md:104` is a single list of obsolete elements,
and it contains, in order, *"continuity/clamp/freeze/recovery/accumulated-travel/
HMM protocols"*, *"off-route `R4` zeroing"*, and *"runtime authority of any final
lateral envelope"*. The first forecloses `C50`'s cursor bound, the second `D14`'s
exit (b), the third the `K4` shape. So the position is not that no remedy has been
found: **every shape of remedy that would work was withdrawn by one approved
document**, and what it leaves is measured at 3.7 % of the index. `AGENTS.md`'s
Scientific Argument Standards has the case exactly — when the only argument
against something is the specification, that is a finding, and usually the
specification is the thing to fix. Framing adjudicated with the A7 ExecPlan
session, which read the clause independently.

**And the scalar arm cannot be fixed this way at any admissible weight.** Priced
by the A7 session and **verified independently here** from the A7 block and
`_validate_six_level_weights` rather than relayed. A violated `K4` step costs
`w₅·(1 + σ)` and a step of progress earns `λ₄·Δq`, so the crossover is
`w₅ = λ₄·Δq / (1 + σ)`, and §5.4 caps `w₅` at `(a − λ₄·ΔQ_MAX) / (1 + σ)` because
`K3` must dominate everything below it.

| | value |
|---|---:|
| §5.4 cap on `w₅` | **0.384615** |
| crossover at the expert's mean pace (`Δq = 0.204089`) | **0.313983** — admissible |
| crossover at the clip (`Δq = 1`) | **1.538462** — inadmissible |

So a weight exists that opposes a *slow* off-route drive and none that opposes a
*fast* one, which is the wrong way round, since speed is what banks the exposure
sooner. At the recommended `w₅ = 0.15` a fully violated step costs 0.195 against
2.000 of progress at the clip, so it opposes nothing at any pace.

**The factor of four is algebraic, not numerical.** The ratio of the clip-pace
requirement to the cap is `λ₄ / (a − λ₄·ΔQ_MAX)`, in which `σ` cancels: at
`a = 2.5, λ₄ = 2.0` it is exactly 4, and it stays 4 for every admissible `σ`. The
gap is therefore a property of how close `λ₄` sits to `a`, not of a severity
choice — which is worth knowing, because it means no re-tuning of `σ` reaches it.

**And that denominator is the §5.4 tail**, the whole budget left below the
non-negotiable level once progress has taken its maximum step. So the identity
reads `ratio = λ₄ / tail = 0.80 / 0.20`, and *"no admissible `w₅` opposes a fast
off-route drive, by a factor of four"* is the same statement as *"progress
consumes four fifths of the per-step budget"* — the review's own headroom
diagnosis, in behavioural units instead of weight-allocation ones. It also names
the only thing that would move it: not `σ`, not `w₅`, but `λ₄` relative to `a`.
That makes a `λ₄` change the one remedy in this whole family that §8 does **not**
foreclose, because it is a weight decision rather than a specification one — the
A7 ExecPlan session prices it at about 1.1 pp of below-standstill per 0.5 of
`λ₄`, which is their figure and not re-derived here. Neither session proposes it;
`λ₄ = 2.0` stands. Reading contributed by that session on top of the identity.

*Independent corroboration of the relayed pace*: 40.58 channel units per mean
mission at `Δq = 0.204089` implies 198.8 steps, against a median episode of 199.
That figure comes from the `C51` row and not from the bench that produced the
pace, so the two agree without sharing a source.

That makes this a third measured asymmetry in the same direction as the collision
one: the ordered arms can express a constraint no admissible scalarization can.

What this measurement changes is that the decision now has its magnitude:
3.7 % of missions, up to a whole mission's worth of progress banked off the
assigned route.

## What this did not verify

* **No trajectory-level evidence.** Every figure here is a property of the frozen
  route geometry. Whether a trained policy goes where the exposure lives is what
  `route_fully_outside_max_run` would answer, and no `eval_episodes.csv` exists on
  disk. `C51`'s signed statistic is now the detector, but it has not been run over
  a panel since the change.
* **The off-route reach in the `C50` bands is not shown to be drivable.** Those
  bands are labelled by lateral excursion, not by whether an ego may legally sit
  there. `D14`'s walk below answers the adjacent question — whether legal
  same-direction surface runs *beside* the route — but not this one, which is
  whether the specific surface between two folds of one route is drivable.
* **`A′`'s station lag is bounded, not measured on a trajectory.** The table
  bounds the per-step truncation from geometry; the accumulated lag over an
  episode depends on the policy's lateral behaviour.
* **ADR-035's claim is not amended.** It is approved, and correcting a claim
  inside it is a user decision; `C52` records what is wrong with it.
