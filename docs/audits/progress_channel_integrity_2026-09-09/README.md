# Progress-channel integrity, 2026-09-09

Continues `docs/audits/rulebook_architecture_2026-09-09/`, whose §8 opened the
three defects of the mission-progress channel. This directory holds what closed
them and the decision brief for the one that did not.

**Status.** `V3` and `C51` are closed, `C49`'s guard half is closed and its `γ`
half is not, `C50` is a decision. No approved behaviour changed: the reward, the
projection and the mission station are exactly what they were.

| file | what it holds |
|---|---|
| `README.md` | this: the `V3` result, and the `C50` decision brief with its three candidates priced |
| `run_2026-09-09.txt` | the `V3` audit's output over the current frozen index, verbatim |

The instrument itself is `scripts/audit_route_near_revisits.py`, with acceptance
tests in `tests/test_route_near_revisit_audit.py`. The per-route JSON is 4 MB and
is regenerated with `--output`, so it is not committed.

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
this audit measured and found to be worth 6.2 cm.

## What this did not verify

* **No trajectory-level evidence.** Every figure here is a property of the frozen
  route geometry. Whether a trained policy goes where the exposure lives is what
  `route_fully_outside_max_run` would answer, and no `eval_episodes.csv` exists on
  disk. `C51`'s signed statistic is now the detector, but it has not been run over
  a panel since the change.
* **The off-route reach is not shown to be drivable.** The bands above are
  labelled by lateral excursion, not by whether an ego may legally sit there;
  that is `D14`, and the backwards route walk it needs is still designed and
  unrun.
* **`A′`'s station lag is bounded, not measured on a trajectory.** The table
  bounds the per-step truncation from geometry; the accumulated lag over an
  episode depends on the policy's lateral behaviour.
* **ADR-035's claim is not amended.** It is approved, and correcting a claim
  inside it is a user decision; `C52` records what is wrong with it.
