# Specification: Unified direction sub-rule and bounded control-reference route polyline

## Metadata

- Feature: `rulebook_v2_wrong_direction_and_control_reference_polyline`
- Specification ID: `rulebook-v2-wrong-direction-control-reference`
- Version: `4.13`
- Status: `APPROVED`
- Date: `2026-08-09`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version
  `4.7-final-implementation-complete`, for §7.3 only, and its §2.9.5/§2.9.6
  only in the respect stated in §4 below. It also supersedes
  `docs/specifications/rulebook_v4.12_specification.md` in full (that document's
  entire subject, the `wrongway` cost deadband, is absorbed here) and
  `docs/specifications/rulebook_v4.10_specification.md` §7.3-bis
  (`wrong_carriageway` becomes a component of the unified sub-rule rather than a
  sub-rule of its own; its formula is unchanged). Every other section of v4.7
  and of the v4.8–v4.12 amendments remains authoritative and unchanged.
- Extends: `docs/specifications/rulebook_v4.11_specification.md` (route-successor
  extension of the control-relevance predicate)
- Related ADRs: `docs/decisions/ADR-060-unified-memoryless-wrong-direction-subrule.md`,
  `docs/decisions/ADR-061-bounded-control-reference-route-polyline.md`;
  supersedes `ADR-050` and `ADR-056`
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
- Approval evidence: explicit user approval on 2026-08-09 in the originating
  conversation, for the unified direction sub-rule (*"Vai con la decisione
  consigliata"*), for its memoryless reformulation after the Markov
  observability constraint was raised, and for the traffic-control coverage fix
  (*"Ok"*).
- Authoritative: `YES` for §3 and §4 below; not authoritative for anything else.

## 1. Purpose And Context

This amendment resolves two independent defects that surfaced together while
diagnosing why a trained policy stands still and only moves once the logged
scenario ends.

**Direction sub-rules.** v4.7 §7.3 (`wrongway`) charges reverse motion above a
`0.1 m/s` deadband, normalized by the ego speed cap `v_max = 22.2 m/s`. That
deadband is a physics-solver noise floor introduced by `ADR-050` and extended to
the cost by `ADR-056`; it was never a legal tolerance. The rule therefore charges
any real reverse manoeuvre — a legal act — as a road-rule violation, while its
normalization makes full cost require reversing at 80 km/h, which never occurs.
v4.10 §7.3-bis added `wrong_carriageway` as a separate positional sub-rule; the
two are complementary, not redundant, but the reference closed-loop benchmarks
expose a single direction metric, and because §2.4 aggregates R3 by `max` the
separation carries no numeric consequence.

**Traffic-control coverage.** §2.9.6 derives a control's `route_s_m` as a
coordinate on the canonical assigned-route polyline, and discards the control
when its control line does not intersect that polyline. For fixed-window
recordings the assigned route reflects the human driver's actually-recorded
path, which legitimately stops short of a junction. `ADR-051`/v4.11 relaxed the
*selection* predicate for a single unambiguous successor and raised the
working-signal rate from 42.9% to 74.8% on the 828 signalised Waymo records of
the frozen catalog; it did not relax the *construction* geometry. The measured
residue is 52 records (6.3%) with no control constructed and 157 (19%) still
excluded by the approach filter — 209/828, **25.2%**, of signalised scenarios in
which the Rulebook never selects a signal.

## 2. Notation

Notation, units, coordinate frames, tolerances and the applicability/evaluability
vocabulary are those of v4.7 §2 and are unchanged. `v_∥(t)` is the ego velocity
projected on the unit tangent returned by the canonical projection of the ego
centre on the `RoutePolyline`, oriented along the route order, exactly as in
v4.7 §7.3.1. `L` denotes the length of the canonical assigned-route polyline.

## 3. Amended §7.3 — `wrong_direction`

### 3.1 Sub-rule

R3 exposes exactly one direction sub-rule, `wrong_direction`, replacing both
`wrongway` (v4.7 §7.3) and `wrong_carriageway` (v4.10 §7.3-bis) as separately
registered components:

$$
q_{\mathrm{wrong\_direction}}(t)=\max\!\big(q_{\mathrm{rev}}(t),\,q_{\mathrm{carr}}(t)\big).
$$

### 3.2 Kinematic component

$$
q_{\mathrm{rev}}(t)=
\operatorname{clip}\!\left(
\frac{[-v_{\parallel}(t)]_+ - v_{\mathrm{tol}}}{v_{\mathrm{viol}} - v_{\mathrm{tol}}},
\,0,\,1\right),
\qquad
v_{\mathrm{tol}}=2.0\ \mathrm{m/s},\quad
v_{\mathrm{viol}}=6.0\ \mathrm{m/s}.
$$

Behavior:

- ego at rest, or moving forward: `0`;
- reverse motion up to `2.0 m/s`: `0` (manoeuvre allowance);
- reverse motion at `4.0 m/s`: `0.5`;
- reverse motion at or above `6.0 m/s`: `1`.

`q_rev(t)` is a function of the current state only. No `RulebookMemory` field
may be introduced for it, and none of `v_max`, the previous route station, or
any history window may enter the formula.

### 3.3 Positional component

`q_carr(t)` is v4.10 §7.3-bis's `wrong_carriageway` area fraction, unchanged in
formula, applicability and evaluability. It remains memoryless.

### 3.4 Parameters and provenance

| Parameter | Value | Provenance |
|---|---:|---|
| `v_tol` | `2.0 m/s` | nuPlan *Driving Direction Compliance* compliant bound (2 m of reverse displacement per 1 s), reinterpreted per §3.5 |
| `v_viol` | `6.0 m/s` | nuPlan *Driving Direction Compliance* violation bound (6 m per 1 s), reinterpreted per §3.5 |

`WRONGWAY_SPEED_EPSILON_MPS` and the separate status epsilon of `ADR-050` are
removed. `v_tol` is twenty times the `0.1 m/s` physics-noise floor those patches
addressed, so it subsumes them.

`status` is `VIOLATED` iff `q_wrong_direction > 0`, following v4.12's
simplification.

### 3.5 Declared deviation from the source metric

nuPlan's thresholds are *displacements over a 1 s window*; §3.2 applies them to
*instantaneous speed*. The two coincide for sustained constant-speed reverse
travel and diverge for a brief high-speed burst — 0.3 s at 5 m/s is 1.5 m,
compliant under nuPlan and partially charged here.

The reinterpretation is mandatory, not cosmetic. A 1 s signed-displacement
window is not reconstructible from the policy's observation: the only block
carrying signed route-relative velocity is 5 steps deep (0.5 s) and the 21-step
context block carries unsigned speed. A cost defined on that window would depend
on information the policy does not observe. This document must therefore never
be cited as implementing the nuPlan metric; it implements a Markov-observable
adaptation of its anchors.

### 3.6 Limitation

The reference tangent is the ego's own canonical route, which follows
source-declared legal lane direction by `DRIVING-MISSION-V1.1.1` §6. Where the
ego is far from its route the tangent is no longer the local legal direction and
the sub-rule's interpretation degrades. This limitation is inherited from v4.7
§7.3.1 and is unchanged.

## 4. Amended §2.9.5/§2.9.6 — control reference polyline

### 4.1 Definition

A **control reference polyline** is defined for each scenario and used
**exclusively** as the route argument of the §2.9.6 control-line derivation:

> Begin with the canonical assigned-route polyline. While the terminal lane has
> exactly one successor and the cumulative extension beyond `L` remains below the
> bounded distance `D`, append that successor's centerline.

### 4.2 Prefix-identity invariant

> The control reference polyline is point-identical to the canonical
> assigned-route polyline on `[0, L]`.

This invariant is normative. It is what keeps a control's `route_s_m` and the
ego's front-bumper station in one curvilinear frame; violating it reintroduces
the frame mismatch that `REQ-EF-05` removed, in which a control carried a
coordinate from a different frame and appeared already passed. An implementation
must assert it.

### 4.3 Scope

Only §2.9.6's control-line derivation consumes this polyline. R4 progress,
mission completion, the final gate, §7.3's reference tangent, and every other
route consumer continue to use the canonical assigned-route polyline unchanged.

### 4.4 Behavior beyond the canonical route end

A control with `route_s_m > L` remains selectable, because the ego's front-bumper
station saturates at `L` and the control therefore always satisfies the
"ahead" predicate. It can never register a swept-bumper crossing, so only the
graded approach cost of §7.6 applies. This is the intended behavior for a
mission whose goal is a stop line: an ego approaching a red light too fast is
charged, and an ego stopped at the line is not (`d_req = 0` yields cost `0`).

### 4.5 Ambiguity

A terminal lane with more than one successor stops the extension. Ambiguity is
never resolved by guessing, exactly as in v4.11.

### 4.6 `D`

`D` is a bounded distance in metres, not a hop count. Its value is fixed from
the coverage measurement recorded in the ExecPlan — the recovered fraction of
the 157 residual records at each candidate budget — and not chosen a priori.
A proximity-only criterion remains forbidden by v4.7 §2.9.5.

## 5. Acceptance criteria

- `q_rev(1.9) = 0`, `q_rev(4.0) = 0.5`, `q_rev(6.0) = 1`, `q_rev(8.0) = 1`.
- Two states identical in the present but differing in history produce the same
  `wrong_direction` cost, and `RulebookMemory` gains no field.
- R3's component set contains `wrong_direction` and neither `wrongway` nor
  `wrong_carriageway` as separately aggregated entries; both remain present in
  diagnostics.
- The control reference polyline and the canonical route polyline are
  point-identical on `[0, L]`.
- `route_s_m` is unchanged for every control already constructible under v4.11.
- The fraction of `has_route_traffic_light` Waymo records in the frozen catalog
  with zero selectable `SIGNAL` control is strictly below the 25.2% baseline.

## 6. Out of scope

The R2 cost scale, the episode horizon, the evaluation reporting contract, the
terminal reward channel and the R4 normalizer are not addressed here. They are
covered by `ADR-058`, `ADR-059`, and the open `DEC-RSEC-001` in the ExecPlan.
