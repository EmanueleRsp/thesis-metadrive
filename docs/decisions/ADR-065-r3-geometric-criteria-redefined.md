# ADR-065: `offroad`, `solid_line` and `wrong_carriageway` are redefined geometrically

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.4 (amends `rulebook_v4.7_specification.md` §7.2 and the §7.3-bis added by
  v4.10).
- Resolves: `DEC-RSEC-008` (off-road geometry).

## Context

Measured on 1100 logged Waymo `train` records:

| sub-rule | production | redefined |
|---|---:|---:|
| `offroad` | 1.190 % of steps, mean cost 0.097 | **0.593 %**, mean 0.080 |
| `solid_line` | 1.120 %, cost **binary 1.000** | **0.349 %**, mean 0.328 |
| `wrong_carriageway` | 0.843 %, mean 0.066 | **0.000 %** (0 of 217,189) |

Without these three redefinitions, 18.09 % of expert episodes still score below
standstill; with them, 10.55 %. They are load-bearing, not polish.

## Decision

- `offroad`: the drivable surface is widened by **0.3 m** before the outside-area
  fraction is taken.
- `solid_line`: the cost becomes **graded lateral penetration** with tolerance
  0.3, replacing the binary 1.0 charged on any contact; detection buffers the
  marking by its real half width, 0.075 m, instead of production's 1 cm
  numerical epsilon.
- `wrong_carriageway`: applies only when the ego **centroid** lies inside the
  exclusive opposing surface. The invaded-area fraction is otherwise unchanged.

## Rationale

**0.3 m is nuPlan's own `drivable_area_compliance` tolerance**, and it exists
for the same reason here: the oriented bounding box over-approximates the
vehicle. This is a measurement artifact, not permissiveness about leaving the
road. For a 1.85 m wide vehicle the 0.3 penetration tolerance on solid lines
corresponds to approximately the same 0.3 m of lateral slack, so the two are one
decision rather than two independent calibrations. The tolerance was **swept**
(0.1 / 0.2 / 0.3 / 0.4 giving 0.599 / 0.430 / 0.349 / 0.249 % of steps) and the
sweep is reported, rather than a single value being chosen and presented as
given.

**A binary solid-line cost cannot distinguish grazing from crossing.** At cost
1.000 on any contact, an ego clipping the marking with a bumper corner for one
centimetre is charged exactly as much as one straddling it. R3 outranks R4, so
both behaviours were taught the same penalty.

**`wrong_carriageway` at exactly zero expert violations** is the sharpest single
result of the campaign. The expert never places its centre on the opposing
carriageway; production's any-overlap criterion was measuring bounding-box
corners clipping the opposing surface in curves. That is geometric noise, not a
normative event, and the rule was pricing normal cornering.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
