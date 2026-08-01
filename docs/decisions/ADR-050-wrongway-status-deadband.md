# ADR-050: Deadband on `wrongway` status classification

- Status: Approved
- Date: 2026-08-01
- Approval evidence: explicit user approval in this conversation ("procedi
  pure senza stare a rifare poi la smoke run"), following a user-reported
  visual flicker of the `wrongway` status marker (`[ok]`/`[!]`) on a
  stationary ego in
  `videos/final_eval/test_arm_stratified/eval_0007/episode_0001` (scenario
  `pg:scenarionet_v1:PGMap-24921253`), diagnosed by extracting and reading
  consecutive GIF frames (steps 91-106) and tracing the mechanism in
  `components/road.py:evaluate_wrongway`.
- Affected specification: none. `evaluate_wrongway`'s `cost` formula
  (`rulebook_v4.7_specification.md` §7.3.2) is unchanged; only the
  `RuleComponentResult.status` classification, a diagnostic field not
  covered by any specification formula, is amended.

## Context

`evaluate_wrongway` computed
`status = VIOLATED if cost > 0.0 else SATISFIED`, with
`cost = min(max(-longitudinal_speed, 0.0) / cap, 1.0)`. A stationary ego
retains a small residual velocity from the physics solver's contact
resolution even while at rest; when that residual has a hairline negative
longitudinal component, `cost` becomes a strictly positive but negligible
value (rounds to `0.00` at two decimals) and `status` flips to `VIOLATED`
with no tolerance. The user observed this as the `wrongway` status marker
alternating `[ok]`/`[!]` every few steps in the GIF overlay while the ego was
correctly parked, initially suspected to be the newly added
`wrong_carriageway` sub-rule (`ADR-049`) but confirmed by frame-by-frame
inspection to be the pre-existing `wrongway` component instead.

This is the same physics noise floor `RSS_STANDSTILL_SPEED_MPS = 0.1 m/s`
already addresses in `components/rss.py` (`ADR-048`), applied here to a
different component's status classification.

## Decision

Add `WRONGWAY_STATUS_SPEED_EPSILON_MPS = 0.1` (`components/road.py`) and
classify status as
`VIOLATED if -longitudinal_speed > WRONGWAY_STATUS_SPEED_EPSILON_MPS else
SATISFIED`. `cost` itself is untouched: it stays the same continuous
function of `longitudinal_speed`, so the scalarizer's input and every other
consumer of the numeric cost are unaffected. Only the boolean `status` field
— used today by the GIF overlay's `[ok]`/`[!]` marker
(`runtime/io/video_diagnostics.py`) and available to any future consumer —
gains the deadband.

## Consequences

The `wrongway` status marker no longer flickers under standstill physics
noise; a genuine reverse manoeuvre (longitudinal speed opposite the route by
more than `0.1 m/s`) is still classified `VIOLATED`. No behavioural change to
training, since `cost` (the only field the reward/scalarizer path consumes)
is unchanged. Regression test:
`test_wrongway_status_has_a_deadband_against_standstill_physics_noise`
(`tests/test_rulebook_v2_road.py`).
