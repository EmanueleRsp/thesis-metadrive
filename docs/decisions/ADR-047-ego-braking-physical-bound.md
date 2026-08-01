# ADR-047: Ego RSS braking bound raised to the physical tyre-road limit

- Status: Approved
- Date: 2026-08-01
- Approval evidence: explicit user approval of
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
  (`DEC-RBCOST-002`) in this conversation, following
  `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md` (F3a).
- Affected specification: `rulebook_v4.7_specification.md` §6.2.4 step 10 and
  the §6.2 parameter table (amended by `rulebook_v4.10_specification.md`).

## Context

`calibrate_ego_braking` (`rulebook/v2/calibration.py`) capped the calibrated
ego braking value at `MAX_REFERENCE_BRAKE_MPS2 = 4.0 m/s²`. A 12-trial
re-measurement of the normative one-shot protocol
(`docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/braking_trials_sample_12.json`)
found mean decelerations of `10.73..16.92 m/s²`, so the cap discarded a factor
of `2.7` from the measured ego capability. The cap was also internally
inconsistent with `FRONT_MAX_BRAKE_MPS2 = 8.0 m/s²` (`components/rss.py`),
which the same specification already assumes for the identical vehicle class
used as the RSS front actor: the model assumed the ego braked half as well as
surrounding traffic of its own type.

With `b_e = 4.0`, RSS `d_safe` at equal speeds was `28.28 m` at `10 m/s` and
`65.78 m` at `20 m/s`, so a routine 2 s car-following headway scored
`q_rss ≈ 0.29..0.39` at every speed — a near-permanent R2 violation in any
traffic.

## Decision

Replace the bound with `MAX_REFERENCE_BRAKE_MPS2 = 8.0 m/s²`, the dry-asphalt
tyre-road deceleration limit and the same value already assumed for
surrounding vehicles. With the measured `b_meas ≈ 10.7 m/s²` this yields
`ego_min_brake_mps2 = 8.0`. The calibration formula itself
(`floor(10 * lower_5th_percentile) / 10`) is unchanged; only the upper bound
moves.

The persisted calibration artifact validates `cap_mps2` against this constant
on both write and load, so an artifact produced under the old bound is
rejected fail-closed rather than silently accepted with a stale cap. The ego
config (and therefore `config_hash`) is untouched, so the frozen ScenarioNet
selection index remains valid; only the calibration artifact needs
regeneration (`make rulebook-v2-collect-trials && make rulebook-v2-calibrate
&& make rulebook-v2-validate-calibration`).

## Consequences

`d_safe` halves at every speed pair (e.g. `16.89 m` at `10/10 m/s`,
`31.27 m` at `20/20 m/s`), removing the near-permanent car-following
violation. `d_req` (signal approach) and `d_stop` (crosswalk, vehicle-yield),
which reuse the same calibrated brake value, shrink by the same factor and
become physically accurate instead of twice too conservative. Every run using
a calibration artifact produced before this change must regenerate it; the
production artifact at `$DATA_ROOT/scenarionet/rulebook_v2/calibration_b_e.json`
was regenerated as part of this change
(`ego_min_brake_mps2 = 8.0`, `config_hash` unchanged).
