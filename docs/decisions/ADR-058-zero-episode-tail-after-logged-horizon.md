# ADR-058: Zero episode tail after the logged scenario horizon

- Status: Approved
- Date: 2026-08-09
- Approval evidence: explicit user approval in this conversation. The user set
  the acceptance condition themselves — *"Se è supportata dalla letteratura va
  bene congelare gli attori e +10 di horizon come coda, altrimenti 0 coda"* —
  and, once the literature review returned no precedent for freezing and a
  uniform precedent for ending at the logged horizon, confirmed
  `extra_steps_after_scenario = 0`.
- Affected specification:
  `docs/specifications/scenarionet_integration_v1.4_specification.md`
  (amends `scenarionet_integration_v1.1_specification.md` §49).
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
  (`DEC-RSEC-002`, `DEV-RSEC-001`).

## Context

`extra_steps_after_scenario` has been frozen at `50` for every source since
`scenarionet_integration_v1`, where it was introduced for visual replay rather
than for reinforcement learning. At `Δt = 0.1 s` it grants **5 s** beyond the
exported scenario length, which for Waymo `training_20s` records (198–200 steps)
is a 25% extension.

That window is not a neutral extension of the episode. At
`episode_step >= current_scenario_length` upstream MetaDrive clears **every**
replay participant in one frame (`manager/scenario_traffic_manager.py:110-139`,
the `else: replay_done = True` branch) and this repository's
`SourceBoundScenarioTrafficManager` removes IDM vehicles as their source
validity ends. Traffic lights do not disappear but freeze at their last state
(`manager/scenario_light_manager.py:68` returns early past the scenario length),
so a red light stays red forever and a green light becomes a permanent pass.

The consequence is an exploitable regime: the world empties while the episode
continues. Measured on the frozen catalog, **31.9%** of Waymo missions are
shorter than 40 m, which at ~8 m/s is fully coverable inside the 5 s window
(13.6% are under 20 m; 44.9% under 60 m). The observed `smoke-gpu` behavior —
the ego waits, the actors vanish, the ego then drives to the goal — is exactly
this regime being found by the optimizer.

Two clarifications matter for scoping this decision correctly:

- The agent does not *predict* the horizon. The observation carries no clock;
  the behavior is reactive, and the tail merely **reveals** a policy that the
  reward scale already makes optimal (see `ADR` for `DEC-RSEC-001` when taken).
  Removing the tail therefore removes an exploit, not the root cause.
- The tail is not needed for feasibility. Each mission's goal is the logged
  SDC's own final position, reached by the human inside `scenario_length`, so
  every mission is completable without any extension. The tail only ever
  provided slack for a policy slower than the expert.

Literature was consulted specifically because the alternative under
consideration — freezing the actors in place instead of deleting them — was an
invention of this conversation with no external basis. Waymax and V-Max run
9 s scenarios at 10 Hz with logged agents replaying for the log duration and
terminate at the horizon or on a critical failure; nuPlan's closed-loop
simulation likewise runs for the scenario duration. Neither extends the horizon
beyond the log, and neither freezes agents. Under the user's own stated
condition, that settles it.

## Decision

Set `extra_steps_after_scenario = 0` for every source. The only temporal
truncation is the exported scenario length itself.

`scenario_time_limit_reached` and the `episode_control` configuration key are
retained unchanged in form, so the contract remains explicit and a non-zero tail
stays expressible; only the frozen value changes.

The actor-freezing alternative is rejected. It has no precedent in the reference
simulators, it invents post-log dynamics (a vehicle stopping instantaneously
mid-junction), and it would charge the ego a full R1 collision for striking a
vehicle whose sudden immobility is an artifact.

## Consequences

- No episode contains a step in which the logged actors have been removed while
  the episode is still running. The empty-world regime ceases to exist rather
  than being shortened.
- The frozen-traffic-light artifact disappears with it: no scenario can end with
  a permanently red light blocking a goal beyond the stop line, or a permanently
  green one granting free passage.
- A policy slower than the expert can no longer reach the geometric final gate
  in the scenarios where it previously used the tail. This is acceptable **only
  in combination with** `ADR-059`, which makes the continuous
  `route_completion` the primary reported metric, so such a policy is scored on
  the progress it achieved instead of receiving zero. The two ADRs must be
  released together.
- Episode returns from any prior run are not comparable, because episode length
  changes for every scenario.
- PG records (length 501) are affected identically; they had the same tail, and
  no PG mission is short enough for the window to have been exploitable
  (0.0% under 40 m), so the change is behaviourally inert for PG apart from the
  5 s truncation.

Regression tests: `TEST-RSEC-003` (boundary of `scenario_time_limit_reached` at
`extra_steps=0`) and `TEST-RSEC-004` (no step executes with the world emptied),
per the ExecPlan's mandatory matrix.
