# Specification: Zero episode tail after the logged scenario horizon

## Metadata

- Feature: `scenarionet_zero_episode_tail`
- Specification ID: `scenarionet-integration-zero-tail`
- Version: `1.4`
- Status: `APPROVED`
- Date: `2026-08-09`
- Supersedes: `docs/specifications/scenarionet_integration_v1.1_specification.md`
  §49 and §18.1 only, for the frozen value of `extra_steps_after_scenario`.
  Every other section of v1.1, and the whole of the v1.2 and v1.3 amendments
  (evaluation-pool cardinality and frozen subset artifacts), remains
  authoritative and unchanged.
- Related ADR: `docs/decisions/ADR-058-zero-episode-tail-after-logged-horizon.md`
- Related specification: `docs/specifications/evaluation_protocol_v1.3_specification.md`
  — this amendment is **not** independently releasable; see §4.
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
- Approval evidence: explicit user approval on 2026-08-09 in the originating
  conversation, under a condition the user set themselves (*"Se è supportata
  dalla letteratura va bene congelare gli attori e +10 di horizon come coda,
  altrimenti 0 coda"*), resolved against the literature as stated in §2.
- Authoritative: `YES` for §3 below; not authoritative for anything else.

## 1. Purpose And Context

v1.1 §49 freezes `extra_steps_after_scenario = 50` for every source. The value
was inherited from `scenarionet_integration_v1`, where the episode contract was
designed around visual replay rather than reinforcement learning. At
`Δt = 0.1 s` it grants 5 s beyond the exported scenario length — a 25% extension
for Waymo `training_20s` records of 198–200 steps.

The extension is not neutral. At `episode_step >= current_scenario_length` the
upstream scenario traffic manager clears every replay participant in a single
frame, this repository's source-bounded manager removes IDM vehicles as their
source validity ends, and the light manager freezes every signal at its last
state. The episode therefore continues in a world with no logged dynamic actors
and with permanently frozen signals.

Measured on the frozen catalog, 31.9% of Waymo missions are under 40 m, which at
~8 m/s is fully coverable inside that window (13.6% under 20 m, 44.9% under
60 m). An optimizer can complete roughly a third of the Waymo panel entirely in
the empty-world regime, which is the behavior observed in the `smoke-gpu` run.

The window is also unnecessary. Each mission's goal is the logged SDC's own final
position, which the human reached inside `scenario_length`; every mission is
therefore completable without any extension. The tail only ever supplied slack
for a policy slower than the expert.

## 2. Literature basis

The alternative considered — freezing the actors in place instead of deleting
them — has no precedent in the reference simulators and was rejected on that
ground:

- Waymax and V-Max simulate 9 s scenarios at 10 Hz with logged agents replaying
  for the log duration; the episode ends at the horizon or on a critical
  failure. No extension, no freezing.
- nuPlan's closed-loop simulation runs for the scenario duration.

Freezing would additionally invent post-log dynamics (a vehicle stopping
instantaneously mid-junction) and would charge the ego a full R1 collision for
striking a vehicle whose immobility is an artifact.

## 3. Amended contract

`extra_steps_after_scenario = 0` for every source and every run profile. The
only temporal truncation is the exported scenario length:

$$
\text{truncate} \iff \text{episode\_steps} \ge \text{scenario\_length}.
$$

The `episode_control.extra_steps_after_scenario` configuration key and the
`scenario_time_limit_reached` predicate are retained unchanged in form, so a
non-zero tail remains expressible and the contract stays explicit. Only the
frozen value changes.

Termination semantics are unchanged: collision, physical out-of-road and
final-gate success terminate; the time limit truncates.

## 4. Release constraint

This amendment must not be released without
`docs/specifications/evaluation_protocol_v1.3_specification.md`. With a zero
tail, a policy slower than the expert can fail to reach the geometric final gate
and, under the previous binary-success reporting, would be scored zero —
indistinguishable from a policy that never moved. v1.3 makes the continuous
`route_completion` the primary reported metric, which is what makes a zero tail
scientifically acceptable.

## 5. Acceptance criteria

- `scenario_time_limit_reached` truncates at exactly `scenario_length` with
  `extra_steps_after_scenario = 0`.
- No episode step executes after the logged replay participants have been
  removed.
- `make config` and `make config-gpu` remain valid.

## 6. Consequences

- Episode returns from any prior run are not comparable, because episode length
  changes for every scenario.
- The frozen-signal artifact disappears: no scenario can end with a permanently
  red light blocking a goal beyond the stop line, or a permanently green one
  granting free passage.
- PG records (length 501) lose the same 5 s. No PG mission is short enough for
  the window to have been exploitable (0.0% under 40 m), so the change is
  behaviourally inert for PG beyond the truncation itself.
