# ADR-071: at-fault classification for R1, with truncation instead of termination

- Status: **Approved — not yet implemented**
- Date: 2026-08-11
- Approval evidence: user instruction "Sì, procedi con ADR-071", 2026-08-12,
  after the feasibility analysis and risk list below were presented.
- Affected specifications: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.1, §9; `docs/specifications/scenarionet_integration_v1.4_specification.md`
  (episode contract). Coordinates with ADR-058, which settled the tail side of
  the same contract.
- Related: ADR-070 applied the same blame principle to R2 only, and its
  follow-up section names this gap.

## Context

R1 has no fault attribution. `evaluate_collision_impact`
(`src/thesis_rl/rulebook/v2/components/collision.py:55`) charges every contact
onset the MAIS3+F injury-risk probability, described in its own docstring as
"a function of the pre-state normal closing speed and the actor class alone".
Whoever caused the contact is not consulted.

Consequently a stopped ego struck from behind pays **full cost at the highest
priority level** — with `a = 2.2`, weight 10.648, which by RULEBOOK-V5.0 §11.8
is 52 steps of typical expert progress, and 3.8 such steps cancel an entire
episode's R4. `crash_vehicle_done: true` also terminates the episode.

This is the canonical case the RSS literature uses to introduce the notion of
blame — being struck while waiting at a red light, where prevention is
impossible — and it is the level at which nuPlan actually applies at-fault
logic, excluding `STOPPED_EGO_COLLISION` and `ACTIVE_REAR_COLLISION` from the
at-fault set (ADR-070 quotes `_get_collision_type`).

The direction of the resulting pressure is the opposite of the degeneracy this
project has been chasing: punishing an unavoidable event teaches the agent to
avoid *legitimate* stops — at lights, at crossings — which would be a new
degeneracy rather than a fix for the old one.

## Feasibility

Every input nuPlan's classifier needs is already present at the evaluation
point. `evaluate_collision_impact` receives `pre_ego: ActorSnapshot`,
`pre_actors_by_id: Mapping[str, ActorSnapshot]` and `onset_records`, where each
`ContactOnsetRecord` carries `actor_id` and `actor_class`:

| nuPlan input | available here |
|---|---|
| `ego_state.speed <= 5e-02` | `pre_ego.velocity_xy` |
| `is_track_stopped(other)` | `pre_actors_by_id[actor_id]` |
| `is_agent_behind(rear_axle, other_center)` | ego position and heading |
| front-bumper line intersection | ego footprint, already used by `clearance`/`offroad` |
| "footprint not fully in a single lane" | `associate_route_lane` + `cache.route_lanes` — the only input to add |

The classification is **memoryless** (all functions of the pre-state; it adds no
memory beyond the contact tracking that already exists) and **observable**
(other agents' speeds and relative positions are in the observation), so it
passes both falsification tests of §2.

## Proposed decision

1. Classify each contact onset with nuPlan's taxonomy. At fault:
   `ACTIVE_FRONT_COLLISION`, `STOPPED_TRACK_COLLISION`, and
   `ACTIVE_LATERAL_COLLISION` when the ego footprint is not fully within a single
   lane or lane connector. Not at fault: `STOPPED_EGO_COLLISION`,
   `ACTIVE_REAR_COLLISION`.
2. **At-fault contact** → R1 charged as today, episode **terminated**
   (bootstrap `V = 0`).
3. **Not-at-fault contact** → **no R1 cost**, episode **truncated**
   (bootstrap from `V(s)`), and the event reported as a diagnostic.

## Why truncation is the load-bearing half

Zeroing the cost while keeping termination would be worse than the status quo:
the agent would still be punished for the event, through the zero bootstrap, and
would additionally have learned that provoking one is free. The two halves are
not separable.

The obvious objection to any blame-based rule is that the agent learns to
*provoke* a not-at-fault collision — brake hard in front of a follower, get
struck, pay nothing. Truncation closes this by construction: a truncated episode
returns the agent its own expected continuation value, i.e. exactly what it would
have obtained by continuing to drive. Provoking the impact buys nothing.

This is the standard treatment of "the episode ended, but not because the agent
failed", i.e. partial-episode bootstrapping (Pardo, Tavakoli, Levdik,
Kormushev, *Time Limits in Reinforcement Learning*, ICML 2018), and it is the
termination/truncation distinction this repository already commits to
preserving.

## Risks and caveats

1. **Residual second-order incentive.** Cashing out at `V(s)` also avoids future
   risk. Under the current policy `V(s)` already prices that risk, so the trade
   is neutral in expectation; it becomes exploitable only if the critic
   overestimates. That is a function-approximation concern, not a structural
   hole, but it must be stated rather than omitted.
2. **Not validatable offline.** R1 is `NOT_MEASURED` in the replay, which has no
   physics contacts, so Test A cannot reach this. It weighs less here than
   elsewhere: no threshold is being calibrated from data, a published
   classification is being implemented. It is deterministically testable with
   fixtures — stopped ego + agent from behind → not at fault; moving ego +
   stopped track → at fault.
3. **The lateral case is the fragile one.** "Footprint not fully within a single
   lane or lane connector" requires footprint-in-lane containment. It is
   computable from data already in the cache but has the most surface for error
   of the five branches.
4. **It changes the episode contract.** When an episode terminates versus
   truncates is a convention this repository requires explicit approval to
   change, and it must be coordinated with ADR-058.
5. **Provocation is only neutralised in expectation.** An agent whose value
   function is poorly calibrated early in training may still find the truncation
   branch attractive. This argues for reporting the not-at-fault rate as a
   training diagnostic, not for abandoning the design.

## Status

**Approved, not yet implemented.** The user approved both halves together — the
classification and the truncation — since adopting the first alone is the worse
outcome described above. RULEBOOK-V5.0 §5.1 still reads "Unchanged" and no
measurement in that document is affected, because R1 is `NOT_MEASURED` offline
either way; the specification text, the episode-contract coordination with
ADR-058, and the implementation (with the fixture tests from risk 2) remain to
be done as a separate milestone, acceptance-test-first per the required
workflow.
