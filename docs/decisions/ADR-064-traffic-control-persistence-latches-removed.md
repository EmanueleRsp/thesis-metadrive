# ADR-064: Traffic-control persistence latches are removed from `crosswalk` and `vehicle_yield`

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.4 (amends `rulebook_v4.7_specification.md` §7.4).

## Context

`components/controls.py:374` and `:485` both compute

```
cost = 1.0 if active_latch_for_zone else approach_cost
```

Once the ego is inside the zone the cost is pinned at 1.0 until the zone is
left. Attribution over 217,189 expert transitions:

| branch | steps |
|---|---:|
| `vehicle_yield:latch` | 353 |
| `vehicle_yield:approach` | 55 |
| `crosswalk:latch` | 11 |
| `crosswalk:approach` | 0 |
| latch share | **86.9 %** |

Removing the latches alone moves the expert's mean episode return from -203.35
to -201.42.

## Decision

The latch branch is removed. The memoryless approach term is retained.

## Rationale

The change is **not** justified by its magnitude: it is worth about two points
of two hundred. It is justified on two other grounds.

**Observability.** The latch is a memory the agent cannot see. Non-Markovian
reward specification in the literature - NMRDPs (Bacchus, Boutilier & Grove,
1996) and reward machines (Toro Icarte et al., ICML 2018 / JAIR 2022) - always
exposes the automaton state to the agent. A hidden latch makes the reward
non-Markovian in the agent's own state space, which is exactly the constraint
the supervisor raised.

**Credit assignment.** Inside the zone no action changes the cost. The agent is
charged repeatedly for a decision already taken, with no gradient toward any
better behaviour. The approach term prices that same decision while it is still
being made, which is where the learning signal belongs.

## Alternatives rejected

1. **Expose the latch in the observation** (Test B branch 2). Admissible in
   principle, but the latch is unbounded in duration - it persists for as long
   as the ego occupies the zone - so it is not a bounded, plausible perception
   feature, and the approach term already covers the decision.
2. **Keep it, given the small cost.** Rejected: the defect is structural, and
   the same reasoning that removes it is what keeps `dashed_line` (ADR-066),
   whose memory *is* bounded and *is* already observable.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
