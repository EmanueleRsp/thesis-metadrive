# ADR-066: `wrongway` is deleted; `dashed_line` is retained unchanged

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`. The user
  raised both questions directly: whether `dashed_line` could be made observable
  rather than removed, and whether the earlier fixes still applied.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.5, §5.6.
- Supersedes: `ADR-060` (unified memoryless wrong-direction sub-rule). The
  correct action is deletion, not reformulation.

## Context

**`wrongway`** fires on **1 step in 217,189** logged expert transitions, with a
cost of 0.003. Reverse motion on the assigned route is already covered by
`wrong_carriageway` (direction relative to the carriageway) and by R4, whose
margin is negative for negative route advance.

**`dashed_line`** computes `cost = penetration x time_factor`, where the time
factor ramps from 0 at `DASHED_T0_S = 1.0 s` to 1 at `DASHED_TCAP_S = 2.0 s` of
continuous contact with the same marking. It fires on 0.530 % of expert steps
with mean cost 0.219, roughly 3.8 reward points per episode against a gross of
40. It carries memory: `previous_boundary_id` and `previous_timer_s`.

## Decision

- `wrongway` is removed from the registry and from aggregation.
- `dashed_line` is retained, unchanged, with no observation amendment.

## Rationale

**`wrongway`.** An empirically inert, semantically duplicated sub-rule adds
specification surface without adding specification. `ADR-060` reformulated it to
be memoryless, which fixed the observability objection but left the redundancy
and the emptiness untouched.

**`dashed_line`.** The memory exists for a sound reason: a lane change
legitimately crosses a dashed marking, while sustained straddling does not. It
is therefore a Test B branch-2 rule - bounded memory (it saturates at 2 s) whose
state is a plausible mid-perception feature, since "how long have I been
overlapping this marking" follows from lane-level localisation that every
production autonomous-driving stack performs.

The branch is **already satisfied, at zero cost**: both observation paths carry
21 steps of history, and the 21 was derived from `DASHED_TCAP_S` itself.
`src/thesis_rl/envs/observations/stacked_lidar_v2.py` states this in its module
docstring, and `conf/obs/semantic_v3.yaml` sets `context_history_length: 21`. At
the 10 Hz control period 21 samples span 2.1 s, so the timer saturates strictly
inside the observation window and is reconstructible from it. No schema change,
no dimension change, no observation ADR.

Sustained straddling of a lane marking was an explicit supervisor requirement,
and it is met without weakening the Markov property.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
