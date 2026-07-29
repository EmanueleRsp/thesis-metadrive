# ADR-033: Rulebook Latch Exclusion from the Policy Observation

- Status: Approved
- Date: 2026-07-29
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-29
- Affected specifications:
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`
    (SS7.8 interaction-token feature list; SS8.3 traffic-control ranking)
  - `docs/specifications/observation_v1.2_specification.md`, ID `OBS-V1.2`
    (SS1, SS12 — the prohibition this record confirms)
  - `docs/specifications/observation_v1.3_specification.md`, ID `OBS-V1.3`
    (to be drafted; carries the resulting field list)
- Affected ExecPlan:
  `docs/implementation/semantic_observation_causal_correctness_v1.3_exec_plan.md`
  (`DEC-005`, `DEC-009`)

## Context

Two approved specifications disagree about whether Rulebook-owned state may
reach the policy.

`OBS-V1.1` SS7.8 lists `Incompatible entry latched` among the 35 features of the
conflict-interaction token, and SS8.3 criterion 3 ("non ancora risolto") ranks
traffic controls by whether the Rulebook has already resolved them.

`OBS-V1.2` SS1 states that the observation "does not contain a future simulator
state, Rulebook output, or Rulebook-owned temporal latch", and its acceptance
criterion in SS12 requires that "no Rulebook timer/latch/result is
policy-visible". SS5 keeps the OBS-V1.1 field definitions normative "unless this
document explicitly overrides their source, frame, category, or eligibility
semantics", which SS1 and SS12 do — but the field list in SS7.8 was never
amended, so the contradiction survived into the implementation.

`CausalSemanticBatchBuilder._build_interactions`
(`src/thesis_rl/envs/observations/causal_semantic.py:1303`-`:1307`) follows
OBS-V1.1: it reads `preexisting_ego_occupancy_zone_ids`,
`vehicle_yield_illegal_entries` and `crosswalk_illegal_entries` from
`CausalSceneContext.memory`, which is a `RulebookMemory`
(`src/thesis_rl/contracts/causal_scene_context.py:21`). The method is inherited
unchanged by the OBS-V1.2 builder, which does not override it.

A static review on 2026-07-29 additionally established that the illegal-entry
read has never been functional. Both latch sets are keyed `(actor_id, zone_id)`
(`src/thesis_rl/rulebook/v2/components/controls.py:435`, consumed as
`for actor_id, zone_id in ...` at `src/thesis_rl/rulebook/v2/transition.py:495`
and `:727`), while the observation constructs `pair = (zone_id, actor.actor_id)`
(`causal_semantic.py:1255`). The membership test can succeed only when
`zone_id == actor_id`, so the feature is constant `0.0` at runtime. The
`preexisting_ego_occupancy_zone_ids` read uses a plain `frozenset[str]` and does
succeed.

## Decision

1. `OBS-V1.2` SS1/SS12 prevail. `Incompatible entry latched` is removed from the
   interaction token. `OBS-V1.3` will not list it.
2. `Preexisting occupancy active` is retained, but reconstructed inside the
   observation builder from current zone geometry plus builder-owned memory
   recorded at the first step the zone becomes a candidate. It is no longer read
   from `RulebookMemory`.
3. `OBS-V1.1` SS8.3 criterion 3 is permanently dropped from the traffic-control
   ranking. It was implementable only by reading
   `context.memory.resolved_signal_group_ids` /
   `resolved_stop_group_ids` (`causal_semantic.py:1062`-`:1065`).
4. The Rulebook keeps every internal latch and timer unchanged. This record
   constrains the observation only.

## Rationale

The reward signal is produced by the Rulebook. A policy input computed by that
same Rulebook from the same event that generates the penalty makes two
hypotheses experimentally indistinguishable once the agent starts avoiding the
penalty: that it learned the intended behaviour (not entering a conflict zone a
vulnerable road user is committed to), or that it learned to react to the flag.
This is label leakage, and it makes the thesis claim "the agent learned
rule-compliant driving from perception" unfalsifiable rather than merely weak.

The latch is also set *after* the incompatible entry, so it cannot support
prevention. Its only usable role is reacting to a violation already committed,
which is precisely the shortcut the exclusion is meant to prevent.

Retaining `Preexisting occupancy active` in reconstructed form is not an
inconsistency: its content is causally lawful (a zone the ego already occupied
before the event became attributable to the policy) and derivable from geometry
the observation already has. Only its source was illegitimate.

The dropped ranking criterion is not reconstructed builder-side because doing so
would rebuild a rule-aligned notion of "resolved" for a marginal ordering
benefit, reintroducing the same class of coupling for a much smaller payoff.

## Consequences

- The interaction token loses one feature; with the roundabout-relation
  duplicate also removed, its width goes from 35 to 33.
- No completed experiment is invalidated by the illegal-entry removal, because
  the channel was inert. The behavioural delta of this ADR is therefore confined
  to the pre-existing-occupancy reconstruction and the control ranking.
- The tuple-order defect is *not* fixed. Correcting it would activate a leak
  this record forbids; the surrounding code is removed instead.
- A regression test asserts that an observation built from a context with a
  populated `RulebookMemory` is bit-identical to one built from the same context
  with an empty memory.
- `OBS-V1.3` must restate the interaction-token field list and the
  traffic-control ranking, and record this ADR as the reason OBS-V1.1 SS7.8 and
  SS8.3 are not carried forward unchanged.

## Alternatives considered

**Amend OBS-V1.2 SS12 to admit this specific latch, and fix the tuple order so
the feature works.** Rejected: it would make the leak real for the first time,
for a feature that cannot support prevention, at the cost of the falsifiability
of the central experimental claim.

**Keep the feature but leave it inert.** Rejected: the coupling to
`RulebookMemory` would remain in the source, where the tuple-order mismatch
reads as an ordinary bug. Any future contributor correcting it would silently
activate the leak, with no test to catch it.

**Reconstruct the "resolved control" ranking criterion from ego-owned memory.**
Rejected as disproportionate; see Rationale.
