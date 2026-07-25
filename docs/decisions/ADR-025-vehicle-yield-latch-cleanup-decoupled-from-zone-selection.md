# ADR-025: Vehicle-Yield Illegal-Entry Latch Cleanup Decoupled From Zone Selection

- Status: Approved
- Date: 2026-07-24
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-24
- Supersedes: NONE
- Affected specifications: `docs/specifications/rulebook_v4.7_specification.md`,
  version 4.7-final-implementation-complete, authoritative (no change to
  §2.8.2 itself)
- Affected ExecPlans:
  `docs/implementation/vehicle_yield_illegal_entry_latch_cleanup_exec_plan.md`

## Context

`select_first_ahead_or_occupied_zone`
(`src/thesis_rl/rulebook/v2/geometry/conflict_zones.py:345`) implements §2.8.2
of the rulebook v4.7 specification faithfully: it selects the ego-occupied
component, otherwise the first component ahead within
`SIGNED_DISTANCE_EPSILON_M = 5.0e-2` m of the ego front curvilinear
coordinate.

For any conflict-zone component of curvilinear exit `s_exit` and an ego
footprint of length `L` (a real vehicle, `L` on the order of metres):

- the "ahead" predicate becomes false once `front_s > s_exit + epsilon`
  (`epsilon = 0.05` m);
- the ego footprint stops intersecting the component only once
  `front_s > s_exit + L`.

Because `L` is always orders of magnitude larger than `epsilon` for any real
vehicle, the "ahead" predicate is already false for metres of ego travel by
the time `ego_occupied` genuinely transitions from `True` to `False` for that
component. At that exact transition frame, `select_first_ahead_or_occupied_zone`
returns `None` for that component (assuming no other candidate is currently
occupied or ahead).

`_vehicle_yield_inputs` (`src/thesis_rl/rulebook/v2/transition.py:514`) treats
"no selected zone" as the sentinel domain
(`zone_id = "__no_vehicle_priority__"`, `ego_occupied = False`). The
illegal-entry latch cleanup performed by the pure evaluator
`evaluate_vehicle_yield` (`src/thesis_rl/rulebook/v2/components/controls.py:365`)
only ever clears entries whose `zone_id` matches the *currently selected*
zone (`entry[1] != zone_id` at lines 402-403 and 448-449). Once the real
zone drops out of selection at the same frame `ego_occupied` becomes `False`
for it, the cleanup is invoked with the sentinel `zone_id` instead, and the
real `(actor_id, zone_id)` latch entry in
`memory.vehicle_yield_illegal_entries` is never removed for the remainder of
the episode.

This is not a synthetic-test artifact: it is a geometric certainty for every
vehicle-yield conflict-zone crossing in the live pipeline, confirmed by
inequality analysis (`epsilon = 0.05` m versus `L` on the order of metres),
independent of simulation step size. The stale latch entry has two
consequences beyond the internal component memory:

1. `evaluate_vehicle_yield`'s own cost formula
   (`src/thesis_rl/rulebook/v2/components/controls.py:458`) treats any
   surviving `(actor_id, zone_id)` entry as a maximal-cost illegal occupancy
   should ego ever occupy that same `zone_id` again (e.g. a revisited or
   cyclic route segment), independent of whether a genuine new conflict
   exists.
2. `src/thesis_rl/envs/observations/causal_semantic.py:1263-1266` exposes
   `pair in context.memory.vehicle_yield_illegal_entries` as a policy
   observation feature. With the latch never cleared, this feature reports an
   active illegal entry for the rest of the episode after a legitimate,
   completed zone exit, corrupting a live policy input rather than only an
   internal cost computation.

## Decision

Decouple the vehicle-yield illegal-entry latch cleanup from the single
zone selected for this step's cost evaluation. In `_vehicle_yield_inputs`,
before constructing either the sentinel `empty` domain or the full domain
dict, compute a cleared view of `memory.vehicle_yield_illegal_entries`://
for every `(actor_id, zone_id)` entry, look up `zone_id` in
`cache.conflict_zones` (populated for every zone ever selected via
`CacheDelta.new_conflict_zones`/`apply_cache_delta`) and drop the entry if
the post-state ego footprint no longer intersects that zone's cached
polygon. Use this cleared set — not the raw memory field — as
`previous_illegal_entries` in every return path of `_vehicle_yield_inputs`.

This is a wiring-layer (transition/memory-management) change only:

- `select_first_ahead_or_occupied_zone` and §2.8.2 are unchanged; the
  currently selected zone for this step's cost/approach computation is
  exactly as specified.
- `evaluate_vehicle_yield`'s pure contract, signature, and existing unit
  tests are unchanged; its own per-current-zone cleanup remains and is now
  redundant-but-harmless for the currently selected zone.
- Only the previously-accumulated latch state supplied to the evaluator is
  corrected so it no longer contains entries for zones the ego footprint has
  demonstrably and independently left.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Extend the "ahead" predicate in §2.8.2 to also select zones with an active latch | Single selection mechanism keeps handling everything | Changes the specification's §2.8.2 selection criterion itself; requires a formal specification amendment, not just an ADR; risks altering which zone is reported as "selected" for cost/observation purposes during the post-exit window | Rejected: broader observable impact than the bug requires, and touches an approved specification clause |
| Leave latch cleanup coupled to selection, add a periodic "sweep" component pass | Centralizes cleanup in the component evaluation step | Requires a new component-level side channel outside the existing per-zone evaluator contract; more invasive than a memory pre-filter | Rejected: higher complexity for the same observable fix |
| Do nothing now, document only | No implementation risk this session | Confirmed corruption of a live policy observation feature for the remainder of affected episodes | Rejected: user explicitly approved implementing the wiring-layer fix in this session |

## Consequences

- `memory.vehicle_yield_illegal_entries` no longer retains stale entries for
  zones the ego footprint has left, once `cache.conflict_zones` has recorded
  that zone's geometry (guaranteed by the time an entry can exist, since an
  entry is only ever added for a zone that was itself selected and thus
  cached).
- The policy observation feature at `causal_semantic.py:1263-1266` reflects
  genuinely active illegal entries only, correcting a source of persistent
  spurious signal for episodes with revisited or long-lived vehicle-yield
  zones.
- No policy input shape, reward-vector shape, or public interface changes;
  only the temporal accuracy of an existing boolean feature changes.
- Any training run that observed the stale-latch behaviour before this fix
  saw a superset of "illegal entry" signal versus the corrected behaviour;
  this is a bug fix, not a new experimental condition, and does not require
  ADR-023 or DEC-005 revision.

## Validation And Traceability

- Affected requirement: `REQ-001` in
  `docs/implementation/vehicle_yield_illegal_entry_latch_cleanup_exec_plan.md`.
- Mandatory regression test: an end-to-end `evaluate_transition` sequence
  (not only the pure `evaluate_vehicle_yield` unit tests) driving ego through
  and completely past a vehicle-yield conflict zone, asserting
  `next_memory.vehicle_yield_illegal_entries` becomes empty once the ego
  footprint has left the zone by more than `SIGNED_DISTANCE_EPSILON_M`.
- Existing pure-function tests in `tests/test_rulebook_v2_vehicle_yield.py`
  remain valid and unmodified.

## Approval Record

- Approved by: user
- Approval evidence: explicit selection of "Pulizia latch disaccoppiata
  (Recommended)" when asked how to proceed with the fix, in the session on
  2026-07-24, after the geometric analysis, the policy-observation impact,
  and the alternatives were presented.
- Notes: NONE
