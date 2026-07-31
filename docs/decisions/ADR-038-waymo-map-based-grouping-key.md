# ADR-038: Waymo Map-Based Grouping Key, Adopted Unconditionally

- Status: Approved
- Date: 2026-07-31
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-31
- Supersedes: NONE
- Affected specifications:
  - `docs/specifications/scenarionet_integration_v1.2_specification.md`, ID
    `SCENARIONET-INTEGRATION`, version `1.2`, §3.2 (amends v1.1 §6.1)
- Affected ExecPlans:
  `docs/implementation/empirical_holdout_split_and_dual_test_panels_v1.2_exec_plan.md`
  (`DEC-002`, `M2`)

## Context

`SCENARIONET-INTEGRATION` v1.1 §6.1 specifies a group-id preference order
(`source_log_id` / `segment_id` / `source_file_id`, falling back to
`f"scenario:{uid}"` when no superior identifier demonstrably identifies a
shared log or segment). In the currently converted Waymo `training_20s`
pool, `source_log_id` always resolves to the TFRecord shard filename, which
the pipeline explicitly treats as unreliable provenance and discards
(`src/thesis_rl/scenarios/pipeline.py:53`), falling back to per-scenario
grouping. This conforms to v1.1 §6.1, which anticipated the absence of a
superior identifier, but means the Waymo grouping used for group-disjoint
splitting is, in practice, no grouping at all: two 20-second windows drawn
from the same road geometry can be assigned to different splits with no
mechanism preventing it.

An audit was started to measure how many map-identity groups in the
converted pool (54,104 scenarios) actually straddle the currently frozen
v1.1 train/test split, using a map-feature-polyline fingerprint. It was
deliberately stopped at 35,000/54,104 scenarios once it became clear that
the audit's result could not change the implementation decision: the
fingerprint needed to *measure* co-location is the same fingerprint needed
to *fix* it, so building the fix costs the same regardless of the measured
magnitude, and no possible measured value would justify skipping the fix
before constructing empirical, group-disjoint test/validation holdouts
(ADR-037).

## Decision

The Waymo grouping key gains a map-identity fallback, inserted between the
existing log/segment identifier and the per-scenario fallback:

```text
1. source_log_id / segment_id, if it demonstrably identifies a shared log or
   segment (not a TFRecord shard name)
2. map-identity fingerprint: a deterministic digest over the scenario's
   map_features polylines, spatially quantized
3. scenario_id, only if neither 1 nor 2 is available
```

This key is adopted **unconditionally** — not conditioned on the audit's
measured co-location rate. The exact fingerprint definition (fields used,
spatial quantization granularity, hash function) is fixed during `M2` of the
linked ExecPlan and recorded in the split manifest under
`grouping.waymo_fingerprint_evidence`, together with whatever partial or
complete co-location count the audit (or its resumed continuation) produces.
That count is retained only as descriptive evidence for the thesis
limitations section, not as an adoption gate.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Complete the co-location audit first, adopt the fix only if the measured rate exceeds a threshold | Quantified justification before code changes | The fix's cost does not depend on the measured rate (same fingerprint code either way); waiting only delays holdout construction for a number that cannot change the decision | Rejected: no decision-relevant information is gained by waiting |
| Defer to a separate ExecPlan, re-freeze the dataset a second time later | Smaller immediate scope | Risks two dataset freezes and two `selection_hash` changes instead of one, doubling the re-run cost when official experiments eventually depend on the frozen dataset | Rejected: adopt now, before the v1.2 freeze, per ADR-037 |
| Accept the per-scenario fallback and document it as a limitation | Zero implementation cost | Undermines the primary empirical Waymo endpoint this same session introduces (ADR-037); a documented-but-unaddressed leakage risk is a weaker basis for the thesis's generalization claim | Rejected |

## Consequences

- **Scientific validity**: removes a leakage risk that would otherwise
  apply specifically to the new empirical holdouts introduced by ADR-037 —
  without this fix, `test_waymo_empirical` results could be optimistic for
  scenarios geographically co-located with training scenarios.
- **Capacity**: may reduce usable Waymo group count relative to per-scenario
  grouping, since co-located scenarios collapse into one group for
  split-assignment purposes; the magnitude is unknown until `M2` completes
  (or the interrupted audit is resumed) and is not a blocker.
- **Compatibility**: changes the `grouping.waymo` field of the split
  manifest schema (`SCENARIONET-INTEGRATION` v1.2 §3.4); consumed only by
  the v1.2 re-split, already breaking by construction per ADR-037.
- **Migration**: none required; the v1.1 frozen dataset and its manifest are
  retained unchanged for historical traceability.

## Validation And Traceability

Affected requirement: `SCENARIONET-INTEGRATION` v1.2 `REQ-006`. Mandatory
test: `TEST-008` (map co-location grouping: two synthetic scenarios sharing
map identity receive the same group id; two with different map identities do
not) in the linked ExecPlan. New regression risk: the fingerprint function
must remain deterministic across pipeline re-runs, verified by `TEST-014`
(determinism/reproducibility).

## Approval Record

- Approved by: repository maintainer (explicit user approval)
- Approval evidence: user message "mi sembra vada tutto bene" in this
  conversation, following the decision recorded earlier in the same session
  to resolve `DEC-002` unconditionally rather than conditionally on audit
  completion
- Notes: the co-location audit that motivated this ADR was intentionally
  interrupted at 35,000/54,104 processed scenarios once its result was
  determined not to be decision-relevant; it may be resumed and completed
  during `M2` purely for the descriptive count referenced in
  `SCENARIONET-INTEGRATION` v1.2 §3.6, at the implementer's discretion.
