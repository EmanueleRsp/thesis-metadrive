# ADR-001: ScenarioNet v1.1 Dataset And Runtime Policy

- Status: `Approved`
- Date: `2026-07-16`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-16`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version `1.1`
- Affected ExecPlans: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

ScenarioNet v1 artifacts and planning contained incompatible policies for split
grouping, arm balancing, Rulebook eligibility, the episode tail, and the ACL
semantic-arm interface. The checked-out ScenarioNet converter preserves the
Waymo TFRecord source file as provenance but does not expose a verified log or
segment identifier. The approved v1.1 specification must therefore freeze the
experimental policy before production reconciliation begins.

## Decision

1. Use the existing six semantic ScenarioNet arms A0–A5 unchanged for dataset
   classification and the ScenarioNet ACL MAB (`K=6`).
2. Use `source_log_id` or `segment_id` as a Waymo split group only when it
   proves a shared source group. Otherwise treat `source_file`/TFRecord shard
   as provenance and group by original `scenario_id`.
3. Require `rulebook_eligible=true`, hard quality validity, and the configured
   signal-reliability policy for every selected runtime split record. Preserve
   excluded records and reasons in the audit catalog.
4. Set the custom time limit to `scenario.length + 50` control steps for both
   Waymo and PG, preserving separate termination and truncation semantics.
5. Acquire Waymo incrementally in batches of 16 unseen shards, with a cap of
   128 newly processed shards. At the cap, fail with the recorded deficit
   report rather than relaxing filters or quotas.
6. Limit this specification's causality acceptance to the ScenarioNet pipeline
   boundary. The full semantic-observation contract remains deferred to its
   dedicated specification.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Treat every TFRecord shard as a split group | Conservative provenance grouping | Usually incompatible with exact v1.1 split targets and not evidence of log overlap | The converter field is provenance, not a verified log/segment ID |
| Rename or remap semantic arms for v1.1 | Could align labels with a new taxonomy | Breaks catalog, config, test, and ACL compatibility; remapping would change semantics | Existing six-arm taxonomy is retained |
| Admit `partial` or `missing` Rulebook records into runtime pools | Retains more candidates | Violates the Rulebook fail-closed eligibility contract | Runtime pools require `rulebook_eligible=true` |
| Choose the episode tail after a visual pilot | Can react to observed behavior | Leaves termination and bootstrap behavior unfrozen | `+50` is frozen and matches the ScenarioNet experimental convention |

## Consequences

The v1.1 dataset is a reproducible, balanced experimental benchmark rather
than a direct continuation of the v1 artifact. Existing code, catalogs,
manifests, reports, and tests require reconciliation. Historical generator ACL
arms remain a separate legacy mechanism and are not the ScenarioNet semantic
MAB arm space. A final Waymo dataset may fail explicitly if the configured cap
cannot satisfy all hard constraints.

## Validation And Traceability

- Requirements: `REQ-SN-001` through `REQ-SN-015` in the v1.1 ExecPlan.
- Acceptance criteria: `AC-SN-001`, `AC-SN-003`, `AC-SN-005`, `AC-SN-007`,
  `AC-SN-009`, `AC-SN-011`, and `AC-SN-013`.
- Mandatory tests: `TEST-SN-001` through `TEST-SN-015`, including grouping,
  eligibility, arm identity, time limit, cap/deficit, and leakage-boundary
  cases.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message in this Codex conversation:
  “Approvo la specifica ScenarioNet Integration v1.1.”
- Notes: the approval includes the six review decisions recorded in the
  approved specification.
