# Specification: Feature Name

> Delete instructional placeholders only after completing the section. Use
> `Not applicable` when appropriate. Do not mark the specification approved while
> a material decision remains open.

## Metadata

- Feature: `<name>`
- Specification ID: `<stable ID>`
- Version: `<version>`
- Status: `DRAFT | UNDER_REVIEW | APPROVED | SUPERSEDED`
- Date: `<YYYY-MM-DD>`
- Supersedes: `<path and version or NONE>`
- Related specifications: `<paths and versions or NONE>`
- Related ADRs: `<paths or NONE>`
- Authoritative: `YES | NO`

## 1. Purpose And Context

Define the externally meaningful capability, research role, upstream and
downstream components, scientific sources, project adaptations, original project
choices, and dependency-imposed constraints. Do not present an adaptation as a
direct result from literature.

## 2. Scope

### In Scope

- `<required behavior>`

### Out Of Scope

- `<excluded behavior>`

### Optional Or Deferred

- `<non-required behavior>`

## 3. Terminology, Assumptions, And Preconditions

Define relevant symbols, units, coordinate frames, masks, categories, and time
conventions. For each assumption state its source, validation, and failure
behavior.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `<name>` | | | | | | `YES/NO` |

List prohibited future, privileged, leaked, offline-only, curriculum-generated,
evaluator-generated, and diagnostic-only information.

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `<name>` | | | | | |

## 6. Functional Requirements

### REQ-001: Requirement Title

- Required observable behavior: `<precise behavior>`
- Applicability: `<conditions>`
- Invariants: `<properties>`
- Edge and missing-data cases: `<behavior>`
- Failure or fallback behavior: `<explicit result>`
- Interactions: `<related requirements/components>`

Avoid undefined terms such as “appropriate”, “reasonable”, or “nearby”. Repeat
for every requirement.

## 7. Mathematical And Algorithmic Contract

Define formulas, domains, units, normalization, clipping, tolerances,
singularities, execution order, temporal aggregation, terminal/truncated cases,
and missing-data behavior. Add pseudocode when equations are insufficient.

## 8. Applicability, State, And Timing

Define required data and reliability, applicability detection, fallbacks and
their logging/training eligibility, state ownership, initialization, update
order, reset, history, timers, control frequency, episode boundaries,
serialization, and deterministic replay. Distinguish termination from time-limit
truncation.

## 9. Configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen for experiments |
|---|---|---|---|---|---|---|
| `<name>` | | | | | `YES/NO` | `YES/NO` |

Define validation and invalid-value behavior. Changing a frozen value requires a
new approved specification version or an approved ADR.

## 10. Errors, Logging, And Diagnostics

Define recoverable and fatal behavior for invalid input/configuration, shape or
mask errors, missing dependencies/data, numerical anomalies, simulator states,
and serialization failures. Define required logs for fallbacks, applicability,
experiments, and metrics.

Classify exposed values as policy observation, training signal, diagnostic-only,
evaluation metric, or reproducibility metadata.

## 11. Reproducibility And Compatibility

Define seed ownership, deterministic limits, serialized configuration, dataset
and split identity, dependency/simulator/specification versions, stored
artifacts, and effects on APIs, configs, spaces, datasets, replay buffers,
checkpoints, evaluation, and previous experiments. Define migration when needed.

## 12. Acceptance Criteria

### AC-001: Criterion Title

- Given: `<preconditions and input>`
- When: `<operation>`
- Then: `<objective expected result>`
- Related requirements: `REQ-...`

Acceptance criteria must test observable behavior rather than private structure.

## 13. Required Validation Categories

Mark each as required or `Not applicable`:

- nominal and boundary behavior;
- invalid and incomplete inputs;
- masks, padding, state, reset, and update order;
- termination and truncation;
- deterministic seeds and reproducibility;
- numerical stability, NaN, and infinity;
- compatibility and migration;
- absence of future and privileged information;
- upstream, downstream, and end-to-end integration;
- regressions for known bugs.

Exact test files, fixtures, and available commands belong in the ExecPlan.

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-001` | `AC-001` | `<source or ADR>` |

## 15. Open Decisions And Limitations

| ID | Question | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|
| `DEC-001` | | | | | `OPEN/APPROVED/REJECTED` |

An open material decision blocks approval. Separate intentional limitations from
missing required behavior.

## 16. References

Identify the exact concept supported by each paper, official document, approved
decision, ADR, related specification, or verified dependency constraint.

## 17. Approval Record

- Approved by: `<name or role>`
- Approval date: `<YYYY-MM-DD>`
- Approval evidence: `<user decision reference>`
- Approval notes: `<notes or NONE>`
- Repository path: `<final path>`
- Project index updated: `YES | NO`
