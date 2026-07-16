# ExecPlan Requirements

An ExecPlan is the living implementation record for a non-trivial feature,
integration, behavioral change, or refactor. Store it as
`docs/implementation/<feature>_<spec-version>_exec_plan.md`.

The plan must be understandable to a contributor who knows the repository but
not the task. Keep it current; preserve important discoveries instead of
rewriting history.

## Required Structure

### 1. Metadata

Record:

- feature and stable plan ID;
- authoritative specification path, ID, version, and approval status;
- status: `DRAFT`, `AWAITING_DECISIONS`, `APPROVED`, `IN_PROGRESS`, `BLOCKED`,
  `IMPLEMENTED`, or `VERIFIED`;
- creation and last-update dates;
- branch, related ADRs, and owner when applicable.

`VERIFIED` is allowed only after mandatory validation and final reconciliation.

### 2. Objective And Scope

Describe the observable capability, why it is needed, how success is recognized,
in-scope work, out-of-scope work, and compatibility constraints. Cover public
APIs, configuration, checkpoints, datasets, logs, metrics, and migration when
relevant.

### 3. Authoritative Requirements

Summarize without redefining the specification:

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Observable required behavior | Exact section or anchor |

Every in-scope requirement needs a stable ID. An exclusion requires recorded
approval.

### 4. Current Repository Analysis

Before production changes, record exact paths and symbols for current code,
call flow, configuration, tests, interfaces, behavior to preserve, directly
relevant debt, and dependency constraints. Label statements as `SPECIFIED`,
`VERIFIED`, `INFERRED`, or `AWAITING_CONFIRMATION`. An inference that affects
behavior is an approval gate.

### 5. Assumptions And Invariants

Record applicable units, coordinate frames, shapes, ranges, masks, timing,
state/reset behavior, termination/truncation semantics, seeds, and versions.
State how each item was established and how violations are handled.

### 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Clarification | Question | A / B | A | Behavior and tests | Awaiting approval |

Use `implementation detail`, `specification clarification`, `specification
deviation`, or `blocking technical issue`. Do not start dependent work while a
gate is unresolved. Link approved material decisions to an ADR.

### 7. Proposed Design

Describe affected modules, interfaces, data and state flow, configuration,
errors, fallbacks, logging, integration boundaries, and realistic alternatives.
Internal choices must not silently change the specification.

### 8. Traceability

Maintain this throughout implementation:

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `src/...` | `tests/...::test_...` | Planned |

### 9. Test Strategy Defined Before Implementation

Define observable acceptance criteria with stable `AC-*` IDs, then freeze the
minimum mandatory matrix before production changes:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Required behavior | Deterministic fixture | Exact result | `REQ-001` |

Cover the applicable nominal, boundary, invalid, missing-data, mask/padding,
state/reset, termination, truncation, determinism, numerical, compatibility,
causality, integration, smoke, and regression cases.

After approval, tests may be added or strengthened. Mandatory tests may not be
deleted, weakened, skipped, marked expected-to-fail, or have expected behavior
changed without approval. Every discovered bug requires a regression test.

List exact commands already supported by the repository for focused tests,
regressions, integration, smoke, lint, formatting, and type checking. Mark an
unconfigured category as unavailable; do not invent a command.

### 10. Milestones

For each independently verifiable milestone, record:

- objective and status;
- expected files;
- implementation tasks;
- tests and commands;
- completion evidence;
- decision dependencies.

Use checkboxes and keep them synchronized with actual progress.

### 11. Progress And Findings Log

Append dated entries containing completed work, commands and results, findings,
decisions needed, and next step. For unexpected problems record evidence,
affected requirements, severity, consequence, proposed resolution, approval
need, and final resolution.

### 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|

If empty, write `No deviations identified.`

### 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/...` | Planned modification | Requirement implemented |

Keep the table aligned with the final diff.

### 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `...` | `PASS`, `FAIL`, or `NOT_RUN` | `YYYY-MM-DD` | Summary |

Never record a pass without execution. For failure or `NOT_RUN`, include cause,
remaining risk, and follow-up.

### 15. Final Reconciliation

For every requirement and acceptance criterion, record `IMPLEMENTED`,
`VERIFIED`, `PARTIAL`, `NOT_IMPLEMENTED`, or `NOT_APPLICABLE`, explaining every
status other than both implemented and verified.

Separate known limitations, deferred required work, and optional improvements.
Conclude with resulting behavior, architecture, compatibility, executed tests,
approved decisions, deviations, limitations, and readiness for experimental use.
