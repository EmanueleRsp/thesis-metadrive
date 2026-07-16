# Repository Documentation Restructure ExecPlan

## 1. Metadata

- Plan ID: `PLAN-DOCS-001`
- Feature: specification-driven documentation structure
- Authority: explicit user instructions and approvals in the 2026-07-16 conversation
- Status: `VERIFIED`
- Created: 2026-07-16
- Last updated: 2026-07-16
- Related ADRs: none

## 2. Objective And Scope

Separate existing documents by role so future work can reliably distinguish
approved specifications, living implementation plans, operational protocols,
architecture, workflows, and archives. Record the user's authority confirmations,
align Copilot guidance, and provide an ignored remote document inbox.

No production code, runtime configuration, dependency, scientific formula,
acceptance behavior, or existing test may change.

## 3. Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-DOC-001` | Approved scientific specifications have exact authoritative paths | User confirmation |
| `REQ-DOC-002` | ExecPlans are separate from specifications | Initialization requirements |
| `REQ-DOC-003` | Candidate protocols are not promoted implicitly | Authority policy |
| `REQ-DOC-004` | All repository links remain valid after migration | Documentation integrity |
| `REQ-DOC-005` | Copilot guidance defers to `AGENTS.md` and current authority/configuration | User approval |
| `REQ-DOC-006` | Remote uploads use a documented ignored inbox | User approval |

## 4. Current Repository Analysis

- `VERIFIED`: the former mixed documentation directory contained three approved
  specifications, three implementation plans, and three protocol-like documents.
- `VERIFIED`: `docs/setup/scenarionet_waymo_conversion.md` is the only inspected
  document outside the directory with a relative link to a file being moved.
- `VERIFIED`: `.github/copilot-instructions.md` says the initial observation is
  `LidarStateObservation`, while `conf/config.yaml` currently selects
  `semantic_state`.
- `VERIFIED`: `/data/` is ignored, so `data/repo/` is not a reviewable or
  semantically appropriate document handoff location.
- `SPECIFIED`: the user confirmed the Rulebook v4.6, curriculum, and ScenarioNet
  specification documents as authoritative.
- `SPECIFIED`: the semantic observation v1.1 specification has not yet been
  supplied.

## 5. Decisions And Approval Gates

| ID | Category | Decision | Status |
|---|---|---|---|
| `DEC-DOC-001` | Implementation detail | Use `docs/protocols/` for unapproved protocol-like documents | Approved by task scope |
| `DEC-DOC-002` | Implementation detail | Use root `incoming/`, ignored except for its README | Approved by user delegation |
| `DEC-DOC-003` | Specification authority | Register Rulebook, curriculum, and ScenarioNet documents as authoritative | Approved by user |

No unresolved gate blocks this work.

## 6. Proposed Structure

```text
docs/
  specifications/  # user-approved scientific and functional contracts
  implementation/  # living ExecPlans and implementation trackers
  protocols/       # candidate or operational validation/reporting protocols
  decisions/       # ADRs
  templates/
  architecture/
  setup/
  workflows/
  archive/
incoming/           # ignored remote handoff area; README tracked
```

## 7. Traceability And Acceptance Tests

| Requirement | Acceptance criterion | Validation |
|---|---|---|
| `REQ-DOC-001` | `AC-001`: index records all three confirmations as `AUTHORITATIVE` | Inspect index and specification metadata |
| `REQ-DOC-002` | `AC-002`: specifications and plans reside in separate directories | Inspect final tree |
| `REQ-DOC-003` | `AC-003`: protocols remain `CANDIDATE` unless separately approved | Inspect index |
| `REQ-DOC-004` | `AC-004`: no active reference to the removed mixed directory remains and introduced paths exist | `rg` plus path existence audit |
| `REQ-DOC-005` | `AC-005`: Copilot instructions point to durable sources and contain no stale Lidar default | Inspect file and compare `conf/config.yaml` |
| `REQ-DOC-006` | `AC-006`: `incoming/README.md` is tracked while uploaded contents are ignored | `git check-ignore` and status inspection |

Mandatory validation commands for this documentation-only change:

- `git diff --check`
- `git status --short --untracked-files=all`
- focused `rg` reference and placeholder audits

Production tests, smoke tests, Ruff, formatting, and mypy are `NOT_APPLICABLE`
because no executable or runtime configuration file is in scope.

## 8. Milestones

- [x] Analyse document roles, links, authority, and current guidance.
- [x] Move documents by role without changing scientific content.
- [x] Record approval metadata and update all links/index entries.
- [x] Align Copilot guidance and document `incoming/`.
- [x] Run mandatory documentation validation and reconcile the diff.

## 9. Progress And Findings

### 2026-07-16

- Completed repository analysis and received authority confirmations.
- Found no configured general formatter, Ruff target, mypy target, or coverage
  threshold; no tooling change is required for this restructure.
- Moved specifications, implementation plans, and protocols into role-specific
  directories. Content comparison found only approval metadata changes in
  specifications and path changes in plans; protocol contents are byte-identical.
- Replaced the ignored `data/repo/` staging area with the documented ignored
  `incoming/` handoff directory.

## 10. Deviations

No deviations identified.

## 11. Files

- Added process files: `AGENTS.md`, `.agent/PLANS.md`, workflow, templates,
  directory READMEs, authority index, and this ExecPlan.
- Moved three specifications to `docs/specifications/`.
- Moved three implementation trackers to `docs/implementation/`.
- Moved three candidate protocols to `docs/protocols/`.
- Updated `.gitignore`, root/documentation READMEs, Copilot instructions, and the
  ScenarioNet conversion link.
- Added `incoming/README.md` and removed the processed ignored `data/repo/`
  copies.

## 12. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `git diff --check` | `PASS` | 2026-07-16 | No whitespace errors |
| Old-path and placeholder `rg` audit | `PASS` | 2026-07-16 | No active old paths or unresolved repository placeholders |
| Introduced path existence audit | `PASS` | 2026-07-16 | All referenced destinations exist |
| `git check-ignore` inbox audit | `PASS` | 2026-07-16 | Uploads ignored; `incoming/README.md` not ignored |
| Moved-content comparison | `PASS` | 2026-07-16 | Scientific content unchanged; protocols byte-identical |

## 13. Final Reconciliation

All six requirements and acceptance criteria are implemented and verified. No
production test was run because executable behavior and runtime configuration
were outside scope. Known gaps remain intentionally visible in the authority
index: the semantic observation v1.1 specification is pending, three protocols
remain candidates, and some approved legacy documents do not follow the new
template. These gaps do not invalidate the completed repository structure.
