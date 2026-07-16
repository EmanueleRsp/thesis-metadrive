# Repository Quality Commands ExecPlan

## 1. Metadata

- Plan ID: `PLAN-QUALITY-001`
- Feature: canonical Ruff lint and formatting commands
- Authority: user-provided process decision dated 2026-07-16
- Status: `VERIFIED`
- Created: 2026-07-16
- Last updated: 2026-07-16
- Related ADRs: none

## 2. Objective And Scope

Provide repeatable Ruff lint, format, and format-check commands for Python files
owned by this repository. Exclude vendored projects, data, outputs, generated
artifacts, caches, and environments. Do not reformat production code during this
task, add dependencies, introduce global mypy enforcement, or set a coverage
threshold.

## 3. Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-QUAL-001` | A canonical general Ruff lint target covers owned Python code | User decision |
| `REQ-QUAL-002` | Canonical Ruff format and format-check targets use the same controlled scope | User decision |
| `REQ-QUAL-003` | The scope excludes `third_party`, data, outputs, generated files, caches, and environments | User decision |
| `REQ-QUAL-004` | No mass formatting occurs in this initialization | User decision |
| `REQ-QUAL-005` | Global mypy and mandatory coverage thresholds remain deferred | User decision |

## 4. Verified Baseline

- Ruff is already a development dependency in `pyproject.toml` and the lock file.
- Project-owned Python files currently live under `src/` and `tests/`; `scripts/`
  currently contains shell scripts but remains in scope for future Python tools.
- `docker compose run --rm dev uv run --no-sync ruff check src tests` passes.
- `docker compose run --rm dev uv run --no-sync ruff format --check src tests`
  reports 186 files requiring formatting and 157 already formatted.
- No general Ruff or formatting Make target exists before this change.

## 5. Decisions

| ID | Decision | Rationale | Status |
|---|---|---|---|
| `DEC-QUAL-001` | Default scope is `src tests scripts` | Owned code only; future Python scripts are included | Approved |
| `DEC-QUAL-002` | Expose `PYTHON_QUALITY_PATHS` as an override | Enables safe file-level adoption before baseline cleanup | Approved |
| `DEC-QUAL-003` | Do not add a composite blocking quality target yet | Global format-check currently fails on legacy formatting debt | Approved |
| `DEC-QUAL-004` | Use existing conservative Ruff rule selection | Avoids enabling unrelated rule families | Approved |

## 6. Proposed Commands

- `make lint`
- `make format`
- `make format-check`
- Focused example:
  `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/example.py tests/test_example.py"`

All targets execute Ruff through the existing Docker Compose and `uv --no-sync`
workflow.

## 7. Acceptance And Traceability

| Requirement | Acceptance criterion | Validation |
|---|---|---|
| `REQ-QUAL-001` | `AC-001`: `make lint` resolves to Ruff check over the controlled scope and passes | Execute `make lint` |
| `REQ-QUAL-002` | `AC-002`: format targets resolve to Ruff formatter commands over the same scope | `make -n`; execute a focused format-check |
| `REQ-QUAL-003` | `AC-003`: Ruff configuration explicitly excludes non-owned heavy paths | Inspect `pyproject.toml` |
| `REQ-QUAL-004` | `AC-004`: no Python production or test file is modified | Git status/diff audit |
| `REQ-QUAL-005` | `AC-005`: no mypy target, coverage threshold, or dependency change is introduced | Configuration and lock-file audit |

## 8. Milestones

- [x] Inspect existing tools, commands, and owned Python scope.
- [x] Measure lint and format baselines without modifying files.
- [x] Add conservative Make targets and Ruff exclusions.
- [x] Update permanent process documentation.
- [x] Run acceptance checks and reconcile the diff.

## 9. Protected Baseline And Adoption Policy

The global lint target is immediately usable. The global formatting check is a
canonical target but not yet a completion gate because the repository baseline
is not clean. Until a dedicated formatting-only change establishes the baseline,
new or materially modified Python files must be formatted and checked through a
focused `PYTHON_QUALITY_PATHS` override. Do not mix mass formatting with semantic
changes.

## 10. Deviations

No deviations identified.

## 11. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| Existing Ruff lint baseline | `PASS` | 2026-07-16 | `src tests` |
| Existing Ruff format baseline | `FAIL` | 2026-07-16 | 186 files would change; no formatting applied |
| `make -n lint format format-check` | `PASS` | 2026-07-16 | Targets resolve to Docker Compose, `uv --no-sync`, and Ruff |
| `make lint` | `PASS` | 2026-07-16 | `src tests scripts`; all checks passed |
| Focused `make format-check` on `src/thesis_rl/__init__.py` | `PASS` | 2026-07-16 | File already formatted |
| Focused `make format-check` including `tests/test_common_paths.py` | `FAIL` | 2026-07-16 | Correctly exposed legacy formatting debt; no file changed |
| `git diff --check` | `PASS` | 2026-07-16 | No whitespace errors |
| Production/test/lock-file diff audit | `PASS` | 2026-07-16 | No changes under `src/`, `tests/`, or `uv.lock` |

## 12. Final Reconciliation

All five requirements and acceptance criteria are implemented and verified. The
global lint command is immediately usable. Formatting commands are canonical and
support focused adoption, while the measured 186-file legacy formatting debt is
explicitly non-blocking until a separate formatting-only change is approved. No
mass formatting, global mypy target, coverage threshold, dependency change, or
production/test modification was introduced.
