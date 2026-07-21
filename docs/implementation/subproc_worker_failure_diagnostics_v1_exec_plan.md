# Subprocess Worker Failure Diagnostics v1 ExecPlan

## 1. Metadata

- Feature: actionable diagnostics for process-vector environment worker failures.
- Plan ID: `SUBPROC-WORKER-DIAGNOSTICS-V1`.
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md` (`SCENARIONET-INTEGRATION`, v1.1, approved), §§23--24 and §27.2.
- Related ADRs: ADR-016 and ADR-018.
- Status: `IMPLEMENTED`.
- Created and last updated: 2026-07-21.

## 2. Objective And Scope

Replace parent-only `EOFError` reports from `DeterministicSubprocVecEnv` with an
actionable failure that identifies the worker slot, command, PID, exit code and,
for Python exceptions, the remote exception type, message and traceback. Preserve
all environment, provider, seed, reset, sampling, termination and ACL semantics.

In scope: worker-to-parent error envelopes, EOF exit-code reporting, focused
regressions, and an operator reproduction command. Out of scope: retrying or
masking failures, fallback sampling, changing data eligibility, or changing
experimental configuration.

## 3. Requirements And Acceptance Criteria

| ID | Requirement | Acceptance criterion |
|---|---|---|
| `REQ-SWD-001` | Process workers and cleanup remain supported. | `AC-SWD-001`: normal reset/step tests remain green. |
| `REQ-SWD-002` | Failures are reproducible from recorded diagnostics. | `AC-SWD-002`: a worker Python exception reports slot, command, PID, exception and traceback. |
| `REQ-SWD-003` | Abrupt worker termination is distinguishable from a Python exception. | `AC-SWD-003`: an EOF includes slot, command, PID and process exit code. |
| `REQ-SWD-004` | Scientific behavior is unchanged. | `AC-SWD-004`: no provider/configuration/data-policy code is changed. |

## 4. Current Analysis And Invariants

- `VERIFIED`: `_worker` lets non-EOF exceptions terminate the process; parent
  `remote.recv()` then raises uncontextualized `EOFError`.
- `VERIFIED`: builder crash-log wrappers cover ordinary `Exception` paths but
  cannot capture a native signal/OOM and their files may be outside the mounted
  run directory.
- `SPECIFIED`: ScenarioNet uses one process per environment, `spawn`, strict
  no-fallback provider sampling, and meaningful per-episode provenance.
- Invariant: a worker failure is fatal for the current vector environment; this
  change diagnoses it and never retries, resamples, or continues collection.

## 5. Design

The worker wraps each received command. On a Python `Exception`, it sends one
serializable error envelope containing command, PID and formatted traceback,
then closes. Parent receive helpers decode that envelope and raise a dedicated
`RuntimeError`. On `EOFError`, they inspect the matching `Process.exitcode` and
raise the same diagnostic type. All vector receive paths use the helpers.

## 6. Traceability And Test Matrix

| Requirement | Implementation | Test |
|---|---|---|
| `REQ-SWD-001` | `deterministic_subproc_vec_env.py` receive helpers | existing deterministic reset tests |
| `REQ-SWD-002` | worker error envelope and decoder | `test_worker_python_exception_reports_remote_traceback` |
| `REQ-SWD-003` | EOF decoder with process metadata | `test_worker_eof_reports_slot_and_exit_code` |
| `REQ-SWD-004` | bounded diff review | `git diff --check` |

Commands: `uv run --no-sync python -m pytest -q tests/test_deterministic_subproc_vec_env.py`; focused Ruff check/format check for the two modified Python files; `git diff --check`. Real ScenarioNet reproduction is intentionally deferred to the user-owned runtime fixture.

## 7. Milestones

- [x] M1: inspect the authoritative execution, provider and test contracts.
- [x] M2: implement fatal worker envelopes and EOF diagnostics.
- [x] M3: add deterministic regression tests and run focused validation.
- [x] M4: reconcile the plan and publish exact user reproduction commands.

## 8. Decisions, Deviations, And Findings

No approval gate or deviation: error observability does not change an observable
scientific or compatibility contract. The failure remains fatal by design.

2026-07-21: user requested diagnostic recovery and improved propagation after a
five-worker ScenarioNet SAC run exposed only `EOFError` in `step_wait`.
The implementation now makes Python exceptions cross the pipe as a structured
fatal failure, while native termination reports the child exit code. Focused
Docker pytest passed 5 tests in 8.57s after the reset-envelope regression;
focused Ruff format/check and
`git diff --check` passed. The local `uv` environment cannot run pytest because
it lacks pytest; the provisioned Docker environment is the validation source.
The first user reproduction exposed a decoder regression: normal reset replies
start with a NumPy observation, so marker matching must first require a string.
The fix adds a reset-envelope regression before the reproduction is retried.

## 9. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py` | Modify | Propagate worker failure context. |
| `tests/test_deterministic_subproc_vec_env.py` | Modify | Regression coverage. |
| `docs/project_index.md` | Modify | Register this implementation record. |
| This ExecPlan | Modify | Traceability and validation record. |

## 10. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_deterministic_subproc_vec_env.py` | `PASS` | 2026-07-21 | 5 passed in 8.57s, including normal reset, Python and abrupt-exit regressions. |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py` | `PASS` | 2026-07-21 | Both files formatted. |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py` | `PASS` | 2026-07-21 | No lint findings. |
| `git diff --check` | `PASS` | 2026-07-21 | No whitespace errors. |

## 11. Final Reconciliation

`REQ-SWD-001`--`REQ-SWD-004` are implemented and verified by the focused test
matrix and bounded diff review. A real ScenarioNet run remains user-owned
validation because its dataset/container fixture is external to this workspace.
No experimental result may be derived from the failed run.
