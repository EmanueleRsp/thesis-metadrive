# Subprocess Worker Fail-Fast v2 ExecPlan

## 1. Metadata

- Feature: fail-fast cleanup for process-vector environment worker failures.
- Plan ID: `SUBPROC-WORKER-FAIL-FAST-V2`.
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md` (`SCENARIONET-INTEGRATION`, v1.1, approved), §§23--24; `docs/specifications/rl_baselines_v1_specification.md` (`RL-BASELINES`, v1.0, approved), §8.
- Related ADRs: ADR-016, ADR-018, ADR-019.
- Status: `IMPLEMENTED`.
- Created and last updated: 2026-07-21.

## 2. Objective And Scope

Make a failure in any `DeterministicSubprocVecEnv` child immediately fatal to
the vector environment, without waiting for an unrelated slow worker and
without leaving children alive.  Preserve sampling, seed, reset, ACL,
termination, truncation, evaluation, and data-policy behavior.

In scope: parent-side readiness-based response collection, contextual worker
errors, bounded child cleanup, and deterministic regressions. Out of scope:
retrying a failed scenario, suppressing Rulebook or semantic-observation
errors, excluding dataset records, changing worker counts, or changing ACL.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-SWFF-001` | Process-vector workers retain deterministic normal collection and reset behavior. | ScenarioNet v1.1 §23; RL-BASELINES §8.2--8.3 |
| `REQ-SWFF-002` | A worker failure is observable and reproducible with worker context. | ScenarioNet v1.1 §24 |
| `REQ-SWFF-003` | Vector worker lifecycle has correct cleanup and no systematic reset errors. | ScenarioNet v1.1 §23.4 |
| `REQ-SWFF-004` | Evaluation failures remain fatal and are never silently continued. | ADR-019 decision 4 |

## 4. Current Repository Analysis And Invariants

- `VERIFIED`: `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py`
  already transports Python exceptions as tagged envelopes and reports EOF with
  slot/PID/exit code.
- `VERIFIED`: batch methods send all commands then call `_receive()` in slot
  order. A failure in a later slot can therefore wait behind a slow earlier
  slot; `close()` uses unbounded `join()`.
- `VERIFIED`: worker creation and reset/step crash files are persisted by
  `_CrashLoggingEnvWrapper` in `src/thesis_rl/runtime/wiring/builders.py`.
- Invariant: a failure remains fatal. No retry, resampling, fallback, data
  filtering, or continuation is introduced.
- Invariant: responses are reduced in requested slot order after all normal
  replies are available; readiness arrival order must not affect scientific
  outputs.

## 5. Proposed Design

Use `multiprocessing.connection.wait()` for every multi-worker receive. Decode
the first ready reply; if it is an error envelope or EOF, terminate/reap all
managed child processes, close parent pipe endpoints, mark the vector closed,
and re-raise the contextual `SubprocessWorkerError`. Normal responses are held
by slot and returned in deterministic slot order. Send failures use the same
fatal path. `close()` does not wait for an in-flight command: it reaps children
with a bounded join, then terminates a remaining child. This affects only
shutdown after caller abandonment/failure, not normal training transitions.

The existing per-worker crash files remain the durable diagnostics for
scenario-specific Rulebook/observation failures. A generic vec-env cannot
reliably extract `scenario_uid`; its remote traceback preserves the original
exception, while the wrapper log remains the source for full local context.

## 6. Traceability And Test Strategy

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SWFF-001` | `AC-SWFF-001`: normal auto-reset and manual reset behavior is unchanged. | vector send/receive helpers | existing normal tests | Verified |
| `REQ-SWFF-002` | `AC-SWFF-002`: Python and abrupt failures state slot, command and worker details. | error decoder | existing failure tests | Verified |
| `REQ-SWFF-003` | `AC-SWFF-003`: failure in a later slot returns before a slow earlier slot completes and leaves no child alive. | readiness collector and cleanup | new mixed slow/failing test | Verified |
| `REQ-SWFF-003` | `AC-SWFF-004`: closing with an in-flight slow command returns in bounded time and reaps the child. | close cleanup | new in-flight close test | Verified |
| `REQ-SWFF-004` | `AC-SWFF-005`: failure is propagated, not converted to a result. | fatal dispatch | mixed failure test | Verified |

Mandatory commands:

```text
uv run --no-sync python -m pytest -q tests/test_deterministic_subproc_vec_env.py
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py"
make lint PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py"
git diff --check
```

If the local environment lacks the required test dependency, the equivalent
already-provisioned Docker command is used and recorded.

## 7. Milestones

- [x] M1: inspect contracts, current implementation, and existing diagnostics.
- [x] M2: add regressions for mixed-worker failure and in-flight cleanup.
- [x] M3: implement readiness-based fatal collection and bounded cleanup.
- [x] M4: execute mandatory checks and reconcile documentation.

## 8. Decisions And Deviations

| ID | Category | Issue | Recommendation | Impact | Status |
|---|---|---|---|---|---|
| `DEC-SWFF-001` | implementation detail | Receive order currently delays a later failure. | Use readiness polling but preserve slot-ordered normal results. | Failure latency only. | Approved by user request 2026-07-21 |
| `DEC-SWFF-002` | implementation detail | `close()` can wait forever for an in-flight child. | Bounded join then terminate only owned children. | Shutdown robustness only. | Approved by user request 2026-07-21 |

No specification deviation is proposed.

## 9. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py` | Modify | Readiness-based failure propagation and owned-child cleanup. |
| `tests/test_deterministic_subproc_vec_env.py` | Modify | Deterministic regressions. |
| `docs/project_index.md` | Modify | Register v2 implementation record. |
| This file | Modify | Plan and validation evidence. |

## 10. Progress And Findings Log

- 2026-07-21: user approved implementation. The prior v1 envelope handling
  fixes uncontextualized `EOFError`, but does not satisfy prompt failure
  propagation when a lower-numbered worker is slow.
- 2026-07-21: `multiprocessing.connection.wait()` now receives replies by
  readiness. Normal replies are retained by slot, while a tagged exception,
  EOF, or failed send closes pipe endpoints and reaps only the vector's owned
  children. The mixed slow/failing regression and in-flight-close regression
  passed in the project Docker environment.

## 11. Deviations

No deviations identified.

## 12. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `uv run --no-sync python -m pytest -q tests/test_deterministic_subproc_vec_env.py` | NOT_RUN | 2026-07-21 | Local virtual environment lacks `pytest`; the equivalent Docker command below is authoritative. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_deterministic_subproc_vec_env.py` | PASS | 2026-07-21 | 7 passed in 13.22s, including later-slot failure and in-flight cleanup regressions. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_parallel_evaluation.py tests/test_parallel_evaluation_config.py tests/test_scenario_acl_vectorized_state.py` | PASS | 2026-07-21 | 20 passed in 2.39s. |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py` | PASS | 2026-07-21 | Both files already formatted. |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py tests/test_deterministic_subproc_vec_env.py` | PASS | 2026-07-21 | No lint findings. |
| `git diff --check` | PASS | 2026-07-21 | No whitespace errors at validation time. |

## 13. Final Reconciliation

`REQ-SWFF-001` is implemented and verified by existing normal reset tests and
the parallel integration suite. `REQ-SWFF-002` is implemented and verified by
the Python-exception and abrupt-exit regressions. `REQ-SWFF-003` is implemented
and verified by the mixed slow/failing-worker and in-flight-close regressions.
`REQ-SWFF-004` is implemented and verified because failures are re-raised to
the caller after cleanup; no continuation path was added.

Known limitation: native child failures can provide only PID and exit code when
the operating system prevents an error envelope. The existing worker crash log
and the parent error together remain the diagnostic path. No scenario is
retried, skipped, or excluded automatically.
