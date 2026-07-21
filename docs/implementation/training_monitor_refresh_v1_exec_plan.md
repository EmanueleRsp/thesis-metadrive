# Training Monitor Refresh v1

## 1. Metadata

- Feature/plan ID: `TRAINING-MONITOR-REFRESH-V1`
- Authoritative specification: user request, 2026-07-21
- Status: `VERIFIED`
- Created: 2026-07-21
- Last update: 2026-07-21
- Branch: current working branch

## 2. Objective and scope

Refresh the live training monitor every 200 training steps in every supported
run profile. This changes the console refresh/metric-render cadence only; it
does not change evaluation cadence, checkpoint cadence, or scientific budgets.

## 3. Requirements and traceability

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| `REQ-TMR-001` | Training monitor refresh interval is 200 steps. | `conf/experiment/default.yaml`, `conf/run_profile/*.yaml` | `tests/test_training_monitor_config.py` |
| `REQ-TMR-002` | Evaluation and checkpoint intervals remain unchanged. | Only `log_interval` values are modified. | Configuration diff review |

## 4. Current repository analysis

- **VERIFIED:** `experiment.log_interval` is passed by `run_training()` to both
  scalar and vectorized `Agent.train*` paths.
- **VERIFIED:** `Agent.train()` and `Agent.train_vectorized()` use this value
  to decide when to update the live Rich training monitor.
- **VERIFIED:** run profiles override the base experiment value, so all active
  profiles must be updated together.

## 5. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_training_monitor_config.py` | PASS | 2026-07-21 | All profiles resolve to 200 |
| focused Ruff check/format check | PASS | 2026-07-21 | Test file checked |
| `git diff --check` | PASS | 2026-07-21 | No whitespace errors |

## 6. Deviations and limitations

No deviations identified. The monitor can render slightly after the exact
boundary when a vectorized iteration advances multiple environments; the
configured cadence remains 200 global training steps.
