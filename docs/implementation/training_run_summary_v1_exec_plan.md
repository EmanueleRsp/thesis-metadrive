# Training Run Summary v1

## 1. Metadata

- Feature/plan ID: `TRAINING-RUN-SUMMARY-V1`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`, `RL-BASELINES v1.0`, approved
- Status: `VERIFIED`
- Created: 2026-07-21
- Last update: 2026-07-21
- Branch: current working branch
- Related ADRs: none

## 2. Objective and scope

Make the Rich `Training Run` and `Evaluation Run` setup tables concise and
consistent with the requested experiment identity. Keep metadata and Hydra
configuration artifacts persisted; only their console presentation is removed.

## 3. Authoritative requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-TRS-001` | Show the requested run, pipeline, algorithm, replay, seed, and worker fields. | User request, 2026-07-21 |
| `REQ-TRS-002` | Do not show the explicitly removed fields. | User request, 2026-07-21 |
| `REQ-TRS-003` | Render disabled curriculum, Rulebook, scalarization, vectorization, PER, and single-worker features as `off`. | User request, 2026-07-21 |

## 4. Current repository analysis

- **VERIFIED:** `print_run_setup()` in `src/thesis_rl/runtime/io/console.py`
  is shared by training and evaluation.
- **VERIFIED:** training and Scenario ACL supplied observation/action/device
  rows through `extra_rows`; these rows duplicated or exposed removed fields.
- **VERIFIED:** resolved Hydra locations provide dataset/environment, curriculum,
  reward/Rulebook, scalarization, observation, encoder/decoder, algorithm replay,
  and evaluation/test worker values.
- **INFERRED:** `env.name` is the dataset display fallback because the current
  root configuration does not compose a separate dataset group.

## 5. Decisions and invariants

- `Rulebook` displays `off`, `monitor only`, or the configured top-level Rulebook
  version.
- `Scalarization function` displays `off` unless reward behavior is
  `scalar_reward`, then uses `scalarization.mode`.
- `n-steps` uses transition replay `n_steps` for off-policy algorithms and the
  algorithm-level `n_steps` for PPO.
- Artifact creation and function parameters remain compatible; metadata and
  Hydra paths are no longer rendered.

## 6. Traceability and test strategy

| Requirement | Implementation | Test |
|---|---|---|
| `REQ-TRS-001` | `_run_setup_rows()` and `print_run_setup()` | `test_run_setup_displays_requested_configuration_fields` |
| `REQ-TRS-002` | shared setup table row list and removed loop extras | same test |
| `REQ-TRS-003` | normalized `off` handling | `test_run_setup_marks_disabled_features_as_off` |

Commands: `pytest -q tests/test_console.py`, focused Ruff check/format check,
and `git diff --check`.

## 7. Milestones

- [x] Identify shared formatter and configuration sources.
- [x] Add acceptance/regression tests.
- [x] Implement the concise setup table.
- [x] Run focused validation and reconcile the final diff.

## 8. Deviations

No deviations identified.

## 9. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/io/console.py` | Modified | Derive and render requested setup fields |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | Remove obsolete training-only rows |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Remove obsolete ACL-only rows |
| `tests/test_console.py` | Modified | Acceptance and regression coverage |

## 10. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_console.py` | PASS | 2026-07-21 | 4 passed |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/io/console.py src/thesis_rl/runtime/loops/train_loop.py src/thesis_rl/curriculum/scenario_acl/driver.py tests/test_console.py` | PASS | 2026-07-21 | All checks passed |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/runtime/io/console.py src/thesis_rl/runtime/loops/train_loop.py src/thesis_rl/curriculum/scenario_acl/driver.py tests/test_console.py` | PASS | 2026-07-21 | 4 files already formatted |
| `docker compose config --quiet` | PASS | 2026-07-21 | Compose configuration valid |
| `git diff --check` | PASS | 2026-07-21 | No whitespace errors |

## 11. Final reconciliation

`REQ-TRS-001`, `REQ-TRS-002`, and `REQ-TRS-003` are VERIFIED by the focused
console tests and shared formatter. No scientific behavior, dataset policy,
checkpoint format, or persisted metadata behavior is changed. No deviations or
unresolved issues remain. `docs/project_index.md`,
`docs/engineering_workflow.md`, and
`docs/templates/specification_template.md` were not modified; no ChatGPT
Project source replacement is required.
