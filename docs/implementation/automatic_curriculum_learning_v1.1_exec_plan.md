# Automatic Curriculum Learning v1.1 ExecPlan

## 1. Metadata

- Feature/plan ID: `ACL-SN-EMA-001`
- Authoritative specification: `docs/specifications/automatic_curriculum_learning_v1.1_specification.md`, `APPROVED`, `Authoritative: YES`
- Status: `VERIFIED`
- Created/updated: 2026-07-23
- Related ADRs: ADR-014, ADR-016

## 2. Objective and scope

Implement the proposed ScenarioNet ACL redesign: bounded EMA semantic-arm scores, temperature-scaled sampling, 40/60 Generate/Replay after warm-up, unchanged 70/30 replay ranking, checkpoint schema/version compatibility, and default `make run` composition. Preserve six frozen arms, LP-only usefulness, immutable catalog data, deterministic vector commits, and algorithm-specific LP formulas.

## 3. Current repository analysis

`src/thesis_rl/curriculum/scenario_acl/mab.py` currently stores cumulative softmax logits plus `target_weights`, applies inverse-probability correction, clips logits, and synchronizes periodically. `src/thesis_rl/curriculum/config.py` exposes the legacy fields and defaults. `driver.py` uses `exploit_probability` for replay and serializes bandit state through the ACL vector state. `conf/config.yaml` and `Makefile` already select `scenario_acl_scenarionet` by default, but that profile inherits legacy parameters. Existing tests assert legacy behavior and must be replaced/strengthened only after specification approval.

## 4. Approval gates

| ID | Issue | Recommendation | Status |
|---|---|---|---|
| DEC-001 | EMA, no importance correction/target, temperature, 40/60, and frozen defaults change experimental behavior. | Approve ACL v1.1 specification. | APPROVED 2026-07-23 |
| DEC-002 | Legacy cumulative checkpoints cannot be interpreted as EMA. | Reject old checkpoints by schema identity; restart by default. | APPROVED 2026-07-23 |

No production implementation may begin while these gates are unresolved.

## 5. Traceability and acceptance matrix

| Requirement | Acceptance | Implementation | Tests |
|---|---|---|---|
| REQ-001 | AC-001 | `scenario_acl/mab.py` | `tests/test_scenario_acl_mab.py` |
| REQ-002 | AC-002 | `scenario_acl/mab.py`, config validation | MAB/config tests |
| REQ-003 | AC-003 | `scenario_acl/driver.py`, config | driver/config tests |
| REQ-004 | AC-004 | existing buffer and driver paths | buffer/ACL separation tests |
| REQ-005 | AC-005 | bandit/vector checkpoint serializers | checkpoint/resume tests |
| REQ-006 | AC-006 | `conf/curriculum/scenario_acl*.yaml`, `Makefile` | Hydra composition tests |

## 6. Test strategy (frozen after approval)

- Unit: EMA bounds, selected-arm-only update, no inverse correction, stable temperature softmax, exploration floor, invalid values, finite outputs.
- Integration: warm-up Generate-only, seeded 40/60 mode selection, six-arm availability, replay 70/30 ranking, Rulebook diagnostic invariance.
- Compatibility: v1.1 checkpoint round-trip and explicit rejection of legacy state.
- Configuration: root/default `make run` resolves ACL v1.1; explicit disabled/staged overrides remain valid.
- Regression/smoke: focused ACL tests, `make test` (or container equivalent), `make config`, `make smoke`, `git diff --check`.

## 7. Milestones

- [x] M0: read proposal, authoritative v1/ADR-014, template, and current implementation.
- [x] M1: write v1.1 specification under review and this plan.
- [x] M2: obtain explicit approval; rename spec, set `APPROVED`/`Authoritative: YES`, update `project_index.md`.
- [x] M3: implement config and EMA bandit with migration guard.
- [x] M4: implement driver mode schedule/default config and diagnostics.
- [x] M5: update tests and run validation/smoke.
- [x] M6: final reconciliation and ChatGPT source synchronization report.

## 8. Deviations

No deviations identified.

## 9. Validation results

| Command | Result | Notes |
|---|---|---|
| `python -c 'import ast; ...'` | PASS | Modified Python files parse successfully. |
| `git diff --check` | PASS | No whitespace errors. |
| Isolated EMA/softmax/checkpoint harness | PASS | Scores remain bounded, probabilities sum to one, exploration floor holds, and `acl_ema_v1` round-trips. |
| `uv run --no-sync python -m pytest ...` | NOT_RUN | Validation was run through the repository's Docker environment instead. |
| `make test` | PASS | Official Docker container: 857 passed in 88.66s. |
| `make config` | PASS | Official Docker Compose configuration validation succeeded. |
| `make smoke` | PASS | Official Docker smoke training completed 2000/2000 environment steps and evaluation. |
| `git diff --check` | PASS | No whitespace errors after final edits. |

## 10. Final reconciliation

- REQ-001 through REQ-006: implemented and covered by focused ACL/config tests plus the full repository suite.
- Legacy cumulative/target/importance-correction configuration and checkpoints are rejected explicitly.
- Default `make run` composition selects the ScenarioNet ACL v1.1 profile without additional CLI overrides.
- No unresolved approval gate or intentional deviation remains.
