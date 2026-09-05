# ExecPlan — Startup Coverage For The Production Composition (`SMOKE-COV`)

## 1. Metadata

| Field | Value |
|---|---|
| Feature | Automated coverage for startup-level breakage of the production training composition |
| Plan ID | `SMOKE-COV` |
| Authoritative specification | None amended. The driver is `AGENTS.md` ("add a regression test for every discovered bug"; validation-command scope is a repository convention) plus the defect recorded in §4 |
| Status | `VERIFIED` for the selected option; the rejected option is recorded as deferred with its rationale |
| Created | 2026-09-02 |
| Last updated | 2026-09-02 |
| Branch | `main` (work done on `scenarionet-implementation`, merged 2026-09-05) |
| Related | `AB-LEARN` (the screening whose smoke exposed the defect) |

## 2. Objective And Scope

**Observable capability.** A startup-level breakage of the canonical training
composition — `env=scenarionet`, `obs=semantic_v3`, an `lq_v3`-family encoder,
`reward=scalar_reward` — is caught by an automated check rather than by a person
launching a long run.

**Why it is needed.** On 2026-09-02 the `AB-LEARN` smoke failed at environment
construction with `KeyError` on three `semantic_v3_signal_*` configuration keys.
The defect had been present since 2026-08-01 (commit `52b8ad4`) and made **every**
`obs=semantic_v3` run unstartable, including `make run-train`, the canonical
production entry point. Nothing caught it for a month because the repository's
only end-to-end check, `make smoke`, selects `obs=lidar_state` and
`agent/planner/encoder=none` — a different code path.

**In scope.** The defect fix, a regression test for its class, and the decision
about how the repository's automated checks should cover the production
composition.

**Out of scope.** Broadening `make smoke` into a general matrix; GPU-dependent
checks in the default test path; any change to the observation or encoder
contracts.

**Compatibility.** The fix declares three configuration keys with the values the
readers already used as defaults, so no behavior changes: the environment either
failed to construct, or now constructs with exactly the values the code expected.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-COV-001` | Every discovered bug carries a regression test | `AGENTS.md`, Testing And Traceability |
| `REQ-COV-002` | A startup-level breakage of the production composition is caught automatically | This plan's objective |
| `REQ-COV-003` | No invented validation command; any new target is real and documented | `AGENTS.md`, Verified Commands |
| `REQ-COV-004` | The default test path stays fast and needs neither GPU nor dataset | Verified repository convention: `make test` runs without the GPU overlay |

## 4. Current Repository Analysis

All `VERIFIED` on 2026-09-02.

| Fact | Evidence |
|---|---|
| The passthrough injects three keys for `semantic_v3` | `envs/factory.py:128-133` |
| The environment reads them back with the same defaults | `envs/thesis_scenario_env.py:469-472` |
| `default_config()` declared none of them | `envs/thesis_scenario_env.py:52-73` before this change |
| MetaDrive rejects undeclared keys at construction | `metadrive/envs/base_env.py:293` calls `default_config().update(config, False, ["agent_configs", "sensors"])`; `metadrive/utils/config.py:143` raises `KeyError` |
| **The rejection is recursive** | `Config._update_dict_item` calls `self[k].update(v, allow_add_new_key=allow_overwrite)`, so `allow_add_new_key=False` propagates into nested dictionaries. A flat key-set comparison would miss an unknown key under `vehicle_config.lidar`, where the semantic passthrough also writes |
| `make smoke` does not exercise the production composition | `Makefile:423` runs `presets/test/smoke_train`, which overrides `obs: lidar_state` and `encoder: none` |
| `make run-train` does | `Makefile:459` passes `obs=semantic_v3` |
| The Hydra preset tests validate composed values, not environment construction | `tests/test_hydra_preset_run_configs.py` asserts configuration fields only |
| Introduced 2026-08-01 | `git log -S "semantic_v3_signal_range_m"` returns `52b8ad4` alone |

## 5. Assumptions And Invariants

- The three baseline values `80.0 / 65.0 / 1.2` are `OBS-V1.2` §6.2's signal-camera
  baseline, overridable through `conf/obs/semantic_v3.yaml` per ADR-045. They are
  declared here identical to the defaults the readers already applied, so the
  declaration is not a calibration choice.
- Every shipped observation configuration declares a `type` field (`VERIFIED`
  across all six files in `conf/obs/`), so the parametrization can be driven by
  the directory.
- `ThesisScenarioEnv.default_config()` is cheap enough for a unit test: the whole
  file runs in 2.3 s.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-COV-001` | Implementation detail | How should the production composition be covered? | (a) a second smoke preset wired into a Make target; (b) a fast non-GPU test performing MetaDrive's own construction merge for every observation type | **(b)**, with (a) deferred | Determines what runs on every `make test` and what the default gate costs | Selected; see §7 |

**Why (b).** It catches the entire "the environment cannot be constructed" class
**exhaustively over every shipped observation**, in milliseconds, with no GPU and
no dataset, and it runs on every `make test`. It would have caught this defect on
the day it was introduced. Crucially, it asserts by *calling MetaDrive's own merge*
with the same arguments `BaseEnv.__init__` uses, rather than reimplementing the
acceptance rule — which matters because the rule is recursive, and the first
version of this test, written minutes earlier, compared flat key sets and would
have missed a nested key.

**Why (a) is deferred, not rejected.** A production-composition smoke catches a
strict superset: encoder/observation dimension mismatches, worker spawn, the first
reset, a learner step, checkpoint identity. But it costs minutes, needs the dataset
and realistically the GPU, and wiring it into `make smoke` would double the cost of
the default gate for every use. The failure mode actually observed is fully covered
by (b) at no cost. If (a) is added later, the right shape is a **separate opt-in
target** run before committing to long runs — the workflow moment where it pays —
not an extension of `make smoke`. Recorded as deferred work in §15.

## 7. Proposed Design

Two changes, both small.

1. **The fix.** `ThesisScenarioEnv.default_config()` declares
   `semantic_v3_signal_range_m`, `semantic_v3_signal_fov_degrees` and
   `semantic_v3_signal_camera_height_m` at the `OBS-V1.2` §6.2 baseline, with a
   comment recording why they must be declared.

2. **The regression test.**
   `test_observation_passthrough_is_accepted_by_environment_config` runs
   `_configure_agent_observation` for each file in `conf/obs/*.yaml` and then
   performs `ThesisScenarioEnv.default_config().update(env_cfg, False,
   ["agent_configs", "sensors"])`. Two deliberate properties: the parametrization
   is **driven by the directory**, so a newly added observation configuration is
   covered without editing the test; and the assertion is **the construction merge
   itself**, so it cannot drift from MetaDrive's semantics.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-COV-001` | `AC-COV-001` | `envs/thesis_scenario_env.py:70-82` | `TEST-COV-001` | Implemented, verified |
| `REQ-COV-002` | `AC-COV-002` | `tests/test_thesis_scenario_env.py` | `TEST-COV-001`, `TEST-COV-002` | Implemented, verified |
| `REQ-COV-003` | `AC-COV-003` | No new command introduced | — | Not applicable |
| `REQ-COV-004` | `AC-COV-004` | Test uses no GPU and no dataset | `TEST-COV-001` | Implemented, verified |

## 9. Test Strategy Defined Before Implementation

- `AC-COV-001` — the three keys are declared, at the `OBS-V1.2` §6.2 baseline.
- `AC-COV-002` — every configuration in `conf/obs/` produces an environment
  configuration MetaDrive's merge accepts.
- `AC-COV-003` — no validation command is invented.
- `AC-COV-004` — the check runs without GPU or dataset, in seconds.

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-COV-001` | Unit | Construction merge accepts the passthrough output | Each `conf/obs/*.yaml` | No `KeyError` | `REQ-COV-002` |
| `TEST-COV-002` | Regression | The test fails on the pre-fix code | Same, with the three declarations removed | `semantic_v3` fails, other five pass | `REQ-COV-001` |
| `TEST-COV-003` | Integration | The production composition starts end to end | `AB-LEARN` smoke, both arms | Past environment construction | `REQ-COV-002` |

Commands: `uv run --no-sync python -m pytest tests/test_thesis_scenario_env.py -q`,
`make lint`, `make format-check`, all with a focused scope.

## 10. Milestones

### `M1` — Fix and regression test — **complete**

Evidence in §14.

### `M2` — Production-composition smoke target — **deferred** (`DEC-COV-001`)

Not started by decision, not by omission. See §6 and §15.

## 11. Progress And Findings Log

**2026-09-02.** Defect found by the `AB-LEARN` smoke, which existed precisely
because the `semantic_v3` + `lq_v3` path had never been exercised since `RB51`
changed the observation dimension. The prediction was wrong about *which* failure
would appear — the expectation was a `D = 3011` rejection — but right that the
path was unverified.

Finding worth keeping: the first version of the regression test, written the same
hour, asserted a **flat** key-set difference. Reading
`metadrive/utils/config.py:167` showed the rejection recurses into nested
dictionaries, and the semantic passthrough writes into `vehicle_config.lidar`, so
the flat form had a real blind spot. Replaced with the construction merge itself.
The lesson generalizes: when an external library owns an acceptance rule, assert by
calling it, not by restating it.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Declares the three keys (`REQ-COV-001`) |
| `tests/test_thesis_scenario_env.py` | Modified | Regression test (`REQ-COV-002`) |
| `docs/implementation/production_composition_startup_coverage_exec_plan.md` | Added | This plan |
| `docs/open_items.md` | Modified | Records the defect and why it stayed invisible |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `pytest tests/test_thesis_scenario_env.py -q` | `PASS` | 2026-09-02 | 42 passed, 2.26 s |
| `pytest tests/test_thesis_scenario_env.py tests/test_scenarionet_config.py -q` | `PASS` | 2026-09-02 | 41 passed at the earlier revision of the test |
| `TEST-COV-002`, pre-fix behaviour | `PASS` | 2026-09-02 | The three declarations were removed from a backed-up copy: `semantic_v3` failed with the diagnostic message, the other five passed, and the file was restored. The test discriminates |
| `make lint` (focused) | `PASS` | 2026-09-02 | Both changed files |
| `make format-check` (focused) | `PASS` | 2026-09-02 | Both changed files, after formatting only the new test block |
| `AB-LEARN` smoke, both arms (`TEST-COV-003`) | `PASS` | 2026-09-02 | Both arms `exit 0` after a second, unrelated defect was cleared (`C6`). Seven CSV artifacts each, `final_eval.csv` populated, no NaN or infinity, `reward_behavior` correctly `monitor_only` and `scalar_reward` |
| `make test` (full suite) | `PASS` | 2026-09-02 | **1598 passed**, 4m35s. The stated risk -- a test elsewhere asserting the exact key set of `default_config()` -- did not materialize |

## 15. Final Reconciliation

`REQ-COV-001`, `REQ-COV-002` and `REQ-COV-004` are `IMPLEMENTED` and `VERIFIED`.
`REQ-COV-003` is `NOT_APPLICABLE`: no command was added.

**Known limitation.** The selected option covers *construction*, not *execution*.
An observation and encoder that agree on configuration keys but disagree on tensor
dimensions still fails only at run time. That is the residual gap `M2` would close.

**Deferred required work.** `M2`, a production-composition smoke, as an opt-in
target rather than an extension of `make smoke` (`DEC-COV-001`).
