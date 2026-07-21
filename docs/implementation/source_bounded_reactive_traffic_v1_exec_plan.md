# Source-Bounded Reactive Traffic v1 ExecPlan

## 1. Metadata

- Feature: source-bounded lifecycle for ScenarioNet reactive traffic.
- Plan ID: `SOURCE-BOUNDED-REACTIVE-TRAFFIC-V1`.
- Authoritative specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md` §§17, 23--24; `docs/specifications/observation_v1.1_specification.md` §§7.4, 10, 13.5.
- Status: `IMPLEMENTED`.
- Created and last updated: 2026-07-21.
- Related ADR: `docs/decisions/ADR-021-source-bounded-reactive-traffic.md`.

## 2. Objective And Scope

Retain the approved `reactive_traffic=true` behaviour only while a reactive
actor has a valid source-track state. At the first invalid source frame, remove
the actor through MetaDrive's normal manager cleanup path. The change is local
to `ThesisScenarioEnv`; vendored MetaDrive, dataset files, sampling, policy
inputs, reward, Rulebook semantics, and episode termination are out of scope.

## 3. Requirements, Analysis, And Invariants

| ID | Requirement | Source | Status |
|---|---|---|---|
| `REQ-SBRT-001` | A reactive vehicle is present only at source-valid frames. | ADR-021 | Approved |
| `REQ-SBRT-002` | Replay traffic retains upstream source-valid cleanup behaviour. | ADR-021 | Approved |
| `REQ-SBRT-003` | No replacement, fallback features, or changed experimental defaults are introduced. | ADR-021 | Approved |

Verified current behaviour: upstream `ScenarioTrafficManager.after_step()`
removes expired replay-policy objects but does not remove
`TrajectoryIDMPolicy` objects. The observed Waymo actor `2057` was source-valid
only through frame 48 and was still live at frame 99 under IDM. The source flag
is an indexed Boolean `track["state"]["valid"]`; an out-of-range index is
invalid. Manager cleanup is keyed by ScenarioNet track ID.

Invariant: for a reactive object mapped to source ID `s` at simulation step `t`,
its controller is invoked only if `state.valid[t]` exists and is true. If not,
`s` is queued for the existing same-frame cleanup, so the object cannot appear
after that transition.

## 4. Decision And Design

`DEC-SBRT-001` is approved by ADR-021: preserve reactive traffic, bounded by
source-track validity. `SourceBoundScenarioTrafficManager` overrides only the
upstream reactive pre-step lifecycle and delegates cleanup and replay behaviour
to the upstream manager. `ThesisScenarioEnv.setup_engine()` replaces the
already-registered traffic manager only when traffic is enabled.

## 5. Traceability And Test Matrix

| Requirement | Acceptance criterion | Implementation | Test | Status |
|---|---|---|---|---|
| `REQ-SBRT-001` | `AC-SBRT-001`: valid samples act; invalid and out-of-range samples queue removal without actuation. | owned manager | manager unit regression | Verified |
| `REQ-SBRT-002` | `AC-SBRT-002`: no-traffic setup is untouched and the normal manager lifecycle remains delegated. | environment setup | existing environment/config regression | Verified |
| `REQ-SBRT-003` | `AC-SBRT-003`: no config/default change and no fallback path. | bounded diff | focused checks | Verified |

Mandatory commands: focused pytest for the manager and ScenarioNet config,
focused Ruff format/check, `git diff --check`, and a user-run vectorized
ScenarioNet reproduction after implementation. A full live training smoke is
not required for the deterministic lifecycle regression but remains the final
runtime confirmation.

## 6. Milestones And Findings

- [x] M1: diagnose the worker failure using enriched projection diagnostics.
- [x] M2: establish that the actor was an IDM continuation after its Waymo
  source record ended; obtain approval for source-bounded reactive traffic.
- [x] M3: implement the owned manager and environment registration.
- [x] M4: add regression tests and run the focused validation matrix.
- [x] M5: provide the exact live reproduction command; runtime confirmation is user-owned.

## 7. Deviations And Files

No deviations identified.

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/scenario_traffic_manager.py` | Add | Source-valid reactive lifecycle. |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modify | Register the owned manager. |
| `tests/test_source_bounded_reactive_traffic.py` | Add | Lifecycle regression coverage. |
| `docs/decisions/ADR-021-source-bounded-reactive-traffic.md` | Add | Approved material decision. |
| `docs/project_index.md` | Modify | ADR and ExecPlan registry. |

## 8. Validation Results And Final Reconciliation

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/test_source_bounded_reactive_traffic.py tests/test_scenarionet_config.py tests/test_thesis_scenario_env.py` | `PASS` | 2026-07-21 | 39 passed in 2.38s. |
| Focused Ruff format/check for the two environment modules and new regression | `PASS` | 2026-07-21 | All checks passed. |
| `git diff --check` | `PASS` | 2026-07-21 | No whitespace errors. |

`REQ-SBRT-001`--`REQ-SBRT-003` are implemented and verified by the focused
matrix. The required remaining confirmation is the user-owned vectorized Waymo
reproduction: it must pass the previously failing setup without the expired
actor entering the semantic observation.
