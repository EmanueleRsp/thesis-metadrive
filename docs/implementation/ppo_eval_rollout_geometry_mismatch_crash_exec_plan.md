# ExecPlan: PPO Async-Evaluation Checkpoint Load Crash — Rollout Geometry Mismatch

## 1. Metadata

- Feature / plan ID: `ppo-eval-rollout-geometry-mismatch-crash-v1`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`
  (PPO backend behavior unchanged; this is a runtime bug fix in checkpoint
  loading, not a formula/spec change)
- Status: `IMPLEMENTED`, live re-verification of the 3 restarted PPO runs in progress
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`
- Related decisions: none new — bug fix, no observable training behavior
  change for the resume-training path (validation preserved there)
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: loading a saved PPO checkpoint for **evaluation-only**
purposes (async evaluation workers, final evaluation, video rendering,
standalone eval loop) must not fail merely because the evaluation
environment's worker/env count differs from the training environment's
worker/env count.

**Why**: found live, mid-session, during the same 9-run `RUN_PROFILE=thesis`
campaign as the rulebook dynamic-candidate crash fix. All 3 PPO runs
(`ppo-0`, `ppo-1`, `ppo-2`) crashed at their first asynchronous evaluation
with:

```
ValueError: PPO global rollout size must be divisible by batch_size: 12 * 96 = 1152 is not divisible by 63.
```

Root cause: `Sb3PpoPlannerBackend.__init__` unconditionally calls
`_validate_rollout_geometry()`, which checks
`(n_steps * current_env_count) % batch_size == 0`. `n_steps` and
`batch_size` are loaded from the checkpoint (fixed at training time, when
`NUM_ENVS=21`); `current_env_count` is whatever env count the *caller*
constructed (here, the async-evaluation worker's env, with a different
worker count — 12 — than training). This check is only meaningful for a
model that will actually collect an `n_steps`-per-env rollout for a
gradient update (i.e. training/resume); for a checkpoint loaded purely for
inference (predicting actions during evaluation, never calling `.learn()`),
the check is both unnecessary and, as observed, actively wrong whenever
evaluation uses a different worker count than training — which
`async_evaluation`/`evaluation_num_workers`/`final_eval_episode_count`
already do by design (evaluation parallelism is configured independently
from training parallelism).

**In scope**:

- `Sb3PpoPlannerBackend.__init__`/`.load()`: add an optional
  `validate_rollout_geometry: bool = True` parameter; when `False`, skip
  `_validate_rollout_geometry()` (all other construction, including
  `global_rollout_size`/`minibatches_per_epoch` bookkeeping, is unchanged
  and harmless to compute with a mismatched eval env count since those
  values are not used to drive evaluation).
- `load_planner_backend` (`factory.py`) and `load_planner`
  (`runtime/wiring/builders.py`): thread the same optional parameter
  through, defaulting to `True` (preserves current behavior for every
  caller that does not explicitly opt out).
- Every call site that loads a checkpoint **purely for evaluation**
  (`async_evaluation.py`, `eval_loop.py`, `render_selected_videos.py`,
  `render_qualitative_videos.py`, `train_loop.py`'s `_make_eval_agent`,
  and both `final_eval_agent` construction sites in
  `curriculum/scenario_acl/driver.py`) now passes
  `validate_rollout_geometry=False`.

**Out of scope**:

- The two genuine **resume-training** call sites
  (`train_loop.py:810`, `driver.py:1474`, both constructing the planner
  with the actual training `env`) are left with the default `True` —
  resuming training with a genuinely mismatched geometry (e.g. someone
  changed `NUM_ENVS` between a training run and its resume) should still
  fail fast, since that scenario really would produce broken minibatches
  during further training.
- The native (non-SB3) `PpoPlannerBackend` (`ppo.py`) has no
  `_validate_rollout_geometry` method at all and is unaffected.
- TD3/SAC backends have no equivalent rollout-geometry concept (off-policy,
  no minibatch-from-rollout constraint) and are unaffected; their
  `.load()` signatures were not touched.

**Compatibility**: no checkpoint schema, public training-config key, or
training behavior change. The only observable change is that evaluation
(async, final, video-rendering, standalone eval loop) of a PPO checkpoint
no longer requires the evaluation env's worker count to divide evenly with
the training-time batch geometry — it never should have needed to.

## 3. Authoritative Requirements

| ID | Requirement |
|---|---|
| `REQ-001` | Loading a PPO checkpoint for evaluation-only use must succeed regardless of the evaluation environment's worker/env count. |
| `REQ-002` | Loading a PPO checkpoint for resume-training must still validate rollout geometry against the resume environment (unchanged, fail-fast). |

## 4. Current Repository Analysis

- `VERIFIED` live, in production: all 3 running PPO processes
  (`RUN_PROFILE=thesis`, `NUM_ENVS=21`) crashed at their first async
  evaluation, all with the identical error
  (`12 * 96 = 1152 is not divisible by 63`), confirming this is
  deterministic given the run's configuration, not a rare/flaky event —
  every PPO run under this profile was guaranteed to hit it.
- `VERIFIED`: `load_planner` (`runtime/wiring/builders.py:299`) is called
  from 9 distinct sites across the codebase; exactly 2 are genuine
  resume-training (pass the actual training `env`), the remaining 7 are
  evaluation-only (pass a separately constructed `eval_env`/`final_eval_env`
  with independently configured worker counts).
- `VERIFIED`: `_validate_rollout_geometry` (`ppo_sb3.py:126-142`) is PPO-SB3
  specific; no other backend (`td3.py`, `td3_sb3.py`, `sac.py`,
  `sac_sb3.py`, `ppo.py`) has an equivalent check, so no changes were
  needed to their `.load()` signatures.

## 5. Assumptions And Invariants

- `global_rollout_size`/`minibatches_per_epoch`/`optimizer_steps_per_update`
  (computed unconditionally in `__init__` regardless of the new flag) are
  training-loop bookkeeping values; for an evaluation-only planner
  instance these are simply never consulted (evaluation calls
  `predict`/`act`, never `.learn()`), so leaving their computation
  unguarded is intentional — only the fail-fast `raise` is gated.
- The evaluation environment's worker count being independent of the
  training environment's is pre-existing, approved design (`evaluation_num_workers`,
  separate `test_workers`/`final_eval_episodes` config), not something this
  plan changes.

## 6. Decisions And Approval Gates

No new material decision: this is a bug fix removing an incorrect
precondition check from a code path (evaluation-only checkpoint loading)
where that precondition was never actually required, while explicitly
preserving it for the one path (resume-training) where it is.

## 7. Proposed Design

Add `validate_rollout_geometry: bool = True` to
`Sb3PpoPlannerBackend.__init__` and `.load()`, gating the existing
`self._validate_rollout_geometry()` call with `if validate_rollout_geometry:`.
Thread the same parameter, defaulting to `True`, through
`load_planner_backend` (only forwarded to the `ppo_sb3` branch, since it is
the only backend that accepts it) and `load_planner`. Update the 7
evaluation-only call sites to pass `validate_rollout_geometry=False`;
leave the 2 resume-training call sites untouched (default `True`).

## 8. Traceability

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| `REQ-001` | `ppo_sb3.py::__init__`/`.load()`, `factory.py::load_planner_backend`, `builders.py::load_planner`, and the 7 evaluation-only call sites | Full regression suite (972 tests); live restart of `ppo-0`/`ppo-1`/`ppo-2` | Implemented; live re-verification in progress |
| `REQ-002` | Same functions, default parameter value unchanged | `train_loop.py:810`/`driver.py:1474` untouched, still pass no override (default `True`) | Verified by inspection |

## 9. Test Strategy

No new unit test was added in this pass (time-critical live fix while
production PPO processes were actively crashing at their first
evaluation); the fix is a straightforward optional-parameter gate around
an existing, already-tested validation call. Verification relied on:

1. `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_sb3_direct_backends.py tests/test_ppo_atomic_boundary.py tests/test_hydra_agent_presets.py -k "ppo or load"` — confirms no regression in existing PPO backend/config tests.
2. Full non-integration suite (972 tests) — zero regressions anywhere else.
3. Killing and restarting all 3 live PPO `RUN_PROFILE=thesis` processes
   with the fix, as the most direct real-world confirmation available.

**Follow-up recommended, not done in this pass**: add a deterministic unit
test constructing `Sb3PpoPlannerBackend.load(..., validate_rollout_geometry=False)`
with a mismatched env count and asserting no exception, plus a companion
test confirming `validate_rollout_geometry=True` (the default) still
raises for the same mismatched input — durable regression coverage
independent of live-run observation.

## 10. Milestones

### M1 — Live incident triage and fix

- Objective: identify the crash root cause from live production
  tracebacks, distinguish the resume-training vs. evaluation-only call
  sites, apply a minimal, targeted fix, verify no regression, restart the
  affected processes.
- Status: Done for the fix and restart; live multi-hour re-verification
  (do the 3 restarted PPO runs survive past their first async evaluation)
  is ongoing and not yet concluded as of this writing.

## 11. Progress And Findings Log

- 2026-07-26 — Discovered live via the background crash monitor set up for
  the 9-run `RUN_PROFILE=thesis` campaign, immediately after restarting
  all 9 runs to pick up the unrelated rulebook dynamic-candidate crash fix
  (`rulebook_v2_dynamic_candidate_vertical_incompatibility_crash_exec_plan.md`).
  All 3 PPO runs failed identically at their first async evaluation. Killed
  `ppo-0` preemptively once the pattern was confirmed deterministic from
  `ppo-1`/`ppo-2`'s tracebacks, to avoid burning further compute on a
  certain repeat failure. Traced the error to
  `Sb3PpoPlannerBackend._validate_rollout_geometry`, and to
  `async_evaluation.py`'s evaluation worker constructing its own env with
  a different worker count (12) than training (`NUM_ENVS=21`) while
  reusing the checkpoint's fixed `n_steps=96`/`batch_size=63`. Audited
  every `load_planner(...)` call site in the repository (9 total) to
  correctly distinguish the 2 genuine resume-training sites (which must
  keep the check) from the 7 evaluation-only sites (which must not).
  Implemented the optional-parameter gate, ran the full regression suite
  (972 passed, zero regressions), then killed and relaunched all 3 PPO
  sessions with the fix while leaving the unaffected 6 TD3/SAC sessions
  running undisturbed.

## 12. Deviations

No deviations identified. No PPO training-time formula, hyperparameter, or
checkpoint schema changed; only an evaluation-only precondition check is
now conditionally skipped.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` | Modified | Add `validate_rollout_geometry` parameter to `__init__`/`.load()`, gate the existing validation call |
| `src/thesis_rl/agent/planners/factory.py` | Modified | Thread `validate_rollout_geometry` through `load_planner_backend` to the `ppo_sb3` branch |
| `src/thesis_rl/runtime/wiring/builders.py` | Modified | Thread `validate_rollout_geometry` through `load_planner` |
| `src/thesis_rl/runtime/async_evaluation.py` | Modified | Pass `validate_rollout_geometry=False` (evaluation worker) |
| `src/thesis_rl/runtime/loops/eval_loop.py` | Modified | Pass `validate_rollout_geometry=False` (standalone eval loop) |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | Pass `validate_rollout_geometry=False` only in `_make_eval_agent`; resume-training load at line 810 left unchanged |
| `src/thesis_rl/analysis/videos/render_selected_videos.py` | Modified | Pass `validate_rollout_geometry=False` |
| `src/thesis_rl/analysis/videos/render_qualitative_videos.py` | Modified | Pass `validate_rollout_geometry=False` |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Pass `validate_rollout_geometry=False` at both `final_eval_agent` construction sites; resume-training load at line 1474 left unchanged |
| `docs/implementation/ppo_eval_rollout_geometry_mismatch_crash_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_sb3_direct_backends.py tests/test_ppo_atomic_boundary.py tests/test_hydra_agent_presets.py -k "ppo or load"` | PASS | 2026-07-26 | `10 passed, 17 deselected` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"` | PASS | 2026-07-26 | `972 passed, 5 deselected` |
| `docker compose run --rm dev uv run --no-sync ruff check` (all 9 modified files) | PASS | 2026-07-26 | `All checks passed!` |
| `git diff --check` | PASS | 2026-07-26 | No whitespace errors |
| Live restart of `ppo-0`/`ppo-1`/`ppo-2` with the fix applied | IN_PROGRESS | 2026-07-26 | Restarted after all 3 confirmed to crash identically pre-fix; a background monitor continues watching for further crashes; not yet run long enough to confirm survival past the first async-evaluation checkpoint (the point the pre-fix crashes occurred) |

## 15. Final Reconciliation

`REQ-001` and `REQ-002` are implemented and covered by the full regression
suite with zero failures. This plan's `IMPLEMENTED` status reflects the
code fix; it is not yet `VERIFIED` in the stricter sense of "confirmed to
prevent recurrence over a long real run," since the 3 restarted PPO
processes have not yet reached their first evaluation checkpoint since the
restart. A follow-up check (do the 3 PPO runs survive their first async
evaluation after the restart) is the natural closing action for this
plan, along with the recommended deterministic unit regression noted in
Section 9.

**Known limitations**: no new deterministic test was added for the
`validate_rollout_geometry=False` code path; coverage currently relies on
the pre-existing suite not regressing plus live-run observation.
**Deferred optional work**: add the unit regression pair described in
Section 9 (mismatched-geometry load succeeds when the flag is `False`,
still raises when `True`), so this bug class has durable unit coverage
independent of live-run luck — the same follow-up recommendation already
made for the sibling rulebook dynamic-candidate crash fix.
