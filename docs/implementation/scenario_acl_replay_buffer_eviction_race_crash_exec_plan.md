# ExecPlan: Scenario ACL Vectorized Replay-Buffer Eviction Race Crash

## 1. Metadata

- Feature / plan ID: `scenario-acl-replay-buffer-eviction-race-crash-v1`
- Authoritative specification: `docs/specifications/automatic_curriculum_learning_v1_specification.md`
  (ACL replay/generate mechanism unchanged; this is a runtime concurrency
  bug fix, not a formula/spec change)
- Status: `IMPLEMENTED`, live re-verification of `td3-0` in progress
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`
- Related decisions: none new — bug fix converting an unhandled race
  condition crash into a benign, logged, expected outcome
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: an in-flight vectorized "replay" episode whose parent
`ScenarioBuffer` record was legitimately evicted by a concurrent
completion elsewhere in the vector (before the replay episode itself
finishes and commits) must not crash the whole training run.

**Why**: found live, mid-session, during the same 9-run `RUN_PROFILE=thesis`
campaign as the two prior fixes today (rulebook dynamic-candidate crash,
PPO eval rollout-geometry crash). `td3-0` crashed with:

```
ValueError: Replay ACL record is missing from the parent buffer: pg:scenarionet_v1:PGMap-9920638
```

Root cause: `ScenarioBuffer.insert()` (`buffer.py:42-54`) evicts the
lowest-usefulness record whenever a new record with higher usefulness
arrives and the buffer is at capacity. In vectorized training
(`NUM_ENVS=21` parallel environments), a scenario selected for "replay" in
one slot can run for many steps while, concurrently, a *different* slot's
episode completes with a "generate" (fresh) scenario whose usefulness
exceeds the in-flight replay scenario's — legitimately evicting it from
the buffer per the approved ranking/eviction contract. When the long-running
replay episode in the first slot eventually finishes,
`commit_event`/`run_scenario_acl_training` (`driver.py:807-816`, pre-fix)
looks up that scenario's record to update it and finds it gone, raising an
unconditional `ValueError` that aborted the entire training process. This
is not a data-corruption bug or a violation of the eviction contract —
eviction working exactly as designed is *what causes* the crash, because
the code path that consumes a completed replay episode's outcome assumed
the record could never disappear out from under it.

**In scope**:

- `src/thesis_rl/curriculum/scenario_acl/driver.py`'s scenario-completion
  commit path: when a "replay"-mode completion's parent record is no
  longer present in the buffer, skip the buffer update for that episode
  (log it as a distinct, auditable outcome) instead of raising.
- No change to `ScenarioBuffer.insert()`/eviction logic itself — the
  eviction behavior is correct and approved; only the crash on the
  consuming side is fixed.

**Out of scope**:

- Any mechanism to "pin"/"reserve" in-flight replay records against
  eviction (a more invasive design change that would alter the buffer's
  eviction contract and require a new decision/approval — not attempted
  here; the chosen fix accepts the race as a normal, logged outcome
  instead).
- The MAB bandit update (`bandit.update(...)`, `driver.py:800-804`) and
  the JSONL/event logging that follow this block — both already execute
  unconditionally before/after this point and are unaffected.

**Compatibility**: no public interface, configuration key, or checkpoint
schema change. Observable behavior change: an episode whose replay record
was evicted before it could commit now produces a
`buffer_action: "skipped_evicted_before_commit"` log entry (both in the
buffer-events JSONL and the events log) instead of crashing the process.
No change to which scenarios are selected, how usefulness is computed, or
the eviction/ranking contract itself.

## 3. Authoritative Requirements

| ID | Requirement |
|---|---|
| `REQ-001` | A "replay"-mode episode completion whose parent buffer record has been evicted by a concurrent completion must not crash `run_scenario_acl_training`. |
| `REQ-002` | The skipped-update outcome must remain auditable (logged with a distinct action label), not silently dropped. |
| `REQ-003` | No change to `ScenarioBuffer` eviction/ranking behavior, MAB bandit updates, or "generate"-mode completion handling. |

## 4. Current Repository Analysis

- `VERIFIED` live, in production: `td3-0` crashed with this exact error
  after ~100,000 training steps under `NUM_ENVS=21` vectorized execution;
  the other 8 concurrently running processes (2 more TD3 seeds, 3 SAC, 3
  PPO) had not hit it at the same point, consistent with this being a
  timing-dependent race rather than a deterministic per-step failure (unlike
  the PPO rollout-geometry bug fixed earlier today, which was deterministic
  for every PPO run).
- `VERIFIED`: `ScenarioBuffer.insert()` (`buffer.py:42-54`) evicts the
  minimum-usefulness record once at capacity; this is the approved,
  intentional ranking/replacement contract (`ACL-SN-EMA-001`,
  `automatic_curriculum_learning_v1.1_specification.md`) — not itself a
  bug.
- `VERIFIED`: no existing test exercises the specific
  "replay episode commits after its record was concurrently evicted"
  scenario (`grep` for the exact error message and for
  `commit_event`/`run_scenario_acl_training` across `tests/` found no
  matches) — this is a genuine coverage gap, not a previously-known and
  accepted risk.
- `VERIFIED`: `record`/`updated` (the two variables only meaningfully set
  inside the `if record is not None:` branch after this fix) are not
  referenced anywhere later in the enclosing function beyond the
  `action`/`metrics`/`scenario_uid`/`lp` values already used for logging —
  confirmed by reading the full remainder of the function
  (`driver.py:840-864`).

## 5. Assumptions And Invariants

- The race is inherent to vectorized execution with a finite-capacity,
  usefulness-ranked buffer and variable episode lengths; it is expected to
  recur (rarely) across any of the 9 running processes over a long enough
  horizon, not specific to TD3 or to seed 0 — the fix applies uniformly to
  all algorithms since `driver.py`'s commit path is shared, backend-agnostic
  ACL infrastructure.
- Skipping the buffer update for an evicted-before-commit episode does not
  corrupt curriculum-learning state: the MAB arm-selection update (which
  drives future sampling) already happened earlier in the same function,
  unconditionally, before this block; only the *specific record's*
  usefulness/metrics refresh is skipped, and that record no longer exists
  in the buffer to refresh in the first place.

## 6. Decisions And Approval Gates

No new material decision: this converts an unhandled, previously-unknown
race condition (which — per the "Never claim a check passed unless it was
executed successfully" principle — no prior test or review had exercised)
into an explicitly logged, non-fatal, already-consistent-with-the-approved-
eviction-contract outcome. No change to the approved eviction/ranking
formula or acceptance behavior.

## 7. Proposed Design

In the `commit_event` closure inside `run_scenario_acl_training`
(`driver.py`), change the "replay" branch: if the record lookup returns
`None`, set `action = "skipped_evicted_before_commit"` and skip
`buffer.update(...)`; otherwise proceed exactly as before. The subsequent
JSONL buffer-events append and `log_event(...)` calls are unchanged and
naturally pick up the new `action` value, since both already read
`action` generically.

## 8. Traceability

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| `REQ-001` | `driver.py`'s `commit_event` "replay" branch | Full regression suite (972 tests, plus 77 ACL-adjacent focused tests); live restart of `td3-0` | Implemented; live re-verification in progress |
| `REQ-002` | Same location — `action` variable flows unchanged into the existing JSONL/event logging calls | Confirmed by reading the unchanged downstream logging code | Verified by inspection |
| `REQ-003` | No change to `buffer.py`, `bandit.update(...)`, or the "generate"-mode branch | Full regression suite passes; `tests/test_scenario_acl_buffer.py`, `tests/test_scenario_acl_mab.py` unmodified and passing | Verified |

## 9. Test Strategy

No new unit test was added in this pass (time-critical live fix while a
production process was down and the remaining 8 were at ongoing risk of
the same race); the fix is a straightforward `if/else` around an existing
lookup, converting a `raise` into a logged skip. Verification relied on:

1. `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py tests/test_scenario_acl_config.py tests/test_scenario_acl_scenario_env.py tests/test_scenario_acl_usefulness.py tests/test_scenario_acl_vectorized_state.py tests/test_curriculum_architecture.py tests/test_curriculum_manager.py tests/test_train_curriculum_helpers.py` — `77 passed`, zero regression in every ACL-adjacent focused suite.
2. Full non-integration suite (972 tests) — zero regressions anywhere else.
3. Restarting the actual crashed live process (`td3-0`) with the fix, as
   the most direct real-world confirmation available; the other 8
   processes, never having hit this race, were left running undisturbed.

**Follow-up recommended, not done in this pass**: add a deterministic unit
test that constructs a `ScenarioBuffer`, inserts a record, simulates its
eviction (a second `insert()` with higher usefulness at capacity), then
calls the commit path for a "replay" completion referencing the evicted
scenario's UID and asserts `action == "skipped_evicted_before_commit"`
with no exception — durable coverage for this specific race independent
of live-run luck. This is the same category of follow-up already
recommended for both prior fixes today.

## 10. Milestones

### M1 — Live incident triage and fix

- Objective: identify the crash root cause from the live production
  traceback, distinguish it as a race condition inherent to the approved
  eviction contract (not a data-corruption bug), apply a minimal,
  non-fatal fix, verify no regression, restart the affected process.
- Status: Done for the fix and restart; live re-verification (does
  `td3-0` survive without recurrence, and do the 8 other already-running
  processes avoid the same race going forward) is ongoing and not yet
  concluded as of this writing.

## 11. Progress And Findings Log

- 2026-07-26 — Discovered live via the background crash monitor for the
  9-run `RUN_PROFILE=thesis` campaign, after the monitor's regex had
  already been hardened against the earlier false-positive
  ("Avg error value") and after the rulebook dynamic-candidate and PPO
  eval-geometry crashes had both already been fixed and all 9 (then 3, for
  PPO) processes restarted. `td3-0` was the only one of the 9 to hit this
  particular error, at ~100k steps. Traced the root cause to
  `ScenarioBuffer.insert()`'s legitimate capacity-based eviction racing
  against a long-running vectorized replay episode's eventual commit.
  Confirmed no existing test exercised this path. Read the full
  surrounding function to confirm the fix's variables
  (`record`/`updated`) have no other downstream consumers, applied the
  minimal `if/else` fix, ran the full ACL-adjacent focused suite (77
  passed) plus the full non-integration suite (972 passed) with zero
  regressions, then restarted only `td3-0` (the one process actually
  affected) while leaving the other 8 running undisturbed, and restarted
  the background crash monitor with reset log offsets to account for
  `td3-0`'s freshly truncated log file.
- 2026-07-26 (continued) — `td3-2` then crashed with the identical error,
  confirming the fix was necessary (not a one-off): `td3-2` had been
  running since before this fix was written, so it still held the pre-fix
  code in memory (source edits do not affect an already-running Python
  process). Checked every other process's restart timestamp and found
  `td3-1`, `sac-0`, `sac-1`, `sac-2`, `ppo-0`, `ppo-1`, `ppo-2` were *all*
  still running pre-fix code (their most recent restart, for the earlier
  PPO/rulebook fixes, predated this ACL fix) — i.e. still exposed to the
  same race. Rather than wait for each to crash individually and lose
  more progress per incident, restarted all 7 proactively alongside
  `td3-2`, so every one of the 9 processes now runs the fully-fixed code
  (rulebook dynamic-candidate fix + PPO eval-geometry fix + this ACL
  buffer-eviction-race fix).

## 12. Deviations

No deviations identified. No ACL specification formula, ranking contract,
or MAB update logic changed; only the previously-unhandled consuming side
of a legitimate, approved eviction outcome.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | `commit_event`'s "replay" branch: skip (not crash) when the parent buffer record was concurrently evicted |
| `docs/implementation/scenario_acl_replay_buffer_eviction_race_crash_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py tests/test_scenario_acl_config.py tests/test_scenario_acl_scenario_env.py tests/test_scenario_acl_usefulness.py tests/test_scenario_acl_vectorized_state.py tests/test_curriculum_architecture.py tests/test_curriculum_manager.py tests/test_train_curriculum_helpers.py` | PASS | 2026-07-26 | `77 passed` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"` | PASS | 2026-07-26 | `972 passed, 5 deselected` |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/curriculum/scenario_acl/driver.py` | PASS | 2026-07-26 | `All checks passed!` |
| `git diff --check` | PASS | 2026-07-26 | No whitespace errors |
| Live restart of `td3-0` with the fix applied | IN_PROGRESS | 2026-07-26 | Restarted after the confirmed crash; other 8 processes left running undisturbed since none had hit this race; a background monitor continues watching all 9 for further crashes; not yet run long enough (the original crash occurred at ~100k steps) to confirm the specific race does not recur |

## 15. Final Reconciliation

`REQ-001`-`REQ-003` are implemented and covered by the full regression
suite with zero failures. This plan's `IMPLEMENTED` status reflects the
code fix; it is not yet `VERIFIED` in the stricter sense of "confirmed to
prevent recurrence over a long real run," since `td3-0` has not yet run
past the point where its pre-fix crash occurred, and the race is
inherently probabilistic (could in principle recur, now harmlessly logged
instead of fatal, in any of the 9 processes). A follow-up check (does
`skipped_evicted_before_commit` appear in any buffer-events JSONL without
a corresponding process crash) is the natural closing action for this
plan, along with the recommended deterministic unit regression noted in
Section 9.

**Known limitations**: no new deterministic test was added for this race
condition; coverage currently relies on the pre-existing suite not
regressing plus live-run observation. Because the fix makes the race
*silently survivable by design* (that is the point), there is no way to
observe from training-process exit status alone whether it recurs — only
the buffer-events JSONL's `buffer_action` field will show it, so a
periodic grep of that field is the appropriate ongoing signal, not the
crash monitor.

**Deferred optional work**: add the unit regression described in Section
9. Also worth considering in a future session (out of scope here, and
would need explicit approval as noted in Section 2): whether in-flight
replay records should be protected from eviction altogether, which would
eliminate the race rather than merely surviving it — a genuine design
trade-off (protecting stale in-flight scenarios vs. keeping the buffer
maximally responsive to newly-discovered high-usefulness scenarios) not
decided in this pass.
