# TODO

## Baseline and refactor

### Code Tasks

- [x] Refactor Hydra config files to keep only active, meaningful configuration groups and remove redundant or misleading fields.
  >Done: cleaned up `conf/`, removed unused/legacy fields, and aligned comments with current architecture.
- [x] Add structured logging and log file output instead of relying only on terminal prints.
  >Done: added JSONL rule-margin logging (`rule_margin_log_path`) and rule diagnostics via logger.
- [x] Review training monitor UX in `Agent.train()` and decide whether to replace periodic `print(...)` blocks with a progress-bar-based monitor.
  >Done: replaced print-style monitor with `rich` Live/Progress monitor.
- [x] Remove silent fallbacks in training-critical code paths and replace them with explicit fail-fast validation.
  >Done: added explicit validation/errors in critical TD3/reward paths (no silent fallback for invalid training config).
- [x] Route component-level logs to monitor during Live monitoring so `rich` progress output stays clean (console: warning/error only).
  >Done: routed component logs to Live monitor.

- [ ] Update result visualization so training/evaluation outputs are easier to inspect after runs:
    - [ ] Logs
    - [ ] Checkpoints
    - [ ] Videos
    - [ ] Csv
    - [ ] Plots
    - [x] Artifacts
- [ ] Decide whether `step_info` logging should support configurable keys for debugging instead of hard-coded fields only.

### Analysis Tasks

#### train.py

- [x] Analyze `_maybe_wrap_env_with_reward_manager(...)` and confirm the hybrid reward wrapper is applied exactly where intended.
  >Done: confirmed wrapper is applied only when `reward.mode=rulebook` (`scalar_default` stays native env reward).
- [x] Analyze the `agent.train(...)` call inside the main loop and validate the chunk-based training flow end-to-end.
  >Done: verified chunk flow + global progress handoff (`chunk_timesteps/global_total_timesteps/global_steps_done`).
- [x] Analyze the `agent.evaluate(...)` call inside the main loop and confirm evaluation uses the expected planner/preprocessor/adapter pipeline.
  >Done: verified evaluation path through preprocessor -> planner -> adapter and fixed eval guardrails.
- [x] Re-check curriculum setup and progression logic after the recent refactors to ensure train/eval environment switching stays correct.
  >Done: validated stage env switching, enforced disjoint train/eval pools per stage in auto mode, and aligned standalone evaluate stage resolution.

#### agent.py

- [x] Analyze lifecycle and adapter setup at the beginning of `Agent.train()`.
  >Done: validated begin/end training flow for lifecycle + adapter.
- [x] Analyze action selection inside the training loop, especially how `lifecycle.act(...)` behaves for TD3.
  >Done: aligned TD3 act/noise behavior with SB3-style flow.
- [x] Analyze transition recording, including whether the new `Transition` dataclass now captures all information needed by current and future planner lifecycles.
  >Done: introduced and integrated `Transition` dataclass end-to-end.
- [x] Analyze planner and adapter updates after each step, with focus on `lifecycle.maybe_update()`.
  >Done: reviewed update scheduling and alignment with chunked training loop.
- [x] Decide how to track rulebook reward components during training if they should become part of logging, debugging, or learning logic.
  >Done: rule components exposed in `info` and persisted via margin JSONL logs.

#### TD3 / Lifecycle

- [x] Verify whether `Td3Lifecycle.act()` correctly reproduces TD3 exploration behavior compared with standard SB3 training.
  >Done: updated act path to follow SB3-compatible exploration semantics.
- [x] Verify whether replay buffer insertion through `observe_transition(...)` matches SB3 expectations closely enough for TD3.
  >Done: fixed transition path with explicit env-action vs buffer-action handling.
- [x] Verify whether `maybe_update()` is fully aligned with SB3 TD3 scheduling, including `learning_starts`, `train_freq`, and gradient update timing.
  >Done: reviewed and corrected lifecycle scheduling behavior in TD3 path.
- [x] Confirm that global progress tracking across chunks is now handled correctly after the recent `chunk_timesteps/global_total_timesteps/global_steps_done` refactor.
  >Done: global-progress tracking now passed explicitly into lifecycle begin-training.

### Cleanup Candidates

- [ ] Remove stale comments once the corresponding analysis tasks are completed.
- [x] Review whether TODOs currently embedded in code should be reduced after they are captured in this file.

### Possible Improvements (Not Needed Now)

- [ ] Improve curriculum learning mechanism (e.g., more advanced progression logic, optional demotion, richer stage adaptation policies).
- [ ] Review and refine `neural_adapter` and `policy_adapter` configs/behavior (cleanup legacy notes, validate defaults, and align docs/comments with current architecture).

---

## Next Phases (After Current Refactor)

### Phase 1: Full Validation

- [ ] Plan and execute complete validation of the refactored pipeline.
  Validation checklist:
  1. [ ] Smoke test end-to-end: short run to verify train + eval + checkpoint save/load all work.
  2. [ ] Baseline validation: curriculum OFF, scalar native reward only (validate core agent/planner/adapter path).
  3. [ ] Curriculum validation: curriculum ON, scalar native reward only.
  4. [ ] Rulebook validation: curriculum ON, rulebook scalar reward enabled.
  5. [ ] Compare runs and sanity-check key metrics/logs (success, collision/out_of_road, route completion, rule saturation).

### Phase 2: Lexicographic RL Algorithms

- [ ] Implement lexicographic algorithms identified in the literature (define exact implementation checklist at execution time).

### Phase 3: Rule Semantics Unification for TLRL

- [ ] Evaluate and possibly redesign selected rules as constraint-like margins with uniform semantics across all rules.
