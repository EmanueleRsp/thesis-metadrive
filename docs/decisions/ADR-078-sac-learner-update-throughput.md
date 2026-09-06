# ADR-078: Off-policy learner update throughput — batch 512 at half update-to-data ratio, diagnostic learning potential off

- Status: **Approved**
- Date: 2026-09-06
- Approval evidence: explicit user approval on 2026-09-06, in two steps. First
  the user asked for the single most defensible throughput change that needs no
  further test runs, and accepted the recommendation of `batch_size 512` with
  `update_to_data_ratio 0.5` ("decidiamo e fine"). Then, after the assistant
  explained what the post-update learning-potential batch computes and that it
  is diagnostic-only under ACL `v2.0`, the user asked to switch it off as well
  ("disattiviamo anche quel learning potential per ora così comunque un minimo
  si guadagna"). Third step, same day: asked whether TD3 and PPO should follow,
  the assistant recommended extending the setting to TD3 and leaving PPO until
  its cost is measured; the user replied "procedi".
- Affected specifications:
  `docs/specifications/rl_baselines_v1_specification.md` (`RL-BASELINES` v1.0,
  §3.4 profile matrix, `REQ-RLB-006` TD3 learning, `REQ-RLB-007` SAC learning,
  §9.3 TD3 configuration, §9.4 SAC configuration — the `batch_size = 256` and
  `gradient_steps = auto -> n_envs` rows for both off-policy learners);
  `docs/specifications/evaluation_protocol_v1.0_specification.md`
  (`EVAL-PROTOCOL` v1.0, `DEC-015`, the parenthetical "`train_freq=1`,
  `gradient_steps=auto`" for TD3/SAC — the atomic unit is unchanged, the
  resolution of `auto` is not);
  `docs/specifications/automatic_curriculum_learning_v2.0_specification.md`
  (`ACL-SN-EMA-001` v2.0, `REQ-013` — the diagnostic chain "continues to run"
  and "no additional learner work is introduced").
- Related: ADR-075 (undiscounted return, same SAC configuration file), ADR-077
  (ACL v2.0, which made the learning potential diagnostic-only), ADR-044
  (`step_timing.csv`, the instrument behind the measurements below), the
  `AB-LEARN` ExecPlan §11 entries of 2026-09-06.

## Context

The `AB-LEARN` screening runs (SAC, `lq_v3`, 20 workers, `run_profile=medium`)
measured, on `step_timing.csv` of the live runs rather than on the smoke:

| component | arm A | arm B |
|---|---:|---:|
| `learner_update` | 179.0 ms/step, 81.4 % | 178.8 ms/step, 81.5 % |
| `worker_wrapped_env_step` | 173.4 ms/step, 78.9 % | 184.2 ms/step, 83.9 % |
| of which `rulebook_evaluator` | 70.5 ms | 75.2 ms |
| of which bare `env_step` | 35.4 ms | 35.2 ms |
| `rulebook_scalarization` | — | 0.04 ms |

Learner and workers overlap, so the wall clock per step (~220 ms, ~3.9 fps) is
roughly the larger of the two plus the non-overlapped remainder. Regime
throughput projects to ~25 h per `medium` run and ~107 h per `thesis` run.
The GPU is shared with three other tenants at 100 % utilisation; when arm B
died, arm A rose from 3.93 to 4.7 fps, which confirms the learner's GPU work is
the binding resource. Worker count is at its measured ceiling (20–21), and the
Rulebook v2 evaluator exposes no tunable cost knob (`rulebook/v2/config.py`
validates every geometry and prediction parameter as immutable).

Two learner-side facts were established by code reading:

1. With `gradient_steps: auto` and 20 workers, `Sb3SacPlannerBackend` runs
   **20 gradient steps of batch 256 per 20 collected transitions** —
   update-to-data ratio 1.0, 5 120 replay samples per update call.
2. After `model.train(...)`, `maybe_update` unconditionally draws a **second
   replay batch** and runs actor, twin critic and critic-target forward passes
   to compute a TD-residual learning potential
   (`sac_sb3.py`, `td3_sb3.py`). Under ACL `v2.0` (`ADR-077`, `REQ-013`) that
   quantity is **diagnostic-only** and must not influence the curriculum; the
   diagnostic chain `REQ-013` actually names is the collection-time one in
   `agent/planners/core/lifecycle.py`, while this post-update value only feeds
   two logging fields in `agent/agent.py`. Historical measurement of the
   equivalent TD3 pass: ~3 % of learner-update time.

## Decision

1. **SAC and TD3 replay minibatch `256 -> 512` and `update_to_data_ratio
   1.0 -> 0.5`**, for every non-smoke run profile (`fast`, `default`,
   `medium`, `long`, `tune`, `thesis`) and for
   `conf/agent/planner/algorithm/sac_sb3.yaml` and `td3_sb3.yaml`.
   `gradient_steps: auto` now resolves to
   `round(train_freq * n_envs * update_to_data_ratio)`, floored at 1; the
   default ratio `1.0` reproduces the `RL-BASELINES` v1 resolution
   `gradient_steps = n_envs`, so no other configuration changes meaning.
   With 20 workers: 10 gradient steps of batch 512 per 20 transitions —
   **the same 5 120 replay samples per update call as before, in half the
   optimizer steps**, same learning rate, same target-update cadence per
   gradient step; TD3's `policy_delay = 2` still updates the actor every
   second gradient step. `smoke` keeps its diagnostic batch 64.
2. **The post-update learning-potential batch is switched off** by the new
   planner key `update_learning_potential_diagnostic: false` in both
   `sac_sb3.yaml` and `td3_sb3.yaml`. The code default is `true` (the previous
   behaviour) so that the flag is an explicit opt-out; when off,
   `maybe_update` reports `learning_potential: None`, which the lifecycle and
   `compute_learning_potential` already handle, and the TD3 timing field
   `timing_acl_replay_learning_potential_seconds` is `0.0`.
3. **TD3 receives the same setting as SAC** (decided in the third approval
   step, closing `open_items` `D8`). The measurement was taken on SAC, but
   the TD3 backend has the same update structure — `auto -> n_envs` gradient
   steps of batch 256, four `lq_v3`-backed networks, update overlapped with
   the workers — and a different update-to-data ratio between the two
   off-policy arms would be a second varied factor in the `EVAL-PROTOCOL`
   algorithm comparison. **PPO is unchanged.** Its analogue would be
   minibatch `64 -> 128` (same samples, 320 -> 160 optimizer steps per
   2 048-transition rollout), but its update is synchronous rather than
   overlapped, it runs on CPU by the builder's choice, and no PPO
   `step_timing.csv` exists under `lq_v3`; the decision waits for the first
   measured PPO run (`open_items` `D9`).
4. The setting is **global**: it applies identically to every reward setting
   and every curriculum setting, so no comparison inside `EVAL-PROTOCOL`
   varies it. It is **not** applied to a run already in progress; a screening
   arm relaunched after this date runs under it, and both arms of a pair must
   share it (`AC-AB-002`).

## Rejected alternatives

- **`update_to_data_ratio 0.5` at batch 256.** Same learner cost, half the
  replay samples per collected transition. The user's constraint was no further
  test runs, and this is the only variant whose learning dynamics stay
  quantitatively close to the validated configuration without one.
- **`update_to_data_ratio 0.25` or lower.** No wall-clock gain beyond 0.5:
  once the learner drops under ~90 ms/step the worker's 175–185 ms becomes the
  binding ceiling, and it cannot be reduced without changing the Rulebook.
- **Encoder `lq_v3_lite`** (1.6× per learner pass). Excluded by `DEC-AB-006`,
  is labelled `diagnostic-only`, was never measured for learning quality, and
  would buy nothing on wall clock for the same reason as above.
- **More workers, FP16 autocast, batch 512 at ratio 1.0, `torch.compile`.**
  All measured or attempted earlier (`semantic_v3_lq_v3_..._exec_plan.md`,
  2026-07-22): ceiling at 20–21 workers, autocast and larger batch slower per
  update, `torch.compile` unavailable in the image.
- **Sub-sampling or thinning the Rulebook evaluation.** Changes the semantics
  of every per-transition rule; not a throughput knob.
- **Removing the learning-potential code instead of gating it.** `REQ-013`
  requires the diagnostic chain to remain available; a switch preserves it.

## Consequences

- Expected: learner ~179 -> ~95 ms/step; wall clock ~220 -> ~190 ms/step;
  ~3.9 -> ~5.2 fps; `medium` ~25 -> ~19 h, `thesis` ~107 -> ~80 h. These are
  projections from the measured decomposition, **not measurements**; the first
  run under this ADR must report its `step_timing.csv` and replace them.
- Learning behaviour is expected to be neutral to mildly different (larger,
  less noisy gradient estimates; half as many target updates per transition).
  No run has verified this; the next SAC run — the relaunched screening pair if
  the user relaunches it, otherwise the first core run — is the evidence, and
  a degradation visible against arm A's control would reopen this decision.
- Runs are no longer bit-reproducible against runs made before this ADR (batch
  size and buffer RNG consumption both changed). Determinism at fixed seed and
  configuration is preserved.
- `RL-BASELINES` v1 §3.4, `REQ-RLB-007` and §9.4 now describe the SAC minibatch
  and `auto` resolution incorrectly; this ADR amends them, following the
  precedent of ADR-075 (which amended `gamma` without rewriting the document).
- `ACL v2.0 REQ-013`'s statement that the retained diagnostic "introduces no
  additional learner work" is inaccurate for the post-update batch; the
  collection-time chain it names is untouched and still runs. When a run needs
  the post-update channel again, set `update_learning_potential_diagnostic:
  true` — it is one key, not a code change.
- Mandatory tests `test_sb3_sac_run_profiles_use_budget_appropriate_warmup_and_batch`
  and `test_td3_run_profiles_use_budget_appropriate_warmup_and_batch` change
  their expected batch from 256 to 512 on the non-smoke profiles, as a direct
  consequence of the approved decision. Regression tests:
  `tests/test_learner_update_throughput_adr078.py` (the SAC half was verified
  to fail on the pre-change backends on 2026-09-06: 9 of 22 cases; the TD3
  resolver cases were verified the same way after the extension).
