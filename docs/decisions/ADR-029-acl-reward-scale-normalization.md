# ADR-029: Per-Arm Reward-Scale Normalization and Positive-Part Off-Policy Learning Potential

- Status: APPROVED
- Date: 2026-07-27
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-27
- Supersedes: none
- Affected specifications: `automatic_curriculum_learning_v1.2_specification.md` (REQ-007, REQ-008)
- Affected ExecPlan: `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` (`ACL-SN-INIT-002`, `DEC-006`, `DEC-007`)
- Amended 2026-07-27 (same day, follow-up in the same session): extended `REQ-007` to also govern replay-buffer retention/eviction priority (`DEC-007`), not only the MAB feedback path. See "Amendment: `DEC-007`" below.

## Context

Measurement across 15 in-progress/completed runs (`FIND-004`) established that the ACL bandit's arm ordering stabilizes almost immediately (within ~7000 steps, one EMA horizon at `alpha=0.10`) and then stays essentially fixed for the rest of training (`MEAS-003`, median Spearman rank correlation `+0.66` between the first and last recorded chunk). The cause is not a bug in `_normalize_learning_potential`, `ScenarioArmBandit.update`, or the LP attribution path (`FIND-002`): all behave as specified. It is a property of the signal itself.

`delta = r + gamma*Q' - Q` carries the units of the return, so `mean(|delta|)` (and `mean(max(delta,0))` equally) is proportional to the reward scale of the arm producing it. The six semantic arms differ by 2x-4x in typical `|reward|`, driven by rulebook violation propensity — a structural property of the scenario class, not of remaining learnability. Because `_normalize_learning_potential` ranks every episode against a window shared across all arms (`RAT-002`), this static scale gap becomes a stable per-arm rank percentile within roughly one EMA horizon, and stays there because the reward scale does not move. Measured Spearman correlation between median LP and median `|reward|`, per arm, across four runs spanning three algorithms: `+0.771`, `+0.886`, `+0.829`, `+0.943`. Dividing raw LP by median `|reward|` per arm removes 27%-72% of the between-arm dispersion.

Separately, PPO's formula `mean(max(A,0))` already filters "worse than expected" outcomes (the ZPD hopelessness filter), but the production TD3/SAC formula `mean(|delta|)` does not: a negative residual (an outcome worse than the critic predicted) inflates LP identically to a positive one, which is not internally consistent with PPO's treatment of the same failure mode.

## Decision

Two changes, approved together because both require re-running the in-progress experiments and address complementary axes of the same signal:

**1. Per-arm reward-scale normalization (REQ-007).** Maintain a per-arm reward-scale EMA `s_i`, sharing `mab.alpha` (no new hyperparameter, per `RAT-004`/`RAT-007`'s parsimony argument), initialized to `1.0` identically across all arms. Because the initial value is shared, normalization is a no-op during warm-up (~10-60 updates per arm) and the correction phases in only as per-arm estimates diverge from the common prior — degrading gracefully to pre-`v1.2` behavior exactly when there is insufficient data to trust a per-arm estimate. Every committed episode's raw LP, Generate or Replay, is divided by `max(s_i, 1e-3)` (a numerical safety clamp, not a tuned parameter) using the pre-update estimate, before entering the shared rank window. Applying this to Replay episodes too, not only Generate, is required to preserve the softmax shift-invariance property (`RAT-002`/`MEAS-001`): normalizing only Generate LP would turn the previously-benign uniform level shift from Replay contamination into an arm-dependent scale shift, reintroducing the exact confound being fixed. The reward-scale-normalized value, `LP_scaled`, is also what is stored in `ScenarioRecord.learning_potential`/`usefulness` (`DEC-007`, amendment below); only diagnostic logs (`live_event_context["U"]`, `buffer_events`) keep the raw value.

**2. Positive-part off-policy learning potential (REQ-008).** `mean(|delta|)` becomes `mean(max(delta,0))` for TD3 and SAC, at the actual production site (`agent/planners/core/lifecycle.py: acl_learning_potential`, `acl_ready_learning_potentials`, per `FIND-002`) and its documented fallback (`curriculum/scenario_acl/usefulness.py`). This is not a step toward closer fidelity to the reference paper — Peng's Eq. 10 normalizes signed episode reward with a cumulative accumulator, a different construction already rejected as non-transferable (`RAT-001`). It is internal consistency: it gives TD3/SAC the same hopelessness filter PPO's `max(GAE,0)` already has.

**Declared limitation, not resolved by either change (`LIM-002`):** variance inflation. At critic convergence `E[delta]=0`, so both the pre-`v1.2` `mean(|delta|)` and the `v1.2` `mean(max(delta,0))` are proportional to `sigma(delta)` for zero-mean noise. A noisy-but-unlearnable scenario still yields nonzero LP. This predates this ADR, is not introduced by it, and affects PPO's existing formula identically — it is not a regression specific to this change. No fix within the prediction-error LP family is known. The principled fix is a learning-progress signal (slope of return per arm over repeated visits); explicitly out of scope here for lack of ablation budget and because it requires an architectural change, not a formula change.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Per-arm rank window (instead of reward-scale normalization) | Removes the scale confound by construction, no new state | At steady state every arm returns to `~0.5` (`RAT-002`), degrading the bandit toward uniform whenever no arm has a moving LP; a much weaker curriculum | User selected reward-scale normalization as the less invasive change |
| Normalize with a windowed per-arm median `|reward|` instead of an EMA | No decay-rate assumption | Requires a second window-size hyperparameter and more per-arm state (a list, not a scalar); no justification for a rate different from `mab.alpha` | Rejected for parsimony, consistent with `RAT-004`/`RAT-007` |
| Learning-progress signal (slope of return per arm) | Directly measures the intended quantity; immune to both the scale confound and variance inflation | Lagged (needs multiple visits per arm), architectural change, no ablation budget | Deferred; flagged as the principled fix for `LIM-002` |
| Leave TD3/SAC on `|delta|`, change only PPO's treatment | No off-policy code change | Leaves the inconsistency in place; already established that `|delta|` inflation is symmetric with PPO's pre-fix behavior, not a reason to leave it | Rejected; both changes bundled since both require re-running experiments anyway |
| Normalize `delta` per-transition instead of the aggregated per-episode LP | Finer granularity | Needs the scale estimate at transition granularity, complicates the sign-gate interaction, more state; no identified benefit over episode-level normalization | Rejected for simplicity; single point of change matching existing pipeline structure |

## Consequences

- `ScenarioArmBandit` checkpoint schema bumps from `acl_ema_v1` to `acl_ema_v2` (new persisted `reward_scale` field); `acl_ema_v1` checkpoints are rejected explicitly and cannot resume, consistent with `REQ-005`.
- The three in-progress experiments (TD3, SAC, PPO) must be stopped and restarted; this is a resource-affecting decision the user confirmed explicitly before implementation began.
- `live_event_context` gains a `U_scaled` diagnostic field (raw LP and rank-normalized LP were already logged; the reward-scale-normalized intermediate is now also visible) to allow future audits analogous to `FIND-004`/`MEAS-003` without re-deriving the pipeline.
- Prior completed runs remain valid under their recorded `v1.1` configuration; they are not retroactively affected.

## Amendment: `DEC-007` — extend REQ-007 to replay-buffer retention priority

The initial version of this ADR deliberately excluded `ScenarioRecord.learning_potential` (used by `buffer.insert`'s worst-of-buffer eviction) from `REQ-007`'s normalization, recording it as a residual limitation. On the same day, the user asked directly whether it should be normalized too. Re-derivation: `exploit_probability=0.60`, i.e. 60% of post-warm-up training time is Replay, sampled from this buffer. If buffer composition remains skewed toward high-reward-scale arms (measured corroborating evidence, `MEAS-002`: `A4_vru=333` vs `A0_simple_low_traffic=18` of 1000 retained slots) after `REQ-007` fixes only the Generate-side MAB feedback, the same root cause (`FIND-004`) continues to bias training time through a second, un-fixed channel. Since the three experiments already require a restart for `DEC-006`, deferring this would risk a second restart later at strictly higher cost than fixing it now.

**Decision:** `_build_record_from_catalog_entry` and `_update_replay_record` (`driver.py`, both the vectorized `commit_event` path and the sequential `collect_catalog_episode` path) receive `LP_scaled`, not raw `LP_i`, as their `learning_potential` argument. This is the same value already computed for the rank window in `REQ-007`; no new computation, only a different consumer of an existing value. Diagnostic fields (`live_event_context["U"]`, `buffer_events`, `log_event`) are unaffected and continue to report the raw value.

**Consequence:** `LIM-001` (from the original ADR text) is resolved, not merely narrowed. The `MEAS-002` buffer-composition evidence cited above will not reproduce identically under `v1.2`, by design — that is the intended effect, not a regression to explain.

**Consequence for validation:** the call-site wiring (`lp_scaled`/`scaled_episode_usefulness` substituted for `lp`/`episode_usefulness` at the two buffer-record construction sites) is verified by direct code reading and by the unchanged, still-passing focused test suite, but has no dedicated closure-level regression test — `commit_event` and `collect_catalog_episode` are private closures inside the ~800-line vectorized/sequential training-loop functions with no existing isolated-testing harness. Recorded as `LIM-004` in the specification, with the follow-up test described there.

## Validation And Traceability

- `tests/test_scenario_acl_mab.py`: reward-scale initialization/EMA update, floor clamping, checkpoint round-trip, and legacy-schema rejection.
- `tests/test_scenario_acl_usefulness.py`: `compute_td3_learning_potential`/`compute_sac_learning_potential` use `max(delta,0)`, including an all-negative case yielding `LP=0`.
- `tests/test_scenario_acl_vectorized_state.py`: production-path regression at `_DelegatingLifecycle.acl_learning_potential` confirming positive-part behavior at the actual collection-time attribution site identified in `FIND-002`, not only at the documented fallback.
- Full targeted suite (`tests/` filtered to `acl`/`lifecycle`/`curriculum`, 102 tests) and repository lint/format-check on all touched files pass; recorded in `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` §14.

## Approval Record

- Approved by: user
- Approval evidence: explicit design confirmation during the 2026-07-27 session (choice of `R-B` combined with the positive-part formula change, with explicit parameter guidance for the reward-scale estimator: shared floor of `1.0`, delegated remaining choices), followed by "D'accordo comunque, procedi con l'implementazione"
