# Session Handoff — A/B Learnability Screening, 2026-09-02

**Branch note (2026-09-05):** the work described here was done on
`scenarionet-implementation` and has since been merged; it now lives on **`main`**,
which is where the follow-up runs should be launched.

Working record of the session that pre-registered the `AB-LEARN` screening,
cleared the two defects blocking it, and left the four screening runs ready to
launch. It is a handoff, not a contract: the authoritative documents are
`AB-LEARN`, `SMOKE-COV`, `RULEBOOK-V5.1`, `SCAL-V1.4` and `EVAL-PROTOCOL`.

## 1. What this work is for

`docs/open_items.md` `D4` records the user's sequencing decision of 2026-09-01:
the serious runs come **after** the A/B/C/D learnability tests. The question is
whether the `RULEBOOK-V5.1` six-level reward scalarized by `SCAL-V1.4` is
*optimizable* comparably to MetaDrive's native reward — and therefore whether the
rulebook and the scalarization can be frozen and stopped being revisited.

The factorial is the project's own historical design (`EVAL-PROTOCOL` v1.0 §229),
already implemented as a pairing primitive in
`analysis/tables/make_factor_effect_tables.py`:

| | ACL off | ACL on |
|---|---|---|
| native reward | **A** | C |
| rulebook + scalarization | **B** | D |

**C and D are blocked** on the unimplemented ACL (`open_items` `F5`;
`ACL-SN-EMA-001` v2.0 approved 2026-09-01 by ADR-077, no ExecPlan yet).

## 2. Decisions taken, all by the user

| ID | Decision |
|---|---|
| `DEC-AB-001` | Arm A is `reward=monitor_only`, **not** `reward=native`. `native` sets `behavior: "off"` and detaches the Rulebook wrapper entirely, leaving arm A with no compliance metric and making A–B incomparable |
| `DEC-AB-002` | Budget `run_profile=medium` (350 000 steps), seeds `[0, 1]`. A `REQ-018` diagnostic, recorded as `DEV-AB-001` |
| `DEC-AB-003` | Probe algorithm SAC only |
| `DEC-AB-005` | A passing screening freezes rulebook and scalarization **jointly**; the `thesis`-budget confirmation arrives as a by-product of the core runs that follow |
| `DEC-AB-006` | Encoder `lq_v3` (ENC-V1.3 production, 16 latents, depth 4), **not** `lq_v3_lite`, which the repository labels `architecture_version: diagnostic-only` |
| `DEC-COV-001` | The production-composition coverage gap is closed by a fast non-GPU test, not by a second smoke preset |
| — | The mission-station guard is enforced from the first tracked step; tolerance unchanged |
| — | Runs go **two at a time, paired by seed** (A and B at seed 0 together, then seed 1) |

**The ACL is frozen in the opposite order** to the rulebook, and this is normative:
`ACL-SN-EMA-001` v2.0 §11 requires approval to precede any comparison run under it.
Arms C/D therefore **measure** the curriculum; a C/D result showing no benefit is a
thesis result to report, not a defect to remove by adjusting the ACL until it helps.

## 3. Two defects found and fixed

Both on the `obs=semantic_v3` path, which no automated check had touched since
2026-08-01. The second was invisible until the first was cleared.

**`C5` — every `obs=semantic_v3` run was unstartable for a month.**
`envs/factory.py:128-133` injects three `semantic_v3_signal_*` keys;
`ThesisScenarioEnv.default_config()` declared none of them; MetaDrive's
`BaseEnv.__init__` merges with `allow_add_new_key=False` and raises `KeyError`
before the first reset. `make run-train` itself could not start. Invisible because
`make smoke` selects `obs=lidar_state` and `encoder=none`.

**`C6` — the evaluation path could not start.** `_ego_local_projection` required
the re-projected ego station to equal the tracker's committed station within
`1e-6 m`. Measured divergence at reset: **0.0229 m** on a 23.17 m route, because
`mission/runtime.py:281` constructs the tracker with a hard-coded
`initial_s_m=0.0`, so the guard compared a zero *by definition* against a
projection of the actual spawn pose. Magnitude consistent with the projection
jitter ADR-073 measured at up to 0.053 m.

**Method note worth carrying forward.** The magnitude *was* the diagnosis. The
original message reported no number, and both plausible readings — floating-point
noise and a stale snapshot — were excluded only once it was printed. Adding the
measured quantity to a fail-fast message costs nothing and converts an
unactionable error into a decision.

## 4. Measurements taken

**Encoder parameters** (exact, `scripts/benchmark_semantic_encoders.py`):

| | parameters | attention | latent FFN | token projectors |
|---|---:|---:|---:|---:|
| `lq_v3` | 1 120 000 | 528 384 | 527 360 | 11 072 |
| `lq_v3_lite` | 588 544 | 264 192 | 263 680 | 11 072 |
| `lq_v3_micro` | 322 816 | 132 096 | 131 840 | 11 072 |

Attention plus latent FFN are **94.3 %** of `lq_v3`. A defect was found and fixed
in the tool before reporting: the grouping folded `output_projection` into
`token_projectors`, overstating the latter by 33 536.

**Step-time attribution** (measured, `step_timing.csv`, arm B smoke, 2000 steps):

| component | total |
|---|---:|
| `learner_update` | **300.6 s** |
| `worker_wrapped_env_step` | 227.4 s |
| ├─ `rulebook_evaluator` | 87.9 s |
| └─ `env_step` (bare MetaDrive) | **30.5 s** |

**This corrects an earlier claim in this session** that the simulator dominates.
It does not: the GPU learner is the largest component and the bare simulator is
about 6 % of the training loop, with the rulebook costing three times MetaDrive.
The earlier claim came from two uncontrolled single runs.

## 5. `lq_v4`, the V-Max-aligned encoder

**It does not exist** — no `lq_v4`, no `ENC-V1.5`, no ReZero, no shared recurrent
core. `lq_encoder.py` still carries `token_to_latent` and four *independent*
blocks, the two structures such a redesign targets. Derived estimate for the
proposed architecture: **~304 000 parameters**, 3.7× fewer than `lq_v3` and below
even `lq_v3_micro`, while keeping 16 latents and four iterations.

**Recommendation given and accepted: do not implement it now.** It would reset the
A/B screening (the encoder is part of the function approximator); the wall-clock
case is weak because the four iterations still run and only the attention kernels
narrow; and the sample-efficiency case is unevidenced — ADR-036 already rejected a
comparable upgrade citing V-Max's own plateau (LQ 0.87 against LQH/MTR/Wayformer
0.84). **Caveat added after the step-time measurement**: the learner being the
largest component means a cheaper encoder may buy more wall clock than argued at
the time. This does not change the sequencing recommendation, only one of its
supporting arguments.

Two discrepancies between the drafted target and the repository, both to resolve
before drafting: the target states `D = 3009` but `RB51` took the observation to
**3011** via `OBS-V1.3.1`; and the premise that the full encoder was "too slow"
does not survive measurement. The V-Max primary sources are **not vendored** —
`third_party/` holds `metadrive`, `scenarionet`, `stable-baselines3` only — so the
exact ReZero semantics cannot be documented without fetching them.

## 6. Resource picture, 2026-09-02 16:41

| | state |
|---|---|
| GPU | 85 452 / 97 871 MiB, 100 % util, **~12 GB free**, three other tenants' processes |
| CPU | 72 cores, load average **3.67** — nearly idle |
| RAM | 573 GB total, **405 GB available**; four runs need ~29 GB of replay buffer |

The binding resource is the **GPU**, which is also the bottleneck component
(`learner_update`). Four runs in parallel would time-slice one contended device
and finish at roughly the same aggregate time as two plus two, with a real OOM
risk against 12 GB. Two at a time **paired by seed** is preferred because the
scientifically useful unit is the pair: a complete comparable (A, B) result at
~17–20 h, with the option not to spend the second pair.

## 7. State of the work

| milestone | status |
|---|---|
| `M1` presets and pre-registration | complete; single-factor invariance measured |
| `M2` per-arm smoke | complete; both arms `exit 0` at the third attempt |
| `M3` four screening runs | **ready**, not started |
| `M4` analysis and verdict | not started |

Both smoke arms produced seven CSV artifacts, `final_eval.csv` populated, no NaN
or infinity, `reward_behavior` correctly `monitor_only` and `scalar_reward`.

Full suite: **1600 passed** (4m11s), re-run *after* the `C6` fix, which changed
the signatures of `_ego_local_projection` and `_record_ego_frame`. The earlier
1598-passing run predates that fix and is superseded. `make lint` and focused
`make format-check` clean.

The exact launch command was validated with `--cfg job --resolve`: it composes to
`seed=0`, `total_timesteps=350000`, `eval_interval=25000`, `eval_episodes=50`,
`reward.behavior=monitor_only`, encoder `latent_query_v3`, experiment group
`SCREEN-LEARNABILITY-A-NATIVE-01`.

## 8. Open

- **`DEC-AB-004`** — authorization to launch. The user has approved the seed-0
  pair; the seed-1 pair is a separate decision.
- **`SMOKE-COV` `M2`** — a production-composition smoke, deferred by
  `DEC-COV-001`; if added, as an opt-in pre-launch target, not inside `make smoke`.
- **`C7`, the typed data abort works only on the vectorized path.** Investigated
  after this handoff was first written. The conversion machinery lives inside the
  subprocess vector environment — the worker catches the error and hands the parent
  a `RuntimeScenarioDataAbort` marker — while `RuleRewardWrapper` only annotates and
  re-raises. With `vectorized.enabled=false` nothing performs the bootstrap, so an
  ordinary data condition (`UNKNOWN` signal pre/post states) terminates training and
  the process exits 139 tearing the engine down. **Production is unaffected**: every
  official configuration is vectorized. It bites whoever disables vectorization to
  get a cleaner traceback while debugging.
- ~~**Commit hygiene**: another session committed `fd4f3a3` with a comfort-diagnostics
  message containing 16 files from four unrelated workstreams, this work included.~~
  **Resolved 2026-09-05** by the user: the repository was tidied, this handoff got its
  own scoped commit (`0a229bb`), and everything is merged into `main`.

## 9. Files touched in this session

| path | action |
|---|---|
| `conf/presets/learnability/sac_a_native.yaml` | added — arm A |
| `conf/presets/learnability/sac_b_rulebook.yaml` | added — arm B |
| `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | added — `AB-LEARN` |
| `docs/implementation/production_composition_startup_coverage_exec_plan.md` | added — `SMOKE-COV` |
| `scripts/benchmark_semantic_encoders.py` | added — encoder microbenchmark |
| `src/thesis_rl/envs/thesis_scenario_env.py` | modified — `C5` fix |
| `src/thesis_rl/envs/observations/causal_semantic.py` | modified — `C6` fix and the diagnostic magnitude |
| `tests/test_thesis_scenario_env.py` | modified — `C5` regression test |
| `tests/test_causal_semantic_batch.py` | modified — `C6` two-sided regression test |
| `docs/open_items.md` | modified — `C5` and `C6` recorded and closed |
| `docs/project_index.md` | modified — two registry rows |
