# ExecPlan — Reward Learnability A/B Screening (`AB-LEARN`)

## 1. Metadata

| Field | Value |
|---|---|
| Feature | Pre-registered A/B screening of reward learnability: MetaDrive's native reward against the `RULEBOOK-V5.1` + `SCAL-V1.4` scalarized reward |
| Plan ID | `AB-LEARN` |
| Authoritative specifications | `EVAL-PROTOCOL` v1.0 (`docs/specifications/evaluation_protocol_v1.0_specification.md`, `AUTHORITATIVE`) with v1.1/v1.2/v1.3/v1.3.1 amendments; `RULEBOOK-V5.1` (`docs/specifications/rulebook_v5.1_specification.md`, `AUTHORITATIVE`); `SCAL-V1.4` (`RULEBOOK-V5.1` §5) |
| Status | `IN_PROGRESS` — `M1` and `M2` complete and verified; `M3` seed-0 pair launched 2026-09-05 21:47 UTC and **both runs died 2026-09-06** (`C9`, CUDA OOM); relaunch blocked on the arm-A control-validity decision (`open_items` `C11`) and on ADR-078; the seed-1 pair is a separate decision |
| Created | 2026-09-01 |
| Last updated | 2026-09-05 |
| Branch | `main` (work done on `scenarionet-implementation`, merged 2026-09-05) |
| Related ADRs | ADR-063…ADR-076 (the rulebook under test), ADR-077 (ACL v2.0, which gates arms C/D), ADR-026 (`D4` sequencing) |

This plan pre-registers an experiment. It changes no production behavior: its
only repository artifacts are two Hydra presets and this document.

## 2. Objective And Scope

**Observable capability.** A decision, backed by measurement, on whether the
`RULEBOOK-V5.1` six-level margin vector scalarized by `SCAL-V1.4` is
*optimizable* by a standard continuous-control learner in a way comparable to
MetaDrive's native reward — and therefore whether the rulebook and the
scalarization function can be frozen and stopped being revisited.

**Why it is needed.** `docs/open_items.md` `D4` records the user's sequencing
decision of 2026-09-01: the serious runs come *after* the A/B/C/D learnability
tests. `RULEBOOK-V5.1` establishes that the rulebook is *correct* — it does not
pay the human expert a nonsensical return (expert mean `+70.70`, `3.36 %` of
episodes below standstill, §5.5) — but correctness of the preference order and
optimizability by gradient methods are different properties, and no training
run exists under v5.1 beyond `make smoke`.

**How success is recognized.** Not by "arm B wins". By arm B showing a genuine
*learning trend* on the driving metrics and not exhibiting the historical
pathology in which scalar return rises while driving quality degrades. See the
decision rule in §7.

**In scope.** Arms A and B of the A/B/C/D factorial, at reduced budget, with
the curriculum disabled; the two presets that express them; the analysis of the
resulting artifacts through the existing `make analyze` pipeline.

**Out of scope.** Arms C and D (they require the ACL, `open_items` `F5`, not
implemented — `ACL-SN-EMA-001` v2.0 was approved 2026-09-01 but has no
ExecPlan); algorithm selection; hyper-parameter tuning; observation or encoder
selection; any change to the rulebook, the scalarization, the termination
contract or the evaluation panels. If arm B fails, the response is analysis,
not an immediate specification change.

**Compatibility.** No public interface, checkpoint schema, dataset or metric
convention changes. The two new presets are additive.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-AB-001` | A reduced-budget, reduced-seed comparison is a descriptive engineering diagnostic and may never be promoted post hoc into a core result; it carries its own comparison-block ID | `EVAL-PROTOCOL` v1.0 `REQ-018`, `DEC-007` |
| `REQ-AB-002` | Conditions are compared under an identical environment-interaction budget and an identical evaluation protocol | `EVAL-PROTOCOL` v1.0 `REQ-002` (§8 of the superseded comparison protocol) |
| `REQ-AB-003` | Seeds are the independent replication unit; the same seed list is used for every condition, and a failed run is relaunched on the same seed rather than replaced | `EVAL-PROTOCOL` v1.0 `REQ-003`, `DEC-001` |
| `REQ-AB-004` | All conditions use identical ordered validation and test panels, independent of the training seed | `EVAL-PROTOCOL` v1.0 `REQ-004` |
| `REQ-AB-005` | Reporting is descriptive: raw seed-level values plus mean and sample standard deviation; no confidence interval, bootstrap or significance test is claimed from this seed count | `EVAL-PROTOCOL` v1.0 `DEC-003` |
| `REQ-AB-006` | `route_completion` is the primary continuous metric; the geometric gate remains the binary success event; nuPlan's making-progress gate at `0.2` zeroes degenerate episodes in evaluation only | `EVAL-PROTOCOL` v1.3 |
| `REQ-AB-007` | Never publish a single score pooled across sources; report Waymo and PG separately | `open_items` `D5`, closed 2026-09-01 |
| `REQ-AB-008` | Every Waymo success figure is reported alongside the feasibility ceiling | `EVAL-PROTOCOL-V1.3.1` |
| `REQ-AB-009` | The reward under test is `RULEBOOK-V5.1`'s six-level vector scalarized by `SCAL-V1.4` as configured in `conf/scalarization/default.yaml`; no scalarization parameter is tuned during the screening | `RULEBOOK-V5.1` §5 |

## 4. Current Repository Analysis

All statements below are `VERIFIED` on 2026-09-01 unless labeled otherwise.

| Fact | Evidence |
|---|---|
| `reward=native` disables the rulebook wrapper entirely and therefore produces no compliance metrics | `conf/reward/native.yaml`: `behavior: "off"`, `rulebook_config: none`; `runtime/wiring/builders.py:413` returns the bare env for `mode == "off"` |
| `reward=monitor_only` keeps the identical Rulebook v2 instrument attached and withholds only the scalarizer | `runtime/wiring/builders.py:372-411`: both behaviors construct `RulebookV2MonitorWrapper`; `scalarizer` is built only when `behavior == "scalar_reward"` |
| Both reward configs inherit `rulebook_config: selection`, so the analysis pipeline can pair them | `conf/reward/rulebook_defaults.yaml`; `analysis/tables/make_factor_effect_tables.py:161` keys on `rulebook_config` and pairs `monitor_only` against `scalar_reward` |
| The curriculum factor is already an analysis primitive, so arms C/D need no new machinery once the ACL exists | `analysis/tables/make_factor_effect_tables.py:140` pairs `curriculum_enabled` false/true |
| The authoritative scalarization is the top-level `scalarization` group, not the legacy `a`/`scales` keys under `reward` | `runtime/wiring/builders.py:373-382` consumes `cfg.scalarization`; `conf/scalarization/default.yaml` declares `SCAL-V1.4`, `priority_base: 2.2`, `mode: six_level_priority_weighted_rank` |
| The historical 3×2 factorial presets under `conf/presets/td3/` are stale: they select the legacy `td3` backend, predate `RULEBOOK-V5.1`, and one arm uses `reward: native` | `conf/presets/td3/*.yaml`; `EVAL-PROTOCOL` v1.0 §229 records them as retained `REQ-018` ablation material |
| The project already expresses a "native contract" arm as `monitor_only` | `conf/presets/selection/sac_sb3_native_contract_fast.yaml` |
| `make run-train` hard-codes `curriculum=scenario_acl_scenarionet` and cannot express arm A or B without overrides | `Makefile:459` |
| `env` defaults already pin `provider.strict: true`, `allow_fallback: false`, `source_probability` 0.5/0.5 and `num_scenarios: -1`; only vectorization needs enabling | `conf/env/scenarionet.yaml` |
| The prior four-configuration diagnostic (2026-08-08, SAC-lite, 120k steps, one seed) is void as evidence | `docs/audits/acl_v2_teacher_power_analysis_2026-07-31/README.md:284`; the reward it measured was subsequently falsified (expert mean `-203.35`) and replaced |
| That diagnostic's methodological finding survives and motivates this plan's design | Same source, §"The improvement is a level shift, not a learning trend": at 120k the ranking of configurations was not stable, and only the native arm showed a real slope (`route +0.087` early→late) |
| Arm C (native reward + ACL) has never been run in any form | The 2026-08-08 set substituted a rulebook variant for that cell |

## 5. Assumptions And Invariants

- **Single varied factor.** `VERIFIED` by diff of the two fully resolved Hydra
  configurations: the only differences are `reward.name/type/behavior`, the
  dependent `lambda_env`/`lambda_rule`, and the run-identity paths. Algorithm,
  encoder, decoder, observation, environment, budget, seed, evaluation panels
  and the `scalarization` block are bit-identical. In arm A the `scalarization`
  block is present but inert, because no scalarizer is constructed.
- **Encoder.** `lq_v3` (`latent_query_v3`, ENC-V1.3), the production
  architecture, identical in both arms.
- **Expected wall clock.** Anchored on a real `medium` SAC run of 2026-07-27
  (`EXP_sac-lite-cmp_RP_medium`, 21 envs, curriculum enabled): 350,000 steps in
  **14.6 h** at 6.68 fps. With the production encoder, roughly 16-17 h per run,
  so four runs are about 2.5 days sequentially. `INFERRED` from the
  micro/lite throughput ratio, not measured for `lq_v3`.
- **Budget.** `run_profile=medium`, `experiment.total_timesteps = 350000`
  environment transitions per run, `eval_interval = 25000`,
  `eval_episodes = 50`, `final_eval_episodes = 100`.
- **Seeds.** `[0, 1]`, identical across both arms. `seed` also seeds the
  ScenarioNet provider (`env.global_seed: ${seed}`), so record choice and
  learner belong to one reproducible run.
- **Source mix.** Waymo 0.5 / PG 0.5. `RULEBOOK-V5.1` `M8a` established that
  five of six L3 sub-rules never apply on PG, so the two sources are graded by
  substantially different rulebooks — hence `REQ-AB-007`.
- **Termination/truncation.** Unchanged from production: at-fault contacts
  terminate and are charged (ADR-071); not-at-fault contacts cost zero and
  truncate through the `MAX_STEP` channel. `γ = 1` (ADR-075).
- **Units.** L4 is signed route advance normalized by `D_REF = v_ref · Δt = 2.2222 m`
  (ADR-073); speeds in m/s; the vehicle is capped at `max_speed_km_h = 80`.
- **Instrument identity.** Both arms report the same `RULEBOOK-V5.1` margins,
  so any compliance difference is a difference in behavior, not in measurement.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-AB-001` | Implementation detail | How is arm A's "native reward" expressed? | `reward=native` (no instrument) / `reward=monitor_only` (instrument kept, native reward trains) / both | `monitor_only` | Without it arm A yields no compliance metrics and the A–B contrast collapses to MetaDrive's own metrics | **Approved** (user, 2026-09-01) |
| `DEC-AB-002` | Specification clarification | What budget licenses the freeze decision? | `thesis` 1.5M × 3 seeds (conclusion-grade under `REQ-018`) / `medium` 350k × 2 seeds (diagnostic) / staged | `medium` first, then escalate | A `medium` result is descriptive only and cannot by itself close the freeze; it de-risks before committing 6 × 1.5M runs | **Approved** (user, 2026-09-01): `medium`, seeds `[0, 1]` |
| `DEC-AB-003` | Implementation detail | Probe algorithm | SAC only / TD3 only / both | SAC only | The question is about the reward, not the learner; algorithm selection is a separate, later step | **Approved** (user, 2026-09-01) |
| `DEC-AB-006` | Implementation detail | Which encoder? | `lq_v3` (ENC-V1.3 production, 16 latents, depth 4) / `lq_v3_lite` (`architecture_version: diagnostic-only`) | `lq_v3` | A screening that licenses a freeze must not be able to fail for lack of encoder capacity, which is indistinguishable from an unlearnable reward. Measured: micro (4, 1) 7.71 fps against lite (8, 2) 6.68 fps on `medium`, so the simulator dominates and the production encoder costs roughly 15 %, not a multiple | **Corrected 2026-09-01**; the first draft inherited `lq_v3_lite` from the `Makefile` `ENCODER` default |
| `DEC-AB-004` | Blocking technical issue | Authorization to consume GPU time for four runs | Launch now / wait for the GPU to free / stage sequentially | Two at a time, **paired by seed** | GPU 0 is shared with other tenants and was at ~12 GB free with `learner_update` measured as the largest component of the loop, so four concurrent runs time-slice one contended device for no aggregate gain and risk OOM. Pairing by seed also makes the first complete (A, B) comparison available at ~17-20 h | **Resolved 2026-09-05**: the seed-0 pair is authorized; the seed-1 pair is a separate decision the user has not taken |
| `DEC-AB-005` | Specification clarification | What exactly does a passing screening license freezing? | Rulebook + scalarization jointly / scalarization only / nothing without the `thesis` budget | Freeze both on a passing screening; the `thesis`-budget confirmation arrives as a by-product of the core runs that follow | Determines whether `D1`'s `τ₄` and the algorithm-selection phase can start | **Approved** (user, 2026-09-01). See §7.5 |

`DEC-AB-004` is resolved for the seed-0 pair. **No open gate remains for `M3` at
seed 0**; launching the seed-1 pair requires a separate user decision.

## 7. Proposed Design

### 7.1 Arms

| Arm | Reward driving the learner | Rulebook instrument | Curriculum | Comparison-block ID |
|---|---|---|---|---|
| **A** | MetaDrive native | attached, measured, not optimized | disabled | `SCREEN-LEARNABILITY-A-NATIVE-01` |
| **B** | `SCAL-V1.4` over the `RULEBOOK-V5.1` six-level vector | attached, measured **and** optimized | disabled | `SCREEN-LEARNABILITY-B-RULEBOOK-01` |
| C | native | attached | ACL enabled | *blocked on `F5`* |
| D | `SCAL-V1.4` | attached | ACL enabled | *blocked on `F5`* |

### 7.2 Pre-registered hypotheses

- `H1` (**learnability**): under arm B, `route_completion` and `success_rate`
  improve from the early to the late portion of training by more than the
  within-arm seed spread. This is the property the 2026-08-08 diagnostic found
  *absent* under the old reward, and it is the primary question.
- `H2` (**comparability**): arm B's late-training driving metrics are not
  materially worse than arm A's. "Materially worse" is read against the seed
  spread, descriptively — no significance is claimed (`REQ-AB-005`).
- `H3` (**no inverted incentive**): under arm B, scalar return and driving
  quality move together. The falsified historical pathology was return rising
  while off-road rate rose with it. Operationalized as the sign of the
  chunk-wise association between `mean_scalar_rule_reward` and
  `route_completion` within arm B.
- `H4` (**compliance dividend**): arm B's rulebook violation metrics are lower
  than arm A's. This is what optimizing the rulebook is *for*; it is expected
  but is not the freeze criterion, because a policy can reduce violations by
  refusing to move — which `H1` and the below-standstill fraction detect.

### 7.3 Decision rule, pre-registered

- `H1` holds and `H3` holds → the screening **passes**; escalate to the
  `thesis`-budget confirmation before declaring the freeze (`DEC-AB-005`).
- `H1` fails → the reward is not optimizable as configured. Diagnose before
  changing anything: first distinguish a *learner* problem (arm A also flat)
  from a *reward* problem (arm A learns, arm B does not). The 2026-08-08
  precedent shows the second is the informative case.
- `H3` fails → stop. An inverted incentive is a rulebook or scalarization
  defect and reopens `RULEBOOK-V5.1`, which is exactly what this screening
  exists to detect before 1.5M-step runs are spent.
- `H2` fails while `H1` and `H3` hold → record it and continue; a rulebook
  reward that trades raw task performance for compliance is the intended
  trade-off, not a failure, provided the loss is bounded and reported.

### 7.5 What a passing screening licenses, and what it does not

Approved 2026-09-01. A passing screening freezes **the rulebook and the
scalarization jointly**, and the algorithm-selection phase and `D1`'s `tau_4`
may then start.

Three qualifications make that coherent rather than a shortcut.

1. **The two components are not in the same position.** The rulebook's
   *correctness* is already established by falsification against 1100 logged
   Waymo records and by the O1-O6 orderings on constructed fixtures; the
   screening adds the second half only -- that the preference order is
   optimizable by gradient methods. The **scalarization** is the component the
   screening actually tests: `SCAL-V1.4` was verified only for
   rank-preservation, which is a mathematical property, and `RULEBOOK-V5.1`
   sec. 5.5 records that `eta` "is the one weight in this document not pinned by
   measurement" because the expert almost never relaxes and the panel cannot
   discriminate. This screening is the first and only empirical test that
   function will receive before the core runs.

2. **A `medium` result can license the freeze even though it cannot be a
   reported number.** The screening exists to stop the core runs being wasted,
   not to produce a thesis figure. On a pass, the freeze is declared, the core
   runs proceed at the `thesis` budget as `EVAL-PROTOCOL` requires, and those
   runs are what produce the reportable quantities. The residual risk accepted
   by `DEV-AB-001` is a pathology that only appears beyond 350k steps; the gross
   failure modes -- the ones that would waste six full-budget runs -- are
   exactly what the screening detects.

3. **The gate is legitimate only because it was pre-registered.** A negative
   `H3` licenses reopening `RULEBOOK-V5.1` because sec. 7.3 defined an inverted
   incentive as a *defect* before any run existed, not because the result was
   disliked. A gate declared after seeing the numbers would be the post-hoc
   promotion `REQ-018` prohibits.

**The ACL is frozen in the opposite order, and this is normative, not stylistic.**
`ACL-SN-EMA-001` v2.0 sec. 11 requires that the specification "must be approved
and frozen **before** any comparison run under it begins", because "the claim
that no parameter was chosen by observing run performance is only defensible if
the approval precedes the runs". The specification was accordingly approved on
2026-09-01 (ADR-077), before any learnability run. Arms C and D therefore
**measure** the curriculum; they are not an acceptance gate for it. A C/D result
showing that the curriculum does not help is a thesis result to be reported, not
a defect to be removed by adjusting the ACL until it helps -- that adjustment
would forfeit the ACL as a scientific claim. The ACL's remaining freeze step is
implementation verified against the approved specification (`F5`), which is
independent of what C/D subsequently measure.

### 7.4 Artifacts and analysis

Both arms write the standard run tree (`csv/`, `artifacts/`, `checkpoints/`,
`videos/`). Analysis uses the existing pipeline; the paired factor-effect table
for the reward factor is produced by
`analysis/tables/make_factor_effect_tables.py`, which is excluded from
`make analyze` by default per `REQ-018` and must be requested explicitly.

Reporting obligations: per source (`REQ-AB-007`), with the Waymo feasibility
ceiling (`REQ-AB-008`), seed-level values plus mean and sample standard
deviation only (`REQ-AB-005`).

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-AB-001` | `AC-AB-001` | `conf/presets/learnability/*.yaml` (`analysis.experiment_group`) | `TEST-AB-001` | Implemented, unverified |
| `REQ-AB-002` | `AC-AB-002` | `override /run_profile: medium` in both presets | `TEST-AB-002` | Implemented, verified |
| `REQ-AB-003` | `AC-AB-003` | `seed=` override at launch | `TEST-AB-004` | Planned |
| `REQ-AB-004` | `AC-AB-004` | `evaluation: scenarionet_panels` inherited unchanged | `TEST-AB-002` | Implemented, verified |
| `REQ-AB-005` | `AC-AB-005` | Report only | — | Planned |
| `REQ-AB-006` | `AC-AB-006` | Existing evaluation runtime | — | Pre-existing |
| `REQ-AB-007` | `AC-AB-007` | Report only | — | Planned |
| `REQ-AB-008` | `AC-AB-008` | Report only | — | Planned |
| `REQ-AB-009` | `AC-AB-009` | `scalarization: default` inherited unchanged | `TEST-AB-002` | Implemented, verified |

## 9. Test Strategy Defined Before Implementation

Acceptance criteria:

- `AC-AB-001` — each arm resolves to its own `analysis.experiment_group`, and
  neither collides with `BASELINE-SCALAR-01` or `EXTENSION-ALGORITHM-01`.
- `AC-AB-002` — the two fully resolved configurations differ **only** in the
  reward group and the derived run identity.
- `AC-AB-003` — both arms complete on seeds `0` and `1`; a crashed run is
  relaunched on the same seed.
- `AC-AB-004` — both arms report `final_eval` over identical panels.
- `AC-AB-005` — the report contains seed-level values, mean and sample standard
  deviation, and no CI or p-value.
- `AC-AB-006` — `route_completion` is the primary reported continuous metric.
- `AC-AB-007` — no metric is published pooled across Waymo and PG.
- `AC-AB-008` — every Waymo success figure is accompanied by the feasibility ceiling.
- `AC-AB-009` — the resolved `scalarization` block equals `conf/scalarization/default.yaml` in both arms.

Mandatory matrix:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-AB-001` | Config | Each preset composes and carries a distinct comparison-block ID | `--cfg job --resolve` on each preset | Exit 0; `SCREEN-LEARNABILITY-A-NATIVE-01` / `SCREEN-LEARNABILITY-B-RULEBOOK-01` | `REQ-AB-001` |
| `TEST-AB-002` | Config | Single-factor invariance | Diff of the two resolved configurations | Differences confined to `reward.*` and run identity | `REQ-AB-002`, `REQ-AB-004`, `REQ-AB-009` |
| `TEST-AB-003` | Smoke | Each arm starts, logs, evaluates and terminates | `run_profile=smoke` override on each preset | Exit 0; `final_eval.csv` present; no NaN | `REQ-AB-002` |
| `TEST-AB-004` | Integration | Both seeds complete at the screening budget | The four screening runs | Four completed runs, 350k transitions each | `REQ-AB-003` |
| `TEST-AB-005` | Regression | The repository suite is unaffected by the added presets | `uv run --no-sync python -m pytest -q` | No new failures | — |

Commands (all verified to exist in this repository):

- Config composition: `docker compose -f compose.yaml run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name presets/learnability/<arm> --cfg job --resolve`
- Smoke: `make smoke`
- Full suite: `make test`
- Lint: `make lint`
- Analysis regeneration: `make analyze RUN_PROFILE=medium ANALYSIS_ARGS="--include-effects-tables"`

No mypy target exists; static checking is not part of this plan.

## 10. Milestones

### `M1` — Presets and pre-registration — **complete**

- Objective: express arms A and B as two Hydra presets differing in one factor, and record the hypotheses and decision rule *before* any run.
- Files: `conf/presets/learnability/sac_a_native.yaml`, `conf/presets/learnability/sac_b_rulebook.yaml`, this document.
- Evidence: `TEST-AB-001` and `TEST-AB-002` executed 2026-09-01, both `PASS` (§14).
- Dependencies: `DEC-AB-001`, `DEC-AB-002`, `DEC-AB-003` — all approved.

### `M2` — Per-arm smoke — **complete**

- Objective: confirm each preset starts, evaluates and terminates before spending the screening budget.
- Tests: `TEST-AB-003` -- **PASS**, both arms `exit 0` on 2026-09-02.
- Evidence: seven CSV artifacts per arm, `final_eval.csv` populated, no NaN or infinity, `reward_behavior` recorded as `monitor_only` and `scalar_reward` respectively.
- **It took three attempts and uncovered two independent defects**; see the findings log.
- Dependencies: none.

### `M3` — Screening runs — **running at seed 0**

- Objective: four runs — arms A and B × seeds 0 and 1 — at `run_profile=medium`.
- Tests: `TEST-AB-004`.
- Dependencies: `DEC-AB-004`, resolved for seed 0 on 2026-09-05. The seed-1 pair
  needs a separate user decision.

**Execution.** Two runs in parallel, **paired by seed**, each in its own named
`tmux` session teeing to a log, so the job survives an SSH disconnect and the user
can attach without asking:

```bash
tmux new-session -d -s ab_a_seed0 \
  "docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
     uv run --no-sync python -m thesis_rl.cli.train \
     --config-name presets/learnability/sac_a_native seed=0 \
   2>&1 | tee /tmp/ab_a_seed0.log"
```

The `seed=0` override is required: `conf/config.yaml` defaults to `42`. Arm B is
the same with `presets/learnability/sac_b_rulebook` and session `ab_b_seed0`.
`scripts/tmux_seed_grid.sh` does not fit: it runs *one* command across many seeds,
where this needs *two* commands at one seed.

**Operational finding, 2026-09-05: `| tee` silences the live monitor.** Piping the
command makes stdout a pipe, so `docker compose run` allocates no TTY
(`docker inspect --format '{{.Config.Tty}}'` returns `false` on both containers),
`rich` reports a non-terminal console, and the `rich.Live` training monitor at
`agent/agent.py:1127` renders nothing for the whole run. The pane is not dead --
`print_evaluation_summary` uses `Console.print`, which writes plain text and flushes,
so the per-evaluation table still reaches pane and log -- but the between-evaluation
step counter, fps, EMA rewards and critic loss are lost, and they exist nowhere else:
`_LiveEventLogHandler` buffers into an in-memory `deque(maxlen=8)`, not a file. There
is therefore **no intra-chunk step counter in any artifact**; the first exact step
figure is the chunk boundary at 25,000.

Accepted for the seed-0 pair (user, 2026-09-05): the runs continue as launched, because
recovering the TTY costs a relaunch and `Tty` is fixed at container creation. For the
seed-1 pair, launch **without** the pipe so the pane keeps its TTY. Do not then capture
the pane wholesale: the monitor refreshes 8 times a second
(`TRAINING-MONITOR-REFRESH-V1`), so `tmux pipe-pane` or `script` would record every
redraw and produce tens of gigabytes of ANSI over a 17-hour run. The durable record is
already the run's own `logs/train.log`, `logs/events.jsonl`, `logs/errors.log` and the
CSV artifacts; the stdout log adds little beyond a crash traceback.

**Monitoring protocol.** At every new `eval_type=intermediate` row in
`csv/evals.csv` — 14 per run — report a cumulative table with both arms side by
side (`global_step`, `route_completion`, `success_rate`, `collision_rate`,
`out_of_road_rate`, `mean_reward`, and `mean_scalar_rule_reward` for arm B only,
since arm A builds no scalarizer), plus a short prose reading.

Report `validation_waymo_empirical` and `validation_pg` **separately**:
`open_items` `D5` forbids a score pooled across sources, because five of the six
L3 sub-rules never apply on PG and the two numbers do not measure the same
rulebook.

Take metrics from the CSV artifacts, not from the log: they are structured,
complete and unaffected by `tmux` scrollback. The log is for diagnosing failures,
which is exactly where a truncated history hurts.

**What the three observations mean**, since they answer different questions:

1. **Slope, not level** (`H1`, the primary criterion). Whether the reward is
   *optimizable* is a claim about change across evaluations, not about where the
   metric sits at 350k steps. The 2026-08-08 diagnostic is the precedent: the
   scalar arm had the better *level* on route (0.259 against 0.193) and a
   *negative* slope (−0.041), while native had the worse level and the only
   clearly positive slope (+0.087). Ranking by level would have selected the arm
   that was getting worse.
2. **Arm B flat while arm A rises.** This is why the control exists. Both flat
   means the *setup* cannot learn and nothing has been learned about the rulebook;
   A rising with B flat isolates the **reward** as the only differing factor and is
   the one configuration that licenses a conclusion about it.
3. **`H3`, inverted incentive — report immediately, do not wait for the next
   evaluation.** `mean_scalar_rule_reward` *is* arm B's objective. If it rises
   while `route_completion` falls, `collision_rate` rises, `ep_len_mean` collapses
   or mean speed tends to zero, then the optimizer is working correctly and the
   **objective is wrong**: the agent has found a way to score well that is not
   driving well. This is the pathology already falsified once — the pre-v5.1
   rulebook paid the human expert **-203.35** with 46.55 % of expert episodes below
   standstill, so standing still beat driving. Offline falsification against expert
   replay proves the good driver scores well; it cannot prove that no *other*
   behavior scores better while driving badly, because it holds one trajectory per
   scenario and no counterfactual. **Only an optimizer can find that, and this run
   is the first adversarial search ever run against this reward.** If `H3` fires the
   run is worthless and `RULEBOOK-V5.1` reopens, which is a user decision, not a fix.

### `M4` — Analysis and verdict — **not started**

- Objective: evaluate `H1`–`H4` against §7.3 and record the verdict.
- Dependencies: `M3`; `DEC-AB-005` for what the verdict licenses.

## 11. Progress And Findings Log

**2026-09-01.** Plan created. Recovered the A/B/C/D design from `open_items`
`D4`, from the user's message of 16:12 the same day, and from `EVAL-PROTOCOL`
v1.0 §229, which shows the factorial is the project's own historical design and
is already implemented as a pairing primitive in the analysis pipeline.

Finding, material: `reward=native` would have made arm A unmeasurable on
compliance, because it detaches the rulebook wrapper. Resolved as `DEC-AB-001`
by using `monitor_only`, which is also the convention the repository already
follows in `conf/presets/selection/sac_sb3_native_contract_fast.yaml`.

Finding, methodological: the 2026-08-08 four-configuration diagnostic is void
as evidence — its reward was subsequently falsified — but its *method* finding
survives and shaped `H1`: at 120k steps the ranking of configurations was not
stable, and level differences between arms were not the same quantity as the
learning slope within an arm. This plan therefore pre-registers the slope, not
the level, as the primary criterion.

`TEST-AB-001` and `TEST-AB-002` executed and passed; the resolved-config diff
confirms single-factor invariance by measurement rather than by inspection.

**2026-09-01, later.** `DEC-AB-005` resolved by the user: a passing screening
freezes the rulebook and the scalarization jointly. Recorded as sec. 7.5, together
with the asymmetry the user's question surfaced -- the ACL is frozen *before* its
comparison runs, not after them, under `ACL-SN-EMA-001` v2.0 sec. 11's
pre-registration requirement, so arms C/D measure the curriculum rather than
gating it. `DEC-AB-004` (GPU authorization) remains the only open gate; the user
is arranging for memory to be freed.

**2026-09-01, third pass.** The user challenged the framing of the ACL freeze and
asked for the encoder and budget facts. Three outcomes.

*Framing corrected.* The previous entry conflated two questions that the ACL
specification keeps apart: **verification** that the implementation works, which
is `F5`, must precede the freeze and a failure there is a defect to fix; and the
**C/D comparison**, whose outcome §11 forbids feeding back into parameter
choice. The specification is not a straitjacket on the first: `DEC-205` states
that the deferred `SNR` measurement for `H` is "a post-hoc validation, **not a
gate**", with `H = 30` pre-declared as the revision if it returns below `0.5`.
A pre-declared revision driven by a real learning run is legitimate precisely
because it was declared in advance.

*Encoder corrected.* `DEC-AB-006`. The first draft inherited `lq_v3_lite` from
the `Makefile` `ENCODER` default; the repository labels it
`architecture_version: diagnostic-only`. Both presets now select `lq_v3`, and
`TEST-AB-001`/`TEST-AB-002` were re-run and pass.

*Throughput measured, not guessed.* See §5.

**2026-09-01, fourth pass — encoder profiling.** The user clarified that the
earlier question about "the LQ encoder aligned with the reference paper" meant a
planned **V-Max-aligned successor** (`lq_v4`), not `lq_v3` against its diagnostic
reductions. Repository verdict: that successor **does not exist** — no `lq_v4`,
no `latent_query_v4`, no `ENC-V1.5`, no ReZero, no shared recurrent core. The
current encoder still carries `token_to_latent` (`lq_encoder.py:152`, `:334`) and
four **independent** blocks (`nn.ModuleList([_LatentQueryBlock(...) for _ in
range(depth)])`), which are precisely the two structures such a redesign targets.
This does not affect `AB-LEARN`: the encoder is held identical across arms, so it
cannot confound the reward contrast, and `lq_v3` is the authoritative architecture
today.

`scripts/benchmark_semantic_encoders.py` added to answer whether the encoder
dominates step time. Exact trainable parameters: `lq_v3` **1,120,000**, `lq_v3_lite`
**588,544**, `lq_v3_micro` **322,816**. Attention plus the latent feed-forward are
**94.3 %** of `lq_v3`; the token projectors are 11,072, or 1 %. Latency, measured
round-robin under contention on both devices: `lq_v3` costs **1.6x** `lq_v3_lite`
per learner pass (GPU 26.4 vs 16.6 ms, CPU 345.6 vs 215.3 ms at batch 256), a ratio
stable across devices even though the absolute medians are not usable — the GPU was
at 98 % from another tenant and the host load average was 14.5.

Two findings recorded against earlier statements in this plan. First, the estimate
in §5 that the production encoder costs "roughly 15 %" derived from two
uncontrolled single runs; the microbenchmark makes the encoder look materially more
expensive than that, and neither figure settles the question. Second, a defect
found and fixed in the tool itself before reporting: the parameter grouping matched
`output_projection` against the token-projector prefix and silently folded the
output head into it, overstating token projectors by 33,536.

The decisive measurement is still missing and needs an idle device: two matched
short runs differing only in the encoder (`lq_v3` against `mlp` on the same
`semantic_v3` observation), which yields numerator and denominator under identical
conditions. It also discharges `TEST-AB-003`, so `M2` and the attribution are one
action.

**2026-09-02, `M2`.** The smoke passed at the **third** attempt and was worth every
minute: it uncovered two independent defects, the second invisible until the first
was cleared, on a code path no automated check had touched since 2026-08-01.

1. **`C5`** -- `KeyError` on three `semantic_v3_signal_*` configuration keys that
   `envs/factory.py` injects and `ThesisScenarioEnv.default_config()` never
   declared. It made **every** `obs=semantic_v3` run unstartable, `make run-train`
   included. Fixed and covered by `SMOKE-COV`.
2. **`C6`** -- the first asynchronous evaluation died in a vector worker because
   `_ego_local_projection` required the re-projected ego station to equal the
   mission tracker's committed station within `1e-6 m`. Measured divergence at
   reset: **0.0229 m** on a 23.17 m route, caused by `mission/runtime.py`
   constructing the tracker with a hard-coded `initial_s_m=0.0`, so the guard
   compared a zero *by definition* against a projection of the actual spawn pose.
   Resolved by the user as: enforce the check from the first tracked step onward,
   tolerance unchanged. A two-sided regression test pins both the exemption and
   the fact that the guard still binds mid-episode.

Method note worth keeping: the magnitude was the whole diagnosis. The original
error message reported no number, and both plausible readings -- floating-point
noise and a stale snapshot -- were excluded only once it was printed. Adding the
measured quantity to a fail-fast message costs nothing and converts an unactionable
error into a decision.

`M1` and `M2` are complete.

**2026-09-05.** Repository tidied by the user and merged to `main`; the branch
fields above were corrected because a session starting from the handoff would
otherwise have looked for this work on a stale branch. `DEC-AB-004` resolved for
the seed-0 pair, which is handed to a dedicated session together with
`ab_screening_session_handoff_2026-09-02.md`.

**2026-09-05, `M3` launched at seed 0.** Both arms started in parallel at 21:47 UTC in
their own named `tmux` sessions, `ab_a_seed0` and `ab_b_seed0`, teeing to
`/tmp/ab_a_seed0.log` and `/tmp/ab_b_seed0.log`. Run directories:
`SCREEN-LEARNABILITY-A-NATIVE-01/sac_sb3/seed_0/20260905_214425/` and
`SCREEN-LEARNABILITY-B-RULEBOOK-01/sac_sb3/seed_0/20260905_214430/`.

Pre-launch re-verification on `main`, because `TEST-AB-002` had last been run on the
pre-merge branch: both configurations recomposed with `--cfg job --resolve` and diffed.
Single-factor invariance **holds** -- the only differences are `reward.name/type/behavior`,
the dependent `lambda_env`/`lambda_rule` (1.0/0.0 against 0.0/1.0) and the derived run
identity. Budget confirmed as `seed: 0` (the override bit; `conf/config.yaml` defaults to
`42`), 350,000 steps, `eval_interval` 25,000, `eval_episodes` 50, `final_eval_episodes` 100,
20 vectorized environments, `curriculum.enabled: false`, `scalarization` equal to
`conf/scalarization/default.yaml`. The runtime banner corroborates it: arm A reports
`Scalarization function: off`, arm B `six_level_priority_weighted_rank`, both on encoder
`latent_query_v3`.

Resource state at launch: GPU **23.7 GB free** of 97.9 GB against three other tenants at
100 % utilization -- roughly double the ~12 GB that `DEC-AB-004` reasoned about, so the OOM
margin is wider than the decision assumed. Host load average 4.0 over 72 cores, 410 GB RAM
available. Each arm settled at ~1.8 GB of GPU memory and ~25 GB of RSS.

Finding, provenance, **not behavioral**: the runtime banner names the rulebook
`4.7-final-implementation-complete` on a run whose object is `RULEBOOK-V5.1`, and the run's
own artifacts contradict each other -- `artifacts/run_metadata.yaml` records
`implementation_family: v1` while the checkpoint reward-semantics sidecar records the version
string, and neither says `v2`, which is what actually runs. Traced to `conf/config.yaml`
declaring only `rulebook.version`, with `runtime/io/metadata.py:207` and
`contracts/reward_semantics.py:47` applying **different defaults** to the same missing key
while `runtime/wiring/builders.py:349-358` treats that version string as a family selector
for the v2 machinery. The instrument under test is unaffected and was verified independently:
`MACRO_RULE_ORDER` carries six levels and the scalarizer runs
`six_level_priority_weighted_rank` on `vector_schema_id: rulebook_v5_1_six_level_v1`.
Recorded as `open_items` `C8` and **not fixed mid-run**, because the same string is stamped
into the fail-closed checkpoint-compatibility identity and into cached rulebook-catalog
provenance; that is a provenance-contract decision for the user, not an implementation
detail. It reproduces on the `M2` smoke of 2026-09-02, so it predates the screening.

**2026-09-06, `M3`: arm B died at 100 000 steps and was relaunched on the same seed.**

Measured throughput, replacing the `INFERRED` 6.68 fps anchor of §5: chunk 1 ran at
**4.55 fps** in both arms, chunk 2 at **3.97/3.94** once the learner updated on every
step (chunk 1 carries 5 000 `learning_starts` steps that skip the update), and chunk 3
at 3.93/3.90. The projection for 350 000 steps is therefore **~25 h per run**, not the
17-20 h §5 estimated -- that anchor came from a run with the `lite` encoder, and
`lq_v3` costs 1.6x per learner pass. Step-time attribution on these runs, not on the
smoke: `learner_update` 179 ms/step at 81 % of elapsed, `worker_wrapped_env_step`
173-184 ms/step at 79-84 %, the two overlapping because the workers step while the
learner updates. Inside the worker step, `rulebook_evaluator` costs 70-75 ms against
bare `env_step` at 35 ms, so **the rulebook costs 2x MetaDrive**, not the 3x the smoke
suggested. `rulebook_scalarization` costs **0.04 ms/step**: the scalarizer is free, and
arm A pays the full rulebook cost too, by `DEC-AB-001`'s design. Corroboration that
the GPU is the binding resource: with arm B dead, arm A rose from 3.93 to **4.7 fps**.

**The crash was `open_items` `C9`**, a dimensional error in the rulebook geometry:
`_constrained_components_after_ear_exhaustion` budgeted the *aggregate* area of the
triangulation slivers it discards against the *per-piece* constant that licensed
discarding each one. It fired inside `vehicle_yield` conflict-zone occupancy
prediction after 7 hours, on a shortfall of **0.13 mm2 on a 261 m2 polygon** -- a
relative error of 5e-7. Fixed by budgeting the shortfall relatively, at one part in
100 000 of the polygon's area; overshoot stays fatal at any magnitude. Both messages
now carry the measured magnitude, which the original did not -- the `C6` lesson,
applied before hypothesising.

**The fix was applied to the live tree under the running arm A** (user decision), which
is defensible only because it is inert: the per-triangle filter is untouched, so every
input the pre-fix code could decompose returns a bit-identical decomposition. Verified
over **1107 generated polygons, 1107 identical, zero differences**. It converts crashes
into results and cannot alter a number the previous code was able to produce, so arm
A's completed evaluations stand and its future ones are unchanged except that it can no
longer die this way. Both arms therefore run identical code from 2026-09-06 08:39 CEST.

**Arm B relaunched on `seed=0`** as `REQ-AB-003` requires, into
`SCREEN-LEARNABILITY-B-RULEBOOK-01/sac_sb3/seed_0/20260906_063938/`. The crashed run's
artifacts are retained at `20260905_214430/` and its log at
`/tmp/ab_b_seed0_crashed_run1.log`; its three completed evaluations (25k, 50k, 75k) are
**not** carried into the analysis, because the relaunched run re-measures them under the
same seed and mixing the two would pair evaluations from different processes.

Results through 75 000 steps, before the crash, descriptive only. On
`validation_waymo_empirical` arm A's `route_completion` rose 0.1372 -> 0.2381 -> 0.2743
while arm B went 0.1598 -> 0.0923 -> 0.1268: A has the clearly larger slope, and B's
50k dip proved to be a dip rather than a trend, which is exactly why §10 pre-registers
slope across many evaluations rather than a two-point line. `H3` held at both
transitions: `mean_scalar_rule_reward` and `route_completion` moved **together** in both
directions (-7.14/-0.068, then +12.21/+0.035), a positive association, which is the
opposite of the inverted incentive. On `validation_pg` at 25 000 arm A was **stationary**
-- `route_completion` 1.37e-05, zero collisions, zero off-road, `comfort_rate` 1.0 across
all 150 episodes -- the standstill degeneracy the rulebook exists to prevent, appearing
in the control arm.

**2026-09-06, learner throughput (ADR-078).** The `step_timing.csv` attribution
recorded above (learner 179 ms/step, worker 173–184 ms/step, overlapping; wall
clock ~220 ms, ~3.9 fps, ~25 h per `medium` run and ~107 h projected per
`thesis` run) led the user to ask which change could accelerate *every* run
without a further test campaign. Decision, approved the same day and recorded
as ADR-078: SAC minibatch `256 -> 512` with `update_to_data_ratio 0.5`, so the
learner consumes the same 5 120 replay samples per update call in half the
optimizer steps; and the post-update learning-potential batch — diagnostic-only
under ACL v2.0 `REQ-013` — switched off. Projection, not measurement: ~5.2 fps,
`medium` ~19 h, `thesis` ~80 h. Anything below `update_to_data_ratio 0.5`
buys nothing more, because the worker's Rulebook cost (70–75 ms) then becomes
the binding component and has no tunable knob.

Consequence for this plan: the setting is global and must be shared by both
arms of a pair (`AC-AB-002`). It is **not** applied to a run in progress. If the
user relaunches arm A together with arm B, the screening also becomes the first
measurement of the new learner setting and the core `thesis` runs inherit a
configuration the screening exercised; if arm A is kept, arm B must be relaunched
on the pre-ADR-078 configuration (`planner.sac.batch_size=256
agent.planner.algorithm.update_to_data_ratio=1.0`) and the new setting starts
with the core runs. That choice is the user's and is open at the time of writing.
`DEV-AB-002` records the deviation from `RL-BASELINES` v1 for whichever runs
adopt it.

**2026-09-06, `M3` seed-0 readout: arm A is not a valid control (`open_items`
`C11`).** Both runs are dead -- A at 10:42:31 UTC in a training worker on the
`C9` decomposition defect (chunk 8, 7 of 14 evaluations complete), B at 09:47:44
UTC on a CUDA out-of-memory while the 50k evaluation worker loaded its checkpoint
on the shared GPU (1 of 14 evaluations complete). Reading A's seven evaluations
per panel before deciding anything about a relaunch: on `validation_pg` the
policy is stationary at five of six points, and on both panels stationary
episodes score about 0 while moving episodes score -9 to -40 under the native
reward. The mechanism is MetaDrive's per-step `on_lane_line_penalty` (-1)
surviving the `no_negative_reward` clamp and accumulating without bound because
ADR-058's physical-exit termination never ends a line- or boundary-contact
episode (-198.9 over 200 steps in the worst Waymo episode; -452.8 over 465 on
PG). A's Waymo `route_completion` is mostly the logged initial speed (5.1 m/s
mean; 0 on PG) being braked away. The critic loss growing 22 -> 506 between 150k
and 175k is the expected consequence of returns spanning 0 to -450.

Consequence for §7.3: the pre-registered reading "arm A also flat => learner
problem" is not available, because arm A is flat for a reward reason of its own.
Arm B's single point at 25k is consistent with the Rulebook not sharing the
optimum (PG: 59 of 150 episodes below 0.01 completion, and the scalar reward
ranks the moving episodes above the stationary ones, +3.6 against -4.3), but one
point decides nothing. Two instruments were added and are recorded under `C11`:
the runtime return-ordering test (`still < partial < full` via logged-expert
replay, both arm rewards, both validation panels) and the constant-action
baseline (`brake`, `random`) evaluated through the panels with the run CSV
schemas, so the relaunched pair is read against its floors. Making arm A a valid
control is a user decision that changes observable behaviour and is **open**;
no relaunch is made until it is taken (user instruction of 2026-09-06).

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-AB-001` | `EVAL-PROTOCOL` v1.0 `REQ-002`/`REQ-003`: `thesis` profile, 1,500,000 steps, three seeds `[0, 1, 2]` | `medium` profile, 350,000 steps, two seeds `[0, 1]` | Screening before committing six full-budget runs under a reward that has never been trained on | User, 2026-09-01 (`DEC-AB-002`) | The result is a descriptive diagnostic under `REQ-018` and is not a core result; it cannot by itself close the freeze |
| `DEV-AB-002` | `RL-BASELINES` v1 §3.4 / `REQ-RLB-007` / §9.4: SAC `batch_size=256`, `gradient_steps=auto` resolved to `n_envs` | SAC `batch_size=512`, `update_to_data_ratio=0.5` (so `auto` resolves to `0.5 * n_envs`), post-update learning-potential diagnostic off, for every screening or core run launched after 2026-09-06 | Measured learner cost of 179 ms/step on the live screening; same replay samples per update call in half the optimizer steps, the one throughput change defensible without a test campaign | User, 2026-09-06 (ADR-078) | `tests/test_learner_update_throughput_adr078.py`; `tests/test_hydra_preset_run_configs.py` expected SAC batch; applies only to runs launched after that date, and a pair must be homogeneous (`AC-AB-002`) |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `conf/presets/learnability/sac_a_native.yaml` | Added | Arm A |
| `conf/presets/learnability/sac_b_rulebook.yaml` | Added | Arm B |
| `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | Added | This pre-registration |
| `docs/project_index.md` | Planned modification | Record the plan in the implementation register |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `... --config-name presets/learnability/sac_a_native --cfg job --resolve` | `PASS` | 2026-09-01 | Exit 0; `reward.behavior=monitor_only`, `curriculum.enabled=false`, `run_profile.name=medium`, `scalarization.version=1.4`, `mode=six_level_priority_weighted_rank`, `priority_base=2.2`, `experiment_group=SCREEN-LEARNABILITY-A-NATIVE-01` |
| `... --config-name presets/learnability/sac_b_rulebook --cfg job --resolve` | `PASS` | 2026-09-01 | Exit 0; `reward.behavior=scalar_reward`, same budget/curriculum/scalarization, `experiment_group=SCREEN-LEARNABILITY-B-RULEBOOK-01` |
| Diff of the two resolved configurations | `PASS` | 2026-09-01 | Re-run after `DEC-AB-006` switched both arms to `lq_v3`; invariance preserved. Differences confined to `reward.name/type/behavior`, `lambda_env`/`lambda_rule`, `experiment.name`, `analysis.experiment_group` and the derived paths. Algorithm, encoder, decoder, observation, environment, budget, seed, panels and `scalarization` identical |
| `make smoke` on each preset (`TEST-AB-003`) | `NOT_RUN` | — | `M2`. **Load-bearing, not a formality**: `conf/presets/test/smoke_train.yaml` selects `obs=lidar_state` and `encoder=none`, so the `semantic_v3` + `lq_v3` path has not been smoke-tested since `RB51` took the observation to `D = 3011` (`factory.py:76`). Risk: a startup failure on the production encoder discovered only at launch. Follow-up: `run_profile=smoke env.vectorized.num_envs=2` with the preset selected |
| `make smoke` on each preset (`TEST-AB-003`) | `PASS` | 2026-09-02 | Both arms `exit 0` at the third attempt, after `C5` and `C6` were fixed. Artifacts verified, not just the exit code |
| Diff of the two resolved configurations, re-run on `main` | `PASS` | 2026-09-05 | Re-verified after the merge, immediately before launch. Differences confined to `reward.name/type/behavior`, `lambda_env`/`lambda_rule` and the derived run identity; budget, seed, encoder, panels and `scalarization` identical |
| The four screening runs (`TEST-AB-004`) | `FAILED`, stopped | 2026-09-06 | `M3`. The seed-0 pair launched 2026-09-05 21:47 UTC; **both runs died on 2026-09-06** (A at 10:42 UTC, `C9`, 175k reached; B at 09:47 UTC, CUDA OOM, 50k reached). Arm A's seven evaluations are retained as descriptive evidence only (`C11`). Relaunch awaits the user's decision on arm A's control validity and on ADR-078; the seed-1 pair awaits a separate decision |
| `make test` (`TEST-AB-005`) | `NOT_RUN` | — | The change is configuration-only and adds no code path; to be run before the plan is marked `VERIFIED` |
| `pytest -m integration tests/test_reward_return_ordering_runtime.py` | `PASS` | 2026-09-06 | Both arms, first scenario of each validation panel, live runtime. Native: Waymo 9.52 < 198.71 < 386.40, PG 0.01 < 43.99 < 97.95; scalarized Rulebook: Waymo 4.60 < 176.67 < 347.36, PG -2.72 < 37.29 < 86.43. 2 passed in 20 min 55 s |
| `scripts/evaluate_constant_action_baseline.py` (`brake`, `random`), preset `sac_b_rulebook`, `seed=0`, full validation panels | `PASS` | 2026-09-06 | `outputs/BASELINE-CONSTANT-ACTION-01/sac_sb3/seed_0/20260906_1128{34,36}`. Floors, native / scalar mean return: **brake** Waymo 3.33 / -1.06 (route 0.041, collision 0.080 -- hit while stopped), PG 0.006 / -4.90 (route 0.00002); **random** Waymo -36.37 / -37.00 (route 0.201, collision 0.135, off-road 0.035, worst episode -420.8), PG 1.18 / -3.16 (route 0.011). Read against arm A's 25k-175k evaluations: A's Waymo route completion (0.125-0.274) sits between the two floors and its PG completion equals the brake floor |
| Full suite `python -m pytest -q` on the rebased branch (`adr078-on-main` + this change) | `PASS` | 2026-09-06 | 1677 passed, 1 pre-existing warning, 28 min 05 s under a machine load of ~20 |

## 15. Final Reconciliation

Not reachable: the plan is `AWAITING_DECISIONS` and no screening run has been
executed.

**Known limitations, stated in advance.**

1. Two seeds and 350k steps cannot support an inferential claim, and the plan
   does not make one (`DEC-AB-003` of `EVAL-PROTOCOL` v1.0 forbids it even at
   three seeds and 1.5M steps).
2. The screening cannot validate the rulebook's *correctness*. That was
   established by a different instrument — Test A/Test B falsification against
   1100 logged Waymo records and the O1–O6 orderings on constructed fixtures —
   and no training run can add to or subtract from it.
3. A negative result on `H1` does not localize the cause between the reward,
   the learner, the observation and the budget. Arm A is the control that
   separates the first two; the remaining two require further work.
4. The calibration underlying the reward is Waymo-only, and `M8a` measured that
   five of six L3 sub-rules never apply on PG. Pooled reporting is therefore
   prohibited (`REQ-AB-007`), and PG results speak to a materially smaller
   rulebook.

**Deferred required work.** Arms C and D, which answer the second half of
`D4`'s question — whether the ACL actually helps — and which cannot start
before `F5`.
