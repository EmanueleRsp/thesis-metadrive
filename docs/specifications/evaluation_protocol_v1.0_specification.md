# Specification: Experimental Evaluation And Algorithm Comparison Protocol

## Metadata

- Feature: Experimental evaluation and algorithm comparison protocol
- Specification ID: `EVAL-PROTOCOL`
- Version: `1.0`
- Status: `APPROVED`
- Date: `2026-07-24`
- Supersedes: replaces the normative statistical/seed content of `docs/protocols/algorithm_comparison_protocol.md` (10-seed protocol, `1.96*s/sqrt(n)` 95% CI formula, and the reward-setting × curriculum ablation framing), consolidates `docs/protocols/csv_evaluation_objectives.md` as a subordinate implementation-level CSV schema referenced by this protocol, and amends `docs/protocols/live_eval_video_protocol.md` only where `REQ-014`'s post-hoc case-selection scheme changes it. All three remain retained historical/supporting material per `DEC-008` (`APPROVED`); none is deleted.
- Related specifications:
  - `docs/specifications/rl_baselines_v1_specification.md`, `RL-BASELINES` v1.0
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`, v1.1
  - `docs/specifications/automatic_curriculum_learning_v1_specification.md`, v1 amended by §28 / ADR-014
  - `docs/specifications/automatic_curriculum_learning_v1.1_specification.md`, `ACL-SN-EMA-001` v1.1, `Status: APPROVED`, `Authoritative: YES`; verified directly against the current repository `docs/project_index.md` on 2026-07-24. It supersedes the v1 specification only for the selected ScenarioNet scalar ACL core; v1 remains authoritative for any part it does not explicitly supersede. See `DEC-011`.
  - `docs/specifications/rulebook_v4.7_specification.md`, v4.7
  - `docs/specifications/rulebook_scalarization_v1.0_specification.md`, `SCAL-V1.0` v1.0
  - `docs/specifications/observation_v1.2_specification.md`, `OBS-V1.2`
  - `docs/specifications/encoder_v1.1_specification.md`, `ENC-V1.1`
  - `docs/specifications/transition_replay_v1_specification.md`, `TRANSITION-REPLAY` v1.0
  - future approved lexicographic and distributional algorithm specifications
- Related ADRs:
  - `docs/decisions/ADR-018-parallel-evaluation-and-test.md`
  - `docs/decisions/ADR-019-asynchronous-evaluation-queue.md`
  - `docs/decisions/ADR-020-evaluation-video-diagnostics.md`
  - `docs/decisions/ADR-021-source-bounded-reactive-traffic.md`
  - `docs/decisions/ADR-024-runtime-scenario-data-abort.md`; verified directly present and `APPROVED` in the current repository `docs/project_index.md` (row dated 2026-07-23). See `DEC-011`.
- Authoritative: `YES`

## Review Notice (2026-07-24, second pass)

This document underwent three review passes on 2026-07-24. The first integrated the user's initial decisions (seeds, checkpoint policy, uncertainty convention, data-abort tolerance, primary metric family, qualitative selection scheme) and corrected several repository-fact and traceability errors. The second pass responded to detailed user feedback that identified:

1. two scientifically problematic formulations (the "exactly 1,500,000 completed timesteps" wording, and treating R4 as an ordinary violable constraint rule);
2. several requirements disproportionate to the agreed scope (mandatory paired seed differences, an over-broad mandatory report/validation matrix, an over-strong `AC-005`);
3. one repository fact requiring reconciliation with a prior audit (data-abort coverage computation).

All corrections from the second pass are recorded inline (marked **Verified repository fact (2026-07-24)** where newly checked) and in the §15 decision table. That pass also discovered exactly one new material decision, `DEC-015`, concerning PPO's rollout-buffer boundary at the exact end of the training budget; every other previously open decision (`DEC-006`, `DEC-008`, `DEC-009`, `DEC-010`) was resolved in that pass per explicit user instruction.

The third pass recorded the user's explicit resolution of `DEC-015`: the approved atomic collection unit for PPO is one complete global rollout (2,048 transitions); training terminates at the first algorithm-valid atomic collection/update boundary reaching or exceeding the 1,500,000-transition target; PPO therefore completes 733 rollouts and terminates after 1,501,184 transitions (overshoot 1,184, ≈0.079% of the target). Partial-rollout training and discarding collected-but-unused transitions are prohibited. With `DEC-015` resolved, no material decision recorded in §15 remained `OPEN`.

A fourth pass, on explicit user instruction ("Procedi con l'approvazione formale e la canonicalizzazione"), promotes this document: `Status: APPROVED`, `Authoritative: YES`, canonical path `docs/specifications/evaluation_protocol_v1.0_specification.md`, registered in `project_index.md` as `AUTHORITATIVE`. Approval of this specification is a scientific-contract decision; it is not by itself proof that the current repository implementation conforms (§11.10). Several verified repository gaps remain open implementation work, most notably `DEC-015` itself (bounded PPO overshoot is approved but not yet implemented — see REQ-002), the panel-manifest artifact (§11.12), checkpoint hash/role persistence (§11.13), and data-abort-coverage persistence (§11.14/REQ-012). These are tracked as required repository changes, not open scientific decisions, and are the subject of the implementation ExecPlan referenced in `project_index.md`.

Content is marked inline as one of: **Verified repository fact**, **Approved decision**, **Project adaptation**, **Limitation**, or **Open**.

## 1. Purpose And Context

This specification defines how completed reinforcement-learning runs are evaluated, compared, aggregated, and reported for the thesis experiments on multi-objective autonomous driving.

The protocol separates the scientific evaluation contract from the implementation of individual algorithms and shared pipeline components. It therefore does not redefine PPO, TD3, SAC, the Rulebook, scalarization, ScenarioNet integration, automatic curriculum learning, the observation, the encoder, or transition replay. Those behaviors are governed by their own authoritative specifications.

The main scientific question is algorithmic: how different reinforcement-learning methods perform under a shared and frozen autonomous-driving pipeline and a fixed environment-interaction budget. **Approved decision**: the protocol compares algorithm-specific systems under the same environment-interaction budget (`experiment.total_timesteps`), not under an equalized number of gradient updates or equalized wall-clock time; wall-clock time and gradient/update counts remain recorded diagnostics only (REQ-002).

The protocol defines two comparison phases:

1. **Phase 1 — Scalar baseline block** (`BASELINE-SCALAR-01`): PPO, TD3, and SAC using the approved scalar pipeline. This phase is approved in full by this specification and does not depend on any future algorithm specification.
2. **Phase 2 — Algorithm-extension block** (`EXTENSION-ALGORITHM-01`): approved lexicographic, distributional, and combined lexicographic-distributional variants, evaluated base-matched against the scalar parent baseline once their dedicated specifications are approved. This specification defines only the common admission requirements for Phase 2 (`REQ-019`); the concrete parent algorithm is deferred (`DEC-006`, `APPROVED` as an explicit deferral that does not block Phase 1).

Within each comparison block, all non-algorithmic components shall remain frozen. **Approved decision**: the dataset, environment (including the approved reactive-traffic lifecycle of ADR-021), observation, encoder, Rulebook, applicable scalarization, ACL, and the environment-step budget are frozen for the duration of a comparison block according to their respective authoritative specifications; only the algorithm identity and the native mechanisms its own approved specification requires may differ.

Differences required by the approved algorithm contracts are permitted and must be declared. For example, transition-level replay, N-step returns, and prioritized replay may be applicable to off-policy TD3/SAC variants and inapplicable to on-policy PPO, and PPO's on-policy rollout buffer has no off-policy analogue. **This specification shall not describe such a comparison as isolating algorithm identity alone.** When native components required by an algorithm's own approved specification (replay, PER, N-step horizon, PPO rollout construction, or similar) differ between compared conditions, the comparison block declaration must state exactly which components differ and why, rather than presenting the conditions as identical except for the algorithm's core update rule.

This is a project-specific evaluation design. The choice to use a fixed environment-step budget, three training seeds, the `thesis` run profile, and a component-frozen comparison is an original experimental decision (**project adaptation**), not a result prescribed by the referenced algorithm literature.

## 2. Scope

### In Scope

- definition of valid experimental conditions and comparison blocks;
- frozen shared-component requirements;
- training seed count and pairing across algorithms;
- environment-step budget and evaluation cadence;
- validation/test split use and scenario-panel identity;
- deterministic policy evaluation;
- checkpoint and model-selection semantics;
- per-episode, per-seed, and cross-seed aggregation;
- metric classification into primary, secondary, and diagnostic outcomes, with R1--R3 constraint metrics kept distinct from the R4 progress metric;
- failed, incomplete, duplicate, aborted, and non-convergent run handling, including reproducible condition-attributable resource/numerical failures;
- periodic learning-curve reporting;
- qualitative video selection and traceability;
- reproducibility metadata and official-run validity;
- the minimum core outputs required for thesis tables, plots, and appendices;
- future inclusion of approved lexicographic and distributional methods without changing the shared evaluation contract.

### Out Of Scope

- defining or modifying PPO, TD3, SAC, lexicographic, or distributional learning algorithms;
- defining Rulebook formulas, applicability, tolerances, missing-data semantics, scalarization, observation features, encoder architecture, ACL usefulness, N-step returns, or PER;
- hyperparameter optimization procedures;
- dataset generation, filtering, arm assignment, or split construction;
- changing termination, truncation, reactive-traffic, or data-abort runtime semantics;
- using wall-clock time as a substitute for the primary sample-budget comparison;
- equalizing gradient updates or wall-clock time across compared conditions;
- claiming formal statistical significance from three training seeds;
- treating evaluation episodes as independent training replicates;
- human validation of Rulebook outcomes;
- paired episode-by-episode statistical significance testing;
- hierarchical bootstrap analysis;
- multiple-comparison correction;
- a mandatory exhaustive report/validation matrix beyond the core outputs defined in `REQ-016` and the validation categories defined in §13.

### Optional Or Deferred

- component ablation studies;
- Waymo-only/PG-only source-mix ablations;
- formal null-hypothesis significance testing;
- multiple-comparison correction;
- hierarchical bootstrap analysis;
- paired episode-by-episode statistical tests;
- full-test-split evaluation beyond the fixed `thesis` panel;
- stochastic-policy evaluation in addition to the approved deterministic deployment mode;
- wall-clock- or compute-matched secondary studies;
- gradient-update-matched secondary studies;
- pre-rendering of all episodes before qualitative case selection;
- human validation of Rulebook outcomes;
- paired seed-level descriptive differences (§7.5) for selected scientifically relevant comparisons — not a mandatory output of the core v1.0 protocol (`REQ-010`);
- secondary/diagnostic tables, exhaustive per-sub-rule plots, ACL exposure plots, a consolidated cross-run reproducibility manifest beyond per-run `REQ-015` metadata, and dual Markdown+LaTeX export (`REQ-016`).

Optional studies shall be reported separately from the core algorithm comparison and shall not silently change the primary protocol.

## 3. Terminology, Assumptions, And Preconditions

### 3.1 Terminology

- **Condition**: one fully resolved algorithm and configuration combination included in a comparison.
- **Comparison block**: a predefined set of conditions intended to answer one scientific question while holding the remaining components fixed.
- **Run**: one training execution for one condition and one training seed.
- **Training seed**: the root seed controlling learner initialization, exploration, environment workers, ACL state, replay state, and other seeded stochastic components according to their authoritative specifications.
- **Evaluation seed**: an optional root seed controlling any evaluation-time stochastic element that is not itself determined by the frozen scenario panel or the deterministic policy. It is distinct from the training seed and, when used, identical across every condition and seed within a comparison block. **Approved decision** (`DEC-012`).
- **Scenario panel**: the ordered, frozen, deduplicated, hashed set of scenario UIDs used by one periodic-validation or final-test evaluation, identical across every algorithm, extension, and training seed within a comparison block.
- **Validation evaluation**: evaluation on the validation split during training. It may support diagnostics and an approved checkpoint-selection policy, but it shall not use the test split.
- **Final test evaluation**: evaluation on the frozen test panel after training and after any validation-only model-selection decision.
- **Valid episode**: an evaluation episode accepted by the evaluator under the approved runtime semantics and not classified as a runtime scenario data abort.
- **Data-abort episode**: an episode excluded according to ADR-024. A data abort is not a collision, failure, success, termination, or time-limit truncation, and by itself does not invalidate an official run (`REQ-012`, `DEC-004`).
- **Constraint macro-rule group**: one of the three worst-of-applicable-subcomponents macro groups produced by the authoritative Rulebook v4.7 aggregation — `collision_impact` (R1), `dynamic_interaction_safety` (R2), `road_traffic_compliance` (R3). **Verified repository fact**: `src/thesis_rl/rulebook/v2/aggregation.py` (`aggregate_rulebook_result`) computes `margins=(-costs[0], -costs[1], -costs[2], progress_margin)` with each `cost` clipped to `[0,1]`; each macro group is `<= 0`, `0` when satisfied, `< 0` when violated. Each macro group and sub-rule component also carries a canonical `applicable` boolean and a `NOT_APPLICABLE`/`VIOLATED`/`SATISFIED` status (`RuleComponentResult.applicable`, `src/thesis_rl/rulebook/v2/types.py:375`; `aggregate_max_component`, `src/thesis_rl/rulebook/v2/aggregation.py`).
- **R4 progress margin**: the route-progress dimension `progress_margin ∈ [-1,1]` produced by the same aggregation function but computed directly, not through `aggregate_max_component`. **Verified repository fact**: it carries no `applicable`/`NOT_APPLICABLE` status in the current implementation. Positive values indicate progress and negative values indicate regression. R4 is a task-completion objective, not a constraint-like safety rule, and this protocol never reports it as a "violation rate" (`REQ-007`, `DEC-013`).
- **Macro-rule group** (umbrella term): any of the four R1--R4 dimensions above, used only where a statement applies to both the constraint groups and the R4 progress margin.
- **Seed-level statistic**: a metric computed from the valid evaluation episodes of one run.
- **Cross-seed statistic**: a statistic computed from the seed-level values of the same condition.
- **Primary sample budget**: the target number of environment transitions for training, represented by `experiment.total_timesteps`; the actual completed count may differ at the algorithm-valid collection boundary that concludes training (`REQ-002`, `DEC-015`).
- **Official result**: the value eligible for the thesis main tables under all validity requirements of this specification.
- **Non-convergent run**: a technically completed run whose measured learning performance remains poor, unstable, or non-convergent. Non-convergence is a valid experimental outcome, not a run-invalidity reason, and shall be manually distinguished from an infrastructure/software failure and from a reproducible condition-attributable resource/numerical failure (`REQ-011`).

### 3.2 Preconditions And Assumptions

1. The ScenarioNet dataset, train/validation/test split manifests, and scenario catalogs are frozen by the authoritative ScenarioNet specification.
2. Train, validation, and test scenario identities are disjoint.
3. Validation and test use `FixedSequenceScenarioProvider` semantics: fixed scenario identities, deterministic order, exactly-once traversal of the requested panel, and no fallback to another split. **Verified repository fact**: `FixedSequenceScenarioProvider` (`src/thesis_rl/scenarios/provider.py`) walks the ordered `records` sequence supplied at construction and rejects duplicates or exhaustion without `repeat=True`; it does not itself persist or hash the resulting UID sequence as a separate frozen artifact (see `DEC-005` and §11).
4. The approved Rulebook, scalarization, observation, encoder, ACL, environment semantics, and replay contracts are available before official runs begin.
5. The scalar baseline block uses the same frozen shared stack for PPO, TD3, and SAC, except for algorithm-specific behavior explicitly authorized by their specifications.
6. The `thesis` run profile is the official duration profile for the core comparison. **Verified repository fact**: `conf/run_profile/thesis.yaml` currently sets `experiment.total_timesteps=1500000`, `experiment.eval_interval=25000`, `experiment.eval_episodes=100`, and `experiment.final_eval_episodes=300`, with `experiment.eval_deterministic=true` inherited from `conf/experiment/default.yaml`.
7. Evaluation is deterministic according to the approved deployment policy of each algorithm.
8. The test split is unavailable to training, curriculum decisions, checkpoint selection, hyperparameter selection, debugging decisions, and run acceptance before final test execution.
9. Three training seeds are used for each official condition: `[0, 1, 2]`, identical across all conditions. **Approved decision** (`DEC-001`).
10. The finite set of three seeds limits the strength of inferential claims. The protocol therefore prioritizes transparent descriptive reporting (raw seed-level values, mean, and sample standard deviation only) and does not compute a confidence interval, bootstrap estimate, or significance test. **Approved decision** (`DEC-003`).
11. If an evaluation seed is used, it is fixed before official runs begin and applied identically across every condition and training seed within a comparison block.
12. **Verified repository fact (2026-07-24)**: the current training loop stops at an exact environment-step count rather than overshooting a collection boundary; whether this exact-stop behavior is acceptable, given its interaction with PPO's rollout-buffer boundary, is `DEC-015` (`OPEN`).

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| Resolved run configuration | Full condition configuration | Structured configuration | Fixed before run start | Hydra-resolved authoritative configuration | Fatal for official run | `NO` as evaluator metadata |
| Training seed | Root experiment seed | Integer | Fixed per run; one of `[0,1,2]` | Official seed list | Fatal if absent or outside official list | Indirectly affects policy |
| Evaluation seed | Optional root seed for residual evaluation-time stochasticity | Integer | Run-constant; identical across conditions | Official evaluation-seed value, when used | No effect if not applicable to the evaluated component | `NO` |
| Run profile | Duration/evaluation profile | Enum | `thesis` for official runs | Resolved configuration | Run invalid if mismatched | `NO` |
| Dataset and split manifests | Frozen data identity | Paths, hashes, UID sets | Pre-run and evaluation-time | ScenarioNet artifacts | Fatal if missing or mismatched | `NO` |
| Validation panel | Ordered validation scenario UIDs | Sequence of UIDs | 100 intended episodes per validation evaluation under `thesis` | Frozen, hashed, deduplicated before official runs | Evaluation invalid if identity cannot be verified | `NO` |
| Test panel | Ordered test scenario UIDs | Sequence of UIDs | 300 intended episodes per final test under `thesis` | Frozen, hashed, deduplicated before official runs, identical for every algorithm, extension, and training seed | Official final result invalid if identity cannot be verified | `NO` |
| Checkpoint | Serialized policy and required companion state | Algorithm-specific | `checkpoints/final.zip` only for official results | Run artifacts plus compatibility checks | Fatal on absence or incompatibility | `YES`, through loaded policy only |
| Per-episode evaluator outputs | Outcomes, applicability, and Rulebook metrics | One row per attempted scenario | Episode lifetime | Evaluator | Missing row requires explicit disposition | `NO` |
| Per-rule and macro-rule metrics | Margins, applicability, violations | Rule/macro-rule-indexed values | Step and episode aggregations | Rulebook evaluator | Missing required macro-rule invalidates evaluation | `NO` |
| Data-abort records | Forensic exclusion metadata | One record per abort | Evaluation episode | ADR-024 runtime path | Must be preserved and reported with coverage | `NO` |
| Runtime and dependency metadata | Reproducibility metadata | Structured values | Run start/end | Runtime environment | Official-run validity depends on the metadata required by REQ-015 | `NO` |

### Prohibited Information And Practices

The following shall not affect training, checkpoint selection, hyperparameter selection, curriculum arm scores, run inclusion, or the definition of metrics after official experiments begin:

- test-split metrics or test episode outcomes;
- future scenario trajectories or privileged signals prohibited by the observation specification;
- evaluator-generated Rulebook outcomes exposed to the policy unless explicitly authorized by another specification;
- test scenario IDs used to tune behavior after inspecting results;
- post-hoc replacement of a failed seed with a different seed;
- selection among duplicate completed runs by choosing the better result;
- post-hoc removal of a technically valid but poorly performing or non-convergent run;
- post-hoc redefinition of primary metrics, checkpoint policy, smoothing, or comparison blocks based on observed results;
- use of evaluator metrics as ACL usefulness or replay-priority inputs;
- aggregation of evaluation episodes as if they were independent training seeds;
- silently modifying an official condition's frozen configuration (batch size, worker count, encoder dimensions, PER, N-step horizon, algorithm/reward/observation mode) to work around a reproducible resource or numerical failure, per `RL-BASELINES` REQ-RLB-022.

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| Official run record | Validity and identity record for one run | Structured document | One per condition/seed | Aggregator, thesis audit | Identifies canonical run, any replacement lineage, and its three-way manual disposition (infrastructure failure / non-convergent / condition-attributable reproducible failure) |
| Frozen panel manifest | Persisted, ordered, deduplicated, hashed validation/test scenario-UID list | One per split per protocol version | Ordered, split-tagged | Run launcher, evaluator, auditor | Immutable once frozen; regenerating it defines a new protocol version. **Limitation**: not yet backed by a persisted artifact in the current repository (§11) |
| Per-episode evaluation table | Scenario-level outcomes | One row per attempted episode | Ordered by frozen panel; includes UID and validity | Analysis and forensic review | Data aborts represented explicitly, not silently dropped from raw records |
| Per-rule and macro-rule evaluation table | Rule/macro-rule-level margins, applicability, and violations | Rule/macro-rule × evaluation rows | Canonical rule/macro-rule priority/order | Analysis | Missing required macro-rule is fatal; R4 kept structurally distinct from R1--R3; sub-rule detail preserved when already available |
| Seed-level summary | Aggregate for one condition/seed | Metric vector | Based only on valid episodes | Cross-seed aggregator | Includes episode count and coverage |
| Cross-seed summary | Aggregate across three seeds | Metric × condition | Canonical condition ordering | Main tables | Includes raw seed values and mean/standard-deviation fields only |
| Learning-curve table | Periodic validation metrics vs. environment steps | Evaluation point × seed × condition | Exact environment-step axis | Plots/sample-efficiency discussion | No primary smoothing or post-hoc interpolation |
| Reproducibility manifest | Complete experimental identity | Structured document | One per run (per `REQ-015`) | Audit/reproduction | Includes hashes and version metadata required by REQ-015. A separate consolidated cross-run manifest is optional (`REQ-016`) |
| Qualitative-case manifest | Traceable video selection | One row per selected case | Four canonical categories; comparisons keyed by shared `scenario_uid` | Thesis qualitative analysis | Produced only when cases are selected; unavailable categories explicitly recorded |
| Final comparison report | Tables, plots, coverage, limitations | Report directory | Predefined core structure (`REQ-016`) | Thesis | Distinguishes primary, secondary, and diagnostic outputs; core vs. optional outputs are explicitly separated |

Pairwise seed differences, when produced, are an optional output (§7.5, `REQ-010`), not a mandatory row of this table.

## 6. Functional Requirements

### REQ-001: Define Frozen Comparison Blocks

- Required observable behavior: every official comparison shall declare its conditions before results are inspected and shall identify exactly which component is allowed to differ.
- Applicability: all official algorithm comparisons.
- Invariants:
  - dataset, split manifests, scenario panels, Rulebook, observation, encoder, environment semantics (including the approved reactive-traffic lifecycle), ACL contract, evaluation cadence, metric definitions, and sample budget remain fixed within a block;
  - algorithm-specific mechanisms required by authoritative algorithm specifications are allowed and declared;
  - no condition may receive a different test panel or evaluation metric definition;
  - a comparison shall not be described as isolating algorithm identity alone when native mechanisms required by an algorithm's own approved specification (replay, PER, N-step horizon, PPO rollout construction, or similar) differ between compared conditions; such differences are permitted but must be explicitly declared in the block definition, not concealed or presented as identical configurations.
- Edge and missing-data cases: a condition without an approved algorithm contract cannot enter an official block.
- Failure or fallback behavior: undeclared configuration differences invalidate the block until reconciled.
- Interactions: RL-BASELINES, future lexicographic/distributional specifications, ScenarioNet, ACL, Rulebook, observation, encoder, transition replay.

The two comparison phases are:

- **Phase 1** — `BASELINE-SCALAR-01`: PPO vs. TD3 vs. SAC scalarized, under the approved scalar pipeline. Fully approved by this specification (`DEC-006`).
- **Phase 2** — `EXTENSION-ALGORITHM-01`: base-matched extensions once their algorithm specifications are approved: baseline (the Phase 1 scalar parent), lexicographic, distributional, and lexicographic + distributional. This specification approves only the common admission requirements (`REQ-019`); the exact parent algorithm and condition identities are explicitly deferred to a future amendment or protocol version after the relevant algorithm specifications are approved, and this deferral does not block Phase 1 approval (`DEC-006`, `APPROVED`).

This replaces the earlier reward-setting (native/monitor-only/scalar-reward) × curriculum-enabled/disabled factorial design historically encoded in preset files such as `conf/presets/td3/td3_native_curr.yaml`, `td3_native_no_curr.yaml`, `td3_monitor_only_curr.yaml`, `td3_monitor_only_no_curr.yaml`, `td3_scalar_reward_curr.yaml`, and `td3_scalar_reward_no_curr.yaml` (**verified repository fact**: this 3×2 preset matrix exists only under `conf/presets/td3/`, with no equivalent full matrix under `conf/presets/sac/` or `conf/presets/ppo/`). Those presets remain available only as separately labeled ablation/diagnostic material under `REQ-018`; they are not part of `BASELINE-SCALAR-01` or `EXTENSION-ALGORITHM-01`.

### REQ-002: Use The Thesis Sample Budget

- Required observable behavior: every official run shall use `run_profile=thesis` with a common target environment-interaction budget of `experiment.total_timesteps=1,500,000` environment transitions. Training terminates at the first algorithm-valid atomic collection/update boundary that reaches or exceeds the target; the target, the actual completed timestep count, the algorithm's atomic collection unit, and any overshoot are recorded.
- Applicability: core baseline and extension comparisons.
- Invariants:
  - environment timesteps are the primary equalized resource;
  - wall-clock time, gradient updates, batch reuse, and samples processed are recorded as diagnostics when available but are not equalized by this protocol;
  - runs shall not be extended post hoc because their current result appears poor;
  - partial-rollout training, partial-buffer optimization, and collection of transitions that are subsequently discarded without training on them are prohibited (`DEC-015`, `APPROVED`);
  - no additional collection unit may be granted on the basis of observed performance.
- Edge and missing-data cases: a run stopped before reaching its algorithm's atomic collection/update boundary at or beyond the target is incomplete and not an official result; a run that discards a partially collected on-policy rollout instead of completing it is likewise not an official result.
- Failure or fallback behavior: restart the same condition with the same seed according to REQ-011, unless the failure is a reproducible condition-attributable resource/numerical failure, in which case REQ-011's third disposition applies.
- Interactions: RL-BASELINES run-profile contract, `DEC-015`.

**Approved decision** (`DEC-015`, resolved 2026-07-24): for PPO, the approved atomic collection unit is the complete global rollout of 2,048 transitions (`n_steps * n_envs`). Under the current configuration, PPO therefore completes exactly 733 rollouts and terminates after `733 * 2,048 = 1,501,184` transitions — an overshoot of 1,184 transitions (≈0.079% of the 1,500,000-transition target) — with the final rollout fully consumed by the standard PPO/GAE update, exactly as every prior rollout. For TD3/SAC, the atomic collection/update unit is one transition (`train_freq=1`, `gradient_steps=auto`), so they terminate at exactly 1,500,000 transitions with no overshoot. This preserves the authoritative PPO rollout/GAE/minibatch/update semantics, follows the underlying on-policy learner's lower-bound-budget semantics, and avoids introducing a project-specific partial-rollout mechanism.

**Verified repository fact (2026-07-24)**: the implementation as it exists today does not yet realize this decision — it exact-stops instead of overshooting. `Agent.train()` runs a hard `for step in range(1, chunk_timesteps + 1)` loop (`src/thesis_rl/agent/agent.py:511`), and the outer chunk loop in `train_loop.py` decrements `remaining` by exactly `chunk_timesteps` every chunk with `chunk_steps = min(eval_interval, remaining)` (`src/thesis_rl/runtime/loops/train_loop.py:1245-1325`), so a single-environment ACL run currently stops at exactly `total_timesteps=1,500,000` env steps regardless of PPO rollout-buffer fullness. The PPO rollout buffer only triggers a policy update when `self.model.rollout_buffer.full` (`src/thesis_rl/agent/planners/algorithms/ppo_sb3.py:461-469`, `maybe_update`); no code path forcing a flush/update of a partially filled buffer at run end was found in `ppo_sb3.py` or `agent.py`. Implementing this approved decision therefore requires changing the outer training-loop stop condition from "stop at exactly `total_timesteps` env steps" to "stop at the first PPO rollout-buffer-full boundary at or beyond `total_timesteps`," which is a repository change required after approval (not performed by this specification review) — see the closing summary of required repository changes.

### REQ-003: Use Three Shared Training Seeds

- Required observable behavior: each condition shall be trained with the same three training seeds `[0, 1, 2]`. **Approved decision** (`DEC-001`).
- Applicability: every official comparison block.
- Invariants:
  - seed identities are paired across conditions;
  - no seed substitution based on observed performance;
  - seed-level values are retained and shown;
  - if an evaluation seed is used, it is distinct from every training seed and identical across all conditions and seeds within a comparison block (`DEC-012`).
- Edge and missing-data cases: a comparison with fewer than three valid runs for any condition is incomplete.
- Failure or fallback behavior: the missing seed shall be rerun with the same seed; no aggregate shall be presented as the official three-seed result until complete.
- Interactions: run launcher, RNG/checkpoint specifications, ACL/replay persistence.

### REQ-004: Freeze Validation And Test Scenario Panels

- Required observable behavior: all conditions and seeds shall use identical ordered validation and test panels, independent of any training seed.
- Applicability: periodic validation and final test.
- Invariants:
  - validation uses only validation-split UIDs;
  - final test uses only test-split UIDs;
  - intended panel sizes under `thesis` are 100 validation episodes and 300 final-test episodes;
  - panel identities and order are frozen, deduplicated, and hashed before official training begins;
  - the frozen panel is identical across every algorithm, extension, and training seed within a comparison block;
  - the exact attempted UID sequence is stored in the artifacts.
- Edge and missing-data cases: duplicate, unknown, wrong-split, or unverified UIDs invalidate the evaluation.
- Failure or fallback behavior: no replacement scenario or fallback sampling is permitted.
- Interactions: ScenarioNet v1.1, `FixedSequenceScenarioProvider`, ADR-024.

### REQ-005: Preserve Deterministic, Read-Only Evaluation

- Required observable behavior: evaluation shall call the approved deterministic inference mode and shall not update the learner, optimizer, target networks, replay buffer, normalization statistics, ACL state, exploration state, or training RNG streams.
- Applicability: validation and final test.
- Invariants:
  - the evaluator receives an immutable policy snapshot;
  - parallel or asynchronous evaluation preserves the ordered scenario sequence and deterministic aggregation semantics;
  - policy inference semantics do not branch implicitly by algorithm beyond the approved policy implementation.
- Edge and missing-data cases: a mutable evaluator snapshot or failed compatibility check invalidates the evaluation.
- Failure or fallback behavior: evaluation fails closed.
- Interactions: RL-BASELINES REQ-RLB-017, ADR-018, ADR-019.

### REQ-006: Apply One Explicit Official Checkpoint Policy

- Required observable behavior: every official result, quantitative or qualitative, shall use the same declared checkpoint type.
- Applicability: final test, main tables, and qualitative case selection.
- Invariants:
  - test metrics never select the checkpoint;
  - `final.zip` and `best_*` results shall never be conflated;
  - the same policy applies to all conditions in the comparison block.
- Edge and missing-data cases: if the required checkpoint is missing, the run is invalid.
- Failure or fallback behavior: no silent fallback to another checkpoint.
- Interactions: checkpoint index, best-checkpoint logic, final evaluation loop, REQ-014.

**Approved decision** (`DEC-002`): the sole official checkpoint for the primary quantitative results and for qualitative case selection/rendering (`REQ-014`) is `checkpoints/final.zip`, produced after the common 1,500,000-step target budget. `best_*` checkpoints are diagnostic/sensitivity artifacts only and never feed the primary ranking, tables, or qualitative selection.

**Verified repository fact**: `final.zip` is produced unconditionally at the end of training (`src/thesis_rl/runtime/loops/train_loop.py`, guarded by `checkpoint.save_final`). Three independently-gated `best_*` criteria currently coexist with no declared precedence among them (`best_lexicographic.zip`, `best_lexicographic_rulebook.zip`, `best_thresholded_lexicographic_rulebook.zip`), plus `latest`/`periodic`/`eval_snapshot` artifacts. This specification deliberately does not resolve a canonical ranking among the `best_*` variants because none of them feed official results (see Intentional Limitations).

### REQ-007: Report Method-Neutral Primary Outcomes

- Required observable behavior: cross-algorithm conclusions shall be based on environment/task outcomes and Rulebook metrics, not solely on the training reward.
- Applicability: final test and primary comparison tables.
- Invariants:
  - primary metrics are reported separately and are not collapsed into one unapproved scalar score;
  - reward channels remain distinct and diagnostic-only;
  - lower-is-better and higher-is-better directions are declared;
  - the R4 progress margin is never reported using constraint-rule terminology ("violation rate", "satisfied/violated") because it is not a constraint-like Rulebook rule (see §7.3).
- Edge and missing-data cases: missing primary metrics invalidate the official result.
- Failure or fallback behavior: no substitution with reward return.
- Interactions: Rulebook v4.7, scalarization v1.0, evaluator CSV schema.

**Approved decision** (`DEC-013`, revised in this review pass to separate the R1--R3 constraint macro-rule groups from the R4 progress margin). Primary metrics:

1. `success_rate` — higher is better;
2. `route_completion` — higher is better;
3. `collision_rate` — lower is better;
4. `out_of_road_rate` — lower is better;
5. for each constraint macro-rule group R1--R3 (`collision_impact`, `dynamic_interaction_safety`, `road_traffic_compliance`): episode violation rate (lower is better) and episode-minimum margin (higher is better), as defined in §7.2;
6. for the R4 progress margin: episode mean progress margin (higher is better) and `negative_progress_rate` (lower is better), as defined in §7.3.

Secondary metrics:

- per-canonical-sub-rule violation rate and margin for R1--R3, preserved in artifacts whenever the authoritative Rulebook implementation already exposes them, without changing Rulebook margin, applicability, or aggregation semantics;
- `top_rule_violation_rate`;
- `counterexample_rate`;
- `avg_error_value` and `max_error_value`;
- `violated_rules_ratio` and `unique_violation_patterns`;
- episode length and termination/truncation outcome counts.

Diagnostic-only metrics:

- selected training reward return;
- native environment reward return;
- scalar Rulebook reward return;
- hybrid reward return, when applicable;
- cumulative route progress across the episode, when available;
- FPS, wall-clock duration, update counts, samples processed, replay statistics;
- ACL arm/source visitation and replay diagnostics;
- data-abort counts and coverage.

**Verified repository fact**: the authoritative Rulebook v4.7 aggregation (`src/thesis_rl/rulebook/v2/aggregation.py`) already computes exactly three worst-of-applicable-subcomponents macro groups (`collision_impact`, `dynamic_interaction_safety`, `road_traffic_compliance`) plus a structurally distinct route-progress margin, and retains every sub-rule's raw result (including which sub-rule drove each macro value) in the same structure. Reporting R1--R3 macro-rule aggregates and the R4 progress margin as primary, and sub-rule detail as secondary, therefore requires no Rulebook semantic change.

### REQ-008: Compute Seed-Level Metrics From Episodes

- Required observable behavior: for each run and metric, compute the seed-level statistic over valid final-test episodes only, and over applicable steps only for any per-step Rulebook indicator.
- Applicability: validation and final test.
- Invariants:
  - Boolean outcome rates use the arithmetic mean of episode indicators;
  - continuous episode metrics use the arithmetic mean, with sample standard deviation and relevant extrema retained;
  - per-rule and constraint-macro-rule (R1--R3) violation rates are computed only over steps the Rulebook marks applicable, first per episode, then averaged across episodes with at least one applicable step, unless the authoritative Rulebook output defines a different aggregation;
  - the R4 progress margin is aggregated over all steps of the episode, per §7.3;
  - intended, valid, and aborted episode counts are stored.
- Edge and missing-data cases: no division by zero; zero valid episodes invalidate the evaluation; zero applicable steps for a given rule in an episode excludes that episode from that rule's aggregate, and the exclusion count is reported.
- Failure or fallback behavior: fail closed or mark the run invalid, never emit a misleading zero.
- Interactions: evaluator aggregation and ADR-024.

### REQ-009: Aggregate Across Training Seeds Transparently

- Required observable behavior: the official condition summary shall use the three seed-level values as the independent replication unit.
- Applicability: final tables and learning-curve summaries.
- Invariants:
  - report all three raw seed values;
  - report the arithmetic mean across seeds;
  - report the sample standard deviation across seeds;
  - do not treat the 300 evaluation episodes per seed as 900 independent training replicates;
  - do not omit a low-performing completed seed;
  - a confidence interval, when produced, is an optional, off-by-default supplementary column, never a replacement for the mandatory raw values/mean/SD, and is visually/textually marked as a weak `n=3` estimate.
- Edge and missing-data cases: no official cross-seed aggregate with fewer than three valid seed-level values.
- Failure or fallback behavior: mark comparison incomplete.
- Interactions: statistical report generator.

**Approved decision** (`DEC-003`): cross-seed reporting uses exactly the raw seed-level values, the arithmetic mean, and the sample standard deviation as the mandatory, always-computed core. This closes `DEC-003` and removes the earlier candidate protocol's 10-seed 95% CI convention (`mean_a(x) ± 1.96 * sd_a(x) / sqrt(n)`, defined for `n=10` in `docs/protocols/algorithm_comparison_protocol.md` §8) and the previously recommended `n=3` Student-t interval as *mandatory* conventions. Amended 2026-07-25: a confidence interval MAY additionally be computed and reported when the analyst opts in (e.g. an `--include-ci`/`analysis.include_confidence_interval` flag), using the same `1.96 * sd_a(x) / sqrt(n)` normal-approximation formula for consistency with the historical convention; it is never computed or shown by default and never substitutes for the mandatory mean/SD reporting.

### REQ-010: Paired Seed Differences (Optional)

- Required observable behavior: this requirement is not mandatory for the core v1.0 protocol. When paired seed differences are produced for a selected, scientifically relevant comparison, they shall align conditions by the same training seed and report per-seed differences.
- Applicability: optional, at the analyst's discretion, for comparisons within one block.
- Invariants, when produced:
  - define the metric direction before computing differences;
  - retain the three paired differences;
  - report their mean and sample standard deviation;
  - no p-value, significance label, bootstrap estimate, or multiple-comparison claim is computed or reported.
- Edge and missing-data cases: if one member of a seed pair is missing, the pairwise comparison is incomplete.
- Failure or fallback behavior: no unpaired substitution.
- Interactions: comparison report (§7.5).

This requirement, its output table, and §7.5 were mandatory in an earlier draft of this specification and are downgraded to optional in this review pass: with three seeds, raw values, mean, and sample standard deviation (`REQ-009`) are sufficient core reporting, and a separate mandatory paired-difference pipeline is disproportionate to the agreed scope.

### REQ-011: Handle Failed, Duplicate, Non-Convergent, And Condition-Attributable Failure Runs Without Selection Bias

- Required observable behavior:
  - failed or interrupted runs are preserved as failed artifacts;
  - exactly one canonical completed run is registered per condition/seed;
  - a run's disposition is classified manually and explicitly into one of three disjoint categories, based on cause and reproducibility rather than on exception type alone, and the classification and its rationale are recorded in the run registry:
    - (a) **infrastructure/software failure** — an external or non-reproducible cause unrelated to the official algorithm/configuration itself (worker/node crash, external resource preemption, transient hardware fault, uncaught defect in shared infrastructure code, corrupted artifact); invalid, rerun with the same condition and seed;
    - (b) **non-convergent/unstable completed run** — a technically completed run (reaches the target budget per REQ-002) whose measured performance is poor, unstable, or non-convergent; a valid experimental outcome, remains included;
    - (c) **reproducible condition-attributable resource or numerical failure** — a deterministic out-of-memory, NaN, or divergence failure caused by the official frozen configuration of the condition itself, reproducible across reruns with the same seed; per `RL-BASELINES` REQ-RLB-022, this shall not be worked around by silently modifying batch size, worker count, encoder dimensions, PER, N-step horizon, or algorithm/reward/observation mode; it is recorded as an incomplete run and reported as an explicit limitation/negative finding for that condition, and resolving it requires an approved specification or ADR change, not an ad hoc rerun.
- Applicability: run registry and aggregation.
- Invariants:
  - a different seed shall not replace a failed seed;
  - the latest timestamp shall not automatically win among multiple completed runs;
  - a replacement run must declare the run it replaces and the reason;
  - rerunning solely to obtain a more favorable outcome is prohibited;
  - the three-way classification is never inferred automatically from the metric values or exception type alone; it requires an explicit determination of cause and reproducibility.
- Edge and missing-data cases: unresolved duplicate completed runs block official aggregation.
- Failure or fallback behavior: require explicit reconciliation in the run registry.
- Interactions: run metadata and analysis pipeline, `RL-BASELINES` REQ-RLB-022.

### REQ-012: Make Data-Abort Coverage Explicit

- Required observable behavior: data-abort episodes are excluded from performance aggregates according to ADR-024, are not backfilled, and are represented in raw and summary artifacts together with intended (planned), attempted, valid, and aborted episode counts and coverage (valid/attempted).
- Applicability: validation and final test.
- Invariants:
  - attempted, valid, and aborted counts are reported per condition/seed and per evaluation;
  - scenario UIDs and abort reasons are retained;
  - an abort is not classified as an agent failure or success;
  - cross-condition scenario coverage differences are visible;
  - the presence of data-abort episodes does not by itself invalidate an official run.
- Edge and missing-data cases: silent omission of coverage reporting invalidates the evaluation.
- Failure or fallback behavior: an official evaluation missing the required attempted/valid/aborted/coverage report is invalid until it is reconstructed or rerun.
- Interactions: ADR-024 and evaluator artifacts.

**Approved decision** (`DEC-004`): an official run may contain data-abort episodes. They remain excluded from aggregates without backfill, and every official evaluation record reports attempted, valid, and aborted episode counts plus coverage. A run is invalid only when it is classified as an infrastructure/software failure under `REQ-011`(a); the mere presence of data-abort episodes, or a non-convergent/unstable but technically completed run, does not by itself invalidate official status.

Reconsidered and reconfirmed 2026-07-25: a deterministic seed-driven replacement-draw ("backfill") on runtime abort was proposed and rejected. Which scenario aborts is a function of the interaction between the policy under evaluation and the simulator, not of the seed alone; backfilling would therefore make the *effective* evaluated scenario set policy-dependent even though the base draw is seed-fixed, silently reintroducing exactly the cross-condition panel-divergence risk `DEC-005`'s frozen panel is meant to eliminate. No-backfill-with-coverage-reporting remains the official policy.

**Repository fact reconciled in this review pass (2026-07-24)**: the earlier draft of this specification stated that a computed planned/valid/aborted/coverage summary "is not yet produced" by the evaluation loop. Direct verification shows this was imprecise. `Agent._evaluate_parallel` (`src/thesis_rl/agent/agent.py:2279-2284`) already computes exactly this summary as `metrics["data_abort_coverage"] = {"attempted": ..., "valid": ..., "invalid": ..., "invalid_episodes": [...]}`, where `attempted` is the requested episode count for that evaluation call, `invalid_episodes` carries `episode_idx`, `scenario_uid`, and `reason_code` per aborted episode, and this is covered by a passing regression test (`tests/test_parallel_evaluation_data_abort.py::test_parallel_evaluation_excludes_data_abort_episode_from_aggregates`). The genuine, narrower gap, confirmed by a repository-wide search that found zero references to `data_abort_coverage` in `src/thesis_rl/runtime/loops/train_loop.py` or `src/thesis_rl/runtime/io/csv_recorder.py`, is that this computed summary is **not currently persisted** into any per-run CSV or metadata artifact by the training loop — it is returned in-memory from `Agent.evaluate()` and, as far as verified, not written to disk. `REQ-012`'s requirement is therefore already satisfied at the computation level; only the persistence step remains to be implemented (§11 item 14, corrected).

### REQ-013: Report Learning Curves On The Environment-Step Axis

- Required observable behavior: periodic validation metrics shall be plotted against cumulative environment timesteps.
- Applicability: sample-efficiency and convergence discussion.
- Invariants:
  - `thesis` evaluation cadence is every 25,000 environment timesteps (**verified against `conf/run_profile/thesis.yaml`**);
  - each evaluation uses the fixed 100-scenario validation panel;
  - main curves show the cross-seed mean and raw seed-level values, for the primary metrics only (§7.1--§7.3);
  - the primary curve is unsmoothed;
  - missing evaluation points are not imputed or interpolated in official tables.
- Edge and missing-data cases: incomplete curves are visibly truncated or marked missing.
- Failure or fallback behavior: no smoothing-based replacement of the raw curve.
- Interactions: asynchronous evaluation snapshots and analysis pipeline.

### REQ-014: Select Qualitative Cases Post Hoc By Category And Shared Scenario UID

- Required observable behavior: qualitative video/case selection is performed after final-test data and the run registry are available, using the recorded per-episode data and manifest. A small, fixed, pre-declared subset of tracked `scenario_uid`s is rendered as GIFs unconditionally for every evaluation (validation and final test); every other episode is recorded only in a per-episode manifest (scenario type/arm and characteristics, algorithm behavior, outcome, and R1--R4 metrics), from which GIFs may be rendered later, out of band, by a separate script driven by the manifest.
- Applicability: video/GIF appendix and selected thesis figures.
- Invariants:
  - every selected case is classified into exactly one of four approved categories: `representative_success`, `representative_failure`, `severe_rule_violation`, `algorithm_disagreement`;
  - a direct qualitative comparison between two or more conditions uses the identical `scenario_uid` for every compared condition;
  - every case records condition, comparison-block ID, seed, checkpoint identity (`final.zip` only, per `REQ-006`), scenario UID, split, category, and availability;
  - rendering and overlays do not change policy actions or metrics;
  - beyond the tracked subset, no episode is pre-rendered before post-hoc selection; the manifest is always produced.
  - the tracked-subset `scenario_uid`s are fixed before training starts, drawn once from the frozen validation/test panels, and identical across periodic validation and final test so that visual progression over training is comparable episode-for-episode; default tracked-subset size is 5 scenario UIDs per split unless the analyst configures otherwise.
- Edge and missing-data cases: an unavailable category for a condition is recorded explicitly as unavailable, not silently omitted.
- Failure or fallback behavior: no substitution with an unlabeled or uncategorized case.
- Interactions: ADR-020 and candidate live-video artifacts.

**Approved decision** (`DEC-014`): qualitative selection may be entirely post hoc, using the four categories above and the recorded data/manifest, with no requirement to predeclare cases before inspecting results, and no requirement to pre-render all episodes beyond the small fixed tracked subset defined above (amended 2026-07-25 to add the tracked-subset GIF mechanism and full per-episode manifest; the post-hoc four-category selection scheme is otherwise unchanged).

### REQ-015: Record Reproducibility Metadata Sufficient To Identify The Executed System

- Required observable behavior: every official run shall record at least:
  - resolved configuration;
  - condition ID and comparison-block ID;
  - training seed and, when used, evaluation seed;
  - run profile;
  - target and actual completed timesteps, and any overshoot (`REQ-002`, `DEC-015`);
  - git branch and commit;
  - working-tree clean/dirty state;
  - hashes of frozen dataset, split, catalog, validation-panel, and test-panel artifacts;
  - Python, PyTorch, Stable-Baselines3/fork, MetaDrive, ScenarioNet, CUDA, and relevant dependency versions;
  - Rulebook, scalarization, observation, encoder, ACL, replay, baseline, and this evaluation protocol's own specification IDs/versions;
  - checkpoint identity (path, hash, role) and reward-semantics identity;
  - device and host information;
  - start/end timestamps and completion status.
- Applicability: every official run.
- Invariants: metadata are written before training and finalized atomically after completion.
- Edge and missing-data cases: absent required metadata invalidates official status until reconstructed from verifiable artifacts.
- Failure or fallback behavior: fail closed for official aggregation.
- Interactions: run metadata writer and checkpoint manifest.

**Approved decision** (`DEC-009`): official runs require a clean, tracked Git working tree. Files that are generated or Git-ignored do not constitute dirty state for this purpose.

### REQ-016: Produce A Minimal Core Comparison Report

- Required observable behavior: the analysis pipeline shall generate a deterministic report tree from the canonical run registry, limited to the core outputs below plus any optional outputs the analyst chooses to enable.
- Applicability: each completed comparison block.
- Required (core) outputs:
  - run completeness and validity table, including the three-way manual disposition (`REQ-011`);
  - primary-metric table with raw seed values, mean, and standard deviation for every primary metric defined in `REQ-007` (including the R1--R3 and R4 breakdown);
  - unsmoothed validation learning curves for the primary metrics (`REQ-013`);
  - data-abort and evaluation-coverage report (`REQ-012`);
  - machine-readable CSV/JSON aggregate artifacts;
  - qualitative-case manifest and rendered artifacts, produced only when qualitative cases are selected (`REQ-014`).
- Optional outputs, enabled at the analyst's discretion and never required for a comparison to be reported as complete:
  - secondary and diagnostic metric tables;
  - the paired seed-difference table (§7.5, `REQ-010`);
  - per-sub-rule margin/violation plots;
  - ACL exposure diagnostics plots;
  - a consolidated cross-run reproducibility manifest beyond the per-run metadata already required by `REQ-015`;
  - a second machine-readable/thesis-ready export format (for example LaTeX, when Markdown is already produced, or vice versa).
- Invariants: rerunning analysis on the same canonical inputs produces the same output values and ordering; a comparison block is not withheld from reporting merely because an optional output was not generated.
- Edge and missing-data cases: incomplete conditions remain in the completeness report and are excluded from official aggregate tables.
- Failure or fallback behavior: analysis fails on schema incompatibility rather than silently dropping fields.
- Interactions: `src/thesis_rl/analysis/*`, Makefile/documented entry point.

**Approved decision** (`DEC-010`): a single documented entry point shall exist to regenerate the analysis deterministically from canonical artifacts. Its exact form (Makefile target vs. a `python -m thesis_rl.analysis...` command) is a repository-dependent implementation detail for the ExecPlan, not an open scientific decision.

### REQ-017: Use The Test Split Only For Final Unbiased Evaluation

- Required observable behavior: the test panel shall be evaluated only after training and any validation-only checkpoint decision are complete.
- Applicability: official runs.
- Invariants:
  - test results do not trigger further training, tuning, checkpoint selection, or condition removal;
  - any method change after test inspection creates a new experimental version and invalidates reuse of the inspected test panel as an untouched final test for that version;
  - final-test artifacts are immutable.
- Edge and missing-data cases: accidental early test use is recorded as leakage and blocks official use of the affected result.
- Failure or fallback behavior: define a new held-out evaluation panel or acknowledge the limitation; do not conceal the leakage.
- Interactions: run workflow and ScenarioNet split contract.

### REQ-018: Keep Core Comparisons Separate From Ablations

- Required observable behavior: optional ablations shall have distinct comparison-block IDs, hypotheses, and reports.
- Applicability: any component-removal or alternative-stack study.
- Invariants:
  - ablations do not alter the core baseline or extension result tables;
  - an ablation intended for a scientific conclusion uses the same three-seed and 1,500,000-step protocol unless a separately approved reduced-budget diagnostic label is used;
  - reduced-budget or one-seed ablations are descriptive engineering diagnostics only.
- Edge and missing-data cases: no post-hoc promotion of a diagnostic ablation to a core result.
- Failure or fallback behavior: relabel or rerun under the official protocol.
- Interactions: comparison registry.

**Approved decision** (`DEC-007`): ablations, including Waymo-only/PG-only source-mix studies, are excluded from the core v1.0 comparison and are optional/deferred only.

### REQ-019: Admit Future Algorithms Only Through Approved Contracts

- Required observable behavior: lexicographic, distributional, and combined variants enter `EXTENSION-ALGORITHM-01` only after their specifications define training signals, deployment action selection, checkpoint compatibility, evaluation mode, and required diagnostics.
- Applicability: future algorithm extensions.
- Invariants:
  - the shared dataset, observation, encoder, Rulebook evaluation, ACL, panels, seed count, and sample budget remain frozen unless an explicitly approved new protocol version changes them;
  - training-return metrics specific to an algorithm are diagnostic and do not replace method-neutral outcomes;
  - the parent baseline used for extension comparisons is declared before final-test results are inspected.
- Edge and missing-data cases: exploratory implementations remain excluded from official comparison.
- Failure or fallback behavior: report as engineering prototype only.
- Interactions: future algorithm specifications and `project_index.md`.

## 7. Mathematical And Algorithmic Contract

### 7.1 Per-Episode Task Outcomes

For valid episode `e`, define:

- `S_e ∈ {0,1}`: success indicator;
- `C_e ∈ {0,1}`: collision indicator;
- `O_e ∈ {0,1}`: out-of-road indicator;
- `R_e ∈ [0,1]`: route completion;
- `T_e ∈ N+`: number of executed environment steps.

For `N_s` valid episodes in seed/run `s`:

`SuccessRate_s = (1 / N_s) * Σ_e S_e`

`CollisionRate_s = (1 / N_s) * Σ_e C_e`

`OutOfRoadRate_s = (1 / N_s) * Σ_e O_e`

`RouteCompletion_s = (1 / N_s) * Σ_e R_e`

`N_s` is the number of valid episodes, not the intended panel size. Intended, attempted, valid, and aborted counts shall be reported separately.

### 7.2 Constraint Macro-Rule And Sub-Rule Metrics (R1--R3)

**Primary reporting in this protocol uses the three constraint macro-rule dimensions R1--R3** (`REQ-007`, `DEC-013`). Individual canonical sub-rule margins and violations (for example `rss`, `ttc`, `clearance`, `offroad`, `wrong_way`, `solid_line`, `signal`, `stop`, `crosswalk`, `vehicle_yield`, and any other sub-rule feeding a macro group) are secondary/diagnostic and shall be preserved in per-episode artifacts whenever the authoritative Rulebook implementation already exposes them, without changing Rulebook margin, applicability, or aggregation semantics.

For macro-rule or sub-rule index `i ∈ {R1, R2, R3, sub-rules}`, step `t` of valid episode `e`, this specification does not redefine applicability, tolerance, or missing-data handling: it reads two values directly from the authoritative Rulebook v2 result, without recomputing them:

- `a_(i,e,t) ∈ {0,1}`: the canonical applicability indicator (`RuleComponentResult.applicable`, equivalently `status != NOT_APPLICABLE`);
- `v_(i,e,t) ∈ {0,1}`: the canonical violation indicator, defined only where `a_(i,e,t) = 1` (`status == VIOLATED`, equivalently `cost > 0`, equivalently the already-computed `m_(i,e,t) < 0` for these cost-based macro groups under the current aggregation).

Aggregation uses only applicable steps:

`EpisodeViolationRate_(i,e) = ( Σ_t a_(i,e,t) * v_(i,e,t) ) / ( Σ_t a_(i,e,t) )`, undefined for an episode with `Σ_t a_(i,e,t) = 0`.

`EpisodeMinimumMargin_(i,e) = min_{t : a_(i,e,t)=1} m_(i,e,t)`, likewise undefined when no step is applicable.

Seed-level values are computed over the `N'_s` valid episodes with at least one applicable step for `i`:

`RuleViolationRate_(i,s) = (1 / N'_s) * Σ_e EpisodeViolationRate_(i,e)`

`RuleMinimumMarginMean_(i,s) = (1 / N'_s) * Σ_e EpisodeMinimumMargin_(i,e)`

The number of episodes excluded from `i`'s aggregate because no step was applicable is reported alongside the seed-level value. The global minimum across episodes may be retained as a secondary worst-case diagnostic but shall not replace the mean episode-minimum margin in the main table unless explicitly approved.

The Rulebook specification remains authoritative for applicability, missing-data behavior, margin formulas, canonical rule/macro-rule identity, and tolerance semantics; this section only defines how already-produced per-step Rulebook outputs are aggregated across steps, episodes, and seeds.

### 7.3 R4 Progress-Margin Metrics

R4 is a task-completion objective, not a constraint-like safety rule, and is aggregated differently from R1--R3 (see the **R4 progress margin** definition in §3.1). **Verified repository fact**: the R4 `progress_margin` carries no `applicable`/`NOT_APPLICABLE` status in the current implementation, so this protocol aggregates it over every step of the episode rather than over an applicability-filtered subset.

For valid episode `e` with `T_e` steps and R4 step margin `m_(4,e,t) ∈ [-1,1]`:

`EpisodeMeanProgressMargin_e = (1 / T_e) * Σ_t m_(4,e,t)`

`NegativeProgressIndicator_(e,t) = 1[m_(4,e,t) < 0]`

`NegativeProgressRate_e = (1 / T_e) * Σ_t NegativeProgressIndicator_(e,t)`

Seed-level values are the arithmetic mean of the above over the `N_s` valid episodes of run `s`:

`MeanProgressMargin_s = (1 / N_s) * Σ_e EpisodeMeanProgressMargin_e`

`NegativeProgressRate_s = (1 / N_s) * Σ_e NegativeProgressRate_e`

`route_completion` (§7.1) remains a distinct task-outcome metric and is not derived from `progress_margin`. `NegativeProgressRate` and `MeanProgressMargin` are never labeled "R4 violation rate" or "R4 satisfaction rate".

### 7.4 Cross-Seed Descriptive Statistics

For condition `a`, metric `x`, and `n=3` seed-level values `x_(a,1), ..., x_(a,n)`:

`mean_a(x) = (1 / n) * Σ_s x_(a,s)`

`sd_a(x) = sqrt((1 / (n - 1)) * Σ_s (x_(a,s) - mean_a(x))^2)`

The official condition summary reports every raw seed value together with `mean_a(x)` and `sd_a(x)`. **No confidence interval, bootstrap estimate, or significance test is computed or reported in v1.0** (`REQ-009`, `DEC-003`). The official three-seed result is incomplete if `n != 3`.

### 7.5 Paired Seed Differences (Optional)

This subsection is optional and not part of the mandatory core protocol (`REQ-010`). When produced for a selected comparison, and for conditions `a` and `b` and a higher-is-better metric:

`Delta_s = x_(a,s) - x_(b,s)`

For a lower-is-better metric, the report shall either use:

`Delta_s = x_(b,s) - x_(a,s)`

so positive always favors `a`, or retain the raw subtraction and display the direction explicitly. One convention shall be fixed in the analysis schema.

Report `Delta_1`, `Delta_2`, `Delta_3`, their mean, and their sample standard deviation. No significance classification, p-value, or bootstrap estimate is computed or required. This is a descriptive summary only and is not a substitute for paired episode-by-episode statistical testing, which remains out of scope (§2).

### 7.6 Reward Channels

For each episode, available returns shall remain separately named:

- selected planner-training reward return;
- native environment reward return;
- scalarized Rulebook reward return;
- hybrid reward return, when applicable.

These channels are not assumed to share scale or semantics and shall not be used as the sole basis for cross-algorithm ranking. All reward channels in this section are diagnostic-only per `REQ-007` and never substitute for the primary or secondary outcome metrics.

### 7.7 Learning-Curve Aggregation

At evaluation environment-step index `k`, aggregate only seed-level validation statistics recorded at exactly that index. No forward filling, linear interpolation, or smoothing is allowed in the primary curve.

An optional smoothed curve may be included only as a secondary visualization when its method and fixed window are declared before generation. It shall be plotted alongside, not instead of, the raw curve.

### 7.8 Checkpoint Semantics

The selected checkpoint identity shall include:

- checkpoint type;
- training timestep (target and actual, per `DEC-015`);
- source run and seed;
- content hash;
- compatible observation, encoder, reward, Rulebook, algorithm, and dependency identity.

Under the approved primary policy (`REQ-006`, `DEC-002`), `final.zip` corresponds to the policy serialized at the end of training under the target 1,500,000-step budget and is the sole source of the primary final-test result and of qualitative case selection/rendering (`REQ-014`).

## 8. Applicability, State, And Timing

1. The condition matrix, seed list, panel manifests, checkpoint policy, primary metrics, and analysis version shall be frozen before official training starts.
2. Training uses the train split only.
3. Periodic validation occurs every 25,000 environment timesteps under `run_profile=thesis` and uses the fixed 100-scenario validation panel.
4. Periodic evaluation may be asynchronous only through immutable policy snapshots and the approved FIFO/fatal-error semantics.
5. Training ends at the first algorithm-valid atomic collection/update boundary reaching or exceeding the target budget of 1,500,000 environment timesteps, unless it fails. For PPO this boundary is the completion of a full 2,048-transition global rollout (terminating at 1,501,184 transitions under the current configuration); for TD3/SAC it is the next single transition (terminating at exactly 1,500,000). The target, the actual completed timestep count, the atomic collection unit, and any overshoot are recorded (`REQ-002`, `DEC-015`, `APPROVED`).
6. The official checkpoint is `final.zip`, resolved according to `REQ-006`.
7. Final test uses the frozen 300-scenario test panel and deterministic inference.
8. Test results are written once and are not fed back into training or model selection.
9. Episode state, history, evaluator metrics, and Rulebook monitors reset at episode boundaries according to their specifications.
10. Termination, time-limit truncation, and data abort remain distinct outcomes in raw artifacts.
11. Data-abort episodes are excluded and not backfilled at runtime. They do not by themselves invalidate official-run acceptance (`REQ-012`, `DEC-004`).
12. Analysis starts only from the canonical run registry and immutable raw artifacts.
13. Evaluation seed, if used, is fixed before official runs and applied identically across conditions (`REQ-003`).
14. A new algorithm, metric definition, panel, or checkpoint rule requires a new approved protocol version or applicable approved ADR before additional official results are combined with the original block.

## 9. Configuration

| Field | Type | Default/Proposed Value | Valid range | Meaning | Required | Frozen for experiments |
|---|---|---|---|---|---|---|
| `evaluation.protocol_id` | string | `EVAL-PROTOCOL` | Exact registered ID | Evaluation contract | `YES` | `YES` |
| `evaluation.protocol_version` | string | `1.0` after approval | Approved version | Evaluation contract version | `YES` | `YES` |
| `evaluation.comparison_block_id` | string | Block-specific | Registered ID | Scientific question/condition matrix | `YES` | `YES` |
| `run_profile` | enum | `thesis` | Approved profiles | Duration/evaluation profile | `YES` | `YES` |
| `experiment.total_timesteps` | integer | `1500000` | Target for core | Environment-step target budget | `YES` | `YES` |
| `experiment.eval_interval` | integer steps | `25000` | Exact for core | Periodic validation cadence | `YES` | `YES` |
| `experiment.eval_episodes` | integer | `100` | Exact for core | Validation panel size | `YES` | `YES` |
| `experiment.final_eval_episodes` | integer | `300` | Exact for core | Final test panel size | `YES` | `YES` |
| `experiment.eval_deterministic` | bool | `true` | `true` for core | Deployment evaluation mode | `YES` | `YES` |
| `evaluation.training_seeds` | list[int] | `[0,1,2]` | Exactly these three integers | Shared paired seed list | `YES` | `YES` |
| `evaluation.evaluation_seed` | int or null | `null` | Distinct from every training seed when set | Optional shared evaluation-time seed | `YES` if applicable, else `Not applicable` | `YES` |
| `evaluation.checkpoint_policy` | enum | `final` | `final` | Official model source | `YES` | `YES` |
| `evaluation.primary_smoothing` | enum | `none` | `none` | Primary learning-curve smoothing | `YES` | `YES` |
| `evaluation.require_clean_tree` | bool | `true` | bool | Official-source reconstructability | `YES` | `YES` |
| `evaluation.fixed_validation_panel_manifest` | path/hash | Repository artifact | Existing valid manifest | Validation identities/order | `YES` | `YES` |
| `evaluation.fixed_test_panel_manifest` | path/hash | Repository artifact | Existing valid manifest | Test identities/order | `YES` | `YES` |
| `evaluation.analysis_schema_version` | string | To be implemented | Registered version | Output schema identity | `YES` | `YES` |
| `evaluation.collection_boundary_policy` | enum | `overshoot_to_atomic_boundary` | `overshoot_to_atomic_boundary` | End-of-run atomic collection/update boundary disposition (PPO: complete the final 2,048-transition rollout, 1,184-transition bounded overshoot; TD3/SAC: exact stop at the next transition) | `YES` | `YES` |

Invalid, unresolved, or mismatched frozen fields shall fail closed for official aggregation. A CLI override that changes a frozen field creates a non-official run unless a new comparison block or protocol version explicitly permits it.

`evaluation.confidence_level` and `evaluation.interval_method`, present in an earlier draft, are removed: no confidence interval is computed in v1.0 (`REQ-009`, `DEC-003`). `evaluation.require_zero_data_abort`, present in an earlier draft, is removed: official-run validity no longer depends on a zero-abort condition (`REQ-012`, `DEC-004`).

## 10. Errors, Logging, And Diagnostics

### 10.1 Fatal Conditions For Official Evaluation

- wrong run profile or timestep budget;
- unapproved condition or algorithm specification;
- missing/incompatible checkpoint;
- dataset, split, panel, or catalog hash mismatch;
- validation/test split leakage;
- missing primary metric or required macro-rule (R1--R4);
- malformed or duplicate scenario UID in a panel;
- mutable learner state during evaluation;
- missing required run metadata;
- unresolved duplicate completed runs;
- missing one or more official seeds;
- test results used for tuning or model selection;
- missing attempted/valid/aborted/coverage report for an official evaluation containing at least one data abort;
- an infrastructure/software failure that is not classified and not rerun with the same condition and seed;
- a reproducible condition-attributable resource/numerical failure silently worked around by modifying the frozen configuration instead of being reported as a limitation;
- a PPO run terminated before completing its final global rollout, i.e. any collected-but-untrained partial-rollout transitions at run end (`REQ-002`, `DEC-015`).

### 10.2 Recoverable Runtime Events

Runtime scenario data aborts are recoverable for worker survival under ADR-024 but remain scientifically visible. They do not by themselves invalidate official-run status (`REQ-012`, `DEC-004`).

### 10.3 Required Logs

- condition and comparison-block identity;
- resolved frozen configuration;
- checkpoint identity (path, hash, role);
- panel UID sequence and split;
- attempted, valid, terminated, truncated, and aborted episode counts;
- termination/truncation/data-abort reason per episode;
- policy deterministic flag;
- per-episode primary and secondary metrics, with R4 kept structurally distinct from R1--R3;
- per-rule and macro-rule applicability, margins, and violations;
- data-abort forensic record and coverage;
- evaluation worker failures and fatal propagation;
- timing, FPS, wall-clock, and resource diagnostics when available;
- ACL arm/source exposure diagnostics;
- manual three-way disposition (infrastructure failure / non-convergent / condition-attributable reproducible failure) and rationale for every invalid or non-convergent run;
- target vs. actual completed timesteps and overshoot (`DEC-015`);
- analysis inclusion/exclusion reason for every discovered run.

### 10.4 Classification Of Values

- **Policy observation**: governed only by OBS-V1.2 and related algorithm specifications.
- **Training signal**: reward, advantages, critic targets, replay priorities, ACL usefulness; not evaluator metrics unless separately authorized.
- **Evaluation metric**: task outcomes and Rulebook macro-rule/sub-rule margins/violations, coverage.
- **Diagnostic-only**: reward decomposition, FPS, wall-clock, update counts, replay/ACL diagnostics, forensic video overlays.
- **Reproducibility metadata**: versions, hashes, source state, configuration, seed, checkpoint identity.

## 11. Reproducibility And Compatibility

1. The run seed shall deterministically derive or record every owned RNG stream according to the authoritative component specifications.
2. Parallel environment and evaluation worker seed partitioning shall be deterministic and collision-free.
3. The exact dataset, catalog, split, validation-panel, and test-panel hashes shall be recorded and verified at evaluation time.
4. The official run shall record source commit and clean/dirty state. **Approved decision** (`DEC-009`): only clean-tree runs are official; Git-ignored/generated files do not count as dirty.
5. Exact dependency versions shall be recorded, including the Stable-Baselines3 fork identity.
6. Checkpoints shall carry or be accompanied by compatibility metadata for algorithm, observation, encoder, reward semantics, Rulebook/scalarization, and dependency identity.
7. Raw per-episode and per-rule/macro-rule artifacts shall be sufficient to independently recompute every reported aggregate.
8. Analysis shall be deterministic and versioned.
9. Existing historical runs produced under candidate protocols shall not be silently reclassified as v1.0 official runs. They may be reanalyzed only if all mandatory inputs and invariants can be verified; otherwise they remain historical/preliminary.
10. Approval of this specification does not by itself prove that the current repository implementation conforms. Codex must create an ExecPlan, verify the current code paths, implement missing requirements, and reconcile tests before `VERIFIED` status is possible.
11. **Verified repository gap (2026-07-24)**: the per-run reproducibility record shall additionally capture fields confirmed absent from the current `run_metadata.yaml` writer (`src/thesis_rl/runtime/io/metadata.py`): working-tree dirty/clean state for the training run itself (currently a dirty flag exists only in the separate `src/thesis_rl/scenarios/bootstrap.py` dataset-provenance manifest, not per training run); the exact `torch` version string (currently only device/CUDA name is captured); the populated Stable-Baselines3 fork version and commit (the `sb3_version`/`sb3_commit` fields already exist in `src/thesis_rl/contracts/checkpoint_manifest.py` but that manifest path is never invoked from `train_loop.py`); MetaDrive and ScenarioNet package versions; and the observation (`OBS-V1.2`), encoder (`ENC-V1.1`), ACL (`ACL-SN-EMA-001` v1.1), transition-replay (`TRANSITION-REPLAY` v1.0), and this protocol's own (`EVAL-PROTOCOL`) specification identity (only Rulebook and scalarization identity are currently captured, at `metadata.py`).
12. **Verified repository gap (2026-07-24)**: a persisted, hashed, ordered validation and test scenario-UID panel manifest shall exist as a distinct artifact and be referenced by run and evaluation records. The current `FixedSequenceScenarioProvider` re-derives panel order deterministically from split/catalog/eligible-UID configuration at each run without persisting or hashing a separate frozen panel artifact.
13. **Verified repository gap (2026-07-24)**: official checkpoint consumption shall record the exact checkpoint path, a content hash, and its role (`final`, or the specific diagnostic `best_*`/`latest`/`periodic`/`eval_snapshot` identity). The production `agent.save()` path used by `train_loop.py` records neither a hash nor a role; the unused `CheckpointGeneration`/`publish_checkpoint_generation` path (`src/thesis_rl/sb3_extensions/checkpointing.py`) computes a hash but is not invoked by the training loop.
14. **Corrected in this review pass (2026-07-24)**: the planned/valid/aborted/coverage computation required by `REQ-012` **already exists** (`Agent._evaluate_parallel`, `src/thesis_rl/agent/agent.py:2279-2284`, tested by `tests/test_parallel_evaluation_data_abort.py`), contrary to the earlier draft's "not yet produced" claim. The verified gap is narrower: this computed summary is not currently persisted to any per-run CSV or metadata artifact by `train_loop.py` (zero references to `data_abort_coverage` found in `train_loop.py` or `csv_recorder.py`). Only the persistence step remains to be implemented.
15. **Resolved in this review pass (2026-07-24)**: `DEC-015` — PPO's atomic collection unit at the end-of-budget boundary is the complete 2,048-transition global rollout; the run overshoots the 1,500,000-transition target by 1,184 transitions (terminating at 1,501,184) rather than discarding a partial rollout. **Verified repository gap**: this is not yet implemented — the current `train_loop.py`/`agent.py` stop condition is an exact `total_timesteps` env-step count independent of PPO rollout-buffer fullness (see REQ-002). Implementing `DEC-015` requires changing this stop condition to an atomic-collection-unit-aware boundary check, recording the atomic unit and overshoot in `run_metadata.yaml`, and adding a regression test that a PPO run under `run_profile=thesis` completes at exactly 1,501,184 transitions with its final rollout trained.

## 12. Acceptance Criteria

### AC-001: Frozen Baseline Block

- Given: approved scalar PPO, TD3, and SAC conditions.
- When: their resolved official configurations are compared.
- Then: only declared algorithm-specific fields differ; all shared frozen fields and panel hashes are identical.
- Related requirements: `REQ-001`, `REQ-002`, `REQ-004`.

### AC-002: Thesis Budget Enforcement

- Given: a candidate official run.
- When: run validity is checked.
- Then: `run_profile=thesis`; the run reaches or exceeds the 1,500,000-step target at an algorithm-valid collection boundary and records the actual completed timestep count and overshoot; validation cadence is 25,000; validation panel size is 100; and final panel size is 300.
- Related requirements: `REQ-002`, `REQ-013`.

### AC-003: Three Paired Seeds

- Given: one complete comparison block.
- When: the run registry is validated.
- Then: every condition has exactly one canonical completed run for each of the same three approved seeds `[0,1,2]` and no substituted seed.
- Related requirements: `REQ-003`, `REQ-011`.

### AC-004: Split And Panel Isolation

- Given: frozen train, validation, and test manifests.
- When: validation and final evaluation are constructed.
- Then: every attempted UID belongs to the correct split, no train/validation/test overlap exists, no duplicate UID exists within a panel, order matches the panel manifest, and no fallback UID is used.
- Related requirements: `REQ-004`, `REQ-017`.

### AC-005: Deterministic Read-Only Evaluation

- Given: a frozen checkpoint and evaluator snapshot.
- When: a representative fixture or evaluation-smoke-scale episode subset (not the full 300-episode final-test panel) is evaluated twice under identical dependencies and seed state.
- Then: attempted UID order is identical, the deterministic-evaluation flag is active, the policy snapshot is immutable, learner/replay/ACL state is unchanged, and outcomes/aggregate values match within the numerical-equivalence tolerances defined by `RL-BASELINES` for deterministic inference. A full repeated 300-episode final-test run is not required as the acceptance mechanism.
- Related requirements: `REQ-005`.

### AC-006: Explicit Checkpoint Consumption

- Given: a run containing `final.zip` and one or more `best_*` checkpoints.
- When: official final evaluation or qualitative case rendering starts.
- Then: it loads exactly `final.zip`, records that identity (path, hash, role), and never silently substitutes another checkpoint.
- Related requirements: `REQ-006`, `REQ-014`.

### AC-007: Method-Neutral Metrics With R4 Kept Distinct

- Given: a final evaluation of any scalar, lexicographic, distributional, or combined condition.
- When: the primary result table is generated.
- Then: success rate, route completion, collision rate, out-of-road rate, the R1--R3 constraint macro-rule violation rates/margins, and the R4 mean progress margin/`negative_progress_rate` are present; reward return is not used as their substitute; and R4 is never labeled or reported as a "violation rate".
- Related requirements: `REQ-007`.

### AC-008: Seed-Level Replication Unit

- Given: three runs with 300 intended test episodes each.
- When: cross-seed statistics are computed.
- Then: the aggregator consumes exactly three seed-level values, exposes those values plus mean and standard deviation, and does not calculate an interval using 900 episodes as independent replicates or any confidence/bootstrap estimate.
- Related requirements: `REQ-008`, `REQ-009`.

### AC-009: Failed, Non-Convergent, And Condition-Attributable Failure Handling

- Given: (1) one seed that failed for an external/non-reproducible infrastructural cause, (2) one completed seed with poor/unstable/non-convergent performance, and (3) one seed that reproducibly fails with an out-of-memory, NaN, or divergence outcome tied to its condition's frozen official configuration.
- When: the comparison registry is prepared.
- Then: (1) is explicitly classified as an infrastructure/software failure and rerun with the same seed and declared lineage; (2) is explicitly classified as non-convergent/unstable and remains included; (3) is explicitly classified as a condition-attributable reproducible failure, is not rerun with a silently modified configuration, and is reported as an incomplete-run limitation for that condition.
- Related requirements: `REQ-011`.

### AC-010: Data-Abort Visibility

- Given: an evaluation containing a runtime scenario data abort.
- When: raw and aggregate artifacts are written.
- Then: the abort UID and reason are present, the episode is excluded from performance numerators and denominators, no replacement scenario is sampled, attempted/valid/aborted counts and coverage are persisted to the run's artifacts, and the run is not marked invalid solely because of the abort.
- Related requirements: `REQ-012`.

### AC-011: Unsmoothed Primary Learning Curves

- Given: periodic validation artifacts.
- When: the main learning-curve plot is produced.
- Then: points occur at their exact environment-step values, no missing point is imputed, and no smoothing replaces the raw primary curve.
- Related requirements: `REQ-013`.

### AC-012: Traceable Qualitative Cases

- Given: a set of post-hoc qualitative selections across conditions.
- When: the qualitative manifest is inspected.
- Then: every video has condition, comparison-block ID, seed, checkpoint (`final.zip`), split, UID, category (one of the four approved categories), and availability; every direct cross-condition comparison uses the identical `scenario_uid`.
- Related requirements: `REQ-014`.

### AC-013: Reproducibility Metadata Completeness

- Given: a candidate official run.
- When: its metadata is validated.
- Then: all fields required by `REQ-015` are present, hashes match the executed artifacts, and the clean-tree policy (`DEC-009`) is satisfied.
- Related requirements: `REQ-015`.

### AC-014: Deterministic Report Regeneration

- Given: an immutable canonical run registry and raw artifact set.
- When: the single documented analysis entry point (`DEC-010`) is executed twice.
- Then: the core machine-readable aggregate values, row ordering, and thesis tables required by `REQ-016` are identical.
- Related requirements: `REQ-016`.

### AC-015: Test Leakage Prevention

- Given: an official comparison workflow.
- When: repository logs, metadata, and code paths are audited.
- Then: test metrics are absent from training, ACL, replay, hyperparameter, checkpoint-selection, and run-selection inputs.
- Related requirements: `REQ-017`.

### AC-016: Ablation Separation

- Given: a reduced-budget or component-removal experiment.
- When: reports are generated.
- Then: it uses a distinct block ID and is excluded from core algorithm tables unless it satisfies the full approved core protocol.
- Related requirements: `REQ-018`.

### AC-017: Future Algorithm Admission

- Given: a new lexicographic or distributional implementation without an approved specification.
- When: official conditions are enumerated.
- Then: the implementation is rejected from the official extension block and labeled exploratory.
- Related requirements: `REQ-019`.

### AC-018: Evaluation Seed Separation

- Given: a comparison block using an evaluation seed.
- When: conditions are compared.
- Then: the evaluation seed is identical across all conditions and distinct from every training seed.
- Related requirements: `REQ-003`.

## 13. Required Validation Categories

The evaluation protocol's own validation is scoped to what it directly owns: run/panel/checkpoint identity, aggregation correctness, and report regeneration. Full behavioral correctness of upstream components (observation, encoder, Rulebook, replay) is governed by their own authoritative specifications and is not re-validated here.

Required, owned by this protocol:

- nominal and boundary behavior of run/panel/checkpoint configuration;
- invalid and incoherent inputs to the evaluation/analysis pipeline;
- termination/truncation/data-abort handling within aggregates (exclusion without backfill, coverage reporting);
- identity of seed, panel, and checkpoint (`REQ-003`, `REQ-004`, `REQ-006`);
- numerical stability (NaN, division-by-zero) within the analysis aggregation itself;
- artifact schema compatibility for analysis regeneration;
- minimal upstream/downstream integration: evaluator output → CSV/artifact → analysis pipeline;
- regressions for: `final.zip`-only checkpoint consumption, shared/frozen panel identity, data-abort coverage persistence, and absence of test-split leakage.

Governed by the corresponding upstream specification, not re-validated by this protocol (`Not applicable` here):

- observation masks, padding, and anti-leakage correctness (`OBS-V1.2`);
- encoder internal state and architecture correctness (`ENC-V1.1`);
- full Rulebook applicability/margin/tolerance correctness (Rulebook v4.7);
- replay and optimizer update-order correctness (`TRANSITION-REPLAY`, `RL-BASELINES`).

Exact test files, fixtures, implementation paths, and commands belong in the ExecPlan.

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-001` | `AC-001` | User-defined frozen-component algorithm-comparison objective; authoritative shared-component specifications |
| `REQ-002` | `AC-002` | Approved decision: three-seed `thesis` runs with a 1.5M target budget; verified against `conf/run_profile/thesis.yaml`; PPO end-of-budget rollout-boundary disposition approved as bounded overshoot to 1,501,184 transitions (`DEC-015`), verified against `ppo_sb3.py`/`agent.py`/`train_loop.py` as not yet implemented |
| `REQ-003` | `AC-003`, `AC-018` | Approved decision: seeds `[0,1,2]` (`DEC-001`) and evaluation-seed separation (`DEC-012`) |
| `REQ-004` | `AC-004` | ScenarioNet v1.1 fixed sequence and split-isolation contract; approved frozen/hashed/deduplicated panel policy (`DEC-005`) |
| `REQ-005` | `AC-005` | RL-BASELINES deterministic evaluation; ADR-018; ADR-019 |
| `REQ-006` | `AC-006` | Approved decision: `final.zip` is the sole official checkpoint for quantitative and qualitative results (`DEC-002`) |
| `REQ-007` | `AC-007` | Approved decision: primary metric set is success/route-completion/collision/out-of-road plus R1--R3 macro-rule aggregates and the R4 progress margin, kept structurally distinct (`DEC-013`); verified against `src/thesis_rl/rulebook/v2/aggregation.py` |
| `REQ-008` | `AC-008` | Existing evaluator behavior, refined by this protocol; applicability-aware aggregation added in this review pass |
| `REQ-009` | `AC-008` | Approved decision: raw values, mean, and sample standard deviation only; no CI/bootstrap/significance (`DEC-003`) |
| `REQ-010` | none (optional) | Project design proposal, downgraded to optional in this review pass as disproportionate to the agreed scope |
| `REQ-011` | `AC-003`, `AC-009` | Approved decision: three-way manual disposition (infrastructure failure / non-convergent / condition-attributable reproducible failure), the latter added in this review pass per `RL-BASELINES` REQ-RLB-022 |
| `REQ-012` | `AC-010` | ADR-024 runtime semantics; approved tolerant data-abort validity policy with mandatory coverage reporting (`DEC-004`); computation verified already implemented in `agent.py`, persistence remains a gap |
| `REQ-013` | `AC-011` | `thesis` profile, verified against `conf/run_profile/thesis.yaml`; no-smoothing policy |
| `REQ-014` | `AC-012` | ADR-020; approved post-hoc category-based qualitative selection (`DEC-014`) |
| `REQ-015` | `AC-013` | Reproducibility gaps verified directly against the repository (§11 items 11--13); clean-tree policy approved (`DEC-009`) |
| `REQ-016` | `AC-014` | Reduced to a minimal core report set in this review pass as disproportionate scope was identified; single canonical entry point approved (`DEC-010`) |
| `REQ-017` | `AC-015` | ScenarioNet split contract; ACL evaluator-metric prohibition; standard held-out-test role adopted by project |
| `REQ-018` | `AC-016` | Approved decision: ablations, including Waymo-only/PG-only, excluded from core v1.0 (`DEC-007`) |
| `REQ-019` | `AC-017` | `project_index.md` currently marks lexicographic/distributional specs missing; Phase 2 admission deferred without blocking Phase 1 (`DEC-006`) |

## 15. Open Decisions And Limitations

| ID | Question | Alternatives | Recommendation/Decision | Impact | Status |
|---|---|---|---|---|---|
| `DEC-001` | Which exact three training seeds are official? | Any three predeclared unique integers; reuse historical defaults | `[0,1,2]`, identical across all conditions | Reproducibility and pairing; no algorithm semantics | `APPROVED` |
| `DEC-002` | Which checkpoint backs the primary final-test and qualitative results? | `final.zip`; one validation-selected `best_*`; both as co-primary | `final.zip` only, for both quantitative and qualitative official results; `best_*` diagnostic-only | Changes scientific meaning; requires implementation reconciliation | `APPROVED` |
| `DEC-003` | What uncertainty summary is official with three seeds? | Mean±SD only; normal CI; Student-t CI; bootstrap/hierarchical bootstrap | Raw seed values, mean, and sample standard deviation are the mandatory core; a normal-approximation CI may additionally be shown when the analyst opts in (off by default) | Tables and analysis implementation; strength of claims | `APPROVED` (amended 2026-07-25: optional opt-in CI) |
| `DEC-004` | Can an official final-test run contain data-abort episodes? | Accept with coverage; common-UID intersection; minimum coverage threshold; zero-abort requirement; deterministic seed-driven backfill/redraw | Accept with mandatory attempted/valid/aborted/coverage reporting; aborts excluded without backfill; do not by themselves invalidate the run | Evaluation validity and operational burden; policy-dependence risk of backfill | `APPROVED` (reconsidered 2026-07-25: backfill rejected, no-backfill reconfirmed) |
| `DEC-005` | What exact 100-validation and 300-test scenario panels are frozen? | Current provider prefix/order; precomputed balanced panel; full split | Test panel size 300, validation panel size 100 (verified against `conf/run_profile/thesis.yaml`); panels frozen, deduplicated, hashed, and identical across every algorithm/extension/training seed | Representativeness and reproducibility | `APPROVED` (policy); the persisted panel-manifest artifact itself is a verified repository gap (§11 item 12) |
| `DEC-006` | Which scalar parent algorithm anchors the lexicographic/distributional extension block, and does its absence block approval? | Decide now; defer without blocking Phase 1; defer and block the whole specification | `EVAL-PROTOCOL v1.0` approves `BASELINE-SCALAR-01` in full now. `EXTENSION-ALGORITHM-01` defines only the common admission requirements (`REQ-019`); the concrete parent algorithm is explicitly deferred to a future amendment or protocol version after the lexicographic/distributional specifications are approved, and this deferral does not block Phase 1 or the overall specification | Scientific attribution and experiment count for Phase 2 only | `APPROVED` (explicit deferral) |
| `DEC-007` | Are any component ablations included in the thesis core? | None; full-budget three-seed ablations; reduced diagnostic ablations | Exclude from core v1.0, including Waymo-only/PG-only source-mix studies; permit only separately labeled studies if time remains | Scope and compute budget | `APPROVED` |
| `DEC-008` | What happens to the three candidate protocol documents after approval? | Keep alongside; partially supersede; fully supersede selected content | `algorithm_comparison_protocol.md`: normative content (seed count, CI formula, ablation table) replaced by this protocol. `csv_evaluation_objectives.md`: retained as the subordinate implementation-level CSV schema this protocol's artifacts reference, not duplicated. `live_eval_video_protocol.md`: retained as implementation guidance, amended only where `REQ-014`'s post-hoc category taxonomy changes case-selection requirements | Document authority and traceability | `APPROVED` |
| `DEC-009` | Must official runs start from a clean working tree? | Require clean; allow dirty with patch capture; allow dirty with flag only | Require a clean, tracked Git working tree; Git-ignored/generated files do not count as dirty | Operational reproducibility | `APPROVED` |
| `DEC-010` | Must analysis/reporting have a canonical Makefile or documented command? | Manual scripts; documented Python command; Makefile target | A single documented entry point is required to regenerate analysis deterministically; the exact form (Makefile vs. Python module) is an ExecPlan implementation detail, not an open scientific decision | Usability and deterministic regeneration | `APPROVED` |
| `DEC-011` | Which uploaded/repository authority index is current? | Use the previously uploaded index; use the current repository index after direct verification | Resolved: the current repository `docs/project_index.md` (read directly on 2026-07-24) already lists ADR-024 as `APPROVED` and `automatic_curriculum_learning_v1.1_specification.md` (`ACL-SN-EMA-001`) as `APPROVED`/`Authoritative: YES` per its own metadata. One residual documentation gap remains: the index's summary table has not been updated to reference v1.1 alongside v1. This does not block this specification | Source authority and related-document metadata | `APPROVED` (resolved by direct verification) |
| `DEC-012` | Is an evaluation seed distinct from the training seed required? | No separate evaluation seed; a distinct evaluation seed shared across conditions | Optional distinct evaluation seed, identical across all conditions when used | Reproducibility of any residual evaluation-time stochastic element | `APPROVED` |
| `DEC-013` | What is the official primary metric set, and how is R4 treated? | Reward-centric; task-outcome-only; task-outcome plus per-canonical-rule detail; task-outcome plus macro-rule aggregates with R4 folded into the same violation-rate framing as R1--R3 | `success_rate`, `route_completion`, `collision_rate`, `out_of_road_rate`; R1--R3 constraint macro-rule episode violation rate and episode-minimum margin; R4 episode mean progress margin and `negative_progress_rate` reported separately and never as a "violation rate", because R4 is a task-completion objective, not a constraint-like safety rule (revised in the second review pass) | Table structure and primary ranking basis; scientific correctness of R4 framing | `APPROVED` |
| `DEC-014` | Must qualitative cases be predeclared before inspecting final-test results? | Mandatory predeclaration; fully post-hoc; hybrid fixed+forensic split; fixed tracked-subset GIFs plus post-hoc manifest-driven selection | Fully post-hoc selection from data/manifest using four fixed categories (`representative_success`, `representative_failure`, `severe_rule_violation`, `algorithm_disagreement`); direct comparisons keyed by shared `scenario_uid`; no mandatory pre-rendering beyond a small fixed tracked-subset (default 5 UIDs/split) rendered unconditionally at every evaluation for training-progression visibility | Qualitative appendix workflow and selection bias risk | `APPROVED` (amended 2026-07-25: tracked-subset GIF mechanism) |
| `DEC-015` | How should PPO's training-loop behavior interact with the rollout-buffer boundary at the end of the target budget? | (a) Accept exact-stop, documenting the trailing sub-2,048-transition partial rollout as a limitation that never triggers a policy update; (b) allow a bounded overshoot so the final rollout completes at `733 * 2,048 = 1,501,184` transitions and updates normally; (c) force a partial-buffer update (`compute_returns_and_advantage` over fewer than 2,048 transitions) at the exact target boundary | (b): the approved atomic collection unit for PPO is the complete 2,048-transition global rollout; training terminates at the first atomic boundary at or beyond the 1,500,000-transition target, i.e. after 733 rollouts / 1,501,184 transitions (overshoot 1,184, ≈0.079%). The final rollout is fully consumed by the standard PPO/GAE update; partial-rollout training and discarding collected transitions are prohibited. TD3/SAC are unaffected (atomic unit = 1 transition, exact stop at 1,500,000). Preserves authoritative PPO/GAE/minibatch/update semantics and the on-policy learner's lower-bound-budget semantics without a project-specific partial-rollout mechanism | PPO's full rollout is always trained; TD3/SAC unaffected; bounded, recorded overshoot of ≈0.079% of the target budget | `APPROVED` |

### Intentional Limitations

- Three seeds provide limited evidence about the distribution of training outcomes.
- Algorithm-specific update counts, replay reuse, and wall-clock compute are not equalized; environment interactions are the primary fairness axis.
- PPO cannot be made transition-replay-equivalent to TD3/SAC without changing its on-policy algorithmic contract.
- A fixed 300-scenario test panel samples only part of a larger test split when the split contains more than 300 eligible scenarios.
- Deterministic evaluation measures the approved deployment policy mode and does not characterize the full stochastic policy distribution.
- Rulebook metrics assess only the approved formalized rules and do not imply complete legal or safety coverage; no human validation of Rulebook outcomes is performed.
- Simulation, post-perception assumptions, source-bounded reactive traffic, and dataset composition limit external validity.
- Qualitative cases cannot establish population frequencies.
- No formal statistical significance, bootstrap, hierarchical bootstrap, or paired episode-by-episode significance testing is performed with three seeds; all cross-seed and (optional) pairwise reporting is descriptive.
- No single canonical `best_*` checkpoint definition exists in the current repository (three parallel, independently-gated criteria: `best_lexicographic`, `best_lexicographic_rulebook`, `best_thresholded_lexicographic_rulebook`, plus `latest`/`periodic`/`eval_snapshot`); this specification deliberately does not resolve that ambiguity because official results use only `final.zip`.
- The frozen-panel requirement (`REQ-004`, `DEC-005`) is a protocol requirement not yet backed by a persisted, hashed panel-manifest artifact in the current repository; `FixedSequenceScenarioProvider` currently re-derives panel order deterministically at each run instead (verified 2026-07-24, §11 item 12).
- Data-abort coverage is already computed at evaluation time (`Agent._evaluate_parallel`) but is not yet persisted to per-run artifacts by the training loop (verified 2026-07-24, §11 item 14 — corrected from an earlier, imprecise "not yet computed" claim).
- `DEC-015` is approved (bounded PPO overshoot to complete the final rollout) but is not yet implemented: the current `train_loop.py`/`agent.py` stop condition is an exact `total_timesteps` env-step count independent of PPO rollout-buffer fullness, so today's PPO runs still leave a trailing sub-2,048-transition partial rollout without a policy update (verified 2026-07-24; repository change required, see closing summary).

## 16. References

- `docs/project_index.md`: authority map, read directly on 2026-07-24 during this review.
- `docs/engineering_workflow.md`: specification lifecycle, approval, implementation handoff, and verification rules.
- `docs/templates/specification_template.md`: required structure and readiness criteria.
- `docs/specifications/rl_baselines_v1_specification.md`: deterministic evaluation, execution/duration profiles (including the `thesis` profile table and the PPO global-rollout-size contract), and REQ-RLB-022's prohibition on silently modifying configuration to work around resource failures.
- `docs/specifications/scenarionet_integration_v1.1_specification.md`: split isolation and fixed-sequence evaluation provider.
- `docs/specifications/automatic_curriculum_learning_v1_specification.md` and `docs/specifications/automatic_curriculum_learning_v1.1_specification.md` (`ACL-SN-EMA-001`): ACL contract; authority reconciled directly against the repository index on 2026-07-24 (`DEC-011`).
- `docs/specifications/rulebook_v4.7_specification.md`: canonical rules, macro-rule aggregation, margins, applicability, and evaluation semantics.
- `docs/specifications/rulebook_scalarization_v1.0_specification.md`: scalar reward semantics and identity.
- `docs/specifications/observation_v1.2_specification.md`: perception-bounded policy information contract.
- `docs/specifications/encoder_v1.1_specification.md`: encoder and checkpoint identity.
- `docs/specifications/transition_replay_v1_specification.md`: N-step/PER compatibility and boundary semantics.
- `docs/decisions/ADR-018-parallel-evaluation-and-test.md`: deterministic parallel evaluation.
- `docs/decisions/ADR-019-asynchronous-evaluation-queue.md`: immutable asynchronous validation snapshots and synchronous final test.
- `docs/decisions/ADR-020-evaluation-video-diagnostics.md`: non-interfering video diagnostics.
- `docs/decisions/ADR-021-source-bounded-reactive-traffic.md`: frozen reactive-traffic lifecycle relied on by the frozen-environment declaration.
- `docs/decisions/ADR-024-runtime-scenario-data-abort.md`: data-abort exclusion semantics; authority verified directly against the repository index on 2026-07-24.
- `docs/protocols/algorithm_comparison_protocol.md`: historical candidate comparison methodology; its 10-seed/95%-CI-formula content is superseded by this protocol (`DEC-008`).
- `docs/protocols/csv_evaluation_objectives.md`: historical candidate metric/artifact schema; retained as the implementation-level CSV schema (`DEC-008`).
- `docs/protocols/live_eval_video_protocol.md`: historical candidate qualitative-video protocol; its case-selection scheme is amended by `REQ-014` (`DEC-008`).
- Repository evaluation-implementation verification performed directly during this review (2026-07-24, both passes): `conf/run_profile/thesis.yaml`, `src/thesis_rl/runtime/loops/train_loop.py`, `src/thesis_rl/agent/agent.py` (including `Agent.train`, `Agent._evaluate_parallel`, `data_abort_coverage`), `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` (`maybe_update`, rollout-buffer-full trigger), `src/thesis_rl/scenarios/provider.py`, `src/thesis_rl/runtime/data_abort.py`, `src/thesis_rl/runtime/io/metadata.py`, `src/thesis_rl/rulebook/v2/aggregation.py`, `src/thesis_rl/rulebook/v2/types.py`, `src/thesis_rl/sb3_extensions/checkpointing.py`, `src/thesis_rl/contracts/checkpoint_manifest.py`, `tests/test_parallel_evaluation_data_abort.py`.
- External context referenced by an earlier draft as "repository audit supplied on 2026-07-23": no matching standalone repository artifact was located (the only located repository audit, `docs/audits/scalar_pipeline_audit_2026-07-19/audit_report.md`, covers dataset/TD3-integration scope). Treated as unverifiable external input per `incoming/README.md`.
- A second, more recent user-supplied audit (referenced in this round's feedback) correctly identified that `Agent._evaluate_parallel` already computes `data_abort_coverage`; this was verified directly against `src/thesis_rl/agent/agent.py:2279-2284` and is now the basis for §11 item 14 and `REQ-012`.
- User-provided historical LaTeX evaluation report: prior design basis; superseded portions are identified in this review draft.

## 17. Implementation Handoff Checklist

Before setting `Status: APPROVED`, confirm:

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, units, ranges, and validity.
- [x] Prohibited test, future, privileged, leaked, and diagnostic-only data is listed.
- [x] Core formulas and aggregation units are explicit, with R1--R3 and R4 kept structurally distinct.
- [x] State, timing, reset, termination, truncation, and data-abort distinctions are defined.
- [x] Configuration fields and experimentally frozen values are identified.
- [x] Errors, diagnostics, reproducibility, compatibility, and historical-run treatment are covered.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected and scoped to what this protocol owns.
- [x] Scientific sources, repository facts, project adaptations, and open recommendations are distinct.
- [x] Exact seed values are approved (`DEC-001`).
- [x] Official checkpoint policy is approved (`DEC-002`).
- [x] Statistical uncertainty convention is approved (`DEC-003`).
- [x] Official data-abort validity policy is approved (`DEC-004`).
- [ ] Exact validation and test panel manifests are verified and approved as a persisted artifact (policy approved per `DEC-005`; the manifest artifact itself remains a verified repository gap, §11 item 12 — an implementation gap, not a blocking scientific decision).
- [x] Extension parent algorithm policy is approved as an explicit deferral that does not block Phase 1 (`DEC-006`).
- [x] Ablation scope is approved (`DEC-007`).
- [x] Candidate-protocol supersession is approved (`DEC-008`).
- [x] Clean-tree policy is approved (`DEC-009`).
- [x] Canonical analysis command requirement is approved (`DEC-010`).
- [x] The actual repository authority index is reconciled with the previously uploaded copy (`DEC-011`).
- [x] R4's task-completion (non-constraint) treatment is approved (`DEC-013`).
- [x] PPO end-of-budget rollout-boundary disposition is approved (`DEC-015`: bounded overshoot to 1,501,184 transitions).
- [x] No material decision recorded in §15 remains open.

## 18. Approval Record

- Approved by: user
- Approval date: `2026-07-24`
- Approval evidence: explicit user decisions across four review passes in the Codex conversation on 2026-07-24, culminating in the explicit instruction "Procedi con l'approvazione formale e la canonicalizzazione."
- Approval notes: `APPROVED`. Across four review passes on 2026-07-24, all fifteen decisions recorded in §15 were resolved and approved: `DEC-001` through `DEC-014` in the first and second passes, and `DEC-015` (PPO's atomic-collection-unit boundary at the end of the target budget — approved as a bounded overshoot to 1,501,184 transitions so the final rollout is always fully trained) by explicit user instruction in the third pass. `DEC-011` was resolved by direct repository verification rather than a separate user decision. The fourth pass promotes the document itself on explicit user instruction. No material decision recorded in §15 remains open. Two scientifically problematic formulations from the original draft were corrected during review (the "exactly 1,500,000 completed timesteps" wording, and treating R4 as an ordinary violable constraint rule); several requirements disproportionate to the agreed scope were reduced to optional status (`REQ-010` paired seed differences, the mandatory report matrix in `REQ-016`, the "REQUIRED for everything" validation matrix in §13); `AC-005` was softened; and the data-abort-coverage repository fact was reconciled (already computed, not yet persisted). Approval of this specification is a scientific-contract decision only; it does not by itself prove that the current repository implementation conforms (§11.10). Verified repository gaps (§11.11--§11.14, and the not-yet-implemented `DEC-015` boundary logic in REQ-002) remain required implementation work, to be tracked by the implementation ExecPlan.
- Repository path: `docs/specifications/evaluation_protocol_v1.0_specification.md`
- Project index updated: `YES`
