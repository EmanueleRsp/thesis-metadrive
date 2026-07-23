# Specification: Scalarized PPO, TD3, and SAC Baselines

> This document was reviewed against the repository and explicitly approved by
> the user on 2026-07-20. The canonical copy is registered in
> `docs/project_index.md`.

## Metadata

- Feature: Scalarized reinforcement-learning baselines for autonomous driving
- Specification ID: `RL-BASELINES`
- Version: `1.0`
- Status: `APPROVED`
- Date: `2026-07-20`
- Supersedes: `NONE`
- Related specifications:
  - `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete`
  - `docs/specifications/rulebook_scalarization_v1.0_specification.md`, ID `SCAL-V1.0`, version `1.0`
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`, version `1.1-final-implementation-complete`
  - `docs/specifications/encoder_v1.0_specification.md`, ID `ENC-V1.0`, version `1.0-final-implementation-complete`
  - `docs/specifications/transition_replay_v1_specification.md`, ID `TRANSITION-REPLAY`, version `1.0`
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`
  - `docs/specifications/automatic_curriculum_learning_v1_specification.md`, version `v1`, including approved §28 amendment
- Related ADRs:
  - `docs/decisions/ADR-001-scenarionet-v1-1-dataset-policy.md`
  - `docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md`
  - `docs/decisions/ADR-003-causal-ctrv-conflict-zone-prediction.md`
  - `docs/decisions/ADR-004-assigned-route-metadata-for-pg-and-waymo.md`
  - `docs/decisions/ADR-011-rulebook-scalarization-v1.md`
  - `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`
- Authoritative: `YES`
- Canonical repository path: `docs/specifications/rl_baselines_v1_specification.md`

---

## Executive Summary

> Amendment RSA-1 (approved 2026-07-23; ADR-024): TD3/SAC preserve valid
> prefixes with a truncation boundary and PPO preserves valid rollout prefixes
> with final-observation value bootstrap. Typed data-aborts are excluded from
> evaluation policy metrics; generic exceptions remain fatal.

This specification defines the three scalar, non-lexicographic,
non-distributional reinforcement-learning baselines used by the thesis:

```text
ppo_sb3
td3_sb3
sac_sb3
```

They solve the same continuous-control autonomous-driving task and differ only
in their approved algorithm-native learning mechanisms. Their common scientific
pipeline is:

```text
frozen Waymo + PG ScenarioDescription dataset
→ ThesisScenarioEnv
→ Rulebook v4.7
→ bounded_satisfaction_rank scalarization
→ SemanticStateObservationV2
→ LatentQueryEncoderV2
→ PPO-SB3 / TD3-SB3 / SAC-SB3
```

The scalarized baselines are the parent controls for later lexicographic,
distributional, and lexicographic-plus-distributional algorithms. Those later
algorithms shall inherit their corresponding parent baseline configuration and
change only parameters required by the extension.

The core algorithm selections are:

```text
PPO:
    on-policy rollout buffer
    GAE
    no transition-level N-step replay
    no PER

TD3:
    3-step replay targets
    proportional PER enabled
    delayed deterministic actor update
    Gaussian behavior noise

SAC:
    3-step replay targets
    proportional PER enabled
    stochastic squashed-Gaussian actor
    automatically learned entropy coefficient
```

The primary final-pipeline execution profile uses ACL and one environment,
because the approved ACL runtime is currently single-environment. A separate
non-ACL vectorized profile uses four environments. Four workers were selected
so that PPO preserves a global rollout size of exactly 2,048 transitions.

Runtime execution profiles and duration run profiles are orthogonal. The
runtime profile selects ACL/vectorization; the duration profile selects the
global training budget and only the budget-sensitive warm-up/minibatch values
listed in §3.4. The default scientific runtime remains ACL ON with one
environment. ACL vectorization is deferred until its deterministic handoff
contract is implemented.

The runtime profiles are:

```text
ACL profile:
    n_envs = 1
    PPO n_steps per env = 2048

non-ACL vectorized profile:
    n_envs = 4
    PPO n_steps per env = 512

both PPO profiles:
    global rollout size = 2048
```

For TD3 and SAC, `gradient_steps=auto` resolves to `n_envs`, preserving one
gradient step per collected transition after warm-up. The duration profiles and
budget-sensitive values are defined in §3.4; they are not a second execution
dimension.

The selected TD3 configuration deliberately changes the current repository
batch size from 2,048 to 256. This is an approved project adaptation intended to
reduce memory cost, improve consistency with common continuous-control
practice, and support fair inheritance by future distributional TD3 variants.
The selected values are not claimed to be globally optimal.

Implementation cannot be declared verified until the resolved Hydra presets,
manifest-based checkpoint path, ACL learning-potential formulas, and required
learner smoke tests have been reconciled against this specification.

---

## 1. Purpose And Context

### 1.1 Purpose

The purpose of this specification is to define an implementation-complete,
observable contract for scalarized PPO, TD3, and SAC baselines in the thesis
pipeline.

It establishes:

1. the authoritative algorithm family;
2. the shared environment, action, observation, encoder, and reward interfaces;
3. algorithm-specific actor, critic, target, exploration, buffer, and update
   semantics;
4. exact scientifically frozen hyperparameters;
5. termination, truncation, auto-reset, and vectorized-environment behavior;
6. compatibility with N-step replay, PER, ACL, checkpointing, and evaluation;
7. anti-leakage requirements;
8. acceptance criteria needed before scientific training.

### 1.2 Research role

The three algorithms are scalar single-objective controls. They do not directly
optimize an ordered reward vector. They receive a scalar reward generated by
`bounded_satisfaction_rank` from the Rulebook v4.7 margin vector.

Their role is to provide parent baselines for:

```text
scalar PPO       → future Lex-PPO
scalar TD3       → future Lex-TD3 and Distributional-TD3
scalar SAC       → future Lex-SAC and Distributional-SAC
parent baseline  → future lexicographic + distributional algorithm
```

A future extension must not receive a more favorable environment, observation,
reward, encoder, training budget, or unreported hyperparameter configuration
than its parent scalar baseline.

### 1.3 Upstream and downstream components

Upstream components:

```text
frozen ScenarioDescription dataset
ThesisScenarioEnv
Rulebook v4.7 monitor
SCAL-V1.0 scalarizer
OBS-V1.1 observation builder
ENC-V1.0 LQ encoder bridge
optional ACL scenario selector
```

Downstream components:

```text
training loop
evaluation loop
checkpoint publisher/loader
experiment manifest
logging and metric exporters
future lexicographic algorithms
future distributional algorithms
```

### 1.4 Scientific sources, dependency behavior, and project adaptations

#### Literature-supported or upstream algorithm behavior

- PPO clipped policy optimization, rollout reuse, and GAE;
- TD3 clipped double Q-learning, delayed policy updates, target-policy
  smoothing, and deterministic behavior policy with external exploration noise;
- SAC twin critics, stochastic actor, entropy-regularized target, and automatic
  entropy-coefficient optimization;
- Polyak target-network updates for TD3 and SAC;
- off-policy replay for TD3 and SAC;
- N-step bootstrapping and proportional prioritized experience replay;
- separate treatment of true termination and time-limit truncation;
- Stable-Baselines3 semantics for rollout size `n_steps * n_envs` and
  off-policy `gradient_steps`.

#### Approved project adaptations

- only the local fork-backed `*_sb3` backends are scientific baselines;
- Rulebook v4.7 margins are converted with `bounded_satisfaction_rank`;
- `SemanticStateObservationV2` and `LatentQueryEncoderV2` are mandatory in the
  core baseline;
- TD3 and SAC use `n_steps=3` and proportional PER in the baseline profile;
- PPO rejects transition replay and uses GAE only;
- TD3 batch size is 256 rather than the current repository value 2,048;
- TD3 buffer capacity remains 300,000 as a resource-aware project choice;
- PPO preserves a global rollout size of 2,048 across supported execution
  profiles;
- the non-ACL vectorized profile uses four workers;
- the current ACL profile uses one worker;
- scientific checkpoints use a manifest-based generation path;
- replay persistence is optional and disabled by default;
- the 48-record golden suite is reference-only for deterministic Rulebook and
  adapter integration/regression checks; it is never a training, validation, or
  policy-evaluation dataset;
- hyperparameters are frozen after integration smoke validation, not selected by
  short-smoke reward performance.

### 1.5 Dependency constraint

The selected dependency is the local editable Stable-Baselines3 fork:

```text
version: 2.9.0
commit: 4e6c3db1367a4cda96308bc6d0b80e63cd698828
```

The repository index currently contains an older commit and must be corrected
after a final `git submodule status` verification.

The fork includes project-specific weighted replay-sample support for TD3 and
SAC critic losses. The complete deviation ledger is recorded in
`docs/implementation/sb3_fork_deviation_ledger.md`.

---

## 2. Scope

### 2.1 In Scope

- `ppo_sb3`, `td3_sb3`, and `sac_sb3` scientific backends;
- continuous two-dimensional ego action interface;
- `ThesisScenarioEnv` environment contract;
- Rulebook v4.7 scalarized reward;
- reward-vector and Rulebook diagnostics kept outside policy input;
- `SemanticStateObservationV2`, flat dimension 2,541;
- `LatentQueryEncoderV2`, output dimension 256;
- actor, critic, value, and target-network ownership;
- algorithm-native exploration and stochasticity;
- PPO rollout buffer, GAE, clipping, and optimization epochs;
- TD3/SAC 3-step targets and proportional PER;
- termination, truncation, final-observation, and auto-reset behavior;
- one-environment ACL execution;
- four-environment non-ACL execution;
- training and deterministic primary evaluation modes;
- checkpoint, model-only restart, stateful continuation, and manifest
  compatibility;
- fail-fast configuration validation;
- algorithm comparison and extension-inheritance rules;
- mandatory diagnostics and acceptance criteria.

### 2.2 Out Of Scope

- legacy backends `ppo`, `td3`, and `sac` as scientific baselines;
- direct lexicographic optimization;
- distributional critics;
- lexicographic-plus-distributional algorithms;
- ACL vectorization;
- raw sensor perception or sensor-to-control training;
- alternative observation families in the core comparison;
- MLP encoder ablations;
- behavior-cloning or imitation-learning initialization;
- recurrent policies;
- offline RL;
- automatic hyperparameter optimization;
- exhaustive hyperparameter tuning;
- changing Rulebook v4.7 or scalarization formulas;
- dataset construction, filtering, or split generation;
- final seed count, total training budget, and final evaluation episode count,
  which belong to an approved experimental protocol;
- actor-aware, rank-based, multi-criterion, or distributional-loss PER;
- cross-algorithm checkpoint transfer;
- bitwise-identical continuation claims.

### 2.3 Optional Or Deferred

- vectorized ACL with deterministic parent-worker episode handoff;
- replay-buffer persistence for selected long TD3/SAC runs;
- MLP or stacked-LiDAR ablations under a separate approved comparison protocol;
- stochastic evaluation diagnostics in addition to deterministic primary
  evaluation;
- separate learning rates for encoder and planner heads;
- alternative TD3 buffer capacities;
- alternative PPO global rollout sizes;
- future parent-matched controls required by distributional memory limits.
- the 48-record golden suite remains available through the dedicated diagnostic
  command, but is excluded from scientific baseline runs because it is not the
  canonical source-by-arm train/validation/test split.

No deferred capability may be enabled merely because an implementation hook
exists.

---

## 3. Terminology, Assumptions, And Preconditions

### 3.1 Symbols

| Symbol | Meaning |
|---|---|
| \(s_t\) | policy observation at control step \(t\) |
| \(a_t\) | normalized continuous action applied at control step \(t\) |
| \(r_t\) | scalar reward generated by `bounded_satisfaction_rank` |
| \(\mathbf m_t\) | ordered Rulebook v4.7 margin vector, diagnostic only |
| \(\gamma\) | reward discount factor |
| \(\lambda\) | GAE parameter for PPO |
| \(n\) | configured maximum N-step horizon for TD3/SAC |
| \(m_t\) | effective N-step horizon, \(1 \le m_t \le n\) |
| \(z_t\) | true-termination indicator |
| \(u_t\) | truncation indicator |
| \(b_t\) | bootstrap mask after the effective sequence |
| \(Q_1,Q_2\) | online twin action-value critics |
| \(Q'_1,Q'_2\) | target twin critics |
| \(\mu\) | TD3 deterministic actor |
| \(\pi\) | PPO or SAC stochastic policy |
| \(V\) | PPO state-value function |
| \(\alpha_{\mathrm{ent}}\) | SAC entropy coefficient |
| \(w_i\) | PER importance-sampling weight for sample \(i\) |
| `n_envs` | number of simultaneously active training environments |
| `vector_iteration` | one parent call that advances every active environment once |
| `global_env_steps` | total transitions collected across all workers |
| `global_rollout_size` | PPO transitions collected before one policy update |

### 3.2 Time and action convention

The environment control interval is:

\[
\Delta t = 0.02\,\mathrm{s}\times 5 = 0.1\,\mathrm{s}.
\]

One global environment transition corresponds to one ego control action applied
for one control interval in one environment.

The normalized action is:

\[
a_t = (a_t^{\mathrm{steer}}, a_t^{\mathrm{long}})
\in [-1,1]^2,
\]

where the first coordinate is normalized steering and the second is the
MetaDrive normalized longitudinal command, with positive values representing
throttle and negative values representing braking.

### 3.3 Global-step accounting

Scientific budgets, evaluation cadence, warm-up, PER beta annealing, and logs
shall use `global_env_steps`, not vector-iteration count.

For `n_envs=4`, one vector iteration adds four global environment steps.
For `n_envs=1`, one iteration adds one global environment step.

### 3.4 Execution profiles

The following are runtime execution profiles, not duration profiles:

The supported profiles are:

| Profile | ACL | `n_envs` | PPO `n_steps` per env | PPO global rollout | TD3/SAC resolved gradient steps |
|---|---:|---:|---:|---:|---:|
| `acl_single_env` | ON | 1 | 2,048 | 2,048 | 1 |
| `non_acl_vectorized` | OFF | 4 | 512 | 2,048 | 4 |
| `acl_control_single_env` | OFF | 1 | 2,048 | 2,048 | 1 |

`acl_control_single_env` is required when measuring the causal effect of ACL
before ACL vectorization exists.

The repository's duration profiles select the global budget. The primary
single-environment ACL runtime is the default for all duration profiles. The
following budget-sensitive values are the approved project matrix:

| Duration profile | Global steps | TD3 `learning_starts` / `batch_size` | SAC `learning_starts` / `batch_size` | PPO runtime setting |
|---|---:|---:|---:|---|
| `default` | 600,000 | 10,000 / 256 | 10,000 / 256 | global rollout 2,048 |
| `smoke` | 2,000 | 100 / 64 | 100 / 64 | diagnostic rollout 16 / batch 8 |
| `fast` | 120,000 | 5,000 / 256 | 1,000 / 256 | global rollout 2,048 |
| `medium` | 350,000 | 10,000 / 256 | 5,000 / 256 | global rollout 2,048 |
| `long` | 700,000 | 10,000 / 256 | 10,000 / 256 | global rollout 2,048 |
| `tune` | 500,000 | 10,000 / 256 | 5,000 / 256 | global rollout 2,048 |
| `thesis` | 1,500,000 | 10,000 / 256 | 10,000 / 256 | global rollout 2,048 |

The smoke PPO setting is diagnostic-only and is not used to claim conformance
to the core 2,048-transition PPO rollout contract. Gamma, tau, learning rate,
TD3 exploration/target noise, policy delay, N-step horizon, PER parameters, and
PPO optimization parameters remain algorithm-native and frozen across duration
profiles. `per.beta_anneal_steps` always resolves to the selected global budget.

### 3.5 Assumptions and validation

| Assumption | Classification | Validation | Failure behavior |
|---|---|---|---|
| Scenario source is the frozen 3,500-scenario dataset | configuration-provided and manifest-validated | dataset fingerprint and split manifest | fail before reset |
| Observation is `SemanticStateObservationV2` | authoritative upstream specification | exact space and schema fingerprint | fail learner construction |
| Flat observation dimension is 2,541 | authoritative upstream specification | runtime shape check | fail fast |
| LQ encoder produces 256 features | authoritative upstream specification | construction and forward test | fail fast |
| Action space is finite `Box([-1,-1],[1,1])` | runtime-validated | environment/action-space check | fail learner construction |
| Reward scalarization is `bounded_satisfaction_rank` | configuration and manifest | reward-semantics validation | fail before learner load |
| Native reward weight is zero | authoritative scalarization specification | config validation | fail configuration |
| Termination and truncation are provided separately | authoritative ScenarioNet contract | boundary normalizer test | fail transition handling |
| Final observation exists for bootstrappable truncation | runtime-validated | boundary check | fatal; no reset-observation fallback |
| ACL vectorization is unsupported in v1 | verified repository limitation | config validation | reject `ACL ON` with `n_envs>1` |
| GPU execution may be nondeterministic | explicit research limitation | documented in manifest | no bitwise-reproducibility claim |
| Short smoke reward is not evidence of hyperparameter superiority | approved scientific decision | review and reporting check | smoke cannot select configurations |

### 3.6 Dataset identity

The protected dataset contains:

```text
train:       1,000 Waymo + 1,000 PG
validation:    250 Waymo +   250 PG
test:          500 Waymo +   500 PG
total:       3,500 ScenarioDescription
```

Dataset fingerprint:

```text
919872129dccd1bfdbe3ac9b61c6a73fd33fb3eea6d92e31f9a42dc3b1405180
```

A scientific run with a different dataset or split identity is a different
experimental condition and cannot resume the same run.

The validated 48-record golden suite at
`docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_content_validated/`
is a reference fixture with integration/regression purpose only. It is not a
balanced source-by-arm sample and must not be split into policy train/test data
or used to support performance claims.

---

## 4. Inputs And Prohibited Information

### 4.1 Inputs

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `observation` | semantic policy observation | `(2541,)`, `float32`, mixed normalized semantic features | current committed step | `OBS-V1.1` | fail fast | `YES` |
| `action_space` | normalized continuous ego command space | `(2,)`, dimensionless | `[-1,1]^2` | `ThesisScenarioEnv` | fail construction | `YES`, as policy output domain |
| `scalar_reward` | learner reward | scalar finite float | current transition | `SCAL-V1.0`, `bounded_satisfaction_rank` | fail transition | `YES`, as training signal only |
| `rulebook_margins` | ordered Rulebook v4.7 margins | `(4,)`, Rulebook-defined bounded values | current transition | Rulebook monitor | fail diagnostics/reward pipeline | `NO` |
| `rulebook_diagnostics` | costs, statuses, submetrics, applicability | structured diagnostic payload | current transition/episode | Rulebook monitor | fail required logging when configured | `NO` |
| `terminated` | true terminal flag | scalar boolean | current transition | `ThesisScenarioEnv` | fail boundary normalization | `NO` |
| `truncated` | horizon/time-limit flag | scalar boolean | current transition | `ThesisScenarioEnv` | fail boundary normalization | `NO` |
| `final_observation` | pre-reset observation for completed vector worker | `(2541,)`, `float32` | terminal/truncated transition | vector boundary normalizer | fatal if required and absent | `NO`, except normal critic/value bootstrap |
| `run_seed` | root experiment seed | scalar integer | run-constant | approved experimental protocol | fail configuration | `NO` |
| `scenario_assignment` | selected dataset record and ACL arm metadata | structured metadata | reset-time | ScenarioNet/ACL runtime | fail reset | `NO` |
| `algorithm_config` | frozen algorithm and profile parameters | structured configuration | run-constant | this specification | fail configuration | `NO` |
| `dependency_identity` | SB3/MetaDrive/ScenarioNet revisions | strings/digests | run-constant | repository and manifest | fail scientific checkpoint | `NO` |

### 4.2 Prohibited information

The policy, value function, critics, actor, encoder, and action adapter shall not
receive, directly or indirectly:

- future actor tracks or future validity masks;
- future SDC states or trajectory samples;
- future traffic-light phases;
- other actors' ground-truth routes, checkpoints, intents, or future movement
  choices;
- future collision, minimum-distance, success, failure, or episode outcome;
- dataset source label `Waymo` or `PG` as a feature;
- ACL arm, sampling probability, usefulness, staleness, capacity state, or MAB
  state;
- `learning_potential`;
- Rulebook margins, costs, statuses, criticality, satisfaction pattern, or
  safety rank;
- scalarization contributions or continuous tie-breaker;
- native environment reward;
- evaluation metrics;
- actor IDs, object-selection scores, or overflow diagnostics;
- replay priorities or importance-sampling weights as observation features;
- critic values, TD errors, advantages, gradients, or target values as
  observation features.

Offline future information may be used only where an authoritative upstream
specification explicitly permits frozen dataset annotation. It cannot be
accessed at reset, observation construction, policy inference, transition
update, or evaluation inference.

---

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `action` | normalized ego control | `(2,)`, finite, `[-1,1]^2` | steering, longitudinal | `ThesisScenarioEnv` | clipped/constructed within action bounds |
| `policy_state` | trainable actor/policy parameters | algorithm-specific tensors | versioned | training/checkpoint | finite after every accepted update |
| `value_state` | critic/value parameters | algorithm-specific tensors | versioned | training/checkpoint | finite after every accepted update |
| `target_state` | TD3/SAC target parameters | algorithm-specific tensors | ownership defined in §7 | training/checkpoint | no optimizer ownership |
| `rollout_state` | PPO temporary on-policy buffer | global size 2,048 | chronological per worker | PPO updater | discarded after scheduled update |
| `replay_state` | TD3/SAC off-policy replay | capacity 300,000 or 1,000,000 | ring-buffer semantics | TD3/SAC updater | persistence optional, default OFF |
| `loss_metrics` | policy/value/critic/entropy losses | finite scalars | named by algorithm | diagnostics | not policy-visible |
| `replay_metrics` | PER/N-step statistics | finite scalars | named schema | diagnostics | required when applicable |
| `learning_potential` | ACL usefulness input | finite scalar per episode | algorithm-specific contract | ACL only | computed per authoritative ACL formula; no policy-loss effect |
| `checkpoint_generation` | model and scientific metadata generation | artifact set | atomic generation identifier | resume/evaluation | compatibility-validated |
| `evaluation_record` | deterministic evaluation outputs | structured metrics | scenario/episode ordering | reporting protocol | no training-state mutation |

---

## 6. Functional Requirements

### REQ-RLB-001: Scientific Backend Family

- Required observable behavior:
  - accept exactly `ppo_sb3`, `td3_sb3`, and `sac_sb3` for scientific baseline
    runs;
  - reject `ppo`, `td3`, and `sac` in scientific baseline presets.
- Applicability: all training, validation, test, and checkpoint-resume runs
  claiming conformance to `RL-BASELINES v1.0`.
- Invariants:
  - a legacy backend is never selected as a silent fallback;
  - algorithm identity remains fixed for the run.
- Failure behavior: configuration fails before environment or learner creation.
- Interactions: `ENC-V1.0`, local SB3 fork identity, checkpoint manifest.

### REQ-RLB-002: Environment And Action Interface

- Required observable behavior:
  - use `ThesisScenarioEnv` over the frozen ScenarioNet dataset;
  - expose a two-dimensional continuous action space `[-1,1]^2`;
  - apply one action every 0.1 s;
  - reject non-finite actions;
  - ensure the environment receives actions within bounds.
- Invariants:
  - no additional learned low-level controller is inserted;
  - action ordering is steering then longitudinal command;
  - algorithm comparisons use the same action adapter.
- Edge cases:
  - TD3 exploration noise and target noise are clipped to valid bounds;
  - SAC and PPO use bounded action distributions according to the selected SB3
    policy implementation.
- Failure behavior: invalid spaces or action shapes fail fast.

### REQ-RLB-003: Common Observation And Encoder

- Required observable behavior:
  - use `SemanticStateObservationV2`, shape `(2541,)`;
  - use `LatentQueryEncoderV2`, output dimension 256;
  - use 16 latent queries, latent dimension 128, four blocks, four attention
    heads, FFN dimension 256, mean pooling, zero dropout, and zero attention
    dropout;
  - train the encoder end-to-end.
- Invariants:
  - no legacy semantic observation;
  - no dense LiDAR augmentation;
  - no online normalization statistics outside the authoritative observation
    contract;
  - architecture identity is equal across PPO, TD3, and SAC except for approved
    ownership/sharing semantics.
- Failure behavior: schema, shape, token-count, mask, or fingerprint mismatch
  fails construction or checkpoint load.

### REQ-RLB-004: Reward And Diagnostics Boundary

- Required observable behavior:
  - use Rulebook v4.7 margins and `bounded_satisfaction_rank`;
  - set native reward mixing weight to exactly zero;
  - deliver numerically equal scalar rewards to PPO, TD3, and SAC for identical
    canonical margins and scalarizer configuration;
  - retain the ordered margin vector and Rulebook diagnostics separately.
- Invariants:
  - no algorithm-specific reward scaling, offset, clipping, or normalization;
  - TD3/SAC N-step accumulation sums already scalarized per-step rewards;
  - PER priorities derive from scalar critic errors, not directly from margins.
- Failure behavior: missing or incompatible reward semantics fails before
  training or checkpoint load.

### REQ-RLB-005: TD3 Architecture And Ownership

- Required observable behavior:
  - use one online deterministic actor, twin online critics, one target actor,
    and twin target critics;
  - use an actor encoder independent from the critic encoder;
  - share one critic encoder between Q1 and Q2;
  - use distinct target copies of actor and critic encoders;
  - use `[256,256]` ReLU actor and critic heads.
- Invariants:
  - actor optimizer owns actor encoder and actor head only;
  - critic optimizer owns critic encoder and both critic heads only;
  - target parameters belong to no optimizer and receive no gradients;
  - critic updates do not change actor parameters;
  - actor updates do not change critic parameters;
  - target updates occur only through Polyak averaging.
- Failure behavior: ownership or gradient-routing mismatch blocks training
  eligibility.

### REQ-RLB-006: TD3 Learning And Exploration

- Required observable behavior:
  - start gradient updates after 10,000 global environment transitions;
  - use batch size 256, replay capacity 300,000, learning rate `3e-4`,
    `gamma=0.99`, and `tau=0.005`;
  - use `train_freq=1` vector iteration and `gradient_steps=auto`;
  - resolve `gradient_steps=n_envs`;
  - update critics on every gradient step;
  - update actor and all target networks every second global gradient step;
  - use Gaussian behavior noise with mean 0 and standard deviation 0.1 during
    training;
  - use target-policy noise standard deviation 0.2 clipped to `[-0.5,0.5]`;
  - use deterministic action without behavior noise during primary evaluation.
- Invariants:
  - effective critic update-to-data ratio is one after warm-up;
  - batch size 2,048 is not conformant to the v1 scientific preset;
  - OOM does not silently modify batch size or architecture.
- Failure behavior: unresolved `auto`, wrong policy delay, wrong noise, or wrong
  batch fails preset validation.

### REQ-RLB-007: SAC Architecture And Ownership

- Required observable behavior:
  - use one stochastic actor, twin online critics, and twin target critics;
  - use an actor encoder independent from the critic encoder;
  - share one critic encoder between Q1 and Q2;
  - use one distinct target critic encoder;
  - use no target actor;
  - use `[256,256]` ReLU actor and critic heads.
- Invariants:
  - actor optimizer owns actor encoder and actor head only;
  - critic optimizer owns critic encoder and both critic heads only;
  - target critic parameters belong to no optimizer;
  - entropy-coefficient parameters are owned only by the entropy optimizer;
  - target critic updates occur only through Polyak averaging.
- Failure behavior: ownership, target-actor, or sharing mismatch blocks training
  eligibility.

### REQ-RLB-008: SAC Learning And Stochasticity

- Required observable behavior:
  - start gradient updates after 100 global environment transitions;
  - use batch size 256, replay capacity 1,000,000, learning rate `3e-4`,
    `gamma=0.99`, and `tau=0.005`;
  - use `train_freq=1` and `gradient_steps=auto`;
  - resolve `gradient_steps=n_envs`;
  - update critic, actor, and automatic entropy coefficient on every gradient
    step after warm-up;
  - use automatic entropy coefficient with initial `alpha=1.0`;
  - use automatic target entropy equal to `-dim(A)=-2`;
  - use no external action noise;
  - use stochastic reparameterized actions during training;
  - use the deterministic mean action during primary evaluation.
- Invariants:
  - effective update-to-data ratio is one after warm-up;
  - PER importance weights do not weight actor or entropy-coefficient losses;
  - the target-critic update interval and all effective dependency defaults are
    serialized in the manifest and verified against the pinned fork.
- Failure behavior: wrong entropy target, external noise, or unresolved update
  schedule fails validation.

### REQ-RLB-009: PPO Architecture And Ownership

- Required observable behavior:
  - use one encoder shared by policy and value;
  - use policy and value heads `[256,256]` with ReLU;
  - set `share_features_extractor=true` and `ortho_init=false`;
  - include the encoder exactly once in the PPO optimizer;
  - allow both policy and value losses to update the shared encoder.
- Invariants:
  - no separate policy/value encoder in v1;
  - no stock `[64,64]` Tanh encoder-controlled head;
  - PPO construction does not reinitialize the custom encoder.
- Failure behavior: sharing, optimizer duplication, architecture, or
  initialization mismatch blocks training eligibility.

### REQ-RLB-010: PPO Rollout, GAE, And Optimization

- Required observable behavior:
  - preserve `global_rollout_size=2048` in every supported profile;
  - use `n_steps=2048` for `n_envs=1` and `n_steps=512` for `n_envs=4`;
  - use batch size 64, 10 epochs, learning rate `3e-4`, `gamma=0.99`,
    `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0`, `vf_coef=0.5`,
    `normalize_advantage=true`, `max_grad_norm=0.5`, and `use_sde=false`;
  - perform 32 minibatches per epoch and 320 optimizer steps per PPO update;
  - discard rollout data after the update.
- Invariants:
  - PPO `n_steps` is rollout length per environment, not transition-level
    N-step return;
  - PPO uses no off-policy replay and no PER;
  - rollout size is divisible by batch size;
  - update cadence is every 2,048 global transitions in both supported profiles.
- Failure behavior: unsupported worker count, non-divisible rollout, active
  transition replay, or altered global rollout fails configuration.

### REQ-RLB-011: TD3/SAC N-Step And PER Profile

- Required observable behavior:
  - set `transition_replay.enabled=true`;
  - set `n_steps=3`;
  - set `prioritized=true`;
  - use proportional stratified PER;
  - apply normalized importance-sampling weights only to twin critic losses;
  - update priorities from the exact scalar TD target and mean absolute twin TD
    error;
  - use all defaults and boundary semantics from `TRANSITION-REPLAY v1.0`.
- Frozen PER values:
  - `alpha=0.6`;
  - `beta_initial=0.4`;
  - `beta_final=1.0`;
  - annealing horizon `${experiment.total_timesteps}` in global environment
    steps;
  - `epsilon=1e-6`;
  - `new_transition_priority=current_max`;
  - duplicate update reduction `max`;
  - sum-tree accumulation `float64`.
- Invariants:
  - supported N-step domain remains `{1,3}`, but the v1 baseline selects `3`;
  - no support for `5`;
  - no scalarization after margin-vector temporal accumulation;
  - actor and SAC alpha losses are not IS-weighted.
- Failure behavior: any mismatch fails before training.

### REQ-RLB-012: PPO Replay Incompatibility

- Required observable behavior:
  - set `transition_replay.enabled=false`;
  - reject PER, replay persistence, custom replay class, reward-vector replay,
    or off-policy N-step settings for PPO.
- Invariants:
  - temporal credit assignment is GAE-based;
  - no replay option is silently ignored.
- Failure behavior: configuration fails before learner construction.

### REQ-RLB-013: ScenarioNet Termination And Truncation

- Required observable behavior:
  - collision with a vehicle, object, or VRU sets `terminated=true`;
  - valid route success sets `terminated=true`;
  - physical departure from the road surface sets `terminated=true`;
  - crossing only a solid line is non-terminal;
  - route deviation is non-terminal;
  - `episode_steps >= scenario.length + 50` sets `truncated=true`, unless a true
    termination has already occurred;
  - `truncate_as_terminate=false`.
- Invariants:
  - this specification adopts the authoritative ScenarioNet v1.1 contract and
    does not define a separate `strict` or `relaxed` profile;
  - when termination and truncation coincide, true termination prevails for
    bootstrap masking.
- Failure behavior: environment behavior that differs from this contract blocks
  scientific eligibility.

### REQ-RLB-014: Boundary And Auto-Reset Semantics

- Required observable behavior:
  - preserve `terminated` and `truncated` separately;
  - include the boundary-transition reward;
  - provide the pre-reset final observation for completed vector workers;
  - TD3/SAC true termination disables bootstrap;
  - TD3/SAC timeout truncation retains bootstrap using the final observation;
  - PPO timeout truncation adds `gamma * V(final_observation)` to the boundary
    reward before rollout insertion;
  - auto-reset returns the next episode's reset observation only after terminal
    data have been normalized.
- Invariants:
  - reset observation is never substituted for a missing final observation;
  - rewards or N-step sequences never cross episode boundaries.
- Failure behavior: missing bootstrappable final observation is fatal.

### REQ-RLB-015: Vectorized Execution Profiles

- Required observable behavior:
  - permit exactly the profiles in §3.4 for scientific baseline runs;
  - use four process-based workers for the non-ACL vectorized profile;
  - use one environment when ACL is enabled;
  - reject `ACL ON` with `n_envs>1` in v1;
  - freeze `n_envs` for learner and replay lifetime;
  - derive deterministic per-worker scenario seeds and partitions.
- Invariants:
  - global budget accounting is independent of worker count;
  - TD3/SAC `gradient_steps=auto` resolves to one gradient update per collected
    transition;
  - PPO global rollout and sample reuse are invariant across supported profiles.
- Failure behavior: unsupported worker count or runtime worker-count change fails
  initialization or resume.

### REQ-RLB-016: ACL Compatibility

- Required observable behavior:
  - support ACL only through the approved single-environment runtime;
  - expose algorithm-specific learning-potential data according to ACL v1 §12,
    §28, and ADR-014;
  - keep learning potential outside policy optimization and evaluation metrics;
  - allow scenario MAB, Generate/Exploit, replay, staleness, warm-up,
    capacity/replacement, and checkpoint/resume behavior defined by ACL v1.
- Prohibited behavior:
  - mutation or mutation-generated child scenarios;
  - `C_safe`;
  - Rulebook margins, safety rank, or criticality in usefulness;
  - `use_rule_criticality=true`;
  - loss-proxy learning potential that differs from the approved
    algorithm-specific formula.
- Failure behavior: incompatible ACL configuration fails before training.

### REQ-RLB-017: Train And Evaluation Modes

- Required observable behavior:
  - set all trainable modules to training mode during optimization;
  - set policy and encoder modules to evaluation mode for validation/test;
  - use no gradient recording during evaluation;
  - use deterministic primary actions for PPO, TD3, and SAC evaluation;
  - disable TD3 behavior noise during evaluation;
  - perform no replay insertion, optimizer update, target update, ACL update,
    PER update, or running-statistic mutation during evaluation.
- Invariants:
  - validation and test use frozen checkpoints and frozen scalarization;
  - stochastic evaluation, if performed, is a separately labeled diagnostic.
- Failure behavior: detected training-state mutation invalidates evaluation.

### REQ-RLB-018: Hyperparameter Freeze And Fairness

- Required observable behavior:
  - freeze all values in §9 after integration/learner smoke passes;
  - do not select values based on short-smoke reward;
  - use equal dataset, reward, observation, encoder, global transition budget,
    seed set, and evaluation protocol across algorithms;
  - allow algorithm-native values where parameters have different meanings;
  - compare ACL ON versus ACL OFF at `n_envs=1` until ACL vectorization exists.
- Extension inheritance:
  - a lexicographic or distributional variant inherits all parent baseline
    hyperparameters not required to change by its approved extension;
  - any parent/extension deviation requires explicit justification and a paired
    parent control with the same deviation when needed for fairness.
- Failure behavior: unregistered override produces a non-conformant run.

### REQ-RLB-019: Scientific Checkpoint Generation

- Required observable behavior:
  - publish scientific checkpoints through an atomic manifest-based generation
    path;
  - include model, optimizer state available through the SB3 artifact, training
    counters, root and component RNG state, configuration digest, dataset
    fingerprint, algorithm identity, action/observation spaces, encoder identity,
    Rulebook identity, scalarization identity, replay identity, ACL state when
    enabled, dependency versions, and fork commit;
  - validate compatibility before loading the learner;
  - write reward-semantics identity as part of the generation.
- Invariants:
  - checkpoints are not portable across PPO, TD3, and SAC;
  - observation/encoder schema changes fail load;
  - scalarizer or Rulebook changes fail same-run resume;
  - dataset fingerprint mismatch fails same-run resume;
  - fork mismatch fails by default unless an explicitly approved migration is
    performed as a new run.
- Failure behavior: incomplete, torn, or mismatched generation is rejected.

### REQ-RLB-020: Resume Semantics

- Required observable behavior:
  - replay persistence is optional and disabled by default;
  - model plus compatible replay/RNG/training/ACL state may be labeled
    `stateful_continuation`;
  - TD3/SAC load without replay is labeled `model_only_restart`, starts a new
    replay segment, resets PER beta progress according to `TRANSITION-REPLAY`,
    and sets `continuation_equivalent=false`;
  - PPO resume restores model/optimizer/training counters but does not persist a
    partially collected rollout buffer;
  - transfer initialization is a new run with new manifest identity.
- Invariants:
  - no claim of bitwise-identical continuation;
  - no silent replay reconstruction;
  - no change of `n_envs` for stateful continuation.
- Failure behavior: incompatible state is rejected before learning continues.

### REQ-RLB-021: Anti-Leakage

- Required observable behavior:
  - satisfy all anti-leakage tests in `OBS-V1.1`, `SCAL-V1.0`, ScenarioNet v1.1,
    and ACL v1;
  - ensure the learner input changes only when authorized causal policy-visible
    inputs change.
- Invariants:
  - diagnostic, curriculum, future, offline-only, and evaluator-generated values
    remain outside policy input;
  - reward-vector diagnostics do not enter the encoder.
- Failure behavior: any causal-leakage failure blocks scientific training.

### REQ-RLB-022: Resource Behavior And No Silent Degradation

- Required observable behavior:
  - record CPU RAM, allocated/reserved GPU memory, peak GPU memory, and update
    time during learner smoke;
  - start learner smoke only with sufficient available GPU memory;
  - stop with an explicit error on OOM or non-finite update.
- Prohibited fallback:
  - reducing batch size;
  - reducing worker count;
  - changing encoder dimensions;
  - disabling PER;
  - changing N-step horizon;
  - switching algorithm backend;
  - switching reward/observation mode.
- Invariants:
  - any resource-motivated scientific change requires an approved specification
    revision or ADR and parent-matched comparison controls.

---

## 7. Mathematical And Algorithmic Contract

### 7.1 Common scalar objective

Each scalar baseline seeks to maximize expected discounted scalarized return:

\[
J(\pi)=
\mathbb E_{\pi}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t^{\mathrm{scalar}}
\right].
\]

This objective does not establish lexicographic expected-return optimality. The
Rulebook margin vector remains the primary behavioral diagnostic.

### 7.2 N-step scalar return

For TD3 and SAC, the replay sample provides an effective horizon \(m_i\le 3\):

\[
R_i^{(m_i)}=
\sum_{k=0}^{m_i-1}\gamma^k
r_{i+k}^{\mathrm{scalar}}.
\]

Let \(b_i=0\) for true termination and \(b_i=1\) when a valid bootstrap is
permitted, including timeout-only truncation. The bootstrap coefficient is:

\[
d_i = \gamma^{m_i} b_i.
\]

The replay implementation shall provide or preserve this per-sample discount.

### 7.3 TD3 target and losses

Target-policy smoothing:

\[
\epsilon_i \sim \mathcal N(0,0.2^2),
\qquad
\bar\epsilon_i=
\operatorname{clip}(\epsilon_i,-0.5,0.5),
\]

\[
\tilde a_i=
\operatorname{clip}
\left(
\mu'(s'_i)+\bar\epsilon_i,
-1,
1
\right).
\]

Target:

\[
y_i^{\mathrm{TD3}}=
R_i^{(m_i)}+
 d_i
\min_{j\in\{1,2\}}
Q'_j(s'_i,\tilde a_i).
\]

Twin TD errors:

\[
\delta_{i,j}=y_i^{\mathrm{TD3}}-Q_j(s_i,a_i).
\]

Weighted critic loss:

\[
L_Q^{\mathrm{TD3}}
=
\frac{1}{B}
\sum_{i=1}^{B}
 w_i
\frac{1}{2}
\left(
\delta_{i,1}^2+
\delta_{i,2}^2
\right).
\]

Actor loss on the sampled batch, without IS weighting:

\[
L_\mu =
-\frac{1}{B}
\sum_{i=1}^{B}
Q_1(s_i,\mu(s_i)).
\]

The critic is updated every gradient step. The actor and all TD3 target
networks are updated when the global TD3 gradient-step counter is divisible by
`policy_delay=2`.

Polyak update:

\[
\theta'\leftarrow
\tau\theta+(1-\tau)\theta',
\qquad \tau=0.005.
\]

### 7.4 SAC target and losses

Sample the next action through the reparameterized squashed policy:

\[
a'_i\sim\pi_\theta(\cdot\mid s'_i),
\qquad
\log\pi_\theta(a'_i\mid s'_i).
\]

Target:

\[
y_i^{\mathrm{SAC}}=
R_i^{(m_i)}+
 d_i
\left[
\min_{j\in\{1,2\}}Q'_j(s'_i,a'_i)
-
\alpha_{\mathrm{ent}}
\log\pi_\theta(a'_i\mid s'_i)
\right].
\]

Weighted twin-critic loss:

\[
L_Q^{\mathrm{SAC}}
=
\frac{1}{B}
\sum_{i=1}^{B}
 w_i
\frac{1}{2}
\sum_{j=1}^{2}
\left(
y_i^{\mathrm{SAC}}-Q_j(s_i,a_i)
\right)^2.
\]

Actor loss, without IS weighting:

\[
L_\pi^{\mathrm{SAC}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
\alpha_{\mathrm{ent}}
\log\pi_\theta(a_i^\pi\mid s_i)
-
\min_j Q_j(s_i,a_i^\pi)
\right],
\]

where \(a_i^\pi\) is reparameterized from the current actor.

Automatic entropy-coefficient loss:

\[
L_\alpha=
-\frac{1}{B}
\sum_{i=1}^{B}
\log\alpha_{\mathrm{ent}}
\left(
\log\pi_\theta(a_i^\pi\mid s_i)
+
\mathcal H_{\mathrm{target}}
\right),
\]

with:

\[
\mathcal H_{\mathrm{target}}=-2,
\qquad
\alpha_{\mathrm{ent},0}=1.0.
\]

The critic, actor, and entropy coefficient are updated on every SAC gradient
step after warm-up. Target critics use Polyak averaging with `tau=0.005` at the
pinned fork's verified effective target-update interval.

### 7.5 PER priority and sampling

Raw priority:

\[
p_i=
\frac{
|\delta_{i,1}|+|\delta_{i,2}|
}{2}
+
10^{-6}.
\]

Sampling probability:

\[
P(i)=
\frac{p_i^{0.6}}
{\sum_k p_k^{0.6}}.
\]

Importance-sampling weight:

\[
\tilde w_i=
\left(
\frac{1}{N_{\mathrm{active}}P(i)}
\right)^{\beta},
\qquad
w_i=
\frac{\tilde w_i}{\max_j \tilde w_j}.
\]

`beta` is annealed linearly from 0.4 to 1.0 over the configured global
interaction budget of the replay-training segment. Duplicate sampled indices
are reduced with `max` before updating stored priorities.

### 7.6 PPO GAE

For a non-terminal transition:

\[
\delta_t=
 r_t+
 \gamma V(s_{t+1})-
 V(s_t).
\]

For a true terminal transition, the value bootstrap is zero. For timeout-only
truncation, the valid final observation supplies the value bootstrap according
to §6 `REQ-RLB-014`.

GAE:

\[
\hat A_t=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l},
\qquad
\lambda=0.95.
\]

Returns:

\[
\hat R_t=
\hat A_t+V(s_t).
\]

When `normalize_advantage=true`, advantages are normalized within each PPO
optimization minibatch according to the pinned implementation.

### 7.7 PPO clipped objective

Probability ratio:

\[
r_t(\theta)=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
}.
\]

Clipped policy objective:

\[
L^{\mathrm{clip}}(\theta)=
\mathbb E_t
\left[
\min
\left(
 r_t(\theta)\hat A_t,
 \operatorname{clip}(r_t(\theta),0.8,1.2)\hat A_t
\right)
\right].
\]

The optimizer minimizes the SB3-equivalent combined loss:

\[
L_{\mathrm{PPO}}=
-L^{\mathrm{clip}}
+
0.5 L_V
-
0\cdot H[\pi],
\]

with value coefficient 0.5, entropy coefficient 0, and global gradient norm
clipped to 0.5. Any effective unlisted PPO dependency defaults, including value
clipping behavior and optimizer details, must be verified against the pinned
fork and serialized in the manifest before approval.

### 7.8 Vectorized update accounting

For TD3/SAC:

\[
\text{resolved gradient steps}
=
\max(
\text{train frequency}\times n_{\mathrm{envs}},
1
).
\]

With `train_freq=1`:

```text
n_envs=1: collect 1 transition → 1 gradient step
n_envs=4: collect 4 transitions → 4 gradient steps
```

Thus the post-warm-up critic update-to-data ratio is one in both profiles.

For PPO:

\[
\text{global rollout size}=
\text{n_steps per env}\times n_{\mathrm{envs}}=2048.
\]

```text
n_envs=1: 2048 × 1 = 2048
n_envs=4:  512 × 4 = 2048
```

With batch size 64 and 10 epochs:

\[
2048/64=32
\]

minibatches per epoch and:

\[
32\times 10=320
\]

optimizer steps per PPO update.

---

## 8. Applicability, State, And Timing

### 8.1 Initialization order

Normative initialization order:

```text
1. resolve Hydra configuration
2. validate scientific execution profile
3. validate dataset fingerprint and split
4. build ThesisScenarioEnv / vector workers
5. validate action and observation spaces
6. build Rulebook v4.7 and scalarizer
7. build SemanticStateObservationV2
8. build LQ encoder and planner heads
9. build algorithm and buffer
10. validate manifest identity template
11. initialize ACL state if enabled
12. begin collection
```

No component may be silently replaced when a later validation fails.

### 8.2 Environment reset

At reset:

- assigned route and static causal context are validated;
- observation and Rulebook state are reset atomically;
- no transition from a previous episode remains active in rollout/N-step state;
- ACL selects the scenario before reset in the single-environment ACL profile;
- reset observation is never used as the final observation of the preceding
  episode.

### 8.3 Per-step order

Normative per-step ordering:

```text
1. consume committed observation s_t
2. select training or evaluation action a_t
3. apply action to ThesisScenarioEnv
4. obtain post-step state, scalar reward, diagnostics, terminated, truncated
5. normalize boundary and final observation
6. commit causal memory and construct s_{t+1}
7. insert transition into rollout/replay if training
8. close episode and update ACL if applicable
9. execute scheduled algorithm update
10. publish diagnostics/checkpoint when due
```

The exact observation-memory transaction order remains authoritative in
`OBS-V1.1`.

### 8.4 PPO state lifecycle

- rollout buffer capacity is exactly 2,048 global transitions;
- collection uses one frozen behavior-policy version per rollout;
- after collection, 10 shuffled optimization epochs are executed;
- partially collected rollout state is not a scientific checkpoint artifact;
- after update, the rollout buffer is discarded/reset;
- evaluation does not fill the rollout buffer.

### 8.5 TD3/SAC replay lifecycle

- each transition is stored with separate termination/truncation semantics and a
  valid final observation when required;
- N-step returns are assembled without crossing environment or episode
  boundaries;
- PER state uses a buffer-owned RNG;
- model-only restart creates a new replay segment;
- replay persistence, when explicitly enabled, uses the compatibility contract
  of `TRANSITION-REPLAY v1.0`.

### 8.6 ACL lifecycle

The current ACL runtime is single-environment:

```text
select arm/scenario
→ reset
→ collect one complete episode
→ compute algorithm-specific learning potential
→ update MAB and scenario replay state
→ select next scenario
```

The learner must not use a vector auto-reset path that bypasses this episode
closure. Simultaneous multi-worker ACL update ordering is deferred to the
separate ACL-vectorization feature.

### 8.7 Training and evaluation dataset use

- training updates use only the train split;
- validation is used for model selection according to an approved protocol;
- test is used only for final evaluation;
- the golden suite is used only by the dedicated Rulebook/adapter diagnostic;
- validation/test never update policy, critic, replay, scalarizer, ACL, or
  normalization state;
- no test result is used to tune hyperparameters.

---

## 9. Configuration

### 9.1 Common scientific configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen for experiments |
|---|---|---|---|---|---|---|
| `agent.backend` | enum | algorithm-specific `*_sb3` | `ppo_sb3`, `td3_sb3`, `sac_sb3` | scientific backend | YES | YES |
| `env.class` | enum | `ThesisScenarioEnv` | exact value | environment | YES | YES |
| `env.control_dt_s` | float | `0.1` | exact value | control interval | YES | YES |
| `env.action_shape` | tuple | `(2,)` | exact value | normalized action | YES | YES |
| `env.action_low/high` | tuple | `(-1,-1)/(1,1)` | exact value | action bounds | YES | YES |
| `observation.type` | enum | `semantic_v2` | exact authoritative alias | observation | YES | YES |
| `observation.flat_dim` | int | `2541` | exact value | flat input dimension | YES | YES |
| `encoder.type` | enum | `lq` / `LatentQueryEncoderV2` | exact approved alias | encoder | YES | YES |
| `encoder.output_dim` | int | `256` | exact value | planner feature dimension | YES | YES |
| `reward.behavior` | enum | `scalar_reward` | exact value | learner reward path | YES | YES |
| `scalarization.mode` | enum | `bounded_satisfaction_rank` | exact value | scalar formula | YES | YES |
| `reward.native_weight` | float | `0.0` | exact value | native reward mixing | YES | YES |
| `dataset.fingerprint` | string | protected fingerprint | exact value | dataset identity | YES | YES |
| `runtime.profile` | enum | `acl_single_env` | profiles in §3.4 | execution profile | YES | YES |
| `experiment.total_timesteps` | int | external protocol | `>0` | global transition budget | YES | YES per run |
| `run_seed` | int | external protocol | integer | root RNG seed | YES | YES per run |

### 9.2 LQ encoder configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---|---|---|---|---|
| `encoder.num_latents` | int | `16` | exact value | latent queries | YES | YES |
| `encoder.latent_dim` | int | `128` | exact value | latent width | YES | YES |
| `encoder.num_layers` | int | `4` | exact value | LQ blocks | YES | YES |
| `encoder.num_heads` | int | `4` | exact value | attention heads | YES | YES |
| `encoder.ffn_dim` | int | `256` | exact value | FFN width | YES | YES |
| `encoder.pooling` | enum | `mean` | exact value | latent aggregation | YES | YES |
| `encoder.dropout` | float | `0.0` | exact value | dropout | YES | YES |
| `encoder.attention_dropout` | float | `0.0` | exact value | attention dropout | YES | YES |
| `planner.net_arch` | list[int] | `[256,256]` | exact value | policy/value/Q heads | YES | YES |
| `planner.activation` | enum | `ReLU` | exact value | head activation | YES | YES |

### 9.3 TD3 configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---:|---|---|---|---|
| `learning_starts` | int | `10000` | exact core value | global warm-up transitions | YES | YES |
| `batch_size` | int | `256` | exact core value | replay minibatch | YES | YES |
| `buffer_size` | int | `300000` | exact core value | replay capacity | YES | YES |
| `learning_rate` | float | `3e-4` | exact core value | actor/critic LR unless fork separates | YES | YES |
| `gamma` | float | `0.99` | exact core value | discount | YES | YES |
| `tau` | float | `0.005` | exact core value | Polyak factor | YES | YES |
| `train_freq` | int | `1` | exact value | vector iterations per update call | YES | YES |
| `gradient_steps` | enum | `auto` | exact value | resolved to `n_envs` | YES | YES |
| `policy_delay` | int | `2` | exact value | delayed actor/target cadence | YES | YES |
| `target_policy_noise` | float | `0.2` | exact value | target action noise sigma | YES | YES |
| `target_noise_clip` | float | `0.5` | exact value | target noise absolute clip | YES | YES |
| `action_noise.type` | enum | `normal` | exact value | behavior exploration | YES | YES |
| `action_noise.sigma` | float | `0.1` | exact value per action dim | behavior noise | YES | YES |

### 9.4 SAC configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---:|---|---|---|---|
| `learning_starts` | int | `100` | exact core value | global warm-up transitions | YES | YES |
| `batch_size` | int | `256` | exact core value | replay minibatch | YES | YES |
| `buffer_size` | int | `1000000` | exact core value | replay capacity | YES | YES |
| `learning_rate` | float | `3e-4` | exact core value | actor/critic/alpha LR unless verified otherwise | YES | YES |
| `gamma` | float | `0.99` | exact core value | discount | YES | YES |
| `tau` | float | `0.005` | exact core value | Polyak factor | YES | YES |
| `train_freq` | int | `1` | exact value | vector iterations per update call | YES | YES |
| `gradient_steps` | enum | `auto` | exact value | resolved to `n_envs` | YES | YES |
| `ent_coef` | enum | `auto` | exact value | learned entropy coefficient | YES | YES |
| `ent_coef_initial` | float | `1.0` | exact value | initial alpha | YES | YES |
| `target_entropy` | enum | `auto` | resolves to `-2` | entropy target | YES | YES |
| `external_action_noise` | bool | `false` | exact value | extra behavior noise | YES | YES |
| `target_update_interval` | int | effective pinned-fork value, expected `1` | must be verified before approval | target cadence | YES | YES |

### 9.5 PPO configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---:|---|---|---|---|
| `global_rollout_size` | int | `2048` | exact value | transitions per PPO update | YES | YES |
| `n_steps` | int | profile-derived `2048` or `512` | exact profile mapping | steps per environment | YES | YES |
| `batch_size` | int | `64` | exact value | minibatch | YES | YES |
| `n_epochs` | int | `10` | exact value | reuse epochs | YES | YES |
| `learning_rate` | float | `3e-4` | exact value | optimizer LR | YES | YES |
| `gamma` | float | `0.99` | exact value | discount | YES | YES |
| `gae_lambda` | float | `0.95` | exact value | GAE trace parameter | YES | YES |
| `clip_range` | float | `0.2` | exact value | policy ratio clip | YES | YES |
| `ent_coef` | float | `0.0` | exact value | entropy loss weight | YES | YES |
| `vf_coef` | float | `0.5` | exact value | value loss weight | YES | YES |
| `normalize_advantage` | bool | `true` | exact value | advantage normalization | YES | YES |
| `max_grad_norm` | float | `0.5` | exact value | gradient clipping | YES | YES |
| `use_sde` | bool | `false` | exact value | state-dependent exploration | YES | YES |
| `share_features_extractor` | bool | `true` | exact value | policy/value encoder sharing | YES | YES |
| `ortho_init` | bool | `false` | exact value | custom encoder preservation | YES | YES |

### 9.6 Transition replay and PER configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---:|---|---|---|---|
| `transition_replay.enabled` | bool | TD3/SAC `true`; PPO `false` | algorithm mapping | off-policy replay feature | YES | YES |
| `transition_replay.n_steps` | int | TD3/SAC `3` | core exact; subsystem domain `{1,3}` | N-step horizon | conditional | YES |
| `transition_replay.prioritized` | bool | TD3/SAC `true` | exact core value | proportional PER | conditional | YES |
| `transition_replay.optimize_memory_usage` | bool | `false` | exact value | replay memory mode | conditional | YES |
| `transition_replay.store_reward_vector` | bool | `false` | exact value | vector replay | conditional | YES |
| `per.alpha` | float | `0.6` | exact core value | prioritization exponent | conditional | YES |
| `per.beta_initial` | float | `0.4` | exact core value | initial IS correction | conditional | YES |
| `per.beta_final` | float | `1.0` | exact core value | final IS correction | conditional | YES |
| `per.beta_anneal_steps` | int | `${experiment.total_timesteps}` | `>0` | global beta horizon | conditional | YES |
| `per.epsilon` | float | `1e-6` | exact core value | priority floor | conditional | YES |
| `per.priority_aggregation` | enum | `mean_abs_twin_td` | exact value | twin TD aggregation | conditional | YES |
| `per.sampling` | enum | `proportional_stratified` | exact value | sampling | conditional | YES |
| `per.duplicate_update_reduction` | enum | `max` | exact value | duplicate priority reduction | conditional | YES |
| `persistence.enabled` | bool | `false` | `false` core; `true` optional approved run | replay serialization | conditional | YES per run |

### 9.7 ACL and vectorization configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---|---|---|---|---|
| `acl.enabled` | bool | `true` for final pipeline | `true/false` according to profile | curriculum | YES | YES |
| `env.vectorized.enabled` | bool | profile-derived | ACL `false`; non-ACL `true` | vector runtime | YES | YES |
| `env.vectorized.num_envs` | int | profile-derived `1` or `4` | exact profile values | worker count | YES | YES |
| `acl.use_rule_criticality` | bool | `false` | exact value | prohibited usefulness input | YES | YES |
| `acl.mutation.enabled` | bool | `false` | exact value | mutation | YES | YES |
| `acl.learning_potential` | enum | algorithm-specific authoritative formula | exact ACL contract | usefulness | conditional | YES |

### 9.8 Configuration validation

Configuration loading shall fail before learner construction when:

- a scientific preset selects a legacy backend;
- required components do not compose exactly;
- observation or action spaces differ;
- reward mode or scalarization differs;
- native reward weight is nonzero;
- TD3 batch size is 2,048 or any value other than 256 in the v1 core preset;
- TD3/SAC PER is disabled;
- TD3/SAC `n_steps` differs from 3;
- PPO replay is active;
- PPO global rollout differs from 2,048;
- PPO rollout is not divisible by batch size;
- profile, ACL, worker count, and PPO `n_steps` are inconsistent;
- `gradient_steps=auto` cannot resolve to `n_envs`;
- `ACL ON` is combined with vectorization;
- checkpoint/reward/dataset/dependency identities mismatch;
- any frozen value is overridden without a new approved specification or ADR.

---

## 10. Errors, Logging, And Diagnostics

### 10.1 Fatal errors

Fatal conditions include:

- invalid observation/action shape, dtype, bounds, or non-finite value;
- non-finite reward, return, advantage, target, Q value, log probability,
  entropy coefficient, loss, gradient, or parameter;
- missing final observation at a bootstrappable truncation;
- unsupported termination/truncation behavior;
- rollout/replay crossing an episode boundary;
- invalid PER probability, priority, tree total, or IS weight;
- incorrect actor/critic/target ownership;
- target parameter receiving optimizer gradients;
- incompatible manifest or checkpoint generation;
- unsupported ACL/vectorization combination;
- OOM during required smoke or training;
- dataset fingerprint or split mismatch;
- selected fork commit mismatch;
- mutation of training state during evaluation.

### 10.2 No silent fallback

The implementation shall not silently:

- select a legacy backend;
- change worker count;
- change PPO rollout size;
- change TD3 batch size;
- reduce encoder capacity;
- disable PER;
- change `n_steps` from 3 to 1;
- reinterpret truncation as termination;
- replace missing final observation with reset observation or zeros;
- mix native reward;
- ignore PPO replay settings;
- reset or clip invalid priorities without a specified fatal path;
- load a partially compatible checkpoint;
- treat model-only restart as stateful continuation;
- use short-smoke reward to choose hyperparameters.

### 10.3 Required run metadata

Every scientific run shall log at initialization:

```text
specification IDs and versions
algorithm and backend
local SB3 version and commit
dependency and repository revisions
resolved Hydra configuration and digest
dataset fingerprint and split
run seed and component seed derivation
runtime profile and n_envs
global rollout size or train_freq/gradient_steps
observation and encoder fingerprints
Rulebook and scalarization identity
replay/PER/N-step identity
ACL identity and state schema
checkpoint/resume mode
```

### 10.4 Required common diagnostics

```text
train/global_env_steps
train/vector_iterations
train/episodes_completed
train/updates
train/update_to_data_ratio
train/learning_rate
train/gradient_norm
train/nonfinite_count
env/terminated_count
env/truncated_count
env/collision_count
env/route_success_count
env/out_of_road_count
env/horizon_truncation_count
reward/scalar_mean
reward/scalar_min
reward/scalar_max
resource/cpu_ram_bytes
resource/gpu_allocated_bytes
resource/gpu_reserved_bytes
resource/gpu_peak_allocated_bytes
resource/update_wall_time_s
```

### 10.5 TD3 diagnostics

```text
td3/critic_loss
td3/actor_loss
td3/critic_updates
td3/actor_updates
td3/target_updates
td3/action_noise_std
td3/target_noise_std
td3/target_noise_clip
td3/q1_mean
td3/q2_mean
td3/target_q_mean
```

### 10.6 SAC diagnostics

```text
sac/critic_loss
sac/actor_loss
sac/alpha_loss
sac/alpha
sac/entropy
sac/target_entropy
sac/q1_mean
sac/q2_mean
sac/target_q_mean
sac/target_updates
```

### 10.7 PPO diagnostics

```text
ppo/policy_loss
ppo/value_loss
ppo/entropy_loss
ppo/approx_kl
ppo/clip_fraction
ppo/explained_variance
ppo/rollout_size_global
ppo/minibatches_per_epoch
ppo/optimizer_steps_per_update
ppo/advantage_mean
ppo/advantage_std
```

### 10.8 Replay diagnostics

All diagnostics required by `TRANSITION-REPLAY v1.0` are mandatory when
applicable, including effective N-step horizon, early cutoff fractions, beta,
priority statistics, sampling probabilities, IS weights, duplicate fraction,
and replay reset/continuation status.

### 10.9 ACL diagnostics

ACL logs remain governed by ACL v1. The baseline shall additionally log:

```text
acl/learning_potential_formula_id
acl/learning_potential
acl/learning_potential_gamma
acl/learning_potential_lambda_if_applicable
acl/vectorized_supported=false
```

These values are diagnostic/training-control data and are not policy-visible or
primary evaluation metrics.

---

## 11. Reproducibility And Compatibility

### 11.1 Seed ownership

A root run seed shall deterministically derive separate RNG streams for:

- environment/scenario sampling;
- each vector worker;
- policy initialization;
- PPO action sampling;
- TD3 behavior noise;
- SAC action reparameterization;
- replay sampling;
- PER sampling/tree state;
- minibatch shuffling;
- ACL MAB and scenario replay;
- evaluation scenario ordering.

The derivation scheme and resulting seeds shall be stored in the manifest.

### 11.2 Determinism limits

Conformance requires deterministic behavior under controlled CPU unit fixtures
and reproducible seed ownership. It does not guarantee bitwise-identical CUDA,
subprocess scheduling, or resumed simulator trajectories.

Results shall be reported over the approved seed set rather than relying on a
single deterministic trajectory.

### 11.3 Scientific manifest

The manifest shall include at least:

```text
RL-BASELINES version and status
all related specification versions
all related ADR identifiers
algorithm/backend
resolved algorithm parameters
runtime profile and n_envs
PPO global rollout or TD3/SAC update schedule
action and observation spaces
observation schema fingerprint
encoder architecture fingerprint
Rulebook schema and version
scalarization mode and parameters
dataset fingerprint and split
SB3 version and commit
MetaDrive and ScenarioNet revisions
replay/PER/N-step metadata
ACL state/configuration identity
root and component seeds
training counters
artifact digests
resume classification
```

### 11.4 Checkpoint compatibility

Same-run resume requires exact equality of all scientifically frozen identities
and parameters.

Model-only evaluation may use a different number of evaluation environments,
provided policy observation/action contracts are unchanged and evaluation is
clearly labeled. Stateful training continuation cannot change `n_envs`, replay
shape, rollout semantics, algorithm, observation, encoder, reward, dataset, or
fork identity.

### 11.5 Previous checkpoints

The following are incompatible with conformant v1 training resume:

- legacy algorithm checkpoints;
- observation schemas based on 2,363 features or 107 tokens;
- legacy semantic-state checkpoints;
- non-LQ core checkpoints;
- native/monitor-only/hybrid reward checkpoints;
- different scalarization modes;
- TD3 batch-2,048 scientific checkpoints;
- uniform-replay TD3/SAC checkpoints when the run claims core PER;
- N-step values other than 3;
- old SB3 fork commits without an approved migration;
- dataset fingerprints other than the protected fingerprint.

They may be retained only as historical artifacts or used as explicitly new
transfer-initialized runs after approval.

### 11.6 Experimental fairness

The final experimental protocol shall define exact seeds, budgets, evaluation
cadence, model-selection rule, and final evaluation episode count. This
specification requires that those values be common where semantically
applicable and that all budgets be expressed in global environment transitions.

Wall-clock time and worker throughput may be reported but shall not replace
sample-budget comparisons.

---

## 12. Acceptance Criteria

### AC-RLB-001: Scientific Backend Rejection

- Given: each of the six registered backend names.
- When: a conformant scientific baseline configuration is loaded.
- Then: `ppo_sb3`, `td3_sb3`, and `sac_sb3` are accepted for their respective
  presets, while `ppo`, `td3`, and `sac` fail before learner construction.
- Related requirements: `REQ-RLB-001`.

### AC-RLB-002: Environment And Action Contract

- Given: the core ScenarioNet environment configuration.
- When: the environment and each learner are constructed.
- Then: the action space is exactly finite `Box([-1,-1],[1,1])`, action shape is
  `(2,)`, control interval is 0.1 s, and invalid/non-finite actions are rejected.
- Related requirements: `REQ-RLB-002`.

### AC-RLB-003: Observation And LQ Contract

- Given: any core algorithm preset.
- When: a batch of valid observations is encoded.
- Then: input shape is `[B,2541]`, LQ tokenization/masks match `OBS-V1.1`, output
  shape is `[B,256]`, and outputs/gradients are finite.
- Related requirements: `REQ-RLB-003`, `REQ-RLB-021`.

### AC-RLB-004: Scalar Reward Parity

- Given: identical Rulebook v4.7 margins and scalarizer configuration.
- When: the reward is delivered to PPO, TD3, and SAC transition paths.
- Then: all three receive equal scalar values within `1e-6`; native reward and
  diagnostic-vector changes cannot alter the value.
- Related requirements: `REQ-RLB-004`.

### AC-RLB-005: TD3 Ownership And Gradient Routing

- Given: a deterministic TD3 update fixture.
- When: critic-only, actor-only, and target-update operations are executed.
- Then: only the specified online/target parameter sets change; Q1/Q2 share one
  critic encoder; targets receive no optimizer gradients.
- Related requirements: `REQ-RLB-005`.

### AC-RLB-006: TD3 Update Schedule

- Given: TD3 profiles with `n_envs=1` and `n_envs=4` after 10,000 global steps.
- When: one vector iteration is collected and updated.
- Then: one or four critic gradient steps occur respectively; the cumulative
  critic updates equal collected post-warm-up transitions; actor/target updates
  occur every second global gradient step.
- Related requirements: `REQ-RLB-006`, `REQ-RLB-015`.

### AC-RLB-007: TD3 Noise And Evaluation

- Given: fixed seeds and fixed observations.
- When: training and evaluation action selection are executed.
- Then: training includes configured Gaussian behavior noise and bounded target
  smoothing; repeated deterministic evaluation produces the same action and no
  behavior noise is sampled.
- Related requirements: `REQ-RLB-006`, `REQ-RLB-017`.

### AC-RLB-008: SAC Ownership And Gradient Routing

- Given: a deterministic SAC update fixture.
- When: critic, actor, alpha, and target operations are executed separately.
- Then: each optimizer changes only its owned parameters; no target actor exists;
  Q1/Q2 share one critic encoder; target critic has no optimizer ownership.
- Related requirements: `REQ-RLB-007`.

### AC-RLB-009: SAC Entropy Contract

- Given: a two-dimensional action space and automatic entropy mode.
- When: SAC is initialized and performs an update.
- Then: initial alpha is 1.0, target entropy is -2, alpha/actor/critic losses are
  finite, and no external action noise object is active.
- Related requirements: `REQ-RLB-008`.

### AC-RLB-010: SAC Update Schedule

- Given: SAC profiles with `n_envs=1` and `n_envs=4` after 100 global steps.
- When: one vector iteration is collected and updated.
- Then: one or four critic/actor/alpha gradient steps occur respectively, and
  cumulative gradient steps equal collected post-warm-up transitions.
- Related requirements: `REQ-RLB-008`, `REQ-RLB-015`.

### AC-RLB-011: PPO Sharing And Initialization

- Given: the PPO semantic-LQ preset.
- When: the model is constructed and separate policy/value backward fixtures are
  run.
- Then: one shared encoder exists, appears once in the optimizer, receives both
  gradient sources, uses `[256,256]` ReLU heads, and is not orthogonally
  reinitialized.
- Related requirements: `REQ-RLB-009`.

### AC-RLB-012: PPO Global Rollout Invariance

- Given: `acl_single_env`, `acl_control_single_env`, and
  `non_acl_vectorized` PPO profiles.
- When: the rollout configuration is resolved.
- Then: each profile has exactly 2,048 global transitions, 32 minibatches per
  epoch, and 320 optimizer steps per update; invalid worker/step combinations
  fail.
- Related requirements: `REQ-RLB-010`, `REQ-RLB-015`.

### AC-RLB-013: PPO GAE And Timeout Bootstrap

- Given: deterministic true-terminal and timeout-truncated trajectory fixtures.
- When: PPO returns and advantages are computed.
- Then: true termination has zero value bootstrap; timeout truncation uses the
  provided final observation; analytical GAE/return values match the
  implementation.
- Related requirements: `REQ-RLB-010`, `REQ-RLB-014`.

### AC-RLB-014: PPO Replay Rejection

- Given: PPO with any active/non-default transition-replay feature.
- When: configuration is loaded.
- Then: loading fails before model construction and no setting is silently
  ignored.
- Related requirements: `REQ-RLB-012`.

### AC-RLB-015: TD3/SAC Three-Step Scalar Target

- Given: hand-computed normal, terminal, truncation, frontier, and wrap-around
  replay sequences.
- When: TD3/SAC samples are constructed.
- Then: scalarized rewards, effective horizon, bootstrap mask, final
  observation, and per-sample discount match `TRANSITION-REPLAY v1.0`; no
  episode boundary is crossed.
- Related requirements: `REQ-RLB-011`, `REQ-RLB-014`.

### AC-RLB-016: PER Weighted Critic Integration

- Given: a deterministic prioritized batch with known Q predictions and target.
- When: TD3 and SAC critic updates execute.
- Then: priorities equal mean absolute twin TD error plus epsilon; sampling and
  normalized weights match the formula; critic losses are weighted; actor and
  SAC alpha losses are not weighted.
- Related requirements: `REQ-RLB-011`.

### AC-RLB-017: ScenarioNet Episode Semantics

- Given: separate fixtures for collision, route success, out-of-road, solid-line
  crossing, route deviation, and horizon exhaustion.
- When: `ThesisScenarioEnv` steps.
- Then: flags match exactly `REQ-RLB-013`; `truncate_as_terminate` remains false.
- Related requirements: `REQ-RLB-013`.

### AC-RLB-018: Vector Final Observation

- Given: process-based vector workers that terminate or truncate and auto-reset.
- When: the transition boundary is normalized.
- Then: reset observation is returned for the next episode, final observation
  remains available for the previous transition, and missing final observation
  at timeout raises a fatal error.
- Related requirements: `REQ-RLB-014`, `REQ-RLB-015`.

### AC-RLB-019: ACL Profile Validation

- Given: all combinations of ACL enabled/disabled and `n_envs` in `{1,4}`.
- When: configuration is loaded.
- Then: ACL ON is accepted only with one environment; ACL OFF accepts the
  approved one- and four-environment profiles; ACL ON plus vectorization fails.
- Related requirements: `REQ-RLB-015`, `REQ-RLB-016`.

### AC-RLB-020: ACL Learning Potential Conformance

- Given: fixed PPO, TD3, and SAC episode/update fixtures with analytically
  computed authoritative learning-potential values.
- When: the baseline emits ACL usefulness input.
- Then: each value matches ACL v1 §12/§28; changing Rulebook margins,
  criticality, `C_safe`, or source labels without changing learner quantities
  does not change learning potential.
- Related requirements: `REQ-RLB-016`, `REQ-RLB-021`.

### AC-RLB-021: Deterministic Evaluation Isolation

- Given: a frozen checkpoint and validation scenarios.
- When: primary evaluation is repeated with fixed seeds.
- Then: actions use deterministic mode, no training/replay/ACL state changes,
  and model parameters and checkpoint digests remain unchanged.
- Related requirements: `REQ-RLB-017`.

### AC-RLB-022: Hyperparameter And Extension Inheritance

- Given: resolved scalar and future extension manifests.
- When: their parent/shared fields are compared.
- Then: all non-extension-specific fields match; any approved deviation has a
  declared reason and required paired parent control.
- Related requirements: `REQ-RLB-018`.

### AC-RLB-023: Atomic Checkpoint Round Trip

- Given: each algorithm and approved runtime profile after at least one update.
- When: an atomic generation is saved, reloaded on CPU and CUDA-compatible
  device, and evaluated on fixed observations.
- Then: manifest/artifact digests validate, outputs match within configured
  tolerance, and incomplete or mismatched generations are rejected.
- Related requirements: `REQ-RLB-019`.

### AC-RLB-024: Resume Classification

- Given: TD3/SAC checkpoints with and without compatible replay, and a PPO
  checkpoint.
- When: resume is requested.
- Then: compatible full TD3/SAC state is labeled `stateful_continuation`; absent
  replay is labeled `model_only_restart` with replay/beta reset; PPO restores
  supported state without partial rollout; no path claims bitwise equivalence.
- Related requirements: `REQ-RLB-020`.

### AC-RLB-025: Anti-Leakage

- Given: paired scenarios differing only in future tracks, future signals,
  dataset source, ACL arm/usefulness, Rulebook diagnostics, or scalarization
  diagnostics.
- When: current policy input and action are computed under fixed policy RNG.
- Then: observation and action are unchanged whenever all authorized causal
  inputs are unchanged.
- Related requirements: `REQ-RLB-021`.

### AC-RLB-026: Numerical And Resource Smoke

- Given: adequate available GPU memory and the final resolved preset for each
  algorithm.
- When: reset, short collection, at least one learner update, save, load, and
  inference are executed.
- Then: no NaN/Inf/OOM occurs, intended parameters change, resource peaks are
  logged, actions are non-degenerate, and checkpoint/reward/replay contracts
  pass. Reward performance is not used to select hyperparameters.
- Related requirements: `REQ-RLB-022`, all algorithm requirements.

### AC-RLB-027: Invalid Configuration Matrix

- Given: individual violations of every frozen requirement in §9.8.
- When: each configuration is loaded.
- Then: it fails before training with an informative error naming the invalid
  field and expected contract; no fallback configuration is constructed.
- Related requirements: all configuration requirements.

### AC-RLB-028: Dataset And Split Isolation

- Given: train, validation, and test loaders and a checkpoint selection run.
- When: training and evaluation execute.
- Then: gradient/replay/ACL updates consume only train records; validation can
  select checkpoints without updating learner state; test records are not read
  until final evaluation; fingerprint mismatch fails.
- Related requirements: `REQ-RLB-018`, `REQ-RLB-019`, §8.7.

---

## 13. Required Validation Categories

| Category | Requirement |
|---|---|
| nominal and boundary behavior | `REQUIRED` |
| invalid and incomplete inputs | `REQUIRED` |
| masks, padding, state, reset, and update order | `REQUIRED`; observation/encoder details imported from `OBS-V1.1` and `ENC-V1.0` |
| termination and truncation | `REQUIRED` |
| deterministic seeds and reproducibility | `REQUIRED` |
| numerical stability, NaN, and infinity | `REQUIRED` |
| compatibility and migration | `REQUIRED` |
| absence of future and privileged information | `REQUIRED` |
| upstream, downstream, and end-to-end integration | `REQUIRED` |
| regressions for known bugs | `REQUIRED` for every discovered defect |
| action scaling and stochasticity | `REQUIRED` |
| actor/critic/target ownership and gradient routing | `REQUIRED` |
| PPO rollout/GAE/update schedule | `REQUIRED` |
| TD3/SAC N-step and PER integration | `REQUIRED` |
| ACL single-environment integration | `REQUIRED` |
| ACL vectorization | `Not applicable` to v1; separate deferred feature |
| trained-policy performance threshold | `Not applicable` to implementation acceptance; belongs to experimental protocol |

Exact test modules, fixtures, commands, and repository paths belong to the
ExecPlan created by Codex after approval.

---

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-RLB-001` | `AC-RLB-001` | user approval 2026-07-19; `ENC-V1.0` backend contract |
| `REQ-RLB-002` | `AC-RLB-002` | verified repository environment interface; MetaDrive action contract |
| `REQ-RLB-003` | `AC-RLB-003` | `OBS-V1.1`, `ENC-V1.0`, ADR-002, ADR-004 |
| `REQ-RLB-004` | `AC-RLB-004` | `SCAL-V1.0`, ADR-011, Rulebook v4.7 |
| `REQ-RLB-005` | `AC-RLB-005` | TD3; `ENC-V1.0`; pinned SB3 behavior |
| `REQ-RLB-006` | `AC-RLB-006`, `007`, `026` | TD3 literature; SB3/RL practice; approved TD3-HYPER-v1 decision 2026-07-19 |
| `REQ-RLB-007` | `AC-RLB-008` | SAC; `ENC-V1.0`; pinned SB3 behavior |
| `REQ-RLB-008` | `AC-RLB-009`, `010`, `026` | SAC automatic entropy behavior; verified fork defaults |
| `REQ-RLB-009` | `AC-RLB-011` | `ENC-V1.0`; PPO shared extractor decision |
| `REQ-RLB-010` | `AC-RLB-012`, `013`, `026` | PPO and GAE; approved VECTOR-BASELINES-v1 decision 2026-07-19 |
| `REQ-RLB-011` | `AC-RLB-015`, `016` | `TRANSITION-REPLAY v1.0`; user-approved core PER/N-step selection |
| `REQ-RLB-012` | `AC-RLB-014` | `TRANSITION-REPLAY v1.0`; PPO on-policy contract |
| `REQ-RLB-013` | `AC-RLB-017` | ScenarioNet integration v1.1; explicit user/Codex reconciliation 2026-07-19 |
| `REQ-RLB-014` | `AC-RLB-013`, `015`, `018` | ScenarioNet v1.1; transition boundary and SB3 timeout behavior |
| `REQ-RLB-015` | `AC-RLB-006`, `010`, `012`, `018`, `019` | approved VECTOR-BASELINES-v1 decision 2026-07-19 |
| `REQ-RLB-016` | `AC-RLB-019`, `020` | ACL v1 §12/§28 and ADR-014; explicit no-mutation decision |
| `REQ-RLB-017` | `AC-RLB-007`, `009`, `021` | algorithm-native train/eval behavior; user approval 2026-07-19 |
| `REQ-RLB-018` | `AC-RLB-022`, `028` | approved fairness and inheritance decisions 2026-07-19 |
| `REQ-RLB-019` | `AC-RLB-023` | `ENC-V1.0`, `SCAL-V1.0`, approved checkpoint decision |
| `REQ-RLB-020` | `AC-RLB-024` | `TRANSITION-REPLAY v1.0`; approved replay-persistence decision |
| `REQ-RLB-021` | `AC-RLB-020`, `025` | `OBS-V1.1`, `SCAL-V1.0`, ScenarioNet v1.1, ACL v1 |
| `REQ-RLB-022` | `AC-RLB-026`, `027` | approved resource/no-silent-change decision 2026-07-19 |

---

## 15. Open Decisions And Limitations

### 15.1 Open material decisions

No scientific or functional decision remains open in this review candidate.
A new material issue found by Codex shall return to explicit user approval and
shall not be resolved silently in the implementation or ExecPlan.

### 15.2 Repository-dependent review gates

These are facts to verify or implementation gaps to reconcile; they are not
new scientific decisions:

| ID | Gate | Required outcome before approval/implementation |
|---|---|---|
| `RG-001` | exact `ADR-014` path and content | verify authority and register exact path |
| `RG-002` | local SB3 revision | confirmed `4e6c3db1367a4cda96308bc6d0b80e63cd698828`; update stale index |
| `RG-003` | fork deviation ledger | documented in `docs/implementation/sb3_fork_deviation_ledger.md` |
| `RG-004` | resolved presets | create/verify exact PPO, TD3, SAC compositions for ScenarioEnv + semantic v2 + LQ + scalarization + replay + ACL/profile |
| `RG-005` | SAC target-update interval | verify effective pinned-fork value and freeze it in config/manifest |
| `RG-006` | PPO unlisted effective defaults | verify value clipping and optimizer details against pinned fork |
| `RG-007` | manifest training-loop path | route normal scientific save/load through atomic generation APIs |
| `RG-008` | ACL learning potential | replace current generic/loss proxy with exact PPO/TD3/SAC formulas from ACL authority |
| `RG-009` | GPU learner smoke | execute S0, S1/S2, replay/PER, ACL, and all-on smoke when memory is available |
| `RG-010` | final spec/index reconciliation | review diff, rename after approval, register authority and implementation status |

### 15.3 Intentional limitations

1. ACL is single-environment in v1; vectorized ACL is a separate future feature.
2. Hyperparameters are frozen, resource-aware baseline choices, not claims of
   global optimality.
3. The scalar baselines optimize a nonlinear scalar transition reward; they do
   not provide direct lexicographic expected-return guarantees.
4. Current-state post-perception inputs do not establish robustness to
   occlusions or perception errors.
5. Replay persistence is disabled by default, so TD3/SAC model-only restart is
   not continuation-equivalent.
6. Exact seeds, budgets, model-selection rule, and final evaluation count require
   a separate authoritative experimental protocol.
7. Bitwise-identical GPU and subprocess reproducibility is not guaranteed.
8. Resource failures do not authorize silent scientific configuration changes.

---

## 16. References

### 16.1 Authoritative project documents

1. `docs/project_index.md`: document authority and current gaps.
2. `docs/engineering_workflow.md`: specification lifecycle and approval gates.
3. `docs/templates/specification_template.md`: required specification structure.
4. `docs/specifications/rulebook_v4.7_specification.md`: Rulebook transition
   metrics and priority order.
5. `docs/specifications/rulebook_scalarization_v1.0_specification.md`: scalar
   reward, diagnostics boundary, algorithm-independent semantics, and checkpoint
   compatibility.
6. `docs/specifications/observation_v1.1_specification.md`: policy-visible
   semantic state, shape, timing, and anti-leakage.
7. `docs/specifications/encoder_v1.0_specification.md`: LQ architecture,
   ownership, gradient routing, and checkpoint fingerprinting.
8. `docs/specifications/transition_replay_v1_specification.md`: N-step, PER,
   timeout, persistence, and invalid-combination semantics.
9. `docs/specifications/scenarionet_integration_v1.1_specification.md`: dataset,
   `ThesisScenarioEnv`, horizon, termination, and truncation.
10. `docs/specifications/automatic_curriculum_learning_v1_specification.md`,
    including §28: ACL MAB/replay and algorithm-specific learning potential.
11. ADR-001, ADR-002, ADR-003, ADR-004, ADR-011, and ADR-014.

### 16.2 Scientific and dependency references

1. Schulman, J. et al. *Proximal Policy Optimization Algorithms*, 2017.
   Supports the PPO clipped surrogate and multi-epoch on-policy optimization.
2. Schulman, J. et al. *High-Dimensional Continuous Control Using Generalized
   Advantage Estimation*, 2016. Supports GAE.
3. Fujimoto, S., van Hoof, H., and Meger, D. *Addressing Function Approximation
   Error in Actor-Critic Methods*, 2018. Supports TD3 twin critics, delayed
   policy updates, and target-policy smoothing.
4. Haarnoja, T. et al. *Soft Actor-Critic Algorithms and Applications*, 2018.
   Supports entropy-regularized continuous-control actor-critic and automatic
   entropy tuning.
5. Schaul, T. et al. *Prioritized Experience Replay*, 2016. Supports
   proportional priorities and importance-sampling correction.
6. Barth-Maron, G. et al. *Distributed Distributional Deterministic Policy
   Gradients*, 2018. Supports N-step and PER use in off-policy continuous
   control and motivates later parent distributional extensions.
7. Raffin, A. et al. *Stable-Baselines3: Reliable Reinforcement Learning
   Implementations*, 2021. Supports the selected implementation framework.
8. Li, Q. et al. *MetaDrive: Composing Diverse Driving Scenarios for
   Generalizable Reinforcement Learning*, 2022. Supports the continuous-control
   driving environment and scenario diversity motivation.
9. Li, Q. et al. *ScenarioNet: Open-Source Platform for Large-Scale Traffic
   Scenario Simulation and Modeling*, 2023. Supports unified ScenarioDescription
   datasets and interactive ScenarioEnv execution.

### 16.3 Historical source limitation

The supplied historical LaTeX report remains useful for literature motivation
and terminology. Its earlier proposals of uniform one-step core replay,
PER-only ablations, rank-based TD3 PER, and PPO GAE-lambda ablations are not
selected by this specification and are not authoritative project decisions.

---

## 17. Implementation Handoff Checklist

Before setting `Status: APPROVED`, confirm:

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, shapes, units, ranges, and visibility.
- [x] Prohibited future, privileged, leaked, and diagnostic-only data is listed.
- [x] PPO, TD3, SAC formulas and update schedules are explicit.
- [x] State, timing, reset, termination, and truncation behavior is defined.
- [x] Configuration fields and scientifically frozen defaults are identified.
- [x] Errors, diagnostics, reproducibility, compatibility, and migration are covered.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected or marked `Not applicable`.
- [x] Scientific sources, dependency behavior, and project adaptations are distinct.
- [x] No material scientific decision remains open.
- [x] Known limitations are explicit.
- [x] Codex has reviewed the complete document against the current repository.
- [x] Exact ADR-014 path and current SB3 effective defaults are verified.
- [x] Resolved core presets are verified or implementation gaps are reported.
- [x] The user has explicitly approved the complete reviewed specification.
- [x] The canonical filename and repository path have been applied.
- [x] `project_index.md` has been updated with authority and implementation state.

---

## 18. Approval Record

- Approved by: `user`
- Approval date: `2026-07-20`
- Approval evidence:
  - explicit user approval on `2026-07-19` of the material baseline decisions,
    including fork-backed-only baselines, ScenarioNet termination adoption,
    vector profiles, TD3 batch 256, PPO global rollout 2,048, hyperparameter
    freeze, deterministic evaluation, checkpoint semantics, and deferred ACL
    vectorization;
  - explicit approval on `2026-07-20` of the clarified golden-suite boundary,
    ACL-on single-environment default, duration-profile parameter matrix,
    replay persistence OFF, and the remaining reviewed baseline contract.
- Approval notes: `The golden suite is retained only as an optional deterministic
  Rulebook/adapter diagnostic and is excluded from scientific policy claims.`
- Repository path: `docs/specifications/rl_baselines_v1_specification.md`
- Project index updated: `YES`
