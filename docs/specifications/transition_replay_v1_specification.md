# Specification: Transition-Level Replay and Multi-Step Targets

## Metadata

- Feature: Transition-level replay, multi-step returns, and optional prioritized experience replay
- Specification ID: `TRANSITION-REPLAY`
- Version: `1.0`
- Status: `APPROVED`
- Date: `2026-07-17`
- Supersedes: `NONE`
- Related specifications:
  - `docs/specifications/rulebook_v4.6_specification.md`
  - `docs/specifications/observation_v1.1_specification.md`
  - `docs/specifications/encoder_v1.0_specification.md`
  - `docs/specifications/automatic_curriculum_learning_v1_specification.md`
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`
  - `docs/specifications/rulebook_scalarization_v1.0_specification.md`
  - future scalarized RL baseline specification
  - future lexicographic RL specification
  - future distributional RL specification
- Related ADRs: `docs/decisions/ADR-011-rulebook-scalarization-v1.md`
- Authoritative: `YES`
- Recommended review filename: `transition_replay_v1_specification_UNDER_REVIEW.md`
- Repository path: `docs/specifications/transition_replay_v1_specification.md`

> This document is a review candidate. It becomes authoritative only after explicit
> user approval, metadata transition to `APPROVED`, removal of the
> `_UNDER_REVIEW` suffix, placement at the actual repository path, and
> registration in `project_index.md`.

# Executive Summary

This specification defines transition-level replay behavior for the
Stable-Baselines3-backed TD3 and SAC agents used in the thesis project.

The selected core behavior is:

```text
TD3:
    uniform replay
    3-step return

SAC:
    uniform replay
    3-step return

PPO:
    no off-policy replay
    existing rollout buffer and GAE behavior
```

Prioritized Experience Replay (PER) is implemented as an optional extension for
scalar TD3 and scalar SAC, but it is disabled in the initial core
configuration:

```text
n-step returns:
    core, enabled with n_steps = 3

proportional PER:
    optional, implemented, disabled by default

replay-buffer persistence:
    optional, disabled by default
```

The N-step implementation shall preserve the semantics of the pinned
Stable-Baselines3 fork:

- rewards are accumulated over an effective horizon \(m \le n\);
- the bootstrap discount is \(\gamma^m\);
- a true termination disables bootstrapping;
- a time-limit truncation ends the reward sequence but retains bootstrapping
  from the valid final observation;
- the scalar reward is accumulated after per-step scalarization;
- TD3 target-policy smoothing and clipped double Q-learning remain unchanged;
- SAC adds the entropy term only in the final bootstrap term, matching the SB3
  training target.

The custom PER implementation shall use:

\[
p_i =
\frac{
\left|\delta_{i,1}\right|
+
\left|\delta_{i,2}\right|
}{2}
+
\varepsilon_{\mathrm{PER}},
\]

\[
P(i)
=
\frac{p_i^{\alpha_{\mathrm{PER}}}}
{\sum_j p_j^{\alpha_{\mathrm{PER}}}},
\]

with proportional stratified sampling, batch-normalized importance-sampling
weights, and priority updates derived from the exact target used by the critic.

The configuration defaults are:

```yaml
transition_replay:
  enabled: true
  n_steps: 3
  prioritized: false
  optimize_memory_usage: false
  store_reward_vector: false

  persistence:
    enabled: false
    trigger: final_or_manual
    keep_last: 1

  per:
    alpha: 0.6
    beta_initial: 0.4
    beta_final: 1.0
    beta_anneal_steps: ${experiment.total_timesteps}
    epsilon: 1.0e-6
    priority_aggregation: mean_abs_twin_td
    new_transition_priority: current_max
    duplicate_update_reduction: max
    sampling: proportional_stratified
```

The exact files affected by implementation, the local fork capabilities, and
the concrete test commands are repository-dependent facts to be verified by
Codex in the implementation ExecPlan. They are not inferred by this
specification.

---

## 1. Purpose And Context

### 1.1 Purpose

This specification defines how off-policy transitions are stored, sampled, and
converted into temporal-difference targets for the scalar TD3 and SAC agents in
the project.

The externally meaningful capabilities are:

1. use a configurable one-step or three-step critic target;
2. preserve the current SB3 TD3 and SAC actor/critic semantics;
3. optionally prioritize replay transitions according to twin-critic TD error;
4. keep scenario-level ACL separate from transition-level replay;
5. support vectorized environments without sharing one priority across
   different environment transitions;
6. provide reproducible diagnostics and optional replay persistence;
7. leave an explicit extension point for future lexicographic and
   distributional critics without defining their priority semantics in this
   version.

### 1.2 Research role

Multi-step targets improve temporal credit propagation by incorporating several
observed rewards before bootstrapping. They are established in off-policy
continuous-control literature, including D4PG.

Prioritized Experience Replay was introduced to sample transitions according to
a measure of current learning relevance rather than uniformly. It is used in
D4PG-style continuous-control systems, but its benefit is less consistently
dominant than the benefit of multi-step returns. For that reason:

- N-step returns are part of the project core;
- PER is an optional replay ablation and is disabled by default.

### 1.3 Upstream and downstream components

Upstream producers:

```text
ScenarioEnv / wrappers
→ scalar per-step reward
→ terminated
→ truncated
→ final observation
→ transition metadata
```

This specification consumes the scalar reward already produced by the
environment/reward adapter. It does not define the scalarization formula.

Downstream consumers:

```text
TD3 critic target and loss
SAC critic target and loss
TD3/SAC actor update batch
ACL learning-potential computation
replay diagnostics
future distributional/lexicographic implementations
```

### 1.4 Scientific sources versus project adaptations

#### Directly supported by literature or official dependency behavior

- proportional prioritized replay;
- TD-error-based priorities;
- importance-sampling correction;
- annealing \(\beta\) toward \(1\);
- insertion of unseen transitions with maximum priority;
- N-step returns for off-policy continuous control;
- separation between PPO rollout collection and off-policy replay;
- SB3 use of `NStepReplayBuffer` when `n_steps > 1`;
- SB3 use of the per-sample bootstrap discount returned by the replay buffer;
- SB3 treatment of time-limit truncation separately from true termination.

#### Project adaptations and original decisions

- \(n=3\) as the selected project default;
- proportional PER as the only PER variant in v1;
- mean absolute TD error across the two critics;
- batch-wise normalization of importance-sampling weights;
- `max` reduction for duplicate sampled-index priority updates;
- one prioritized batch shared by critic and actor, with IS weights applied only
  to critic losses;
- PER disabled by default;
- replay persistence disabled by default;
- no priority clipping in v1;
- optional reward-vector storage disabled in the scalar v1;
- buffer-owned replay RNG and persisted RNG state;
- future lexicographic and distributional priority semantics explicitly deferred.

### 1.5 Dependency constraint

The project uses a pinned local fork of Stable-Baselines3. The repository
currently records a specific SB3 submodule commit, but this specification does
not assert that the local commit already contains every upstream behavior
described here.

Before implementation planning, Codex shall verify:

1. whether the pinned fork exposes `n_steps` in TD3 and SAC;
2. whether it contains `NStepReplayBuffer`;
3. whether replay samples expose a per-sample `discounts` field;
4. how timeouts and final observations are represented;
5. whether the local training loops match the target equations in this document;
6. whether a custom replay class is already supported by configuration;
7. whether any project-specific fork changes conflict with this contract.

A repository mismatch is an implementation finding. It does not silently modify
the scientific contract.

---

## 2. Scope

### 2.1 In Scope

- scalar TD3 transition replay;
- scalar SAC transition replay;
- uniform one-step replay mode;
- uniform three-step replay mode;
- proportional PER with one-step or three-step targets;
- per-transition priorities for vectorized environments;
- twin-critic TD-error priority aggregation;
- importance-sampling correction of critic losses;
- PER \(\beta\) annealing;
- termination and truncation semantics;
- ring-buffer frontier and wrap-around behavior;
- deterministic replay sampling under a fixed seed and state;
- optional replay-buffer persistence;
- replay-buffer compatibility validation;
- replay diagnostics;
- test and acceptance requirements;
- an optional storage hook for future reward vectors, disabled in v1.

### 2.2 Out Of Scope

- PPO replay prioritization;
- changing PPO GAE or rollout length;
- scalar reward and rulebook scalarization formulas;
- algorithm-wide TD3, SAC, or PPO hyperparameter selection;
- replay capacity selection for final experiments;
- actor-aware PER;
- LA3P-style separate actor and critic replay distributions;
- rank-based PER;
- dynamic or multi-criterion PER;
- reward-based replay priorities;
- uncertainty-based replay priorities;
- priority clipping or priority aging;
- offline RL;
- demonstration replay or behavior cloning;
- hindsight experience replay;
- recurrent replay;
- sequence-model replay beyond N-step targets;
- exact priority semantics for lexicographic critics;
- exact priority semantics for distributional critics;
- risk-sensitive distributional sampling;
- automatic buffer compression;
- replay migration across incompatible observation schemas.

### 2.3 Optional Or Deferred

- enabling PER in a controlled ablation;
- enabling replay persistence for selected long runs;
- storing a four-dimensional rulebook reward vector;
- distributional-loss-based priorities;
- lexicographic objective-aware priorities;
- critic-only prioritized batches with uniform actor batches;
- rank-based or actor-aware replay.

These features shall not be activated merely because extension points exist.

---

## 3. Terminology, Assumptions, And Preconditions

### 3.1 Symbols

| Symbol | Meaning |
|---|---|
| \(s_t\) | observation at environment step \(t\) |
| \(a_t\) | action applied at step \(t\) |
| \(r_t\) | scalar reward already produced for step \(t\) |
| \(\mathbf r_t\) | optional ordered reward vector, not used by scalar v1 |
| \(s_{t+1}\) | valid next observation associated with the transition |
| \(\gamma\) | discount factor provided by the algorithm configuration |
| \(n\) | configured maximum N-step horizon |
| \(m_t\) | effective accumulated horizon, \(1 \le m_t \le n\) |
| \(z_t\) | true-termination indicator |
| \(u_t\) | truncation/time-limit indicator |
| \(b_t\) | bootstrap mask after the effective sequence |
| \(Q_j\) | online critic \(j\), \(j\in\{1,2\}\) |
| \(Q'_j\) | target critic \(j\) |
| \(\delta_{i,j}\) | TD error for sampled transition \(i\), critic \(j\) |
| \(p_i\) | raw PER priority |
| \(P(i)\) | PER sampling probability |
| \(\alpha_{\mathrm{PER}}\) | prioritization exponent |
| \(\beta_{\mathrm{PER}}\) | importance-sampling correction exponent |
| \(w_i\) | normalized importance-sampling weight |
| \(B\) | gradient minibatch size |
| \(N_{\mathrm{active}}\) | number of currently active replay transitions |
| `beta_progress_env_steps` | persisted environment-timestep progress for the current replay-training segment |

### 3.2 Time convention

One replay transition corresponds to one environment control step.

The control timestep is defined by the environment specification and is not
modified by this feature. `n_steps = 3` therefore covers three environment
transitions, not three seconds.

### 3.3 Assumptions and classifications

| Assumption | Classification | Validation | Failure behavior |
|---|---|---|---|
| TD3 and SAC use continuous `Box` actions | guaranteed by selected algorithms | runtime space check | fail initialization |
| Core observations are flat `Box` observations | guaranteed by observation v1.1 | runtime shape/space check | fail initialization |
| The environment provides separate termination and truncation semantics | runtime-validated | wrapper/integration tests | fail final training eligibility |
| A valid final observation is available for bootstrappable truncations | runtime-validated | transition collection test | fail transition insertion or mark run invalid; no silent zero observation |
| Scalar reward is finite | runtime-validated | insertion check | fail fast |
| Scalarization has already occurred per step | configuration/integration precondition | reward-adapter reconciliation | fail integration approval |
| Twin critics exist in TD3 and SAC | guaranteed by selected backends | repository inspection and smoke test | block implementation |
| The local SB3 fork supports custom replay classes | repository-dependent | Codex inspection | implement supported adapter or raise approval gate |
| Exact replay memory capacity is provided elsewhere | configuration-provided | config validation | fail initialization if absent or invalid |
| Full bitwise continuation of the entire simulator is not guaranteed | explicit research limitation | documented in manifest | no claim of bitwise continuation |

### 3.6 Replay-training segment

`beta_progress_env_steps` is the trainer-owned environment-timestep progress of
the current replay-training segment. It is not inferred implicitly from the
model's global timestep counter.

The state lifecycle is:

```text
new run or new replay segment:                 0
collect k environment timesteps:              increment by k
model-only restart without replay:            reset to 0
compatible model + replay load:               restore serialized value
```

The value is included in the trainer checkpoint/manifest. The default
`beta_anneal_steps` resolves to `${experiment.total_timesteps}`.

### 3.4 Replay address

A transition in a vectorized replay buffer is identified by the pair:

```text
(storage_index, env_index)
```

Any flat index used by PER shall be a reversible encoding of this pair. The
normative logical mapping is:

```python
flat_index = storage_index * n_envs + env_index
```

An equivalent internal mapping is permitted only if round-trip behavior and
per-environment independence are preserved.

### 3.5 Effective horizon

The configured N-step horizon is:

\[
n \in \{1,3\}.
\]

The effective horizon \(m_t\) is the number of rewards included before the first
of:

1. reaching \(n\) transitions;
2. encountering true termination;
3. encountering truncation;
4. reaching the current valid replay frontier under the SB3-compatible frontier
   rule.

The boundary transition reward is included.

---

## 4. Inputs And Prohibited Information

### 4.1 Inputs

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `observation` | flat policy observation | `[D]`, `float32` | current step | observation v1.1 | insertion fails | `YES`, through normal policy input |
| `action` | applied continuous action | `[A]`, normalized SB3 action space | current step | algorithm collector | insertion fails | generated by policy |
| `scalar_reward` | scalarized per-step reward | scalar `float32` | step \(t\) | reward adapter | insertion fails | `NO`, training signal |
| `next_observation` | valid next observation | `[D]`, `float32` | step \(t+1\) | collector/final observation | insertion fails | used by target networks |
| `terminated` | true MDP termination | boolean | step boundary | environment/wrapper | insertion fails | `NO` |
| `truncated` | time-limit or external truncation | boolean | step boundary | environment/wrapper | insertion fails | `NO` |
| `final_observation` | valid pre-reset observation for a completed transition | `[D]`, `float32` | only when `terminated` or `truncated` is true | collector-normalized environment info | required for bootstrappable truncation; insertion fails if missing/invalid | `NO` |
| `gamma` | discount factor | scalar | \(0 \le \gamma \le 1\) | algorithm config | initialization fails | `NO` |
| `n_steps` | maximum return horizon | integer | `{1, 3}` | replay config | initialization fails | `NO` |
| `run_seed` | experiment seed | integer | implementation-defined valid integer | experiment manifest | initialization fails for final runs | `NO` |
| `beta` | current PER correction exponent | scalar | `[0,1]` | trainer schedule | sampling fails | `NO` |
| `reward_vector` | optional ordered rulebook vector | `[4]`, `float32` | step \(t\) | future reward adapter | ignored when storage disabled; fail if enabled and missing | `NO` |
| `VecNormalize` state | optional normalization adapter | SB3 object | current training state | algorithm | use unnormalized path if absent | `NO` |

### 4.2 Prohibited information

The replay system shall not consume or derive priority from:

- future ground-truth actor trajectories;
- future SDC trajectory;
- future traffic-light states;
- episode outcome labels unavailable at transition time;
- scenario test-set membership as a priority;
- ACL arm identifier as a transition priority;
- scenario usefulness score as a transition priority;
- rulebook episode-level evaluation summaries;
- final held-out evaluation metrics;
- non-causal future collision labels;
- human-selected “interesting transition” labels;
- actor routes or intentions unavailable online;
- distributional or lexicographic diagnostic values before those algorithms are
  separately specified.

### 4.3 Information classification

| Value | Classification |
|---|---|
| observation/action | policy interaction data |
| scalar reward | training signal |
| reward vector when enabled | future training signal and diagnostics |
| TD errors | learner-internal training diagnostics |
| priority and sampling probability | learner-internal replay state |
| IS weights | critic optimization signal |
| effective N-step count | diagnostic-only |
| buffer size and persistence metadata | reproducibility metadata |
| sampled flat indices | learner-internal replay address |
| replay checkpoint | optional reproducibility artifact |

### 4.4 Canonical transition collection contract

The project collector shall preserve these fields separately before adapting to
SB3-specific storage:

```text
observation
action
scalar_reward
next_observation
terminated
truncated
final_observation, when the environment auto-resets
```

For vectorized environments, the collector shall normalize any source field
named `terminal_observation` or `final_observation` into the canonical
`final_observation` field before insertion. A batch containing only combined
`dones` is not a sufficient project-level collection interface.

The target observation semantics are:

| Case | Observation stored for target | Bootstrap |
|---|---|---:|
| ordinary transition | normal `next_observation` | yes |
| `terminated=true` | terminal pre-reset observation | no |
| `truncated=true` | final pre-reset observation | yes |
| both flags true | final pre-reset observation | no; termination prevails |

An internal adapter may convert the canonical fields to SB3 `dones` and
`timeouts`, provided the observable semantics remain unchanged.

---

## 5. Outputs

### 5.1 Uniform sample output

Uniform `ReplayBuffer` and `NStepReplayBuffer` sampling shall provide the fields
required by the pinned SB3 TD3/SAC training loops:

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `observations` | start observations | `[B,D]` | sampled order | actor/critic | finite |
| `actions` | stored actions | `[B,A]` | sampled order | critic | finite |
| `next_observations` | final observations after effective horizon | `[B,D]` | sampled order | target actor/critic | finite |
| `dones` | non-bootstrap mask source | `[B,1]`, `{0,1}` | sampled order | target equation | `1` only for true termination |
| `rewards` | accumulated discounted scalar return | `[B,1]` | sampled order | target equation | finite |
| `discounts` | bootstrap discount | `[B,1]` or scalar for one-step | sampled order | target equation | equals \(\gamma^{m_i}\) |
| `effective_n_steps` | effective horizon | `[B,1]`, integer | sampled order | diagnostics | \(1\le m_i\le n\) |

If the unmodified one-step SB3 replay sample does not expose
`effective_n_steps`, the adapter/logger may use the constant value `1`.

### 5.2 Prioritized sample output

The custom sample type shall be semantically equivalent to:

```python
class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    importance_weights: torch.Tensor
    flat_indices: torch.Tensor
    effective_n_steps: torch.Tensor
```

Normative shapes:

```text
observations:         [B, D]
actions:              [B, A]
next_observations:    [B, D]
dones:                [B, 1]
rewards:              [B, 1]
discounts:            [B, 1]
importance_weights:   [B, 1]
flat_indices:         [B]
effective_n_steps:    [B, 1]
```

The concrete Python type may be a `NamedTuple`, immutable dataclass, or a
compatible extension of an SB3 sample type. The field names and observable
semantics above are normative.

### 5.3 Priority update input

The replay buffer shall expose behavior equivalent to:

```python
def update_priorities(
    flat_indices: np.ndarray | torch.Tensor,
    raw_priorities: np.ndarray | torch.Tensor,
) -> None:
    ...
```

Requirements:

- input length equals sampled batch length before duplicate reduction;
- indices refer to currently active transition slots;
- raw priorities are finite and strictly positive;
- duplicates are reduced with `max`;
- tree leaves store \(p_i^{\alpha_{\mathrm{PER}}}\), not raw \(p_i\);
- `max_raw_priority` is updated from the raw values.

---

## 6. Functional Requirements

### REQ-001: Algorithm Applicability

- Required observable behavior:
  - TD3 and SAC support transition replay modes defined here.
  - PPO does not instantiate this replay subsystem.
  - PPO may contain `transition_replay` only when it is explicitly inactive with
    `transition_replay.enabled=false`.
- Applicability:
  - scalar SB3-backed algorithms in the current project.
- Invariants:
  - PPO `n_steps` continues to mean rollout length.
  - PPO temporal credit assignment remains GAE-based.
- Edge cases:
  - `transition_replay.enabled=true` for PPO is invalid;
  - `prioritized=true`, replay persistence, a custom replay class, reward-vector
    replay storage, or off-policy N-step settings are invalid for PPO.
- Failure behavior:
  - configuration validation fails before learner construction; inactive PPO
    configuration containing only schema defaults is accepted and does not emit
    an ignored-warning path. Non-default or active replay settings fail.
- Interactions:
  - future PPO changes require a separate specification.

### REQ-002: Supported N-Step Modes

- Required observable behavior:
  - TD3 and SAC accept `n_steps=1` and `n_steps=3`.
  - the project core default is `3`.
- Applicability:
  - uniform and prioritized replay.
- Invariants:
  - no other N-step value is silently coerced.
- Failure behavior:
  - unsupported values fail initialization.

### REQ-003: Uniform N-Step SB3 Compatibility

- Required observable behavior:
  - when PER is disabled and `n_steps=3`, the implementation uses the pinned
    SB3 `NStepReplayBuffer` behavior or a provably equivalent adapter.
- Invariants:
  - target equations consume returned per-sample discounts;
  - `optimize_memory_usage` is false;
  - N-step sampling does not allocate a second N-step copy of the replay data.
- Failure behavior:
  - missing upstream capability blocks implementation until explicitly adapted.

### REQ-004: Per-Step Scalarization Before Accumulation

- Required observable behavior:
  - the replay buffer stores the scalar reward emitted for each individual
    environment transition.
  - the N-step return accumulates those stored scalar rewards.
- Invariant:

\[
G_t^{(m)}
=
\sum_{k=0}^{m-1}\gamma^k
r_{t+k}^{\mathrm{scalar}}.
\]

- Prohibited behavior:
  - accumulating the reward vector and applying a nonlinear scalarization only
    after N-step aggregation.
- Failure behavior:
  - integration reconciliation fails if reward ordering is ambiguous.

### REQ-005: True-Termination Semantics

- Required observable behavior:
  - the canonical collection contract preserves `terminated` separately from
    `truncated`;
  - the reward at the true terminal transition is included;
  - the effective sequence ends at that transition;
  - `dones=1` for the sample;
  - a terminal pre-reset observation may be stored, but it is never used for a
    value bootstrap;
  - no value bootstrap is added.
- Invariant:
  - terminal transitions never read rewards from the next episode.

### REQ-006: Truncation Semantics

- Required observable behavior:
  - the canonical collection contract preserves `truncated` separately from
    `terminated` and provides a normalized `final_observation` when required;
  - the reward at a truncation transition is included;
  - the effective sequence ends at that transition;
  - `dones=0` for the critic bootstrap mask;
  - the valid final observation is used for the bootstrap;
  - the discount is \(\gamma^m\).
- Applicability:
  - time limits and external horizon truncations that preserve a valid final
    observation.
- Failure behavior:
  - missing or invalid final observation is fatal for final training eligibility;
    it shall not be replaced silently by zeros or a reset observation.
- Edge case:
  - when both flags are true, the final pre-reset observation is retained but
    true termination prevails and bootstrapping is disabled.

### REQ-007: Replay-Frontier Semantics

- Required observable behavior:
  - sampling shall not cross from the newest written transition into unwritten or
    stale ring-buffer content.
  - if the selected start is too close to the current frontier for a full
    N-step sequence, the effective horizon is shortened using the
    SB3-compatible frontier rule.
  - bootstrapping remains enabled from the last valid next observation.
- Invariants:
  - no reward from an unrelated old transition is included.
- Limitation:
  - the same recently inserted transition may produce a shorter effective
    horizon before additional transitions are collected.

### REQ-008: TD3 N-Step Target

- Required observable behavior:
  - TD3 uses the target defined in Section 7.3.
- Invariants:
  - target-policy smoothing is unchanged;
  - target noise clipping is unchanged;
  - action clipping is unchanged;
  - the minimum across target critics is unchanged.
- Failure behavior:
  - any divergence from the baseline TD3 target requires an approved revision.

### REQ-009: SAC N-Step Target

- Required observable behavior:
  - SAC uses the target defined in Section 7.4.
- Invariants:
  - accumulated intermediate terms are environment scalar rewards only;
  - the entropy term is applied in the final bootstrap term;
  - the action for the bootstrap state is sampled from the current actor;
  - the minimum across target critics is unchanged.
- Classification:
  - this is the selected SB3-compatible SAC N-step convention.

### REQ-010: Optional Proportional PER

- Required observable behavior:
  - when `prioritized=false`, replay sampling is uniform.
  - when `prioritized=true`, sampling follows proportional PER.
- Invariants:
  - PER is supported only by scalar TD3 and scalar SAC in v1;
  - PER does not modify environment collection or scenario selection;
  - the default remains disabled.
- Failure behavior:
  - invalid algorithm/PER combinations fail initialization.

### REQ-011: Per-Transition Priority With Vectorized Environments

- Required observable behavior:
  - each `(storage_index, env_index)` transition has an independent priority.
- Invariants:
  - one environment's TD error never overwrites another environment's priority
    merely because both transitions share the same storage time index.
- Failure behavior:
  - ambiguous time-only priority indexing is non-conformant.

### REQ-012: Priority Formula

- Required observable behavior:

\[
p_i
=
\frac{
|\delta_{i,1}|+|\delta_{i,2}|
}{2}
+
\varepsilon_{\mathrm{PER}}.
\]

- Invariants:
  - both critics use the same exact critic target;
  - errors are computed before the critic optimizer step;
  - priorities are detached from autograd;
  - priorities are finite and strictly positive.
- Failure behavior:
  - non-finite priority causes fail-fast.

### REQ-013: New Transition Priority

- Required observable behavior:
  - new transitions receive the current maximum raw priority;
  - the initial maximum is `1.0`.
- Invariant:

\[
p_{\mathrm{new}}
=
\max(1, p_{\max,\mathrm{current}}).
\]

- Rationale:
  - new transitions can be sampled before a TD error has been computed.

### REQ-014: Proportional Stratified Sampling

- Required observable behavior:
  - leaves store \(p_i^{\alpha_{\mathrm{PER}}}\);
  - batch sampling divides total priority mass into equal segments and draws one
    uniform mass per segment.
- Invariants:
  - only active transition leaves contribute;
  - `alpha=0` yields uniform sampling over active transition slots;
  - duplicate indices are permitted.

### REQ-015: Importance-Sampling Weights

- Required observable behavior:

\[
\widetilde w_i
=
\left(
N_{\mathrm{active}}P(i)
\right)^{-\beta},
\qquad
w_i
=
\frac{\widetilde w_i}
{\max_{j\in B}\widetilde w_j}.
\]

- Invariants:
  - \(0 < w_i \le 1\);
  - weights are finite;
  - weights have shape `[B,1]`;
  - batch-wise normalization is used in v1.
- Classification:
  - batch-wise maximum normalization is a project adaptation.

### REQ-016: Weighted Critic Loss Only

- Required observable behavior:
  - IS weights multiply per-sample critic losses.
  - actor losses are not multiplied by IS weights.
  - SAC entropy-coefficient loss is not multiplied by IS weights.
  - target-network updates are unchanged.
- Invariants:
  - reduction to a scalar occurs after per-sample weighting.
- Limitation:
  - the actor still receives states from the prioritized batch.

### REQ-017: Priority Update Order

- Required observable order:

```text
sample transition batch
→ construct exact algorithm target
→ compute per-critic pre-update TD errors
→ construct weighted critic loss
→ perform critic optimizer step
→ update sampled priorities using detached pre-update errors
→ perform actor update when scheduled
→ perform target update when scheduled
```

- Equivalent ordering is permitted for actor/target updates if the baseline
  algorithm requires it, provided priorities use the pre-critic-update errors.
- Prohibited behavior:
  - recomputing priority with a simplified target;
  - using actor loss as priority;
  - using post-update Q values without an approved revision.

### REQ-018: Duplicate Priority Reduction

- Required observable behavior:
  - if a flat index occurs multiple times in one batch, its stored raw priority is
    the maximum of the proposed raw priorities for that index.
- Invariant:
  - update result is independent of duplicate processing order.

### REQ-019: Sum-Tree State

- Required observable behavior:
  - PER uses one active priority leaf per addressable transition slot.
  - inactive/unwritten leaves have zero sampling mass.
  - tree aggregates use `float64`.
- Invariants:
  - total mass equals the sum of active powered priorities within numerical
    tolerance;
  - no min tree is required in v1.
- Failure behavior:
  - non-positive or non-finite total mass fails sampling.

### REQ-020: Buffer-Owned Replay RNG

- Required observable behavior:
  - PER sampling uses a replay-buffer-owned NumPy generator.
  - it is initialized deterministically from the run seed.
  - its state is included when replay persistence is enabled.
- Invariants:
  - equal buffer content, equal priorities, equal generator state, and equal
    sampling request produce equal sampled indices.
- Limitation:
  - this does not guarantee bitwise equality of the complete environment/training
    run.

### REQ-021: Beta Schedule

- Required observable behavior:

\[
\beta(t)
=
\beta_0
+
(\beta_{\mathrm{final}}-\beta_0)
\min\left(
\frac{t}{T_{\beta}},
1
\right).
\]

- Inputs:
  - \(t\): persisted `beta_progress_env_steps` for the current replay-training
    segment;
  - \(T_\beta\): resolved `beta_anneal_steps`.
- Invariants:
  - schedule belongs to trainer state, not priority-tree state;
  - `beta_progress_env_steps` starts at zero for a new run or replay segment;
  - it increments by collected environment timesteps;
  - a compatible replay load restores the serialized value;
  - a model-only restart resets it to zero;
  - \(\beta\) is clamped to `[beta_initial, beta_final]`;
  - default final value is `1.0`.
- Failure behavior:
  - PER cannot start without a positive resolved anneal horizon.

### REQ-022: Optional Reward-Vector Storage

- Required observable behavior:
  - scalar v1 defaults to `store_reward_vector=false`.
  - when false, no reward-vector array is allocated.
  - when true in a future approved integration, the configured vector shall be
    stored per transition without changing the scalar target.
- Limitation:
  - v1 does not define how lexicographic or distributional learners consume the
    vector.
- Failure behavior:
  - enabling storage without an approved producer contract is invalid.

### REQ-023: Replay Persistence Disabled By Default

- Required observable behavior:

```text
model checkpoint:
    saved according to checkpoint policy

replay-buffer checkpoint:
    not saved unless persistence.enabled = true
```

- Invariants:
  - no replay `.pkl` artifact is produced by default;
  - disabling replay persistence does not disable model checkpoints;
  - periodic checkpoint callbacks shall not produce replay artifacts in v1.

### REQ-024: Optional Replay Persistence

- Required observable behavior:
  - when enabled, replay persistence is triggered only at final/manual
    checkpoints in the initial implementation;
  - `transition_replay.persistence.enabled` is the canonical control;
  - the legacy `checkpoint.save_replay_buffer=false` is tolerated as inactive;
  - the legacy `checkpoint.save_replay_buffer=true` fails with an actionable
    migration error and does not silently change meaning;
  - only the latest replay checkpoint is retained (`keep_last=1`);
  - periodic replay snapshots are not supported in v1.
- Serialized state includes:
  - transition arrays;
  - timeout and done arrays;
  - position/full state;
  - N-step configuration;
  - PER priority tree;
  - raw maximum priority;
  - replay RNG state;
  - compatibility metadata.
- Model and replay artifacts saved together shall share a verifiable
  `checkpoint_id`, `training_timestep`, `replay_segment_id`, and manifest
  record.
- Replay replacement shall be atomic: write and validate a temporary artifact,
  commit it atomically, and only then retire the previous artifact.
- Failure behavior:
  - serialization errors are fatal to a requested replay-save operation but do
    not silently delete a previously valid checkpoint.

### REQ-025: Resume Classification

- Required observable behavior:
  - loading a model without the corresponding replay state starts a new empty
    replay segment.
  - the event is logged as `replay_reset=true`.
  - it shall not be reported as replay-equivalent continuation.
  - `beta_progress_env_steps` resets to `0` and PER \(\beta\) restarts from
    `beta_initial` for the new empty replay segment.
- When replay state is loaded:
  - replay sampling state can continue from the serialized buffer state;
  - `beta_progress_env_steps` is restored from the compatible trainer/replay
    checkpoint pair;
  - complete bitwise run continuation is not claimed unless all external RNG and
    environment states are also restored and verified.

### REQ-026: Load Compatibility Validation

- Required observable behavior:
  - replay loading validates the compatibility fields defined in Section 11.4.
- Failure behavior:
  - incompatible replay artifacts fail before training resumes;
  - no implicit reshaping, truncation, or priority reconstruction is permitted.

### REQ-027: Fail-Fast Numerical Behavior

- Required observable behavior:
  - targets, Q predictions, TD errors, priorities, probability masses, and IS
    weights are checked for finiteness in debug/validation paths and before any
    invalid value can corrupt the tree.
- Prohibited fallbacks:
  - replacing NaN/Inf with epsilon;
  - silently clipping non-finite priority;
  - silently resetting the tree.
- Failure behavior:
  - raise an informative numerical error and mark the run failed.

### REQ-028: ACL Separation

- Required observable behavior:
  - ACL selects scenarios;
  - replay selects transitions;
  - neither module reuses the other's priority value.
- Initial experimental behavior:
  - PER remains disabled in the primary ACL comparison.
- Future behavior:
  - simultaneous ACL and PER activation requires explicit experiment
    configuration and reporting.

### REQ-029: Logging

- Required observable behavior:
  - the diagnostics in Section 10.3 are emitted at the configured logging
    cadence when applicable.
- Invariants:
  - logging does not modify sampling behavior;
  - diagnostics are not policy inputs.

### REQ-030: Future Algorithm Boundary

- Required observable behavior:
  - this priority formula is normative only for scalar TD3 and scalar SAC.
- Prohibited inference:
  - distributional or lexicographic implementations shall not reuse the scalar
    priority formula without a separately approved specification.

### REQ-031: Canonical Transition Collection

- Required observable behavior:
  - scalar and vectorized collectors preserve `terminated`, `truncated`, and
    `final_observation` separately before SB3 adaptation;
  - auto-reset vector environments use the pre-reset final observation rather
    than the reset observation;
  - one batch may contain independent termination and truncation outcomes for
    different environments.
- Failure behavior:
  - combined `dones` without a validated timeout/final-observation mapping is
    rejected at the project integration boundary.

### REQ-032: Reward Semantic Compatibility

- Required observable behavior:
  - replay artifacts and manifests record the reward and scalarization identity
    that produced stored scalar rewards;
  - replay loading validates that identity before training resumes.
- Required compatibility fields:

```text
rulebook_specification_id
rulebook_version
rulebook_margin_schema_id
scalarization_specification_id
scalarization_version
scalarization_mode
scalarization_config_digest
native_environment_reward_weight
```

- Legacy-mode fields, when applicable:

```text
legacy_vector_schema_id
legacy_rule_scales_digest
legacy_scale_source_digest
```

- Failure behavior:
  - any mismatch prevents replay loading; no replay re-scaling or reward
    reinterpretation is performed implicitly.
- Compatibility limitation:
  - model weights may be used for transfer initialization in a new run, but a
    replay with semantically different scalar rewards requires a separately
    approved re-scalarization procedure based on stored reward vectors.

### REQ-033: Checkpoint Pairing And Legacy Migration

- Required observable behavior:
  - a compatible model/replay continuation requires a matching checkpoint ID,
    training timestep, replay segment ID, and manifest;
  - a model-only checkpoint is classified as a new replay segment;
  - legacy `checkpoint.save_replay_buffer=true` fails before training with a
    migration message to `transition_replay.persistence.enabled`.
- Failure behavior:
  - mismatched or partially committed model/replay pairs fail before resume;
  - a failed replay save preserves the previous valid artifact.

---

## 7. Mathematical And Algorithmic Contract

### 7.1 Stored transition

The logical one-step transition is:

\[
\tau_t
=
\left(
s_t,
a_t,
r_t,
s_{t+1},
z_t,
u_t
\right),
\]

where \(z_t\) and \(u_t\) remain conceptually distinct even if the pinned SB3
storage internally derives timeout masks from `info`.

It is invalid for both values to imply incompatible final-observation
semantics. The environment integration shall define precedence if Gymnasium
returns both flags true; the project expectation is that a true terminal event
is treated as non-bootstrappable.

### 7.2 Effective N-step return

Let \(m_t\) be the effective sequence length. The stored scalar return exposed to
the learner is:

\[
G_t^{(m_t)}
=
\sum_{k=0}^{m_t-1}
\gamma^k r_{t+k}.
\]

The returned bootstrap discount is:

\[
d_t^{\mathrm{boot}}
=
\gamma^{m_t}.
\]

The final bootstrap mask is:

\[
b_t
=
\begin{cases}
0,
&
\text{if the final included transition is a true termination},
\\[4pt]
1,
&
\text{otherwise, including a bootstrappable truncation or replay frontier}.
\end{cases}
\]

The generic target form is:

\[
y_t^{(m_t)}
=
G_t^{(m_t)}
+
b_t
\gamma^{m_t}
V_{\mathrm{target}}(s_{t+m_t}).
\]

### 7.3 TD3 target

For TD3:

\[
\widetilde a_{t+m}
=
\operatorname{clip}_{\mathcal A}
\left(
\pi'(s_{t+m})
+
\operatorname{clip}
\left(
\epsilon,
-c,
c
\right)
\right),
\]

with the baseline TD3 target smoothing distribution and clip \(c\).

The target is:

\[
y_t^{\mathrm{TD3},(m)}
=
G_t^{(m)}
+
b_t\gamma^m
\min_{j\in\{1,2\}}
Q'_j
\left(
s_{t+m},
\widetilde a_{t+m}
\right).
\]

The current critic errors are:

\[
\delta_{t,j}
=
y_t^{\mathrm{TD3},(m)}
-
Q_j(s_t,a_t).
\]

N-step replay shall not change policy delay, Polyak averaging, exploration
noise, target smoothing, or actor objective.

### 7.4 SAC target

Sample:

\[
a_{t+m}
\sim
\pi(\cdot\mid s_{t+m}).
\]

Define:

\[
V_{\mathrm{soft,target}}(s_{t+m})
=
\min_{j\in\{1,2\}}
Q'_j(s_{t+m},a_{t+m})
-
\alpha_{\mathrm{ent}}
\log\pi(a_{t+m}\mid s_{t+m}).
\]

The selected SB3-compatible target is:

\[
y_t^{\mathrm{SAC},(m)}
=
G_t^{(m)}
+
b_t\gamma^m
V_{\mathrm{soft,target}}(s_{t+m}).
\]

The intermediate accumulated terms are the scalar environment rewards only.
Entropy is not retroactively added for stored behavior-policy actions at each
intermediate step.

The current critic errors are:

\[
\delta_{t,j}
=
y_t^{\mathrm{SAC},(m)}
-
Q_j(s_t,a_t).
\]

N-step replay shall not change automatic entropy-coefficient learning, target
entropy, actor objective, or Polyak updates.

### 7.5 PER raw priority

For a sampled transition \(i\):

\[
e_i
=
\frac{
|\delta_{i,1}|
+
|\delta_{i,2}|
}{2},
\]

\[
p_i
=
e_i
+
\varepsilon_{\mathrm{PER}}.
\]

The tree mass is:

\[
q_i
=
p_i^{\alpha_{\mathrm{PER}}}.
\]

No priority cap is applied in v1.

### 7.6 Sampling probability

\[
P(i)
=
\frac{q_i}{\sum_{j\in\mathcal A}q_j},
\]

where \(\mathcal A\) is the set of active transition slots.

For \(\alpha_{\mathrm{PER}}=0\):

\[
q_i = 1
\]

for all active leaves, and sampling becomes uniform.

### 7.7 Stratified batch sampling

Let:

\[
S
=
\sum_{j\in\mathcal A}q_j.
\]

For batch item \(k\in\{0,\dots,B-1\}\), draw:

\[
x_k
\sim
\mathcal U
\left[
\frac{kS}{B},
\frac{(k+1)S}{B}
\right).
\]

The sum tree returns the active leaf whose cumulative interval contains \(x_k\).

### 7.8 Importance-sampling correction

\[
\widetilde w_i
=
\left(
N_{\mathrm{active}}P(i)
\right)^{-\beta}.
\]

Project v1 uses batch-wise normalization:

\[
w_i
=
\frac{
\widetilde w_i
}{
\max_{j\in B}\widetilde w_j
}.
\]

### 7.9 Critic loss

Let \(\ell\) be the per-element critic loss used by the selected SB3 baseline,
currently squared error unless the baseline specification explicitly changes
it.

The weighted twin-critic loss is:

\[
L_Q
=
\frac{1}{B}
\sum_{i=1}^{B}
w_i
\left[
c_{\mathrm{alg}}
\sum_{j=1}^{2}
\ell(\delta_{i,j})
\right],
\]

where \(c_{\mathrm{alg}}\) preserves the baseline algorithm's existing twin-loss
scaling:

- TD3 retains its baseline sum convention;
- SAC retains its baseline \(0.5\) twin-loss convention.

The implementation shall use `reduction="none"` or an equivalent computation so
that weighting occurs before batch reduction.

### 7.10 Beta schedule

The resolved annealing horizon is \(T_\beta>0\).

In this section, \(t\) is exactly the persisted
`beta_progress_env_steps` value for the current replay-training segment.

\[
\beta(t)
=
\beta_0
+
(\beta_f-\beta_0)
\min
\left(
\frac{t}{T_\beta},
1
\right).
\]

Default:

\[
\beta_0=0.4,
\qquad
\beta_f=1.0.
\]

### 7.11 Duplicate update reduction

For a sampled flat index \(i\) appearing multiple times:

\[
p_i^{\mathrm{updated}}
=
\max_{k:\,i_k=i}
p_{i_k}.
\]

### 7.12 Ring overwrite

When a ring slot is overwritten:

1. the old transition data become inaccessible;
2. the old tree leaf is replaced;
3. the new raw priority is `max(1.0, max_raw_priority)`;
4. any optional reward-vector metadata is overwritten atomically;
5. ancestor sums are updated before the slot is sampleable.

### 7.13 Reference pseudocode

```python
def sample_and_update(batch_size: int) -> None:
    if prioritized:
        beta = beta_schedule(num_timesteps)
        replay_data = replay_buffer.sample(
            batch_size=batch_size,
            env=vec_normalize_env,
            beta=beta,
        )
    else:
        replay_data = replay_buffer.sample(
            batch_size=batch_size,
            env=vec_normalize_env,
        )

    target = build_exact_algorithm_target(replay_data)
    q1, q2 = critic(replay_data.observations, replay_data.actions)

    td1 = target - q1
    td2 = target - q2

    per_sample_loss = algorithm_twin_loss_without_batch_reduction(
        td1,
        td2,
    )

    if prioritized:
        critic_loss = (
            per_sample_loss
            * replay_data.importance_weights
        ).mean()
    else:
        critic_loss = per_sample_loss.mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if prioritized:
        raw_priorities = (
            0.5 * (td1.detach().abs() + td2.detach().abs())
            + per_epsilon
        )
        replay_buffer.update_priorities(
            replay_data.flat_indices,
            raw_priorities,
        )

    run_algorithm_specific_actor_and_target_updates(replay_data)
```

This pseudocode defines semantics, not exact repository file placement.

---

## 8. Applicability, State, And Timing

### 8.1 State ownership

The replay buffer owns:

- transition arrays;
- done and timeout arrays;
- ring position and full flag;
- optional reward-vector array;
- PER sum tree when enabled;
- current maximum raw priority;
- replay sampling RNG;
- replay compatibility metadata.

The trainer owns:

- current environment timestep count;
- resolved \(\beta\) annealing horizon;
- `beta_progress_env_steps` for the current replay-training segment;
- current \(\beta\);
- algorithm networks and optimizers;
- checkpoint policy;
- run manifest.

### 8.2 Initialization

At initialization:

```text
pos = 0
full = false
active_count = 0
max_raw_priority = 1.0
beta_progress_env_steps = 0
PER inactive unless configured
replay RNG initialized from run seed
```

Unwritten priority leaves have zero mass.

### 8.3 Update order during collection

For each vectorized `env.step()` result:

1. preserve separate `terminated` and `truncated` flags;
2. normalize any auto-reset `terminal_observation`/`final_observation` into the
   canonical `final_observation` field;
3. validate observation, action, reward, flags, and final-observation rules;
4. write each environment transition independently;
5. assign new priority if PER is enabled;
6. advance ring position using the base replay-buffer convention;
7. update active transition count.

### 8.4 Episode reset

Replay state is not cleared on environment episode reset.

N-step sampling shall stop at each individual environment's termination or
truncation boundary and shall never cross into the next episode.

### 8.5 Vectorized environment behavior

Each environment contributes a separate transition at a shared collection
iteration. Priority, termination, truncation, effective N-step count, and final
observation are per environment.

### 8.6 Training start

The baseline algorithm's `learning_starts` behavior remains authoritative.
This specification does not change when gradient updates begin.

When PER is enabled, transitions collected before learning starts receive
maximum insertion priority.

### 8.7 Timing and schedules

PER \(\beta\) is computed once per gradient sampling operation or once per
gradient-step loop iteration, consistently for the full batch.

After collecting \(k\) environment timesteps, the trainer increments
`beta_progress_env_steps` by \(k\). The implementation shall not use episode
count or an implicitly restored model counter as the \(\beta\) progress axis.

### 8.8 Buffer frontier

The implementation shall preserve the pinned SB3 protection against sampling
across the current ring-buffer write frontier.

The exact local implementation may temporarily mark the final valid position as
a truncation or use an equivalent immutable mask. Observable semantics shall
match REQ-007.

### 8.9 Serialization timing

Default:

```text
periodic model checkpoint:
    model only

final model checkpoint:
    model only

replay checkpoint:
    disabled
```

When enabled:

```text
final/manual checkpoint:
    model + replay state

periodic replay checkpoint:
    unsupported in v1
```

The model artifact, replay artifact, and manifest record form one continuation
pair and share `checkpoint_id`, `training_timestep`, and `replay_segment_id`.
Replay serialization uses a temporary artifact, validates it, atomically
commits it, and only then replaces the previous artifact.

### 8.10 Determinism limits

The replay subsystem shall be deterministic for:

- identical stored data;
- identical active priorities;
- identical replay RNG state;
- identical sample request and configuration.

This does not guarantee deterministic CUDA execution, environment stepping,
asynchronous worker scheduling, SAC action sampling, or complete training
continuation.

---

## 9. Configuration

### 9.1 Consolidated configuration

```yaml
transition_replay:
  enabled: true
  n_steps: 3
  prioritized: false
  optimize_memory_usage: false
  store_reward_vector: false
  reward_vector_dim: null

  persistence:
    enabled: false
    trigger: final_or_manual
    periodic_frequency_steps: null
    keep_last: 1

  per:
    alpha: 0.6
    beta_initial: 0.4
    beta_final: 1.0
    beta_anneal_steps: ${experiment.total_timesteps}
    epsilon: 1.0e-6
    priority_aggregation: mean_abs_twin_td
    new_transition_priority: current_max
    duplicate_update_reduction: max
    sampling: proportional_stratified
    tree_dtype: float64
```

### 9.2 Configuration table

| Field | Type | Default | Valid range | Meaning | Required | Frozen for core experiments |
|---|---|---:|---|---|---|---|
| `transition_replay.n_steps` | int | `3` | `{1,3}` | maximum return horizon for TD3/SAC | YES | YES |
| `transition_replay.enabled` | bool | `true` for TD3/SAC, `false` for PPO | boolean | activate this replay subsystem | YES | YES |
| `transition_replay.prioritized` | bool | `false` | boolean | enable proportional PER | YES | YES |
| `transition_replay.optimize_memory_usage` | bool | `false` | must be `false` | SB3 replay storage mode | YES | YES |
| `transition_replay.store_reward_vector` | bool | `false` | boolean | allocate optional vector reward storage | YES | YES |
| `transition_replay.reward_vector_dim` | int/null | `null` | `4` only when approved/enabled | optional vector dimension | conditional | YES |
| `transition_replay.persistence.enabled` | bool | `false` | boolean | save replay artifact | YES | YES |
| `transition_replay.persistence.trigger` | enum | `final_or_manual` | `final_or_manual` | replay-save trigger | conditional | YES |
| `transition_replay.persistence.periodic_frequency_steps` | null | `null` | must remain `null` in v1 | periodic replay save | NO | YES |
| `transition_replay.persistence.keep_last` | int | `1` | exactly `1` in v1 | retained replay artifacts | conditional | YES |
| `per.alpha` | float | `0.6` | `[0,1]` | priority exponent | conditional | YES |
| `per.beta_initial` | float | `0.4` | `[0,1]` | initial IS correction | conditional | YES |
| `per.beta_final` | float | `1.0` | `[beta_initial,1]` | final IS correction | conditional | YES |
| `per.beta_anneal_steps` | int | `${experiment.total_timesteps}` | `>0` | annealing horizon | conditional | YES |
| `per.epsilon` | float | `1e-6` | finite, `>0` | non-zero priority floor | conditional | YES |
| `per.priority_aggregation` | enum | `mean_abs_twin_td` | exact value | twin-critic aggregation | conditional | YES |
| `per.new_transition_priority` | enum | `current_max` | exact value | unseen transition priority | conditional | YES |
| `per.duplicate_update_reduction` | enum | `max` | exact value | duplicate-index update rule | conditional | YES |
| `per.sampling` | enum | `proportional_stratified` | exact value | batch sampling method | conditional | YES |
| `per.tree_dtype` | dtype | `float64` | exact value | sum-tree accumulation dtype | conditional | YES |
| `buffer_size` | int | external | `> batch_size`, memory-valid | replay capacity | YES, external | frozen by baseline protocol |
| `gamma` | float | external | `[0,1]` | discount factor | YES, external | frozen by baseline specification |
| `run_seed` | int | external | valid integer | replay RNG seed | YES for final runs | YES |

### 9.3 Algorithm mapping

```text
PPO:
    transition_replay.enabled=false only
    any active replay feature fails before learner construction
    no PER
    no off-policy N-step target

TD3:
    prioritized=false, n_steps=1 → uniform one-step replay
    prioritized=false, n_steps=3 → SB3-compatible uniform N-step replay
    prioritized=true,  n_steps=1 → custom proportional PER one-step
    prioritized=true,  n_steps=3 → custom proportional PER N-step

SAC:
    same replay mode matrix as TD3
```

### 9.4 Invalid configurations

The following fail before training:

- PPO with `prioritized=true`;
- PPO with `transition_replay.enabled=true`;
- PPO with replay persistence, custom replay class, reward-vector replay
  storage, or off-policy N-step settings enabled;
- PPO with `transition_replay.enabled=false` but any non-default replay setting
  that would alter replay behavior;
- PPO routed through `PrioritizedNStepReplayBuffer`;
- `n_steps` outside `{1,3}`;
- `optimize_memory_usage=true`;
- PER with missing \(\beta\) anneal horizon;
- `beta_final < beta_initial`;
- non-positive epsilon;
- unsupported priority aggregation;
- reward-vector storage enabled without approved producer contract;
- persistence `keep_last != 1`;
- periodic replay persistence configured in v1;
- unsupported observation space for the selected pinned-fork buffer;
- insufficient replay capacity;
- mismatched loaded replay metadata;
- `checkpoint.save_replay_buffer=true` without the canonical
  `transition_replay.persistence.enabled` migration;
- incompatible checkpoint pairing metadata;
- incompatible rulebook or scalarization metadata.

---

## 10. Errors, Diagnostics, And Resource Behavior

### 10.1 Fatal errors

Fatal conditions include:

- invalid shapes or dtypes;
- non-finite reward;
- non-finite target or Q prediction;
- non-finite TD error;
- non-finite or non-positive raw priority;
- non-finite tree total;
- non-finite or non-positive sampling probability;
- non-finite IS weight;
- invalid flat index;
- priority update for inactive/unwritten slot;
- missing final observation at bootstrappable truncation;
- incompatible replay artifact;
- requested persistence failure with no valid output artifact;
- unsupported local SB3 behavior.

### 10.2 No silent fallback

The implementation shall not silently:

- change `n_steps` from 3 to 1;
- disable PER after a tree error;
- reset all priorities;
- replace NaN/Inf values;
- reinterpret truncation as termination;
- use reset observation as final observation;
- omit IS weighting;
- save replay despite persistence being disabled;
- load a mismatched replay buffer.

### 10.3 Required diagnostics

Always when N-step is active:

```text
replay/current_size
replay/capacity
replay/beta_progress_env_steps
replay/effective_n_steps_mean
replay/early_cutoff_fraction
replay/termination_cutoff_fraction
replay/timeout_cutoff_fraction
replay/frontier_cutoff_fraction
```

When PER is active:

```text
replay/per_beta
replay/priority_mean
replay/priority_max
replay/tree_total
replay/sampling_probability_min
replay/sampling_probability_max
replay/is_weight_mean
replay/is_weight_min
replay/duplicate_fraction
```

When a model-only replay reset occurs:

```text
replay/replay_reset = true
replay/continuation_equivalent = false
```

### 10.4 Resource behavior

N-step sampling shall reuse the one-step transition storage and compute
accumulated returns at sample time.

PER adds priority-tree state of order \(O(N)\), which is expected to be small
relative to the two observation arrays for the current high-dimensional
observations.

Approximate observation-only replay memory, before actions/rewards/flags and
Python overhead, is:

\[
M_{\mathrm{obs}}
\approx
2 N D \cdot 4\ \mathrm{bytes},
\]

because `optimize_memory_usage=false` stores both observations and next
observations as `float32`.

For example, at \(N=10^6\):

- \(D=1540\): approximately \(12.32\) decimal GB;
- \(D=2541\): approximately \(20.33\) decimal GB.

These are diagnostic estimates, not a prescribed replay capacity.

Replay persistence may create a disk artifact comparable to the in-memory
transition arrays. Therefore it is disabled by default.

---

## 11. Reproducibility And Compatibility

### 11.1 Required manifest fields

Every TD3/SAC run shall record:

```text
transition replay specification ID/version
local SB3 commit
replay class
transition replay enabled
n_steps
prioritized
optimize_memory_usage
buffer_size
gamma
n_envs
observation schema ID/fingerprint
action shape
timeout handling
PER alpha/beta/epsilon when enabled
beta anneal horizon when enabled
priority aggregation
sampling method
replay persistence enabled/disabled
reward-vector storage enabled/disabled
beta_progress_env_steps
checkpoint_id
training_timestep
replay_segment_id
rulebook specification ID/version
rulebook margin schema ID
scalarization specification ID/version
scalarization mode
scalarization configuration digest
native environment reward weight
legacy vector schema ID when applicable
legacy rule scales digest when applicable
legacy scale source digest when applicable
run seed
```

### 11.2 Replay class version

The custom buffer shall expose a stable version identifier, for example:

```python
replay_schema_id = "PRIORITIZED_N_STEP_REPLAY"
replay_schema_version = "1.0"
```

Renaming the concrete class does not permit bypassing compatibility checks.

### 11.3 Seed ownership

The experiment owns the canonical run seed.

The replay buffer shall construct its own NumPy generator from that seed, rather
than relying on unrelated global sampling state.

The exact seed-derivation helper is an implementation detail, but it must be
deterministic and recorded or reproducible.

### 11.4 Load compatibility fields

Replay load shall validate at least:

```text
replay schema ID/version
replay class/mode
observation shape and dtype
observation schema fingerprint when available
action shape and dtype
n_envs
allocated capacity
n_steps
gamma
timeout-handling convention
PER enabled flag
alpha
epsilon
priority aggregation
sampling method
tree dtype
reward-vector enabled flag and dimension
rulebook specification ID/version
rulebook margin schema ID
scalarization specification ID/version
scalarization mode
scalarization configuration digest
native environment reward weight
legacy vector schema ID when applicable
legacy rule scales digest when applicable
legacy scale source digest when applicable
checkpoint ID and replay segment ID
```

`beta` and `beta_progress_env_steps` are trainer schedule state and need not be
stored as tree invariants, but the resolved schedule configuration and progress
value shall be stored in the trainer checkpoint/manifest and validated by the
run-resume layer. A compatible model + replay pair restores the progress value.

Device may change if the pinned SB3 load path supports explicit device mapping.

### 11.5 Persistence artifact

When enabled, replay state is saved separately from the model checkpoint using
the SB3 replay-buffer persistence mechanism or a compatible wrapper. The model,
replay artifact, and manifest must form a matching continuation pair with the
same `checkpoint_id`, `training_timestep`, and `replay_segment_id`.

The replay artifact is written to a temporary path, completed and validated,
committed atomically, and only then used to replace the previous artifact. A
failed save shall leave the previous valid artifact untouched.

Default checkpoint callbacks shall pass:

```python
save_replay_buffer=False
```

### 11.6 Model-only restart

A model-only restart:

- creates an empty replay buffer;
- repeats the algorithm's configured replay warm-up behavior;
- resets `beta_progress_env_steps` to `0` and PER \(\beta\) to `beta_initial`;
- is logged as a new replay segment;
- is not treated as replay-state continuation.

### 11.7 Replay-state continuation

Loading the replay artifact restores:

- stored transitions;
- ring position and active size;
- priority tree;
- maximum raw priority;
- replay RNG state.

Loading a compatible model + replay checkpoint pair also restores
`beta_progress_env_steps` and the associated replay segment identity.

This is necessary for replay-state continuation but is insufficient by itself to
guarantee bitwise continuation of the complete training process.

### 11.8 Compatibility with prior experiments

Existing one-step uniform experiments remain valid as baseline results when
their exact configuration and pinned dependency are recorded.

They shall not be relabeled as three-step results.

Existing model checkpoints without replay state remain usable for inference.
Continuing training from them starts a new replay segment.

---

## 12. Acceptance Criteria

### AC-001: PPO Rejection

- Given: PPO configuration with `transition_replay.prioritized=true`.
- When: configuration is validated.
- Then: initialization fails with an algorithm-applicability error.
- Related requirements: `REQ-001`.

### AC-002: Supported N-Step Values

- Given: TD3 or SAC with `n_steps` equal to `1` or `3`.
- When: replay is initialized.
- Then: initialization succeeds with the expected replay mode.
- Given: any other value.
- Then: initialization fails.
- Related requirements: `REQ-002`.

### AC-003: One-Step Equivalence

- Given: deterministic replay contents, fixed networks, and `n_steps=1`.
- When: the custom N-step path is compared with the baseline one-step target.
- Then: rewards, next observations, dones, discounts, and target values agree
  within configured floating-point tolerance.
- Related requirements: `REQ-002`, `REQ-003`.

### AC-004: Three-Step Hand Calculation

- Given: rewards `[1,2,3]`, no boundary, and known \(\gamma\).
- When: the start transition is sampled with `n_steps=3`.
- Then:

\[
G=1+2\gamma+3\gamma^2,
\qquad
discount=\gamma^3,
\qquad
done=0.
\]

- Related requirements: `REQ-003`, `REQ-004`.

### AC-005: True Termination At Each Position

- Given: deterministic sequences with true termination at step 1, 2, or 3.
- When: the start transition is sampled.
- Then:
  - rewards include the terminal transition;
  - `effective_n_steps` equals the boundary position;
  - `done=1`;
  - no bootstrap term is added.
- Related requirements: `REQ-005`.

### AC-006: Truncation At Each Position

- Given: deterministic sequences with truncation at step 1, 2, or 3 and a valid
  final observation.
- When: the start transition is sampled.
- Then:
  - rewards include the truncation transition;
  - `effective_n_steps` equals the boundary position;
  - `done=0`;
  - discount equals \(\gamma^m\);
  - the final observation is used for bootstrap.
- Related requirements: `REQ-006`.

### AC-007: Missing Final Observation

- Given: a bootstrappable truncation without a valid final observation.
- When: transition collection or sampling occurs.
- Then: the operation fails with an informative error and no reset observation
  is substituted.
- Related requirements: `REQ-006`, `REQ-027`.

### AC-008: Episode Isolation

- Given: two consecutive episodes in the ring buffer.
- When: an N-step sample starts near the first episode boundary.
- Then: no reward or observation from the second episode is included.
- Related requirements: `REQ-005`, `REQ-006`.

### AC-009: Frontier Isolation

- Given: a partially filled buffer or current ring frontier.
- When: a sample starts within fewer than `n_steps` valid transitions of the
  frontier.
- Then: no unwritten/stale transition is read and the effective horizon is
  shortened according to the selected rule.
- Related requirements: `REQ-007`.

### AC-010: TD3 Target Preservation

- Given: fixed replay data and fixed target networks.
- When: the N-step TD3 target is computed.
- Then: it matches the equation in Section 7.3, including target smoothing,
  twin-min, effective discount, and bootstrap mask.
- Related requirements: `REQ-008`.

### AC-011: SAC Target Preservation

- Given: fixed replay data, controlled SAC action sample, fixed target networks,
  and fixed entropy coefficient.
- When: the N-step SAC target is computed.
- Then: it matches Section 7.4 and contains the entropy term only in the final
  bootstrap term.
- Related requirements: `REQ-009`.

### AC-012: Alpha-Zero Uniformity

- Given: active transitions with unequal raw priorities and `alpha=0`.
- When: a statistically sufficient number of samples is drawn.
- Then: empirical frequencies are consistent with uniform sampling within a
  predeclared statistical tolerance.
- Related requirements: `REQ-010`, `REQ-014`.

### AC-013: Proportional Frequency

- Given: a small replay buffer with known powered priorities.
- When: a statistically sufficient number of samples is drawn.
- Then: empirical frequencies are consistent with normalized priority mass
  within a predeclared statistical tolerance.
- Related requirements: `REQ-014`.

### AC-014: Vectorized Priority Independence

- Given: two environments sharing one storage index but assigned different raw
  priorities.
- When: priorities are updated and sampling probabilities inspected.
- Then: the two transitions retain distinct probabilities.
- Related requirements: `REQ-011`.

### AC-015: Twin-Critic Priority

- Given: known target and known Q1/Q2 predictions.
- When: priority is computed.
- Then: raw priority equals the mean absolute twin TD error plus epsilon.
- Related requirements: `REQ-012`.

### AC-016: New Transition Maximum

- Given: active priorities with known maximum.
- When: a new transition is inserted.
- Then: its raw priority equals `max(1.0, current_max_raw_priority)`.
- Related requirements: `REQ-013`.

### AC-017: Importance Weights

- Given: known probabilities, active count, and beta.
- When: IS weights are computed.
- Then:
  - they match Section 7.8;
  - they are finite;
  - their maximum in the batch is `1` within tolerance;
  - all weights are positive and at most `1`.
- Related requirements: `REQ-015`.

### AC-018: Critic-Only Weighting

- Given: a fixed prioritized batch.
- When: one TD3 or SAC update is executed.
- Then:
  - critic per-sample losses are weighted;
  - actor loss is unweighted;
  - SAC entropy-coefficient loss is unweighted;
  - target update is unchanged.
- Related requirements: `REQ-016`.

### AC-019: Priority Update Uses Exact Target

- Given: one sampled batch.
- When: critic target, loss, and priorities are computed.
- Then: the target tensor used for both critic loss and priority residuals is
  identical.
- Related requirements: `REQ-012`, `REQ-017`.

### AC-020: Pre-Update Priority Residual

- Given: a critic optimizer step that measurably changes predictions.
- When: priorities are updated.
- Then: stored priorities correspond to detached pre-update residuals.
- Related requirements: `REQ-017`.

### AC-021: Duplicate Max Reduction

- Given: the same flat index sampled multiple times with different proposed raw
  priorities.
- When: priorities are updated.
- Then: the stored raw priority is the maximum proposed value.
- Related requirements: `REQ-018`.

### AC-022: Tree Integrity

- Given: insertion, update, overwrite, and wrap-around operations.
- When: tree totals are checked after each operation.
- Then: root mass equals the sum of active leaf masses within tolerance and
  inactive leaves contribute zero.
- Related requirements: `REQ-019`.

### AC-023: Replay RNG Reproducibility

- Given: two equivalent buffers with identical RNG state.
- When: identical sample calls are made.
- Then: sampled flat indices are identical.
- Related requirements: `REQ-020`.

### AC-024: Beta Schedule Endpoints

- Given: configured \(\beta_0,\beta_f,T_\beta\).
- When: beta is evaluated at `t=0`, `t=T_beta/2`, and `t>=T_beta`.
- Then: it equals the linear schedule and clamps at \(\beta_f\).
- Related requirements: `REQ-021`.

### AC-025: Reward-Vector Disabled Memory

- Given: `store_reward_vector=false`.
- When: the buffer is initialized.
- Then: no reward-vector array is allocated and scalar replay behavior remains
  unchanged.
- Related requirements: `REQ-022`.

### AC-026: Persistence Disabled

- Given: default persistence configuration.
- When: periodic and final model checkpoint callbacks run.
- Then:
  - model artifacts are produced as configured;
  - no replay-buffer artifact is produced.
- Related requirements: `REQ-023`.

### AC-027: Persistence Round Trip

- Given: persistence enabled and a non-empty prioritized N-step buffer.
- When: the buffer is saved and loaded.
- Then:
  - transitions, ring state, tree, maximum priority, configuration metadata,
    `beta_progress_env_steps`, checkpoint-pairing metadata, and RNG state are
    preserved;
  - an identical next sample request produces identical flat indices.
- Related requirements: `REQ-024`, `REQ-020`.

### AC-028: Keep-Last Behavior

- Given: persistence enabled and two manual/final replay saves.
- When: the second save completes.
- Then: only one current replay artifact is retained and a previously valid
  artifact is not removed before the new artifact is safely completed.
- Related requirements: `REQ-024`.

### AC-029: Model-Only Replay Reset

- Given: a model checkpoint without replay state.
- When: training restarts.
- Then:
  - the replay buffer is empty;
  - replay warm-up behavior is applied;
  - `beta_progress_env_steps` is `0` and beta restarts at `beta_initial` if PER
    is enabled;
  - manifest/logging records non-equivalent replay continuation.
- Related requirements: `REQ-025`.

### AC-030: Incompatible Replay Rejected

- Given: a replay artifact with mismatched observation shape, N-step setting,
  PER mode, alpha, epsilon, vector reward schema, rulebook identity,
  scalarization identity, or checkpoint-pairing metadata.
- When: load is attempted.
- Then: loading fails before training.
- Related requirements: `REQ-026`.

### AC-031: Numerical Failure

- Given: injected NaN/Inf in reward, Q value, target, priority, probability, or
  IS weight.
- When: the affected path executes.
- Then: the run fails with an informative diagnostic and the invalid value is
  not inserted into the tree or optimizer.
- Related requirements: `REQ-027`.

### AC-032: ACL Separation

- Given: ACL scenario selection active and PER disabled.
- When: training runs.
- Then: transition sampling remains uniform and no ACL score is stored as a
  transition priority.
- Related requirements: `REQ-028`.

### AC-033: Required Diagnostics

- Given: N-step and optionally PER enabled.
- When: the logging cadence is reached.
- Then: all applicable fields from Section 10.3 are emitted with finite values.
- Related requirements: `REQ-029`.

### AC-034: Scalar-Only Priority Boundary

- Given: a future lexicographic or distributional learner requests v1 scalar
  priority semantics.
- When: configuration is validated.
- Then: activation fails until a compatible approved specification is provided.
- Related requirements: `REQ-030`.

### AC-035: TD3 Smoke Training

- Given: each core observation/encoder combination supported by the encoder
  specification and TD3 with uniform three-step replay.
- When: a short collect-update-save-load-infer smoke run is executed.
- Then: at least one critic and actor update completes without NaN/Inf or shape
  errors.
- Related requirements: `REQ-003`, `REQ-008`.

### AC-036: SAC Smoke Training

- Given: each core observation/encoder combination supported by the encoder
  specification and SAC with uniform three-step replay.
- When: a short collect-update-save-load-infer smoke run is executed.
- Then: critic, actor, and entropy update complete without NaN/Inf or shape
  errors.
- Related requirements: `REQ-003`, `REQ-009`.

### AC-037: PER Smoke Training

- Given: TD3 and SAC with proportional PER and three-step replay.
- When: a short training run crosses `learning_starts`.
- Then: sampling, weighted critic updates, priority updates, actor updates, and
  logging all execute without invalid state.
- Related requirements: `REQ-010` through `REQ-021`.

### AC-038: Beta Progress Lifecycle

- Given: a PER-enabled run with a resolved
  `beta_anneal_steps=${experiment.total_timesteps}`.
- When: a new segment starts, (k) environment timesteps are collected, a
  model-only restart occurs, or a compatible model + replay pair is loaded.
- Then:
  - progress is respectively `0`, incremented by (k), reset to `0`, or
    restored from the serialized value;
  - the computed beta uses that progress value rather than the model's implicit
    global timestep.
- Related requirements: `REQ-021`, `REQ-025`.

### AC-039: Canonical Termination And Final Observation Contract

- Given: scalar and vectorized collectors, including vector auto-reset, with
  ordinary transitions, termination, truncation, both flags true, and missing
  final observations.
- When: transitions are normalized before replay insertion.
- Then:
  - `terminated` and `truncated` remain separately observable;
  - pre-reset final observations replace reset observations when required;
  - termination disables bootstrap;
  - valid truncation preserves bootstrap;
  - both flags true preserves the final observation but disables bootstrap;
  - missing final observation for a bootstrappable truncation fails;
  - a batch may contain independent termination and truncation outcomes.
- Related requirements: `REQ-005`, `REQ-006`, `REQ-031`.

### AC-040: Reward Semantic Compatibility

- Given: two replay artifacts with identical tensor shapes and replay settings
  but different rulebook or scalarization metadata.
- When: replay loading is attempted.
- Then: loading fails before training resumes and no implicit re-scaling or
  reward reinterpretation occurs.
- Given: matching reward metadata, including applicable legacy digests.
- Then: compatibility validation succeeds.
- Related requirements: `REQ-004`, `REQ-026`, `REQ-032`.

### AC-041: Legacy Migration And Checkpoint Pairing

- Given: `checkpoint.save_replay_buffer=true` without the canonical
  `transition_replay.persistence.enabled` setting.
- When: configuration is validated.
- Then: initialization fails with an actionable migration message.
- Given: a final/manual model + replay save, including a replacement of an
  existing artifact.
- When: writing, validation, commit, or replacement fails.
- Then: the previous valid replay artifact remains available; a successful pair
  shares `checkpoint_id`, `training_timestep`, `replay_segment_id`, and manifest
  metadata.
- Related requirements: `REQ-024`, `REQ-033`.

---

## 13. Required Validation Categories

| Category | Requirement |
|---|---|
| nominal and boundary behavior | REQUIRED |
| invalid and incomplete inputs | REQUIRED |
| masks and padding | Not applicable to replay data itself; observation schema tests remain upstream |
| state, reset, and update order | REQUIRED |
| termination and truncation | REQUIRED |
| deterministic seeds and reproducibility | REQUIRED |
| numerical stability, NaN, and infinity | REQUIRED |
| compatibility and migration | REQUIRED |
| absence of future and privileged information | REQUIRED by integration review |
| upstream SB3 integration | REQUIRED |
| downstream TD3/SAC integration | REQUIRED |
| end-to-end smoke training | REQUIRED |
| replay persistence disabled behavior | REQUIRED |
| replay persistence round trip | REQUIRED even though feature is disabled by default |
| replay/model checkpoint pairing and atomic replacement | REQUIRED |
| reward and scalarization semantic compatibility | REQUIRED |
| legacy replay-persistence migration | REQUIRED |
| vectorized environments | REQUIRED |
| ring wrap-around and overwrite | REQUIRED |
| regressions for known bugs | REQUIRED as bugs are discovered |

Exact test files, fixtures, statistical sample counts, tolerances, and commands
belong in the implementation ExecPlan. Statistical tolerances shall be declared
before inspecting the resulting frequency test.

Protected acceptance behavior cannot be weakened to accommodate an
implementation deviation without explicit approval.

---

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-001` | `AC-001` | PPO/SB3 rollout architecture; project decision 2026-07-17 |
| `REQ-002` | `AC-002`, `AC-003`, `AC-004` | D4PG N-step use; project selection \(n=3\) |
| `REQ-003` | `AC-003`, `AC-035`, `AC-036` | official SB3 N-step behavior |
| `REQ-004` | `AC-004` | rulebook per-step learner signal; project scalarization ordering decision |
| `REQ-005` | `AC-005`, `AC-008` | episodic return semantics; SB3 done handling |
| `REQ-006` | `AC-006`, `AC-007` | Gymnasium/SB3 timeout handling |
| `REQ-007` | `AC-009` | official SB3 N-step frontier behavior |
| `REQ-008` | `AC-010`, `AC-035` | TD3 target; official SB3 target |
| `REQ-009` | `AC-011`, `AC-036` | SAC target; official SB3 target |
| `REQ-010` | `AC-012`, `AC-013`, `AC-037` | Schaul et al.; project default disabled |
| `REQ-011` | `AC-014` | project vectorized replay contract |
| `REQ-012` | `AC-015`, `AC-019`, `AC-020` | TD-error PER; project twin aggregation |
| `REQ-013` | `AC-016` | Schaul et al. maximum insertion priority |
| `REQ-014` | `AC-012`, `AC-013` | proportional PER; stratified sampling |
| `REQ-015` | `AC-017` | PER importance sampling; project batch normalization |
| `REQ-016` | `AC-018` | PER bias correction; project actor/critic boundary |
| `REQ-017` | `AC-019`, `AC-020` | project update-order decision |
| `REQ-018` | `AC-021` | project duplicate reduction decision |
| `REQ-019` | `AC-022` | proportional PER efficient sampling; project sum-tree contract |
| `REQ-020` | `AC-023`, `AC-027` | project reproducibility contract |
| `REQ-021` | `AC-024`, `AC-038` | Schaul et al. beta annealing; approved persisted segment progress |
| `REQ-022` | `AC-025` | future compatibility decision |
| `REQ-023` | `AC-026` | user decision 2026-07-17; official SB3 separate replay persistence |
| `REQ-024` | `AC-027`, `AC-028`, `AC-041` | approved final/manual storage, atomic replacement, and checkpoint pairing |
| `REQ-025` | `AC-029`, `AC-038` | project resume classification and beta progress lifecycle |
| `REQ-026` | `AC-030`, `AC-040`, `AC-041` | compatibility and migration requirements |
| `REQ-027` | `AC-031` | engineering workflow fail-fast requirement |
| `REQ-028` | `AC-032` | ACL specification separation; project experiment decision |
| `REQ-029` | `AC-033` | project diagnostics decision |
| `REQ-030` | `AC-034` | explicit future-scope boundary |
| `REQ-031` | `AC-039` | approved project collection contract; user review 2026-07-17 |
| `REQ-032` | `AC-040` | approved reward-semantic compatibility decision; user review 2026-07-17 |
| `REQ-033` | `AC-041` | approved persistence migration and checkpoint-pairing decision; user review 2026-07-17 |
| all core requirements | `AC-035`–`AC-037` | encoder/SB3 integration contract |
| replay lifecycle additions | `AC-038`–`AC-041` | user review resolutions 2026-07-17 |

---

## 15. Open Decisions And Limitations

### 15.1 Decision record

No material scientific or functional decision remains open for this review
candidate.

| ID | Question | Alternatives | Approved recommendation in this draft | Impact | Status |
|---|---|---|---|---|---|
| `DEC-001` | default N-step horizon | 1, 3, 5 | `3` | credit assignment and target variance | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-002` | PER default | enabled/disabled | disabled | stability and attribution | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-003` | PER variant | proportional/rank/actor-aware | proportional | implementation and sampling semantics | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-004` | twin priority aggregation | max/mean/RMS/Q1 | mean absolute | critic symmetry and outlier sensitivity | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-005` | actor batch | same prioritized/separate uniform | same prioritized, unweighted actor loss | implementation complexity | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-006` | replay persistence | default on/off | off | disk and memory use | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-007` | duplicate priority update | first/last/mean/max | max | deterministic duplicate handling | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-008` | SAC multi-step entropy | all intermediate/final bootstrap | final bootstrap, SB3-compatible | SAC target semantics | `APPROVED_IN_DESIGN_DISCUSSION` |
| `DEC-009` | beta progress state | implicit model timestep / explicit segment state | persisted `beta_progress_env_steps`, reset or restored by replay-segment lifecycle | beta reproducibility and resume semantics | `APPROVED_BY_USER_REVIEW_2026-07-17` |
| `DEC-010` | PPO replay configuration | reject / ignore / explicit inactive section | `transition_replay.enabled=false` only; active replay behavior fails before learner construction | fail-fast configuration and PPO compatibility | `APPROVED_BY_USER_REVIEW_2026-07-17` |
| `DEC-011` | transition collection contract | combined done / separate flags and final observation | canonical separate `terminated`, `truncated`, and normalized pre-reset `final_observation` | termination, truncation, and bootstrap correctness | `APPROVED_BY_USER_REVIEW_2026-07-17` |
| `DEC-012` | persistence migration and pairing | silent alias / legacy behavior / explicit migration | canonical nested setting, legacy true fails, atomic model-replay-manifest pairing | resume safety and compatibility | `APPROVED_BY_USER_REVIEW_2026-07-17` |
| `DEC-013` | reward semantic compatibility | shape-only / reward-aware metadata | validate rulebook, scalarization, and legacy scale identities before replay load | prevents semantically invalid replay reuse | `APPROVED_BY_USER_REVIEW_2026-07-17` |

The user explicitly approved `DEC-009` through `DEC-013` in the review response
received on 2026-07-17. The approval covers segment-owned beta progress,
explicitly inactive PPO configuration, canonical transition collection,
non-silent persistence migration and atomic pairing, and reward-semantic replay
compatibility. These decisions do not yet approve the complete specification.

The material decisions above are resolved for this revision based on the user's
review response. The document itself remains `UNDER_REVIEW` until the user
explicitly approves the complete updated specification.

### 15.2 Intentional limitations

1. PER is not active in the initial core configuration.
2. PPO is not modified.
3. The scalar priority is not normative for future lexicographic or
   distributional critics.
4. The same prioritized batch is used by actor and critic.
5. Importance weights use batch-wise rather than global-max normalization.
6. No min tree is maintained.
7. No priority cap, aging, uncertainty correction, or actor-aware sampling is
   implemented.
8. Replay persistence retains only one final/manual artifact when enabled.
9. N-step replay cannot use SB3 memory-efficient storage in this version.
10. Replay-state restoration does not guarantee bitwise restoration of the
    complete simulation and training process.
11. Exact replay capacity is defined by the future baseline/protocol
    configuration because it depends materially on available RAM and selected
    observation dimension.
12. The local SB3 fork must be inspected before implementation planning.

### 15.3 Information to verify in the repository

These are not open scientific decisions:

- local fork file paths and exact class signatures;
- pinned fork support for N-step;
- current timeout/final-observation adapter behavior;
- existing custom replay abstractions;
- current configuration schema;
- current save/load callbacks;
- test fixtures and commands;
- actual RAM target and replay capacity;
- whether reward vectors are already present in transition `info`;
- whether ACL learning potential currently uses one-step or N-step residuals.

A repository finding that would change the specified behavior must return to
the approval process rather than being silently accommodated.

---

## 16. References

### Scientific literature

1. T. Schaul, J. Quan, I. Antonoglou, and D. Silver,
   “Prioritized Experience Replay,” ICLR 2016.
   https://arxiv.org/abs/1511.05952

   Supports proportional and rank-based prioritization, TD-error-based
   relevance, maximum insertion priority, importance-sampling correction, and
   beta annealing.

2. G. Barth-Maron et al.,
   “Distributed Distributional Deterministic Policy Gradients,” ICLR 2018.
   https://openreview.net/forum?id=SyZipzbCb
   https://arxiv.org/abs/1804.08617

   Supports the use of N-step returns and prioritized replay in off-policy
   continuous actor-critic learning. It does not directly prescribe the
   project-specific \(n=3\), twin-error aggregation, or persistence policy.

3. S. Fujimoto, H. van Hoof, and D. Meger,
   “Addressing Function Approximation Error in Actor-Critic Methods,” ICML 2018.
   https://arxiv.org/abs/1802.09477

   Supports the baseline TD3 clipped double-Q and target-policy-smoothing
   target retained by this specification.

4. T. Haarnoja et al.,
   “Soft Actor-Critic Algorithms and Applications,” 2018.
   https://arxiv.org/abs/1812.05905

   Supports the entropy-regularized SAC objective. The selected N-step target
   convention is specifically aligned to the SB3 implementation.

5. A. Saglam et al.,
   “Actor Prioritized Experience Replay,” 2022.
   https://arxiv.org/abs/2209.00532

   Motivates the stated limitation that high-TD-error prioritization is
   directly aligned with critic correction but may not be ideal for actor
   updates. Actor-aware replay remains deferred.

### Official dependency sources

6. Stable-Baselines3, SAC documentation, version line consulted 17 July 2026.
   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

   Documents `n_steps`, replay-buffer customization, replay persistence, and
   SAC parameters.

7. Stable-Baselines3,
   `stable_baselines3/common/buffers.py`, consulted 17 July 2026.
   https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

   Defines upstream `NStepReplayBuffer`, effective discounts, termination,
   truncation, frontier behavior, and the memory-optimization limitation.

8. Stable-Baselines3,
   `stable_baselines3/common/off_policy_algorithm.py`, consulted 17 July 2026.
   https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/off_policy_algorithm.py

   Defines automatic replay-class selection, custom replay construction, and
   replay save/load behavior.

9. Stable-Baselines3,
   `stable_baselines3/td3/td3.py`, consulted 17 July 2026.
   https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py

   Defines the TD3 target and consumption of replay-provided discounts.

10. Stable-Baselines3,
    `stable_baselines3/sac/sac.py`, consulted 17 July 2026.
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py

    Defines the SAC target, final entropy bootstrap, and consumption of
    replay-provided discounts.

11. Stable-Baselines3,
    checkpoint callback source, consulted 17 July 2026.
    https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/callbacks.html

    Confirms that replay-buffer checkpointing is separately enabled and
    disabled by default.

### Project sources and decisions

12. `docs/specifications/rulebook_v4.6_specification.md`.

    Defines the rulebook learner signals and separation between live rule
    evaluation and downstream learner/reward adaptation.

13. `docs/specifications/observation_v1.1_specification.md`.

    Defines the supported flat observation schemas and dimensions.

14. `docs/specifications/encoder_v1.0_specification.md`.

    Defines SB3 TD3/SAC/PPO integration and required smoke combinations.

15. `docs/specifications/automatic_curriculum_learning_v1_specification.md`.

    Defines scenario-level ACL as separate from transition-level replay.

16. `docs/specifications/rulebook_scalarization_v1.0_specification.md`.

    Defines the scalar reward producer, scalarization modes, reward-vector
    provenance, and compatibility identity consumed by replay.

17. `docs/decisions/ADR-011-rulebook-scalarization-v1.md`.

    Records the approved scalarization default, reward metadata, transfer
    boundary, and conditional future N-step/PER compatibility.

18. `docs/engineering_workflow.md`.

    Defines specification approval, fail-fast behavior, protected tests,
    reconciliation, and authority handling.

19. Project design decisions explicitly confirmed by the user on
    17 July 2026:

    - N-step project core with \(n=3\);
    - PER proportional, optional, and disabled by default;
    - mean absolute twin-critic TD-error priority;
    - critic-only IS weighting;
    - advanced PER variants deferred;
    - replay persistence optional and disabled by default.

---

## 17. Implementation Handoff Checklist

Before setting `Status: APPROVED`, confirm:

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, shapes, timing, and consumers.
- [x] Prohibited future, privileged, leaked, and diagnostic-only data is listed.
- [x] N-step formulas and effective-horizon behavior are explicit.
- [x] TD3 and SAC target semantics are explicit.
- [x] PPO non-applicability is explicit.
- [x] Termination, truncation, frontier, and episode boundaries are explicit.
- [x] Proportional PER formulas and sampling are explicit.
- [x] Twin-critic priority aggregation is explicit.
- [x] IS weighting and actor/critic behavior are explicit.
- [x] Vectorized-environment priority identity is explicit.
- [x] State ownership, initialization, update order, and overwrite are explicit.
- [x] Persistence is optional and disabled by default.
- [x] Resume classification and compatibility are explicit.
- [x] Configuration fields and frozen defaults are explicit.
- [x] Numerical errors and prohibited fallbacks are explicit.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected.
- [x] Scientific sources, dependency behavior, and project adaptations are
      distinct.
- [x] No material behavior decision remains open.
- [ ] User has approved the complete document.
- [ ] Codex has inspected the pinned local SB3 fork.
- [ ] Repository-specific ExecPlan has been produced.
- [ ] Final repository path has been registered in `project_index.md`.

### Recommended Codex handoff

After approval, Codex shall:

1. verify the pinned SB3 fork and report deviations;
2. update metadata to `APPROVED`;
3. move/rename the specification to its canonical path;
4. register it in `project_index.md`;
5. create an ExecPlan following `.agent/PLANS.md`;
6. map each requirement and acceptance criterion to code and tests;
7. implement uniform N-step integration first;
8. run protected N-step tests and smoke training;
9. implement the optional custom PER path;
10. implement persistence-disabled behavior before optional save/load;
11. run final reconciliation before marking `VERIFIED`.

---

## 18. Approval Record

- Approved by: `user`
- Approval date: `2026-07-17`
- Approval evidence: User explicitly authorized proceeding after the review resolutions were integrated and no material blockers remained: `"se non sono rimasti altri punti da sbloccare, procedi"`.
- Approval notes: The approval covers the complete revised candidate, including D1–D6, persisted beta progress, explicit inactive PPO configuration, canonical termination/truncation collection, persistence migration and pairing, and reward-semantic compatibility.
- Repository path: `docs/specifications/transition_replay_v1_specification.md`
- Project index updated: `YES; registered 2026-07-17`
