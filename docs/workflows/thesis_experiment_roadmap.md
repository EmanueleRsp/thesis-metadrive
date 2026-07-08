# Thesis Experiment Roadmap

This note defines the current operational sequence for reaching, cleanly and
incrementally:

- vector-valued rulebook reward
- scalarized rulebook reward
- a strong scalar baseline on the final task
- lexicographic and distributional algorithms

## Decisions Already Locked

- `env.vectorized.num_envs=5` remains the current operational default
- the real experimental target is no longer the `native` baseline alone, but
  the final `rulebook-aware` task
- termination flags should be chosen in the context of the final reward, not
  separately and too early
- the `native` baseline remains useful as a reference and control, but it
  should not by itself drive the final algorithm choice

## Experimental Contracts

### Native Reference Contract

Use:

- reporting reference baseline
- historical control against the MetaDrive native reward

Note:

- it is no longer the main pipeline for deciding the final thesis backbone

### Rulebook-Aware Contract

Use:

- selection of termination flags consistent with the final framework
- algorithm selection
- tuning
- observation selection
- encoder selection
- curriculum
- final scalar baseline
- comparison against lexicographic and distributional methods

Principle:

- if a violation should be treated as a graded preference or a penalizable
  rule, it should remain in the reward or rulebook rather than being turned
  into `done=True` too early

## General Principle

Do not treat everything as a single large grid search. The correct procedure
is:

1. implement the final reward and rulebook
2. lock the task and environment contract of the final problem
3. select the algorithm on the final task
4. tune the chosen algorithm
5. select observation and encoder
6. evaluate algorithm-level augmentations
7. introduce and evaluate curriculum
8. use the best scalar rulebook-aware baseline as the reference for
   lexicographic and distributional methods

Methodological rule:

- if you change reward or termination flags in a substantial way, you are
  changing the task
- when the task changes, the most sensitive upstream decisions should be
  revalidated, especially the algorithm choice

## Phase 0 - Technical Smoke Test

Budget:

- `run_profile=smoke`
- `seed=0`

Purpose:

- verify that each new config starts, logs, and terminates correctly

Lock:

- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Inspect:

- crashes
- NaN values
- missing artifacts
- presence of `final_eval.csv`

## Phase 1 - Rulebook Reward Implementation And Validation

Purpose:

- define rulebook metrics and violations
- produce a consistent vector-valued reward
- produce an initial scalarization usable by scalar algorithms

Budget:

- technical smoke: `run_profile=smoke`, `1` seed
- functional debug: `run_profile=fast`, `1` seed

Vary:

- implementation of reward vector components
- scalarization function

Lock:

- probe algorithm: `SAC`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- initial debug termination flags, preferably a `strict-like` variant

Inspect:

- semantic correctness of reward components
- strange saturation behavior or inconsistent signs
- qualitative correlation between observed events and penalty or reward
- absence of obviously broken metrics

Note:

- if a metric looks suspicious, such as `goal_progress violation` always being
  `1`, fix it here before using it to decide anything else

## Phase 2 - Lock The Rulebook-Aware Task Contract

Purpose:

- choose termination flags consistent with the final reward
- decide whether the final task should be `strict`, `relaxed`, or an
  intermediate variant

Budget:

- initial screening: `run_profile=fast`, `1` seed
- clean closure: `run_profile=fast`, `3` seeds

Recommended config:

- probe algorithm: `SAC`
- `reward=scalar_reward`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Vary:

- `strict`
- `relaxed`
- optionally an intermediate variant if a specific doubt emerges

Lock:

- algorithm
- hyperparameters
- observation
- encoder
- scalarization

Inspect:

- learnability
- stability
- degenerate behavior such as a stationary agent
- `collision_rate`
- `out_of_road_rate`
- `route_completion`
- `success_rate`
- per-rule violations

Correct interpretation:

- if `strict` reduces collisions only because the episode gets censored early,
  that alone is not enough to call it better
- if `relaxed` exposes useful rulebook trade-offs without destroying
  learnability, it is a strong candidate for the final task

Expected output:

- a locked `rulebook-aware contract` used in all subsequent phases

## Phase 3 - Algorithm Selection On The Final Task

Budget:

- qualification: `3` seeds (`0,1,2`)
- profile: selection preset or equivalent with
  `experiment.eval_interval=50000`,
  `experiment.eval_episodes=20`,
  `experiment.final_eval_episodes=100`

Recommended config:

- `reward=scalar_reward`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Vary:

- `td3_sb3`
- `sac_sb3`
- `ppo_sb3`

Lock:

- task contract locked in Phase 2
- scalarization fixed
- decoder consistent with the algorithm
- no serious tuning in this phase

Inspect for decision:

- `collision_rate`
- `out_of_road_rate`
- `top_rule_violation_rate`
- `success_rate`
- `route_completion`
- stability across seeds

Rule:

- compliance and safety come before average reward
- if one algorithm is clearly worse, discard it
- if two are close, carry both into the confirmation phase

## Phase 4 - Algorithm Confirmation

Budget:

- `5-10` seeds on the best `1-2` algorithms
- same protocol as Phase 3

Vary:

- only the algorithm among the remaining candidates

Lock:

- everything else

Inspect:

- final means
- `95% CI`
- curve stability
- robustness across seeds

Expected output:

- official baseline algorithm choice for the rulebook-aware task

## Phase 5 - Tuning The Chosen Algorithm

Recommended budget:

- `run_profile=tune`
- suggested target:
  - `total_timesteps` around `400k-600k`
  - `eval_interval=25000`
  - `eval_episodes=20`
  - `final_eval_episodes=50`
- `3` seeds per trial
- `8-12` random-search trials

Vary:

- a small number of high-impact hyperparameters

Suggested parameters:

- `PPO`: `learning_rate`, `n_steps`, `batch_size`, `ent_coef`
- `SAC`: `learning_rate`, `batch_size`, `learning_starts`, `buffer_size`
- `TD3`: `learning_rate`, `batch_size`, `learning_starts`,
  `action_noise_sigma`

Lock:

- reward
- task contract
- observation
- encoder
- curriculum

Inspect:

- safety-first ranking
- stability across seeds
- sample efficiency

Confirmation:

- top `2` settings with longer budget and `5` seeds
- top `1` setting with `10` seeds only if you need extra rigor

## Phase 6 - Observation Selection

Budget:

- screening: `run_profile=long`, `3` seeds
- confirmation: `run_profile=thesis`, `5-10` seeds

Vary:

- `obs=lidar_state`
- `obs=semantic_state`

Lock:

- algorithm already selected and tuned
- `reward=scalar_reward`
- `curriculum=disabled`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`
- task contract locked

Inspect:

- safety and performance
- training stability
- robustness across seeds

## Phase 7 - Encoder Selection

Note:

- this phase mainly makes sense if `semantic_state` remains competitive

Budget:

- screening: `run_profile=long`, `3` seeds
- confirmation: `run_profile=thesis`, `5-10` seeds

Vary:

- `encoder=none`
- `encoder=mlp`
- `encoder=lq`

Lock:

- tuned algorithm
- fixed observation
- scalarized reward
- curriculum disabled
- decoder consistent with the selected encoder pipeline
- task contract locked

Inspect:

- safety and performance
- sample efficiency
- seed variance

## Phase 8 - Algorithm-Level Augmentations

Purpose:

- evaluate internal algorithm changes only after the backbone, main
  hyperparameters, observation, and encoder are already fixed

Recommended order:

1. replay buffer and sampling strategy
2. temporal credit assignment and traces-like variants

### Phase 8a - Replay Buffer / Prioritized Sampling

Note:

- this sub-phase mainly makes sense for off-policy algorithms, in practice
  `SAC` and `TD3`
- for `PPO`, prioritized replay is not the natural lever

Budget:

- screening: `run_profile=long`, `3` seeds
- confirmation: `run_profile=thesis`, `5-10` seeds

Vary:

- `uniform replay`
- `prioritized replay`

Lock:

- chosen algorithm
- chosen hyperparameters
- observation
- encoder
- scalarized reward
- curriculum disabled
- task contract locked

Inspect:

- safety and performance
- sample efficiency
- robustness across seeds
- any numerical instability introduced by prioritized replay

### Phase 8b - Eligibility Traces / Credit Assignment

Note:

- on `PPO`, the natural lever is often `gae_lambda`
- on `SAC` and `TD3`, traces are not a plug-and-play modification

Budget:

- screening: `run_profile=long`, `3` seeds
- confirmation: `run_profile=thesis`, `5` seeds

Vary:

- baseline credit assignment
- traces-like variant or `lambda` / credit-assignment tuning

Lock:

- everything else

Inspect:

- sample efficiency
- stability
- real gain on final metrics

## Phase 9 - Curriculum

Budget:

- screening: `run_profile=thesis`, `3` seeds
- confirmation: `run_profile=thesis`, `5-10` seeds

Vary:

- `curriculum=disabled`
- `curriculum=staged` base

Lock:

- algorithm
- hyperparameters
- observation
- encoder
- scalarized reward
- task contract locked

Inspect:

- final metrics
- `steps_to_final_stage`
- `final_stage_reached`
- `failed_evals_before_promotion`

Only if curriculum truly helps:

- tune promotion threshold
- tune number of consecutive evaluations
- tune stage definition

## Phase 10 - Native Reference Baseline

Purpose:

- keep a comparable `native` baseline for the final report

Budget:

- lower priority
- `run_profile=fast` or `thesis`
- `3` seeds for an initial check

Recommended config:

- `reward=monitor_only`
- `strict` contract as the native reference baseline
- the same best algorithm found on the final task, or `SAC` as a probe

Note:

- this phase is for control and reporting, not for selecting the final
  framework backbone

## Phase 11 - Lexicographic And Distributional Algorithms

Budget:

- smoke and fast for debugging
- official comparison with `thesis` budget
- at least `5` seeds, preferably `10`

Reference baseline:

- the best scalar rulebook-aware baseline found in previous phases

Objective:

- evaluate the real gain relative to a strong scalar baseline consistent with
  the final task

## Practical Notes On GPU And Parallelism

- `num_envs=5` alone does not guarantee stability if the GPU is shared
- with heavy encoders, the number of seeds and concurrently launched groups
  also matters a lot
- if OOM appears, first reduce run-level parallelism and only then consider
  further cuts to `num_envs`
- for comparisons with heavy encoders, do not launch many groups together
