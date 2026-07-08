# PPO SB3 Parity Porting Plan

Objective: bring the custom `PPO` implementation as close as possible to
`stable-baselines3` before building lexicographic or distributional variants.

## Scope

This document covers:

- the current custom `PPO` algorithm
- confirmed differences relative to SB3
- recommended porting order
- validation checks to run after each change

External references:

- SB3 PPO docs: <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>
- SB3 repository: <https://github.com/DLR-RM/stable-baselines3>

## Current Status In The Repository

Custom PPO implementation:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Relevant configs:

- `conf/agent/planner/algorithm/ppo.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/ppo_sb3.yaml`

## Confirmed Differences Vs SB3

### 1. Network Architecture

The largest initial difference was the network:

- old setup: `encoder=mlp` + `decoder=mlp_encoded`
- SB3-like setup: `encoder=none` + `decoder=ppo_sb3`

With vector observations, SB3 PPO effectively uses:

- flattened observations
- a smaller policy/value MLP
- typically `[64, 64]`

The SB3-like decoder already exists in the repository:

- `conf/agent/planner/decoder/ppo_sb3.yaml`

### 2. Gaussian Policy

In the current custom implementation:

- the Gaussian policy is unsquashed
- the environment action is obtained by clipping to bounds
- the buffer stores the raw action

Files:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

This has already been moved closer to SB3 and is conceptually correct, but it
still needs to be checked in detail.

### 3. Rollout Buffer

The custom rollout buffer is minimal:

- `obs/actions/rewards/dones/values/log_probs`
- custom GAE
- custom minibatch iteration

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`

This is not necessarily wrong, but it is a structural difference from SB3's
infrastructure.

### 4. Timeout Handling

This point has already been fixed:

- reward bootstrapping on timeouts uses the value of
  `terminal_observation`

Files:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `src/thesis_rl/agent/agent.py`

### 5. Update Scheduling

In the current custom implementation:

- updates happen when the rollout buffer is full
- `n_epochs`, `batch_size`, `clip_range`, `clip_range_vf`, `target_kl`
- `normalize_advantage`

Files:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

This is already fairly close to SB3, but it should still be checked
line-by-line.

### 6. Planner / Lifecycle / Agent Integration

Audit completed:

- identity adapter
- lifecycle without anomalous logic
- rollout buffer populated correctly
- updates called correctly

Current conclusion:

- no gross bug appears in the global wiring
- if there is a problem, it is more likely in the PPO policy details or in the
  training setup

## Recommended Porting Priority

Recommended order:

1. policy/value network
2. action and log-prob handling
3. rollout and GAE semantics
4. update loop
5. logging and metrics parity

## Operational Plan

### Phase 1 - SB3-Faithful Network

Target:

- use `encoder=none` only
- use `decoder=ppo_sb3`
- verify that the policy and value paths match SB3

To inspect:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Expected outcome:

- no additional deep encoder
- compact SB3-style MLP policy

### Phase 2 - Action And Log-Prob

Target:

- verify the distinction between:
  - `raw_action`
  - clipped environment action
  - action stored in the buffer
- verify `log_prob` computation

Files:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Note:

even a small discrepancy here directly alters the surrogate loss.

### Phase 3 - Rollout And GAE

Target:

- verify GAE step-by-step
- verify `last_values`
- verify `last_dones`
- verify timeout bootstrapping

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/ppo.py`

### Phase 4 - Update Loop

Target:

- verify:
  - `n_steps`
  - `batch_size`
  - `n_epochs`
  - `clip_range`
  - `clip_range_vf`
  - `target_kl`
  - `normalize_advantage`
  - `max_grad_norm`

Files:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `conf/agent/planner/algorithm/ppo.yaml`

### Phase 5 - Parity Checks

Target:

- same observation config
- same reward config
- same network
- same hyperparameters
- same seeds

Minimum comparisons:

- `ep_rew_mean` trend
- `route_completion`
- `success_rate`
- `approx_kl`
- action magnitude
- update counts
- visual behavior

## Recommended Implementation In The Next Session

Suggested tasks, in order:

1. line-by-line audit of `ppo.py` against SB3 PPO
2. verification of action / raw_action / log_prob handling
3. verification of rollout buffer and GAE
4. verification of `ppo.yaml` config
5. short debug run (`50k` / `100k`) with `ppo_sb3` network
6. comparison between custom metrics and expected behavior

## Decision Criterion

If, after an SB3-faithful PPO port:

- PPO starts producing sensible behavior again, the problem was in the custom
  implementation or config
- PPO remains too aggressive or unstable, the issue should be searched in:
  - reward setup
  - observation
  - action semantics
  - hyperparameter tuning

## Reference Command For PPO Testing

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_ppo_sb3like" \
  --seed-start 0 \
  --seed-end 2 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name config \
    run_profile=thesis \
    reward=monitor_only \
    curriculum=disabled \
    obs=lidar_state \
    env.vectorized.num_envs=4 \
    experiment.eval_interval=50000 \
    experiment.eval_episodes=20 \
    agent/planner/encoder=none \
    agent/planner/decoder=ppo_sb3 \
    agent/planner/algorithm=ppo \
    analysis.experiment_group=EXP_dbg_ppo_sb3like_lidar
```

## Final Note

For the thesis objective, the best strategy is:

1. bring the custom implementation to credible parity with SB3
2. verify that parity empirically
3. build lexicographic and distributional variants on top of that base
