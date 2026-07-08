# TD3 SB3 Parity Porting Plan

Objective: bring the custom `TD3` implementation as close as possible to
`stable-baselines3` before building lexicographic or distributional variants.

## Scope

This document covers:

- the current custom `TD3` algorithm
- confirmed differences relative to SB3
- recommended porting order
- validation checks to run after each change

External references:

- SB3 TD3 docs: <https://stable-baselines3.readthedocs.io/en/master/modules/td3.html>
- SB3 repository: <https://github.com/DLR-RM/stable-baselines3>

## Current Status In The Repository

Custom TD3 implementation:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Relevant configs:

- `conf/agent/planner/algorithm/td3.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/td3_sb3.yaml`

## Confirmed Differences Vs SB3

### 1. Network Architecture

The largest difference was the network:

- old setup: `encoder=mlp` + `decoder=mlp_encoded`
- SB3-like setup: `encoder=none` + `decoder=td3_sb3`

With vector observations, SB3 effectively uses:

- flattened observations
- actor MLP `[400, 300]`
- critic MLP `[400, 300]`

The SB3-like decoder already exists in the repository:

- `conf/agent/planner/decoder/td3_sb3.yaml`

### 2. Exploration

In the current custom implementation:

- uniform random warmup until `learning_starts`
- then always additive Gaussian noise in `predict()`

Files:

- `src/thesis_rl/agent/planners/algorithms/td3.py`

Current parameters:

- `action_noise_sigma`
- `learning_starts`

Difference relative to SB3:

- in SB3, action noise is an explicit `ActionNoise` object
- the wiring is cleaner and separate from the policy
- the behavior may be equivalent, but today it is embedded inside `predict()`

### 3. Train / Update Semantics

In the current custom implementation:

- `train_freq` is a plain integer
- `gradient_steps=auto` is resolved as `train_freq * n_envs`

Files:

- `src/thesis_rl/agent/planners/algorithms/td3.py`

Important note:

with `train_freq=1` and `num_envs=4`, this is close to SB3 behavior when the
desired update-to-data ratio is 1:1. So this does not look like the main bug.

### 4. Replay Buffer

The custom buffer is intentionally minimal:

- only `obs/actions/rewards/dones/next_obs`
- no specialized buffer classes
- no memory optimizations
- no additional helpers from the SB3 framework

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`

This does not automatically imply a bug, but it is a structural difference.

### 5. Timeout Handling

This used to be a problematic point and has already been fixed:

- timeouts are not treated as true terminals in replay
- `terminal_observation` is used when available

Files:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/agent.py`

### 6. Planner / Lifecycle / Agent Integration

Audit completed:

- the adapter is identity
- the lifecycle delegates without anomalous logic
- the buffer really receives transitions
- updates are called correctly

Current conclusion:

- no gross bug appears in the global wiring
- if a problem exists, it is more likely in the algorithm details or the
  representation

## Recommended Porting Priority

Recommended order:

1. policy / network
2. exploration
3. train / update schedule
4. replay semantics
5. logging and metrics parity

## Operational Plan

### Phase 1 - SB3-Faithful Network

Target:

- use `encoder=none` only
- use `decoder=td3_sb3`
- verify that actor and critic really use flat MLP stacks

To inspect:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Expected outcome:

- no additional deep encoder
- no `layer_norm`
- policy closer to SB3

### Phase 2 - TD3 Exploration

Target:

- clearly separate:
  - random warmup
  - post-warmup action noise
  - target policy smoothing noise

To verify or align:

- `action_noise_sigma` in the custom code
- noise semantics relative to SB3
- optional support for `action_noise=None`

Files:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `conf/agent/planner/algorithm/td3.yaml`

Note:

this is one of the potentially most influential behavioral differences.

### Phase 3 - Update Scheduling

Target:

- check that the meaning of:
  - `train_freq`
  - `gradient_steps`
  - `policy_delay`
  - `tau`
  - `learning_starts`

  is aligned with SB3 as much as possible

Focus:

- do not change everything at once
- initially keep `train_freq=1`
- test with `num_envs=4`

To verify:

- whether a more explicit SB3-style semantic is worth introducing
- whether the `auto` value should remain or be replaced by a more controlled
  mapping

### Phase 4 - Replay Semantics

Target:

- confirm parity of stored transitions
- confirm parity on timeouts
- confirm parity of bootstrap with `next_obs`

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/agent.py`

This phase mainly prevents subtle discrepancies.

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
- action magnitude
- update counts
- visual behavior

## Recommended Implementation In The Next Session

Suggested tasks, in order:

1. line-by-line audit of `td3.py` against SB3 TD3
2. exploration refactor to align it with SB3 semantics
3. verification of `td3.yaml` config
4. short debug run (`50k` / `100k`) with `td3_sb3` network
5. comparison against off-policy debug logs

## Decision Criterion

If, after an SB3-faithful TD3 port:

- TD3 starts moving and learning again, the problem was in the custom
  implementation or config
- TD3 remains stuck or keeps timing out, the issue should be searched outside
  the algorithm:
  - environment
  - reward
  - observation
  - action-space semantics

## Reference Command For TD3 Testing

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_td3_sb3like" \
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
    agent/planner/decoder=td3_sb3 \
    agent/planner/algorithm=td3 \
    analysis.experiment_group=EXP_dbg_td3_sb3like_lidar
```

## Final Note

For the thesis objective, the best strategy is not "just use SB3 as-is", but:

1. bring the custom implementation to credible parity with SB3
2. verify that parity empirically
3. build lexicographic and distributional variants on top of that base
