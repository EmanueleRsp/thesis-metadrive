# SAC SB3 Parity Porting Plan

Objective: bring the custom `SAC` implementation as close as possible to
`stable-baselines3` before building lexicographic or distributional variants.

## Scope

This document covers:

- the current custom `SAC` algorithm
- confirmed differences relative to SB3
- recommended porting order
- validation checks to run after each change

External references:

- SB3 SAC docs: <https://stable-baselines3.readthedocs.io/en/master/modules/sac.html>
- SB3 repository: <https://github.com/DLR-RM/stable-baselines3>

## Current Status In The Repository

Custom SAC implementation:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Relevant configs:

- `conf/agent/planner/algorithm/sac.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/sac_sb3.yaml`

## Confirmed Differences Vs SB3

### 1. Network Architecture

The largest initial difference was the network:

- old setup: `encoder=mlp` + `decoder=mlp_encoded`
- SB3-like setup: `encoder=none` + `decoder=sac_sb3`

With vector observations, SB3 effectively uses:

- flattened observations
- actor MLP `[256, 256]`
- critic MLP `[256, 256]`

The SB3-like decoder already exists in the repository:

- `conf/agent/planner/decoder/sac_sb3.yaml`

### 2. Stochastic Policy And Log-Prob

In the current custom implementation:

- Gaussian actor with `tanh`
- squash correction in `log_prob`
- state-dependent `log_std`

Files:

- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/algorithms/sac.py`

This is conceptually close to SB3, but it still needs a line-by-line check to
align sampling, log-prob, bounds, and initialization.

### 3. Temperature / `alpha`

In the current custom implementation:

- `ent_coef=auto`
- `target_entropy=auto`
- separate `log_alpha` update

Files:

- `src/thesis_rl/agent/planners/algorithms/sac.py`

This is close to SB3, but it must be checked carefully:

- initialization
- update formula
- speed of `alpha` collapse

### 4. Train / Update Semantics

In the current custom implementation:

- `train_freq` is a plain integer
- `gradient_steps=auto` is resolved as `train_freq * n_envs`

Files:

- `src/thesis_rl/agent/planners/algorithms/sac.py`

With `train_freq=1` and `num_envs=4`, behavior is close to SB3 when you want a
1:1 ratio between updates and collected data, but it is not a general replica
of the full SB3 semantics.

### 5. Replay Buffer

The custom buffer is minimal:

- only `obs/actions/rewards/dones/next_obs`
- no SB3 framework buffer variants
- no memory optimizations or additional helpers

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`

This is not necessarily a bug, but it is a structural difference.

### 6. Timeout Handling

This point has already been fixed:

- timeouts are not treated as true terminals in replay
- `terminal_observation` is used when available

Files:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/agent.py`

### 7. Planner / Lifecycle / Agent Integration

Audit completed:

- identity adapter
- lifecycle without anomalous logic
- buffer populated correctly
- updates called correctly

Current conclusion:

- no gross bug appears in the global wiring
- if a problem exists, it is more likely in the algorithm details or the
  representation

## Recommended Porting Priority

Recommended order:

1. policy / network
2. sampling and log-prob
3. temperature `alpha`
4. train / update schedule
5. replay semantics
6. logging and metrics parity

## Operational Plan

### Phase 1 - SB3-Faithful Network

Target:

- use `encoder=none` only
- use `decoder=sac_sb3`
- verify that actor and critic really use flat MLP stacks

To inspect:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Expected outcome:

- no additional deep encoder
- no `layer_norm`
- policy closer to SB3

### Phase 2 - Sampling And Log-Prob

Target:

- verify `Normal -> rsample -> tanh`
- verify the `log_prob` correction
- verify the distinction between stochastic and deterministic action

To verify or align:

- `sample()`
- `evaluate()`
- `log_std` initialization
- `log_std` bounds

Files:

- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Note:

even a small discrepancy here can significantly alter learning.

### Phase 3 - Temperature `alpha`

Target:

- align `ent_coef=auto` behavior
- verify `target_entropy = -action_dim`
- verify the learning update for `log_alpha`

Files:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `conf/agent/planner/algorithm/sac.yaml`

Signals to monitor:

- `alpha`
- `logp`
- action magnitude

### Phase 4 - Update Scheduling

Target:

- check that the meaning of:
  - `train_freq`
  - `gradient_steps`
  - `tau`
  - `learning_starts`

  is as consistent as possible with SB3

Focus:

- do not change too many variables at once
- initially keep `train_freq=1`
- test with `num_envs=4`

### Phase 5 - Replay Semantics

Target:

- confirm parity of stored transitions
- confirm parity on timeouts
- confirm parity of bootstrap with `next_obs`

Files:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/agent.py`

### Phase 6 - Parity Checks

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
- `alpha`
- action magnitude
- update counts
- visual behavior

## Recommended Implementation In The Next Session

Suggested tasks, in order:

1. line-by-line audit of `sac.py` against SB3 SAC
2. verification of `SquashedGaussianActor`
3. refactor of `alpha` / `ent_coef=auto` if needed
4. verification of `sac.yaml` config
5. short debug run (`50k` / `100k`) with `sac_sb3` network
6. comparison against off-policy debug logs

## Decision Criterion

If, after an SB3-faithful SAC port:

- SAC starts moving and learning again, the problem was in the custom
  implementation or config
- SAC remains stuck or keeps timing out, the issue should be searched outside
  the algorithm:
  - environment
  - reward
  - observation
  - action-space semantics

## Reference Command For SAC Testing

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_sac_sb3like" \
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
    agent/planner/decoder=sac_sb3 \
    agent/planner/algorithm=sac \
    analysis.experiment_group=EXP_dbg_sac_sb3like_lidar
```

## Final Note

For the thesis objective, the best strategy is:

1. bring the custom implementation to credible parity with SB3
2. verify that parity empirically
3. build lexicographic and distributional variants on top of that base
