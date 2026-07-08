# Algorithm Selection Playbook

This document captures the current operational sequence for native-baseline
algorithm selection.

It is no longer the only decision path for the full thesis, but it remains the
reference playbook for the scalar-native comparison stage.

## Current Assumptions

The current qualification setup assumes:

- `run_profile=thesis`
- `reward=monitor_only`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env=metadrive_native_strict`
- `env.vectorized.num_envs=5`
- `experiment.eval_interval=50000`
- `experiment.eval_episodes=20`
- `experiment.final_eval_episodes=100`

## 0. Preset Sanity Check

```bash
cd /workspace/thesis-metadrive
uv run --no-sync python -m pytest -q tests/test_hydra_agent_presets.py
```

## 1. Historical Triage

Use historical `EXP_recheck_*` runs only as context, not as the final verdict.

```bash
cd /workspace/thesis-metadrive

export OUT_ROOT="${OUTPUTS_ROOT:-/workspace/outputs}"
export CMP_ROOT="${OUT_ROOT}/_comparisons/exp_recheck_alg_compare"

rm -rf "$CMP_ROOT"
mkdir -p "$CMP_ROOT" "$CMP_ROOT/analysis"
```

```bash
for g in \
  EXP_recheck_td3_sb3_medium \
  EXP_recheck_sac_sb3_medium \
  EXP_recheck_ppo_sb3_medium
do
  rsync -a "$OUT_ROOT/$g/" "$CMP_ROOT/$g/"
done
```

```bash
uv run --no-sync python -m thesis_rl.analysis.run_analysis \
  --outputs-root "$CMP_ROOT" \
  --analysis-root "$CMP_ROOT/analysis" \
  --run-profile medium \
  --only all \
  --no-videos \
  --seed-list 0,1,2 \
  --total-timesteps 350000 \
  --eval-episodes 20 \
  --final-eval-episodes 100 \
  --comparison-dimension algorithm \
  --reward-type native \
  --reward-behavior monitor_only \
  --curriculum-name disabled \
  --rulebook-config selection
```

## 2. Qualification Runs

Host:

```bash
cd /path/to/thesis-metadrive
```

Recommended wrapper:

```bash
scripts/run_algorithm_selection.sh run-parallel --tag v3
```

More conservative variant:

```bash
scripts/run_algorithm_selection.sh run --tag v3
```

Cleanup:

```bash
scripts/run_algorithm_selection.sh cleanup --tag v3
```

## 3. Qualification Analysis

Inside the container:

```bash
cd /workspace/thesis-metadrive

export OUT_ROOT="${OUTPUTS_ROOT:-/workspace/outputs}"
export CMP_ROOT="${OUT_ROOT}/_comparisons/qual_alg_compare_v2"

rm -rf "$CMP_ROOT"
mkdir -p "$CMP_ROOT" "$CMP_ROOT/analysis"
```

```bash
for g in \
  EXP_qual_td3_sb3_lidar_thesis_v2 \
  EXP_qual_sac_sb3_lidar_thesis_v2 \
  EXP_qual_ppo_sb3_lidar_thesis_v2
do
  rsync -a "$OUT_ROOT/$g/" "$CMP_ROOT/$g/"
done
```

```bash
uv run --no-sync python -m thesis_rl.analysis.run_analysis \
  --outputs-root "$CMP_ROOT" \
  --analysis-root "$CMP_ROOT/analysis" \
  --run-profile thesis \
  --only all \
  --no-videos \
  --seed-list 0,1,2 \
  --total-timesteps 1500000 \
  --eval-episodes 20 \
  --final-eval-episodes 100 \
  --comparison-dimension algorithm \
  --reward-type native \
  --reward-behavior monitor_only \
  --curriculum-name disabled \
  --rulebook-config selection
```

## 4. Decision Rule

Use the qualification results to:

- discard clearly weaker algorithms
- promote a clear winner
- carry forward two close candidates if the result is ambiguous

Priority metrics:

1. `final_eval.csv`
2. success rate
3. route completion
4. collisions / out-of-road
5. cross-seed stability

## 5. Winner Expansion To Seeds 0..9

```bash
cd /path/to/thesis-metadrive

scripts/tmux_seed_grid.sh \
  --session "final_<ALG>_sb3_lidar_thesis_v2" \
  --seed-start 0 \
  --seed-end 9 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/selection/<ALG>_sb3_qual_lidar_thesis \
    analysis.experiment_group=EXP_final_<ALG>_sb3_lidar_thesis_v2
```

Replace `<ALG>` with `td3`, `sac`, or `ppo`.

## 6. Only After That: Observations And Encoders

Once the algorithm is chosen:

1. compare `lidar_state` vs `semantic_state`
2. compare encoders in the stronger observation domain
3. evaluate `LQ` only on `semantic_state`
