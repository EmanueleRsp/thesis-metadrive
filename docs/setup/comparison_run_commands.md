# Comparison Run Commands

This document collects the current commands used to launch comparison runs in
the Docker-based setup.

## Scope

These commands assume:

- a running Compose service named `dev`
- the project mounted at `/workspace/thesis-metadrive`
- outputs written under `OUTPUTS_ROOT` inside the container
- `OUTPUTS_ROOT=/workspace/outputs` by default
- commands launched from the host terminal at the repository root

```bash
cd /path/to/thesis-metadrive
```

## Initial Smoke Check

```bash
docker compose exec dev bash -lc 'cd /workspace/thesis-metadrive && pwd && uv run --no-sync python -c "import thesis_rl; print(\"ok\")"'
```

## Smoke

```bash
scripts/tmux_seed_grid.sh \
  --session smoke_alg_sac \
  --seed-list 0 \
  --docker-compose-service dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/agent/sac_sb3 \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled \
    env.vectorized.num_envs=5
```

## Observation Comparison

```bash
for obs in lidar_state semantic_state; do
  scripts/tmux_seed_grid.sh \
    --session "cmp_obs_${obs}" \
    --docker-compose-service dev -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config \
      run_profile=thesis \
      reward=monitor_only \
      curriculum=disabled \
      obs=$obs \
      env.vectorized.num_envs=5 \
      agent/planner/encoder=none \
      agent/planner/decoder=sac_sb3 \
      agent/planner/algorithm=sac_sb3 \
      analysis.experiment_group=EXP_cmp_obs_${obs}_sac_sb3flat_RP_thesis_CUR_disabled_REW_monitor_only
done
```

## Encoder Comparison

```bash
for enc in lq mlp none; do
  if [ "$enc" = "none" ]; then dec=sac_sb3; else dec=mlp_encoded; fi
  scripts/tmux_seed_grid.sh \
    --session "cmp_enc_${enc}" \
    --docker-compose-service dev -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config \
      run_profile=thesis \
      reward=monitor_only \
      curriculum=disabled \
      obs=semantic_state \
      env.vectorized.num_envs=5 \
      agent/planner/encoder=$enc \
      agent/planner/decoder=$dec \
      agent/planner/algorithm=sac_sb3 \
      analysis.experiment_group=EXP_cmp_enc_${enc}_semantic_sac_sb3_RP_thesis_CUR_disabled_REW_monitor_only
done
```

## Algorithm Comparison

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

## GPU Monitoring

```bash
nvidia-smi
```

If the GPU becomes unstable, reduce parallelism before increasing `num_envs`.
