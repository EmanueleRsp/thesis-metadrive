# Comparison Run Commands

Questo documento raccoglie i comandi correnti per lanciare run di confronto
nel setup Docker attuale.

## Scope

Questi comandi assumono:

- container Compose attivo con nome `thesis-metadrive-dev`
- progetto montato in `/workspace/thesis-metadrive`
- output scritti sotto `OUTPUTS_ROOT` nel container
- `OUTPUTS_ROOT=/workspace/outputs` come default
- esecuzione dei comandi dal terminale host, nella root del repo

```bash
cd /path/to/thesis-metadrive
```

## Check rapido iniziale

```bash
docker compose exec dev bash -lc 'cd /workspace/thesis-metadrive && pwd && uv run --no-sync python -c "import thesis_rl; print(\"ok\")"'
```

## Smoke

```bash
scripts/tmux_seed_grid.sh \
  --session smoke_alg_sac \
  --seed-list 0 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/agent/sac_sb3 \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled \
    env.vectorized.num_envs=5
```

## Confronto osservazioni

```bash
for obs in lidar_state semantic_state; do
  scripts/tmux_seed_grid.sh \
    --session "cmp_obs_${obs}" \
    --docker-container thesis-metadrive-dev -- \
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

## Confronto encoder

```bash
for enc in lq mlp none; do
  if [ "$enc" = "none" ]; then dec=sac_sb3; else dec=mlp_encoded; fi
  scripts/tmux_seed_grid.sh \
    --session "cmp_enc_${enc}" \
    --docker-container thesis-metadrive-dev -- \
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

## Confronto algoritmi

Wrapper consigliato:

```bash
scripts/run_algorithm_selection.sh run-parallel --tag v3
```

Variante piu' conservativa:

```bash
scripts/run_algorithm_selection.sh run --tag v3
```

Pulizia rapida:

```bash
scripts/run_algorithm_selection.sh cleanup --tag v3
```

## Monitoraggio GPU

```bash
nvidia-smi
```

Se la GPU non regge, riduci il parallelismo prima di aumentare `num_envs`.
