# Comparison Run Commands

Questo documento raccoglie i comandi correnti per lanciare le run di confronto
nel setup Docker attuale.

## Scope

Questi comandi assumono:

- container Compose attivo con nome `thesis-metadrive-dev`
- progetto montato in `/workspace/thesis-metadrive`
- output scritti in `/scratch/$USER/thesis-metadrive/outputs`
- esecuzione dei comandi dal terminale host, nella root del repo

```bash
cd /home/e.respino/main/thesis/thesis-metadrive
```

## Nota importante

Non lanciare tutti i gruppi di confronto contemporaneamente: alcune
combinazioni, in particolare `sac`/`td3` con encoder pesanti e molti seed,
possono saturare la GPU.

Ordine consigliato:

1. smoke
2. confronto osservazioni
3. confronto encoder
4. confronto algoritmi

Lancia un gruppo, controlla che resti stabile, poi passa al successivo.

## Check rapido iniziale

Verifica che il container veda il progetto e che `thesis_rl` sia importabile:

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
    --config-name config \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled \
    obs=lidar_state \
    env.vectorized.num_envs=4 \
    agent/planner/encoder=mlp \
    agent/planner/decoder=mlp_encoded \
    agent/planner/algorithm=sac
```

## Confronto osservazioni

Per default `scripts/tmux_seed_grid.sh` lancia i seed `0..9`.

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
      agent/planner/encoder=mlp \
      agent/planner/decoder=mlp_encoded \
      agent/planner/algorithm=sac \
      analysis.experiment_group=EXP_cmp_obs_${obs}_sac_mlp_RP_thesis_CUR_disabled_REW_monitor_only
done
```

## Confronto encoder

```bash
for enc in lq mlp none; do
  if [ "$enc" = "none" ]; then dec=mlp_large; else dec=mlp_encoded; fi
  scripts/tmux_seed_grid.sh \
    --session "cmp_enc_${enc}" \
    --docker-container thesis-metadrive-dev -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config \
      run_profile=thesis \
      reward=monitor_only \
      curriculum=disabled \
      obs=semantic_state \
      agent/planner/encoder=$enc \
      agent/planner/decoder=$dec \
      agent/planner/algorithm=sac \
      analysis.experiment_group=EXP_cmp_enc_${enc}_semantic_sac_RP_thesis_CUR_disabled_REW_monitor_only
done
```

## Confronto algoritmi

Per un confronto più vicino possibile alle policy MLP di SB3 con osservazioni
vettoriali:

- `agent/planner/encoder=none`
- `agent/planner/decoder=td3_sb3` per `td3`
- `agent/planner/decoder=sac_sb3` per `sac`
- `agent/planner/decoder=ppo_sb3` per `ppo`

Setup consigliato per stabilità iniziale e confronto più pulito:

- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=4`
- `experiment.eval_interval=50000`
- `experiment.eval_episodes=20`

Per lanciare solo tre seed, usa `--seed-start` e `--seed-end` in modo
esplicito.

```bash
for alg in td3 sac ppo; do
  case "$alg" in
    td3) dec=td3_sb3 ;;
    sac) dec=sac_sb3 ;;
    ppo) dec=ppo_sb3 ;;
  esac
  scripts/tmux_seed_grid.sh \
    --session "cmp_alg_${alg}" \
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
      agent/planner/decoder=$dec \
      agent/planner/algorithm=$alg \
      analysis.experiment_group=EXP_cmp_alg_${alg}_sb3net_lidar_RP_thesis_CUR_disabled_REW_monitor_only
done
```

Se la GPU resta stabile con `3` seed per algoritmo, puoi allargare la finestra
dei seed rilanciando ad esempio con `--seed-start 3 --seed-end 5`.

## Sessioni tmux

Se una sessione esiste già, lo script fallisce. Controlli utili:

```bash
tmux ls
tmux kill-session -t cmp_obs_lidar_state
```

## Monitoraggio GPU

Mentre le run partono, puoi controllare la GPU dal terminale host:

```bash
nvidia-smi
```

Se vedi nuovi errori `CUDA out of memory`, fermati e rilancia meno gruppi in
parallelo.
