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

Per i nuovi confronti tratta come canonici i preset fork-backed sotto
`presets/agent/*_sb3`. Evita di combinare backend legacy
(`agent/planner/algorithm=sac`, `td3`, `ppo`) con decoder "SB3-like" a meno
che tu non stia facendo un confronto storico o di compatibilita`.

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
    --config-name presets/agent/sac_sb3 \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled \
    env.vectorized.num_envs=5
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

Questo wrapper:

- crea le sessioni tmux con naming coerente
- usa i preset di selezione correnti
- lancia `td3`, `sac`, `ppo` insieme
- aspetta il completamento reale di tutti i gruppi
- rilancia l'analisi finale con i path giusti

Se vuoi una variante piu' conservativa lato GPU, resta disponibile anche:

```bash
scripts/run_algorithm_selection.sh run --tag v3
```

Per pulire eventuali sessioni tmux rimaste appese prima di rilanciare:

```bash
scripts/run_algorithm_selection.sh cleanup --tag v3
```

Se preferisci vedere i comandi espliciti o fare piccole varianti manuali, resta
valido anche il blocco seguente.

Per un confronto fork-backed tra algoritmi con setup il piu` vicino possibile
alle policy MLP standard di SB3:

- `agent/planner/encoder=none`
- `agent/planner/algorithm=td3_sb3` per `td3`
- `agent/planner/algorithm=sac_sb3` per `sac`
- `agent/planner/algorithm=ppo_sb3` per `ppo`
- decoder coerente con il backend scelto (`td3_sb3`, `sac_sb3`, `ppo_sb3`)

Setup consigliato per stabilità iniziale e confronto più pulito:

- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`
- `experiment.eval_interval=50000`
- `experiment.eval_episodes=20`

Per lanciare solo tre seed, usa `--seed-start` e `--seed-end` in modo
esplicito.

```bash
for alg in td3 sac ppo; do
  case "$alg" in
    td3) alg_cfg=td3_sb3; dec=td3_sb3 ;;
    sac) alg_cfg=sac_sb3; dec=sac_sb3 ;;
    ppo) alg_cfg=ppo_sb3; dec=ppo_sb3 ;;
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
      env.vectorized.num_envs=5 \
      experiment.eval_interval=50000 \
      experiment.eval_episodes=20 \
      agent/planner/encoder=none \
      agent/planner/decoder=$dec \
      agent/planner/algorithm=$alg_cfg \
      analysis.experiment_group=EXP_cmp_alg_${alg}_sb3fork_lidar_RP_thesis_CUR_disabled_REW_monitor_only
done
```

Se la GPU resta stabile con `3` seed per algoritmo, puoi allargare la finestra
dei seed rilanciando ad esempio con `--seed-start 3 --seed-end 5`. Se usi
encoder pesanti o la GPU e' condivisa, riduci il parallelismo prima di
aumentare `num_envs`.

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
