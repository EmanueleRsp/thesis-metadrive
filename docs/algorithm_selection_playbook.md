# Algorithm Selection Playbook

Nota importante:

- questo documento descrive il playbook di selezione algoritmo per la traccia
  `native` di riferimento
- non e' piu' la pipeline principale per scegliere il backbone finale della
  tesi
- per la sequenza aggiornata del task finale, fare riferimento a
  `docs/thesis_experiment_roadmap.md`

Questa nota raccoglie la sequenza operativa concordata per:

1. chiudere la selezione del miglior algoritmo scalar baseline `native` tra
   `td3_sb3`, `sac_sb3`, `ppo_sb3`
2. mantenere il protocollo di confronto esplicito e ripetibile
3. preparare il terreno per il blocco successivo
   (`obs`, encoder/decoder, curriculum, rulebook, scalarizzazione, replay
   buffer)

## Stato attuale

Le basi gia` fissate nel repo sono:

- contract nativo esplicito in due varianti:
  - `conf/env/metadrive_native_strict.yaml`
  - `conf/env/metadrive_native_relaxed.yaml`
- preset di qualification aggiunti sotto `conf/presets/selection/`
- protocollo comune di qualification:
  - `run_profile=thesis`
  - `reward=monitor_only`
  - `curriculum=disabled`
  - `obs=lidar_state`
  - `encoder=none`
  - `env=metadrive_native_strict`
  - `env.vectorized.num_envs=5`
  - `experiment.eval_interval=50000`
  - `experiment.eval_episodes=20`
  - `experiment.final_eval_episodes=100`

Decisione operativa corrente:

- la Fase 1 di confronto `strict` vs `relaxed` ha dato un vantaggio chiaro al
  contract `strict`, anche se con performance assolute ancora deboli
- il prossimo controllo concordato e' un rerun `strict` con `num_envs=4` per
  verificare che `5` env paralleli non stiano peggiorando la stabilita'
- salvo smentita dal rerun di controllo, `metadrive_native_strict` e' il
  contract da usare per tutta la selezione algoritmo scalar-native
- il trade-off "meglio fuori carreggiata che collisione" verra' modellato piu'
  avanti via reward/rulebook, non rilassando ora il terminal
- `env.vectorized.num_envs=5` e' il default operativo per i confronti correnti,
  per ridurre i crash GPU quando il carico e' condiviso o gli encoder sono
  pesanti

Motivazione della scelta `lidar_state` in questa fase:

- e` l'osservazione piu` baseline e piu` collaudata
- riduce il rischio di confondere differenze algoritmiche con differenze di
  rappresentazione
- `semantic_state` e gli encoder thesis verranno valutati dopo la scelta
  dell'algoritmo

## Sequenza consigliata

### 0. Commit e test del protocollo

Se non ancora fatto dopo le ultime modifiche:

```bash
cd /workspace/thesis-metadrive
uv run --no-sync python -m pytest -q tests/test_hydra_agent_presets.py
```

Poi commit del preset pack / config aggiornata.

### 1. Triage storico delle run gia` fatte

Le run `EXP_recheck_*` servono come contesto storico, non come verdetto finale,
perche' il task MetaDrive e` stato aggiornato.

Directory isolate per il confronto:

```bash
cd /workspace/thesis-metadrive

export SCRATCH_ROOT="$(dirname "$HOME")"
export OUT_ROOT="$SCRATCH_ROOT/thesis-metadrive/outputs"
export CMP_ROOT="$SCRATCH_ROOT/outputs_exp_recheck_alg_compare"

rm -rf "$CMP_ROOT"
mkdir -p "$CMP_ROOT" "$CMP_ROOT/analysis"
```

Copia delle run storiche:

```bash
for g in \
  EXP_recheck_td3_sb3_medium \
  EXP_recheck_sac_sb3_medium \
  EXP_recheck_ppo_sb3_medium
do
  rsync -a "$OUT_ROOT/$g/" "$CMP_ROOT/$g/"
done
```

Analisi:

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

Se i plot falliscono per `promotions_all_runs.csv`, workaround:

```bash
export AGG_DIR="$CMP_ROOT/analysis/medium/comparisons/algorithm/native__monitor_only__disabled__selection/aggregated"
mkdir -p "$AGG_DIR"
printf "condition_id,algorithm,reward_type,reward_behavior,curriculum_name,curriculum_enabled,rulebook_config,seed,run_id,event_type,global_step\n" > "$AGG_DIR/promotions_all_runs.csv"
```

```bash
uv run --no-sync python -m thesis_rl.analysis.run_analysis \
  --outputs-root "$CMP_ROOT" \
  --analysis-root "$CMP_ROOT/analysis" \
  --run-profile medium \
  --only plots \
  --comparison-dimension algorithm \
  --reward-type native \
  --reward-behavior monitor_only \
  --curriculum-name disabled \
  --rulebook-config selection
```

Tabella finale:

```bash
column -s, -t < "$CMP_ROOT/analysis/medium/comparisons/algorithm/native__monitor_only__disabled__selection/tables/final_evaluation.csv" | less -S
```

### 2. Qualification run dei tre algoritmi sul native contract chiuso

Host:

```bash
cd /home/e.respino/main/thesis/thesis-metadrive
```

Wrapper consigliato:

```bash
scripts/run_algorithm_selection.sh run-parallel --tag v3
```

Nota:

- da ora i preset `*_qual_lidar_thesis` puntano esplicitamente a
  `env=metadrive_native_strict`

Variante piu' conservativa:

```bash
scripts/run_algorithm_selection.sh run --tag v3
```

Pulizia rapida di eventuali sessioni tmux precedenti:

```bash
scripts/run_algorithm_selection.sh cleanup --tag v3
```

Comandi espliciti alternativi:

TD3:

```bash
scripts/tmux_seed_grid.sh \
  --session "qual_td3_sb3_lidar_thesis_v2" \
  --seed-start 0 \
  --seed-end 2 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/selection/td3_sb3_qual_lidar_thesis \
    analysis.experiment_group=EXP_qual_td3_sb3_lidar_thesis_v2
```

SAC:

```bash
scripts/tmux_seed_grid.sh \
  --session "qual_sac_sb3_lidar_thesis_v2" \
  --seed-start 0 \
  --seed-end 2 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/selection/sac_sb3_qual_lidar_thesis \
    analysis.experiment_group=EXP_qual_sac_sb3_lidar_thesis_v2
```

PPO:

```bash
scripts/tmux_seed_grid.sh \
  --session "qual_ppo_sb3_lidar_thesis_v2" \
  --seed-start 0 \
  --seed-end 2 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/selection/ppo_sb3_qual_lidar_thesis \
    analysis.experiment_group=EXP_qual_ppo_sb3_lidar_thesis_v2
```

### 3. Analisi delle qualification run

Container:

```bash
cd /workspace/thesis-metadrive

export SCRATCH_ROOT="$(dirname "$HOME")"
export OUT_ROOT="$SCRATCH_ROOT/thesis-metadrive/outputs"
export CMP_ROOT="$SCRATCH_ROOT/outputs_qual_alg_compare_v2"

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

Tabella finale:

```bash
column -s, -t < "$CMP_ROOT/analysis/thesis/comparisons/algorithm/native__monitor_only__disabled__selection/tables/final_evaluation.csv" | less -S
```

### 4. Decisione dopo la qualification

Regola pratica:

- se un algoritmo e` chiaramente peggiore, scartarlo
- se c'e` un vincitore netto, promuoverlo alla fase successiva
- se due algoritmi sono vicini, portare avanti entrambi al confronto `0..9`

Metriche da leggere con priorita`:

1. `final_eval.csv`
2. `success rate`
3. `route completion`
4. collisioni / out-of-road
5. stabilita` tra seed

### 5. Run `0..9` del vincitore

Template:

```bash
cd /home/e.respino/main/thesis/thesis-metadrive

scripts/tmux_seed_grid.sh \
  --session "final_<ALG>_sb3_lidar_thesis_v2" \
  --seed-start 0 \
  --seed-end 9 \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/selection/<ALG>_sb3_qual_lidar_thesis \
    analysis.experiment_group=EXP_final_<ALG>_sb3_lidar_thesis_v2
```

Sostituire `<ALG>` con `td3`, `sac` oppure `ppo`.

### 6. Solo dopo: osservazione ed encoder

Una volta selezionato l'algoritmo:

1. confronto `lidar_state` vs `semantic_state`
2. confronto encoder nel dominio osservativo che regge meglio
3. `LQ` solo su `semantic_state`

Combinazioni sensate per il blocco successivo:

- `lidar_state + none`
- `lidar_state + mlp`
- `semantic_state + none`
- `semantic_state + mlp`
- `semantic_state + lq`

## Nota finale

Non aprire ancora il blocco:

- curriculum
- rulebook reward shaping
- scalarizzazione finale
- replay buffer priority / eligibility traces

prima di aver fissato almeno:

- algoritmo candidato
- osservazione di base
- prima shortlist di encoder
