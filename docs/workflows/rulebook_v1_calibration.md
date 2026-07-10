# Calibrazione della scalarizzazione Rulebook v1

Questa procedura va eseguita **solo dopo il freeze** di:

- quattro regole (`collision_severity`, `allowed_driving_area`,
  `lane_marking_compliance`, `local_route_progress`);
- estrazione degli input runtime MetaDrive;
- semantica dei margini e priorità;
- policy strict (nessun fallback, input non disponibile o errore interrompono la run).

Non riutilizzare log, scale o risultati raccolti prima di tale freeze.

## 1. Raccogliere margini non influenzati dalla scalarizzazione

Usare `monitor_only`: il rulebook calcola e registra i margini, ma l'agente
continua a ricevere il reward nativo. I log sono abilitati solo per questa fase.

```bash
for s in 0 1 2 3 4; do
  docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_scalar_reward_scale_tuning_no_curr \
    reward=monitor_only \
    experiment.name=rulebook_v1_calibration \
    run_profile=medium \
    seed=$s \
    reward.include_violation_vector=true \
    reward.rule_margin_log_path='outputs/calibration/v1/seed_${seed}.jsonl'
done
```

Ogni run deve completare in strict mode: una run interrotta segnala un input
rulebook non affidabile, non un dato da ignorare.

## 2. Coprire i casi rari

Le collisioni, l'uscita dall'area ammessa e le violazioni delle marcature
potrebbero essere troppo rare in rollout normali. Generare scenari forzati
finché ogni regola non raggiunge la copertura minima indicata al passo 3.

```bash
for i in $(seq 0 99); do
  docker compose run --rm dev uv run --no-sync python \
    src/thesis_rl/tools/debug/force_rule_scenarios.py \
    --start-seed $((10000 + i)) \
    --seed $((42 + i)) \
    --map 5 \
    --traffic-density 0.5 \
    --out "outputs/calibration/v1/forced_${i}.json"
done
```

## 3. Stimare le scale dai margini attivi

Il tool usa il percentile 90 dei margini assoluti attivi. In modalità strict
fallisce se una regola ha pochi esempi: non inserire valori inventati.

```bash
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.tools.calibration.aggregate_rule_margins \
  --input 'outputs/calibration/v1/*.jsonl' \
  --output outputs/calibration/v1/aggregated_rule_margins.jsonl

docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.tools.calibration.scale_tuning \
  --input outputs/calibration/v1/aggregated_rule_margins.jsonl \
  --percentile 90 \
  --min-scale 1e-6 \
  --min-active-margin 1e-9 \
  --min-samples 300 \
  --strict \
  --output-json outputs/calibration/v1/scale_report.json
```

Quando il comando passa, copiare il blocco `scales` dal report in
`conf/reward/rulebook_defaults.yaml`.

## 4. Tarare la forma della scalarizzazione

Dopo aver fissato le scale, confrontare una piccola griglia:

- `a`: `1.8`, `2.0`, `2.2`;
- `c`: `1`, `2`, `3`, `5`.

Esempio:

```bash
docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.train \
  --config-name presets/td3/td3_scalar_reward_scale_tuning_no_curr \
  experiment.name=rulebook_v1_scalar_check \
  seed=0 \
  reward.a=2.0 \
  reward.c=3.0 \
  reward.rule_margin_log_path='${paths.logs_dir}/rule_margins.jsonl'
```

Con `reward.behavior=scalar_reward`, `lambda_env` e `lambda_rule` non
influenzano il reward usato per l'apprendimento; servono solo nel successivo
confronto con `hybrid`.

## Criteri di accettazione

- guida sicura in avanti > fermo > collisione o uscita dall'area ammessa;
- una collisione ad alta velocità è peggiore di una a bassa velocità;
- un guadagno di progresso non compensa una violazione seria di sicurezza;
- nessuna regola è sempre zero e le saturazioni non sono quasi sempre al 100%;
- nessun record presenta `available=false` o `fallback_used=true`.

Gli audit JSONL sono opzionali e vanno disabilitati (`rule_margin_log_path: null`)
nelle run normali, una volta completata la calibrazione.
