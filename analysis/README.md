# Analysis Pipeline

Pipeline unica per aggregare run multi-seed, generare tabelle/plot e costruire confronti A/B/C.

## Schema canonico (nuovo)

Tutti i report usano questi campi:

- `reward_type`: `native` | `rulebook`
- `reward_behavior`: `off` | `monitor_only` | `scalar_reward`
- `rulebook_config`: `none` | `<nome file in conf/rulebook/*.yaml>`

## Comandi principali

```bash
# Pipeline quantitativa completa (aggregazione + tabelle + plot core)
python -m analysis.run_analysis --only all --no-videos

# Solo aggregazione
python -m analysis.run_analysis --only aggregate

# Solo tabelle
python -m analysis.run_analysis --only tables

# Solo plot
python -m analysis.run_analysis --only plots
```

## Flag più utili

- `--analysis-root` default `analysis`
- `--outputs-root` default `outputs`
- `--comparison-dimension` `none|curriculum|reward|algorithm`
- `--comparison-id` per rigenerare una sola comparison view
- `--algorithm`
- `--reward-type`
- `--reward-behavior`
- `--rulebook-config`
- `--reward-granularity` `semantic|raw` (solo confronto reward)
- `--include-effects-tables` (tabelle ablation opzionali)
- `--include-diagnostic-plots` (plot diagnostici opzionali)
- `--include-qualitative-pack` (manifest + GIF qualitativi su comparison views)
- `--qualitative-max-per-category` (attualmente supportato: `1`)

## Confronti A/B/C

```bash
# A) Effetto curriculum (varia SOLO curriculum)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension curriculum \
  --algorithm sb3_td3 \
  --reward-type native \
  --reward-behavior monitor_only \
  --rulebook-config selection

# B) Effetto reward (semantic: confronta reward_type)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension reward \
  --algorithm sb3_td3 \
  --curriculum-name stages \
  --rulebook-config selection \
  --reward-granularity semantic

# B-raw) Effetto reward (raw: confronta reward_behavior)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension reward \
  --algorithm sb3_td3 \
  --curriculum-name disabled \
  --reward-type native \
  --reward-granularity raw

# C) Effetto algoritmo (varia SOLO algoritmo)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension algorithm \
  --curriculum-name stages \
  --reward-type rulebook \
  --reward-behavior scalar_reward \
  --rulebook-config selection
```

## Pacchetto qualitativo curato

```bash
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension curriculum \
  --algorithm sb3_td3 \
  --reward-type native \
  --reward-behavior monitor_only \
  --rulebook-config selection \
  --include-qualitative-pack
```

Categorie fisse:

- `best`
- `median`
- `worst`
- `rule_violation_case`
- `curriculum_transition_case`

## Output principali

### `analysis/aggregated/`

- `train_chunks_all_runs.csv`
- `evals_all_runs.csv`
- `eval_episodes_all_runs.csv`
- `promotions_all_runs.csv`
- `rule_metrics_all_runs.csv`
- `final_eval_all_runs.csv`
- `selected_runs.csv`

### `analysis/tables/` (core)

- `final_evaluation.*`
- `curriculum_efficiency.*`
- `sample_efficiency_thresholds.*`
- `generalization_train_vs_eval.*`
- `rulebook_compliance.*`
- `rule_violation_by_rule.*`

### `analysis/plots/` (core)

- `learning_success_vs_global_step.png`
- `learning_collision_vs_global_step.png`
- `learning_out_of_road_vs_global_step.png`
- `learning_route_completion_vs_global_step.png`
- `learning_rule_top_violation_vs_global_step.png`
- `curriculum_stage_index_vs_global_step.png`
- `rule_metrics_violation_rate_by_rule.png`

### `analysis/comparisons/<dimension>/<comparison_id>/`

- `aggregated/*.csv`
- `tables/*`
- `plots/*`
- `qualitative/video_manifest.csv` (se `--include-qualitative-pack`)
- `qualitative/gifs/*.gif` (se `--include-qualitative-pack`)

## Sequenza tipica end-to-end

```bash
# 1) Baseline quantitativa globale
python -m analysis.run_analysis --only all --no-videos

# 2) Confronto A (curriculum)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension curriculum \
  --algorithm sb3_td3 \
  --reward-type native \
  --reward-behavior monitor_only \
  --rulebook-config selection

# 3) Confronto B (reward)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension reward \
  --algorithm sb3_td3 \
  --curriculum-name stages \
  --rulebook-config selection

# 4) Confronto C (algorithm)
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension algorithm \
  --curriculum-name stages \
  --reward-type rulebook \
  --reward-behavior scalar_reward \
  --rulebook-config selection

# 5) Pacchetto qualitativo curato
python -m analysis.run_analysis --only all --no-videos \
  --comparison-dimension curriculum \
  --algorithm sb3_td3 \
  --reward-type native \
  --reward-behavior monitor_only \
  --rulebook-config selection \
  --include-qualitative-pack
```
