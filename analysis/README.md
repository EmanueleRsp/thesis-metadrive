# Analysis Pipeline

Questa cartella contiene gli script post-run per aggregare risultati multi-seed e generare tabelle/plot.

## Script disponibili

- `run_analysis.py`: orchestratore unico (`aggregate + tables + plots`).
- `aggregate_runs.py`: unisce i CSV prodotti nelle run in file aggregati `*_all_runs.csv`.
- `make_final_tables.py`: genera tabella finale `mean ± 95% CI` in `csv` e `md`.
- `make_curriculum_tables.py`: genera tabella curriculum/sample-efficiency in `csv` e `md`.
- `make_rulebook_tables.py`: genera tabelle rulebook compliance globali e per-regola in `csv` e `md`.
- `make_plots.py`: genera i grafici comparativi principali in formato immagine (`.png`).
- `select_video_episodes.py`: seleziona episodi rappresentativi da `eval_episodes.csv` e crea `video_selection.json` + `video_index.csv`.
- `render_selected_videos.py`: replay offline su episodi selezionati, render GIF in `videos/final_eval/` e verifica fidelity (`original_*` vs `replay_*`) in `video_index.csv`.

## Uso rapido

```powershell
python -m analysis.run_analysis
```

Parametri principali:

- `--outputs-root`: root delle run (default: `outputs`)
- `--analysis-root`: root output analisi (default: `analysis`)
- `--only`: `all | aggregate | tables | plots`
- `--no-videos`: disabilita la fase video (in `--only all` i video sono abilitati di default)
- `--video-max`: numero massimo di video selezionati per run (default 5)
- `--seed-list`: lista seed ufficiale (default `0..9`)
- `--total-timesteps`, `--eval-episodes`, `--final-eval-episodes`: filtri protocollo opzionali

## Output

```text
analysis/aggregated/
  train_chunks_all_runs.csv
  evals_all_runs.csv
  eval_episodes_all_runs.csv
  promotions_all_runs.csv
  rule_metrics_all_runs.csv
  final_eval_all_runs.csv

analysis/tables/
  final_evaluation.csv
  final_evaluation.md
  curriculum_efficiency.csv
  curriculum_efficiency.md
  rulebook_compliance.csv
  rulebook_compliance.md
  rule_violation_by_rule.csv
  rule_violation_by_rule.md

analysis/plots/
  learning_success_vs_global_step.png
  learning_collision_vs_global_step.png
  learning_out_of_road_vs_global_step.png
  learning_route_completion_vs_global_step.png
  learning_rule_top_violation_vs_global_step.png
  learning_avg_error_value_vs_global_step.png
  curriculum_stage_index_vs_global_step.png
  safety_performance_tradeoff.png
  rule_metrics_violation_rate_by_rule.png
  episode_error_distribution_boxplot.png

videos/metadata/ (per singola run)
  video_selection.json
  video_index.csv

videos/final_eval/ (per singola run)
  eval_XXXX_ep_XXXX_<tag>.gif
```

Checkpoint/resume artifacts (per singola run):

- `checkpoints/latest.zip`
- `checkpoints/latest_replay_buffer.pkl`
- `checkpoints/latest_training_state.yaml`
- `checkpoints/latest_rng_state.pkl`

## Video Replay Fidelity

La pipeline video e pensata in due fasi:

1. `select_video_episodes.py` seleziona gli episodi rappresentativi da `eval_episodes.csv`.
2. `render_selected_videos.py` fa replay deterministico (checkpoint + `scenario_seed`) e genera le GIF.

Durante il render aggiorna:

- `videos/metadata/video_index.csv` con metadati e confronto `original` vs `replay`.
- `csv/eval_episodes.csv` impostando `video_path` per gli episodi renderizzati.

Campi fidelity principali in `video_index.csv`:

- `original_reward`, `replay_reward`, `reward_abs_diff`
- `original_route_completion`, `replay_route_completion`, `route_completion_abs_diff`
- `original_error_value`, `replay_error_value`, `error_value_abs_diff`
- `original_success/collision/out_of_road`, `replay_*`, `*_match`
- `replay_match`

Regola corrente per `replay_match=true`:

- match esatto dei booleani (`success`, `collision`, `out_of_road`)
- `reward_abs_diff <= 1e-2`
- `route_completion_abs_diff <= 1e-3`
- `error_value_abs_diff <= 1e-3`

Se il replay non matcha, viene stampato warning (`[video][warn] replay mismatch ...`), ma il video viene comunque salvato.

Checkpoint di replay:

- configurabile da `conf/video/default.yaml` con `video.replay_checkpoint`.
- valori supportati: `final`, `latest`, `best_lexicographic`, oppure un path esplicito.
- default consigliato/protocollo: `final`.
