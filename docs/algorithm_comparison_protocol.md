# Algorithm Comparison Protocol

Questo documento definisce come confrontare gli algoritmi in modo scientifico e riproducibile.
Complementa `docs/csv_evaluation_objectives.md`, che invece definisce il formato/logging dei CSV.

## 1) Unita' di analisi

Gerarchia:

- `run` = algoritmo + seed + configurazione + timestamp
- `experiment` = insieme di run dello stesso algoritmo/config su piu' seed
- `comparison` = confronto tra experiment diversi

Regola:

- Una run singola serve per debug/tracciamento.
- Il risultato sperimentale va riportato aggregando su piu' seed.

## 2) Obiettivi del confronto

Il confronto non e' solo "reward piu' alto".
Priorita' della tesi:

1. Safety e compliance al rulebook.
2. Performance di guida/task.
3. Efficienza campionaria e progressione curriculum.
4. Robustezza e stabilita' tra seed/scenari.

## 3) Outcome principali (primary)

Metriche principali per ranking finale:

- `collision_rate` (minore e' meglio)
- `out_of_road_rate` (minore e' meglio)
- `top_rule_violation_rate` / metriche rulebook aggregate (minore e' meglio)
- `success_rate` (maggiore e' meglio)
- `route_completion` (maggiore e' meglio)

Metriche secondarie:

- `mean_reward`, `std_reward`
- `episode_length_mean/std`

Nota:

- In presenza di trade-off, safety/compliance ha precedenza su reward globale.

## 4) Sample efficiency e curriculum progression

Metriche da riportare:

- `steps_to_stage_2`, `steps_to_stage_3`, ..., `steps_to_final_stage`
- `final_stage_reached` (percentuale seed)
- numero evaluation fallite prima della promozione
- stage raggiunto a `total_timesteps`

Scopo:

- Misurare non solo "quanto va bene", ma "quanto rapidamente e stabilmente arriva ai livelli difficili".

## 5) Robustezza / generalizzazione

Valutare, quando disponibile:

- scenari/seed visti durante curriculum
- scenari/seed non visti
- stage finale e/o stress set

Metriche:

- media e variabilita' delle metriche primary
- quantili di coda (es. 5th percentile)
- metriche worst-case/CVaR-like (quando definite formalmente)

## 6) Figure obbligatorie

### Figura 1 - Learning curves principali

Pannelli consigliati (vs `global_step`):

- `success_rate`
- `collision_rate`
- `top_rule_violation_rate` (o metrica rulebook equivalente)
- `route_completion`

Linee = algoritmi, bande = variabilita' tra seed (std, stderr o CI).

### Figura 2 - Progressione curriculum

- `stage_index` vs `global_step` (aggregato sui seed), oppure tabella "steps to stage".

### Figura 3 - Trade-off safety/performance

Scatter consigliato:

- asse X: safety metric (es. `collision_rate` o `Avg EV`)
- asse Y: task metric (es. `success_rate` o `route_completion`)

## 7) Tabelle obbligatorie

### Tabella A - Final evaluation (multi-seed)

Campi minimi:

- `success_rate`
- `collision_rate`
- `out_of_road_rate`
- `top_rule_violation_rate` (o equivalente)
- `route_completion`
- `mean_reward`

Formato:

- Convenzione ufficiale: `mean ± 95% CI`.

### Tabella B - Rulebook compliance

Quando disponibili le metriche V2:

- `Avg EV`
- `Max EV`
- `Counterexample ratio`
- `% violated rules`
- `# unique violation patterns`

Nota semantica:

- `EV` e' derivato dai margini di regola (violazione se `margin < 0`, non-violazione se `margin >= 0`).

### Tabella C - Curriculum/sample efficiency

- `final_stage_reached`
- `steps_to_final_stage`
- `failed_evals_before_promotion`

### Tabella D - Ablation (se presente)

Confronto varianti (es. TD3, scalar-rulebook, lexicographic, distributional, curriculum on/off).

## 8) Regole di aggregazione statistica

- Confrontare algoritmi su identico budget (`total_timesteps`) e protocollo eval.
- Aggregare sempre per seed (non per singole run isolate).
- Definire in anticipo:
  - numero seed
  - metrica di variabilita' (adottata: `95% CI`)
  - eventuale smoothing delle curve (default: nessuno)
- Non cambiare metrica/aggregazione a posteriori tra algoritmi.

### Seed protocol (fissato)

- Numero run per algoritmo nel confronto principale: `10`.
- Lista seed ufficiale (uguale per tutti gli algoritmi):
  - `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Regola di bilanciamento:
  - ogni algoritmo deve avere lo stesso numero di run valide sulla stessa seed list.
- Gestione failure:
  - se una run fallisce, si rilancia la stessa configurazione con lo stesso seed;
  - non si sostituisce il seed con un altro valore.

Regole operative:

- Le curve principali si riportano senza smoothing.
- Eventuale versione smooth e' secondaria, con finestra fissa dichiarata in modo esplicito.

## 9) Run validity e selezione

Una run e' valida per il confronto se:

- contiene `artifacts/run_metadata.yaml` con `status: completed`;
- contiene `csv/final_eval.csv`;
- usa lo stesso budget `total_timesteps` previsto per il confronto;
- usa protocollo eval coerente (`eval_episodes` e `final_eval_episodes`) con il gruppo confrontato;
- non e' una run debug/dev esclusa esplicitamente.

Campi consigliati per filtro automatico (in config/metadata):

```yaml
analysis:
  include_in_comparison: true
  experiment_group: td3_curriculum_v1
```

## 10) Mapping domande -> sorgenti CSV

- Come evolve durante il training: `evals.csv`
- Quando e come promuove: `promotions.csv` (+ gate columns in `evals.csv`)
- Distribuzione episodi eval: `eval_episodes.csv`
- Stato finale run: `final_eval.csv` (V2; in V1.5 derivabile da eval finale)
- Analisi per-regola: `rule_metrics.csv` (V2)

Disponibilita' V2 attuale:

- `evals.csv`: include anche `avg_error_value`, `max_error_value`, `counterexample_rate`, `violated_rules_ratio`, `unique_violation_patterns`.
- `eval_episodes.csv`: include `scenario_seed`, `scenario_id`, `error_value`, `violated_rules`, `violation_pattern`, `video_path` (opzionale, puo' essere `null`).

## 11) Protocollo final evaluation

- Le evaluation intermedie usano `experiment.eval_episodes`.
- La final evaluation usa `experiment.final_eval_episodes` (default consigliato: `100`), separata dalle intermedie.
- `final_eval.csv` contiene una riga per run/seed con le metriche finali ufficiali.

## 12) Analysis outputs richiesti

```text
analysis/aggregated/
  evals_all_runs.csv
  final_eval_all_runs.csv
  promotions_all_runs.csv
  eval_episodes_all_runs.csv
  rule_metrics_all_runs.csv

analysis/tables/
  final_evaluation.(csv|tex)
  rulebook_compliance.(csv|tex)
  curriculum_efficiency.(csv|tex)

analysis/plots/
  learning_curves_success_collision_rule_route.(png|pdf)
  curriculum_progression.(png|pdf)
  safety_performance_tradeoff.(png|pdf)
```

## 13) Reporting statement consigliato

Messaggio finale atteso dal confronto:

> L'algoritmo lessicografico (eventualmente distributional) riduce frequenza e severita' delle violazioni delle regole prioritarie, mantiene performance di guida competitiva, e attraversa il curriculum in modo piu' stabile/efficiente rispetto alle baseline scalarizzate.
