# CSV Evaluation Objectives

Questo documento raccoglie gli obiettivi di valutazione e logging CSV per le run, con un piano incrementale per evitare incompatibilita' tra esperimenti.

## Obiettivo

Standardizzare l'output in `outputs/.../csv/` per supportare:

- confronto tra algoritmi/reward mode/curriculum
- confronto tra seed
- analisi della progressione curriculum
- analisi distribuzionale degli episodi in evaluation
- estendibilita' verso metriche rulebook avanzate

## Principi

- Colonne identificative sempre presenti: `algorithm`, `reward_mode`, `curriculum_name`, `seed`, `run_id`, `stage`, `stage_index`, `global_step`.
- Schema stabile e ordinato (fieldnames fissi).
- Nessuna metrica "inventata": ogni colonna deve avere definizione e sorgente dati chiara nel codice.
- Backward compatibility: nuove colonne solo additive.

## Granularita' dei CSV

La granularita' dei CSV principali non e' step-by-step.

I grafici tipo `metric vs timesteps` vanno interpretati cosi':

- `x = global_step` (interazioni ambiente accumulate)
- `y = metrica aggregata` della policy valutata a quel `global_step`

Nel nostro loop (`train chunk -> evaluation -> gate check -> eventual promotion`), ogni CSV ha granularita' specifica:

- `train_chunks.csv`: 1 riga = 1 chunk di training.
- `evals.csv`: 1 riga = 1 evaluation aggregata (su N episodi).
- `eval_episodes.csv`: 1 riga = 1 episodio di evaluation.
- `promotions.csv`: 1 riga = 1 evento curriculum rilevante (es. `promoted`).
- `rule_metrics.csv` (V2): 1 riga = 1 regola valutata in 1 evaluation.
- `final_eval.csv` (V2): 1 riga = 1 final evaluation della run.

Non usare i CSV principali per salvare `1 riga = 1 env step`.
Il livello step-by-step resta materiale diagnostico/debug e va salvato separatamente (log JSONL, trajectory dump, ecc.).

## Fase V1.5 (implementazione prioritaria)

### File CSV previsti

- `train_chunks.csv`
- `evals.csv`
- `promotions.csv`
- `eval_episodes.csv`

### 1) `train_chunks.csv` (una riga per chunk)

Scopo: monitoraggio del training a blocchi (`chunk -> eval`).

Colonne V1.5:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,chunk_id,stage,stage_index,steps_start,steps_end,global_step,chunk_steps,episodes,ep_rew_mean,ep_len_mean,ep_success_rate,ep_collision_rate,ep_out_of_road_rate,ep_route_completion_mean,actor_loss,critic_loss,learning_rate,n_updates,fps,elapsed_seconds
```

Note:

- Le metriche `ep_success_rate`, `ep_collision_rate`, `ep_out_of_road_rate`, `ep_route_completion_mean` sono aggregate per episodio all'interno del chunk.
- Se nel chunk non termina alcun episodio, questi campi possono risultare null.

### 2) `evals.csv` (una riga per evaluation aggregata)

Scopo: curva principale di performance e gate curriculum.

Colonne V1.5:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,eval_id,chunk_id,stage,stage_index,global_step,eval_episodes,deterministic,mean_reward,std_reward,mean_env_reward,std_env_reward,mean_scalar_rule_reward,std_scalar_rule_reward,mean_rule_saturation_max,collision_rate,collision_rate_std,out_of_road_rate,success_rate,success_rate_std,route_completion,top_rule_violation_rate,success_rate_min,collision_rate_max,out_of_road_rate_max,top_rule_violation_rate_max,route_completion_min,gate_success_pass,gate_collision_pass,gate_out_of_road_pass,gate_top_rule_pass,gate_route_completion_pass,passed_eval_gates,consecutive_passes,warmup_evals_required,consecutive_evals_required,promoted,next_stage
```

Note:

- I gate separati evitano l'ambiguita' di un solo `passed_eval_gates`.
- Le soglie usate vengono salvate per mantenere interpretabilita' storica delle run.

### 3) `promotions.csv` (solo eventi promozione)

Scopo: tracciare avanzamento stage e tempo alla promozione.

Colonne V1.5:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,event_type,from_stage,to_stage,from_stage_index,to_stage_index,eval_id,chunk_id,global_step,stage_steps_done,stage_steps_min_required,passed_eval_gates,consecutive_passes,success_rate,collision_rate,out_of_road_rate,top_rule_violation_rate,route_completion,reason
```

Linee guida:

- Scrivere solo eventi reali di promozione (`event_type=promoted`).
- I gate falliti restano in `evals.csv` (niente duplicazione).

### 4) `eval_episodes.csv` (una riga per episodio di evaluation)

Scopo: analisi distribuzionale, worst-case, best/median/worst episode.

Colonne V1.5:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,eval_id,episode_id,stage,stage_index,global_step,deterministic,reward,env_reward,scalar_rule_reward,rule_rewards_by_rule,episode_length,success,collision,out_of_road,timeout,route_completion,top_rule_violation_rate
```

Note:

- In V1.5 non imponiamo ancora `scenario_id/scenario_seed/video_path` perche' non sono stabilizzati nel payload episodio corrente.
- La colonna `top_rule_violation_rate` per episodio e' derivata dai dati per-episodio gia' disponibili in `Agent.evaluate(..., return_episode_metrics=True)`.
- `reward` = reward finale usato dall'agente in eval (dopo eventuale wrapping rulebook).
- `env_reward` = reward nativo MetaDrive (sempre tracciato).
- `scalar_rule_reward` = contributo scalarizzato rulebook (presente quando disponibile nel wrapper rulebook).
- `rule_rewards_by_rule` = JSON per episodio con contributo/margine cumulato per ciascuna regola.

Colonne V2 aggiunte:

```csv
scenario_seed,scenario_id,error_value,violated_rules,violation_pattern,video_path
```

Note V2:

- `scenario_seed` e' il seed effettivo usato in `env.reset(...)` per l'episodio.
- `scenario_id` e' derivato deterministico nel formato `seed_<scenario_seed>`.
- `video_path` e' opzionale: attualmente puo' essere `null` se non e' attiva la registrazione video per episodio.

## Fase V2 (estensioni implementate / richieste per analisi finale)

### File aggiuntivi

- `rule_metrics.csv`
- `final_eval.csv`

### Metriche avanzate candidate

- `avg_error_value`
- `max_error_value`
- `counterexample_rate`
- `violated_rules_ratio`
- `unique_violation_patterns`

Definizioni concordate:

- Il margine di regola e' il segnale base. Convenzione: violazione se `margin < 0`, soddisfazione/non-violazione se `margin >= 0`.
- `error_value` e' derivato dai margini negativi (nessuna metrica scollegata dal margine).
- A livello episodio:
  - `EV_episode = sum_i (w_i * max(0, -margin_i_min_episode))`
  - `margin_i_min_episode` = peggior margine osservato nell'episodio per la regola `i`
  - `w_i` = peso di priorita' (monotono rispetto alla priorita', da implementare in modo coerente con il rulebook attivo)
- A livello evaluation:
  - `avg_error_value = mean(EV_episode)`
  - `max_error_value = max(EV_episode)`

Rappresentazione violazioni per regola:

- In `rule_metrics.csv` (formato long): una riga per (`eval_id`, `rule_name`) con almeno:
  - `violated` (bool aggregato),
  - `violation_rate`,
  - statistiche margine (`mean`, `min`, `max`).
- In `eval_episodes.csv` (V2):
  - `violated_rules` come stringa canonica ordinata per priorita' (tie-break alfabetico),
  - `violation_pattern` (stessa serializzazione),
  - valore `none` quando nessuna regola e' violata.

Identificatori scenario/video (V2):

- `scenario_seed`: seed effettivo usato nel `env.reset(...)`.
- `scenario_id`: derivato deterministico (`seed_<scenario_seed>`).
- `video_path`: opzionale (`null` se video non salvato).

Stato implementazione V2:

- `rule_metrics.csv`: implementato.
- `final_eval.csv`: implementato.
- `evals.csv`/`final_eval.csv`: includono tripletta reward aggregata (`mean_reward`, `mean_env_reward`, `mean_scalar_rule_reward` e rispettive std; i campi rulebook possono essere null fuori da `reward.mode=rulebook`).
- Colonne V2 aggregate in `evals.csv` (`avg_error_value`, `max_error_value`, `counterexample_rate`, `violated_rules_ratio`, `unique_violation_patterns`): implementate.
- Colonne V2 episodio in `eval_episodes.csv` (`scenario_seed`, `scenario_id`, `error_value`, `violated_rules`, `violation_pattern`, `video_path`): implementate.

## Analysis pipeline (post-run)

Pipeline consigliata per trasformare CSV di run in risultati comparabili:

```text
analysis/
  aggregate_runs.py
  make_final_tables.py
  make_learning_curves.py
  make_curriculum_plots.py
  make_rulebook_tables.py
  make_tradeoff_plots.py
```

Output attesi:

```text
analysis/aggregated/
  evals_all_runs.csv
  final_eval_all_runs.csv
  promotions_all_runs.csv
  eval_episodes_all_runs.csv
  rule_metrics_all_runs.csv
```

Note:

- L'aggregazione multi-run e' prerequisito per plot/tabelle robusti.
- Le tabelle/figure del protocollo vanno generate sui file `*_all_runs.csv`, non su singole run.

## Decisioni di protocollo collegate a V2

- Report tabellare: `mean ± 95% CI` come convenzione ufficiale.
- Curve principali: senza smoothing in prima istanza.
- Smoothing opzionale solo in analisi successive e sempre dichiarato esplicitamente (finestra fissa).
- Final evaluation separata e robusta:
  - evaluation intermedie: `experiment.eval_episodes`
  - final evaluation: `experiment.final_eval_episodes` (default consigliato: `100`)

## Criteri di accettazione V1.5

- Ogni run train produce i 4 CSV in `paths.csv_dir`.
- Gli header sono stabili tra run diverse.
- Le righe di `evals.csv` sono allineate 1:1 alle evaluation eseguite.
- Le righe di `promotions.csv` sono allineate 1:1 alle promozioni reali.
- `eval_episodes.csv` contiene `eval_episodes` righe per ogni evaluation.
