# Thesis Experiment Roadmap

Questa nota fissa la sequenza operativa corrente per arrivare in modo pulito a:

- reward rulebook vettoriale
- reward rulebook scalarizzato
- baseline scalar forte sul task finale
- algoritmi lessicografici e distribuzionali

## Decisioni gia' chiuse

- `env.vectorized.num_envs=5` resta il default operativo corrente
- il vero obiettivo sperimentale non e' piu' la sola baseline `native`, ma il
  task finale `rulebook-aware`
- i `termination flags` vanno scelti nel contesto del reward finale, non
  separatamente e troppo presto
- la baseline `native` resta utile come riferimento/controllo, ma non deve
  guidare da sola la scelta finale dell'algoritmo

## Contract sperimentali

### Native reference contract

Uso:

- baseline di riferimento per il report
- controllo storico rispetto al reward nativo MetaDrive

Nota:

- non e' piu' la pipeline principale per decidere il backbone finale della tesi

### Rulebook-aware contract

Uso:

- scelta dei termination flags coerenti col framework finale
- selezione algoritmo
- tuning
- osservazione
- encoder
- curriculum
- baseline scalar finale
- confronto con lessicografici e distribuzionali

Principio:

- se una violazione deve essere trattata come preferenza graduata o regola
  penalizzabile, va lasciata vivere nel reward/rulebook e non trasformata
  troppo presto in `done=True`

## Principio generale

Non trattare tutto come una grid search unica. La procedura corretta e':

1. implementare il reward/rulebook finale
2. chiudere il task/environment contract del problema finale
3. selezionare l'algoritmo sul task finale
4. fare tuning dell'algoritmo scelto
5. selezionare osservazione ed encoder
6. valutare augmentations algorithm-level
7. introdurre e valutare il curriculum
8. usare la migliore baseline scalar rulebook-aware come riferimento per
   lessicografici e distribuzionali

Regola metodologica:

- se cambi reward o termination flags in modo sostanziale, stai cambiando task
- quando cambia il task, le decisioni a monte piu' sensibili vanno
  riconfermate, soprattutto l'algoritmo

## Fase 0 - Smoke tecnico

Budget:

- `run_profile=smoke`
- `seed=0`

Scopo:

- verificare che ogni nuova config parta, logghi e chiuda correttamente

Blocca:

- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Guarda:

- crash
- NaN
- artefatti mancanti
- `final_eval.csv` presente

## Fase 1 - Implementazione e validazione del rulebook reward

Scopo:

- definire le metriche/violazioni del rulebook
- produrre reward vettoriale consistente
- produrre una prima scalarizzazione usabile dagli algoritmi scalar

Budget:

- smoke tecnico: `run_profile=smoke`, `1` seed
- debug funzionale: `run_profile=fast`, `1` seed

Varia:

- implementazione delle componenti del vettore reward
- funzione di scalarizzazione

Blocca:

- algoritmo sonda: `SAC`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- termination flags iniziali di debug, preferibilmente una variante
  `strict-like`

Guarda:

- correttezza semantica delle componenti reward
- saturazioni strane o segni incoerenti
- correlazione qualitativa tra eventi osservati e penalty/reward
- assenza di metriche palesemente rotte

Nota:

- se una metrica appare sospetta, come `goal_progress violation` sempre a `1`,
  sistemarla qui prima di usarla per decidere altro

## Fase 2 - Chiusura del task contract rulebook-aware

Scopo:

- scegliere i termination flags coerenti col reward finale
- decidere se il task finale deve essere `strict`, `relaxed` o una variante
  intermedia

Budget:

- screening iniziale: `run_profile=fast`, `1` seed
- chiusura pulita: `run_profile=fast`, `3` seed

Config consigliata:

- algoritmo sonda: `SAC`
- `reward=scalar_reward`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Varia:

- `strict`
- `relaxed`
- eventualmente una variante intermedia se emerge un dubbio preciso

Blocca:

- algoritmo
- hyperparam
- osservazione
- encoder
- scalarizzazione

Guarda:

- learnability
- stabilita'
- degenerazioni tipo agente fermo
- `collision_rate`
- `out_of_road_rate`
- `route_completion`
- `success_rate`
- violazioni per regola

Interpretazione corretta:

- se `strict` riduce collisioni per censura dell'episodio, non basta da solo
  per dichiararlo migliore
- se `relaxed` lascia emergere trade-off utili al rulebook senza distruggere la
  learnability, e' un candidato forte per il task finale

Output atteso:

- un `rulebook-aware contract` fissato e usato in tutte le fasi successive

## Fase 3 - Selezione algoritmo sul task finale

Budget:

- qualification: `3` seed (`0,1,2`)
- profilo: preset di selezione o equivalente con
  `experiment.eval_interval=50000`,
  `experiment.eval_episodes=20`,
  `experiment.final_eval_episodes=100`

Config consigliata:

- `reward=scalar_reward`
- `curriculum=disabled`
- `obs=lidar_state`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`

Varia:

- `td3_sb3`
- `sac_sb3`
- `ppo_sb3`

Blocca:

- task contract chiuso in Fase 2
- scalarizzazione fissata
- decoder coerente con l'algoritmo
- nessun tuning serio in questa fase

Guarda per decidere:

- `collision_rate`
- `out_of_road_rate`
- `top_rule_violation_rate`
- `success_rate`
- `route_completion`
- stabilita' tra seed

Regola:

- compliance e safety vengono prima del reward medio
- se un algoritmo e' nettamente peggiore, scartalo
- se due sono vicini, portali entrambi alla conferma successiva

## Fase 4 - Conferma algoritmo

Budget:

- `5-10` seed sui migliori `1-2` algoritmi
- stesso protocollo della fase 3

Varia:

- solo l'algoritmo tra i candidati rimasti

Blocca:

- tutto il resto

Guarda:

- medie finali
- `95% CI`
- stabilita' delle curve
- robustezza tra seed

Output atteso:

- scelta dell'algoritmo baseline ufficiale per il task rulebook-aware

## Fase 5 - Tuning dell'algoritmo scelto

Budget consigliato:

- `run_profile=tune`
- target suggerito:
  - `total_timesteps` circa `400k-600k`
  - `eval_interval=25000`
  - `eval_episodes=20`
  - `final_eval_episodes=50`
- `3` seed per trial
- `8-12` trial di random search

Varia:

- pochi iperparametri ad alto impatto

Parametri consigliati:

- `PPO`: `learning_rate`, `n_steps`, `batch_size`, `ent_coef`
- `SAC`: `learning_rate`, `batch_size`, `learning_starts`, `buffer_size`
- `TD3`: `learning_rate`, `batch_size`, `learning_starts`,
  `action_noise_sigma`

Blocca:

- reward
- task contract
- osservazione
- encoder
- curriculum

Guarda:

- ranking safety-first
- stabilita' tra seed
- sample efficiency

Conferma:

- top `2` setting con budget piu' lungo e `5` seed
- top `1` setting con `10` seed solo se ti serve maggiore rigore

## Fase 6 - Selezione osservazione

Budget:

- screening: `run_profile=long`, `3` seed
- conferma: `run_profile=thesis`, `5-10` seed

Varia:

- `obs=lidar_state`
- `obs=semantic_state`

Blocca:

- algoritmo gia' scelto e tuned
- `reward=scalar_reward`
- `curriculum=disabled`
- `agent/planner/encoder=none`
- `env.vectorized.num_envs=5`
- task contract chiuso

Guarda:

- safety/performance
- stabilita' di training
- robustezza tra seed

## Fase 7 - Selezione encoder

Nota:

- questa fase ha senso soprattutto se `semantic_state` resta competitivo

Budget:

- screening: `run_profile=long`, `3` seed
- conferma: `run_profile=thesis`, `5-10` seed

Varia:

- `encoder=none`
- `encoder=mlp`
- `encoder=lq`

Blocca:

- algoritmo tuned
- osservazione fissata
- reward scalarizzato
- curriculum disabled
- decoder coerente con la pipeline encoder scelta
- task contract chiuso

Guarda:

- safety/performance
- sample efficiency
- varianza tra seed

## Fase 8 - Augmentations algorithm-level

Scopo:

- valutare modifiche interne all'algoritmo dopo aver gia' fissato backbone,
  hyperparam principali, osservazione ed encoder

Ordine consigliato:

1. replay buffer / sampling strategy
2. temporal credit assignment / traces-like variants

### Fase 8a - Replay buffer / prioritized sampling

Nota:

- questa sottofase ha senso soprattutto per algoritmi off-policy, quindi in
  pratica `SAC` e `TD3`
- per `PPO` il replay prioritizzato non e' la leva naturale

Budget:

- screening: `run_profile=long`, `3` seed
- conferma: `run_profile=thesis`, `5-10` seed

Varia:

- `uniform replay`
- `prioritized replay`

Blocca:

- algoritmo scelto
- hyperparam scelti
- osservazione
- encoder
- reward scalarizzato
- curriculum disabled
- task contract chiuso

Guarda:

- safety/performance
- sample efficiency
- robustezza tra seed
- eventuale instabilita' numerica introdotta dal replay prioritizzato

### Fase 8b - Eligibility traces / credit assignment

Nota:

- su `PPO` la leva naturale e' spesso `gae_lambda`
- su `SAC` e `TD3` le traces non sono una modifica plug-and-play

Budget:

- screening: `run_profile=long`, `3` seed
- conferma: `run_profile=thesis`, `5` seed

Varia:

- baseline credit assignment
- traces-like variant o tuning di `lambda`/credit assignment

Blocca:

- tutto il resto

Guarda:

- sample efficiency
- stabilita'
- guadagno reale sulle metriche finali

## Fase 9 - Curriculum

Budget:

- screening: `run_profile=thesis`, `3` seed
- conferma: `run_profile=thesis`, `5-10` seed

Varia:

- `curriculum=disabled`
- `curriculum=staged` base

Blocca:

- algoritmo
- hyperparam
- osservazione
- encoder
- reward scalarizzato
- task contract chiuso

Guarda:

- metriche finali
- `steps_to_final_stage`
- `final_stage_reached`
- `failed_evals_before_promotion`

Solo se il curriculum aiuta davvero:

- tuni soglia di promozione
- numero di eval consecutive
- definizione degli stage

## Fase 10 - Baseline native di riferimento

Scopo:

- mantenere una baseline `native` confrontabile nel report finale

Budget:

- non prioritaria
- `run_profile=fast` o `thesis`
- `3` seed per un primo controllo

Config consigliata:

- `reward=monitor_only`
- contract `strict` come baseline di riferimento nativa
- stesso algoritmo migliore trovato sul task finale, oppure `SAC` come sonda

Nota:

- questa fase serve come controllo/reporting, non per scegliere il backbone
  finale del framework

## Fase 11 - Algoritmi lessicografici e distribuzionali

Budget:

- smoke e fast per debug
- confronto ufficiale con budget `thesis`
- almeno `5` seed, meglio `10` seed

Baseline di riferimento:

- la migliore baseline scalar rulebook-aware trovata nelle fasi precedenti

Obiettivo:

- valutare il guadagno reale rispetto a una baseline scalar forte e coerente
  con il task finale

## Note pratiche su GPU e parallelismo

- `num_envs=5` non garantisce da solo stabilita' se la GPU e' condivisa
- per encoder pesanti conta molto anche il numero di seed e di gruppi lanciati
  in parallelo
- se compare OOM, riduci prima il parallelismo tra run e solo dopo considera
  ulteriori tagli a `num_envs`
- per i confronti con encoder pesanti, non lanciare molti gruppi insieme
