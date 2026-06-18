# SAC SB3 Parity Porting Plan

Obiettivo: rendere `SAC` custom il più possibile aderente a `stable-baselines3`
prima di costruire varianti lessicografiche/distribuzionali.

## Scope

Questo documento riguarda:

- algoritmo `SAC` custom attuale
- differenze confermate rispetto a SB3
- ordine consigliato di porting
- check di validazione da eseguire dopo ogni modifica

Riferimenti esterni:

- SB3 SAC docs: <https://stable-baselines3.readthedocs.io/en/master/modules/sac.html>
- SB3 repo: <https://github.com/DLR-RM/stable-baselines3>

## Stato attuale nel repo

Implementazione SAC custom:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Config rilevanti:

- `conf/agent/planner/algorithm/sac.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/sac_sb3.yaml`

## Differenze confermate vs SB3

### 1. Architettura rete

La differenza più grossa iniziale è stata la rete:

- setup vecchio: `encoder=mlp` + `decoder=mlp_encoded`
- setup SB3-like: `encoder=none` + `decoder=sac_sb3`

SB3 con osservazioni vettoriali usa, di fatto:

- flatten osservazione
- actor MLP `[256, 256]`
- critic MLP `[256, 256]`

Nel repo ora il decoder SB3-like esiste già:

- `conf/agent/planner/decoder/sac_sb3.yaml`

### 2. Policy stocastica e log-prob

Nel custom attuale:

- actor gaussiano con `tanh`
- correzione del `log_prob` sullo squash
- `log_std` state-dependent

File:

- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/algorithms/sac.py`

Questo è concettualmente vicino a SB3, ma va verificato riga per riga per
allineare sampling, log-prob, bounds e inizializzazione.

### 3. Temperatura / `alpha`

Nel custom attuale:

- `ent_coef=auto`
- `target_entropy=auto`
- update di `log_alpha` separato

File:

- `src/thesis_rl/agent/planners/algorithms/sac.py`

Questo è vicino a SB3, ma va controllato bene:

- inizializzazione
- formula dell'update
- velocità di collasso di `alpha`

### 4. Semantica train/update

Nel custom attuale:

- `train_freq` è intero semplice
- `gradient_steps=auto` viene risolto come `train_freq * n_envs`

File:

- `src/thesis_rl/agent/planners/algorithms/sac.py`

Con `train_freq=1` e `num_envs=4`, il comportamento è vicino a SB3 quando si
vuole un rapporto 1:1 tra update e dati raccolti, ma non è una replica
generale dell'intera semantica SB3.

### 5. Replay buffer

Il buffer custom è minimale:

- solo `obs/actions/rewards/dones/next_obs`
- niente varianti buffer del framework SB3
- niente ottimizzazioni memoria o supporti aggiuntivi

File:

- `src/thesis_rl/agent/planners/core/buffers.py`

Non è necessariamente un bug, ma è una differenza strutturale.

### 6. Timeout handling

Questo punto è già stato corretto:

- i timeout non vengono trattati come terminali veri nel replay
- viene usata `terminal_observation` quando disponibile

File:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/agent.py`

### 7. Integrazione planner/lifecycle/agent

Audit svolto:

- adapter identità
- lifecycle senza logica anomala
- buffer popolato correttamente
- update chiamati correttamente

Conclusione attuale:

- non emerge un bug grossolano nel wiring globale
- il problema, se presente, è più probabile nel dettaglio dell'algoritmo o nella rappresentazione

## Priorità di porting consigliata

Ordine raccomandato:

1. policy/rete
2. sampling e log-prob
3. temperatura `alpha`
4. train/update schedule
5. replay semantics
6. parità di logging/metriche

## Piano operativo

### Fase 1 — rete SB3-faithful

Target:

- usare solo `encoder=none`
- usare `decoder=sac_sb3`
- verificare che actor e critic abbiano stack MLP “flat”

Da controllare:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Esito atteso:

- nessun encoder profondo aggiuntivo
- nessun `layer_norm`
- policy più vicina a SB3

### Fase 2 — sampling e log-prob

Target:

- verificare `Normal -> rsample -> tanh`
- verificare la correzione del `log_prob`
- verificare la distinzione tra azione stocastica e deterministica

Da verificare/allineare:

- `sample()`
- `evaluate()`
- inizializzazione `log_std`
- bounds di `log_std`

File:

- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Nota:

qui una discrepanza piccola può alterare molto l'apprendimento.

### Fase 3 — temperatura `alpha`

Target:

- allineare il comportamento di `ent_coef=auto`
- verificare `target_entropy = -action_dim`
- verificare il learning update di `log_alpha`

File:

- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `conf/agent/planner/algorithm/sac.yaml`

Segnali da monitorare:

- `alpha`
- `logp`
- action magnitude

### Fase 4 — update scheduling

Target:

- controllare che il significato di:
  - `train_freq`
  - `gradient_steps`
  - `tau`
  - `learning_starts`
  
  sia il più possibile coerente con SB3

Focus:

- non cambiare troppe variabili insieme
- mantenere inizialmente `train_freq=1`
- testare con `num_envs=4`

### Fase 5 — replay semantics

Target:

- confermare parità sulle transizioni salvate
- confermare parità sui timeout
- confermare parità sul bootstrap con `next_obs`

File:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/sac.py`
- `src/thesis_rl/agent/agent.py`

### Fase 6 — parity checks

Target:

- stessa observation config
- stessa reward config
- stessa rete
- stessi hyperparameter
- stessi seed

Confronti minimi:

- andamento `ep_rew_mean`
- `route_completion`
- `success_rate`
- `alpha`
- action magnitude
- update counts
- comportamento visivo

## Implementazione consigliata nella prossima chat

Task suggeriti, in ordine:

1. audit riga per riga di `sac.py` contro SB3 SAC
2. verifica di `SquashedGaussianActor`
3. refactor di `alpha`/`ent_coef=auto` se necessario
4. verifica delle config `sac.yaml`
5. lancio debug corto (`50k`/`100k`) con rete `sac_sb3`
6. confronto con i log debug off-policy

## Criterio decisionale

Se, dopo il porting SAC SB3-faithful:

- SAC torna a muoversi e apprendere → il problema era nell'implementazione/config custom
- SAC continua a bloccarsi/timeoutare → il problema va cercato fuori:
  - env
  - reward
  - observation
  - semantics dell'action space

## Comando di riferimento per il test SAC

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_sac_sb3like" \
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
    agent/planner/decoder=sac_sb3 \
    agent/planner/algorithm=sac \
    analysis.experiment_group=EXP_dbg_sac_sb3like_lidar
```

## Nota finale

Per l'obiettivo tesi, la strategia migliore è:

1. portare il custom a una parità credibile con SB3
2. verificare empiricamente la parità
3. costruire sopra quella base le varianti lessicografiche/distribuzionali

