# TD3 SB3 Parity Porting Plan

Obiettivo: rendere `TD3` custom il più possibile aderente a `stable-baselines3`
prima di costruire varianti lessicografiche/distribuzionali.

## Scope

Questo documento riguarda:

- algoritmo `TD3` custom attuale
- differenze confermate rispetto a SB3
- ordine consigliato di porting
- check di validazione da eseguire dopo ogni modifica

Riferimenti esterni:

- SB3 TD3 docs: <https://stable-baselines3.readthedocs.io/en/master/modules/td3.html>
- SB3 repo: <https://github.com/DLR-RM/stable-baselines3>

## Stato attuale nel repo

Implementazione TD3 custom:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Config rilevanti:

- `conf/agent/planner/algorithm/td3.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/td3_sb3.yaml`

## Differenze confermate vs SB3

### 1. Architettura rete

La differenza più grossa è stata la rete:

- setup vecchio: `encoder=mlp` + `decoder=mlp_encoded`
- setup SB3-like: `encoder=none` + `decoder=td3_sb3`

SB3 con osservazioni vettoriali usa, di fatto:

- flatten osservazione
- actor MLP `[400, 300]`
- critic MLP `[400, 300]`

Nel repo ora il decoder SB3-like esiste già:

- `conf/agent/planner/decoder/td3_sb3.yaml`

### 2. Esplorazione

Nel custom attuale:

- warmup casuale uniforme fino a `learning_starts`
- poi sempre rumore gaussiano additivo in `predict()`

File:

- `src/thesis_rl/agent/planners/algorithms/td3.py`

Parametri correnti:

- `action_noise_sigma`
- `learning_starts`

Differenza rispetto a SB3:

- in SB3 l'action noise è un oggetto esplicito (`ActionNoise`)
- il wiring è più pulito e separato dalla policy
- il comportamento può essere equivalente, ma oggi nel custom è inglobato nella `predict()`

### 3. Semantica train/update

Nel custom attuale:

- `train_freq` è intero semplice
- `gradient_steps=auto` viene risolto come `train_freq * n_envs`

File:

- `src/thesis_rl/agent/planners/algorithms/td3.py`

Nota importante:

con `train_freq=1` e `num_envs=4`, questo è vicino al comportamento SB3
quando si vuole un rapporto update/data raccolti pari a 1:1. Quindi questo
non sembra il bug principale.

### 4. Replay buffer

Il buffer custom è volutamente minimale:

- solo `obs/actions/rewards/dones/next_obs`
- niente classi buffer specializzate
- niente ottimizzazioni memoria
- niente supporti aggiuntivi del framework SB3

File:

- `src/thesis_rl/agent/planners/core/buffers.py`

Questo non implica automaticamente un bug, ma è una differenza strutturale.

### 5. Timeout handling

Questo era un punto problematico ed è già stato corretto:

- i timeout non vengono trattati come terminali veri nel replay
- viene usata `terminal_observation` quando disponibile

File:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/agent.py`

### 6. Integrazione planner/lifecycle/agent

Audit svolto:

- l'adapter è identità
- il lifecycle delega senza logica anomala
- il buffer riceve davvero le transizioni
- gli update vengono chiamati correttamente

Conclusione attuale:

- non emerge un bug grossolano nel wiring globale
- se c'è un problema, è più probabile che sia nel dettaglio dell'algoritmo o nella rappresentazione

## Priorità di porting consigliata

Ordine raccomandato:

1. policy/rete
2. esplorazione
3. train/update schedule
4. replay semantics
5. parità di logging/metriche

## Piano operativo

### Fase 1 — rete SB3-faithful

Target:

- usare solo `encoder=none`
- usare `decoder=td3_sb3`
- verificare che actor e critic abbiano effettivamente stack MLP “flat”

Da controllare:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/planners/modules/actor_critic.py`

Esito atteso:

- nessun encoder profondo aggiuntivo
- nessun `layer_norm`
- policy più vicina a SB3

### Fase 2 — esplorazione TD3

Target:

- separare chiaramente:
  - random warmup
  - action noise post-warmup
  - target policy smoothing noise

Da verificare/allineare:

- `action_noise_sigma` nel custom
- semantica del rumore rispetto a SB3
- eventuale possibilità di configurare `action_noise=None`

File:

- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `conf/agent/planner/algorithm/td3.yaml`

Nota:

questa è una delle differenze potenzialmente più influenti sul comportamento.

### Fase 3 — update scheduling

Target:

- controllare che il significato di:
  - `train_freq`
  - `gradient_steps`
  - `policy_delay`
  - `tau`
  - `learning_starts`
  
  sia il più possibile allineato a SB3

Focus:

- non cambiare subito tutto insieme
- mantenere inizialmente `train_freq=1`
- testare con `num_envs=4`

Da verificare:

- se conviene introdurre una semantica più esplicita in stile SB3
- se il valore `auto` debba restare o essere sostituito da un mapping più controllato

### Fase 4 — replay semantics

Target:

- confermare parità sulle transizioni salvate
- confermare parità sui timeout
- confermare parità sul bootstrap con `next_obs`

File:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/td3.py`
- `src/thesis_rl/agent/agent.py`

Questa fase serve soprattutto a evitare discrepanze sottili.

### Fase 5 — parity checks

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
- action magnitude
- update counts
- comportamento visivo

## Implementazione consigliata nella prossima chat

Task suggeriti, in ordine:

1. audit riga per riga di `td3.py` contro SB3 TD3
2. refactor dell'esplorazione per allinearla alla semantica SB3
3. verifica delle config `td3.yaml`
4. lancio debug corto (`50k`/`100k`) con rete `td3_sb3`
5. confronto con i log debug off-policy

## Criterio decisionale

Se, dopo il porting TD3 SB3-faithful:

- TD3 torna a muoversi e apprendere → il problema era nell'implementazione/config custom
- TD3 continua a bloccarsi/timeoutare → il problema va cercato fuori:
  - env
  - reward
  - observation
  - semantics dell'action space

## Comando di riferimento per il test TD3

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_td3_sb3like" \
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
    agent/planner/decoder=td3_sb3 \
    agent/planner/algorithm=td3 \
    analysis.experiment_group=EXP_dbg_td3_sb3like_lidar
```

## Nota finale

Per l'obiettivo tesi, la strategia migliore non è “usare SB3 e basta”, ma:

1. portare il custom a una parità credibile con SB3
2. verificare empiricamente la parità
3. costruire sopra quella base le varianti lessicografiche/distribuzionali

