# PPO SB3 Parity Porting Plan

Obiettivo: rendere `PPO` custom il più possibile aderente a `stable-baselines3`
prima di costruire varianti lessicografiche/distribuzionali.

## Scope

Questo documento riguarda:

- algoritmo `PPO` custom attuale
- differenze confermate rispetto a SB3
- ordine consigliato di porting
- check di validazione da eseguire dopo ogni modifica

Riferimenti esterni:

- SB3 PPO docs: <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>
- SB3 repo: <https://github.com/DLR-RM/stable-baselines3>

## Stato attuale nel repo

Implementazione PPO custom:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/core/lifecycle.py`
- `src/thesis_rl/agent/agent.py`

Config rilevanti:

- `conf/agent/planner/algorithm/ppo.yaml`
- `conf/agent/planner/encoder/none.yaml`
- `conf/agent/planner/decoder/ppo_sb3.yaml`

## Differenze confermate vs SB3

### 1. Architettura rete

La differenza più grossa iniziale è stata la rete:

- setup vecchio: `encoder=mlp` + `decoder=mlp_encoded`
- setup SB3-like: `encoder=none` + `decoder=ppo_sb3`

SB3 PPO con osservazioni vettoriali usa, di fatto:

- flatten osservazione
- MLP policy/value più piccola
- tipicamente `[64, 64]`

Nel repo ora il decoder SB3-like esiste già:

- `conf/agent/planner/decoder/ppo_sb3.yaml`

### 2. Politica gaussiana

Nel custom attuale:

- policy gaussiana non squashata
- env action ottenuta con clipping ai bound
- buffer action = raw action

File:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Questo è stato già avvicinato a SB3 ed è concettualmente corretto, ma va
verificato su tutti i dettagli.

### 3. Rollout buffer

Il rollout buffer custom è minimale:

- `obs/actions/rewards/dones/values/log_probs`
- GAE custom
- iterazione minibatch custom

File:

- `src/thesis_rl/agent/planners/core/buffers.py`

Non è necessariamente sbagliato, ma è una differenza strutturale rispetto
all'infrastruttura SB3.

### 4. Timeout handling

Questo punto è già stato corretto:

- bootstrap del reward sui timeout usando il valore della `terminal_observation`

File:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `src/thesis_rl/agent/agent.py`

### 5. Update scheduling

Nel custom attuale:

- update quando il rollout buffer è pieno
- `n_epochs`, `batch_size`, `clip_range`, `clip_range_vf`, `target_kl`
- `normalize_advantage`

File:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Questa parte è già abbastanza vicina a SB3, ma va controllata riga per riga.

### 6. Integrazione planner/lifecycle/agent

Audit svolto:

- adapter identità
- lifecycle senza logica anomala
- rollout buffer popolato correttamente
- update chiamati correttamente

Conclusione attuale:

- non emerge un bug grossolano nel wiring globale
- se c'è un problema, è più probabile nel dettaglio della policy PPO o nel setup di training

## Priorità di porting consigliata

Ordine raccomandato:

1. policy/value network
2. action/log-prob handling
3. rollout/GAE semantics
4. update loop
5. parità di logging/metriche

## Piano operativo

### Fase 1 — rete SB3-faithful

Target:

- usare solo `encoder=none`
- usare `decoder=ppo_sb3`
- verificare che policy e value path siano coerenti con SB3

Da controllare:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Esito atteso:

- nessun encoder profondo aggiuntivo
- rete compatta stile MLP policy SB3

### Fase 2 — action e log-prob

Target:

- verificare distinzione tra:
  - `raw_action`
  - env action clippata
  - action salvata nel buffer
- verificare calcolo `log_prob`

File:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`

Nota:

qui una discrepanza altera direttamente il surrogate loss.

### Fase 3 — rollout e GAE

Target:

- verificare il GAE step-by-step
- verificare `last_values`
- verificare `last_dones`
- verificare bootstrap su timeout

File:

- `src/thesis_rl/agent/planners/core/buffers.py`
- `src/thesis_rl/agent/planners/algorithms/ppo.py`

### Fase 4 — update loop

Target:

- verificare:
  - `n_steps`
  - `batch_size`
  - `n_epochs`
  - `clip_range`
  - `clip_range_vf`
  - `target_kl`
  - `normalize_advantage`
  - `max_grad_norm`

File:

- `src/thesis_rl/agent/planners/algorithms/ppo.py`
- `conf/agent/planner/algorithm/ppo.yaml`

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
- `approx_kl`
- action magnitude
- update counts
- comportamento visivo

## Implementazione consigliata nella prossima chat

Task suggeriti, in ordine:

1. audit riga per riga di `ppo.py` contro SB3 PPO
2. verifica di action/raw_action/log_prob
3. verifica del rollout buffer e GAE
4. verifica delle config `ppo.yaml`
5. lancio debug corto (`50k`/`100k`) con rete `ppo_sb3`
6. confronto tra metriche custom e attese

## Criterio decisionale

Se, dopo il porting PPO SB3-faithful:

- PPO torna a produrre comportamento sensato → il problema era nell'implementazione/config custom
- PPO continua a essere troppo aggressivo o instabile → il problema va cercato in:
  - setup reward
  - observation
  - action semantics
  - hyperparameter tuning

## Comando di riferimento per il test PPO

```bash
scripts/tmux_seed_grid.sh \
  --session "dbg_ppo_sb3like" \
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
    agent/planner/decoder=ppo_sb3 \
    agent/planner/algorithm=ppo \
    analysis.experiment_group=EXP_dbg_ppo_sb3like_lidar
```

## Nota finale

Per l'obiettivo tesi, la strategia migliore è:

1. portare il custom a una parità credibile con SB3
2. verificare empiricamente la parità
3. costruire sopra quella base le varianti lessicografiche/distribuzionali

