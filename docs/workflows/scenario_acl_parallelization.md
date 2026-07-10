# Futuro redesign: Scenario ACL con env paralleli

Stato attuale: `scenario_acl` richiede `env.vectorized.enabled=false`.
Questa limitazione è intenzionale: un chunk ACL oggi corrisponde a un singolo
scenario, un record nel buffer e un unico segnale di learning potential.

## Decisione di progetto futura

Non replicare lo stesso scenario su tutti i worker. La semantica desiderata è:

```text
un worker → un scenario selezionato/generato → un ScenarioRecord
```

In un chunk con `N` worker devono quindi esistere `N` scenari distinti, ognuno
con il proprio export/replay, learning potential, aggiornamento del buffer e
traccia di provenienza.

## Requisiti di implementazione

1. **Batch di selezione ACL**
   - il MAB campiona un braccio per worker;
   - il buffer può selezionare record diversi per worker;
   - ogni selezione conserva probabilità e seed separati.

2. **Env vettorizzati con override per-worker**
   - non usare la partizione automatica dei seed di `build_train_env`;
   - costruire ogni worker con il proprio `start_seed`, `num_scenarios=1` e,
     per il replay, il proprio dataset `ScenarioEnv`;
   - il seed del processo worker non deve modificare il seed/scenario scelto.

3. **Attribuzione per scenario**
   - raccogliere transizioni, episodi, loss e learning potential per worker;
   - non usare una media del chunk per aggiornare più record;
   - aggiornare MAB, rank, staleness e `num_seen` per record.

4. **Persistenza e resume**
   - serializzare la lista di selezioni correnti e lo stato RNG;
   - rendere atomici gli aggiornamenti batch del buffer;
   - verificare che il replay buffer del planner mantenga `n_envs` costante.

5. **Test necessari**
   - smoke `generate[N] → buffer[N] → replay[N]`;
   - mix generation/replay nello stesso batch;
   - attribution test: modificare il learning potential di un worker deve
     aggiornare solo il suo record/MAB feedback;
   - resume deterministico con `N > 1`.

Non attivare `env.vectorized.enabled=true` per ACL prima di questo redesign.
