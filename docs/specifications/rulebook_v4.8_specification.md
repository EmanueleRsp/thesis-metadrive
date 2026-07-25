# Specification: Metrica R2 — Sostituzione del clearance geometrico veicolo/statico con RSS laterale scoped

## Metadata

- Feature: `r2_lateral_rss_clearance_replacement`
- Specification ID: `rulebook-v2-r2-clearance`
- Version: `4.8`
- Status: `APPROVED`
- Date: `2026-07-24`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete` (solo per il sottoinsieme §6.4 modificato da questa versione; il resto di v4.7 resta autoritativo e invariato)
- Related specifications: `docs/specifications/rulebook_v4.7_specification.md` (base per §2, §4, §6.1-6.3, §6.5-6.6 non toccati da questa bozza)
- Related ADRs: `docs/decisions/ADR-025-r2-lateral-rss-clearance-replacement.md` (approvato)
- Authoritative: YES

## 1. Purpose And Context

La specifica v4.7 §6.4 definisce il costo di clearance come

```
q_clear,i = [1 - d_i^poly / D_min,tau_i]_+
```

funzione monotona decrescente della sola distanza poligono-poligono
`d_i^poly`, con soglie fisse `D_min` per classe di attore (veicolo 0.8 m,
VRU 1.0 m, oggetto statico 0.5 m). Applicata a veicoli e ostacoli statici,
questa formulazione non è equivalente a un vincolo di collision safety nel
senso di Censi et al. — la stessa motivazione già usata in v4.7 §4.2 per
giustificare `R2 ≻ R3`. Una prossimità geometrica ridotta diventa
un'erosione di `R2` anche quando non esiste dinamica di conflitto reale,
confondendo sei situazioni distinte:

1. distanza ridotta ma stabile (nessuna convergenza);
2. avvicinamento laterale realmente pericoloso;
3. due veicoli entrambi fermi;
4. sorpasso corretto di un veicolo parcheggiato;
5. corsie adiacenti percorse parallelamente;
6. un ostacolo realmente diretto verso il footprint ego.

Censi et al. mostrano inoltre che attribuire priorità alta alla clearance
può indurre una policy a oltrepassare una lane boundary pur di allontanarsi
da un ostacolo che non pone un conflitto reale — la clearance non è
semanticamente equivalente alla collision safety.

Questa versione sostituisce il clearance veicolo/veicolo con una metrica
RSS laterale scoped, che prende dal Responsibility-Sensitive Safety
(Shalev-Shwartz, Shammah, Shashua) soltanto la nozione di distanza laterale
dinamicamente sicura, applicata a coppie ego-veicolo in geometrie locali
compatibili, trasformata in un costo bounded. Non vengono implementati
proper response, safety filter, o RSS generalizzato per intersezioni con
geometrie differenti — questo resta esplicitamente fuori scope (§2). Rende
inoltre il clearance statico diagnostico-only, e lascia invariato il
clearance VRU, per cui la formulazione a distanza pura resta appropriata
(RSS distingue esplicitamente i VRU dagli altri attori per dinamiche,
percorsi e responsabilità differenti).

I valori numerici dei parametri laterali (§9) provengono dai valori
iniziali suggeriti dalla documentazione ufficiale Intel `ad-rss-lib`; la
loro trasformazione in un costo bounded di `R2`, il riuso del response time
già approvato di progetto, e il gate longitudinale di §7 sono adattamenti
originali di questo progetto, non una conseguenza diretta della letteratura.

## 2. Scope

### In Scope

- Rimozione di `q_clear,vehicle` dal costo `c_2(t)` e sua sostituzione con
  una nuova metrica scoped `q_RSS,lat` limitata a coppie ego-veicolo con
  applicabilità esplicita (§8).
- Trasformazione di `q_clear,static` in un valore diagnostico-only (loggato,
  non incluso in `c_2(t)`).
- Nuova aggregazione di `R2`:
  `c_2(t) = max{q_RSS,long, q_RSS,lat, q_TTC, q_clear,VRU}`.
- Gate longitudinale `I_long,unsafe` che riusa la formula RSS longitudinale
  già approvata in v4.7 §6.2.

### Out Of Scope

- Qualunque modifica a RSS longitudinale (§6.2 v4.7), TTC generalizzato
  (§6.3 v4.7) o al sottoinsieme VRU di clearance (§6.4 v4.7) — restano
  quelli già approvati.
- Riclassificazione degli attori (un veicolo fermo o parcheggiato resta
  `VEHICLE`; invariante già garantito dall'implementazione corrente, si
  veda REQ-R2-06).
- Full RSS (proper response, safety filter, geometrie di route differenti,
  intersezioni, priorità non strutturate) — deliberatamente non
  implementato per restare circoscritti al costo scoped di `R2`.
- Modifiche alla gerarchia `R1 ≻ R2 ≻ R3 ≻ R4` o ad altri componenti
  normativi (vehicle-yield, segnaletica, crosswalk, ecc.).

### Optional Or Deferred

- Estensione della stessa formulazione laterale a VRU — esplicitamente
  esclusa in questa versione (§8).
- Estensione a geometrie con lane non parallele/route differenti (full RSS)
  — deferred, non pianificata da questa specifica.

## 3. Terminology, Assumptions, And Preconditions

Riusa per riferimento la terminologia e le convenzioni di v4.7 §2.x (ego,
attore, footprint, `RoutePolyline`, `associate_route_lane`, `Δt`,
`T_pred`). Nuovi simboli:

- **Frame tangente/normale condiviso**: per una coppia (ego, attore `i`)
  applicabile (§8), l'asse tangente `t` è la direzione legale locale della
  strada (derivata dalla lane assegnata all'ego lungo il percorso
  canonico); l'asse normale `n = (-t_y, t_x)` è perpendicolare, orientato
  dal veicolo a sinistra verso quello a destra della coppia. Fonte:
  guaranteed upstream tramite `RoutePolyline.project`/`associate_route_lane`
  quando l'associazione di corsia di entrambi gli attori esiste; altrimenti
  la coppia è `NOT_APPLICABLE` (§8) — nessun frame ambiguo viene mai usato
  per calcolare un costo.
- **Distanza laterale `d_i^lat`**: edge-to-edge gap fra gli intervalli
  ottenuti proiettando i due footprint sull'asse `n` (analogo laterale del
  pattern longitudinale già usato da `footprint_route_coordinates`).
- **Velocità laterale entrante (inward)**: `u_e^in`, `u_i^in`, proiezioni
  delle velocità world-frame su `n`, rappresentate con segno positivo
  quando dirette verso l'altro veicolo.
- **`rho_lat`**: tempo di risposta laterale (§9), un valore dedicato, non
  il `rho=1.0s` di RSS longitudinale.
- **`a_acc_max^lat`, `a_brake_min^lat`**: accelerazione massima di
  allontanamento laterale e decelerazione minima di frenata laterale
  garantita (§9).
- **`mu`**: margine di sicurezza laterale fisso (§9).
- **Snapshot**: la metrica legge dal **pre_state** dello step corrente
  (posizioni, velocità, footprint immediatamente prima dell'azione ego),
  coerente con RSS longitudinale e TTC che già leggono da `pre_state`
  (`transition.py`); decisione confermata dall'utente il 2026-07-24 (si
  veda §15, non più aperta).

Tutte le grandezze sopra sono guaranteed upstream se e solo se
`associate_route_lane`/`RoutePolyline.project` producono un'associazione di
corsia non nulla e compatibile per ego e per l'attore; altrimenti la
coppia è `NOT_APPLICABLE` (§8), mai un costo implicito pari a zero
mascherato da un'assunzione silenziosa.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `ego_lateral_extent_m` | proiezione min/max del footprint ego su `n` | m, frame condiviso | `(-inf, inf)` | estensione laterale di `footprint_route_coordinates` (da implementare) | assente → coppia `NOT_APPLICABLE` | NO |
| `actor_lateral_extent_m` | idem per l'attore `i` | m | idem | idem | idem | NO |
| `ego_inward_lateral_speed_mps`, `actor_inward_lateral_speed_mps` | velocità normali proiettate, convenzione inward | m/s | `(-inf, inf)` | derivate da `velocity_xy` e dal versore tangente `t` del frame condiviso | assente → coppia `NOT_APPLICABLE` | NO |
| `longitudinal_unsafe_gate` | predicato booleano `I_long,unsafe,i` (§7) | bool | `{0,1}` | overlap longitudinale o formula RSS longitudinale esistente | assente → gate falso, coppia esclusa dal costo | NO |

Prohibited: nessuna informazione futura oltre lo stato `pre_state` dello
step corrente (nessun future track, nessuna intention prediction); nessun
dato privilegiato lato simulatore non ricostruibile da osservazione live;
nessuna trajectory prediction oltre la proiezione istantanea già usata da
RSS longitudinale.

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `q_RSS,lat` | costo RSS laterale aggregato worst-of sugli attori applicabili | float `[0,1]` | worst-of | `c_2(t)` | `0.0` se nessuna coppia applicabile o `d_safe^lat=0` |
| `q_clear,static` (diagnostic-only) | distanza poligono-statico, non convertita in costo | float, m | worst-of, solo log | diagnostica/log | non contribuisce a `c_2(t)` |
| `lateral_gap_m`, `lateral_safe_distance_m`, `ego_inward_lateral_speed_mps`, `actor_inward_lateral_speed_mps`, `longitudinal_unsafe_gate`, `lateral_rss_applicable`, `lateral_rss_cost` | diagnostici per coppia worst | vari | — | log/valutazione | diagnostic-only, tranne `lateral_rss_cost` che coincide con `q_RSS,lat,i` della coppia worst (training signal) |
| `vru_clearance_m` | clearance VRU, invariato da v4.7 | float, m | worst-of | `c_2(t)` (tramite `q_clear,VRU`) | invariato |
| `static_polygon_distance_m` (diagnostic-only) | distanza poligono-statico raw | float, m | worst-of, solo log | log | non training signal |

## 6. Functional Requirements

### REQ-R2-01: Rimozione del clearance veicolo dal costo

- Required observable behavior: `q_clear,vehicle` non contribuisce più a
  `c_2(t)`; il componente `"clearance"` esistente resta applicabile
  esclusivamente alle classi `PEDESTRIAN`/`CYCLIST` (VRU).
- Applicabilità: tutte le coppie ego-veicolo, incondizionatamente.
- Invarianti: nessuna soglia veicolo residua influenza `c_2(t)`.
- Edge and missing-data cases: non applicabile (rimozione incondizionata).
- Failure or fallback behavior: non applicabile.
- Interactions: sostituito da REQ-R2-03.

### REQ-R2-02: Clearance statico diagnostico-only

- Required observable behavior: per attori `STATIC_COLLIDABLE`,
  `d_i^poly` resta calcolata e loggata come `static_polygon_distance_m` ma
  non entra in `c_2(t)`.
- Applicabilità: tutti gli ostacoli statici collidibili, vertically
  compatible con l'ego.
- Invarianti: nessuna soglia `D_min` statica viene rimossa dai log.
- Edge and missing-data cases: se non esistono attori statici vertically
  compatible, il valore diagnostico è assente, non zero.
- Failure or fallback behavior: non applicabile (solo logging, nessun
  fallimento possibile).
- Interactions: TTC resta l'unico componente di `R2` che può catturare una
  collisione imminente con uno statico (AC-R2-06).

### REQ-R2-03: Nuova metrica RSS laterale scoped

- Required observable behavior: per ogni coppia ego-veicolo applicabile
  (§8), calcolare `q_RSS,lat,i` secondo la formula di §7 e aggregare
  worst-of in `q_RSS,lat`.
- Applicabilità: solo attori di classe `VEHICLE`; solo quando il gate
  longitudinale `I_long,unsafe,i` è vero (§7); solo quando l'associazione
  di corsia di entrambi gli attori esiste e permette di costruire un unico
  frame tangente/normale condiviso (§8).
- Invarianti: `q_RSS,lat,i in [0,1]`; `q_RSS,lat,i = 0` esplicito quando
  `d_safe^lat,i = 0`.
- Edge and missing-data cases: coppia non applicabile → esclusa dal
  worst-of, non contribuisce con uno zero implicito (pattern coerente con
  `evaluate_clearance`: assenza di candidati ⇒ componente
  `NOT_APPLICABLE`).
- Failure or fallback behavior: un errore nella route o nei polygon
  richiesti dopo che lo scenario è stato validato produce
  `RulebookEvaluationError` (stesso contratto di RSS longitudinale v4.7
  §6.2.1).
- Interactions: complementare a RSS longitudinale (gate condiviso, §7) e a
  TTC; nessun doppio conteggio perché l'aggregatore di `R2` resta il
  massimo.

### REQ-R2-04: Clearance VRU invariato

- Required observable behavior: `q_clear,VRU` resta esattamente quello di
  v4.7 §6.4 (soglia 1.0 m, stessa formula, stesso insieme di candidati).
- Applicabilità: invariata da v4.7.
- Invarianti: nessuna modifica di formula, soglia, o candidate set.
- Edge and missing-data cases: invariati da v4.7.
- Failure or fallback behavior: invariato da v4.7.
- Interactions: nessuna nuova interazione.

### REQ-R2-05: Nuova aggregazione R2

- Required observable behavior:
  `c_2(t) = max{q_RSS,long(t), q_RSS,lat(t), q_TTC(t), q_clear,VRU(t)}`,
  `m_2(t) = -c_2(t)`.
- Applicabilità: sempre (l'aggregazione worst-of gestisce già i componenti
  non applicabili come oggi).
- Invarianti: `c_2(t) in [0,1]`.
- Edge and missing-data cases: se tutti i sotto-componenti sono
  `NOT_APPLICABLE`, `c_2(t) = 0` (pattern invariato da v4.7 §6.5).
- Failure or fallback behavior: invariato dal pattern esistente in
  `aggregate_max_component`.
- Interactions: aggiorna il gruppo `dynamic_interaction_safety` in
  `aggregation.py` (oggi `{"rss","ttc","clearance"}` → deve includere il
  nuovo componente laterale e continuare a includere `clearance` solo per
  il sottoinsieme VRU) e la registry dei componenti in `registry.py`.

### REQ-R2-06: Invariante di classificazione attori preservato

- Required observable behavior: un veicolo fermo o parcheggiato resta
  classificato `VEHICLE`, mai riclassificato `STATIC_COLLIDABLE` in base
  alla velocità.
- Applicabilità: tutti gli attori live.
- Invarianti: la classificazione dipende esclusivamente dal tipo
  dell'attore, mai dalla sua velocità istantanea.
- Edge and missing-data cases: non applicabile.
- Failure or fallback behavior: non applicabile.
- Interactions: precondizione di REQ-R2-03 (un veicolo parcheggiato deve
  restare candidato a RSS laterale, non silenziosamente escluso passando a
  `STATIC_COLLIDABLE`, si veda AC-R2-04).
- Stato attuale: già soddisfatto dall'implementazione corrente
  (`src/thesis_rl/rulebook/v2/context/metadrive_live.py::_actor_class`,
  classificazione puramente basata sul nome del tipo — verificato tramite
  grep di `parked`/`stationary`/`is_static` nel repository, nessuna logica
  basata su velocità esiste). Nessuna modifica di codice richiesta da
  questo requisito; manca però un test di regressione esplicito in
  `tests/test_rulebook_v2_metadrive_live.py` che verifichi che un attore
  `VEHICLE` a velocità zero/bassa resti `VEHICLE` — da aggiungere
  nell'ExecPlan di implementazione (indipendentemente dall'approvazione di
  questa specifica, è un gap di copertura preesistente).

## 7. Mathematical And Algorithmic Contract

Frame condiviso: per la coppia (ego, attore `i`) applicabile, si usa il
frame tangente/normale della corsia assegnata all'ego lungo il percorso
canonico (coerente con `RoutePolyline.project`/`associate_route_lane`);
quando le corsie di ego e attore non sono localmente parallele o non è
possibile costruire un frame condiviso univoco, la coppia è
`NOT_APPLICABLE` (§8) — non esiste quindi un caso di frame ambiguo da
risolvere con una convenzione separata.

Distanza laterale corrente (edge-to-edge gap fra le proiezioni dei due
footprint su `n`):

```
d_i^lat(t) = gap( proj_n P_e , proj_n P_i )
```

Velocità laterali entranti (convenzione inward, positive quando dirette
verso l'altro veicolo):

```
u_e^in , u_i^in    (proiezione di v_e, v_i su n, con segno verso l'altro attore)
```

Velocità proiettate al tempo di risposta `rho_lat` (assumendo
accelerazione massima di allontanamento durante la finestra di risposta):

```
u_{a,rho}^in = u_a^in + rho_lat * a_acc_max^lat ,   a in {e, i}
```

Spostamento laterale worst-case durante `rho_lat` più frenata successiva:

```
Delta_a^lat = (u_a^in + u_{a,rho}^in) / 2 * rho_lat
              + [u_{a,rho}^in]_+^2 / (2 * a_brake_min^lat) ,   a in {e, i}
```

Distanza laterale minima di sicurezza:

```
d_safe,i^lat = [ mu + Delta_e^lat + Delta_i^lat ]_+
```

Gate longitudinale (evita di applicare RSS laterale a coppie senza
sovrapposizione o avvicinamento longitudinale rilevante):

1. proiettare i footprint di ego e attore `i` sull'asse tangente `t`;
2. se gli intervalli longitudinali si sovrappongono, porre
   `I_long,unsafe,i = 1`;
3. altrimenti identificare quale dei due è il veicolo posteriore e quale
   l'anteriore lungo `t`;
4. riutilizzare la formula RSS longitudinale già presente in v4.7 §6.2.2
   (`d_safe,i` e il gap bumper-to-bumper) applicata a questa coppia
   posteriore/anteriore;
5. porre `I_long,unsafe,i = 1` soltanto se il gap longitudinale è inferiore
   alla distanza longitudinale sicura così calcolata; altrimenti
   `I_long,unsafe,i = 0`.

Costo bounded (zero esplicito quando `d_safe,i^lat = 0`):

```
q_RSS,lat,i =
  0                                                  se I_long,unsafe,i = 0
  0                                                  se d_safe,i^lat = 0
  clip( [d_safe,i^lat - d_i^lat(t)]_+ / d_safe,i^lat , 0, 1 )   altrimenti
```

Aggregazione:

```
q_RSS,lat(t) = max_i q_RSS,lat,i(t)     (worst-of sugli attori applicabili)
c_2(t) = max{ q_RSS,long(t), q_RSS,lat(t), q_TTC(t), q_clear,VRU(t) }
m_2(t) = -c_2(t)
```

Nota numerica di verifica (dall'analisi originale, da usare come test di
non-regressione della formula): per due veicoli inizialmente a velocità
laterale nulla (`u_e^in = u_i^in = 0`), con i parametri congelati di §9
(`rho_lat=0.5s`, `a_acc_max^lat=0.2 m/s^2`, `a_brake_min^lat=0.8 m/s^2`,
`mu=0.10 m`), si ottiene `d_safe^lat ≈ 0.1625 m`.

Snapshot: tutte le grandezze di questa sezione (`d_i^lat`, `u_e^in`,
`u_i^in`, il gate longitudinale) sono calcolate sul **pre_state** dello
step corrente (§3), coerentemente con RSS longitudinale e TTC.

## 8. Applicability, State, And Timing

`q_RSS,lat,i` è applicabile soltanto quando, simultaneamente:

- l'attore `i` è di classe `VEHICLE`;
- ego e l'attore hanno entrambi un'associazione di corsia valida e non
  ambigua sul percorso canonico;
- le lane di ego e attore appartengono alla stessa struttura stradale o
  sono lane adiacenti compatibili (nessuna intersezione di rami diversi,
  nessun merge non ancora ricondotto a una stessa lane locale);
- le direzioni legali locali delle due lane sono concordi;
- è possibile costruire un unico frame tangente/normale condiviso (§7).

È `NOT_APPLICABLE` per la coppia quando una qualunque delle condizioni
sopra non è soddisfatta, in particolare: traiettorie su rami diversi di
un'intersezione; lane che si incrociano; relazione di merge non ancora
ricondotta a una stessa lane locale; associazione di corsia ambigua per
ego o per l'attore; geometrie con direzioni legali incompatibili. In questi
casi restano attivi TTC, collisione, crosswalk/vehicle-conflict-zone e
vehicle-yield, invariati da v4.7 — nessuna copertura di sicurezza viene
persa, solo la sotto-metrica laterale scoped non si applica a quella
coppia in quello step.

Nessuno stato persistente fra step è richiesto da questa metrica oltre
all'associazione di corsia già derivata correntemente (nessun nuovo campo
di `RulebookMemory`). Snapshot sorgente: `pre_state` (§3, §7) — decisione
risolta, non più aperta.

## 9. Configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen for experiments |
|---|---|---|---|---|---|---|
| `rho_lat_s` | float | `0.5` | `> 0` | Tempo di risposta laterale | YES | YES |
| `lateral_acc_max_mps2` | float | `0.2` | `> 0` | Accelerazione massima di allontanamento laterale | YES | YES |
| `lateral_brake_min_mps2` | float | `0.8` | `> 0` | Decelerazione minima di frenata laterale garantita | YES | YES |
| `lateral_margin_mu_m` | float | `0.10` | `>= 0` | Margine di sicurezza laterale fisso | YES | YES |

Tutti i valori sono presi dai valori iniziali suggeriti dalla
documentazione ufficiale Intel `ad-rss-lib` per la dinamica laterale
(stesso trattamento di valore-congelato-da-fonte già applicato a
`rho`/`a_max^acc`/`b_i` in v4.7 §6.2.4). Un valore non positivo o non
finito invalida la configurazione. Modificare uno di questi valori richiede
una nuova versione approvata della specifica o un ADR approvato — non è
consentita una ricalibrazione tramite training.

## 10. Errors, Logging, And Diagnostics

Un'associazione di corsia ambigua o assente per ego o per l'attore produce
`NOT_APPLICABLE` per quella specifica coppia (non un errore); TTC e RSS
longitudinale restano attivi. Un errore nella route o nei polygon richiesti
dopo che lo scenario è stato validato produce `RulebookEvaluationError`
(stesso contratto di v4.7 §6.2.1).

Nuovi campi diagnostici (tutti diagnostic-only salvo indicato):
`lateral_gap_m`, `lateral_safe_distance_m`, `ego_inward_lateral_speed_mps`,
`actor_inward_lateral_speed_mps`, `longitudinal_unsafe_gate`,
`lateral_rss_applicable`, `lateral_rss_cost` (== `q_RSS,lat,i` della coppia
worst, training signal), `vru_clearance_m` (invariato da v4.7, training
signal via `q_clear,VRU`), `static_polygon_distance_m` (diagnostic-only,
non training signal).

## 11. Reproducibility And Compatibility

Cambia la composizione del gruppo `dynamic_interaction_safety` in
`aggregation.py`/`registry.py` e lo shape del vettore diagnostico di `R2`
— è un cambio comportamentale del costo `R2`, non solo diagnostico:
`c_2(t)` può cambiare valore per scenari con veicoli vicini ma non
convergenti (atteso: diminuisce, casi 1/3/4/5 di §1) o con avvicinamenti
laterali reali non catturati oggi da RSS longitudinale/TTC (atteso: può
aumentare, caso 2 di §1). Richiede rivalutazione di eventuali baseline o
esperimenti già calibrati sul costo `R2` di v4.7: la scala del segnale non
è direttamente comparabile fra le due versioni, solo la sua distribuzione
numerica potrebbe apparire simile in scenari senza convergenza laterale.
Nessun checkpoint di policy già addestrata è compatibile in modo
trasparente con questo cambio. Seed, split, versioni di dipendenze e
formato di configurazione non cambiano.

## 12. Acceptance Criteria

### AC-R2-01: Distanza ridotta ma stabile (nessun avvicinamento)

- Given: ego e un veicolo su corsie affiancate, `u_e^in = u_i^in = 0`,
  `d_i^lat` costante sotto la vecchia soglia `D_min=0.8 m`.
- When: si valuta un transition step.
- Then: `q_clear,vehicle` non esiste più nell'output; `q_RSS,lat,i = 0`
  (con i margini tipici di §9, `d_safe^lat ≈ 0.1625 m ≤ d_i^lat`).
- Related requirements: `REQ-R2-01`, `REQ-R2-03`.

### AC-R2-02: Avvicinamento laterale reale

- Given: l'attore `i` (o ego) ha velocità laterale entrante positiva verso
  l'altro veicolo, `d_i^lat(t)` in diminuzione, `I_long,unsafe,i = 1`.
- When: `d_i^lat(t) < d_safe,i^lat(t)`.
- Then: `q_RSS,lat,i > 0`, contribuisce a `c_2(t)`.
- Related requirements: `REQ-R2-03`, `REQ-R2-05`.

### AC-R2-03: Due veicoli fermi lateralmente (caso di riferimento numerico)

- Given: `u_e^in = u_i^in = 0`, parametri congelati di §9,
  `d_i^lat = 0.1625 m` esatti.
- When: si calcola `q_RSS,lat,i`.
- Then: `q_RSS,lat,i = 0` (bordo, non violazione) — usato come test di
  regressione numerica della formula.
- Related requirements: `REQ-R2-03`.

### AC-R2-04: Sorpasso di un veicolo parcheggiato

- Given: veicolo bersaglio a velocità ~0, classificato `VEHICLE`
  (`REQ-R2-06`), ego lo supera mantenendo margine laterale variabile nel
  tempo ma senza convergenza netta.
- When: nessuna convergenza laterale durante il sorpasso
  (`u_e^in`/`u_i^in` non entrambe positive e crescenti verso l'altro).
- Then: `q_RSS,lat,i` resta 0 o basso per l'intera manovra corretta; il
  veicolo bersaglio non viene mai riclassificato `STATIC_COLLIDABLE`.
- Related requirements: `REQ-R2-03`, `REQ-R2-06`.

### AC-R2-05: Corsie parallele adiacenti, nessun conflitto

- Given: ego e attore su corsie parallele stabili, nessuna convergenza per
  l'intero episodio.
- When: si valuta ogni step dell'episodio.
- Then: `q_RSS,lat,i = 0` per l'intero episodio, anche con `d_i^lat`
  ridotta.
- Related requirements: `REQ-R2-03`.

### AC-R2-06: Ostacolo statico realmente invadente

- Given: attore `STATIC_COLLIDABLE` con `d_i^poly` che tende a zero
  (invasione di corsia verso il footprint ego).
- When: la traiettoria ego converge geometricamente con l'ostacolo entro
  l'orizzonte TTC.
- Then: `q_TTC` cattura l'evento (nessuna altra sotto-metrica di `R2` lo fa
  più, per costruzione, dato che il clearance statico è diagnostic-only).
  Se la geometria non converge entro l'orizzonte TTC, resta un limite noto
  da documentare esplicitamente (non un fallimento silente non tracciato).
- Related requirements: `REQ-R2-02`.

## 13. Required Validation Categories

- nominal and boundary behavior: Required (AC-R2-01..03).
- invalid and incomplete inputs: Required (associazione di corsia assente
  o ambigua → `NOT_APPLICABLE`).
- masks, padding, state, reset, and update order: Not applicable (nessuno
  stato persistente introdotto).
- termination and truncation: Not applicable.
- deterministic seeds and reproducibility: Not applicable (formula
  deterministica pura, nessuna fonte di casualità).
- numerical stability, NaN, and infinity: Required (`d_safe^lat=0` gestito
  esplicitamente).
- compatibility and migration: Required (§11).
- absence of future and privileged information: Required (§3, §4 —
  `pre_state` only).
- upstream, downstream, and end-to-end integration: Required (aggregazione
  `R2` end-to-end tramite `aggregate_rulebook_result`).
- regressions for known bugs: Required (`REQ-R2-06`, invariante di
  classificazione già garantito ma non ancora coperto da test esplicito).

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| REQ-R2-01, REQ-R2-03, REQ-R2-05 | AC-R2-01, AC-R2-02, AC-R2-03, AC-R2-05 | Shalev-Shwartz, Shammah, Shashua, "On a Formal Model of Safe and Scalable Self-driving Cars" (arXiv:1708.06374), Lemma 4; Intel `ad-rss-lib` |
| REQ-R2-02 | AC-R2-06 | Censi et al.; v4.7 §4.2 |
| REQ-R2-04 | — | v4.7 §6.4 (sottoinsieme VRU), invariato |
| REQ-R2-06 | AC-R2-04 | Verifica del repository corrente (nessuna modifica di codice necessaria) |

## 15. Open Decisions And Limitations

| ID | Question | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|
| DEC-R2-02 | Quale snapshot (`pre_state` o `post_state`) alimenta la nuova metrica laterale | (A) `pre_state`, coerente con RSS longitudinale/TTC; (B) `post_state`, coerente con l'attuale clearance | (A) | Determina quale step "vede" per primo un avvicinamento laterale | **APPROVED** — risolto dall'utente il 2026-07-24: `pre_state` (opzione A) |
| DEC-R2-04 | Come esporre `static_polygon_distance_m` senza farlo contribuire a `c_2(t)` | (A) solo in `diagnostics`, mai in `raw`/costo; (B) componente separato sempre `NOT_APPLICABLE`-only, presente solo per il log | (A), pattern minimo | Solo logging, nessun impatto sul costo | **APPROVED** — risolto in sede di ExecPlan il 2026-07-24: opzione (A), `static_polygon_distance_m` esposto esclusivamente nel campo `diagnostics` del componente `clearance` esistente, mai in `raw`/costo |

Nessuna decisione resta aperta. Il frame tangente/normale, la convenzione
di segno delle velocità, il gate longitudinale, l'applicabilità e
l'esposizione diagnostica dello statico sono interamente specificati da
§7-8 e da `DEC-R2-04`.

## 16. References

- A. Censi, K. Slutsky, T. Wongpiromsarn, D. Yershov, S. Pendleton, J.
  Fu, E. Frazzoli, "Liability, Ethics, and Culture-Aware Behavior
  Specification using Rulebooks" — motivazione della gerarchia `R1 ≻ R2 ≻
  R3` e della non-equivalenza fra clearance geometrica e collision safety
  (già citata in v4.7 §4.2, riusata qui come motivazione principale del
  cambio).
- ScenicRules — soglie `D_min` veicolo/VRU riusate (invariate) per il
  sottoinsieme VRU di v4.7 §6.4.
- Intel `ad-rss-lib`, Overview e Appendix "Parameter Discussion"
  (`intel.github.io/ad-rss-lib`) — valori iniziali suggeriti per
  `rho_lat`, `a_acc_max^lat`, `a_brake_min^lat`, `mu`; nota esplicita che
  sono valori iniziali configurabili, non costanti universali.
- S. Shalev-Shwartz, S. Shammah, A. Shashua, "On a Formal Model of Safe
  and Scalable Self-driving Cars" (arXiv:1708.06374), in particolare
  Lemma 4 — formula RSS laterale di riferimento, da bloccare con test di
  conformità ai segni.

## 17. Implementation Handoff Checklist

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, shapes, units, frames, ranges, and masks.
- [x] Prohibited future, privileged, leaked, and diagnostic-only data is listed.
- [x] Formulas, algorithms, applicability, and fallbacks are unambiguous.
- [x] State, timing, reset, termination, and truncation behavior is defined.
- [x] Configuration fields and scientifically frozen defaults are identified.
- [x] Errors, diagnostics, reproducibility, compatibility, and migration are covered.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected or marked `Not applicable`.
- [x] Scientific sources, project adaptations, and approved decisions are distinct.
- [x] No material decision remains open. — `DEC-R2-02` and `DEC-R2-04` are
      both `APPROVED`; no scientific or implementation decision is open.
- [x] Known limitations are intentional and do not hide missing requirements
      (AC-R2-06's noted limit is explicit, not silent).

## 18. Approval Record

- Approved by: user
- Approval date: 2026-07-24
- Approval evidence: explicit instruction "Approvo la specifica v4.8,
  procedi con l'implementazione" in this conversation, after the full
  specification (context, scope, functional requirements, mathematical
  contract, configuration, acceptance criteria, and open decisions) was
  presented for review and `DEC-R2-02` was separately resolved by the user
  during drafting.
- Approval notes: `DEC-R2-04` resolved as an implementation detail (§15)
  concurrently with approval, per `AGENTS.md`'s decision authority for
  non-material choices; no new user approval required for it.
- Repository path: `docs/specifications/rulebook_v4.8_specification.md`
  (moved from `incoming/rulebook_v4.8_specification_UNDER_REVIEW.md`)
- Project index updated: YES
