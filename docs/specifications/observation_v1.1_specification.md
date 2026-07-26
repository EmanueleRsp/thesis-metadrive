---
title: "Specificazione finale delle osservazioni per MetaDrive ScenarioEnv"
subtitle: "Baseline LiDAR-state, osservazione semantica object-centric, storia temporale, anti-leakage e contratto implementativo"
author: "Report tecnico per la tesi"
date: "15 luglio 2026"
lang: it-IT
version: "1.1-final-implementation-complete"
specification_id: "OBS-V1.1"
status: "APPROVED"
authoritative: true
approval_date: "2026-07-16"
approval: "Explicit user approval in this Codex conversation"
related_adrs:
  - "docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md"
  - "docs/decisions/ADR-004-assigned-route-metadata-for-pg-and-waymo.md"
  - "docs/decisions/ADR-026-dynamic-slot-identity-removal-and-context-quota-ranking.md"
amendments:
  - "2026-07-17: assigned-route task metadata for PG and Waymo"
  - "2026-07-26: clarified actor-cap fallback in the relative-speed scale (§10.2) for actors without configured_speed_cap_mps (VRUs); fixed corresponding bug in causal_semantic.py"
  - "2026-07-26: replaced the Context quota ranking key (§8.1) with plain Euclidean distance, removing lane-relation/ahead precedence; see ADR-026"
---

# Sintesi esecutiva

Questa specifica definisce le due osservazioni usate nella pipeline RL:

1. **`StackedLidarStateObservation`**: baseline compatta derivata dalla `LidarStateObservation` di MetaDrive, con route ego ricostruita dalla mappa, sensori ray-based configurati, rumore leggero e stack temporale;
2. **`SemanticStateObservationV2`**: osservazione custom post-perception, object-centric e temporalmente strutturata, progettata per fornire alla policy le informazioni causalmente disponibili necessarie a prevedere le conseguenze del rulebook v4.6 senza esporre costi, margini, future ground truth o decisioni già prese dal monitor.

Decisioni principali:

- setting **single-agent, closed-loop, post-perception / mid-to-end**;
- percezione corrente ideale, completata da preprocessing deterministico e causale di mappa, route e contesto normativo;
- la route ego è una missione assegnata, costruita da centerline canoniche di
  mappa e congelata al reset;
- la traiettoria futura SDC non è mai letta online come navigation; per Waymo
  può soltanto aver prodotto l’annotazione di missione offline congelata;
- non si usano route, checkpoint o intenzioni ground-truth degli altri attori;
- history breve con \(H=5\);
- tutti gli stati storici di ego e attori vengono riespressi nel frame ego corrente;
- capacità dinamica \(K_{\mathrm{dyn}}=16\), con \(8\) slot conflict-reserved e \(8\) context-reserved; la persistenza è subordinata al ranking corrente e i conflict actor possono preemptare slot context;
- gli ID degli attori servono solo per tracking e slot persistence e non sono input della policy;
- route tokens \(M=10\), distanziati di circa \(5\,m\);
- interaction tokens pairwise \(K_{\mathrm{int}}=8\), uno per coppia `(zone_id, actor_id)`;
- nessun dense LiDAR ray viene aggiunto alla semantic observation;
- gli stati causali minimi degli automi temporali del rulebook sono esposti, ma non i loro costi o margini;
- `SemanticStateObservationV2` ha dimensione flat totale:

\[
D_{\mathrm{semantic}}=2541;
\]

- la baseline LiDAR usa 240 raggi principali, 4 nearby vehicles, 12 raggi SideDetector e 12 LaneLineDetector;
- la baseline LiDAR ha dimensione per frame:

\[
D_{\mathrm{lidar}}=308,
\]

e con stack \(H=5\):

\[
D_{\mathrm{lidar,stack}}=1540;
\]

- rumore ray-based nominale: Gaussian noise normalizzato \(\sigma=0.001\), corrispondente a circa \(5\,cm\) su un range di \(50\,m\);
- il rumore nativo dei sensori MetaDrive è disattivato: un solo wrapper, alimentato dall’RNG dell’ambiente, perturba main LiDAR, SideDetector e LaneLineDetector una sola volta;
- dropout ray-based nel core: \(0\).

La specifica non introduce un nuovo modulo scientifico chiamato `SharedSceneContextExtractor`. Richiede soltanto che observation builder e rulebook riusino le stesse primitive canoniche di route, lane, boundary, traffic controls e conflict zones già definite dalla specifica del rulebook.

# 1. Ambito

## 1.1 Ambiente

La specifica è progettata per:

- `MetaDrive ScenarioEnv`;
- scenari in formato `ScenarioDescription`;
- dataset Waymo convertito tramite ScenarioNet;
- dataset procedurale PG esportato nello stesso formato;
- azioni continue ego `(steering, throttle/brake)`;
- control timestep nominale:

\[
\Delta t = 0.1\,s;
\]

- rulebook `v4.6-final-implementation-complete`.

## 1.2 Livello informativo

Il contributo della tesi riguarda planning e decision-making RL, non perception end-to-end. La semantic observation assume quindi un modulo percettivo ideale che rende disponibili al planner:

- stato corrente e history osservata dell’ego;
- oggetti dinamici tracciati fino al tempo corrente;
- ostacoli statici correnti;
- geometria locale della mappa;
- route ego assegnata;
- lane markings;
- traffic controls correnti;
- conflict zones derivate causalmente;
- memoria temporale minima costruita dal passato.

Questo setting deve essere descritto come:

> post-perception / mid-to-end planning with ideal current-state perception and deterministic causal map/rule-context preprocessing.

La seconda parte della definizione è necessaria perché associazione lane-control, conflict zones, right-of-way esplicito e automi temporali non sono output grezzi della percezione: sono elaborazioni deterministiche di informazioni disponibili online. Non deve essere descritto come robusto alle occlusioni o come sensor-to-control.

# 2. Contratto anti-leakage

## 2.1 Informazioni ammesse

Una feature è ammessa se è ricostruibile online usando soltanto:

- stato ego al tempo \(t\);
- stati ego precedenti;
- azioni applicate fino a \(t-1\);
- route ego assegnata e congelata al reset;
- route mission metadata frozen before reset; for Waymo only, this offline
  annotation may be map-matched from the complete SDC trajectory before the
  episode is made available to the runtime;
- mappa locale;
- lane graph e lane markings;
- oggetti osservati/tracciati fino a \(t\);
- traffic-control state corrente;
- memoria causale aggiornata usando transizioni già avvenute;
- predizioni cinematiche locali calcolate da stato corrente/passato con un modello dichiarato;
- associazioni di movimento e diritto di precedenza ricavate esclusivamente da lane corrente, heading/history osservata, mappa e traffic controls correnti; se tali dati non sono sufficienti, il valore deve essere `unknown`/`undefined`.

Sono ammesse, per esempio:

- relative position e relative velocity;
- constant-velocity CPA;
- occupancy intervals entro \(T_{\mathrm{pred}}\);
- timer dashed e stop;
- stato corrente del semaforo;
- dati salvati all’onset del giallo;
- indicazione che un ingresso incompatibile già avvenuto è latched fino all’uscita.

## 2.2 Informazioni vietate

Non devono essere consultate né direttamente né indirettamente:

- posizioni future registrate degli altri attori;
- future validity masks;
- route o checkpoint ground-truth degli altri attori;
- futura lane scelta dagli altri;
- futura traiettoria SDC usata online come navigation; una route assegnata
  ottenuta offline e congelata nei metadati è invece ammessa;
- future SDC samples accessed, indexed, replayed, or compared during reset,
  `observe()`, transition evaluation, or policy inference; the frozen assigned
  route metadata is the sole approved exception;
- future traffic-light phases;
- futura lane, futura uscita o futura `MovementKey` degli altri attori;
- precedenza inferita dalla futura traiettoria registrata di un altro attore;
- collisione futura;
- distanza minima futura registrata;
- successo/fallimento futuro dello scenario;
- label di difficoltà;
- arm del curriculum;
- source Waymo/PG come feature;
- learning potential;
- usefulness;
- reward scalarizzato;
- costi \(c_k\);
- margini \(m_k\);
- status `VIOLATED`;
- `unsafe_action`;
- score o quota di selezione degli oggetti.

## 2.3 Future supervision e input online

Le future tracks possono essere usate offline per:

- validazione del dataset;
- filtraggio degli scenari;
- generazione di label di training per eventuali moduli separati;
- metriche di valutazione.

Non possono essere lette durante `observe()` o usate per costruire l’input corrente della policy.

# 3. Primitive condivise con il rulebook

Non è obbligatorio implementare una nuova classe monolitica. È invece obbligatorio riusare gli stessi risultati canonici già definiti nel rulebook:

- `RoutePolyline`;
- route projection \(s\);
- lane association;
- `DrivableLaneRecord`;
- logical boundary ID;
- boundary type;
- `TrafficControlRecord`;
- control line;
- `MovementKey`;
- `ConflictZoneRecord`;
- occupancy interval;
- vertical compatibility 2.5D.

Architettura ammessa:

```text
canonical utilities / episode cache
              ├── RulebookMonitor
              └── ObservationBuilder
```

Architettura vietata:

```text
RulebookResult
      └── ObservationBuilder
```

L’observation builder può leggere memoria causale esplicitamente whitelisted, ma non costi o risultati aggregati.

## 3.1 Runtime causal-context contract

The environment owns one immutable `CausalSceneContext` per observation step.
It exposes only the frozen task route and route projection; lane, boundary,
control, and conflict-zone primitives; current signal states; current/past
actor snapshots; and explicitly whitelisted causal-memory fields. Static
fields are built and validated at reset. Dynamic fields are rebuilt from the
current or committed past state only.

`RulebookMonitor` and `ObservationBuilder` consume the same canonical
primitives. The observation builder may read only the whitelisted causal-memory
fields after their transactional commit; it must never receive a
`RulebookResult`, costs, margins, statuses, rewards, or diagnostics. A missing
or invalid required context field is a fail-closed scenario-validation error,
not a fallback to native navigation or future scenario data.

Sono inoltre ammesse primitive di **causal rule context**, purché prodotte senza consultare il risultato del monitor: associazione lane-control, `MovementKey` corrente, stato di occupazione già avvenuto, onset del giallo, stop/dashed timers e diritto di precedenza esplicitamente derivabile dalla mappa o dal controllo corrente. Queste primitive devono avere un fallback `unknown`/`undefined` e non possono essere retro-riempite usando il futuro del dataset.

# 4. Assigned ego route and map geometry

## 4.1 Motivazione

The ego route is a navigation mission known before control begins. Each source
must persist it as an offline task annotation. Waymo may obtain it by
map-matching the full SDC trajectory only in offline dataset preparation,
before the scenario is released to the runtime. PG route-annotation provenance
must be explicitly recorded. The resulting lane-ID sequence is immutable task
metadata, not an online future-state feature.

## 4.2 Costruzione

Before reset, every scenario must publish a non-empty, contiguous
`assigned_route_lane_ids` sequence. At reset the runtime validates that each
lane exists and concatenates its canonical map centerline into a
`RoutePolyline`; it then freezes the lane sequence and polyline for the full
episode. The runtime must not read any SDC sample or compare route alternatives.

The annotation's provenance is recorded in the dataset and experiment manifest:
an explicit PG assignment provenance or `waymo_sdc_offline_task_annotation`. It
is not included in the policy input. A missing, invalid, or non-contiguous assignment
is a fail-closed scenario-validation error. No native trajectory-navigation,
geometric nearest-route, or runtime future-SDC fallback is permitted.

## 4.3 Conseguenze

Dalla route assegnata e costruita su geometria di mappa devono derivare:

- navigation della baseline LiDAR;
- route tokens della semantic observation;
- route completion;
- distance to route goal;
- route-relative \(\Delta s\);
- route corridor;
- object relevance;
- traffic-control relevance;
- conflict-zone ordering.

# 5. Baseline `StackedLidarStateObservation`

## 5.1 Scopo

Questa osservazione è la baseline MetaDrive compatta. Mantiene la struttura della built-in `LidarStateObservation`, ma:

- usa route assegnata e costruita su geometria di mappa;
- configura esplicitamente i detector;
- disattiva la navigation degli altri veicoli;
- applica uno stack temporale;
- applica un rumore leggero ai ray measurements.

Non viene resa semanticamente equivalente alla custom observation e non riceve traffic-control tokens o conflict-zone tokens.

## 5.2 Configurazione finale

```yaml
observation:
  type: stacked_lidar_state
  implementation: custom_wrapper_over_metadrive_state_observation
  metadrive_revision: 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
  history_length: 5
  history_fill: repeat_first_valid_frame

  navigation:
    implementation: MapRouteNavigationObservation22
    source: assigned_route_metadata_and_map_geometry
    replace_native_scenario_navigation: true
    num_waypoints: 10
    waypoint_spacing_m: 5.0
    output_dim: 22
    future_sdc_trajectory: false

  lidar:
    num_lasers: 240
    distance_m: 50.0
    num_others: 4
    add_others_navi: false
    native_gaussian_noise: 0.0
    native_dropout_prob: 0.0

  side_detector:
    enabled: true
    num_lasers: 12
    distance_m: 50.0
    native_gaussian_noise: 0.0
    native_dropout_prob: 0.0

  lane_line_detector:
    enabled: true
    num_lasers: 12
    distance_m: 50.0
    native_gaussian_noise: 0.0
    native_dropout_prob: 0.0

  ray_noise_wrapper:
    enabled: true
    sole_noise_owner: true
    distribution: iid_gaussian_on_normalized_ray_fraction
    sigma_normalized: 0.001
    dropout_prob: 0.0
    clip: [0.0, 1.0]
    rng_source: engine.np_random
```

La classe non è la `LidarStateObservation` stock configurata soltanto tramite YAML. Deve sostituire la navigation nativa con un adapter custom da 22 dimensioni costruito sulla `RoutePolyline` della missione assegnata e su geometria di mappa. La revisione esatta di MetaDrive deve essere registrata nel manifest sperimentale e la dimensione per-frame deve essere verificata con un’asserzione runtime, così modifiche upstream non cambiano silenziosamente lo schema.

## 5.3 Perché attivare entrambi i detector

### SideDetector

Il `SideDetector` rileva:

- continuous lane lines;
- sidewalk/road-side boundaries.

Fornisce una descrizione ray-based dei limiti laterali locali.

### LaneLineDetector

Il `LaneLineDetector` rileva:

- continuous lane lines;
- broken lane lines;
- sidewalk.

Usando stesso numero di raggi, stessa fase angolare e stesso range del `SideDetector`, le due sequenze forniscono alla rete indizi per distinguere una marking visibile soltanto al lane-line detector da un confine visibile anche al side detector.

La distinzione non è una classe semantica esplicita. Rimane una baseline sensor-like.

### Alternative scartate

**Detector entrambi disattivi**

- vantaggio: baseline più vicina al vettore compatto tradizionale;
- svantaggio: nessuna informazione ray-based sulle lane markings; la policy non può distinguere adeguatamente presenza di marking e road-side structure.

**Solo SideDetector**

- mantiene road-boundary awareness;
- non rappresenta broken lane markings.

**Solo LaneLineDetector**

- rappresenta marking e sidewalk;
- non separa gli elementi rilevati dal subset continuous/sidewalk.

Per il rulebook corrente la configurazione migliore è quindi:

\[
\boxed{\text{SideDetector ON, LaneLineDetector ON}}.
\]

## 5.4 Rumore

MetaDrive rappresenta ogni ray measurement come hit fraction normalizzata in \([0,1]\). Il rumore è applicato come:

\[
\tilde r_j
=
\operatorname{clip}
\left(
r_j+\epsilon_j,
0,1
\right),
\qquad
\epsilon_j\sim\mathcal N(0,\sigma^2).
\]

Si fissa:

\[
\sigma=0.001.
\]

Su un ray range di \(50\,m\):

\[
0.001\cdot 50\,m = 0.05\,m.
\]

Il valore rappresenta quindi circa \(5\,cm\) di standard deviation equivalente sulla distanza massima.

La scelta è deliberatamente lieve:

- evita input perfettamente deterministici;
- è nello stesso ordine di grandezza della precisione di sensori LiDAR reali;
- non altera significativamente la geometria locale;
- non richiede tuning;
- evita di trasformare il confronto in uno studio completo di perception robustness.

## 5.5 Perché dropout zero

Si fissa:

\[
p_{\mathrm{drop}}=0.
\]

Un dropout uniforme indipendente non riproduce fedelmente i missing returns reali, che dipendono da:

- distanza;
- materiale;
- angolo d’incidenza;
- intensità;
- condizioni atmosferiche;
- occlusioni.

Inoltre, studi sim-to-real mostrano che un semplice Gaussian range noise può ridurre il domain gap, mentre un forte random downsampling può non migliorare e persino peggiorare le prestazioni.

Il dropout può essere usato soltanto come robustness ablation opzionale:

```yaml
dropout_prob: 0.01
```

ma non appartiene alla configurazione core.

## 5.6 Applicazione del rumore

L’ownership del rumore è univoca:

```text
MetaDrive native sensor noise = OFF
RayNoiseWrapper             = unico proprietario del rumore
```

Il wrapper applica una sola volta la stessa trasformazione ai tre gruppi ray-based:

- main LiDAR;
- SideDetector;
- LaneLineDetector.

```python
def perturb_rays(values, sigma, dropout, rng):
    noisy = values + rng.normal(0.0, sigma, size=values.shape)
    if dropout > 0:
        mask = rng.random(values.shape) < dropout
        noisy[mask] = 0.0
    return np.clip(noisy, 0.0, 1.0)
```

The wrapper samples noise from the seeded `engine.np_random` stream, so equal
environment seeds reproduce the ray perturbations. It must sample the raw
main-LiDAR, SideDetector, and LaneLineDetector blocks explicitly because the
checked-out MetaDrive implementation applies its native noise only to the main
LiDAR. The observation constructor must fail if any native sensor noise is
non-zero while `RayNoiseWrapper.enabled=true`, preventing double perturbation.

Non viene aggiunto rumore a:

- ego dynamics;
- navigation;
- nearby vehicle relative-state features.

Questa baseline non viene presentata come full perception error model.

## 5.7 Feature per frame

Quando i detector sono attivi, sostituiscono le rispettive feature scalari:

- SideDetector sostituisce le due road-boundary distances;
- LaneLineDetector sostituisce il lane-center offset.

| Gruppo | Feature | Dim. | Significato |
|---|---|---:|---|
| Ego | Heading difference | 1 | Differenza fra heading ego e riferimento locale |
| Ego | Speed | 1 | Velocità ego normalizzata |
| Ego | Steering | 1 | Comando/stato di sterzo corrente |
| Ego | Previous throttle/brake | 1 | Comando longitudinale precedente |
| Ego | Previous steering | 1 | Comando laterale precedente |
| Ego | Yaw rate | 1 | Velocità angolare ego |
| Navigation | Map-route waypoints | 20 | Dieci waypoint 2D nel frame ego |
| Navigation | Route lateral/heading state | 2 | Relazione locale con la route |
| Side detector | Side cloud points | 12 | Distanze ray-based a continuous line/sidewalk |
| Lane detector | Lane-line cloud points | 12 | Distanze ray-based a continuous/broken/sidewalk |
| Nearby vehicles | Relative state | 16 | 4 veicoli × \(x,y,v_x,v_y\) relativi |
| Main LiDAR | Occupancy rays | 240 | Hit fractions su 360° |
| **Totale** |  | **308** |  |

Quindi:

\[
D_{\mathrm{lidar}}
=
6+22+12+12+16+240
=
308.
\]

## 5.8 Stack

Si usa:

\[
H_{\mathrm{lidar}}=5.
\]

Lo stack contiene i cinque frame più recenti, dal meno recente al corrente:

\[
O_t^{\mathrm{lidar}}
=
[o_{t-4},o_{t-3},o_{t-2},o_{t-1},o_t].
\]

Dimensione:

\[
D_{\mathrm{lidar,stack}}
=
5\cdot308
=
1540.
\]

Al reset, il primo frame valido viene replicato fino a riempire lo stack. Non si usano frame zero artificiali, perché potrebbero essere confusi con misure fisiche valide.

## 5.9 Limiti dichiarati

La baseline non contiene esplicitamente:

- stato dei semafori;
- stop sign;
- control lines;
- `MovementKey`;
- conflict zones;
- pairwise priority;
- stop timer;
- yellow onset state.

È quindi information-limited rispetto ad alcune componenti di \(R_3\). Questa limitazione deve essere dichiarata e il confronto LiDAR–semantic non deve essere descritto come confronto a parità di informazione.

# 6. `SemanticStateObservationV2`

## 6.1 Capacità

```yaml
semantic_observation:
  history_length: 5
  history_fill: zero_and_mask_unavailable_past
  causal_track_cache_length: 5
  route_tokens: 10
  dynamic_tokens: 16
  dynamic_conflict_reserved_slots: 8
  dynamic_context_reserved_slots: 8
  allow_conflict_preemption_of_context: true
  persistence_subordinate_to_current_ranking: true
  static_tokens: 8
  traffic_control_tokens: 8
  interaction_tokens: 8

  dynamic_radius_m: 50.0
  static_radius_m: 50.0
  control_radius_m: 80.0
  interaction_prediction_horizon_s: 3.0
```

## 6.2 Struttura e dimensione

| Gruppo | Tensor shape | Feature dim. | Mask | Contributo |
|---|---:|---:|---:|---:|
| Ego history | \(5\times10\) | 10 | 5 | 55 |
| Ego current | \(1\times3\) | 3 | — | 3 |
| Route | \(10\times7\) | 7 | 10 | 80 |
| Dynamic actors | \(16\times5\times22\) | 22 | 80 | 1840 |
| Static objects | \(8\times13\) | 13 | 8 | 112 |
| Lane/road | \(1\times14\) | 14 | — | 14 |
| Traffic controls | \(8\times17\) | 17 | 8 | 144 |
| Conflict interactions | \(8\times35\) | 35 | 8 | 288 |
| Temporal compliance | \(1\times5\) | 5 | — | 5 |
| **Totale** |  |  |  | **2541** |

\[
\boxed{D_{\mathrm{semantic}}=2541}
\]

La tokenizzazione LQ produce:

\[
5+1+10+80+8+1+8+8+1=122
\]

raw tokens prima delle group-specific projections.

# 7. Feature della semantic observation

## 7.1 Ego history

Shape:

\[
E_t\in\mathbb R^{5\times10}.
\]

Ogni frame storico è riespresso nel frame ego corrente.

| Feature | Dim. | Significato | Normalizzazione |
|---|---:|---|---|
| Longitudinal velocity | 1 | Velocità ego lungo l’asse forward corrente | ego speed cap |
| Lateral velocity | 1 | Velocità laterale nel frame ego corrente | ego speed cap |
| Longitudinal acceleration | 1 | Accelerazione longitudinale recente | \(20\,m/s^2\) |
| Lateral acceleration | 1 | Accelerazione laterale recente | \(20\,m/s^2\) |
| Yaw rate | 1 | Velocità angolare verticale | \(1\,rad/s\) |
| Steering | 1 | Comando/stato di sterzo | range azione |
| Throttle/brake | 1 | Comando longitudinale firmato | range azione |
| Lane offset | 1 | Offset laterale firmato dalla lane associata | \(6\,m\) |
| Route heading error \(\sin\) | 1 | Seno dell’errore di heading rispetto alla route | già bounded |
| Route heading error \(\cos\) | 1 | Coseno dell’errore di heading rispetto alla route | già bounded |
| **Totale** | **10** |  |  |

Mask:

\[
m_t^E\in\{0,1\}^5.
\]

## 7.2 Ego current

Shape:

\[
E_t^{current}\in\mathbb R^3.
\]

| Feature | Dim. | Significato |
|---|---:|---|
| Ego length | 1 | Lunghezza del footprint ego |
| Ego width | 1 | Larghezza del footprint ego |
| Route completion | 1 | Frazione percorsa della route assegnata |
| **Totale** | **3** |  |

La dimensione del veicolo è necessaria per interpretare clearance e footprint-relative geometry. Route completion è calcolata soltanto sulla route canonica.

## 7.3 Route tokens

Shape:

\[
R_t\in\mathbb R^{10\times7}.
\]

I token sono campionati ogni circa \(5\,m\) davanti alla proiezione ego.

| Feature | Dim. | Significato |
|---|---:|---|
| Relative \(x,y\) | 2 | Posizione del route point nel frame ego |
| Tangent error \(\sin,\cos\) | 2 | Orientamento locale della route rispetto all’ego |
| Local curvature | 1 | Curvatura della route nel punto |
| Lane width | 1 | Larghezza della lane associata al punto |
| Relative route distance | 1 | Distanza curvilinea davanti all’ego |
| **Totale** | **7** |  |

Mask:

\[
m_t^R\in\{0,1\}^{10}.
\]

Vicino al goal i token mancanti vengono azzerati e mascherati.

## 7.4 Dynamic actors

Shape:

\[
D_t\in\mathbb R^{16\times5\times22}.
\]

| Feature | Dim. | Significato |
|---|---:|---|
| Relative position \(x,y\) | 2 | Posizione attore nel frame ego corrente |
| Relative velocity \(v_x,v_y\) | 2 | Velocità attore meno velocità ego |
| Relative heading \(\sin,\cos\) | 2 | Heading attore rispetto all’ego |
| Length, width | 2 | Footprint attore |
| Actor type | 4 | vehicle, pedestrian, cyclist, other |
| Lane relation | 5 | same, left-adjacent, right-adjacent, crossing/other, unknown |
| Route \(\Delta s\) | 1 | Differenza di proiezione curvilinea sulla route ego |
| Route lateral distance | 1 | Distanza laterale dalla route ego |
| CPA valid | 1 | Esiste un avvicinamento entro \(3\,s\) |
| \(t_{\mathrm{CPA}}\) | 1 | Tempo al closest point of approach |
| \(d_{\mathrm{CPA}}\) | 1 | Distanza al closest point of approach |
| **Totale** | **22** |  |

Non vengono passati:

- actor ID;
- track age;
- current Euclidean distance ridondante;
- absolute speed ridondante;
- conflict/context selection flag;
- future trajectory;
- other-agent route.

Mask:

\[
m_t^D\in\{0,1\}^{16\times5}.
\]

### CPA

\[
\mathbf p_i=\mathbf x_i-\mathbf x_e,
\qquad
\mathbf v_i=\dot{\mathbf x}_i-\dot{\mathbf x}_e.
\]

\[
\tau_i^{raw}
=
-\frac{\mathbf p_i^\top\mathbf v_i}
{\|\mathbf v_i\|^2+\epsilon}.
\]

\[
\tau_i
=
\operatorname{clip}
(\tau_i^{raw},0,T_{\mathrm{pred}}).
\]

\[
d_i^{CPA}
=
\|\mathbf p_i+\tau_i\mathbf v_i\|.
\]

`CPA valid=1` se:

\[
\mathbf p_i^\top\mathbf v_i<0
\land
0\le\tau_i^{raw}\le3\,s.
\]

Non vengono consultate future tracks.

## 7.5 Static objects

Shape:

\[
S_t\in\mathbb R^{8\times13}.
\]

| Feature | Dim. | Significato |
|---|---:|---|
| Relative position \(x,y\) | 2 | Posizione nel frame ego |
| Relative heading \(\sin,\cos\) | 2 | Orientamento dell’oggetto |
| Length, width | 2 | Ingombro planare |
| Static type | 5 | cone, barrier, wall/building, static vehicle, generic |
| Route \(\Delta s\) | 1 | Posizione curvilinea rispetto all’ego |
| Route lateral distance | 1 | Distanza dalla route corridor |
| **Totale** | **13** |  |

Mask:

\[
m_t^S\in\{0,1\}^{8}.
\]

## 7.6 Lane/road state

Shape:

\[
L_t\in\mathbb R^{14}.
\]

| Feature | Dim. | Significato |
|---|---:|---|
| Current lane width | 1 | Larghezza della lane associata |
| Left boundary distance | 1 | Signed footprint clearance al boundary sinistro |
| Right boundary distance | 1 | Signed footprint clearance al boundary destro |
| Left boundary type | 4 | solid, dashed, road edge, none/unknown |
| Right boundary type | 4 | solid, dashed, road edge, none/unknown |
| Left adjacent lane available | 1 | Lane raggiungibile a sinistra |
| Right adjacent lane available | 1 | Lane raggiungibile a destra |
| Current lane curvature | 1 | Curvatura locale della lane |
| **Totale** | **14** |  |

Il lane offset è già incluso nella ego history. Il dashed timer è nel temporal state. Le boundary distances sono definite come **signed footprint clearances**: positive prima del contatto, zero al contatto e negative durante l’overlap. Non sono distanze centro-veicolo–boundary.

## 7.7 Traffic controls

Shape:

\[
C_t\in\mathbb R^{8\times17}.
\]

Sono rappresentati:

- signal groups;
- stop groups.

Crosswalk e vehicle conflicts vengono rappresentati come interaction tokens.

| Feature | Dim. | Significato |
|---|---:|---|
| Control-line midpoint \(x,y\) | 2 | Centro della control line nel frame ego |
| Signed route distance | 1 | Distanza del front bumper dalla linea |
| Control-line direction \(\sin,\cos\) | 2 | Orientamento della linea |
| Control type | 2 | signal, stop |
| Signal state | 5 | green, yellow, red, flashing-yellow, not-signal |
| Controls ego movement | 1 | Associazione map/control-derived alla `MovementKey` ego corrente |
| Active control | 1 | Primo controllo pertinente non risolto |
| State valid | 1 | Stato corrente disponibile e valido |
| Yellow-onset distance | 1 | Distanza memorizzata all’inizio del giallo |
| Yellow-onset required stopping distance | 1 | Distanza di arresto richiesta all’onset |
| **Totale** | **17** |  |

Per uno stop:

- signal-state è `not-signal`;
- yellow-onset features sono zero.

Per un signal state invalido:

- `state_valid=0`;
- le signal one-hot sono zero;
- nel pool rulebook-based tale situazione deve normalmente essere esclusa dal validatore.

Non si passa direttamente `yellow_must_stop`.

## 7.8 Conflict interaction tokens

Shape:

\[
I_t\in\mathbb R^{8\times35}.
\]

Ogni token rappresenta una coppia:

```text
(zone_id, actor_id)
```

Esempi:

- crosswalk + pedestrian;
- crosswalk + cyclist;
- intersection conflict zone + vehicle;
- merge zone + vehicle;
- roundabout entry zone + circulating vehicle.

| Feature | Dim. | Significato |
|---|---:|---|
| Zone centroid \(x,y\) | 2 | Centroide relativo |
| Route distance to entry/exit | 2 | Distanza curvilinea alla zona |
| Zone type | 4 | crosswalk, intersection, merge, roundabout |
| Ego inside | 1 | Ego attualmente nella zona |
| Other inside | 1 | Attore associato nella zona |
| Other actor type | 3 | vehicle, pedestrian, cyclist |
| Ego \(t_{in},t_{out}\) | 2 | Predicted occupancy interval |
| Other \(t_{in},t_{out}\) | 2 | Predicted occupancy interval |
| Ego interval valid | 1 | Intervallo esistente entro \(3\,s\) |
| Ego open-end | 1 | Ego ancora nella zona a fine orizzonte |
| Other interval valid | 1 | Intervallo altro attore valido |
| Other open-end | 1 | Altro attore ancora nella zona |
| Ego approach control | 4 | none, stop, signal, unknown |
| Other approach control | 4 | none, stop, signal, unknown |
| Map/control-derived right-of-way | 3 | ego, other, undefined |
| Roundabout relation | 1 | Ego entry, altro sulla componente circolante |
| Preexisting occupancy active | 1 | Zona già occupata prima che l’evento fosse attribuibile alla policy |
| Incompatible entry latched | 1 | Ingresso incompatibile già avvenuto e ancora attivo |
| **Totale** | **35** |  |

Non vengono passati:

- \(q_{\mathrm{crosswalk}}\);
- \(q_{\mathrm{yield}}\);
- `must_yield`;
- rule margin;
- macro-rule status.

La feature `map/control-derived right-of-way` non è il risultato `must_yield` del monitor. Può essere valorizzata soltanto quando la relazione è esplicitamente determinabile da lane/approach corrente, traffic control corrente o regola di mappa dichiarata. Se la relazione dipende dalla futura svolta, futura uscita o futura lane dell’altro attore, il valore normativo è `undefined`.

### Occupancy intervals

Gli intervalli derivano dal modello dichiarato dal rulebook:

- constant velocity;
- constant heading;
- no rotation del footprint;
- horizon \(3\,s\);
- continuous SAT.

Sono predizioni online, non future ground truth.

### Derivazione causale del movimento degli altri attori

`Other approach control`, `MovementKey` e right-of-way dell’altro attore devono essere derivati esclusivamente da:

1. lane association corrente;
2. heading corrente e history osservata;
3. connettività della mappa;
4. traffic-control association corrente;
5. predizione cinematica dichiarata, quando necessaria.

Non è ammesso usare futura exit lane, futura posizione o futura traiettoria registrata. Se più movimenti restano compatibili con le informazioni correnti, il movimento è `unknown` e il right-of-way è `undefined`.

Here, ambiguity means that multiple future manoeuvres remain compatible with the
actor's current lane, heading, observed history, map connectivity, and current
controls. It does not mean uncertain current position or track identity. A
dedicated lane whose topology permits one manoeuvre is therefore not ambiguous.

An interaction token may be emitted only when its zone geometry is derivable
without choosing one of those future manoeuvres. If the zone geometry is known
but the priority relation depends on the unresolved manoeuvre, the token remains
present and its right-of-way feature is `undefined`. If identifying the zone
itself requires selecting a future exit lane, no pairwise token is emitted:
its payload is zero and its mask is zero. The actor remains eligible for the
dynamic tokens and the causal CPA ranking. Unions of hypothetical zones and
future-data disambiguation are forbidden.

## 7.9 Temporal compliance state

Shape:

\[
T_t\in\mathbb R^5.
\]

| Feature | Dim. | Significato |
|---|---:|---|
| Dashed boundary active | 1 | Sovrapposizione corrente con logical dashed boundary |
| Dashed timer | 1 | \(\tau_{\mathrm{dash}}/2\,s\) |
| Active stop zone | 1 | Ego nella fascia valida di arresto |
| Continuous stop timer | 1 | \(T_{\mathrm{cont}}/1\,s\) |
| Best stop timer | 1 | \(T_{\mathrm{best}}/1\,s\) |
| **Totale** | **5** |  |

Questi valori sono memoria causale, non leakage del reward.

# 8. Selezione degli elementi

## 8.1 Dynamic candidates

\[
\mathcal C_{\mathrm{dyn}}
=
\{
i:
i\text{ live},
d_i\le50\,m,
\text{verticalmente compatibile}
\}.
\]

Si usa:

\[
K_{\mathrm{dyn}}=16,
\qquad
K_{\mathrm{conf,res}}=8,
\qquad
K_{\mathrm{ctx,res}}=8.
\]

Le quote sono **slot riservati**, non limiti rigidi. I primi otto slot sono conflict-reserved e gli ultimi otto context-reserved. Slot conflict inutilizzati possono ospitare context actor; viceversa, un nuovo conflict actor può preemptare immediatamente un context slot se gli otto slot conflict sono occupati. La persistenza non può mai impedire l’ingresso di un candidato con priorità corrente superiore.

### Conflict quota

Chiave deterministica, in ordine:

1. attore associato a una conflict zone corrente;
2. CPA valid;
3. minore \(t_{\mathrm{CPA}}\);
4. minore \(d_{\mathrm{CPA}}\);
5. relation same/adjacent/crossing;
6. minore current distance;
7. actor ID stabile come tie-break.

### Context quota

Fra i candidati non già selezionati:

1. minore Euclidean distance;
2. actor ID stabile.

Il motivo di selezione non entra nel token.

> **Emendamento 2026-07-26 (ADR-026).** La regola precedente (same lane →
> adjacent lane → davanti sulla route → minore route lateral distance →
> minore Euclidean distance → actor ID) dava precedenza alla relazione di
> corsia rispetto alla distanza fisica, permettendo a un attore molto vicino
> ma su una corsia diversa di perdere uno slot a favore di un attore lontano
> sulla stessa corsia. La quota conflict-zone (sopra) resta invariata — CPA/
> TTC restano il criterio corretto per il piccolo insieme di attori
> effettivamente in conflitto attivo, coerentemente con l'evidenza che un
> criterio di interattività batte la distanza pura solo per N piccolo
> (Sun, Zhao, Sadigh, Zhan, Anguelov, *Identifying Driver Interactions via
> Conditional Behavior Prediction*, ICRA 2021). Per il pool generale
> (context quota), dove N può essere maggiore, si è scelto la distanza
> euclidea pura invece di estendere CPA/TTC a tutti gli attori, per due
> motivi: (1) è la pratica dominante negli encoder agent-centric per guida
> autonoma quando la capacità è satura (es. GameFormer, Wayformer — selezione
> per k-nearest); (2) il CPA assume estrapolazione a velocità costante, la
> stipula meno affidabile proprio per gli attori che stanno per sterzare,
> frenare o accelerare — gli stessi che il criterio ufficiale di selezione
> agenti del Waymo Open Motion Dataset individua come "di interesse" tramite
> variazione di heading, deviazione laterale e accelerazione. Dettagli
> completi e alternative valutate in ADR-026.

## 8.2 Static selection

Candidati:

- static collidable;
- verticalmente compatibili;
- entro \(50\,m\).

Ordine:

1. intersezione col route corridor corrente;
2. davanti o attualmente occupato;
3. minore route lateral distance;
4. minore Euclidean distance;
5. static ID.

## 8.3 Traffic-control selection

Candidati:

- control record pertinente o raggiungibile;
- entro \(80\,m\);
- verticalmente compatibile;
- non definitivamente irrilevante.

Ordine:

1. active control;
2. controls ego movement;
3. non ancora risolto;
4. minore signed route distance non negativa;
5. minore absolute route distance se già parzialmente superato;
6. control group ID.

## 8.4 Interaction selection

Candidati pairwise:

- crosswalk zone + VRU;
- vehicle-yield zone + vehicle;
- intervallo ego o attore valido, oppure zona già occupata/latched.

A candidate must additionally satisfy the causal zone-geometry rule in §7.8.
Pairs requiring an unresolved future exit lane are not candidates.

Ordine:

1. incompatible-entry latched;
2. ego/other inside;
3. intervalli sovrapposti;
4. preexisting occupancy active;
5. minore route distance all’entry;
6. minore \(t_{in}\) valido;
7. minore actor distance;
8. `(zone_id, actor_id)`.

## 8.5 Overflow diagnostics

Per dynamic, static, controls e interactions il builder deve produrre soltanto nel canale di logging/debug:

- numero totale di candidati;
- numero selezionato;
- numero scartato per capacità;
- numero di candidati conflict-critical scartati;
- chiave di ranking dell’ultimo selezionato e del primo escluso.

Questi valori non fanno parte dell’osservazione, del reward o del curriculum. Servono a verificare empiricamente che le capacità fissate non eliminino sistematicamente elementi rilevanti.

# 9. History e slot persistence

## 9.1 Frame

Gli snapshot passati vengono memorizzati in coordinate globali internamente. Al tempo \(t\), tutti i dati storici sono trasformati nel frame ego corrente:

\[
\mathbf p_{i,t-k}^{(ego_t)}
=
R(\psi_t)^\top
(\mathbf p_{i,t-k}^{global}-\mathbf p_{e,t}^{global}).
\]

La stessa trasformazione è applicata alle velocità.

Le coordinate globali non vengono esposte.

Il sistema mantiene una `causal_track_cache` di lunghezza massima \(H\) per **tutti gli attori effettivamente osservati** entro il dominio percettivo, non soltanto per quelli già selezionati. Quando un attore entra per la prima volta nei token selezionati, la sua history viene letta da questa cache. È vietato retro-riempirla consultando frame precedenti non osservati o future tracks dello `ScenarioDescription`.

## 9.2 Slot persistence

Per dynamic actors:

1. gli slot `0–7` sono conflict-reserved e gli slot `8–15` context-reserved;
2. un actor ID selezionato conserva preferibilmente lo stesso slot finché resta eleggibile nella stessa classe;
3. se non è osservabile in un frame, il token relativo è zero e la mask è zero;
4. lo slot può restare riservato fino a \(H\) step consecutivi di assenza soltanto se non serve a un candidato corrente più prioritario;
5. un candidato conflict può preemptare immediatamente il context slot meno prioritario quando necessario;
6. all’interno di ogni classe, un nuovo candidato con ranking superiore può preemptare l’occupante meno prioritario anche prima della grace period;
7. se un attore passa da context a conflict, viene promosso nel conflict bank e l’eventuale vecchio slot viene liberato atomicamente;
8. dopo \(H\) step di assenza uno slot non preemptato viene comunque liberato;
9. actor ID e slot ID non sono feature.

La risoluzione dei conflitti di slot è deterministica: ranking corrente, poi actor ID stabile come ultimo tie-break.

## 9.3 Reset

Al reset, la semantica è unica e normativa:

- il frame corrente ego contiene valori reali e mask \(1\);
- i quattro frame ego passati non disponibili sono zero e hanno mask \(0\);
- la `causal_track_cache` è vuota;
- dynamic slots vuoti e relative mask false;
- timer e latch inizializzati dalla procedura causale di reset, senza leggere eventi precedenti all’inizio dell’episodio;
- nessuna informazione pre-reset viene mantenuta.

La replica del primo frame valido è usata soltanto nella baseline LiDAR stacked; non è usata nella semantic observation.

# 10. Normalizzazione

## 10.0 Contratto delle unità

Prima della normalizzazione, tutte le primitive canoniche devono usare:

- metri per posizioni, distanze, dimensioni e clearance;
- secondi per tempi e timer;
- radianti per angoli;
- metri al secondo per velocità;
- metri al secondo quadrato per accelerazioni;
- radianti al secondo per yaw rate.

Ogni conversione da unità del dataset o del simulatore avviene prima della costruzione dei token. Nessun gruppo può applicare convenzioni di unità differenti.

## 10.1 Regole

Per quantità signed:

\[
\hat x
=
\operatorname{clip}
\left(
\frac{x}{s_x},
-1,1
\right).
\]

Per quantità non-negative:

\[
\hat x
=
\operatorname{clip}
\left(
\frac{x}{s_x},
0,1
\right).
\]

Angoli:

- seno/coseno;
- nessun raw angle discontinuo.

Categoriche:

- one-hot;
- nessun ordinal encoding.

## 10.2 Scale

| Quantità | Scala |
|---|---:|
| Dynamic/static position | \(50\,m\) |
| Route point position | \(50\,m\) |
| Control position/distance | \(80\,m\) |
| Interaction entry/exit distance | \(50\,m\) |
| CPA/occupancy time | \(3\,s\) |
| Ego speed | configured ego speed cap |
| Relative speed | ego cap + actor cap |
| Acceleration | \(20\,m/s^2\) |
| Yaw rate | \(1\,rad/s\) |
| Length | \(10\,m\) |
| Width | \(5\,m\) |
| Lane width/offset | \(6\,m\) |
| Curvature | \(0.2\,m^{-1}\) |
| Dashed timer | \(2\,s\) |
| Stop timer | \(1\,s\) |
| Route completion | già \([0,1]\) |

Se una scala è stimata empiricamente, deve usare soltanto il train split ed essere congelata prima di validation/test.

**Nota di chiarimento (actor cap):** quando l'attore osservato non espone una
capacità di velocità configurata (`configured_speed_cap_mps = None`, il caso
sistematico per pedoni e ciclisti, che non hanno un analogo di
`vehicle.max_speed_km_h`), l'`actor cap` usato nella scala "ego cap + actor
cap" DEVE ricadere sull'`ego cap`, non sulla velocità istantanea dell'attore
osservato. Far dipendere la scala dalla velocità corrente dell'attore
renderebbe la stessa velocità relativa fisica codificata in modo diverso a
seconda del comportamento transitorio dell'attore, con la distorsione
massima proprio negli incontri più critici (VRU che si avvicina a velocità
elevata). Questo chiarimento corregge un'implementazione preesistente in
`causal_semantic.py` che usava `max(ego cap, |velocità attore|, 1.0)` come
fallback; vedi
`docs/implementation/semantic_v3_actor_cap_normalization_fix_exec_plan.md`.

# 11. Masks, flattening e schema

## 11.1 Structured representation

Internamente:

```python
@dataclass(frozen=True)
class SemanticObservationBatch:
    ego_history: np.ndarray
    ego_history_mask: np.ndarray
    ego_current: np.ndarray

    route: np.ndarray
    route_mask: np.ndarray

    dynamic: np.ndarray
    dynamic_mask: np.ndarray

    static: np.ndarray
    static_mask: np.ndarray

    lane_road: np.ndarray

    controls: np.ndarray
    controls_mask: np.ndarray

    interactions: np.ndarray
    interactions_mask: np.ndarray

    temporal: np.ndarray
```

## 11.2 Flat adapter

Per SB3/MLP, un’unica `ObservationSchema` definisce:

- ordine dei gruppi;
- shape;
- slice;
- normalizzatore;
- mask slice.

Ordine flat normativo:

```text
ego_history
ego_history_mask
ego_current
route
route_mask
dynamic
dynamic_mask
static
static_mask
lane_road
controls
controls_mask
interactions
interactions_mask
temporal
```

Ogni modifica allo schema incrementa `schema_version`.

## 11.3 LQ adapter

L’LQ encoder riceve i gruppi strutturati. Le mask non vengono proiettate come normali feature, ma usate come attention mask.

Il flat vector non deve essere reshaped usando magic numbers sparsi nel codice. Tutte le shape derivano da `ObservationSchema`.

# 12. Sincronizzazione per-step

Ordine:

```text
1. costruzione obs_t dalla memoria committata
2. policy produce action_t
3. cattura pre_state
4. env.step(action_t)
5. cattura post_state
6. evaluate the rulebook transition and derive the next CausalSceneContext
   exclusively from pre_state, post_state, canonical primitives, and committed
   causal memory
7. validazione atomica
8. commit next_memory, cache_delta, and next CausalSceneContext
9. costruzione obs_{t+1}
10. store transition
```

La `obs_{t+1}` deve quindi riflettere:

- timer aggiornati;
- signal onset memory aggiornata;
- stop timers aggiornati;
- interaction latches aggiornati.

Non deve esistere un ritardo di uno step fra reward e memoria osservabile.
The observation builder receives the committed context, not the rulebook result.

# 13. Test obbligatori

## 13.1 Anti-leakage

1. Cambiare future tracks degli altri lasciando invariato presente/passato: observation invariata.
2. Cambiare future SDC trajectory after the offline task annotation is frozen:
   observation invariata.
3. Cambiare future traffic-light sequence: observation corrente invariata.
4. Cambiare arm, source e usefulness: observation invariata.
5. Cambiare reward scalarization: observation invariata.
6. Verificare nessun accesso a indice temporale \(>t\).
7. Verificare `add_others_navi=False`.
8. Nessun actor ID nel vettore.
9. Nessun selection flag.
10. Nessun rule cost/margin/status.
11. Cambiare futura lane/uscita di un altro attore senza modificare presente e passato: `MovementKey`, approach control e right-of-way correnti invariati.
12. In caso di movimento ambiguo, verificare `unknown`/`undefined` e nessun accesso al futuro per disambiguare.
13. If a zone requires an unresolved future exit lane, no interaction token is
    emitted; changing only that future exit cannot change the current
    observation.

## 13.2 Equivarianza

1. Traslare globalmente la scena: observation ego-centric equivalente.
2. Ruotare globalmente la scena: observation relativa equivalente.
3. Route projection su self-intersection usa tie-break canonico.
4. Cavalcavia/sottopasso non genera lane o interaction tokens errati.

## 13.3 History

1. Same actor mantiene lo slot.
2. Actor assente produce mask zero.
3. Slot liberato dopo \(H\) assenze.
4. Reset pulisce tutti gli slot.
5. Frame storici riespressi nel frame ego corrente.
6. Nessuna contaminazione fra episodi.
7. Attore osservato ma non selezionato conserva history causale nella track cache.
8. Nessun backfill da frame non osservati o future tracks.
9. Nuovo conflict actor preempta un context slot quando necessario.
10. La grace period non blocca candidati con ranking superiore.
11. Promozione context→conflict è atomica e non duplica l’attore.

## 13.4 LiDAR baseline

1. Dimensione single frame uguale a 308.
2. Dimensione stack uguale a 1540.
3. Detector ON sostituiscono scalar boundary/lane features.
4. 240 raggi coprono l’intero scan orizzontale.
5. Noise seed riproducibile.
6. \(\sigma=0.001\) mantiene valori in observation space.
7. `dropout_prob=0`.
8. No nearby-vehicle navigation.
9. Navigation deriva dalla route assegnata custom da 22 dimensioni e sostituisce quella nativa.
10. Stack ordering corretto.
11. Rumore nativo MetaDrive nullo per tutti e tre i sensori ray-based.
12. Il wrapper è l’unico proprietario del rumore e non avviene doppia perturbazione.
13. Revisione MetaDrive registrata nel manifest sperimentale.
14. Modifica simulata della dimensione upstream causa failure dell’asserzione runtime.
15. The wrapper uses the seeded `engine.np_random` stream and perturbs exactly
    the three declared ray blocks once each.

## 13.5 Semantic dimension

1. Ogni group shape coincide con lo schema.
2. Flat dimension uguale a 2541.
3. Token count LQ uguale a 122.
4. Padding sempre accompagnato da mask zero.
5. Feature finite e bounded.
6. Categorical one-hot valide.
7. Nessun gruppo oltre capacità.

## 13.6 Visual checks

Su almeno 20 scenari Waymo e 20 PG visualizzare:

- route points;
- selected dynamic actors;
- selected static objects;
- lane boundaries;
- control lines;
- conflict zones;
- interaction pair;
- current masks;
- slot IDs solo nel debug renderer;
- contatori overflow e primo candidato escluso per ciascun gruppo.

Gli ID e le diagnostiche di debug non entrano nell’osservazione.

# 14. Configurazione consolidata

```yaml
observations:
  schema_version: 1.1-final

  common:
    control_timestep_s: 0.1
    history_length: 5
    current_frame: ego_current
    vertical_compatibility_tolerance_m: 3.0
    future_ground_truth: forbidden
    other_agent_navigation: forbidden
    route_source: assigned_route_metadata_and_map_geometry
    output_dtype: float32
    units:
      distance: meter
      time: second
      angle: radian
      speed: meter_per_second
      acceleration: meter_per_second_squared
      yaw_rate: radian_per_second

  map_route:
    route_source: assigned_route_lane_ids
    pg_route_provenance: explicit_offline_task_annotation
    waymo_route_provenance: waymo_sdc_offline_task_annotation
    runtime_future_sdc_access: false
    invalid_route_policy: exclude_scenario_fail_closed
    waypoint_spacing_m: 5.0
    route_tokens: 10

  lidar_state:
    class: StackedLidarStateObservation
    implementation: custom_wrapper_over_metadrive_state_observation
    metadrive_revision: 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
    history_fill: repeat_first_valid_frame

    navigation:
      implementation: MapRouteNavigationObservation22
      replace_native_scenario_navigation: true
      output_dim: 22

    lidar:
      num_lasers: 240
      distance_m: 50.0
      num_others: 4
      add_others_navi: false
      native_gaussian_noise: 0.0
      native_dropout_prob: 0.0

    side_detector:
      enabled: true
      num_lasers: 12
      distance_m: 50.0
      native_gaussian_noise: 0.0
      native_dropout_prob: 0.0

    lane_line_detector:
      enabled: true
      num_lasers: 12
      distance_m: 50.0
      native_gaussian_noise: 0.0
      native_dropout_prob: 0.0

    ray_noise:
      enabled: true
      owner: RayNoiseWrapper
      sole_noise_owner: true
      gaussian_sigma_normalized: 0.001
      dropout_prob: 0.0
      clip_min: 0.0
      clip_max: 1.0
      rng_source: engine.np_random

    single_frame_dim: 308
    stacked_dim: 1540

  semantic:
    class: SemanticStateObservationV2
    history_fill: zero_and_mask_unavailable_past
    causal_track_cache_length: 5
    no_history_backfill: true
    flat_dim: 2541
    lq_raw_token_count: 122

    capacities:
      ego_history: 5
      route: 10
      dynamic_total: 16
      dynamic_conflict_reserved: 8
      dynamic_context_reserved: 8
      allow_conflict_preemption_of_context: true
      persistence_subordinate_to_current_ranking: true
      static: 8
      controls: 8
      interactions: 8

    lane_road_geometry:
      boundary_distance: signed_footprint_clearance

    radii_m:
      dynamic: 50.0
      static: 50.0
      controls: 80.0

    prediction:
      horizon_s: 3.0
      cpa_model: constant_velocity
      occupancy_model: canonical_rulebook_continuous_sat

    causal_context:
      provider: environment_owned_causal_scene_context
      static_fields_validated_at_reset: true
      invalid_required_field_policy: exclude_scenario_fail_closed
      other_movement_source: current_lane_heading_history_map_control_only
      ambiguous_other_movement: unknown
      right_of_way_source: explicit_map_or_current_control_only
      ambiguous_right_of_way: undefined
      future_disambiguation: forbidden

    overflow_diagnostics:
      enabled: true
      policy_input: false
      log_first_excluded: true
      log_critical_excluded_count: true

    include:
      causal_rule_memory: true
      rule_costs: false
      rule_margins: false
      rule_status: false
      selection_flags: false
      actor_ids: false
      future_tracks: false
      dense_lidar: false
```

# 15. Estensioni non core

Non fanno parte della specifica principale:

- `LidarRuleAwareObservation`;
- dense LiDAR rays nella semantic observation;
- RGB/BEV fusion;
- learned perception error model;
- occlusion-aware semantic observation;
- correlated temporal LiDAR noise;
- material-dependent dropout;
- weather-dependent LiDAR degradation;
- recurrent policy in sostituzione della history esplicita;
- pretraining autoencoder;
- behavior-cloning warm start;
- LQ tokenizzazione dei 240 raggi individuali.

Eventuali esperimenti di robustezza possono usare:

```text
clean evaluation: sigma = 0
nominal evaluation: sigma = 0.001
stress evaluation: sigma = 0.002
optional dropout stress: p = 0.01
```

Queste condizioni non modificano la configurazione nominale.

# 16. Riferimenti principali

1. Li, Q. et al., *MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning*, IEEE TPAMI, 2022.
2. Huch, S. et al., *Quantifying the LiDAR Sim-to-Real Domain Shift: A Detailed Investigation Using Object Detectors and Analyzing Point Clouds at Target-Level*, IEEE Transactions on Intelligent Vehicles, 2023, DOI 10.1109/TIV.2023.3251650.
3. Charraut, V. et al., *V-Max: Learning to Drive from Real-World Video*, 2025.
4. Renz, K. et al., *PlanT: Explainable Planning Transformers via Object-Level Representations*, 2022.
5. Specificazione del rulebook `v4.6-final-implementation-complete`.
6. Trascrizione della riunione del 4 luglio 2026.
7. Nayakanti, N. et al., *Wayformer: Motion Forecasting via Simple & Efficient Attention Networks*, 2022.
8. Jiang, B. et al., *VAD: Vectorized Scene Representation for Efficient Autonomous Driving*, ICCV, 2023.

# Approval Record

- Status: `APPROVED`
- Authoritative: `YES`
- Approval date: `2026-07-16`
- Approved by: user
- Approval evidence: explicit user message in this Codex conversation:
  “approvo”
- Amendment approval evidence: explicit user message in this Codex conversation
  on `2026-07-17`: “va bene”, approving offline SDC route annotation for both
  PG and Waymo as recorded in ADR-004.
- Scope: OBS-V1.1 route-assignment amendment recorded by ADR-004; ENC-V1.0 is
  governed by its own approval record.
