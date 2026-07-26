# Specification: Metrica R1 — Costo di collisione ancorato a curve di rischio lesivo da letteratura

## Metadata

- Feature: `r1_injury_risk_collision_cost`
- Specification ID: `rulebook-v2-r1-injury-risk`
- Version: `4.9`
- Status: `APPROVED`
- Date: `2026-07-26`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete` (solo per il sottoinsieme §5.4/§5.6/§5.8 modificato da questa versione; il resto di v4.7 resta autoritativo e invariato)
- Related specifications:
  - `docs/specifications/rulebook_v4.7_specification.md` (base per §5.1-5.3, §5.5, §5.7, §5.9 non toccati da questa versione)
  - `docs/specifications/rulebook_v4.8_specification.md` (amendment indipendente su §6.4/R2; nessuna interazione)
  - `docs/specifications/rulebook_scalarization_v1.0_specification.md` (consumatore a valle, invariato)
- Related ADRs: `docs/decisions/ADR-027-r1-injury-risk-collision-cost.md`
- Authoritative: YES

## 1. Purpose And Context

### 1.1 Il difetto della formulazione v4.7

La specifica v4.7 §5.4 definisce il costo bounded di `R1` come

```
q_collision,i = max{ eps_col , ( min(u_i, u_cap,i) / u_cap,i )^2 }
```

dove `u_i` è la componente normale pre-state del closing speed e il
normalizzatore è

```
u_cap,i = v_max,e + v_max,i   (veicolo)
u_cap,i = v_max,e             (VRU / statico)
```

cioè i **configured speed normalization cap** dell'ego e dell'attore.

Questi cap sono proprietà della configurazione dello scenario, non costanti
fisiche. Di conseguenza il costo di `R1` **dipende dallo scenario a parità di
severità fisica reale dell'urto**. Controesempio numerico verificabile con la
formula v4.7:

| Scenario | `v_max,e` | `u_i` | `q_collision` v4.7 |
|---|---|---|---|
| residenziale | 5 m/s | 4 m/s | `(4/5)^2 = 0.640` |
| autostradale | 20 m/s | 15 m/s | `(15/20)^2 = 0.5625` |

L'urto a 15 m/s (≈54 km/h) risulta **meno grave** dell'urto a 4 m/s
(≈14 km/h), pur avendo energia d'impatto e rischio lesivo reale
enormemente superiori. L'inversione non è marginale: si presenta ogni volta
che due scenari con cap diversi vengono confrontati o aggregati.

Le conseguenze non sono confinate alla reward:

1. la media di `R1` su un evaluation set eterogeneo (Waymo urbano + PG) non
   è più interpretabile, perché somma costi normalizzati su divisori
   diversi;
2. l'obiettivo `J_1(pi) = E[ sum_t gamma^t m_1(t) ]`, che un learner
   lessicografico ottimizza **direttamente** senza passare per lo
   scalarizzatore, eredita l'inversione senza alcun filtro;
3. nessun'altra sotto-metrica del rulebook ha questa proprietà: TTC
   (v4.7 §6.3) e clearance VRU (v4.7 §6.4) usano soglie **fisse per classe
   di attore**, indipendenti dalla configurazione dello scenario.

### 1.2 Origine della formulazione v4.7 e limiti della letteratura di base

v4.7 §5.6 dichiara la formula come *derivazione originale della tesi*,
ispirata a ScenicRules [10], che per le collisioni usa perdita/guadagno di
energia cinetica

```
VS = sum_j E_ego_loss + E_vru_j_gain          (collisioni con VRU)
VS = sum_i E_ego_loss + E_vehicle_i_loss      (collisioni con veicoli)
```

— una quantità fisica **non normalizzata** e illimitata. Il quadrato del
closing speed normale di v4.7 ne è la versione bounded, e il cap
configurato è esattamente il pezzo aggiunto per ottenere il bound. Il
difetto di §1.1 nasce quindi da un requisito implementativo (limitare il
costo a `[0,1]`), non da una scelta scientifica sostenuta da una fonte.

Si osserva inoltre che ScenicRules tratta collisioni con VRU e collisioni
con veicoli come **due regole distinte a priorità diverse**, non come
un'unica metrica parametrizzata per classe di attore.

### 1.3 La sostituzione adottata

Questa versione sostituisce la mappatura `u_i -> q_collision,i` con la
**curva di rischio lesivo** di Lubbe, Wu e Jeppsson (2022) [11], una
regressione logistica binaria pesata costruita sul German In-Depth Accident
Study (GIDAS, casi 1999-2020), che modella la probabilità di lesione in
funzione della **closing speed** e dell'età dell'utente della strada.

La scelta è motivata da tre proprietà della fonte, non da convenienza:

1. **la variabile indipendente è la closing speed**, definita dagli autori
   come *"the relative speed between two crash partners"* — esattamente la
   grandezza che il monitor già calcola in v4.7 §5.3. Non è richiesta
   alcuna traduzione di unità, alcun modello di massa, alcun `delta-v`;
2. **una sola fonte copre tutte le classi di attore rilevanti** (pedone,
   ciclista, motociclista, conducente auto) con lo stesso dataset, la
   stessa metodologia e le stesse variabili esplicative, quindi le curve
   sono confrontabili fra loro *per costruzione*. Mescolare fonti diverse
   per classi diverse reintrodurrebbe l'incomparabilità che questa versione
   elimina;
3. **il codominio della logistica è `(0,1)`**, quindi il vincolo
   `q_collision,i in [0,1]` del contratto rulebook è soddisfatto per
   costruzione, senza cap, senza clipping e senza alcuna costante di
   normalizzazione da tarare.

Il costo di `R1` diventa quindi una grandezza con significato fisico
esplicito: **la probabilità che l'urto produca una lesione almeno seria**.
Questo rende i costi `R1` confrontabili fra scenari *e* fra classi di
attore, perché esprimono tutti la stessa quantità.

### 1.4 Confine fra letteratura e adattamento di progetto

La letteratura fornisce: la forma funzionale (Eq. 1 di [11]), i
coefficienti di regressione (Tabelle 2, 3 e 5 di [11]), il livello di
severità di riferimento e la convenzione sull'età usata per il confronto
fra utenti (§4.3 di [11]).

Sono adattamenti originali di questo progetto, non conseguenze dirette
della fonte:

- l'uso della curva come **funzione di costo istantanea per transizione**,
  mentre [11] la usa per definire limiti di velocità a livello di sistema;
- l'alimentazione della curva con la **componente normale** del closing
  speed anziché il suo modulo (§4.2);
- la mappatura `STATIC_COLLIDABLE -> curva conducente auto` (§4.4);
- il mantenimento del floor numerico `eps_col` di v4.7 §5.5.

## 2. Scope

### In Scope

- Sostituzione di `q_collision,i` (v4.7 §5.4) con la curva di rischio
  lesivo MAIS3+F di [11].
- Rimozione di `u_cap,i` dal calcolo del costo.
- Definizione della mappatura `ActorClass -> curva` (§4.4).
- Aggiornamento dei diagnostics di `R1` (§6).
- Aggiornamento del comportamento atteso di v4.7 §5.8 (§7).

### Out Of Scope

- Tutto il resto di `R1`: significato (v4.7 §5.1), collision onset e
  deduplicazione (§5.2), definizione della normale e della velocità normale
  pre-state (§5.3), floor numerico (§5.5), fattibilità e contact hook
  (§5.9). Restano quelli già approvati e invariati.
- L'aggregazione `c_1(t) = max_i q_collision,i` — invariata.
- `R2`, `R3`, `R4` e la gerarchia `R1 > R2 > R3 > R4` — invariati.
- La scalarizzazione (`SCAL-V1.0`) — invariata; consuma `m_1` senza sapere
  come è prodotto.
- Le regole di eleggibilità degli scenari basate sui configured speed cap
  (§4.6) — invariate deliberatamente.
- L'introduzione di una classe attore `MOTORCYCLIST` — non implementata,
  si veda §9.2.

### Optional Or Deferred

- Adozione della forma logistica anche per altre sotto-metriche del
  rulebook — non pianificata da questa specifica.
- Uso dei coefficienti MAIS2+F o `Fatal` di [11] come ablation — possibile,
  richiede una nuova identità di run, non è nel core sperimentale.

## 3. Terminology, Assumptions, And Preconditions

Riusa integralmente la terminologia di v4.7 §2.x e §5.1-5.3. In
particolare restano invariati e sono precondizioni di questa specifica:

- `C_t^new`: insieme degli onset di contatto deduplicati per actor ID
  durante la transizione `x_t -> x_{t+1}` (v4.7 §5.2);
- `n_i`: asse unitario pre-state dal centro del footprint canonico ego
  verso quello dell'attore (v4.7 §5.3);
- `u_i = [ (v_e^- - v_i^-)^T n_i ]_+`: componente normale non negativa del
  closing speed, in `m/s`, valutata sul `pre_state` (v4.7 §5.3);
- `v_i^- = 0` per `STATIC_COLLIDABLE` (v4.7 §5.3);
- `eps_col = 1e-6`: floor numerico (v4.7 §5.5);
- comparison tolerance del monitor: `1e-8` (v4.7 §5.5).

Nuovi simboli introdotti da questa versione:

| Simbolo | Significato | Dominio |
|---|---|---|
| `beta_0(tau)` | intercetta della logistica per la classe `tau` | reale |
| `beta_v(tau)` | coefficiente closing speed, per `km/h` | reale positivo |
| `beta_a(tau)` | coefficiente età, per anno | reale positivo |
| `A` | età di riferimento congelata | `65` anni |
| `K` | conversione `m/s -> km/h` | `3.6` |
| `P_tau(u)` | rischio MAIS3+F per classe `tau` a closing speed `u` | `(0,1)` |

## 4. Mathematical Contract

### 4.1 Forma funzionale

Per una classe di attore `tau` e una closing speed normale `u` in `m/s`:

```
z_tau(u) = beta_0(tau) + beta_v(tau) * K * u + beta_a(tau) * A

P_tau(u) = 1 / (1 + exp(-z_tau(u)))
```

Questa è l'Equazione (1) di [11] valutata con le variabili esplicative
`(closing speed, age)` del Modello 2 di [11] §3.2.

Il costo bounded di `R1` diventa:

```
q_collision,i = max{ eps_col , P_{tau(i)}(u_i) }
```

L'aggregazione resta quella di v4.7 §5.4:

```
c_1(t) = 0                             se C_t^new = {}
c_1(t) = max_{i in C_t^new} q_collision,i    altrimenti

m_1(t) = -c_1(t)
```

### 4.2 Closing speed: componente normale, non modulo

[11] definisce la closing speed come modulo della velocità relativa fra i
due partner dell'urto. Questa specifica alimenta la curva con la
**componente normale** `u_i` già definita in v4.7 §5.3, non con il modulo.

Motivazione: v4.7 §5.3 introduce la proiezione sulla normale
deliberatamente, *"per evitare di equiparare uno sfioramento tangenziale a
un impatto frontale"*, ed è una proprietà approvata che questa versione non
intende perdere. Poiché `u_i <= ||v_e^- - v_i^-||`, la sostituzione è
**conservativa in difetto**: il rischio stimato non supera mai quello che
[11] assocerebbe allo stesso urto. La deviazione è dichiarata qui come
limitazione nota (§9.1), non nascosta.

### 4.3 Coefficienti congelati

Livello di severità: **MAIS3+F** — lesione almeno seria (AIS 3 o superiore
in almeno una regione corporea) oppure esito fatale, secondo la revisione
2015 dell'Abbreviated Injury Scale.

Età di riferimento: **`A = 65` anni**, costante e identica per tutte le
classi.

Coefficienti, da [11] Tabelle 2 (pedone), 3 (ciclista) e 5 (conducente
auto), righe `MAIS3+F`:

| Classe `tau` | `beta_0` | `beta_v` | `beta_a` | Fonte |
|---|---|---|---|---|
| `PEDESTRIAN` | `-6.190` | `0.078` | `0.038` | [11] Tab. 2 |
| `CYCLIST` | `-7.467` | `0.079` | `0.047` | [11] Tab. 3 |
| `VEHICLE` | `-7.654` | `0.041` | `0.021` | [11] Tab. 5 |
| `STATIC_COLLIDABLE` | `-7.654` | `0.041` | `0.021` | [11] Tab. 5, per analogia (§4.4) |

I coefficienti sono **congelati** per tutti gli esperimenti conformi. Non
sono iperparametri, non vengono tarati, non possono essere sovrascritti per
algoritmo, arm di curriculum, sorgente dati o seed. Una loro modifica
richiede una nuova specifica approvata e una nuova identità di run.

I coefficienti sono registrati nella forma pubblicata (per `km/h`), non
pre-moltiplicati, in modo che ogni valore sia verificabile per confronto
diretto con le tabelle della fonte.

**Verifica di correttezza della trascrizione.** Con `A` pari all'età
mediana per classe riportata da [11] (46 pedoni, 39 ciclisti, 39
conducenti), la formula di §4.1 riproduce gli ancoraggi pubblicati da [11]
§4.4 (closing speed al 10% di rischio MAIS3+F):

| Classe | Calcolato | Pubblicato in [11] |
|---|---|---|
| `PEDESTRIAN` | 28.8 km/h | 29 km/h |
| `CYCLIST` | 43.5 km/h | 44 km/h |
| `VEHICLE` | 113.1 km/h | 112 km/h |

Questa verifica è normativa e deve essere riprodotta da un test
(`AC-R1-05`). Non usa `A = 65` perché il suo scopo è validare la
trascrizione dei coefficienti contro i numeri pubblicati, non definire il
costo di esercizio.

### 4.4 Mappatura `ActorClass -> curva`

- `PEDESTRIAN -> ` curva pedone di [11];
- `CYCLIST -> ` curva ciclista di [11];
- `VEHICLE -> ` curva conducente auto di [11];
- `STATIC_COLLIDABLE -> ` curva conducente auto di [11], **per analogia**.

La mappatura `STATIC_COLLIDABLE` è un adattamento di progetto. [11] non
modella urti contro ostacoli fissi. Il razionale è che in un urto ego
contro oggetto fisso il soggetto esposto al rischio è l'occupante
dell'ego, e la curva `car driver` di [11] è appunto una curva di lesione
per occupante di autovettura.

La limitazione va dichiarata nella sua direzione: la curva `car driver` di
[11] descrive urti contro il **frontale di un'altra autovettura**, struttura
deformabile che assorbe energia, mentre un ostacolo rigido concentra il
carico. A parità di closing speed il rischio reale contro ostacolo rigido è
quindi **superiore** a quello stimato dalla curva: l'approssimazione
**sottostima** il costo. Si veda §9.1.

Un attore `INFRASTRUCTURE_NON_COLLIDABLE` non genera onset di contatto e
non raggiunge questa metrica; una classe non mappata è un errore fatale
(§8).

### 4.5 Ruolo del floor numerico

`eps_col = 1e-6` resta invariato (v4.7 §5.5). Poiché `P_tau(u) > 0` per
ogni `u` finito, e in particolare `P_tau(0) >= 1.85e-3` per tutte le classi
mappate, il floor **non è più attivo in nessun caso raggiungibile**: la
proprietà che garantiva (ogni nuovo contatto ha costo strettamente
positivo) è ora garantita strutturalmente dalla forma logistica.

Il floor viene mantenuto come rete di sicurezza numerica difensiva e per
non modificare v4.7 §5.5, che resta fuori scope. Non rappresenta una soglia
di danno e non viene tarato.

### 4.6 I configured speed cap restano precondizioni di eleggibilità

`u_cap,i` esce dal calcolo del costo, ma le regole di **eleggibilità dello
scenario** di v4.7 §5.4 restano invariate e devono continuare a essere
applicate: un ego o un veicolo live che non espone un configured speed
normalization cap valido rende lo scenario non eleggibile.

Questa è una scelta deliberata di contenimento del rischio: l'eleggibilità
degli scenari è fuori dallo scope di questa versione, i cap servono
comunque ad altri componenti (`progress` v4.7 §8, `wrongway` v4.7 §7.2) e
modificarne il contratto qui accoppierebbe due cambiamenti indipendenti.
Il codice conserva quindi la validazione ma non usa il valore per il costo
di `R1`.

## 5. Riferimento numerico

Costo `q_collision,i` con `A = 65`, MAIS3+F, per closing speed normale:

| `u` (m/s) | `PEDESTRIAN` | `CYCLIST` | `VEHICLE` / `STATIC` |
|---|---|---|---|
| 0 | 0.0237 | 0.0120 | 0.0019 |
| 2 | 0.0408 | 0.0210 | 0.0025 |
| 5 | 0.0898 | 0.0479 | 0.0039 |
| 10 | 0.2866 | 0.1725 | 0.0081 |
| 15 | 0.6206 | 0.4636 | 0.0167 |
| 20 | 0.8694 | 0.7818 | 0.0343 |
| 25 | 0.9644 | 0.9369 | 0.0692 |
| 30 | 0.9910 | 0.9840 | 0.1346 |

Tutti i valori sono in `(0,1)`, monotoni crescenti in `u`, e ordinati
`PEDESTRIAN > CYCLIST > VEHICLE` a parità di `u` — coerente con il ranking
di vulnerabilità stabilito da [11] §4.3 (*"from lowest to highest
vulnerability: drivers, motorcyclists, cyclists, and finally
pedestrians"*).

## 6. Diagnostics

Il componente `collision` deve esporre, in aggiunta a quanto già previsto
da v4.7:

- `raw.actors[].closing_speed_mps`: `u_i` (nuovo, sostituisce l'esposizione
  del solo quadrato);
- `raw.actors[].actor_class`: la classe usata per selezionare la curva;
- `raw.actors[].injury_risk_curve`: identificatore della curva applicata
  (utile perché `VEHICLE` e `STATIC_COLLIDABLE` condividono la stessa);
- `diagnostics.injury_risk_model`: identità del modello
  (`severity`, `age_years`, riferimento bibliografico).

`raw.actors[].raw_speed_squared` di v4.7 viene rimosso: non entra più in
nessun calcolo e conservarlo suggerirebbe una severità che non è più
quella adottata. `raw.worst_raw_closing_speed_squared` è sostituito da
`raw.worst_closing_speed_mps`.

L'evento binario `new_collision` resta invariato nei diagnostics.

## 7. Comportamento atteso

Sostituisce v4.7 §5.8:

- nessun nuovo contatto: `0`;
- contatto già attivo: `0` nello step successivo;
- nuovo impatto a closing speed nulla: costo piccolo ma strettamente
  positivo, pari a `P_tau(0)` (dipendente dalla classe);
- nuovo impatto a bassa closing speed: costo piccolo e crescente;
- impatti a closing speed elevata: costo che tende a `1` senza mai
  raggiungerlo;
- a parità di closing speed: `PEDESTRIAN > CYCLIST > VEHICLE`;
- **a parità di closing speed e classe, il costo è identico in ogni
  scenario** — proprietà che v4.7 non aveva ed è la ragione di questa
  versione;
- impatti multipli simultanei: massimo.

## 8. Errori

- Classe di attore non presente nella mappatura di §4.4: errore fatale
  `RulebookEvaluationError`, nessun fallback e nessuna curva di default.
- `u_i` non finito: errore fatale (comportamento v4.7 invariato).
- Costo risultante non finito o fuori da `[0,1]`: errore fatale.
- Restano invariati tutti gli errori fatali di v4.7 §5.3 (centri
  coincidenti, velocità pre-state mancante, cap non valido).

Non esiste alcun fallback fra curve.

## 9. Limitazioni note

### 9.1 Limitazioni dichiarate

1. **Componente normale invece del modulo** (§4.2): il costo sottostima il
   rischio che [11] assocerebbe allo stesso urto, in misura crescente con
   l'obliquità dell'impatto.
2. **`STATIC_COLLIDABLE` per analogia** (§4.4): sottostima il rischio reale
   contro ostacoli rigidi non deformabili.
3. **Età fissa a 65 anni**: il simulatore non modella l'età degli utenti
   della strada. Il valore è una convenzione di confronto, non una
   proprietà della scena.
4. **Dati tedeschi 1999-2020**: la flotta veicolare e le strutture di
   protezione del campione GIDAS non coincidono necessariamente con quelle
   degli scenari Waymo statunitensi. [11] §5.2 dichiara la stessa
   limitazione per il proprio studio.
5. **Solo urti contro il frontale dell'auto**: [11] §5.2 restringe il
   campione a impatti frontali dal punto di vista dell'altra auto. Gli urti
   laterali e posteriori sono modellati dalla stessa curva, il che è
   un'estensione oltre il dominio dichiarato della fonte.
6. **Il costo non è più simmetrico fra i due partner dell'urto**: esprime
   il rischio per l'attore urtato secondo la sua classe, non un rischio
   congiunto.

### 9.2 Motociclisti

[11] fornisce una curva per motociclisti (`MAIS3+F`: intercetta `-4.555`,
closing speed `0.040`, età `0.011`; Tabella 4), sostanzialmente più
vulnerabile della curva conducente auto. Il tassonomia attori del progetto
(`ActorClass`, v4.7 §5.7) non ha una classe `MOTORCYCLIST`: eventuali
motocicli presenti negli scenari ScenarioNet ricadono in `VEHICLE` e
vengono quindi trattati come occupanti protetti, sottostimandone il
rischio.

Questa specifica **non** introduce la classe. La decisione è deferred e
subordinata a una verifica empirica della presenza di motocicli nel pool di
scenari selezionato; se la loro frequenza è trascurabile la limitazione
resta annotata e non attuabile.

### 9.3 Impatto sperimentale

La distribuzione di `c_1(t)` cambia. Qualunque baseline già eseguita e
calibrata su `R1` v4.7 non è direttamente comparabile con risultati
prodotti dopo questa versione, e i due insiemi non possono essere messi in
pool come una sola condizione sperimentale.

L'impatto è **asimmetrico rispetto all'algoritmo**, e va riportato come
tale:

- per lo scalarizzatore di default `bounded_satisfaction_rank`
  (`SCAL-V1.0` §7.5) il termine categoriale dipende da
  `I_1 = 1[m_1 = 0]`, che è **invariato** da questa versione: qualsiasi
  collisione produceva e continua a produrre la stessa penalità `a^3`. La
  magnitudine entra solo nel tie-breaker continuo `T(m)`, limitato a
  `+-0.25`. L'effetto sul reward scalare è quindi piccolo;
- per un learner lessicografico, che ottimizza `J_1(pi)` direttamente,
  l'effetto è pieno: a parità di closing speed una collisione con pedone
  pesa circa 25 volte una collisione fra veicoli (`u = 20 m/s`: `0.869`
  contro `0.035`). Questo è il comportamento inteso e coerente con il
  ranking di vulnerabilità di [11], ma è una differenza sostanziale di
  obiettivo rispetto a v4.7 e va dichiarata nell'interpretazione dei
  risultati.

### 9.4 Rapporto con v4.7 §5.6

v4.7 §5.6 motivava l'assenza di masse così: *"Non vengono usate masse
perché non sono disponibili in modo omogeneo per tutti gli attori
ScenarioNet e una massa ridotta penalizzerebbe numericamente meno una
collisione con un VRU leggero. Il tipo di vittima resta nel logging; la
maggiore tutela preventiva dei VRU è affidata a TTC e clearance."*

Questa versione **non introduce masse** — la prima parte della motivazione
resta valida e rispettata. Modifica però la seconda parte: il tipo di
vittima non è più soltanto informazione di logging, ma determina la curva
di rischio applicata. La tutela dei VRU non è più affidata esclusivamente a
`R2`; `R1` la esprime ora direttamente, con la gerarchia di vulnerabilità
presa da dati di incidentalità reali anziché da una scelta di progetto.

Il rischio che v4.7 voleva evitare (una massa ridotta che penalizza *meno*
una collisione con un VRU) è evitato in modo più forte: la curva pedone
domina la curva veicolo su tutto il dominio, quindi una collisione con VRU
non può mai costare meno di una collisione fra veicoli a parità di closing
speed.

## 10. Acceptance Criteria

### AC-R1-01: Indipendenza dallo scenario

- Dato: due valutazioni con identici `u_i` e classe di attore, e
  configured speed cap ego diversi;
- Allora: `q_collision,i` è identico entro `1e-12`.
- Requisito: `REQ-R1-01`.

### AC-R1-02: Regressione del controesempio v4.7

- Dato: caso residenziale (`cap = 5`, `u = 4`) e caso autostradale
  (`cap = 20`, `u = 15`), stessa classe di attore;
- Allora: `q_collision(autostradale) > q_collision(residenziale)`, cioè
  l'ordinamento rispecchia la severità fisica reale.
- Nota: sotto v4.7 questo test fallisce (`0.5625 < 0.640`).
- Requisito: `REQ-R1-01`.

### AC-R1-03: Ordinamento di vulnerabilità fra classi

- Dato: la stessa `u > 0` per `PEDESTRIAN`, `CYCLIST`, `VEHICLE`;
- Allora:
  `q_PEDESTRIAN > q_CYCLIST > q_VEHICLE`.
- Requisito: `REQ-R1-03`.

### AC-R1-04: Dominio, monotonia e riferimento numerico

- Dato: `u` in `[0, 100] m/s` per ogni classe mappata;
- Allora: `q` in `(0,1)`, strettamente crescente in `u`, e i valori della
  tabella §5 sono riprodotti entro `1e-4`.
- Requisito: `REQ-R1-02`.

### AC-R1-05: Verifica della trascrizione dei coefficienti

- Dato: età mediane per classe di [11] (46/39/39) e rischio target `0.10`;
- Allora: le closing speed risolte sono `29 / 44 / 112 km/h` entro
  `1 km/h`, riproducendo gli ancoraggi pubblicati da [11] §4.4.
- Requisito: `REQ-R1-04`.

### AC-R1-06: `STATIC_COLLIDABLE` usa la curva conducente auto

- Dato: la stessa `u` per `VEHICLE` e `STATIC_COLLIDABLE`;
- Allora: i costi sono identici entro `1e-12`.
- Requisito: `REQ-R1-05`.

### AC-R1-07: Classe non mappata è errore fatale

- Dato: un onset di contatto con una classe di attore non presente in §4.4;
- Allora: `RulebookEvaluationError`, nessun fallback.
- Requisito: `REQ-R1-06`.

### AC-R1-08: Invarianti di v4.7 preservati

- Onset già attivo non genera nuova collisione; attore dinamico senza
  pre-state viene ignorato con diagnostics; centri coincidenti sollevano
  errore; aggregazione per massimo su onset multipli; `m_1 = -c_1`.
- Requisito: `REQ-R1-07`.

## 11. Traceability

| Requisito | Descrizione | Criteri | Fonte |
|---|---|---|---|
| `REQ-R1-01` | Il costo non dipende dalla configurazione dello scenario | `AC-R1-01`, `AC-R1-02` | §1.1, decisione utente 2026-07-26 |
| `REQ-R1-02` | Il costo è la probabilità MAIS3+F della curva di [11] | `AC-R1-04` | §4.1, [11] Eq. (1) |
| `REQ-R1-03` | La vulnerabilità per classe segue [11] | `AC-R1-03` | [11] §4.3 |
| `REQ-R1-04` | I coefficienti riproducono gli ancoraggi pubblicati | `AC-R1-05` | [11] Tab. 2/3/5, §4.4 |
| `REQ-R1-05` | `STATIC_COLLIDABLE` usa la curva conducente auto | `AC-R1-06` | §4.4, decisione utente 2026-07-26 |
| `REQ-R1-06` | Nessun fallback per classi non mappate | `AC-R1-07` | §8 |
| `REQ-R1-07` | Gli invarianti di v4.7 §5.1-5.3, §5.5 restano validi | `AC-R1-08` | v4.7 §5 |

## 12. Decisioni approvate

| ID | Decisione | Alternative considerate | Stato | Evidenza |
|---|---|---|---|---|
| `DEC-R1-01` | Sostituire il normalizzatore scenario-dipendente con una curva di rischio da letteratura | (a) cap fisso globale calcolato sul pool di scenari; (b) curva di rischio | `APPROVED` | Discussione 2026-07-26; (a) scartata perché resterebbe un artefatto della composizione del dataset |
| `DEC-R1-02` | Livello di severità `MAIS3+F` | `MAIS2+F` (costo `0.217` già a velocità nulla, dinamica utile bruciata); `Fatal` (quasi piatta sotto 10 m/s, nessuna risoluzione nel regime urbano) | `APPROVED` | Utente 2026-07-26; è inoltre il criterio usato da [11] e dal framework Safe System |
| `DEC-R1-03` | Età di riferimento fissa `A = 65` per tutte le classi | Età mediana per classe (46/39/39): confonde vulnerabilità ed età del campione | `APPROVED` | Utente 2026-07-26; 65 è l'età che [11] §4.3 usa proprio per il confronto fra utenti |
| `DEC-R1-04` | `STATIC_COLLIDABLE` mappato sulla curva conducente auto | Fonte separata per urti contro oggetto fisso (reintroduce incomparabilità metodologica) | `APPROVED` | Utente 2026-07-26 |
| `DEC-R1-05` | Alimentare la curva con la componente normale, non il modulo | Modulo del closing speed, fedele a [11] ma perde la discriminazione sfioramento/impatto frontale di v4.7 §5.3 | `APPROVED` | Preservazione di un invariante v4.7 già approvato; limitazione dichiarata in §9.1 |
| `DEC-R1-06` | Mantenere i configured speed cap come precondizione di eleggibilità | Rimuoverli del tutto da `R1` | `APPROVED` | Contenimento del rischio: l'eleggibilità è fuori scope, §4.6 |
| `DEC-R1-07` | Non introdurre la classe `MOTORCYCLIST` | Introdurla usando [11] Tab. 4 | `DEFERRED` | §9.2; subordinata a verifica empirica sul pool di scenari |

## 13. Riferimenti

I riferimenti `[1]`-`[10]` sono quelli di v4.7 §14. Questa versione
aggiunge:

- `[11]` N. Lubbe, Y. Wu, H. Jeppsson, "Safe speeds: fatality and injury
  risks of pedestrians, cyclists, motorcyclists, and car drivers impacting
  the front of another passenger car as a function of closing speed and
  age", *Traffic Safety Research*, vol. 2, 000006, 2022.
  DOI: `10.55329/vfma7555`.
  Copia locale: `docs/papers/rulebook/Lubbe et Al. - 2022 - Safe speeds.pdf`.

Fonti consultate e non adottate, registrate per tracciabilità della
selezione:

- B. C. Tefft, "Impact speed and a pedestrian's risk of severe injury or
  death", *Accident Analysis & Prevention*, vol. 50, 2013 — copre solo i
  pedoni e usa la velocità d'impatto del veicolo, non la closing speed.
- E. Rosén, U. Sander, "Pedestrian fatality risk as a function of car
  impact speed", *Accident Analysis & Prevention*, vol. 41(3), 2009 — solo
  pedoni; usato da [11] §5.1 come termine di confronto.
- H. C. Joksch, modello `P ~ (delta-v / 31.74 m/s)^4` — richiede il
  `delta-v`, quindi le masse dei due veicoli, non disponibili in
  `ActorSnapshot`. Corrobora indirettamente [11]: la costante `31.74 m/s`
  è prossima ai `112 km/h = 31.1 m/s` di [11] per il conducente auto,
  ricavati da un dataset indipendente.

## 14. Approval Record

- Approved by: `user`
- Approval date: `2026-07-26`
- Approval evidence: `Messaggio utente "confermo MAIS3+F, età 65, static→car driver"`, a valle della presentazione delle tre decisioni con opzioni, conseguenze numeriche e raccomandazione motivata.
- Approval scope: le tre decisioni `DEC-R1-02`, `DEC-R1-03`, `DEC-R1-04` sono
  approvate esplicitamente. `DEC-R1-01` è approvata dalla richiesta iniziale
  di sostituire il normalizzatore. `DEC-R1-05` e `DEC-R1-06` sono scelte di
  preservazione di invarianti v4.7 esistenti, registrate qui per
  tracciabilità.
