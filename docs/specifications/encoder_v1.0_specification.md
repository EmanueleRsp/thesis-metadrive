---
title: "Specificazione finale degli encoder e dell'integrazione con Stable-Baselines3"
subtitle: "MLP flat, Latent Query strutturato, condivisione actor–critic, gradient routing, checkpoint e test"
author: "Report tecnico per la tesi"
date: "15 luglio 2026"
lang: it-IT
version: "1.0-final-implementation-complete"
specification_id: "ENC-V1.0"
status: "APPROVED"
authoritative: true
approval_date: "2026-07-16"
approval: "Explicit user approval in this Codex conversation"
related_adrs:
  - "docs/decisions/ADR-002-semantic-observation-and-encoder-contract.md"
---

# Sintesi esecutiva

This candidate specification defines the proposed encoder contract for the RL
pipeline, aligned to:

- `docs/specifications/observation_v1.1_specification.md`;
- `StackedLidarStateObservation`, dimensione flat \(1540\);
- `SemanticStateObservationV2`, dimensione flat \(2541\);
- tokenizzazione semantica di \(122\) raw token;
- backend normativi `td3_sb3`, `sac_sb3` e `ppo_sb3`;
- fork locale Stable-Baselines3 `2.9.0`, commit registrato nel manifest sperimentale.

Le configurazioni core sono:

1. `StackedLidarStateObservation` + `FlatMLPEncoder`;
2. `SemanticStateObservationV2` + `FlatMLPEncoder`;
3. `SemanticStateObservationV2` + `LatentQueryEncoderV2`.

L’`IdentityEncoder` resta disponibile soltanto come modalità diagnostica e di compatibilità, non come confronto sperimentale principale.

Decisioni principali:

- il backend SB3 fork-backed è normativo;
- i backend legacy interni non condizionano il nuovo design;
- tutti gli encoder producono un vettore latente di dimensione \(256\), salvo l’identity diagnostico;
- l’MLP usa la rappresentazione flat e riceve anche le mask come normali feature;
- l’LQ usa i gruppi strutturati e tratta le mask esclusivamente come attention mask;
- l’LQ usa \(16\) latent queries di dimensione \(128\), quattro blocchi, quattro head e FFN da \(256\);
- il residual gating è disattivato nella configurazione core;
- TD3 e SAC usano encoder actor e critic separati; Q1 e Q2 condividono il critic encoder;
- PPO usa un encoder condiviso tra policy e value;
- nessun autoencoder, behavior-cloning warm start, congelamento o learning rate separato appartiene alla configurazione core;
- l’addestramento è end-to-end;
- i checkpoint precedenti basati su \(2363\) feature e \(107\) token sono incompatibili e non vengono migrati implicitamente;
- schema dell’osservazione e architettura dell’encoder sono versionati e verificati con fail-fast.

# 1. Ambito e dipendenze normative

## 1.1 Ambito

La specifica copre esclusivamente:

- conversione della flat observation SB3 in feature neurali;
- MLP encoder;
- Latent Query encoder;
- connessione con actor, critic e value head;
- condivisione dei moduli e routing dei gradienti;
- inizializzazione, ottimizzazione e serializzazione;
- configurazioni e test.

Non copre:

- costruzione delle osservazioni;
- rulebook e reward;
- N-step returns;
- PER;
- ACL;
- algoritmi lessicografici;
- critic distributional.

Tali componenti possono usare gli encoder definiti qui senza modificarne il contratto.

## 1.2 Fonti normative

Ordine di precedenza:

1. `docs/specifications/observation_v1.1_specification.md`;
2. questa specifica;
3. configurazioni consolidate;
4. implementazione;
5. documentazione precedente.

In caso di conflitto, una specifica precedente basata su:

```text
semantic flat_dim = 2363 oppure 2450
semantic token_count = 107
```

è obsoleta.

## 1.3 Backend normativo

Sono normativi:

```text
td3_sb3
sac_sb3
ppo_sb3
```

Sono legacy e non devono introdurre vincoli nel nuovo design:

```text
td3
sac
ppo
```

Gli encoder rimangono implementati nel package `thesis_rl` e vengono collegati a SB3 tramite un `BaseFeaturesExtractor`.

# 2. Decomposizione dell’agente

L’agente mantiene la decomposizione:

```text
Preprocessor
    ↓
Planner
    ↓
Action Adapter
```

Il preprocessor è deterministico e non appreso. Valida formato, dtype e shape, ma non apprende rappresentazioni.

Il planner contiene:

```text
Observation encoder
Planner heads
RL algorithm
Buffer
Losses
Optimizers
Target networks, quando applicabili
```

L’action adapter applica soltanto trasformazioni deterministiche necessarie all’interfaccia dell’ambiente.

Il termine preferito è **planner head**. In questa specifica, `decoder` non indica un reconstruction decoder o un decoder generativo.

# 3. Configurazioni supportate

## 3.1 Configurazioni core

| ID | Osservazione | Encoder | Output |
|---|---|---|---:|
| `lidar_mlp` | `StackedLidarStateObservation` | `FlatMLPEncoder` | 256 |
| `semantic_mlp` | `SemanticStateObservationV2` | `FlatMLPEncoder` | 256 |
| `semantic_lq` | `SemanticStateObservationV2` | `LatentQueryEncoderV2` | 256 |

## 3.2 Configurazioni non core

| Configurazione | Stato |
|---|---|
| Identity / flat no-compression | diagnostica |
| LiDAR + LQ | non implementata nel core |
| singoli raggi LiDAR come token | esclusa |
| encoder ricorrente | estensione |
| autoencoder pretraining | escluso |
| behavior-cloning warm start | estensione futura |
| encoder unico actor–critic critic-driven | estensione futura custom |
| residual gating | legacy/opzionale, OFF nel core |

Il builder deve rifiutare:

```text
StackedLidarStateObservation + LatentQueryEncoderV2
```

finché non esiste una specifica separata per la tokenizzazione LiDAR.

# 4. Contratto comune degli encoder

## 4.1 Interfaccia

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    input_dim: Final[int]
    output_dim: Final[int]
    observation_schema_version: Final[str]
    encoder_architecture_version: Final[str]

    @abstractmethod
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flat_obs:
                float32 tensor, shape [B, input_dim].

        Returns:
            float32 tensor, shape [B, output_dim].
        """
        ...
```

Contratto normativo:

```text
input rank:        2
input shape:       [B, D]
input dtype:       torch.float32
output shape:      [B, output_dim]
device:            preservato
batch dimension:   mai rimossa
```

Input `[D]` non è ammesso. Il chiamante deve usare `unsqueeze(0)`.

L’encoder non deve:

- leggere direttamente l’ambiente;
- consultare rulebook, reward o curriculum;
- modificare l’osservazione;
- applicare clipping aggiuntivo;
- normalizzare con statistiche online;
- cambiare ordine o significato delle feature.

## 4.2 Validazione

Sempre:

```python
if flat_obs.ndim != 2:
    raise ValueError(...)
if flat_obs.shape[-1] != self.input_dim:
    raise ValueError(...)
```

In modalità test/debug:

```python
if flat_obs.dtype != torch.float32:
    raise TypeError(...)
if not torch.isfinite(flat_obs).all():
    raise ValueError(...)
```

Durante training la conversione a `float32` avviene nel bridge SB3, prima dell’encoder.

## 4.3 Bridge SB3

```python
class ThesisEncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        *,
        encoder_config: dict,
        observation_schema: SemanticObservationSchemaV11 | None,
    ) -> None:
        encoder = build_encoder(
            encoder_config=encoder_config,
            observation_space=observation_space,
            observation_schema=observation_schema,
        )
        super().__init__(
            observation_space=observation_space,
            features_dim=encoder.output_dim,
        )
        self.encoder = encoder

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)

        if observations.ndim != 2:
            raise ValueError(
                "Thesis encoders require a flat SB3 observation [B, D]."
            )

        return self.encoder(observations)
```

È vietato appiattire silenziosamente tensori con rank superiore a 2 nel feature extractor.

SB3 receives only plain encoder configuration and the schema identifier through
`features_extractor_kwargs`; it must not receive a pre-built `BaseEncoder`
instance. Each SB3 construction of `ThesisEncoderFeatureExtractor` builds its
own encoder. This yields independent actor, critic, and target encoders where
the algorithm requires them, while PPO shares its one extractor through
`share_features_extractor=true`.

# 5. Schema dell’osservazione e flat bridge

## 5.1 Unica fonte degli offset

Deve esistere un unico oggetto normativo:

```python
SemanticObservationSchemaV11
```

che definisce:

- schema version;
- ordine flat;
- shape;
- slice;
- dtype;
- dimensione totale;
- token order;
- mask order;
- fingerprint.

L’ordine non deve essere duplicato in:

- observation builder;
- encoder;
- unflatten utility;
- test;
- configurazioni.

## 5.2 Ordine flat semantic

Ordine normativo:

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

Dimensione:

\[
D_{\mathrm{semantic}}=2541.
\]

## 5.3 Structured torch representation

```python
@dataclass(frozen=True)
class SemanticObservationTensorBatch:
    ego_history: torch.Tensor
    ego_history_mask: torch.Tensor
    ego_current: torch.Tensor

    route: torch.Tensor
    route_mask: torch.Tensor

    dynamic: torch.Tensor
    dynamic_mask: torch.Tensor

    static: torch.Tensor
    static_mask: torch.Tensor

    lane_road: torch.Tensor

    controls: torch.Tensor
    controls_mask: torch.Tensor

    interactions: torch.Tensor
    interactions_mask: torch.Tensor

    temporal: torch.Tensor
```

Shape batch-first:

```text
ego_history:       [B, 5, 10]
ego_history_mask:  [B, 5]
ego_current:       [B, 3]

route:             [B, 10, 7]
route_mask:        [B, 10]

dynamic:           [B, 16, 5, 22]
dynamic_mask:      [B, 16, 5]

static:            [B, 8, 13]
static_mask:       [B, 8]

lane_road:         [B, 14]

controls:          [B, 8, 17]
controls_mask:     [B, 8]

interactions:      [B, 8, 35]
interactions_mask: [B, 8]

temporal:          [B, 5]
```

## 5.4 API dello schema

```python
class SemanticObservationSchemaV11:
    version = "1.1-final"
    flat_dim = 2541
    raw_token_count = 122

    def flatten_numpy(
        self,
        batch: SemanticObservationBatch,
    ) -> np.ndarray:
        ...

    def unflatten_torch(
        self,
        flat_obs: torch.Tensor,
    ) -> SemanticObservationTensorBatch:
        ...

    def canonical_dict(self) -> dict:
        ...

    def fingerprint_sha256(self) -> str:
        ...
```

`unflatten_torch()` deve usare `reshape`, non copie CPU/NumPy, e preservare autograd, device e dtype.

## 5.5 Fingerprint

Il fingerprint è:

```text
SHA-256(
    canonical JSON di:
        schema_version
        flat order
        group names
        group shapes
        mask shapes
        flat slices
        token order
)
```

Il JSON canonico usa:

```text
UTF-8
chiavi ordinate
separatori senza whitespace variabile
```

# 6. Identity encoder diagnostico

## 6.1 Definizione

```python
class IdentityEncoder(BaseEncoder):
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        validate(flat_obs)
        return flat_obs
```

Output:

```text
LiDAR:    1540
Semantic: 2541
```

## 6.2 Uso

È ammesso per:

- test del bridge;
- debug dell’observation space;
- riproduzione di vecchie varianti;
- confronto locale non principale.

Non appartiene alla matrice sperimentale core perché delega l’intera trasformazione agli head SB3 e rende meno controllabile il confronto sulla capacità della rete.

Nel naming user-facing usare:

```text
flat_no_compression
```

anziché suggerire che la policy sia priva di rete neurale.

# 7. Flat MLP encoder

## 7.1 Architettura

Per entrambe le osservazioni:

```text
D
→ Linear(D, 512)
→ LayerNorm(512)
→ ReLU

→ Linear(512, 512)
→ LayerNorm(512)
→ ReLU

→ Linear(512, 256)
→ LayerNorm(256)
→ ReLU

→ Linear(256, 256)
→ LayerNorm(256)
→ ReLU
```

Output:

\[
z_t\in\mathbb R^{256}.
\]

Configurazione:

```yaml
encoder:
  type: mlp
  architecture_version: 1.0-final
  hidden_layers: [512, 512, 256]
  output_dim: 256
  activation: relu
  layer_norm: true
  dropout: 0.0
```

## 7.2 Layer normalization

Semantica normativa:

```text
layer_norm=true:
    LayerNorm dopo ogni Linear, incluso il projection block finale

layer_norm=false:
    nessuna LayerNorm
```

È vietato applicare una LayerNorm finale quando `layer_norm=false`.

Configurazione core:

```text
layer_norm = true
```

## 7.3 Dropout e residual

```text
dropout = 0
residual connections = assenti
```

## 7.4 Mask

L’MLP riceve il vettore flat completo. Le mask:

- restano feature numeriche \(\{0,1\}\);
- permettono alla rete di distinguere padding e valori fisici zero;
- non vengono rimosse;
- non vengono usate come hard gating interno all’MLP.

Il builder dell’osservazione deve comunque garantire:

```text
masked payload = 0
```

## 7.5 Parametri attesi

Inclusi bias e parametri LayerNorm, esclusi planner heads:

| Input | Parametri |
|---:|---:|
| 1540 | 1,251,840 |
| 2541 | 1,764,352 |

Il test può verificare questi valori per individuare modifiche architetturali accidentali.

# 8. Latent Query encoder V2

## 8.1 Applicabilità

`LatentQueryEncoderV2` accetta esclusivamente:

```text
SemanticStateObservationV2
schema_version = 1.1-final
flat_dim = 2541
raw_token_count = 122
```

## 8.2 Tokenizzazione

| Gruppo | Raw shape per elemento | Token |
|---|---:|---:|
| Ego history | \(5\times10\) | 5 |
| Ego current | \(1\times3\) | 1 |
| Route | \(10\times7\) | 10 |
| Dynamic actors | \(16\times5\times22\) | 80 |
| Static objects | \(8\times13\) | 8 |
| Lane/road | \(1\times14\) | 1 |
| Traffic controls | \(8\times17\) | 8 |
| Conflict interactions | \(8\times35\) | 8 |
| Temporal compliance | \(1\times5\) | 1 |
| **Totale** |  | **122** |

Ordine token normativo:

```text
0..4      ego_history
5         ego_current
6..15     route
16..95    dynamic
96..103   static
104       lane_road
105..112  controls
113..120  interactions
121       temporal
```

Per dynamic:

```text
token_index =
    16
    + actor_slot * history_length
    + history_index
```

con:

```text
history_index 0 = frame più antico disponibile
history_index 4 = frame corrente
```

## 8.3 Proiezioni per gruppo

Dimensione comune:

\[
d_{\mathrm{token}}=64.
\]

Moduli:

```text
ego_history: Linear(10, 64) → ReLU → LayerNorm(64)
ego_current: Linear(3, 64)  → ReLU → LayerNorm(64)
route:       Linear(7, 64)  → ReLU → LayerNorm(64)
dynamic:     Linear(22, 64) → ReLU → LayerNorm(64)
static:      Linear(13, 64) → ReLU → LayerNorm(64)
lane_road:   Linear(14, 64) → ReLU → LayerNorm(64)
controls:    Linear(17, 64) → ReLU → LayerNorm(64)
interactions:Linear(35, 64) → ReLU → LayerNorm(64)
temporal:    Linear(5, 64)  → ReLU → LayerNorm(64)
```

L’ordine `Linear → ReLU → LayerNorm` è normativo per LQ.

## 8.4 Type embedding

Tabella:

```python
nn.Embedding(9, 64)
```

Type IDs:

```text
0 ego_history
1 ego_current
2 route
3 dynamic
4 static
5 lane_road
6 controls
7 interactions
8 temporal
```

Inizializzazione:

\[
E_{\mathrm{type}}\sim\mathcal N(0,0.02^2).
\]

## 8.5 Temporal embedding

Tabella:

```python
nn.Embedding(5, 64)
```

Applicata a:

```text
ego_history
dynamic
```

Indice:

```text
0 oldest
1
2
3
4 current
```

La stessa tabella temporale può essere condivisa tra ego history e dynamic history perché gli indici hanno identica semantica temporale.

Inizializzazione:

\[
E_{\mathrm{time}}\sim\mathcal N(0,0.02^2).
\]

## 8.6 Slot embedding

Tabelle separate:

```text
route_slot_embedding:        Embedding(10, 64)
dynamic_slot_embedding:      Embedding(16, 64)
static_slot_embedding:       Embedding(8, 64)
control_slot_embedding:      Embedding(8, 64)
interaction_slot_embedding:  Embedding(8, 64)
```

Gli slot embedding:

- identificano posizioni persistenti nel tensore;
- non codificano actor ID o object ID;
- non sono derivati dai metadati dello scenario;
- vengono ripetuti sulle cinque posizioni temporali dello stesso dynamic slot.

Inizializzazione:

\[
E_{\mathrm{slot}}\sim\mathcal N(0,0.02^2).
\]

## 8.7 Composizione del token

Per un token valido:

\[
x
=
\phi_g(f)
+
e_{\mathrm{type}}
+
e_{\mathrm{time}}\;\text{se applicabile}
+
e_{\mathrm{slot}}\;\text{se applicabile}.
\]

Dopo la composizione:

```python
tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
```

Il masked zeroing è obbligatorio anche se viene applicata la `key_padding_mask`.

## 8.8 Mask globale

Ordine:

```text
ego_history_mask        5
ego_current             1, sempre valido
route_mask             10
dynamic_mask           80
static_mask              8
lane_road                1, sempre valido
controls_mask            8
interactions_mask        8
temporal                 1, sempre valido
```

Totale:

\[
5+1+10+80+8+1+8+8+1=122.
\]

Semantica esterna:

```text
1 = valido
0 = padding/non disponibile
```

Semantica PyTorch MHA:

```python
key_padding_mask = ~valid_mask.bool()
```

La mask è applicata alle key/value della cross-attention. Non viene applicata alla self-attention fra latent queries.

Almeno i token:

```text
ego_current
lane_road
temporal
```

sono sempre validi. Un batch con tutte le 122 posizioni mascherate è invalido e deve generare errore nei test/debug.

## 8.9 Passaggio alla latent dimension

Dopo la concatenazione:

```text
scene tokens: [B, 122, 64]
```

Proiezione:

```python
token_to_latent = nn.Linear(64, 128)
```

Output:

```text
scene memory: [B, 122, 128]
```

## 8.10 Learned latent queries

```python
latent_queries = nn.Parameter(
    torch.empty(16, 128)
)
```

Inizializzazione:

\[
Z_0\sim\mathcal N(0,0.02^2).
\]

Per il batch:

```python
z = latent_queries.unsqueeze(0).expand(B, -1, -1)
```

Non usare `repeat`, per evitare copie non necessarie.

## 8.11 Blocco LQ

Numero blocchi:

\[
L=4.
\]

Ogni blocco usa pre-normalization e residual standard:

```text
1. Cross-attention:
   z ← z + CrossAttention(
       query = LN(z),
       key   = LN(scene_memory),
       value = LN(scene_memory),
       key_padding_mask
   )

2. Cross FFN:
   z ← z + FFN_cross(LN(z))

3. Latent self-attention:
   z ← z + SelfAttention(
       query = LN(z),
       key   = LN(z),
       value = LN(z)
   )

4. Latent FFN:
   z ← z + FFN_self(LN(z))
```

Attention:

```text
embed_dim = 128
num_heads = 4
head_dim = 32
batch_first = true
attention_dropout = 0
```

FFN:

```text
Linear(128, 256)
ReLU
Linear(256, 128)
dropout = 0
```

Cross-attention e self-attention hanno parametri indipendenti per ciascun blocco.

## 8.12 Residual gating

Configurazione core:

```text
residual_gating = false
```

È vietata nella configurazione core la formula:

```python
x + tanh(alpha) * branch(x)
```

con gate inizializzato a zero, perché disattiva inizialmente i rami e modifica il flusso dei gradienti.

Un’eventuale modalità legacy può rimanere nel codice soltanto dietro configurazione esplicita e non deve essere usata negli esperimenti principali.

## 8.13 Pooling e output

Dopo l’ultimo blocco:

\[
\bar z
=
\frac{1}{16}
\sum_{i=1}^{16} z_i.
\]

Projection head:

```text
Linear(128, 256)
LayerNorm(256)
ReLU
```

Output:

```text
[B, 256]
```

Il pooling supportato nel core è esclusivamente:

```text
mean
```

## 8.14 Configurazione consolidata LQ

```yaml
encoder:
  type: latent_query_v2
  architecture_version: 1.0-final
  required_observation_schema: 1.1-final

  token_dim: 64
  num_latents: 16
  latent_dim: 128
  output_dim: 256

  depth: 4
  num_heads: 4
  ff_dim: 256

  activation: relu
  projection_order: linear_relu_layernorm
  normalization: pre_norm

  type_embedding: true
  time_embedding: true
  slot_embedding: true

  pooling: mean

  dropout: 0.0
  attention_dropout: 0.0
  residual_gating: false
```

# 9. Planner heads

Gli head vengono mantenuti uguali fra MLP e LQ per isolare il contributo dell’encoder.

## 9.1 TD3 e SAC actor

```text
encoder output 256
→ Linear(256, 256)
→ ReLU
→ Linear(256, 256)
→ ReLU
→ algorithm-specific action output
```

TD3:

```text
deterministic action head
```

SAC:

```text
mean and log_std heads / distribution parameters
```

Il post-processing dell’azione rimane quello SB3.

## 9.2 TD3 e SAC critic

Per ciascun Q head:

```text
concat(encoder(obs), action)
→ Linear(256 + action_dim, 256)
→ ReLU
→ Linear(256, 256)
→ ReLU
→ Linear(256, 1)
```

Q1 e Q2:

- hanno head indipendenti;
- condividono il critic observation encoder;
- non condividono i pesi degli MLP Q.

## 9.3 PPO

```text
encoder output 256
├── policy head: [256, 256] → action distribution
└── value head:  [256, 256] → V(s)
```

Attivazione:

```text
ReLU
```

La baseline PPO stock `[64,64]` con `Tanh` non appartiene alle configurazioni encoder-controlled core.

# 10. Condivisione dei moduli e gradient routing

## 10.1 TD3 e SAC

TD3 configuration:

```text
actor encoder:
    independent module

critic encoder:
    independent from actor
    shared by Q1 and Q2

target actor encoder:
    distinct copy of actor encoder

target critic encoder:
    distinct copy of critic encoder
```

SAC configuration:

```text
actor encoder:
    independent module

critic encoder:
    independent from actor
    shared by Q1 and Q2

target critic encoder:
    distinct copy of critic encoder
```

SAC has no target actor in the normative SB3 implementation. Therefore TD3 has
four encoder instances, SAC has three, and neither algorithm duplicates the
critic encoder for Q1 and Q2.

SB3:

```python
share_features_extractor = False
```

Questo valore deve essere impostato esplicitamente nelle configurazioni, non lasciato al default implicito.

## 10.2 Ownership degli optimizer TD3/SAC

Actor optimizer contiene:

```text
actor encoder
actor head
```

Critic optimizer contiene:

```text
critic encoder
Q1 head
Q2 head
```

La critic loss deve aggiornare:

```text
critic encoder
Q1
Q2
```

e non deve aggiornare:

```text
actor encoder
actor head
```

La actor loss deve aggiornare:

```text
actor encoder
actor head
```

e non deve aggiornare:

```text
critic encoder
Q1
Q2
```

I parametri critic possono essere usati per calcolare il gradiente rispetto all’azione/actor, ma non devono appartenere all’actor optimizer.

Dopo un actor update, i parametri del critic encoder devono essere bitwise invariati nel test deterministico.

## 10.3 Target networks

Le target networks:

- non appartengono ad alcun optimizer;
- non ricevono gradienti;
- vengono aggiornate esclusivamente tramite Polyak averaging SB3;
- include the respective encoders: actor and critic targets for TD3, critic
  target only for SAC.

## 10.4 Perché non usare `share_features_extractor=true`

La modalità SB3 condivisa non implementa:

```text
unico encoder
appreso dal critic
actor con stop-gradient
```

Nel comportamento SB3, la condivisione modifica il routing dei gradienti e può escludere il features extractor dall’aggiornamento critic. Non deve essere usata come scorciatoia per una variante critic-driven.

## 10.5 PPO

Configurazione normativa:

```text
un encoder condiviso tra policy e value
un optimizer
loss PPO congiunta
```

SB3:

```python
share_features_extractor = True
```

L’encoder riceve gradienti da:

```text
policy loss
value loss
entropy term, se dipendente dalle feature
```

pesati secondo la loss PPO configurata.

Una variante PPO con encoder policy/value separati non appartiene al core.

# 11. Ottimizzazione

## 11.1 Training mode

Configurazione core:

```text
training_mode = end_to_end
pretraining = none
freeze_encoder = false
freeze_steps = 0
encoder_learning_rate = inherited
```

## 11.2 Learning rate

L’encoder usa il learning rate del network che lo possiede:

```text
TD3/SAC actor encoder → actor learning rate
TD3/SAC critic encoder → critic learning rate
PPO encoder → PPO optimizer learning rate
```

Non viene creato un parameter group separato.

## 11.3 Optimizer

Si usa l’optimizer del backend SB3:

```text
Adam
```

Le differenze algoritmiche SB3, come `eps=1e-5` nel percorso PPO, restano parte del backend e vengono registrate nella configurazione.

## 11.4 Gradient clipping

```text
PPO:
    max_grad_norm = configurazione SB3, nominalmente 0.5

TD3/SAC:
    nessun clipping encoder-specifico nel core
```

# 12. Inizializzazione

## 12.1 Principio

La stessa classe encoder deve avere la stessa inizializzazione indipendentemente dall’algoritmo RL.

Per evitare che PPO reinizializzi ricorsivamente il custom feature extractor in modo diverso da TD3/SAC:

```python
ortho_init = False
```

deve essere impostato esplicitamente nella policy PPO encoder-controlled.

## 12.2 Parametri

```text
Linear:
    inizializzazione PyTorch del modulo

MultiheadAttention:
    inizializzazione PyTorch del modulo

LayerNorm:
    weight = 1
    bias = 0

type/time/slot embeddings:
    Normal(0, 0.02)

learned latent queries:
    Normal(0, 0.02)
```

Bias delle embedding: non applicabile.

## 12.3 Seed

La costruzione dell’encoder avviene dopo avere inizializzato i seed di:

```text
Python
NumPy
PyTorch
CUDA, se disponibile
SB3
environment
```

Il manifest registra il seed della run.

# 13. Normalizzazione degli input

## 13.1 Semantic

La normalizzazione è quella feature-specific definita dall’observation specification.

L’encoder non usa:

```text
VecNormalize
running mean/variance
BatchNorm
input standardization appresa
```

## 13.2 LiDAR

La `StackedLidarStateObservation` arriva già nel proprio observation space, con ray fractions in \([0,1]\) e altre feature secondo il contratto dell’osservazione.

Non viene applicato un secondo clipping nel feature extractor.

## 13.3 LayerNorm non equivale a input normalization

Le LayerNorm interne normalizzano le attivazioni neurali e non sostituiscono il contratto di normalizzazione dell’osservazione.

# 14. Pretraining, freeze e fine-tuning

Non appartengono alla versione `1.0-final`:

- autoencoder reconstruction pretraining;
- VAE;
- masked-token pretraining;
- behavior cloning;
- encoder pretrained da un agente MetaDrive;
- freeze iniziale;
- learning rate encoder separato;
- caricamento del solo encoder.

Le relative chiavi di configurazione, se presenti per compatibilità, devono avere:

```text
enabled = false
```

e il builder deve fallire se vengono abilitate senza una specifica futura.

Motivazione operativa:

- la semantic observation è già compatta e strutturata;
- un reconstruction objective non coincide necessariamente con le feature utili al critic;
- congelare un encoder preaddestrato su traiettorie nominali ridurrebbe l’adattamento agli stati visitati durante esplorazione, ACL e violazioni;
- l’obiettivo core è prima validare il training RL end-to-end.

# 15. Checkpoint e compatibilità

## 15.1 Incompatibilità esplicita

I checkpoint basati su:

```text
semantic flat_dim = 2363
semantic token_count = 107
```

sono incompatibili con:

```text
semantic flat_dim = 2541
semantic token_count = 122
```

Non esiste migration automatica o partial load implicito.

## 15.2 Manifest del checkpoint

Every checkpoint generation is an immutable directory:

```text
checkpoints/
  <checkpoint_name>-<generation_id>/
    model.zip
    manifest.json
  latest.json
```

`latest.json` is an atomic pointer to one complete generation and includes the
generation identifier and SHA-256 digests of both files. It is optional for an
explicit generation-path load, but required for a `latest` resume.

Campi minimi:

```json
{
  "observation_schema_version": "1.1-final",
  "observation_schema_fingerprint": "<sha256>",
  "observation_type": "semantic_v2",
  "flat_dim": 2541,
  "raw_token_count": 122,

  "encoder_architecture_version": "1.0-final",
  "encoder_type": "latent_query_v2",
  "encoder_config": {},

  "features_dim": 256,
  "share_features_extractor": false,
  "ppo_ortho_init": null,

  "algorithm": "td3_sb3",
  "sb3_version": "2.9.0",
  "sb3_commit": "<commit>",
  "git_commit": "<project commit>",
  "seed": 0
}
```

Per LiDAR:

```text
raw_token_count = null
flat_dim = 1540
observation_type = stacked_lidar_state
```

## 15.3 Atomic publication

Publication order:

```text
1. write model.zip and manifest.json in a temporary generation directory
2. validate both files and their SHA-256 digests
3. fsync when supported
4. atomically rename the complete generation directory
5. write a temporary latest.json for that generation and atomically rename it
```

A loader accepts only a complete generation whose model and manifest digests
match `latest.json` or the explicitly requested generation. A temporary
directory, an orphaned generation, or a checkpoint without a complete manifest
is invalid for resume.

## 15.4 Caricamento

Prima di chiamare `BaseAlgorithm.load()`:

```text
1. resolve the explicit generation or validate latest.json
2. verify model.zip and manifest.json SHA-256 digests
3. leggere manifest
4. costruire schema/config corrente
5. confrontare versioni e fingerprint
6. confrontare encoder type e shape
7. fail-fast se incompatibile
8. caricare il checkpoint SB3
```

Errore richiesto:

```text
CheckpointCompatibilityError
```

Il messaggio deve indicare:

```text
campo incompatibile
valore checkpoint
valore corrente
```

## 15.5 Legacy mode

Una modalità esplicita:

```text
allow_legacy_checkpoint = true
```

può servire soltanto per eseguire vecchi esperimenti con la vecchia observation e il vecchio codice. Non può essere usata per resume sul nuovo schema.

# 16. Configurazioni consolidate

## 16.1 MLP

```yaml
agent:
  backend: sb3

  encoder:
    type: mlp
    architecture_version: 1.0-final
    output_dim: 256
    hidden_layers: [512, 512, 256]
    activation: relu
    layer_norm: true
    dropout: 0.0

    training:
      mode: end_to_end
      pretrained: false
      freeze_steps: 0
      separate_learning_rate: false
```

## 16.2 LQ

```yaml
agent:
  backend: sb3

  encoder:
    type: latent_query_v2
    architecture_version: 1.0-final
    required_observation_schema: 1.1-final

    token_dim: 64
    num_latents: 16
    latent_dim: 128
    output_dim: 256

    depth: 4
    num_heads: 4
    ff_dim: 256

    activation: relu
    projection_order: linear_relu_layernorm
    normalization: pre_norm

    type_embedding: true
    time_embedding: true
    slot_embedding: true

    pooling: mean

    dropout: 0.0
    attention_dropout: 0.0
    residual_gating: false

    training:
      mode: end_to_end
      pretrained: false
      freeze_steps: 0
      separate_learning_rate: false
```

## 16.3 TD3/SAC

```yaml
policy:
  share_features_extractor: false

planner_heads:
  actor: [256, 256]
  critic: [256, 256]
  activation: relu
```

## 16.4 PPO

```yaml
policy:
  share_features_extractor: true
  ortho_init: false

planner_heads:
  policy: [256, 256]
  value: [256, 256]
  activation: relu
```

# 17. File e responsabilità implementative

Struttura consigliata:

```text
src/thesis_rl/contracts/
    observation_schema.py
    encoder_contract.py
    checkpoint_manifest.py

src/thesis_rl/agent/planners/encoders/
    base.py
    factory.py
    identity_encoder.py
    mlp_encoder.py
    lq_encoder_v2.py

src/thesis_rl/agent/planners/encoders/lq/
    tokenization.py
    embeddings.py
    masks.py
    blocks.py

src/thesis_rl/sb3_extensions/
    features_extractors.py
    builders.py
    checkpointing.py
```

Responsabilità:

```text
observation_schema.py:
    ordine, shape, slice, flatten/unflatten, fingerprint

features_extractors.py:
    bridge SB3 e validazione [B,D]

factory.py:
    compatibilità observation–encoder e costruzione

lq/tokenization.py:
    conversione structured batch → 122 token

lq/masks.py:
    costruzione valid mask e key_padding_mask

builders.py:
    policy_kwargs, sharing flags, ortho_init PPO

checkpointing.py:
    manifest, save/load fail-fast
```

È vietato ricreare offset o token order in file diversi dallo schema/tokenizer normativo.

# 18. Test obbligatori

## 18.1 Schema

1. `StackedLidarStateObservation` ha shape `(1540,)`.
2. `SemanticStateObservationV2` ha shape `(2541,)`.
3. `SemanticObservationSchemaV11.flat_dim == 2541`.
4. `raw_token_count == 122`.
5. Tutti gli offset coincidono con l’ordine normativo.
6. `structured → flat → structured` è esatto entro tolleranza float.
7. `unflatten_torch` preserva device, dtype e autograd.
8. Il fingerprint è deterministico.
9. Una modifica a shape/order cambia il fingerprint.

## 18.2 MLP

1. Forward LiDAR: `[B,1540] → [B,256]`.
2. Forward semantic: `[B,2541] → [B,256]`.
3. Backward finito.
4. Nessun NaN/Inf.
5. Parametri attesi:
   - `1,251,840` per LiDAR;
   - `1,764,352` per semantic.
6. `layer_norm=false` rimuove tutte le LayerNorm.
7. Dropout assente.
8. Batch size 1 e batch size maggiore funzionano.

## 18.3 Tokenizzazione LQ

1. Token order esatto.
2. Shape `[B,122,64]` prima di `token_to_latent`.
3. Shape `[B,122,128]` dopo la projection.
4. Global mask shape `[B,122]`.
5. Dynamic flatten slot-major/time-minor.
6. Ego history frame 0 è il più antico.
7. Type IDs corretti.
8. Temporal embedding applicato solo a ego history e dynamic.
9. Slot embedding ripetuto correttamente sulla history dynamic.
10. Actor/object IDs non influenzano embedding.
11. Masked token embedding uguale a zero dopo il gating.

## 18.4 Mask invariance

Per ogni gruppo masked:

1. costruire due input identici sulle posizioni valide;
2. cambiare arbitrariamente il payload delle posizioni invalide;
3. mantenere le mask a zero;
4. verificare output LQ uguali entro tolleranza.

Gruppi:

```text
ego history passata
route
dynamic
static
controls
interactions
```

Testare anche padding totale di ciascun gruppo variabile.

## 18.5 LQ forward/backward

1. Output `[B,256]`.
2. Quattro blocchi presenti.
3. Sedici latent queries.
4. Cross-attention usa key padding mask.
5. Self-attention non usa scene mask.
6. Residual gating assente/disattivato.
7. Forward finito.
8. Backward finito.
9. Tutti i branch ricevono gradiente al primo backward.
10. Determinismo con stesso seed e dropout zero.

## 18.6 Integrazione TD3/SAC

Per TD3 e SAC, MLP e LQ:

1. actor encoder e critic encoder sono oggetti distinti;
2. Q1 e Q2 usano lo stesso critic encoder;
3. TD3 has distinct actor-target and critic-target encoders;
4. SAC has a distinct critic-target encoder and no target actor;
5. target encoder non ha parametri negli optimizer;
6. critic update cambia critic encoder;
7. critic update non cambia actor encoder;
8. actor update cambia actor encoder;
9. actor update non cambia critic encoder;
10. Polyak update cambia ogni target encoder nella direzione dell’online;
11. `share_features_extractor=false` è esplicito.

## 18.7 Integrazione PPO

Per MLP e LQ:

1. policy e value usano lo stesso encoder;
2. encoder appartiene una sola volta all’optimizer;
3. policy loss produce gradiente encoder;
4. value loss produce gradiente encoder;
5. update congiunto produce parametri finiti;
6. `share_features_extractor=true`;
7. `ortho_init=false`;
8. PPO non reinizializza né sovrascrive lo `state_dict` prodotto dalla factory del custom encoder (`ortho_init=false`).

## 18.8 Checkpoint

Per ogni configurazione core e algoritmo:

1. save/load round-trip;
2. output deterministico pre/post load;
3. manifest presente;
4. fingerprint corretto;
5. rifiuto schema `2363/107`;
6. rifiuto encoder type differente;
7. rifiuto sharing flag differente;
8. errore informativo;
9. checkpoint incompleto non usato per resume;
10. CPU load di checkpoint CPU;
11. mapping device esplicito per CUDA.
12. a stale or torn `latest.json` cannot select a mismatched generation.

## 18.9 Smoke training

Per ciascuna combinazione prevista:

```text
TD3 × lidar_mlp
TD3 × semantic_mlp
TD3 × semantic_lq

SAC × lidar_mlp
SAC × semantic_mlp
SAC × semantic_lq

PPO × lidar_mlp
PPO × semantic_mlp
PPO × semantic_lq
```

Eseguire almeno:

- environment reset;
- collect rollout breve;
- un update;
- save;
- load;
- nuova inferenza;
- assenza di NaN/Inf.

Questo smoke matrix verifica integrazione, non prestazioni scientifiche.

# 19. Acceptance gates

L’implementazione encoder è pronta per training quando:

```text
G1. observation schema v1.1 implementato e unico
G2. MLP passa tutti i test shape/gradient
G3. LQ produce esattamente 122 token
G4. mask invariance verificata
G5. TD3/SAC gradient routing verificato
G6. PPO sharing e inizializzazione verificati
G7. checkpoint fail-fast operativo
G8. smoke matrix completata
G9. nessun vecchio magic number 2363/107 nel percorso v1.1
G10. configurazioni core congelate nel manifest
```

Non è necessario ottenere già una policy performante per dichiarare completata l’implementazione architetturale.

# 20. Sequenza di implementazione

Ordine consigliato:

```text
1. Implementare SemanticObservationSchemaV11
2. Spostare flattening/unflattening nello schema unico
3. Aggiornare SemanticStateObservationV2
4. Implementare StackedLidarStateObservation e assert D=1540
5. Correggere ThesisEncoderFeatureExtractor
6. Aggiornare FlatMLPEncoder
7. Implementare tokenizzazione LQ a 122 token
8. Aggiornare proiezioni ed embedding LQ
9. Rimuovere/disattivare residual gating
10. Configurare sharing TD3/SAC/PPO
11. Disattivare ortho_init PPO per i custom encoder
12. Implementare checkpoint manifest
13. Aggiungere test unitari
14. Eseguire smoke matrix
15. Avviare baseline scalarizzata senza ACL
```

# 21. Estensioni future non core

Possibili estensioni dopo la pipeline principale:

- tokenizzazione settoriale LiDAR;
- learned pooling al posto della media;
- auxiliary masked-token prediction;
- behavior-cloning warm start;
- encoder learning rate separato;
- freeze/unfreeze;
- encoder unico actor–critic critic-driven;
- policy/value encoder separati in PPO;
- residual gating;
- recurrent latent state;
- cross-modal semantic + sparse LiDAR;
- diagnostica e visualizzazione delle attention weights.

Ogni estensione deve avere una configurazione separata e non modificare retroattivamente il significato della versione `1.0-final`.

# 22. Riferimenti principali

1. `docs/specifications/observation_v1.1_specification.md`.
2. Charraut, V. et al., *V-Max: Learning to Drive from Real-World Video*, 2025.
3. Jaegle, A. et al., *Perceiver: General Perception with Iterative Attention*, 2021.
4. Raffin, A. et al., *Stable-Baselines3: Reliable Reinforcement Learning Implementations*, 2021.
5. Li, Q. et al., *MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning*, 2022.
6. Dossier tecnico sullo stato degli encoder nel repository, 15 luglio 2026.

# Approval Record

- Status: `APPROVED`
- Authoritative: `YES`
- Approval date: `2026-07-16`
- Approved by: user
- Approval evidence: explicit user message in this Codex conversation:
  “Approvo l'encoder 1.0 comunque”
- Scope: this document and its dependency on
  `docs/specifications/observation_v1.1_specification.md`.
