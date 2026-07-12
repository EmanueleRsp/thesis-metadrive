# Conversione Waymo per ScenarioNet

La conversione Waymo `training_20s` usa un ambiente separato dal runtime RL.
L'immagine principale del progetto non installa TensorFlow né le dipendenze
specifiche del dataset, così training ed evaluation restano riproducibili e
leggeri.

## Prerequisiti

- Docker Compose disponibile;
- dataset Waymo autorizzato dall'utente;
- directory raw contenente file con nome `training_20s.tfrecord*`;
- spazio disco sufficiente per i file convertiti.

L'accettazione della licenza e l'autenticazione Google non sono automatizzate;
il download dei file viene invece automatizzato dalla pipeline dopo il login.

## Autenticazione Google Cloud (una sola volta)

`gcloud` è il programma da riga di comando per autenticarsi a Google Cloud e
leggere il bucket che contiene i TFRecord. Non è una dipendenza Python del
progetto e non richiede di inserire password, token o file JSON in `.env`.

Installa il [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), poi
esegui una sola volta:

Su Ubuntu/WSL puoi installarlo con questi comandi, **uno alla volta**. Non
separare il carattere `|` dalla riga e non eseguire `\` da solo:

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
sudo apt-get update
sudo apt-get install -y google-cloud-cli
gcloud --version
```

Durante il secondo `apt-get update` devi vedere una riga riferita a
`packages.cloud.google.com`. Se `google-cloud-cli` non viene trovato, il
repository non è stato registrato correttamente.

Quando compare il prompt `>` significa che la shell sta aspettando il seguito
di un comando multilinea: premi `Ctrl+C` e riparti dalla riga completa.

```bash
make waymo-auth
```

Il comando avvia `gcloud init`, che apre il flusso di login, configura l'account
e il progetto Google Cloud. L'account deve avere accesso al Waymo Open Dataset
e devi aver accettato le relative condizioni d'uso. Per la conversione locale
non serve creare API key: il download usa le credenziali OAuth del CLI.

Su WSL, se il browser non viene aperto automaticamente, usa:

```bash
make waymo-auth GCLOUD_INIT_FLAGS=--console-only
```

Il comando mostrerà un URL da copiare nel browser Windows.

## Pipeline automatica

Le impostazioni sono in `.env` e partono da `.env.example`:

- `WAYMO_GCS_URI` e `WAYMO_GCS_OBJECT_PATTERN`: sorgente Google Cloud;
- `WAYMO_RAW_DATA_PATH`: directory raw sul host;
- `WAYMO_NUM_FILES=1`: smoke test iniziale; vuoto per tutti i file già scaricati;
- `WAYMO_NUM_WORKERS` e `WAYMO_OVERWRITE`: opzioni del converter.

Con l'autenticazione già configurata, il comando unico è:

```bash
make waymo-pipeline
```

La pipeline controlla l'account attivo, scarica i file mancanti, verifica la
presenza di `training_20s.tfrecord*`, costruisce il container dedicato e avvia
la conversione. Il default scarica un solo shard e lo converte per lo smoke
test, evitando un download completo accidentale.

Per il dataset completo, dopo aver verificato lo smoke test, imposta in `.env`:

```dotenv
WAYMO_NUM_FILES=
WAYMO_GCS_OBJECT_PATTERN=training_20s.tfrecord-*
WAYMO_SKIP_DOWNLOAD_IF_PRESENT=false
```

e rilancia `make waymo-pipeline`. I file già presenti non vengono riscaricati.

## Build manuale dell'immagine dedicata

```bash
make build-waymo
```

L'immagine installa TensorFlow 2.11 e il converter dei commit locali di
ScenarioNet/MetaDrive. Il servizio è esposto dal profilo Compose `waymo` e non
viene avviato da `make up` o dal runtime RL standard.

## Conversione

In alternativa, senza scaricare, si può convertire una directory raw già
disponibile:

```bash
make waymo-convert \
  WAYMO_RAW_DATA_PATH=/percorso/assoluto/waymo_raw \
  NUM_FILES=1 \
  OVERWRITE=1
```

`WAYMO_RAW_DATA_PATH` è un path del **host**: Compose lo monta in sola lettura
come `/workspace/waymo_raw` nel container, quindi non è necessario che il raw
dataset si trovi sotto `./data`.

Senza `NUM_FILES` viene processata l'intera directory. L'output predefinito è
`${SCENARIONET_DATA_ROOT}/waymo/database` (normalmente
`./data/scenarionet/waymo/database` sul host). È possibile cambiare la data
root tramite `SCENARIONET_DATA_ROOT` e il mount host tramite `HOST_DATA_DIR`.

Il wrapper passa sempre al converter:

- `dataset_name=waymo`;
- `version=training_20s`;
- il path raw e il database assoluti;
- il numero di worker configurato (8 di default).

Prima dell'avvio vengono rifiutate directory inesistenti o varianti Waymo
diverse da `training_20s`. Dopo la conversione, la pipeline F5 carica i pickle,
estrae il catalogo e assegna gli split interni senza rompere i gruppi logici.

## Verifiche successive

La conversione reale non è inclusa nei test ordinari perché richiede dati
licenziati. Una volta disponibile il database, eseguire nell'ambiente del
progetto i comandi di validazione e costruzione delle viste runtime descritti
nel tracker [`scenarionet_integration_implementation_plan.md`](../specs/scenarionet_integration_implementation_plan.md).

In caso di errore TensorFlow/protobuf, conservare l'output del container nel
tracker: non installare automaticamente queste dipendenze nell'immagine RL.
