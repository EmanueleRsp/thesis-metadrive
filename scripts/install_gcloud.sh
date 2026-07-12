#!/usr/bin/env bash
set -euo pipefail

if command -v gcloud >/dev/null 2>&1; then
  echo "Google Cloud CLI già installato: $(gcloud --version | head -n 1)"
  exit 0
fi

if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  echo "Installazione Google Cloud CLI tramite il repository ufficiale APT..."
  sudo apt-get update
  sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
  gcloud --version
  exit 0
fi

if command -v brew >/dev/null 2>&1; then
  echo "Installazione Google Cloud CLI tramite Homebrew..."
  brew install --cask google-cloud-sdk
  echo "Riavvia la shell oppure esegui il path.bash.inc del Google Cloud SDK."
  exit 0
fi

echo "Impossibile installare automaticamente Google Cloud CLI su questo sistema." >&2
echo "Consulta: https://cloud.google.com/sdk/docs/install" >&2
exit 2
