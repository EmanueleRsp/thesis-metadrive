#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_sdk="${GCLOUD_SDK_ROOT:-${repo_root}/.gcloud-sdk/google-cloud-sdk}"
local_gcloud="${local_sdk}/bin/gcloud"

if command -v gcloud >/dev/null 2>&1; then
  echo "Google Cloud CLI già installato: $(gcloud --version | head -n 1)"
  exit 0
fi

if [[ -x "$local_gcloud" ]]; then
  echo "Google Cloud CLI già installato localmente: $local_gcloud"
  "$local_gcloud" --version | head -n 1
  exit 0
fi

if command -v brew >/dev/null 2>&1; then
  echo "Installazione Google Cloud CLI tramite Homebrew..."
  brew install --cask google-cloud-sdk
  echo "Riavvia la shell oppure esegui il path.bash.inc del Google Cloud SDK."
  exit 0
fi

command -v curl >/dev/null 2>&1 || {
  echo "curl is required to install Google Cloud CLI locally." >&2
  exit 2
}

install_parent="$(dirname "$local_sdk")"
mkdir -p "$install_parent"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

installer="$tmp_dir/google-cloud-sdk-install.sh"
echo "Installazione Google Cloud CLI locale in: $local_sdk"
curl -fsSL https://sdk.cloud.google.com -o "$installer"
bash "$installer" --disable-prompts --install-dir="$install_parent"

if [[ ! -x "$local_gcloud" ]]; then
  echo "Google Cloud CLI installation completed, but gcloud was not found at $local_gcloud" >&2
  exit 2
fi

"$local_gcloud" --version | head -n 1
echo
echo "Local gcloud path: $local_gcloud"
echo "The repository Waymo scripts automatically use this local installation."
