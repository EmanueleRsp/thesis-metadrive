#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

local_gcloud="${repo_root}/.gcloud-sdk/google-cloud-sdk/bin/gcloud"
if ! command -v gcloud >/dev/null 2>&1 && [[ -x "$local_gcloud" ]]; then
  PATH="${local_gcloud%/gcloud}:${PATH}"
  export PATH
fi

command -v gcloud >/dev/null 2>&1 || {
  echo "gcloud CLI not found. Run 'make install-gcloud' first." >&2
  exit 2
}

gcloud_account() {
  gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null \
    | head -n 1
}

active_account="$(gcloud_account || true)"
local_config="${repo_root}/.gcloud-sdk/config"
if [[ -z "$active_account" && "${CLOUDSDK_CONFIG:-}" == "$local_config" ]]; then
  # A stale empty repository config must not hide a valid default gcloud login.
  unset CLOUDSDK_CONFIG
  active_account="$(gcloud_account || true)"
fi
if [[ -z "$active_account" && -z "${CLOUDSDK_CONFIG:-}" ]]; then
  if [[ -d "$local_config" ]] && active_account="$(CLOUDSDK_CONFIG="$local_config" gcloud_account || true)" \
    && [[ -n "$active_account" ]]; then
    export CLOUDSDK_CONFIG="$local_config"
  fi
fi

if [[ -z "$active_account" ]]; then
  echo "No active Google Cloud account found; starting OAuth login..."
  gcloud auth login ${GCLOUD_LOGIN_FLAGS:-}
  active_account="$(gcloud_account || true)"
fi

[[ -n "$active_account" ]] || {
  echo "Google Cloud login completed without an active account." >&2
  exit 2
}

gcs_uri="${WAYMO_GCS_URI:-gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s}"
object_pattern="${WAYMO_GCS_OBJECT_PATTERN:-training_20s.tfrecord-*}"
source_uri="${gcs_uri%/}/${object_pattern}"

echo "Google Cloud account: $active_account"
echo "Checking Waymo bucket access..."
gcloud storage ls "$source_uri" >/dev/null
echo "Waymo bucket access: OK"
