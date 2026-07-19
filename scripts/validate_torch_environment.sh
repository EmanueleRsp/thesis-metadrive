#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <expected-torch-version> <torch-backend>" >&2
  exit 2
fi

expected_torch_version="$1"
torch_backend="$2"
validator_args=(--torch-version "$expected_torch_version")
if [[ "$torch_backend" != "cpu" ]]; then
  validator_args+=(--require-cusparselt)
fi
uv run --no-sync python scripts/validate_torch_install.py "${validator_args[@]}"

if check_output="$(uv pip check --no-config --python /opt/venv 2>&1)"; then
  printf '%s\n' "$check_output"
  exit 0
fi

if [[ "$torch_backend" != "cpu" && "$(uname -m)" == "aarch64" \
  && "$check_output" == *"Found 1 incompatibility"* \
  && "$check_output" == *"The package \`nvidia-cusparselt-cu12\` was built for a different platform"* ]]; then
  echo "Accepted ARM64 cuSPARSELt platform-validator false positive after ELF validation." >&2
  exit 0
fi

printf '%s\n' "$check_output" >&2
exit 1
