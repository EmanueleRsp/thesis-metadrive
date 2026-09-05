#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

check_only=0
verify=0
gpu=0
skip_docker_checks=0
skip_compose_config=0
skip_build=0
skip_smoke_check=0
skip_pytest=0

failures=0
warnings=0
changes=0
docker_available=0
docker_daemon_available=0
compose_config_ok=0
build_ok=0

info() {
  printf '[INFO] %s\n' "$*"
}

ok() {
  printf '[ OK ] %s\n' "$*"
}

warn() {
  warnings=$((warnings + 1))
  printf '[WARN] %s\n' "$*"
}

fail() {
  failures=$((failures + 1))
  printf '[FAIL] %s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  ./setup.sh [options]
  ./setup.sh --verify [--gpu]

Bootstraps the local machine setup for this repository and runs a set of
practical compatibility checks for the Docker-first workflow.

What it does:
  - creates .env from .env.example if missing
  - creates the host-mounted directories declared in .env
  - initializes git submodules when needed
  - compares HOST_UID/HOST_GID with the current user
  - checks Docker, Docker Compose, daemon reachability, and compose expansion
  - with --verify, builds the image and runs smoke checks and pytest
  - warns about likely Linux/NVIDIA compatibility issues

Options:
  --verify               Build and run the complete validation suite
  --gpu                  Use compose.gpu.yaml and verify CUDA access
  --check-only           Do not create or modify local files/directories
  --skip-docker-checks   Skip Docker/Compose/NVIDIA checks
  --skip-compose-config  Skip `docker compose config`
  --skip-build           Skip `docker compose build`
  --skip-smoke-check     Skip the container import smoke check
  --skip-pytest          Skip the container pytest run
  -h, --help             Show this help text

Exit codes:
  0 if no blocking issues were found
  1 if one or more blocking issues were found
EOF
}

read_env_var() {
  local file="$1"
  local key="$2"
  local default_value="${3:-}"
  local line
  local value

  if [[ -f "$file" ]]; then
    line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      value="${line#*=}"
      value="${value%$'\r'}"
      if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
          value="${value:1:${#value}-2}"
        elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      printf '%s' "$value"
      return 0
    fi
  fi

  printf '%s' "$default_value"
}

resolve_path() {
  local value="$1"

  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$repo_root" "$value"
  fi
}

ensure_dir() {
  local label="$1"
  local raw_path="$2"
  local resolved_path

  resolved_path="$(resolve_path "$raw_path")"
  if [[ $check_only -eq 1 ]]; then
    if [[ -d "$resolved_path" ]]; then
      ok "$label exists: $resolved_path"
    else
      warn "$label is missing: $resolved_path"
    fi
  else
    if [[ -d "$resolved_path" ]]; then
      ok "$label exists: $resolved_path"
    else
      mkdir -p "$resolved_path"
      ok "$label created: $resolved_path"
      changes=$((changes + 1))
    fi
  fi

  if [[ -d "$resolved_path" && ! -w "$resolved_path" ]]; then
    fail "$label is not writable by the current user: $resolved_path"
  fi
}

check_repo_layout() {
  local required_paths=(
    ".env.example"
    "compose.yaml"
    "pyproject.toml"
    "third_party/metadrive"
    "third_party/stable-baselines3"
    "third_party/scenarionet"
  )
  local path

  for path in "${required_paths[@]}"; do
    if [[ -e "$path" ]]; then
      ok "Found $path"
    else
      fail "Missing required repo path: $path"
    fi
  done
}

prepare_lfs_assets() {
  local pointers

  if ! git check-attr --all -- . >/dev/null 2>&1 || ! grep -q "filter=lfs" .gitattributes 2>/dev/null; then
    return
  fi

  if ! command -v git-lfs >/dev/null 2>&1; then
    fail "git-lfs is not installed, but tracked assets require it (see .gitattributes). Install it and re-run: https://git-lfs.com"
    return
  fi

  # A checkout made without the smudge filter leaves a ~130-byte pointer in
  # place of the asset. The frozen ScenarioNet selection index is one, and the
  # runtime parses it as JSON, so the failure surfaces far from its cause.
  # `git lfs ls-files` marks a materialized object with `*` and a pointer
  # with `-` in its second field.
  pointers="$(git lfs ls-files 2>/dev/null | awk '$2 == "-" {print $3}')"

  if [[ -z "$pointers" ]]; then
    ok "Git LFS assets are materialized"
    return
  fi

  if [[ $check_only -eq 1 ]]; then
    # A warning, not a failure: CI runs this mode on a checkout that leaves LFS
    # pointers in place on purpose (`actions/checkout` does not fetch them, and
    # fetching a 196 MB index on every run would exhaust the LFS bandwidth
    # quota), and none of the checks in that workflow read the asset.
    warn "Git LFS assets are unmaterialized pointers; run \`git lfs pull\` before training"
    return
  fi

  info "Materializing Git LFS assets"
  if git lfs pull; then
    ok "Git LFS assets pulled"
    changes=$((changes + 1))
  else
    fail "Failed to materialize Git LFS assets with \`git lfs pull\`"
  fi
}

prepare_submodules() {
  if ! command -v git >/dev/null 2>&1; then
    fail "git is not installed; cannot manage submodules"
    return
  fi

  if [[ ! -f .gitmodules ]]; then
    warn ".gitmodules not found; skipping submodule verification"
    return
  fi

  if [[ $check_only -eq 1 ]]; then
    info "Check-only mode: not modifying submodules"
    return
  fi

  info "Synchronizing and initializing git submodules"
  if git submodule sync --recursive >/dev/null 2>&1 && git submodule update --init --recursive; then
    ok "Git submodules initialized"
  else
    fail "Failed to initialize git submodules with \`git submodule update --init --recursive\`"
  fi
}

check_submodules() {
  local status_output
  local line
  local prefix

  if ! command -v git >/dev/null 2>&1; then
    fail "git is not installed; cannot verify submodules"
    return
  fi

  if [[ ! -f .gitmodules ]]; then
    warn ".gitmodules not found; skipping submodule verification"
    return
  fi

  if ! status_output="$(git submodule status --recursive 2>/dev/null)"; then
    fail "Unable to read submodule status; run \`git submodule update --init --recursive\`"
    return
  fi

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    prefix="${line:0:1}"
    case "$prefix" in
      "-")
        fail "Submodule not initialized: $line"
        ;;
      "U")
        fail "Submodule has merge conflicts: $line"
        ;;
      "+")
        warn "Submodule checkout differs from recorded commit: $line"
        ;;
      *)
        ok "Submodule ready: ${line#?}"
        ;;
    esac
  done <<<"$status_output"
}

check_uid_gid() {
  local env_file="$1"
  local expected_uid expected_gid actual_uid actual_gid

  expected_uid="$(read_env_var "$env_file" HOST_UID "1000")"
  expected_gid="$(read_env_var "$env_file" HOST_GID "1000")"
  actual_uid="$(id -u 2>/dev/null || true)"
  actual_gid="$(id -g 2>/dev/null || true)"

  if [[ -n "$actual_uid" ]]; then
    if [[ "$actual_uid" == "$expected_uid" ]]; then
      ok "HOST_UID matches current user: $actual_uid"
    else
      warn "HOST_UID=$expected_uid but current user has UID $actual_uid; update .env if needed"
    fi
  else
    warn "Could not determine current UID"
  fi

  if [[ -n "$actual_gid" ]]; then
    if [[ "$actual_gid" == "$expected_gid" ]]; then
      ok "HOST_GID matches current user: $actual_gid"
    else
      warn "HOST_GID=$expected_gid but current user has GID $actual_gid; update .env if needed"
    fi
  else
    warn "Could not determine current GID"
  fi
}

check_torch_backend() {
  local env_file="$1"
  torch_backend="$(read_env_var "$env_file" TORCH_BACKEND "cu128")"
  torch_version="$(read_env_var "$env_file" TORCH_VERSION "2.9.1")"

  if [[ "$torch_version" == "2.9.1" ]]; then
    ok "TORCH_VERSION is supported: $torch_version"
  else
    fail "Unsupported TORCH_VERSION=$torch_version; use the cross-architecture pin 2.9.1"
  fi

  case "$torch_backend" in
    cpu|cu126|cu128)
      ok "TORCH_BACKEND is supported: $torch_backend"
      ;;
    *)
      fail "Unsupported TORCH_BACKEND=$torch_backend; use cpu, cu126, or cu128"
      ;;
  esac

  if [[ $gpu -eq 1 && "$torch_backend" == "cpu" ]]; then
    fail "--gpu requires TORCH_BACKEND=cu126 or cu128"
  fi
  if [[ $gpu -eq 0 && "$torch_backend" != "cpu" ]]; then
    info "TORCH_BACKEND=$torch_backend is CUDA-capable; base Compose can still run it on CPU"
  fi
  if [[ -f "$env_file" ]] && grep -q '^TORCH_INDEX_URL=' "$env_file"; then
    warn "TORCH_INDEX_URL is obsolete and ignored; replace it with TORCH_BACKEND=$torch_backend"
  fi
}

check_host_platform() {
  local os_name

  os_name="$(uname -s 2>/dev/null || printf 'unknown')"
  case "$os_name" in
    Linux)
      ok "Host OS is Linux"
      ;;
    *)
      warn "Host OS is $os_name; this repo is primarily tuned for Linux/WSL with NVIDIA Docker support"
      ;;
  esac
}

check_host_resources() {
  local arch available_kb free_kb
  arch="$(uname -m 2>/dev/null || true)"
  case "$arch" in
    x86_64|aarch64)
      ok "Host architecture is supported for the pinned PyTorch CUDA profiles: $arch"
      ;;
    *)
      warn "Host architecture is ${arch:-unknown}; the pinned PyTorch CUDA profiles are validated only on x86_64 and aarch64"
      ;;
  esac

  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  if [[ -n "$available_kb" && "$available_kb" -lt 12582912 ]]; then
    warn "Less than 12 GiB of RAM is currently available"
  fi
  free_kb="$(df -Pk "$repo_root" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  if [[ -n "$free_kb" && "$free_kb" -lt 12582912 ]]; then
    warn "Less than 12 GiB of disk space is available for the repository filesystem"
  fi
}

check_docker_stack() {
  local docker_info_ok=0
  local runtimes=""
  local compute_cap=""
  local compute_major=""

  if ! command -v docker >/dev/null 2>&1; then
    fail "docker is not installed or not on PATH"
    return
  fi
  ok "docker command is available"
  docker_available=1

  if docker compose version >/dev/null 2>&1; then
    ok "docker compose is available"
  else
    fail "docker compose is unavailable; install Docker Compose v2"
  fi

  if docker info >/dev/null 2>&1; then
    ok "Docker daemon is reachable"
    docker_info_ok=1
    docker_daemon_available=1
  else
    fail "Docker daemon is not reachable; start Docker or fix permissions"
  fi

  if [[ $skip_compose_config -eq 0 ]]; then
    if docker compose config >/dev/null 2>&1; then
      ok "docker compose config succeeded"
      compose_config_ok=1
    else
      fail "docker compose config failed; check .env values and compose mounts"
    fi
  else
    info "Skipping docker compose config"
    compose_config_ok=1
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi can access an NVIDIA GPU on the host"
    if [[ $gpu -eq 1 ]]; then
      compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
      compute_major="${compute_cap%%.*}"
      if [[ "$compute_major" =~ ^[0-9]+$ && "$compute_major" -ge 12 && "$torch_backend" != "cu128" ]]; then
        fail "GPU compute capability $compute_cap requires TORCH_BACKEND=cu128"
      elif [[ -n "$compute_cap" ]]; then
        ok "Detected GPU compute capability: $compute_cap"
      fi
    fi
  else
    if [[ $gpu -eq 1 ]]; then
      fail "nvidia-smi cannot access a GPU but --gpu was requested"
    else
      info "nvidia-smi cannot access a GPU; CPU-only validation remains available"
    fi
  fi

  if [[ $docker_info_ok -eq 1 && $gpu -eq 1 ]]; then
    runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    if [[ -n "$runtimes" && "$runtimes" == *nvidia* ]]; then
      ok "Docker advertises an NVIDIA runtime"
    else
      fail "Docker does not advertise an NVIDIA runtime but --gpu was requested"
    fi
  fi
}

run_docker_build() {
  if [[ $check_only -eq 1 || $skip_docker_checks -eq 1 || $skip_build -eq 1 ]]; then
    return
  fi

  if [[ $docker_available -ne 1 || $docker_daemon_available -ne 1 || $compose_config_ok -ne 1 ]]; then
    warn "Skipping docker compose build because Docker is not ready"
    return
  fi

  info "Building the Docker image with docker compose build"
  if docker compose build; then
    ok "docker compose build succeeded"
    build_ok=1
  else
    fail "docker compose build failed"
  fi
}

run_import_smoke_check() {
  if [[ $check_only -eq 1 || $skip_docker_checks -eq 1 || $skip_smoke_check -eq 1 ]]; then
    return
  fi

  if [[ $skip_build -eq 0 && $build_ok -ne 1 ]]; then
    warn "Skipping import smoke check because the Docker image build did not complete"
    return
  fi

  if [[ $docker_available -ne 1 || $docker_daemon_available -ne 1 || $compose_config_ok -ne 1 ]]; then
    warn "Skipping import smoke check because Docker is not ready"
    return
  fi

  info "Running the container import smoke check"
  if docker compose run --rm dev bash -lc "uv run --no-sync python -c 'import thesis_rl, metadrive, stable_baselines3, torch; assert tuple(map(int, torch.__version__.split(\"+\")[0].split(\".\")[:2])) >= (2, 9); print(\"imports ok; torch=\" + torch.__version__)' && bash scripts/validate_torch_environment.sh '$torch_version' '$torch_backend'"; then
    ok "Container import smoke check succeeded"
  else
    fail "Container import smoke check failed"
  fi
}

run_gpu_smoke_check() {
  if [[ $gpu -ne 1 || $check_only -eq 1 || $skip_docker_checks -eq 1 ]]; then
    return 0
  fi
  if [[ $skip_build -eq 0 && $build_ok -ne 1 ]]; then
    warn "Skipping CUDA check because the Docker image build did not complete"
    return 0
  fi
  if [[ $docker_available -ne 1 || $docker_daemon_available -ne 1 || $compose_config_ok -ne 1 ]]; then
    return 0
  fi
  info "Verifying CUDA access inside the container"
  if docker compose run --rm dev uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; x = torch.tensor([1.0], device="cuda"); assert (x * 2).item() == 2.0; print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")'; then
    ok "Container CUDA smoke check succeeded"
  else
    fail "Container CUDA smoke check failed"
  fi
}

run_pytest_suite() {
  if [[ $check_only -eq 1 || $skip_docker_checks -eq 1 || $skip_pytest -eq 1 ]]; then
    return
  fi

  if [[ $skip_build -eq 0 && $build_ok -ne 1 ]]; then
    warn "Skipping pytest because the Docker image build did not complete"
    return
  fi

  if [[ $docker_available -ne 1 || $docker_daemon_available -ne 1 || $compose_config_ok -ne 1 ]]; then
    warn "Skipping pytest because Docker is not ready"
    return
  fi

  info "Running pytest in the container"
  if docker compose run --rm dev uv run --no-sync python -m pytest -q; then
    ok "Container pytest run succeeded"
  else
    fail "Container pytest run failed"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)
      verify=1
      shift
      ;;
    --gpu)
      gpu=1
      shift
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    --skip-docker-checks)
      skip_docker_checks=1
      shift
      ;;
    --skip-compose-config)
      skip_compose_config=1
      shift
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-smoke-check)
      skip_smoke_check=1
      shift
      ;;
    --skip-pytest)
      skip_pytest=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $verify -eq 0 ]]; then
  skip_build=1
  skip_smoke_check=1
  skip_pytest=1
fi
if [[ $gpu -eq 1 ]]; then
  export COMPOSE_FILE="compose.yaml:compose.gpu.yaml"
fi

info "Repository root: $repo_root"
check_repo_layout
prepare_lfs_assets
prepare_submodules
check_submodules

env_file=".env"
env_source=".env"

if [[ -f "$env_file" ]]; then
  ok ".env already exists"
else
  if [[ $check_only -eq 1 ]]; then
    warn ".env is missing; run ./setup.sh without --check-only to create it from .env.example"
    env_source=".env.example"
  else
    cp .env.example .env
    if [[ "$(uname -s 2>/dev/null || true)" == "Linux" ]]; then
      sed -i "s/^USER_NAME=.*/USER_NAME=$(id -un)/; s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env
    fi
    ok "Created .env from .env.example"
    changes=$((changes + 1))
  fi
fi

host_outputs_dir="$(read_env_var "$env_source" HOST_OUTPUTS_DIR "./outputs")"
host_data_dir="$(read_env_var "$env_source" HOST_DATA_DIR "./data")"
host_container_home_dir="$(read_env_var "$env_source" HOST_CONTAINER_HOME_DIR "./.container-home")"

check_uid_gid "$env_source"
check_torch_backend "$env_source"
ensure_dir "HOST_OUTPUTS_DIR" "$host_outputs_dir"
ensure_dir "HOST_DATA_DIR" "$host_data_dir"
ensure_dir "HOST_CONTAINER_HOME_DIR" "$host_container_home_dir"

check_host_platform
check_host_resources
if [[ $skip_docker_checks -eq 1 ]]; then
  info "Skipping Docker/Compose/NVIDIA checks"
else
  check_docker_stack
fi
run_docker_build
run_import_smoke_check
run_gpu_smoke_check
run_pytest_suite

printf '\n'
info "Summary: $failures blocking issue(s), $warnings warning(s), $changes local change(s)"

if [[ $failures -eq 0 ]]; then
  printf '\n'
  printf 'Setup completed.\n\n'
  printf 'Recommended next steps:\n'
  printf '  1. Review .env, especially TORCH_BACKEND and host paths.\n'
  if [[ "$torch_backend" == "cpu" ]]; then
    printf '  2. Start the CPU container: make up\n'
  else
    printf '  2. Start the NVIDIA container: make up-gpu\n'
  fi
  printf '  3. Enter it: make shell\n'
  printf '  4. Run tests: make test\n'
  if [[ "$torch_backend" == "cpu" ]]; then
    printf '  5. Verify end-to-end training: make smoke\n'
  else
    printf '  5. Check CUDA and smoke training: make gpu-check && make smoke-gpu\n'
  fi
  exit 0
fi

printf '\n'
printf 'Fix the blocking issues above, then rerun ./setup.sh.\n'
exit 1
