#!/usr/bin/env bash

set -euo pipefail

docker_default_workdir="/workspace/thesis-metadrive"

usage() {
  cat <<'EOF'
Usage:
  scripts/tmux_seed_grid.sh [options] -- <command ...>

Creates a tmux session with one tiled pane per seed and runs the given
command in each pane, appending `seed=<seed>` automatically.

Examples:
  scripts/tmux_seed_grid.sh --session obs_sac -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config run_profile=thesis reward=monitor_only curriculum=disabled

  scripts/tmux_seed_grid.sh --session alg_lq --seed-list 0,1,2,3 --attach -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config run_profile=thesis obs=semantic_state \
      agent/planner/encoder=lq agent/planner/decoder=mlp_encoded

  scripts/tmux_seed_grid.sh \
    --session alg_lq \
    --docker-container thesis-metadrive-dev \
    --docker-workdir /workspace/thesis-metadrive \
    --attach -- \
    uv run --no-sync python -m thesis_rl.cli.train \
      --config-name config run_profile=thesis obs=semantic_state \
      agent/planner/encoder=lq agent/planner/decoder=mlp_encoded

Options:
  --session NAME       Session name. Default: auto-generated with timestamp.
  --window NAME        Window name. Default: runs
  --seed-count N       Use seeds 0..N-1. Default: 10
  --seed-start N       Start seed for --seed-count mode. Default: 0
  --seed-list LIST     Comma-separated explicit seeds (overrides --seed-count)
  --workdir DIR        Working directory for each pane. Default: current dir
  --docker-container   Run each pane command inside this Docker container
  --docker-workdir     `cd` here inside the container before running the command
                       Default with `--docker-container`:
                       `/workspace/thesis-metadrive`.
  --docker-shell       Shell used by `docker exec`. Default: bash
  --attach             Attach immediately after creating the session
  --dry-run            Print generated pane commands without creating tmux session
  -h, --help           Show this help message

Notes:
  - The command is sent to each pane exactly as provided, plus `seed=<seed>`.
  - With `--docker-container`, each pane runs `docker exec -it ...`.
  - The pane shell stays open after the command finishes, so logs remain visible.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'." >&2
    exit 1
  fi
}

shell_join_quoted() {
  local out=""
  local arg
  for arg in "$@"; do
    if [[ -n "$out" ]]; then
      out+=" "
    fi
    printf -v out '%s%q' "$out" "$arg"
  done
  printf '%s' "$out"
}

sanitize_session_name() {
  local value="$1"
  value="${value// /_}"
  value="${value//[^[:alnum:]_.:-]/_}"
  printf '%s' "$value"
}

session_name=""
window_name="runs"
seed_count=10
seed_start=0
seed_list_raw=""
workdir="$(pwd)"
attach=0
dry_run=0
docker_container=""
docker_workdir=""
docker_shell="bash"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      session_name="${2:-}"
      shift 2
      ;;
    --window)
      window_name="${2:-}"
      shift 2
      ;;
    --seed-count)
      seed_count="${2:-}"
      shift 2
      ;;
    --seed-start)
      seed_start="${2:-}"
      shift 2
      ;;
    --seed-list)
      seed_list_raw="${2:-}"
      shift 2
      ;;
    --workdir)
      workdir="${2:-}"
      shift 2
      ;;
    --docker-container)
      docker_container="${2:-}"
      shift 2
      ;;
    --docker-workdir)
      docker_workdir="${2:-}"
      shift 2
      ;;
    --docker-shell)
      docker_shell="${2:-}"
      shift 2
      ;;
    --attach)
      attach=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Error: missing command after '--'." >&2
  usage >&2
  exit 1
fi

if [[ -z "$docker_container" && ( -n "$docker_workdir" || "$docker_shell" != "bash" ) ]]; then
  echo "Error: --docker-workdir/--docker-shell require --docker-container." >&2
  exit 1
fi

if [[ -n "$docker_container" && -z "$docker_workdir" ]]; then
  docker_workdir="$docker_default_workdir"
fi

if [[ ! -d "$workdir" ]]; then
  echo "Error: workdir does not exist: $workdir" >&2
  exit 1
fi

if [[ -n "$seed_list_raw" ]]; then
  IFS=',' read -r -a seeds <<<"$seed_list_raw"
else
  if ! [[ "$seed_count" =~ ^[0-9]+$ ]] || ! [[ "$seed_start" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --seed-count must be a non-negative integer and --seed-start an integer." >&2
    exit 1
  fi
  if [[ "$seed_count" -le 0 ]]; then
    echo "Error: --seed-count must be > 0." >&2
    exit 1
  fi
  seeds=()
  for ((offset = 0; offset < seed_count; offset++)); do
    seeds+=("$((seed_start + offset))")
  done
fi

if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "Error: no seeds selected." >&2
  exit 1
fi

for seed in "${seeds[@]}"; do
  if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
    echo "Error: invalid seed '$seed'." >&2
    exit 1
  fi
done

command_args=("$@")
base_command="$(shell_join_quoted "${command_args[@]}")"

if [[ -z "$session_name" ]]; then
  session_name="seed_grid_$(date +%Y%m%d_%H%M%S)"
fi
session_name="$(sanitize_session_name "$session_name")"
window_name="$(sanitize_session_name "$window_name")"

if [[ $dry_run -eq 0 ]]; then
  require_cmd tmux
  if [[ -n "$docker_container" ]]; then
    require_cmd docker
  fi
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Error: tmux session '$session_name' already exists." >&2
    exit 1
  fi

  tmux new-session -d -s "$session_name" -n "$window_name" -c "$workdir"
  tmux set-option -t "$session_name" remain-on-exit on >/dev/null

  for ((idx = 1; idx < ${#seeds[@]}; idx++)); do
    tmux split-window -t "${session_name}:${window_name}" -c "$workdir"
    tmux select-layout -t "${session_name}:${window_name}" tiled >/dev/null
  done
fi

for ((idx = 0; idx < ${#seeds[@]}; idx++)); do
  seed="${seeds[$idx]}"
  pane_target="${session_name}:${window_name}.${idx}"
  run_command="${base_command} seed=${seed}"

  if [[ -n "$docker_container" ]]; then
    inner_command="$run_command"
    if [[ -n "$docker_workdir" ]]; then
      printf -v inner_command 'cd %q && %s' "$docker_workdir" "$inner_command"
    fi
    inner_command="${inner_command}; exec ${docker_shell}"
    pane_command="$(shell_join_quoted docker exec -it "$docker_container" "$docker_shell" -lc "$inner_command")"
  else
    pane_command="$run_command"
  fi

  if [[ $dry_run -eq 1 ]]; then
    printf '[pane %02d] %s\n' "$idx" "$pane_command"
    continue
  fi

  tmux select-pane -t "$pane_target" -T "seed=${seed}" >/dev/null
  tmux send-keys -t "$pane_target" "$pane_command" C-m
done

if [[ $dry_run -eq 1 ]]; then
  exit 0
fi

tmux select-layout -t "${session_name}:${window_name}" tiled >/dev/null
echo "Created tmux session: $session_name"
echo "Attach with: tmux attach -t $session_name"

if [[ $attach -eq 1 ]]; then
  exec tmux attach -t "$session_name"
fi
