#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

docker_container="thesis-metadrive-dev"
container_workdir="/workspace/thesis-metadrive"
seed_start=0
seed_end=2
tag="v3"
algorithms_csv="td3,sac,ppo"
run_profile="thesis"
total_timesteps="1500000"
eval_episodes="20"
final_eval_episodes="100"
window_name="runs"
poll_seconds="60"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_algorithm_selection.sh run [options]
  scripts/run_algorithm_selection.sh run-parallel [options]
  scripts/run_algorithm_selection.sh cleanup [options]
  scripts/run_algorithm_selection.sh launch [options]
  scripts/run_algorithm_selection.sh analyze [options]

Purpose:
  Wrapper per la Fase 2 di selezione algoritmo. Lancia i tre gruppi di run
  in tmux e, in un secondo momento, aggrega e analizza i risultati.

Examples:
  scripts/run_algorithm_selection.sh run --tag v3
  scripts/run_algorithm_selection.sh run-parallel --tag v3
  scripts/run_algorithm_selection.sh cleanup --tag v3
  scripts/run_algorithm_selection.sh launch --algorithms td3 --tag v3
  scripts/run_algorithm_selection.sh launch --algorithms td3,sac,ppo --tag v3
  scripts/run_algorithm_selection.sh analyze --tag v3

Options:
  --algorithms LIST        Comma-separated list from: td3,sac,ppo
                           Default: td3,sac,ppo
  --tag TAG                Suffix for tmux sessions / experiment groups.
                           Default: v3
  --seed-start N           Inclusive start seed. Default: 0
  --seed-end N             Inclusive end seed. Default: 2
  --docker-container NAME  Docker container name. Default: thesis-metadrive-dev
  --container-workdir DIR  Project path inside container.
                           Default: /workspace/thesis-metadrive
  --run-profile NAME       Analysis run-profile label. Default: thesis
  --total-timesteps N      Expected total timesteps for analysis.
                           Default: 1500000
  --eval-episodes N        Expected intermediate eval episodes.
                           Default: 20
  --final-eval-episodes N  Expected final eval episodes.
                           Default: 100
  --window NAME            tmux window name. Default: runs
  --poll-seconds N         Poll interval in seconds for `run`.
                           Default: 60
  -h, --help               Show this help

Notes:
  - `launch` crea le sessioni tmux ma non aspetta che finiscano.
  - `run` esegue i gruppi in sequenza e lancia l'analisi finale da solo.
  - `run-parallel` lancia tutti i gruppi richiesti insieme, aspetta che finiscano
    tutti, poi esegue l'analisi finale.
  - `cleanup` chiude le sessioni tmux associate agli algoritmi richiesti.
  - Esegui `analyze` solo dopo che tutte le run sono completate.
  - Per evitare OOM GPU, e' spesso meglio lanciare un algoritmo per volta con
    `--algorithms td3` / `sac` / `ppo`.
  - Nel container gli output sono letti da `OUTPUTS_ROOT`
    (default: `/workspace/outputs`).
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "missing required command '$1'"
  fi
}

sanitize_tag() {
  local value="$1"
  value="${value// /_}"
  value="${value//[^[:alnum:]_.:-]/_}"
  printf '%s' "$value"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --algorithms)
        algorithms_csv="${2:-}"
        shift 2
        ;;
      --tag)
        tag="${2:-}"
        shift 2
        ;;
      --seed-start)
        seed_start="${2:-}"
        shift 2
        ;;
      --seed-end)
        seed_end="${2:-}"
        shift 2
        ;;
      --docker-container)
        docker_container="${2:-}"
        shift 2
        ;;
      --container-workdir)
        container_workdir="${2:-}"
        shift 2
        ;;
      --run-profile)
        run_profile="${2:-}"
        shift 2
        ;;
      --total-timesteps)
        total_timesteps="${2:-}"
        shift 2
        ;;
      --eval-episodes)
        eval_episodes="${2:-}"
        shift 2
        ;;
      --final-eval-episodes)
        final_eval_episodes="${2:-}"
        shift 2
        ;;
      --window)
        window_name="${2:-}"
        shift 2
        ;;
      --poll-seconds)
        poll_seconds="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option '$1'"
        ;;
    esac
  done
}

validate_common() {
  [[ "$seed_start" =~ ^-?[0-9]+$ ]] || die "--seed-start must be an integer"
  [[ "$seed_end" =~ ^-?[0-9]+$ ]] || die "--seed-end must be an integer"
  [[ "$seed_end" -ge "$seed_start" ]] || die "--seed-end must be >= --seed-start"
  [[ "$eval_episodes" =~ ^[0-9]+$ ]] || die "--eval-episodes must be an integer"
  [[ "$final_eval_episodes" =~ ^[0-9]+$ ]] || die "--final-eval-episodes must be an integer"
  [[ "$total_timesteps" =~ ^[0-9]+$ ]] || die "--total-timesteps must be an integer"
  [[ "$poll_seconds" =~ ^[0-9]+$ ]] || die "--poll-seconds must be an integer"
  tag="$(sanitize_tag "$tag")"
}

collect_algorithms() {
  IFS=',' read -r -a algorithms <<<"$algorithms_csv"
  [[ ${#algorithms[@]} -gt 0 ]] || die "no algorithms selected"
  local alg
  for alg in "${algorithms[@]}"; do
    case "$alg" in
      td3|sac|ppo) ;;
      *)
        die "unsupported algorithm '$alg' (expected td3, sac, or ppo)"
        ;;
    esac
  done
}

session_name_for() {
  local alg="$1"
  printf 'qual_%s_sb3_lidar_thesis_%s' "$alg" "$tag"
}

experiment_group_for() {
  local alg="$1"
  printf 'EXP_qual_%s_sb3_lidar_thesis_%s' "$alg" "$tag"
}

preset_for() {
  local alg="$1"
  printf 'presets/selection/%s_sb3_qual_lidar_thesis' "$alg"
}

seed_list_csv() {
  local items=()
  local seed
  for ((seed = seed_start; seed <= seed_end; seed++)); do
    items+=("$seed")
  done
  local out
  IFS=',' read -r -a _dummy <<< ""
  out="${items[*]}"
  printf '%s' "${out// /,}"
}

launch_runs() {
  require_cmd docker
  require_cmd tmux

  local alg
  for alg in "${algorithms[@]}"; do
    local session_name
    local experiment_group
    local preset_name
    session_name="$(session_name_for "$alg")"
    experiment_group="$(experiment_group_for "$alg")"
    preset_name="$(preset_for "$alg")"

    log "launching group for $alg"
    echo "  session: $session_name"
    echo "  group:   $experiment_group"
    echo "  preset:  $preset_name"
    echo "  seeds:   ${seed_start}..${seed_end}"

    "$repo_root/scripts/tmux_seed_grid.sh" \
      --session "$session_name" \
      --window "$window_name" \
      --seed-start "$seed_start" \
      --seed-end "$seed_end" \
      --workdir "$repo_root" \
      --docker-container "$docker_container" \
      --docker-workdir "$container_workdir" -- \
      uv run --no-sync python -m thesis_rl.cli.train \
        --config-name "$preset_name" \
        "analysis.experiment_group=$experiment_group"
  done

  echo
  log "tmux sessions created"
  echo "  monitor with: tmux ls"
  echo "  analyze with: scripts/run_algorithm_selection.sh analyze --tag $tag"
}

wait_for_group() {
  local alg="$1"
  local group="$2"

  docker exec -i "$docker_container" bash -s -- \
    "$group" \
    "$alg" \
    "$seed_start" \
    "$seed_end" \
    "$poll_seconds" <<'EOF'
set -euo pipefail

group="$1"
alg="$2"
seed_start="$3"
seed_end="$4"
poll_seconds="$5"

OUT_ROOT="${OUTPUTS_ROOT:-/workspace/outputs}"
expected_count=$((seed_end - seed_start + 1))
stale_polls=0

latest_run_dir_for_seed() {
  local seed_root="$1"
  if [[ ! -d "$seed_root" ]]; then
    return 1
  fi
  find "$seed_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

has_live_train_process() {
  pgrep -af "analysis.experiment_group=${group}" >/dev/null 2>&1
}

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  completed=0
  pending=0
  running=0
  failed=0
  interrupted=0
  details=()

  for ((seed = seed_start; seed <= seed_end; seed++)); do
    seed_root="$OUT_ROOT/$group/$alg/seed_${seed}"
    status="pending"
    latest_run="$(latest_run_dir_for_seed "$seed_root" || true)"

    if [[ -n "$latest_run" ]]; then
      metadata_path="$latest_run/artifacts/run_metadata.yaml"
      final_eval_path="$latest_run/csv/final_eval.csv"
      if [[ -f "$metadata_path" ]]; then
        if grep -q '^status: completed$' "$metadata_path" && [[ -s "$final_eval_path" ]]; then
          status="completed"
        elif grep -q '^status: failed$' "$metadata_path"; then
          status="failed"
        elif grep -q '^status: interrupted$' "$metadata_path"; then
          status="interrupted"
        else
          status="running"
        fi
      else
        status="running"
      fi
    fi

    case "$status" in
      completed) completed=$((completed + 1)) ;;
      pending) pending=$((pending + 1)) ;;
      running) running=$((running + 1)) ;;
      failed) failed=$((failed + 1)) ;;
      interrupted) interrupted=$((interrupted + 1)) ;;
    esac
    details+=("seed_${seed}=${status}")
  done

  printf '[%s] [wait][%s] completed=%d/%d pending=%d running=%d failed=%d interrupted=%d\n' \
    "$now" "$alg" "$completed" "$expected_count" "$pending" "$running" "$failed" "$interrupted"
  printf '[%s] [wait][%s] %s\n' "$now" "$alg" "${details[*]}"

  if [[ "$failed" -gt 0 || "$interrupted" -gt 0 ]]; then
    printf '[%s] [wait][%s] stopping because at least one seed failed or was interrupted\n' "$now" "$alg"
    exit 2
  fi

  if [[ "$completed" -eq "$expected_count" ]]; then
    printf '[%s] [wait][%s] all seeds completed\n' "$now" "$alg"
    exit 0
  fi

  if has_live_train_process; then
    stale_polls=0
    printf '[%s] [wait][%s] live training process detected for group=%s\n' "$now" "$alg" "$group"
  else
    stale_polls=$((stale_polls + 1))
    printf '[%s] [wait][%s] no live training process detected for group=%s (stale poll %d)\n' \
      "$now" "$alg" "$group" "$stale_polls"
    if [[ "$stale_polls" -ge 2 ]]; then
      printf '[%s] [wait][%s] stopping because no live process remains and runs are not completed\n' "$now" "$alg"
      exit 3
    fi
  fi

  printf '[%s] [wait][%s] next poll in %ss\n' "$now" "$alg" "$poll_seconds"
  sleep "$poll_seconds"
done
EOF
}

run_all() {
  require_cmd docker
  require_cmd tmux

  local alg
  for alg in "${algorithms[@]}"; do
    local session_name
    local experiment_group
    local preset_name
    session_name="$(session_name_for "$alg")"
    experiment_group="$(experiment_group_for "$alg")"
    preset_name="$(preset_for "$alg")"

    log "starting sequential run for $alg"
    "$repo_root/scripts/tmux_seed_grid.sh" \
      --session "$session_name" \
      --window "$window_name" \
      --seed-start "$seed_start" \
      --seed-end "$seed_end" \
      --workdir "$repo_root" \
      --docker-container "$docker_container" \
      --docker-workdir "$container_workdir" -- \
      uv run --no-sync python -m thesis_rl.cli.train \
        --config-name "$preset_name" \
        "analysis.experiment_group=$experiment_group"

    log "waiting for $alg to complete (poll every ${poll_seconds}s)"
    wait_for_group "$alg" "$experiment_group"
    log "$alg completed successfully"
  done

  log "all algorithm groups completed, starting analysis"
  analyze_runs
}

run_all_parallel() {
  require_cmd docker
  require_cmd tmux

  log "launching all requested algorithm groups in parallel"
  launch_runs

  local alg
  for alg in "${algorithms[@]}"; do
    local experiment_group
    experiment_group="$(experiment_group_for "$alg")"
    log "waiting for $alg to complete (poll every ${poll_seconds}s)"
    wait_for_group "$alg" "$experiment_group"
    log "$alg completed successfully"
  done

  log "all parallel algorithm groups completed, starting analysis"
  analyze_runs
}

cleanup_sessions() {
  require_cmd tmux

  local alg
  for alg in "${algorithms[@]}"; do
    local session_name
    session_name="$(session_name_for "$alg")"
    if tmux has-session -t "$session_name" 2>/dev/null; then
      log "killing tmux session $session_name"
      tmux kill-session -t "$session_name"
    else
      log "tmux session $session_name not present"
    fi
  done
}

analyze_runs() {
  require_cmd docker

  local groups=()
  local alg
  for alg in "${algorithms[@]}"; do
    groups+=("$(experiment_group_for "$alg")")
  done

  log "starting analysis for tag=$tag on algorithms=${algorithms_csv}"

  docker exec -i "$docker_container" bash -s -- \
    "$container_workdir" \
    "$run_profile" \
    "$tag" \
    "$(seed_list_csv)" \
    "$total_timesteps" \
    "$eval_episodes" \
    "$final_eval_episodes" \
    "${groups[@]}" <<'EOF'
set -euo pipefail

container_workdir="$1"
run_profile="$2"
tag="$3"
seed_list="$4"
total_timesteps="$5"
eval_episodes="$6"
final_eval_episodes="$7"
shift 7
groups=("$@")

cd "$container_workdir"

OUT_ROOT="${OUTPUTS_ROOT:-/workspace/outputs}"
CMP_BASE_ROOT="${COMPARISON_ROOT:-$OUT_ROOT/_comparisons}"
CMP_ROOT="$CMP_BASE_ROOT/qual_alg_compare_${tag}"
ANALYSIS_ROOT="$CMP_ROOT/analysis"

rm -rf "$CMP_ROOT"
mkdir -p "$CMP_ROOT" "$ANALYSIS_ROOT"

for group in "${groups[@]}"; do
  echo "[analyze] copying $group"
  rsync -a "$OUT_ROOT/$group/" "$CMP_ROOT/$group/"
done

uv run --no-sync python -m thesis_rl.analysis.run_analysis \
  --outputs-root "$CMP_ROOT" \
  --analysis-root "$ANALYSIS_ROOT" \
  --run-profile "$run_profile" \
  --only all \
  --no-videos \
  --seed-list "$seed_list" \
  --total-timesteps "$total_timesteps" \
  --eval-episodes "$eval_episodes" \
  --final-eval-episodes "$final_eval_episodes" \
  --comparison-dimension algorithm \
  --reward-type native \
  --reward-behavior monitor_only \
  --curriculum-name disabled \
  --rulebook-config selection

FINAL_TABLE="$ANALYSIS_ROOT/$run_profile/comparisons/algorithm/metadrive__native__monitor_only__disabled__selection/tables/final_evaluation.csv"
echo
echo "Final table:"
echo "  $FINAL_TABLE"
echo
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$FINAL_TABLE"
else
  cat "$FINAL_TABLE"
fi
EOF
}

main() {
  [[ $# -ge 1 ]] || {
    usage
    exit 1
  }

  local subcommand="$1"
  shift

  case "$subcommand" in
    run|run-parallel|cleanup|launch|analyze) ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown subcommand '$subcommand' (expected run, run-parallel, cleanup, launch or analyze)"
      ;;
  esac

  parse_args "$@"
  validate_common
  collect_algorithms

  case "$subcommand" in
    run)
      log "mode=run tag=$tag algorithms=$algorithms_csv seeds=${seed_start}..${seed_end} poll=${poll_seconds}s"
      run_all
      ;;
    run-parallel)
      log "mode=run-parallel tag=$tag algorithms=$algorithms_csv seeds=${seed_start}..${seed_end} poll=${poll_seconds}s"
      run_all_parallel
      ;;
    cleanup)
      log "mode=cleanup tag=$tag algorithms=$algorithms_csv"
      cleanup_sessions
      ;;
    launch)
      log "mode=launch tag=$tag algorithms=$algorithms_csv seeds=${seed_start}..${seed_end}"
      launch_runs
      ;;
    analyze)
      log "mode=analyze tag=$tag algorithms=$algorithms_csv seeds=${seed_start}..${seed_end}"
      analyze_runs
      ;;
  esac
}

main "$@"
