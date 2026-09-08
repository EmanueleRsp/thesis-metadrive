#!/usr/bin/env bash
# Merge gate for thesis-metadrive.
#
# GitHub CI runs portability checks only, so this script is the project's only
# test signal. It runs the owned-code checks on the project machine and writes a
# log that can be cited as the executed evidence AGENTS.md Completion requires.
#
# Usage:
#   scripts/gate.sh                                  full gate
#   scripts/gate.sh tests/test_foo.py -k bar         partial: pytest scope
#
# Any pytest argument, or a narrowed PYTHON_QUALITY_PATHS, marks the run PARTIAL
# so a scoped run cannot be mistaken for a full gate in the log or the summary.
#
# A check that had nothing to inspect is recorded as NOT APPLICABLE, never as ok:
# the verdict must never absorb a check that read nothing.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Compose derives its project name from the directory, so a worktree would look
# for an image that does not exist and try to rebuild it. Pin the name instead.
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-thesis-metadrive}"

DEFAULT_QUALITY_PATHS="src tests scripts"
QUALITY_SPEC="${PYTHON_QUALITY_PATHS:-$DEFAULT_QUALITY_PATHS}"
BASE_REF="${GATE_BASE_REF:-origin/main}"
PYTEST_ARGS=("$@")

# pytest-xdist workers. The suite's cost sits in two ~585 s integration cases
# of one file (the Waymo panel of test_reward_return_ordering_runtime.py), so
# the gate is short only if they run on different workers. Every xdist mode
# hands out contiguous runs of the collection, so adjacent long cases land on
# one worker (measured: 19m50s with --dist load, 19m36s with --dist worksteal,
# both at -n 16, against 28m00s sequential). The fix is in the collection
# order, not the scheduler: tests/conftest.py spreads `integration` items
# evenly through the collection, so no chunk holds two of them, and a worker
# running a long case is not offered more work until it finishes. `--dist
# loadfile` would pin the whole file to one worker and hide order dependence
# between tests instead of exposing it, so it is deliberately not used.
# GATE_WORKERS=1 runs the suite sequentially; the worker count is recorded in
# the log header either way.
GATE_WORKERS="${GATE_WORKERS:-16}"
PYTEST_DIST=()
if [ "$GATE_WORKERS" != "1" ]; then
  PYTEST_DIST=(-n "$GATE_WORKERS")
fi

SCOPE_NOTES=()
if [ ${#PYTEST_ARGS[@]} -gt 0 ]; then
  SCOPE_NOTES+=("pytest args: ${PYTEST_ARGS[*]}")
fi
if [ "$QUALITY_SPEC" != "$DEFAULT_QUALITY_PATHS" ]; then
  SCOPE_NOTES+=("ruff paths: $QUALITY_SPEC")
fi
if [ ${#SCOPE_NOTES[@]} -gt 0 ]; then
  SCOPE_DETAIL="$(printf '%s; ' "${SCOPE_NOTES[@]}")"
  SCOPE="PARTIAL (${SCOPE_DETAIL%; })"
else
  SCOPE="FULL"
fi

fail_precondition() {
  printf 'gate: cannot run — %s\n' "$1" >&2
  printf 'gate: fix on this machine with:\n  %s\n' "$2" >&2
  printf 'gate: this is a local gap, not a repository defect.\n' >&2
  exit 2
}

if ! command -v docker >/dev/null 2>&1; then
  fail_precondition "docker is not on PATH" "install Docker, then re-run"
fi

if [ ! -f .env ]; then
  fail_precondition ".env is missing (it is gitignored, so it is never in a fresh checkout)" \
    "cp .env.example .env and set the host paths, or copy an existing .env from another checkout"
fi

# All three submodules are needed, not just one: metadrive and scenarionet are
# imported as directly as stable-baselines3.
for submodule in metadrive scenarionet stable-baselines3; do
  if [ -z "$(ls -A "third_party/$submodule" 2>/dev/null)" ]; then
    fail_precondition "third_party/$submodule is empty, so the suite cannot import it" \
      "git submodule update --init --recursive"
  fi
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

DIRTY="$(git status --porcelain)"
if [ -n "$DIRTY" ]; then
  DIRTY_COUNT="$(printf '%s\n' "$DIRTY" | wc -l | tr -d ' ')"
  TREE="dirty ($DIRTY_COUNT files)"
else
  DIRTY_COUNT=0
  TREE="clean"
fi

# `git diff --check` with no revision sees only tracked, unstaged changes. It is
# blind to untracked files and inspects nothing at all once the branch's work is
# committed — which is the normal state for a merge gate. Resolve the scope
# explicitly and report each part, so "ok" always means something was read.
UNTRACKED="$(git ls-files -o --exclude-standard)"
if [ -n "$UNTRACKED" ]; then
  UNTRACKED_COUNT="$(printf '%s\n' "$UNTRACKED" | wc -l | tr -d ' ')"
else
  UNTRACKED_COUNT=0
fi

RANGE_BASE=""
RANGE_NOTE=""
if git rev-parse --verify --quiet "$BASE_REF" >/dev/null 2>&1; then
  RANGE_BASE="$(git merge-base HEAD "$BASE_REF" || true)"
  if [ -z "$RANGE_BASE" ]; then
    RANGE_NOTE="no merge base with $BASE_REF"
  elif [ "$RANGE_BASE" = "$(git rev-parse HEAD)" ]; then
    RANGE_BASE=""
    RANGE_NOTE="HEAD is not ahead of $BASE_REF"
  fi
else
  RANGE_NOTE="$BASE_REF is not available (fetch it to widen this check)"
fi

LOG_DIR="${GATE_LOG_DIR:-outputs/gate}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$TIMESTAMP-$COMMIT.log"
LOG_ABS="$REPO_ROOT/$LOG"

read -r -a QUALITY_PATHS <<<"$QUALITY_SPEC"
COMPOSE_RUN=(docker compose run --rm -T dev uv run --no-sync)

{
  printf 'gate run %s\n' "$TIMESTAMP"
  printf 'scope       %s\n' "$SCOPE"
  printf 'commit      %s (%s, tree %s)\n' "$COMMIT" "$BRANCH" "$TREE"
  printf 'checkout    %s\n' "$REPO_ROOT"
  printf 'machine     %s %s\n' "$(hostname)" "$(uname -m)"
  printf 'quality     %s\n' "${QUALITY_PATHS[*]}"
  printf 'workers     %s\n' "$GATE_WORKERS"
  if [ -n "$RANGE_BASE" ]; then
    printf 'whitespace  %s..HEAD, %s pending, %s untracked\n' \
      "${RANGE_BASE:0:7}" "$DIRTY_COUNT" "$UNTRACKED_COUNT"
  else
    printf 'whitespace  no range (%s), %s pending, %s untracked\n' \
      "$RANGE_NOTE" "$DIRTY_COUNT" "$UNTRACKED_COUNT"
  fi
  if [ -n "$DIRTY" ]; then
    printf 'dirty files:\n%s\n' "$DIRTY"
  fi
} | tee "$LOG"

FAILED=()
SKIPPED=()

run_step() {
  local name="$1"
  shift
  {
    printf '\n=== %s ===\n' "$name"
    printf '$ %s\n' "$*"
  } | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then
    printf '=== %s: ok ===\n' "$name" | tee -a "$LOG"
  else
    printf '=== %s: FAILED ===\n' "$name" | tee -a "$LOG"
    FAILED+=("$name")
  fi
}

skip_step() {
  local name="$1"
  local reason="$2"
  printf '\n=== %s: NOT APPLICABLE (%s) ===\n' "$name" "$reason" | tee -a "$LOG"
  SKIPPED+=("$name")
}

# 1. Pending tracked work, staged and unstaged. Bare `git diff --check` would
#    miss the staged half.
if git diff --quiet HEAD; then
  skip_step "whitespace/pending" "no pending tracked changes"
else
  run_step "whitespace/pending" git diff --check HEAD
fi

# 2. Untracked files, which no `git diff <rev>` form can see. --exclude-standard
#    already honours .gitignore, so the evidence logs never appear here.
if [ "$UNTRACKED_COUNT" -eq 0 ]; then
  skip_step "whitespace/untracked" "no untracked files"
else
  ws_untracked_status=0
  {
    printf '\n=== whitespace/untracked ===\n'
    printf '$ git diff --no-index --check /dev/null <each of %s untracked files>\n' "$UNTRACKED_COUNT"
  } | tee -a "$LOG"
  while IFS= read -r untracked_file; do
    [ -f "$untracked_file" ] || continue
    # `--no-index` implies `--exit-code`, so the status is 1 for any two files
    # that differ — which /dev/null and a non-empty file always do. Measured on
    # 2026-09-08: clean file 1, file with trailing whitespace 3, empty file 1.
    # The status therefore cannot distinguish "differs" from "has a whitespace
    # error"; a non-empty report can, and it is also what a reader sees in the
    # log. Do not "fix" this back to testing the exit status.
    ws_report="$(git diff --no-index --check /dev/null "$untracked_file" 2>&1 || true)"
    if [ -n "$ws_report" ]; then
      printf '%s\n' "$ws_report" | tee -a "$LOG"
      ws_untracked_status=1
    fi
  done <<<"$UNTRACKED"
  if [ "$ws_untracked_status" -eq 0 ]; then
    printf '=== whitespace/untracked: ok ===\n' | tee -a "$LOG"
  else
    printf '=== whitespace/untracked: FAILED ===\n' | tee -a "$LOG"
    FAILED+=("whitespace/untracked")
  fi
fi

# 3. The committed range this branch would merge — the part a clean-tree gate
#    would otherwise report as ok without reading anything.
if [ -n "$RANGE_BASE" ]; then
  run_step "whitespace/range" git diff --check "$RANGE_BASE" HEAD
else
  skip_step "whitespace/range" "$RANGE_NOTE"
fi

run_step "ruff" "${COMPOSE_RUN[@]}" ruff check "${QUALITY_PATHS[@]}"
# --durations makes the slowest tests part of the recorded evidence, so "the
# suite is slow" is answerable from any log instead of needing a special run.
run_step "pytest" "${COMPOSE_RUN[@]}" python -m pytest -q --durations=25 \
  "${PYTEST_DIST[@]}" "${PYTEST_ARGS[@]}"

if [ ${#FAILED[@]} -eq 0 ]; then
  VERDICT="PASS"
  if [ ${#SKIPPED[@]} -gt 0 ]; then
    SKIPPED_DETAIL="$(printf '%s, ' "${SKIPPED[@]}")"
    VERDICT="PASS (${#SKIPPED[@]} not applicable: ${SKIPPED_DETAIL%, })"
  fi
else
  VERDICT="FAIL (${FAILED[*]})"
fi

SUMMARY="gate: $VERDICT | $SCOPE | $COMMIT ($BRANCH, tree $TREE) | $TIMESTAMP"
{
  printf '\n%s\n' "$SUMMARY"
  printf 'log: %s\n' "$LOG_ABS"
} | tee -a "$LOG"

# A smoke test is a separate, change-dependent requirement: this gate does not
# run one. See docs/workflows/agent_operations.md for the production-path smoke.
if [ ${#FAILED[@]} -ne 0 ]; then
  exit 1
fi
