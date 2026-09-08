# Agent Operations

Operational recipes for any agent session working on this repository, whichever
assistant or model is driving it. Policy lives in `AGENTS.md`; this file holds
the commands and the traps behind them.

## Realigning with the remote

```
git fetch origin --prune

# is there anything to integrate?
git log --oneline HEAD..origin/main      # what arrived that this checkout lacks
git branch -r --no-merged origin/main    # what is in flight and not yet merged

# does what changed invalidate the plan?
git diff --stat HEAD...origin/main       # which files moved since the work started
```

Run this at the start of a session that will touch code and before each new block
of changes, as `AGENTS.md` requires. If nothing moved, the check cost seconds. If
`origin/main` advanced outside the area being worked on, integrate and continue.
If it touched the same area, stop and restate the plan — that is the outcome the
check exists for, and the one most easily skipped, because integrating and
pressing on always looks like the shorter path.

## The merge gate

```
make gate
```

It runs the whitespace checks, Ruff over `PYTHON_QUALITY_PATHS`, and the test suite
inside the container, writes an evidence log to
`outputs/gate/<timestamp>-<commit>.log`, and prints a summary line to cite:

```
gate: PASS | FULL | 58fc9d0 (branch-name, tree clean) | 20260908T130211Z
```

The script pins `COMPOSE_PROJECT_NAME`, so it also works from a worktree where a
bare `make test` does not. When the machine lacks Docker, `.env`, or any of the
three submodules, it refuses to run and prints the fix — a local gap, not a
repository defect.

**Scope.** Passing pytest arguments (`make gate GATE_ARGS="tests/test_module.py -k
case"`) or narrowing `PYTHON_QUALITY_PATHS` marks the run `PARTIAL` in the header
and the summary line, so a scoped run cannot be cited as a gate.

**A check that read nothing is never `ok`.** It is recorded as
`NOT APPLICABLE (<reason>)` and named in the verdict, because a `PASS` that
absorbs an empty check is worse than a missing check: it turns "not inspected"
into recorded evidence. The whitespace check is in three parts for that reason —
bare `git diff --check` sees only tracked, unstaged changes, so it is blind to
untracked files and inspects nothing at all on a clean tree, which is the state a
merge gate normally runs in. The parts are pending tracked work
(`git diff --check HEAD`), untracked files, and the committed range against
`origin/main` (override with `GATE_BASE_REF`). The header states which scope each
resolved to.

**Test durations are part of the evidence.** The pytest step runs with
`--durations=25`, so every log answers "which tests make the suite slow" without a
special run.

The gate deliberately does not run a smoke test: that is a separate,
change-dependent requirement, covered by the production-path smoke below.

## The working-loop check

```
make check
```

The same steps as the gate minus the seven `integration` tests. Measured on
2026-09-08: **3m55s against the full suite's 27m59s** (1838 passed, 5 skipped, 7
deselected, against 1845 passed and 5 skipped). The point of the short one is that
it is cheap enough to run on *every* change, which removes the need to guess which
subset covers a change; guessing wrong is how defects reach `main`. It is
`PARTIAL` by construction and never a merge gate.

**Two tests are 74% of the suite.**
`test_reward_return_ordering_on_validation_panels` costs 628s for `b_rulebook` and
621s for `a_native` — 1249s of the full run's 1679s. The other five `integration`
tests come to about 48s together, and the two PG-generator ones do not reach the
top 25 at all (under 5s each). So "the seven slow tests" is the wrong picture:
five of the seven are cheap, and the cost is concentrated in one parametrized
test. Each of its two cases builds a fresh env per behaviour and runs a complete
episode, over two panels and three behaviours — six env constructions and six full
episodes per case, twelve in all. The return *is* the episode, so there is nothing
to shorten inside them.

`pytest-xdist` was measured and rejected: `-n 16` on the residual suite bought
only 84 seconds (2m30s against 3m55s), because what is left is many small tests
where starting sixteen processes that import torch and MetaDrive eats most of the
gain. Not worth a new dependency. The parallel run did surface one order-dependent
test — it failed under `-n 16` and passed under `--dist loadfile` — which is a
defect in its own right, tracked in `docs/open_items.md`, not a reason to adopt
xdist.

**Why the five slowest unmarked tests stay unmarked.** In the short run the top
five durations are all in `tests/test_scenarionet_vectorized_integration.py`
(43s, 40s, 20s, 20s, 19s) and carry no `integration` marker, so they account for
143s of its 235s. Marking them would take the short run to about 1m30s and would
remove exactly the vectorized-provider and frozen-catalog coverage where two
defects previously hid for weeks. The 90 seconds are deliberately spent. After
those five the distribution is flat — 3.5s, then 1-2s — so there is no other
concentration of cost to remove.

## Running checks from a git worktree

A fresh worktree lacks three things the main checkout has:

1. **Submodules.** `third_party/metadrive`, `third_party/scenarionet` and
   `third_party/stable-baselines3` exist but are empty, so importing
   `stable_baselines3` fails. Run `git submodule update --init --recursive`.
2. **`.env`.** It is gitignored, so a worktree only gets `.env.example`. Copy the
   main checkout's file (`cp <main-checkout>/.env .env`); Compose reads
   `HOST_UID` and the data and outputs paths from it.
3. **The Compose project name.** Always pass `-p thesis-metadrive`:

   ```
   docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync ...
   ```

   Without `-p`, Compose derives the project name from the directory, looks for
   an image that does not exist and tries to rebuild it, which fails on
   `uv sync`. The virtual environment is baked into the image at `/opt/venv`;
   the host `.venv` does not have SB3.

With those in place a focused suite of about 100 tests takes roughly 20 s.

## Heavy jobs run on the GPU

Smokes, training runs and evaluations use the GPU overlay and `device=cuda`:

```
docker compose -p thesis-metadrive -f compose.yaml -f compose.gpu.yaml run --rm -T dev \
  uv run --no-sync python -m thesis_rl.cli.train ... device=cuda
```

The GPU is shared with other users, but it is available for this project's jobs:
do not pass `device=cpu` or drop the overlay out of caution. Keep the CPU for
unit tests.

**Timing on a busy GPU.** End-to-end timings taken while another run holds the
GPU are worthless: one encoder-forward comparison swung by ±1.7 ms and produced a
negative delta for the slower variant. Measure the isolated operation instead and
take the minimum over a few hundred repeats, then multiply by a forward count
derived from the configuration (`share_features_extractor`,
`update_to_data_ratio`, `n_envs`) rather than timing the whole job.

## `make smoke` does not cover the production path

`conf/presets/test/smoke_train.yaml` overrides `obs: lidar_state`,
`agent/planner/encoder: none`, `agent/planner/algorithm: td3_sb3`,
`reward: monitor_only` and `curriculum: disabled`, and does not enable the
vectorized path. A passing `make smoke` therefore says nothing about the
configuration the thesis actually trains: the semantic observation builder, the
LQ encoder, the scalarized reward, the vectorized subprocess path and the ACL
are all unexercised. Two defects survived in exactly that blind spot — every
`obs=semantic_v3` run unstartable for a month, and a silently halved
compliance-trace window.

Before trusting a change that touches observation, reward, vectorization or
curriculum, run the production-path smoke as well:

```
docker compose -p thesis-metadrive -f compose.yaml -f compose.gpu.yaml run --rm -T dev \
  uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train \
  obs=semantic_v3 agent/planner/encoder=lq_v3 reward=scalar_reward \
  env.vectorized.enabled=true env.vectorized.num_envs=2 device=cuda \
  analysis.experiment_group=<tag>
```

It runs two chunks of 1000 steps, so it also crosses a chunk boundary. Add
`curriculum=scenario_acl_scenarionet` to reach the ACL driver.

## Long jobs survive the session

Anything expected to take more than a couple of minutes goes into a dedicated,
named tmux session that tees its output to a log file:

```bash
tmux new-session -d -s <name> "<cmd> 2>&1 | tee <log-dir>/<name>.log"
```

Read the log incrementally with `tail` or `grep`, and kill the session when the
job is done. Never end the command with a buffering filter such as `| tail -N`:
`tail` emits nothing until stdin closes, so the output does not exist for anyone
until the job ends. The job then outlives the agent session and an SSH
disconnect, the user can attach and watch it without asking, and a named session
is recoverable where a background PID is not. Always say which session name and
log path to attach to.

## Pre-fix evidence for a defect fix

`docs/open_items.md` requires, for every defect fix, the *executed* evidence that
the regression test failed before the fix; a structural argument is explicitly
second best. Three ways, cheapest first:

1. **Write the test before the fix.** Run it, capture the assertion message, then
   fix. This works whenever the test does not import a symbol that the fix
   introduces.
2. **A throwaway script reproducing the pre-fix construct.** When the fix adds a
   new function, the test cannot fail before it for an honest reason — it would
   only raise `ImportError`. Reconstruct the old code's data inline instead and
   run it through the unchanged reader, then delete the script.
3. **A temporary file swap**, when neither of the above applies: copy the fixed
   file to a scratch directory outside the repository, restore the committed
   version in place with `git show HEAD:<path> > <path>`, run the test, then copy
   the fixed file back. This changes no git state and is fully reversible.

Record in the register which of the three produced the evidence.
