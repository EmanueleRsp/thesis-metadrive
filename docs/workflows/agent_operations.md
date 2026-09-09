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

The same steps as the gate minus the nine `integration` tests. Measured on
2026-09-08 with 16 workers: **about 1m35s against the gate's 10m01s** (1838
passed, 5 skipped in 94–95 s over three runs, against 1855 passed and 5 skipped;
run sequentially the same two took 3m55s and 27m59s). The point of the short one is that
it is cheap enough to run on *every* change, which removes the need to guess which
subset covers a change; guessing wrong is how defects reach `main`. It is
`PARTIAL` by construction and never a merge gate.

**Two cases are the whole cost, and it is not the episode.**
`test_reward_return_ordering_on_validation_panels` is parametrized over arm and
panel; its two Waymo cases take **585 s each** and its two PG cases **41 s each**
(sequentially, 628 s for `b_rulebook` and 621 s for `a_native`, 1249 s of the full
run's 1679 s). The other `integration` tests come to about 48 s together. A
cProfile of one Waymo case shows env construction at 16–17 s per behaviour and the
step loop at the rest: `RoutePolyline.project` is a pure-Python scan over every
segment of a centerline, `drivable_surface_for_ego` calls it on every lane of the
scenario (~180 on Waymo, a handful on PG) only to read the lane's height, and that
is invoked about six times per step on identical inputs — about 1 500 projections
per step, 67 % of the case. This is production code that also runs in every
training worker; it is recorded as `F8` in `docs/open_items.md` with the two
semantics-preserving changes that would remove it. Shortening the episodes is not
an option: the return *is* the episode.

**How the parallel run is kept short.** Both gate targets run under
`pytest-xdist` with `GATE_WORKERS` workers (default 16, `make gate GATE_WORKERS=1`
for a sequential run; the count is in the log header). Worker count alone did not
help: at `-n 16` the full suite took **19m50s with `--dist load` and 19m36s with
`--dist worksteal`**, and in both logs the two Waymo cases sum to the whole run.
Reading xdist's schedulers explains it — every mode hands out *contiguous* runs of
the collection (`load` in chunks, `worksteal` as an even split, and it never
steals the last queued item behind a running test), so adjacent long cases land
on one worker and run back to back. The fix is in `tests/conftest.py`: a
collection hook spreads `integration` items evenly through the collection (stride
~206 items, against a maximum chunk of 57 at 16 workers), so no chunk holds two
of them, and xdist only offers a worker more work when it completes a test. With
that the gate is **10m01s**, and the four long cases alone on four workers take
9m53s, so the remaining floor is one Waymo case (`F8`). `--dist loadfile` is
deliberately not used: it would pin a whole file to one worker and hide
order-dependent tests instead of exposing them; the four known ones are `C43`.

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

4. **The first full run after `git submodule update` can fail three tests, and the
   second passes.** Observed 2026-09-09 on a docs-only branch: with a freshly
   cloned `third_party/metadrive`, `make gate` reported
   `test_pg_validation.py::test_bundled_export_validation_can_be_read`,
   `test_scenario_catalog_build.py::test_waymo_loader_deduplicates_reprocessed_batch_uid`
   and `test_scenarionet_smoke.py::test_bundled_waymo_fixture_accepts_random_policy`
   as failures. All three read the bundled Waymo assets, all three **pass when run
   serially**, and re-running `make gate` unchanged gave `PASS | FULL`,
   1909 passed. It is a first-run race between the sixteen `pytest-xdist` workers
   over state the assets generate on first read, which the main checkout already
   has. **A local gap, not a repository defect** — but do not read the first
   verdict as one, and do not chase it: run the gate again.

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
