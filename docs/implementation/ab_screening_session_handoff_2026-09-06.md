# Session Handoff — A/B Learnability Screening relaunch, 2026-09-06

Working record for the session that relaunches the seed-0 pair of the `AB-LEARN`
screening after both first runs died on 2026-09-06. It is a handoff, not a
contract: the authoritative documents are `AB-LEARN`
(`docs/implementation/reward_learnability_ab_screening_exec_plan.md`),
`RULEBOOK-V5.1`, `SCAL-V1.4`, `EVAL-PROTOCOL` and ADR-078. The previous handoff,
`ab_screening_session_handoff_2026-09-02.md`, still describes the design; this
one describes what changed since and how to run and read the relaunch.

## 1. Where the code is

| ref | state on 2026-09-06 14:33 UTC |
|---|---|
| `main` (local checkout) | `dd17696`, clean tree, **15 commits ahead of `origin/main`, not pushed** |
| `origin/main` | `95a6ed0`, stale |
| `origin/adr078-on-main`, `origin/worktree-reward-ordering-baseline` | pushed backups of what `main` now contains |

`main` at `dd17696` contains, in order: the documentation of the first launch,
`C8` (provenance identity), `C9` (decomposition coverage budget, the defect that
killed arm A), `GEOM-ABORT` (a geometry failure ends the episode, not the run),
ADR-078 (SAC and TD3: batch 512, update-to-data 0.5, learning-potential batch
off), the reward-ordering test, the constant-action baseline, and `DEC-AB-007`.

**Launch from the main checkout at `dd17696` or later, and push `main` first.**
Yesterday's runs died on a defect whose fix was applied to the live tree while
they ran; the training workers had already imported the old module and never saw
it (`open_items` `C9`, `C10`). The lesson is procedural: the tree a run imports
must be the tree that was tested, and it must not be edited while the run is
alive. Any code change during the runs goes in a separate worktree.

## 2. What happened to the first seed-0 pair

| arm | died | cause | evidence kept |
|---|---|---|---|
| A | 2026-09-06 10:42:31 UTC, chunk 8 (175k of 350k) | `C9`: `ValueError: Constrained decomposition does not cover the input polygon` in training worker slot 18 | 7 of 14 evaluations per panel, `SCREEN-LEARNABILITY-A-NATIVE-01/sac_sb3/seed_0/20260905_214425/` |
| B (second run) | 2026-09-06 09:47:44 UTC, at the 50k evaluation | CUDA out of memory while the evaluation worker loaded the checkpoint on the shared GPU | 1 evaluation per panel, `SCREEN-LEARNABILITY-B-RULEBOOK-01/sac_sb3/seed_0/20260906_063938/` |

Neither is reusable for the analysis: `REQ-AB-003` relaunches on the same seed
and the code has changed. Arm A's seven points are retained as the evidence
behind `C11`.

## 3. The finding that changed arm A's role (`C11`, `DEC-AB-007`)

Arm A did not learn to drive; it learned to stop. On `validation_pg` 853 of
1 050 episodes were stationary; on both panels stationary episodes scored about
0 and moving episodes -9 to -40 under the native reward. Mechanism: MetaDrive's
`on_lane_line_penalty` (-1 per step on a continuous line or on boundary contact)
is applied after `no_negative_reward` clamps the dense term at 0, and ADR-058's
physical-exit termination never ends such an episode, so the penalty is unbounded
(-198.9 over 200 steps in the worst Waymo episode) while standing still in lane
scores exactly 0. Arm A's Waymo route completion was mostly the logged initial
speed (5.1 m/s mean; 0 on every PG record) being braked away.

**Decision `DEC-AB-007` (user, 2026-09-06): arm A is relaunched unchanged and
relabelled** as the descriptive baseline "MetaDrive native reward under the
thesis episode contract". It is not a control. Consequences for reading:

- `H1` (learnability) and `H3` (no inverted incentive) are read on **arm B alone**,
  exactly as pre-registered in `AB-LEARN` §7.2-7.3.
- `H2` is a **descriptive** comparison of B against the arm-A baseline and against
  the constant-action floors below, not a comparability test.
- The §7.3 reading "A also flat => learner problem" is **not available**; do not use it.
- Arm B's reward is the frozen contract under test and is **not** revisited on
  arm-A evidence. If `H1` fails on B, arm A's failure enters the diagnosis then.

## 4. Instruments added on 2026-09-06

**Return-ordering test**, `tests/test_reward_return_ordering_runtime.py`
(`-m integration`, ~21 min). Replays the logged expert in the live runtime on the
first scenario of each validation panel and asserts `still < partial < full` for
both arm rewards. Measured: native Waymo 9.52 < 198.71 < 386.40, PG 0.01 < 43.99
< 97.95; scalarized Rulebook Waymo 4.60 < 176.67 < 347.36, PG -2.72 < 37.29 <
86.43. **Both rewards pass**: neither has an inverted incentive at this level.
The test is a necessary condition; arm A's pathology lives on the trained
policy's trajectories, not in this ordering. It relies on the diagnostic
`agent_policy: replay_ego_policy` (`src/thesis_rl/envs/policies/replay_ego.py`)
and the env key `replay_ego_stop_fraction`.

**Constant-action baseline**, `scripts/evaluate_constant_action_baseline.py`.
Evaluates `brake`, `coast` or `random` through the frozen validation panels with
the run CSV schemas and prints step-based progress with an ETA. Floors already
measured at `seed=0` on the full panels, preset `sac_b_rulebook` (so each row
carries the native reward as `env_reward` and the scalarized one as `reward`),
in `outputs/BASELINE-CONSTANT-ACTION-01/sac_sb3/seed_0/20260906_1128{34,36}/`:

| policy | panel | native mean return | scalar mean return | route completion | collision |
|---|---|---|---|---|---|
| brake | Waymo | 3.33 | -1.06 | 0.041 | 0.080 (hit while stopped) |
| brake | PG | 0.006 | -4.90 | 0.00002 | 0 |
| random | Waymo | -36.37 | -37.00 | 0.201 | 0.135 |
| random | PG | 1.18 | -3.16 | 0.011 | 0 |

How to use them: an arm sitting at the brake floor has not learned; an arm below
the random floor has learned to do worse than chance. On Waymo both rewards put
random far below brake because a random driver collides in 13.5 % of episodes,
so expect **both** arms to pass through a low-motion phase early; whether B
leaves it is what the slope over 14 evaluations decides.

## 5. Resource picture, 2026-09-06 14:33 UTC

| | state |
|---|---|
| GPU | 86 970 / 97 871 MiB, 100 % util, **~11 GB free**; other tenants at 26 GB and 8.7 GB |
| CPU | 72 cores, load average 5.6 |
| RAM | 395 GB available |

One run needs on the order of 10-15 GB of GPU memory including its evaluation
workers, which load a second copy of the checkpoint. **Launch arm B first**; launch
arm A when free memory exceeds ~25 GB. Two runs in parallel with 11 GB free is
how B died the second time. ADR-078 is a projection (~5.2 fps, `medium` ~19 h),
not a measurement: the first `step_timing.csv` rows of the relaunch replace it.

## 6. Launch protocol

From the main checkout, after `git push origin main`, one `tmux` session per
arm, teeing to a log (`run_profile=medium`, 350k steps, evaluation every 25k on
`validation_waymo_empirical` and `validation_pg`):

```bash
tmux new-session -d -s ab_b_seed0 \
  "docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
     uv run --no-sync python -m thesis_rl.cli.train \
     --config-name presets/learnability/sac_b_rulebook seed=0 \
   2>&1 | tee /tmp/ab_b_seed0_run3.log"

# when GPU free memory > ~25 GB
tmux new-session -d -s ab_a_seed0 \
  "docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
     uv run --no-sync python -m thesis_rl.cli.train \
     --config-name presets/learnability/sac_a_native seed=0 \
   2>&1 | tee /tmp/ab_a_seed0_run2.log"
```

`seed=0` is required (`conf/config.yaml` defaults to 42). Before launching,
`--cfg job --resolve` on each preset should show `planner.sac.batch_size=512`,
`update_to_data_ratio=0.5`, `reward.behavior` `monitor_only` / `scalar_reward`,
and the only differences between the two resolved configs confined to the reward
group (`AC-AB-002`).

## 7. Monitoring rules, learned the hard way

1. **Liveness is `"event": "run_failed"` in `logs/events.jsonl`**, not the tmux
   session, not the container, not the last line of `train.log` (`C10`). A chunk
   that takes twice its predecessor is "possibly dead" before it is "probably
   slow": check `run_failed` and `errors.log` first, GPU contention second.
2. Metrics come from `csv/evals.csv` and `csv/eval_episodes.csv`, never from the
   log. Report the two panels **separately** (`open_items` `D5`), for each new
   `eval_type=intermediate` row: `global_step`, `route_completion`,
   `success_rate`, `collision_rate`, `out_of_road_rate`, `mean_reward`, plus
   `mean_scalar_rule_reward` for B; and put the brake and random floors from §4
   in the same table.
3. Slope, not level (`H1`): no verdict before four or five evaluation points;
   short trajectories have misled twice already.
4. `H3` on B: `mean_scalar_rule_reward` and `route_completion` must move together
   across evaluations. Their moving apart is the stop condition of §7.3.
5. Watch for the standstill signature on B: `route_completion` near 0 with
   `collision_rate` 0 and identical episode lengths across evaluations. Expected
   early; a problem if it persists past the middle of the budget.
6. Do not edit `src/` or `conf/` in the main checkout while a run is alive.
7. If a run dies: read `run_failed`, keep its directory, record it in the plan,
   relaunch on the same seed only after the cause is understood (`REQ-AB-003`).
8. `runtime_scenario_data_abort.jsonl` grows on ordinary data conditions (unknown
   signal states) and is not a failure; GEOM-ABORT quarantines geometry failures
   the same way and counts them toward a cap; a rising count is worth reporting.

## 8. Open

- **Seed-1 pair**: a separate user decision (`DEC-AB-004`), not taken.
- **Stall and off-route truncation**: proposed and **deferred** until B's data
  show whether hopeless episodes are frequent (user, 2026-09-06). They would be a
  new truncation convention and need an ADR.
- **Push of `main`**: the user pushes; sessions do not push `main`.
- **Baseline ETA**: available only after the first completed episode of a panel,
  and PG steps run about ten times faster than Waymo, so the first panel's rate
  does not predict the second.
- `D8`/`D9` (TD3 and PPO under ADR-078) are recorded in `open_items` by the
  ADR-078 session and are outside this screening.

## 9. Files touched on 2026-09-06 by the reward-ordering session

| path | action |
|---|---|
| `src/thesis_rl/envs/policies/replay_ego.py`, `__init__.py` | added — diagnostic logged-ego replay with mid-episode stop |
| `src/thesis_rl/envs/factory.py` | modified — `replay_ego_policy` accepted by the policy resolver |
| `src/thesis_rl/envs/thesis_scenario_env.py` | modified — `replay_ego_stop_fraction` declared in `default_config` |
| `tests/test_reward_return_ordering_runtime.py` | added — return-ordering integration test plus resolver unit test |
| `scripts/evaluate_constant_action_baseline.py` | added — constant-action floors with step progress |
| `docs/open_items.md` | modified — `C11` |
| `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | modified — `DEC-AB-007`, §11 entries, §14 rows, status |
| this file | added |

## 10. Prompt for the relaunch session

Paste as the first message of a new session opened in the main checkout.

```text
Sei nella checkout principale di thesis-metadrive, su `main` a dd17696 o
successivo. Leggi prima, per intero e in quest'ordine:
`docs/implementation/ab_screening_session_handoff_2026-09-06.md`,
`docs/implementation/reward_learnability_ab_screening_exec_plan.md` (§6, §7,
§10 M3, §11), `docs/open_items.md` (C9, C10, C11), `AGENTS.md`.

Compito: rilanciare la coppia seed 0 dello screening AB-LEARN e monitorarla
fino alla fine, secondo l'handoff §6-§7.

1. Verifica che `main` locale sia pushato (`git status`, `git log origin/main..main`);
   se non lo è, fermati e chiedimelo: il push di main lo faccio io.
2. Verifica con `--cfg job --resolve` che i due preset compongano ADR-078
   (batch 512, update_to_data_ratio 0.5) e che differiscano solo nel gruppo
   reward. Riporta il diff.
3. Misura la GPU. Lancia il braccio B in tmux (`ab_b_seed0`) con il comando
   dell'handoff. Lancia A (`ab_a_seed0`) solo quando la memoria libera supera
   25 GB, e dimmelo quando lo fai.
4. Arma un monitor su `logs/events.jsonl` di ciascuna run per `run_failed` e per
   ogni `evaluation_finished`, e uno sulla GPU. La liveness è `run_failed`, non
   tmux.
5. A ogni nuova valutazione riporta la tabella cumulativa per pannello,
   separando Waymo e PG, con le colonne dell'handoff §7.2 e i pavimenti brake e
   random di §4 nella stessa tabella. Leggi H1 e H3 solo su B; H2 solo in modo
   descrittivo; nessun verdetto prima di 4-5 punti. Segnalami pattern, non
   solo conclusioni.
6. Non modificare `src/` o `conf/` nella checkout principale mentre le run
   girano; qualunque modifica va in un worktree. Non rilanciare una run morta
   senza avermi riportato la causa.
7. Comunica in italiano; codice, log e documenti in inglese.
```
