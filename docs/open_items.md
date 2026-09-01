# Open Items

Single register of known-but-unfixed work. It exists because these were
previously scattered across "Notes" cells of `docs/project_index.md`, where they
were easy to lose: several have been open for weeks without being surfaced.

**Scope.** Items recorded in repository documentation, plus items opened by a
decision that has not yet been implemented. It is **not** an audit of the source
tree — an unlisted problem is not thereby absent.

**Convention.** `Blocking` means some approved contract is not yet met by the
code. `Deferred` means a decision was taken to postpone. `Open decision` means
the work is waiting on a choice the user has not yet made. Tick an item by
moving it to *Closed* with the date and the change that closed it. **Identifiers
are never recycled**: a closed `D2` stays `D2`, and the next new decision takes
the next free number.

Last reviewed: 2026-08-20. `D1` was briefly closed on 2026-08-20 and
**reopened the same day**: the value chosen presumed an episodic threshold, and
the mechanism is not chosen yet.

---

## Blocking — approved contracts not yet met

| # | item | source | notes |
|---|---|---|---|
| B1 | **ADR-071 not implemented** — `RB51` milestone `M7`, gated on `DEC-RB51-002`. R1 at-fault classification: at-fault → termination + R1 cost; not-at-fault → truncation, no cost. | ADR-071, approved 2026-08-12 | Part of the approved `RULEBOOK-V5.1` contract. Every classifier input already exists at the R1 evaluation point except lane containment for the lateral branch. |
| B2 | **`RULEBOOK-V5.1` is not implemented in `src/thesis_rl/`.** The six-level hierarchy, `SCAL-V1.4` and L6 exist only in the measurement instrument. | `RULEBOOK-V5.1`, approved 2026-08-14, amended 2026-08-20 | ExecPlan written 2026-08-20: `implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md` (`RB51`), `AWAITING_DECISIONS`. Production is at `SCAL-V1.1` with four margins; the gap includes deleting `wrong_way`, demoting `rss`, and adding `speed_limit`, which does not exist. |
| B3 | **`speed_limit` requires amending `OBS-V1.3` and `OBS-LIDAR-V2.0`**, and intentionally breaks checkpoint compatibility. | `RULEBOOK-V5.0` §7, carried by v5.1 §6 | Compounds with C1 below. |

## Open decisions — waiting on the user

| # | item | opened | notes |
|---|---|---|---|
| D1 | **`τ₄` for the thresholded-lexicographic arm.** `τ₄ = 0` is wrong: exact L4 ties have probability zero, so at 0 the comparison stops at L4 on every pair and **L5 and L6 decide nothing**. A positive `τ₄` is load-bearing — it is the minimum progress a lane relaxation must buy to be worth taking. | 2026-08-20 | **Sequenced after the rulebook is frozen**, and not fixable before the algorithm is chosen: the object a threshold applies to differs by mechanism. *Absolute Thresholding* (Gábor et al. 1998) thresholds **Q-values**; *Absolute Slacking* (Li & Czarnecki 2019) thresholds a slack from the **state's own optimum**, which is already relative and therefore sidesteps the mission-length problem for free; a policy-gradient construction (Tercan & Prabhu 2024) compares **expected returns**, which is episodic. A rule phrased on the completion fraction presumes the third. Analysis kept: missions span 13–247 m, so no single absolute constant serves both ends under an absolute mechanism; `1 − success_route_completion_threshold = 0.05` is an already-declared constant available as a starting point **if** the episodic reading applies. Settled regardless: the **channel is not renormalized** (§4.1.1's rejection stands, scalar calibration untouched). |
| D4 | **Restart of the three production runs.** `sac-0` and `ppo-0` crashed on checkpoint reload and are stopped; `td3-0` remains at risk. | 2026-07-26 (ADR-026) | Now compounded: `γ = 1` (ADR-075) and `speed_limit` both break checkpoint compatibility. |
| D5 | **Arm/source confounding in the frozen train split**: A0–A2 are all PG, A3–A5 all Waymo, so arm difficulty and data source cannot be separated. | 2026-07-27 (ADR-031) | Same *class* of defect as differing `γ` across arms, which ADR-075 refused for exactly this reason. Open for three weeks. |
| D6 | **Should `EVAL-PROTOCOL v1.3` record the feasibility ceiling** from `D3`, so reported success rates are read against it rather than against 100 %? | 2026-08-20 | Recorded in `RULEBOOK-V5.1` limitation 14; the evaluation protocol is a separate approved document. |

## Deferred — postponed on purpose

| # | item | source | notes |
|---|---|---|---|
| F1 | **M5, solid-line penetration deadband**, not implemented, gated on its own measurement. | Rulebook v4.10 ExecPlan | |
| F2 | **Scalarizer binarisation (F0)** deferred to a separate ExecPlan. | Rulebook v4.10 ExecPlan | |
| F3 | **Calibrated stochastic tracking** deferred; ideal semantic tracking remains an explicit baseline limitation. | `OBS-V1.2` | |
| F4 | **`MOTORCYCLIST` collision class** deferred. | ADR-027 | |
| F5 | **ACL implementation stage v3 not started, v4 deferred**; runtime learner smoke, live ScenarioNet/Rulebook wiring and resume validation pending before `VERIFIED`. | ACL ExecPlan | |
| F6 | **PG coverage measurement** for the rulebook (applicability rates and geometric sanity, not Test A). | `RULEBOOK-V5.1` limitation 5 | A dispatch change, not new capability: `build_pg_static_adapter_result` already exists and already reads `polygon`. |

## Known defects — identified, not fixed

| # | item | source | notes |
|---|---|---|---|
| C1 | **RSS standstill-exit transient** identified but not fixed, pending frequency/duration measurement. | Rulebook v4.10 ExecPlan | |
| C2 | **`SIGNAL` never selected at runtime on 209/828 (25.2 %)** of Waymo scenarios carrying `has_route_traffic_light=true`, after the route-membership fix reduced it from 57.1 %. | ADR-051 | A traffic-light rule that never fires on a quarter of the scenarios that have traffic lights. |
| C3 | **M0 control-drop diagnostic counters** are collected but aggregated into no cross-episode report table. | Rulebook v4.10 ExecPlan | |
| C4 | **Mission artifact regeneration pending** a provisioned `pyarrow`. | `DRIVING-MISSION-V1.1.1` | |

## Verification debt — checks not run

| # | item | notes |
|---|---|---|
| V1 | **`make smoke` not run** since ADR-058's episode-contract change, and not run after `γ = 1`. The discount change alters value-target scale, which is what an end-to-end smoke test would surface. | `make smoke` |
| V2 | **Transition replay**: source-backed learner smoke and checkpoint/resume remain pending. | `TRANSITION-REPLAY` ExecPlan |

---

## Closed

| # | item | closed | how |
|---|---|---|---|
| D2 | **L6 rewards speed, and PG admits no speed limit.** | 2026-08-20 | Decided: **accept**. PG blocks are synthetic, so there is no traffic law to encode and MetaDrive's design speeds are simulator parameters, not norms; inventing a PG limit would be the unmotivated calibrated constant this project rejects elsewhere. `offroad` at L3 constrains reactively, and it is a difference between *sources*, not between *arms*, so it does not confound the comparison. Recorded as `RULEBOOK-V5.1` limitation 13, with a new §7 diagnostic `mean_ego_speed_by_source` so the divergence is measured rather than assumed away. |
| D3 | **Whether to raise MetaDrive's `max_speed_km_h = 80`.** | 2026-08-20 | Decided: **do not raise**; declare the ceiling. The reason is quantitative: the episode ceiling is `a · distance / longest single step`, and the cap *is* the longest step, so it sits in the denominator of what a mission can be worth. A 110 km/h cap — the least that covers all nine records — costs **−27 %** of every mission's worth while every per-step penalty stays unchanged, and `λ₄` cannot compensate because `λ₄ + 0.1·(λ₅+λ₆) < a`. Completing those records *would* be legal (posted limits 65–70 mph), so the obstruction is the vehicle, not the rulebook. Recorded as `RULEBOOK-V5.1` limitation 14 with the full table. |

*(move further items here with the date and the change that closed them)*
