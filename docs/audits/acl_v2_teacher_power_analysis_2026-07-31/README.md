# ACL v2.0 Teacher — Synthetic Power And Dynamics Analysis

- Date: 2026-07-31
- Status: `EVIDENCE`; not a specification and not authoritative
- Subject: `docs/specifications/automatic_curriculum_learning_v2.0_specification_UNDER_REVIEW.md`
- Purpose: answer, before freezing `v2.0`, three questions that were being argued from theory —
  (1) does the ordinal signal really have a noise-independent neutral expectation, (2) is the
  Holm correction the reason the curriculum might stay uniform, (3) what does the exact
  permutation deadband cost.
- Artifacts: `teacher_dynamics_simulation.py` (self-contained, numpy only, deterministic seeds),
  `simulation_output.txt` (captured run of that script).

Reproduce with:

```bash
python docs/audits/acl_v2_teacher_power_analysis_2026-07-31/teacher_dynamics_simulation.py
```

No simulator, no learner, no training. Runtime a few minutes on CPU.

---

## 0. What is simulated, and what is therefore *not* established

The script implements the proposed teacher exactly — episodic keys, `2H` sliding windows, the
Vargha–Delaney statistic `G`, the exact conditional permutation p-value, the Goldilocks gate, the
signed feedback `A = 0.5 + D(G − 0.5)`, the EMA score, and the temperature/eta-floor softmax — and
drives it with **synthetic arms whose learning curves are prescribed**.

It therefore establishes properties of the *estimator and the control loop*. It establishes
**nothing** about how much a real policy improves on a real arm. That quantity is the free
parameter of the whole analysis and is defined in §2.

## 1. The exact permutation DP is correct and cheap

The dynamic program over tie groups was validated against exhaustive enumeration of all
`C(20,10) = 184 756` splits, on both tie-free and heavily tied inputs. All six trials match to
`1e-12` (`simulation_output.txt` §0).

Cost, `H = 10`, int64 DP:

| variant | ms/call |
|---|---:|
| cold cache, tie-free (worst case) | **0.059** |
| warm cache | **0.019** |
| four dimensions per Generate commit | **0.24** |

A first, naive implementation using arbitrary-precision integers in an object-dtype array cost
`4.2 ms/call`, i.e. `~17 ms` per commit. The optimised version is ~70× faster. **`LIM-208`'s
concern about the permutation cost is real for a naive implementation and disappears with the
int64 DP.** The implementation must use the validated construction, not a rewritten one.

## 2. The free parameter: SNR

Define, for one arm and one dimension,

```
SNR = ( mean episodic key of the older window − mean of the recent window ) / sd
```

i.e. the policy improvement across the `H`-episode gap between the two window centres, expressed
in units of the between-episode (between-scenario) standard deviation *inside that arm*. The true
Vargha–Delaney value is `A = Phi(SNR / sqrt(2))`.

Detection rate of the deadband, `H = 10`, 4000 repetitions per row:

| SNR | true A | mean G | detect @0.05 (no Holm) | detect @0.0125 (Holm, 4 dims) |
|---:|---:|---:|---:|---:|
| 0.00 | 0.500 | 0.497 | 0.049 | 0.014 |
| 0.25 | 0.570 | 0.571 | 0.072 | 0.021 |
| 0.50 | 0.638 | 0.641 | 0.168 | 0.064 |
| 0.75 | 0.702 | 0.701 | 0.314 | 0.147 |
| 1.00 | 0.760 | 0.760 | 0.511 | 0.302 |
| 1.25 | 0.812 | 0.812 | 0.708 | 0.495 |
| 1.50 | 0.856 | 0.853 | 0.850 | 0.669 |
| 2.00 | 0.921 | 0.920 | 0.981 | 0.922 |
| 3.00 | 0.983 | 0.983 | 1.000 | 1.000 |

`mean G` tracks `true A` to within Monte-Carlo error at every level, so `G` is an unbiased
estimator of the quantity it is supposed to estimate.

## 3. `E[G] = 0.5` is independent of the noise level — measured

This is the property that replaces `LIM-002`, the residual defect of the whole `v1`–`v1.3`
prediction-error family. Two windows drawn from the *same* distribution, with the distribution's
scale varied over 250×:

| noise scale | mean G | sd G | false-fire @0.05 | false-fire @0.0125 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.4989 | 0.131 | 0.045 | 0.008 |
| 1.0 | 0.4987 | 0.134 | 0.043 | 0.014 |
| 5.0 | 0.5021 | 0.131 | 0.050 | 0.013 |
| 25.0 | 0.5008 | 0.134 | 0.043 | 0.010 |

No trend in `mean G`, no trend in the false-fire rate. Under `v1.3`'s `mean(max(delta,0))` the
corresponding quantity is proportional to `sigma(delta)` and would grow monotonically down this
column. **This is the empirical core of the redesign and should become `AC-201`.**

## 4. Closed-loop dynamics: the mechanism works, and the noisy arm gets nothing

Six arms. `A0…A3` improve in sequence over disjoint stretches; `A4_noisy` is **stationary with 4×
the between-episode noise of the others** — the synthetic analogue of `A4_vru`; `A5_hard` is
stationary and hard. 9000 post-calibration Generate episodes.

At `SNR = 2.0`, without Holm (`simulation_output.txt` §B):

| episode | A0 | A1 | A2 | A3 | A4_noisy | A5_hard |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 |
| 1500 | 0.153 | **0.242** | 0.152 | 0.148 | 0.152 | 0.152 |
| 2250 | 0.150 | 0.150 | **0.245** | 0.142 | 0.150 | 0.162 |
| 3000 | 0.151 | 0.151 | 0.151 | **0.246** | 0.151 | 0.151 |
| 6000 | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 |

The intended dynamic is reproduced: an arm's probability rises while it improves, hands over to the
next arm, and the whole system returns to uniform once every arm plateaus.

**Mean feedback `A` per arm, all configurations:**

| arm | SNR 0.75 | SNR 1.25 | SNR 2.0 |
|---|---:|---:|---:|
| A0…A3 (improving) | 0.506–0.514 | 0.515–0.530 | 0.524–0.547 |
| **A4_noisy (stationary, 4× noise)** | **0.499** | **0.499** | **0.498** |
| A5_hard (stationary) | 0.499 | 0.499 | 0.499 |

`A4_noisy` receives exactly neutral mean feedback at every signal level, and ends with the *lowest*
Generate count of all six arms (1436–1491 against 1500–1600). Under `v1.3` this arm is the one that
`FIND-006` observed holding the highest score. **The bias is removed, not merely reduced.**

## 5. Holm is not the deciding factor — SNR is

This contradicts the framing that Holm is one of two blocking problems.

Peak arm probability reached during the arm's own learning phase:

| SNR | no Holm | with Holm | uniform |
|---:|---:|---:|---:|
| 0.75 | 0.21–0.24 | 0.21–0.23 | 0.167 |
| 1.25 | 0.24–0.27 | 0.23–0.26 | 0.167 |
| 2.00 | 0.28–0.29 | 0.27–0.29 | 0.167 |

At `SNR >= 1.25` **both configurations work**, and at `SNR = 0.75` **neither** produces a sustained
departure from uniform. Holm changes the fire rate by roughly 2× (e.g. `0.144 → 0.095` on A0 at
SNR 2.0) and the mean feedback marginally; it does not decide whether the curriculum functions.

**What does justify removing Holm** is the cost side, not the benefit side: the extra false fires it
suppresses are sign-symmetric and cost nothing in bias. Measured on `A4_noisy`, the false-fire rate
is `0.032–0.035` without Holm against `0.006–0.007` with it, yet the mean feedback is `0.498–0.500`
in **both** cases. Propagated through the EMA this is `sd(q) ~ 0.02`, i.e. `sd(p_i) ~ 0.006` around
`1/6`. Removing Holm therefore buys ~2× sensitivity for a negligible variance cost, and the teacher
makes no inferential claim that would require family-wise control.

Recommendation: remove Holm, for that reason. Do not claim it was blocking the curriculum.

## 6. Softmax amplitude is intentionally bounded

Even a fully detected, sustained improvement moves an arm to `p ~ 0.25–0.29` against uniform
`0.167`. With `tau = 0.5` and `eta = 0.2` the curriculum cannot monopolise training. Given
`FIND-006`/`FIND-007`, moving from over-concentration to a bounded curriculum is the intended risk
posture, but it means `v2.0` is a *moderately* adaptive teacher by construction.

## 7. The open question this analysis could not answer

**Nothing in this repository measures the real SNR.** It is not derivable from theory, and the
diagnostic-run logs were not present on the analysis machine (`outputs/` is gitignored and empty in
every checkout). From §2:

- `SNR >= 1.25` → the teacher functions as designed;
- `SNR ~ 0.75` → it stays near uniform regardless of Holm, `H = 10` is too small, and `H = 15–20`
  would be required;
- `SNR <= 0.5` → no window-based ordinal teacher at this budget will work.

### Required measurement, before freezing `H`

From any completed ACL run's per-episode records, grouped by arm, for each of `C_1`, `C_2`, `C_3`,
and `route_completion`:

1. fit the within-arm temporal trend across the run;
2. take `delta` = change in the fitted trend across `H = 10` Generate episodes of that arm;
3. take `sd` = residual standard deviation around the trend, within the arm;
4. report `SNR = delta / sd` per arm and per dimension.

This requires no new training run. It is the last quantity needed to decide `H`, and it should be
recorded here as a follow-up section rather than assumed.

## 8. The SNR was then measured on real runs — see §9

§7 above was written before the completed-run logs were located. They were found at
`/scratch/e.respino/thesis-metadrive/outputs/`, and the measurement was carried out. The result
changes the conclusion of this audit and is reported in §9–§11.

---

## 9. Measured on real runs: SNR ≈ 0, because the policy is not learning to drive

Artifacts: `replay_teacher_on_real_runs.py`, `teacher_replay_output.txt`,
`learning_curve_check.py`, `learning_curve_output.txt`.

Source: `EXP_thesis_RP_thesis_CUR_scenario_acl_scenarionet_REW_scalar_reward`, seed 0 of each
planner, longest run instance of each. **None of the runs reached the 1.5M-step `thesis` budget**:
the longest is PPO seed 0 at 475 209 steps, then TD3 seed 0 at 350 154 and SAC seed 0 at 325 143.
These runs predate `ADR-032`, which affects arm *selection* but not whether the policy improves.

### 9.1 Replaying the proposed `v2.0` teacher over the real Generate sequences

The proposed statistics — `2H` windows, Vargha–Delaney `G`, the exact permutation deadband, the
gate, the signed feedback — were run over each arm's real committed Generate episodes, in commit
order. TD3 seed 0, 3331 Generate episodes (`teacher_replay_output.txt`):

| dimension | measured SNR range | mean G | fire rate @0.05 | mean A |
|---|---|---|---|---|
| route completion | 0.00 – 0.02 | 0.491 – 0.512 | 0.044 – 0.098 | 0.500 – 0.506 |
| task `(success, rc)` | 0.00 – 0.02 | 0.491 – 0.512 | 0.044 – 0.098 | 0.500 – 0.506 |
| R1 proxy (collision) | 0.01 – 0.02 | 0.500 – 0.502 | 0.000 – 0.028 | 0.496 – 0.500 |
| R3 proxy (out-of-road) | 0.01 – 0.02 | 0.495 – 0.498 | 0.000 – 0.028 | 0.500 – 0.504 |

PPO seed 0 and SAC seed 0 give the same picture (SNR `0.00–0.05`, mean `A` `0.499–0.514`).

Against §2's table, `SNR ≈ 0.01` sits at the *first row* — the stationary row. The fire rates
recovered are the nominal false-positive rate of the deadband, not detections. **The proposed
teacher, driven by real data, would have produced a uniform curriculum for the entire run.**

R2 could not be reconstructed: per-macro-rule episodic costs are not in these logs, which is
itself a gap `v2.0` `REQ-002` would close.

### 9.2 Why: the outcome levels, not the local slopes

A near-zero slope is ambiguous — it can mean "slow but real" or "nothing". Binning each arm's
episodes into eight equal parts across the run separates them (`learning_curve_output.txt`).

TD3 seed 0, all arms pooled, first bin → last bin:

| quantity | b1 | b8 | change |
|---|---:|---:|---:|
| route completion | 0.157 | 0.175 | +0.019 |
| **success rate** | **0.013** | **0.009** | **−0.004** |
| collision rate | 0.219 | 0.183 | −0.036 |
| **out-of-road rate** | **0.723** | **0.797** | **+0.074** |
| **episode reward** | **−80.4** | **−54.8** | **+25.5** |

PPO seed 0 (the best of the nine), all arms pooled:

| quantity | b1 | b8 | change |
|---|---:|---:|---:|
| route completion | 0.230 | 0.325 | +0.095 |
| success rate | 0.014 | 0.075 | +0.061 |
| **collision rate** | **0.248** | **0.333** | **+0.084** |
| **out-of-road rate** | **~0.15** | **~0.42** | **+0.27** |
| **episode reward** | **−233.1** | **−112.5** | **+120.6** |

SAC seed 0 matches PPO's pattern (reward +99.7, driving outcomes flat or worse).

### 9.3 The finding

**In all three algorithms the scalar reward improves substantially while driving competence does
not.** Success rate stays between 0% and 8% for the whole run and does not trend up in TD3 or SAC.
Out-of-road rate *rises* in every case — for PPO it nearly triples. Collision rate rises in PPO and
in TD3's A1/A2.

The optimiser is working: the reward is being maximised, by a large margin, monotonically.
Maximising *this* reward does not produce a policy that drives. This is a reward-specification
result, not an RL result and not a curriculum result.

## 10. Consequences

1. **The ACL was never the blocking problem.** Any learning-progress curriculum — `v2.0`, TSCL,
   Graves — measures improvement. With no improvement to measure, all of them correctly return
   neutral and sample uniformly. `H`, Holm, and the gate definition are all downstream of a signal
   that is not there.
2. **`FIND-007` is not evidence about the curriculum.** The ACL-on/ACL-off comparison
   (out-of-road `0.60` vs `0.525`, collision `0.275` vs `0.10`) compares two configurations
   *neither of which learns to drive*. The differences are dispersion around a non-learning
   baseline, not a curriculum effect. This weakens `FIND-007` further than §15.2 of the `v2.0`
   draft already did.
3. **`FIND-006` keeps its force.** That the `v1.3` teacher preferred the noisiest arm while nothing
   was being learned is exactly the defect: a correct teacher returns neutral on a stationary arm,
   and the replay in §9.1 shows `v2.0` does (mean `A` `0.496–0.514` everywhere).
4. **The `v2.0` design is validated as far as this evidence can validate it** — it is correctly
   silent when there is nothing to detect, and §3–§4 show it responds when there is. It cannot be
   validated *end to end* until a run exists in which the policy improves.
5. **`H` cannot be frozen from this evidence.** Choosing `H` requires an SNR from a run where
   learning occurs. Deferring the choice is the honest position; `H = 10` remains provisional.

## 10bis. The 120k diagnostic set: promoting offroad into R1 is a real effect

Artifacts: `compare_diagnostic_runs.py`, `diagnostic_comparison_output.txt`.
Source: the four `EXP_diag-sac-lite-*_RP_fast` runs, SAC, seed 0, 120 000 steps each, one seed per
configuration. Chunk 1 is dropped everywhere as the initial exploration transient.

Episode-weighted over chunks 2–12, with binomial 95% CIs on the pooled episode count:

| configuration | N | off-road | collision | success | route | ep_len |
|---|---:|---:|---:|---:|---:|---:|
| native reward, ACL off | 486 | 0.239 ±0.038 | 0.210 ±0.036 | 0.027 ±0.014 | 0.193 | 154.5 |
| **scalar + offroad in R1, ACL off** | 511 | **0.192 ±0.034** | 0.256 ±0.038 | **0.065 ±0.021** | **0.273** | 144.0 |
| scalar rulebook, ACL off | 572 | 0.374 ±0.040 | 0.187 ±0.032 | 0.033 ±0.015 | 0.259 | 130.3 |
| scalar rulebook, ACL on | 677 | 0.313 ±0.035 | 0.239 ±0.032 | 0.030 ±0.013 | 0.273 | 108.4 |

**Promoting off-road into R1 roughly halves the off-road rate** (`0.192` against `0.374`), with
clearly non-overlapping intervals, and roughly doubles the success rate (`0.065` against `0.033`,
intervals barely disjoint). It costs collision rate (`0.256` against `0.187`), which is coherent:
a policy that stays on the road spends more time interacting with traffic. On this evidence it is
the best of the four configurations on three of five metrics.

### The improvement is a level shift, not a learning trend

Early (chunks 2–5) against late (chunks 9–12):

| configuration | metric | early | late | trend |
|---|---|---:|---:|---:|
| native reward | off-road | 0.273 | 0.224 | −0.049 |
| | route | 0.157 | 0.245 | **+0.087** |
| | success | 0.016 | 0.044 | **+0.028** |
| scalar + offroad in R1 | off-road | **0.153** | **0.172** | +0.019 |
| | route | 0.262 | 0.257 | −0.005 |
| | success | 0.038 | 0.056 | +0.017 |
| scalar rulebook | off-road | 0.380 | 0.347 | −0.032 |
| | route | 0.278 | 0.237 | **−0.041** |
| | success | 0.032 | 0.026 | −0.007 |
| scalar rulebook, ACL on | off-road | 0.368 | 0.246 | −0.122 |
| | route | 0.260 | 0.289 | +0.029 |
| | success | 0.019 | 0.035 | +0.017 |

Two separable results:

1. **The `r1road` variant is already at `off-road = 0.153` by chunk 2** and stays there. It moves the
   operating point immediately; it does not accelerate learning. It is a better reward
   *specification*, not a fix for the missing learning signal.
2. **The native MetaDrive reward is the only configuration with a consistent improving trend on all
   three outcomes** (route `+0.087`, success `+0.028`, off-road `−0.049`), while the unmodified
   scalar rulebook trends *negative* on route and success. The scalarisation is therefore suspect on
   two independent counts: a worse operating point *and* a weaker or absent learning gradient.

**Evidentiary weight.** One seed per configuration, 120k steps, 486–677 episodes. The off-road
contrast is wide relative to its interval and survives as a level shift across every chunk; the
success contrast is marginal; the trends of `±0.02–0.03` are within noise and only the native-reward
route trend (`+0.087`) is comfortably outside it. None of this is a replacement for the
`EVAL-PROTOCOL` v1.0 seed protocol.

## 11. Recommended priority change

Freezing and implementing `v2.0` now would deliver a curriculum that is provably neutral on the
only data available. The order that follows from this audit is:

1. **Diagnose why reward improves while driving degrades.** §10bis narrows this considerably: the
   scalarisation is implicated on two counts, and the `r1road` variant already shows that the macro-
   rule priority assignment materially changes the operating point. Remaining suspects are the
   aggregation itself (`rulebook_scalarization_v1.0`, `hybrid_rulebook_manager.py`:
   `a**exponent * sigmoid(c*rho) + rho/n`) and the termination configuration — with
   `out_of_road_done: false`, `out_of_route_done: false`, and `on_continuous_line_done: false` in
   `conf/curriculum/scenario_acl.yaml:38`, an episode continues off-road for the remainder of the
   horizon. Both are checkable offline, by reconstructing the scalarised return of an off-road
   episode against one that completes the route, with no training.
2. **Establish a configuration in which a policy measurably learns**, on the fixed evaluation panel,
   with `curriculum=disabled`. §10bis makes `r1road` — or a principled successor to it — the natural
   starting point, and the native reward the reference for what a working learning gradient looks
   like on this task.
3. **Then measure SNR on that run**, freeze `H`, and finish `v2.0`.

Doing (3) before (1) would freeze a parameter against an SNR of zero.

## 12. Findings for the specification

| # | Finding | Specification impact |
|---|---|---|
| 1 | `E[G] = 0.5` invariant over a 250× noise range, measured | becomes `AC-201`; the core claim of §15.5 |
| 2 | Noisy stationary arm gets mean `A = 0.499` and the fewest Generate draws | new acceptance criterion; direct answer to `FIND-006` |
| 3 | Sequential emerge→plateau→handover reproduced at `SNR >= 1.25` | new acceptance criterion on the closed loop |
| 4 | Holm is not decisive; removing it is justified by negligible bias cost, not by unblocking | rewrite `LIM-201`; remove `progress.multiplicity_correction` |
| 5 | Exact DP validated against exhaustive enumeration; `0.059 ms/call` | resolves the permutation half of `LIM-208`; pins the implementation |
| 6 | Peak `p ~ 0.25–0.29` even under full detection | `v2.0` is a bounded teacher; state it in §15.5 |
| 7 | **Measured SNR ≈ 0.01 on every arm and dimension of every completed run** | `H` cannot be frozen; `LIM-201` becomes a blocking open decision |
| 8 | **Reward rises while success stays ~0–8% and out-of-road rises, in all three algorithms** | outside `v2.0`'s scope; blocks its end-to-end validation and reorders the work (§11) |
| 9 | Per-macro-rule episodic costs are absent from current logs | `REQ-002` instrumentation is needed for diagnosis too, not only for the teacher |
