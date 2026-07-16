# Curriculum Learning Specification for MetaDrive Autonomous Driving RL

**Version:** not declared in the source document  
**Status:** user-approved authoritative specification  
**Approval confirmed:** 2026-07-16  
**Scope:** scenario-level automatic curriculum learning for MetaDrive-based autonomous driving experiments  
**Main objective:** define a complete, implementable curriculum architecture compatible with scalar, lexicographic, and distributional RL agents.

---

## 0. High-level design decision

The curriculum operates at **scenario level**, while the RL algorithm operates at **transition level**.

```text
Curriculum Learning:
    selects, stores, replays, mutates scenarios

RL algorithm:
    collects transitions and updates actor/critic/value/distributional networks
```

This separation is mandatory.

The curriculum does **not** replace the RL replay buffer. It adds a separate **scenario buffer** used to select which scenarios the agent should experience.

---

## 1. Final curriculum strategy

The selected strategy is:

```text
Scenario-description-based automatic curriculum learning
with generator-arm selection, prioritized scenario replay,
and controlled scenario mutation.
```

The resulting workflow is:

```text
MetaDrive procedural generation
        ↓
MAB over generator arms
        ↓
scenario rollout
        ↓
export / store ScenarioDescription
        ↓
compute rule-aware usefulness U_s
        ↓
scenario buffer Λ
        ↓
curriculum iteration:
    GENERATE new scenario
    or
    EXPLOIT scenario from Λ
        ├── REPLAY parent scenario
        └── MUTATE parent scenario
```

The method combines these ideas:

| Component | Origin / inspiration | Use in this project |
|---|---|---|
| Procedural generation | MetaDrive / PGDrive | main source of scenarios |
| MAB over curricula | Peng-style reward-driven ACL | adaptive selection of generator profiles |
| Scenario buffer | PLR / Abouelazm / ACCEL-style ACL | store useful scenarios |
| Learning potential | PLR / Abouelazm / ACCEL | estimate scenario usefulness for learning |
| Staleness | PLR-style replay | avoid replaying only the same high-ranked scenarios |
| Scenario mutation | ACCEL / Abouelazm | generate useful variants of informative scenarios |
| Failure / near-violation focus | failure-driven curriculum / safety-critical AD | prioritize safety-relevant cases |
| ScenarioDescription | MetaDrive / ScenarioNet | canonical format for saving, replaying, mutating scenarios |
| Rulebook criticality | rulebook / lexicographic specification | preserve safety hierarchy before scalarization |

---

## 2. Scenario source

### 2.1 Core source: MetaDrive procedural generation

The core training curriculum uses:

```text
MetaDriveEnv + procedural generation
```

The curriculum selects a **generator arm**. Each generator arm defines a distribution over MetaDrive procedural generation parameters.

```text
MAB selects arm A_i
        ↓
sample θ ~ A_i
        ↓
create MetaDriveEnv(θ)
        ↓
rollout current agent
```

This is the preferred core source because it provides:

- controllable scenario distributions;
- clean comparison between RL algorithms;
- explicit difficulty and diversity knobs;
- easy train/test split via seeds/maps;
- no dependency on large real-world datasets;
- a focused thesis scope centered on RL, lexicographic objectives, and distributional RL.

### 2.2 Alternative / extension: ScenarioNet / Waymo

ScenarioNet / Waymo is **not** the core training source for the first implementation.

It is kept as a possible extension or evaluation setting.

With ScenarioNet/Waymo the curriculum would become:

```text
ScenarioNet / Waymo dataset
        ↓
scenario descriptor / metadata / features
        ↓
scenario selection
        ↓
ScenarioEnv
        ↓
buffer + replay + mutation
```

In that case the MAB would select **scenario classes or clusters**, not procedural generator parameters.

Possible ScenarioNet arms would be:

```text
low traffic scenarios
high traffic scenarios
intersections
lane-change / merge scenarios
pedestrian / cyclist scenarios
traffic-light scenarios
safety-critical / near-conflict scenarios
```

This is not used in the core implementation because it would introduce additional complexity:

- dataset selection;
- dataset preparation/conversion;
- scenario feature extraction;
- clustering or class assignment;
- dataset imbalance;
- real-world scenario mining;
- compatibility verification with custom observations, rulebook, reward, and training loop.

---

## 3. Main objects and notation

| Symbol / object | Meaning |
|---|---|
| `G_i` or `A_i` | generator arm / generation profile |
| `K` | number of generator arms |
| `theta` | sampled MetaDrive procedural config |
| `s` | scenario |
| `Lambda` / `Λ` | scenario buffer |
| `R` | standard RL replay buffer of transitions |
| `w_i` | main MAB weight for arm `i` |
| `w_target_i` | target / temporary MAB weight for arm `i` |
| `p_i(t)` | probability of selecting arm `i` |
| `U_s` | final scenario usefulness |
| `C_rule_s` | rulebook-based scenario criticality |
| `LP_alg_s` | algorithm-dependent learning potential |
| `U_tilde_s` | normalized scalar usefulness for MAB |
| `D` | exploitation probability |
| `1-D` | generation probability |
| `N` | scenario buffer capacity |
| `B_min` | minimum buffer size before enabling exploit/mutation |
| `N_m` | number of mutation attempts/children during exploitation |
| `omega` | replay sampling weight between usefulness and staleness |
| `beta` | rank-priority exponent |
| `c_s` | staleness of scenario `s` |
| `W_recent` | recent evaluated scenarios window |

---

## 4. Initialization

```python
generator_arms = [A0, A1, A2, A3, A4, A5, A6]
K = 7

Lambda = ScenarioBuffer(capacity=1000)
R = RLReplayBuffer(...)

w = np.ones(K)
w_target = np.ones(K)

B_min = 100
```

Final default parameters:

```yaml
curriculum:
  buffer_capacity: 1000
  warmup_buffer_size: 100

  exploit_probability: 0.8
  generate_probability: 0.2

  mutation_per_exploit: 2

  replay_usefulness_weight: 0.7
  replay_staleness_weight: 0.3
  rank_priority_exponent: 1.0
```

---

## 5. Curriculum iteration

At every curriculum iteration:

```python
if len(Lambda) < B_min:
    mode = "GENERATE"
else:
    if Bernoulli(1 - D):
        mode = "GENERATE"
    else:
        mode = "EXPLOIT"
```

where:

```yaml
D: 0.8          # probability of EXPLOIT
1-D: 0.2        # probability of GENERATE
```

Important naming:

```text
D is NOT generation probability.
D is exploitation probability.
```

---

## 6. GENERATE mode

### 6.1 Select generator arm

The MAB selects a generator arm.

The selection probability is:

```math
p_i(t) =
(1 - eta) * exp(w_i(t)) / sum_j exp(w_j(t))
+ eta / K
```

where:

- `w_i(t)` is the MAB weight of arm `i`;
- `eta` is the exploration coefficient;
- `K` is the number of arms.

Final parameters:

```yaml
mab:
  num_arms: 7
  eta: 0.2
  alpha: 0.005
  initial_weight: 1.0
  use_target_mab: true
  target_sync_interval: 5
  weight_clip_min: -5.0
  weight_clip_max: 5.0
```

### 6.2 Sample MetaDrive config

```python
i = sample_categorical(p)
theta = sample_from_generator_arm(A_i)
env = MetaDriveEnv(theta)
```

### 6.3 Rollout

Run the current agent in the sampled environment:

```python
trajectory = rollout(agent, env)
```

The trajectory must contain:

```text
observations
actions
rewards
reward vectors / rule margins if available
dones / truncations
infos
critic/value/distributional losses or enough data to compute them
```

### 6.4 Export ScenarioDescription

After the episode, export/store the scenario in ScenarioDescription format.

The stored scenario is the canonical object used for:

```text
buffer insertion
replay
mutation
reproducibility
diagnostics
```

If the ScenarioDescription pipeline is not immediately available during implementation, a temporary fallback is allowed:

```text
store env_config + seed for replay
store ScenarioDescription later when export is available
```

However, the target architecture remains ScenarioDescription-based.

### 6.5 Compute usefulness

Compute:

```math
U_s = (C_rule_s, LP_alg_s)
```

See Section 11.

### 6.6 Insert into scenario buffer

Insert scenario into `Lambda` if useful enough.

See Section 14.

### 6.7 Update MAB

The generated scenario is used as feedback for the arm `A_i`.

First compute normalized usefulness:

```math
U_tilde_s in [0, 1]
```

Then importance-weighted feedback:

```math
feedback_i(t) = U_tilde_s / p_i(t)
```

Update the target weight:

```math
w_target_i(t+1) = clip(w_target_i(t) + alpha_MAB * feedback_i(t), -5, 5)
```

Every `N_MAB = 5` generated episodes:

```math
w_i <- w_target_i
```

If unstable, set:

```yaml
target_sync_interval: 1
```

which synchronizes after every generated episode.

---

## 7. EXPLOIT mode

In EXPLOIT mode:

```text
1. sample parent scenario from Lambda
2. replay parent scenario
3. generate mutated children
4. validate children
5. rollout valid children
6. compute U_child
7. insert useful children into Lambda
```

Pseudocode:

```python
parent = sample_replay_scenario(Lambda)

trajectory_parent = rollout(agent, ScenarioEnv(parent))
U_parent_new = compute_usefulness(trajectory_parent)
update_record(parent, U_parent_new)

for _ in range(N_m):
    child_sd, mutation_report = mutate_scenario_description(parent.sd, mutation_cfg)

    is_valid, validation_report = validate_mutated_scenario(
        child_sd,
        parent_sd=parent.sd,
        cfg=validation_cfg,
    )

    if not is_valid:
        continue

    child_record = save_child_scenario(child_sd, parent, mutation_report)
    trajectory_child = rollout(agent, ScenarioEnv(child_record))
    U_child = compute_usefulness(trajectory_child)

    maybe_insert_or_replace(Lambda, child_record, U_child)
```

Final parameters:

```yaml
exploit:
  probability: 0.8
  replay_parent: true
  mutate_inside_exploit: true
  mutation_children_per_parent: 2
```

---

## 8. Generator arms

The curriculum uses `K = 7` generator arms.

Each arm is a distribution over MetaDrive procedural-generation parameters.

General note:

```text
The exact validity of block_sequence strings must be verified against the installed MetaDrive version.
```

### 8.1 A0 — `broad_random`

Purpose: maintain broad exploration.

```python
A0_broad_random = {
    "map_mode": Categorical({
        "block_num": 0.70,
        "block_sequence": 0.30,
    }),

    "block_num": UniformDiscrete({3, 4, 5, 6, 7, 8}),

    "block_sequence": Categorical({
        "S": 0.10,
        "C": 0.10,
        "X": 0.10,
        "O": 0.10,
        "SC": 0.15,
        "SX": 0.15,
        "OC": 0.10,
        "XCO": 0.10,
        "rCR": 0.05,
        "SCrRX": 0.05,
    }),

    "traffic_density": Uniform(0.00, 0.50),

    "traffic_mode": Categorical({
        "trigger": 0.40,
        "respawn": 0.30,
        "hybrid": 0.30,
    }),

    "random_lane_width": Bernoulli(0.50),
    "random_lane_num": Bernoulli(0.50),

    "accident_prob": Mixture({
        Constant(0.0): 0.45,
        Uniform(0.05, 0.20): 0.30,
        Uniform(0.20, 0.40): 0.20,
        Uniform(0.40, 0.60): 0.05,
    }),

    "static_traffic_object": True,
}
```

### 8.2 A1 — `simple_low_risk`

Purpose: easy scenarios, useful during early training and as a control profile.

```python
A1_simple_low_risk = {
    "map_mode": Categorical({
        "block_num": 0.80,
        "block_sequence": 0.20,
    }),

    "block_num": UniformDiscrete({3, 4}),

    "block_sequence": Categorical({
        "S": 0.35,
        "C": 0.25,
        "SC": 0.25,
        "SX": 0.15,
    }),

    "traffic_density": Uniform(0.00, 0.10),

    "traffic_mode": Categorical({
        "trigger": 0.70,
        "respawn": 0.10,
        "hybrid": 0.20,
    }),

    "random_lane_width": False,
    "random_lane_num": False,

    "accident_prob": Constant(0.0),
    "static_traffic_object": True,
}
```

### 8.3 A2 — `traffic_heavy`

Purpose: many vehicles and dense traffic.

```python
A2_traffic_heavy = {
    "map_mode": Categorical({
        "block_num": 0.75,
        "block_sequence": 0.25,
    }),

    "block_num": UniformDiscrete({4, 5, 6}),

    "block_sequence": Categorical({
        "SC": 0.25,
        "SX": 0.25,
        "X": 0.20,
        "XCO": 0.15,
        "OC": 0.15,
    }),

    "traffic_density": Uniform(0.25, 0.50),

    "traffic_mode": Categorical({
        "trigger": 0.20,
        "respawn": 0.40,
        "hybrid": 0.40,
    }),

    "random_lane_width": Bernoulli(0.30),
    "random_lane_num": Bernoulli(0.30),

    "accident_prob": Mixture({
        Constant(0.0): 0.60,
        Uniform(0.05, 0.20): 0.40,
    }),

    "static_traffic_object": True,
}
```

### 8.4 A3 — `obstacle_heavy`

Purpose: static obstacles, cones, barriers, warning objects, accident-like situations.

```python
A3_obstacle_heavy = {
    "map_mode": Categorical({
        "block_num": 0.75,
        "block_sequence": 0.25,
    }),

    "block_num": UniformDiscrete({4, 5, 6}),

    "block_sequence": Categorical({
        "S": 0.15,
        "C": 0.20,
        "SC": 0.20,
        "SX": 0.15,
        "OC": 0.15,
        "XCO": 0.15,
    }),

    "traffic_density": Uniform(0.05, 0.25),

    "traffic_mode": Categorical({
        "trigger": 0.50,
        "respawn": 0.20,
        "hybrid": 0.30,
    }),

    "random_lane_width": Bernoulli(0.25),
    "random_lane_num": Bernoulli(0.25),

    "accident_prob": Uniform(0.20, 0.60),
    "static_traffic_object": True,
}
```

### 8.5 A4 — `complex_topology`

Purpose: geometrically complex maps.

```python
A4_complex_topology = {
    "map_mode": Categorical({
        "block_num": 0.40,
        "block_sequence": 0.60,
    }),

    "block_num": UniformDiscrete({6, 7, 8}),

    "block_sequence": Categorical({
        "X": 0.10,
        "O": 0.10,
        "XCO": 0.25,
        "rCR": 0.20,
        "OC": 0.15,
        "SCrRX": 0.20,
    }),

    "traffic_density": Uniform(0.10, 0.30),

    "traffic_mode": Categorical({
        "trigger": 0.40,
        "respawn": 0.25,
        "hybrid": 0.35,
    }),

    "random_lane_width": Bernoulli(0.30),
    "random_lane_num": Bernoulli(0.30),

    "accident_prob": Mixture({
        Constant(0.0): 0.50,
        Uniform(0.05, 0.25): 0.50,
    }),

    "static_traffic_object": True,
}
```

### 8.6 A5 — `lane_randomized`

Purpose: force variation in lane width and lane number.

```python
A5_lane_randomized = {
    "map_mode": Categorical({
        "block_num": 0.65,
        "block_sequence": 0.35,
    }),

    "block_num": UniformDiscrete({4, 5, 6, 7}),

    "block_sequence": Categorical({
        "SC": 0.20,
        "SX": 0.20,
        "OC": 0.20,
        "XCO": 0.20,
        "rCR": 0.20,
    }),

    "traffic_density": Uniform(0.10, 0.30),

    "traffic_mode": Categorical({
        "trigger": 0.40,
        "respawn": 0.30,
        "hybrid": 0.30,
    }),

    "random_lane_width": True,
    "random_lane_num": True,

    "accident_prob": Mixture({
        Constant(0.0): 0.50,
        Uniform(0.05, 0.25): 0.50,
    }),

    "static_traffic_object": True,
}
```

### 8.7 A6 — `safety_critical_mix`

Purpose: combine topology complexity, traffic, and obstacles.

```python
A6_safety_critical_mix = {
    "map_mode": Categorical({
        "block_num": 0.40,
        "block_sequence": 0.60,
    }),

    "block_num": UniformDiscrete({5, 6, 7, 8}),

    "block_sequence": Categorical({
        "X": 0.10,
        "O": 0.10,
        "XCO": 0.25,
        "rCR": 0.20,
        "OC": 0.15,
        "SCrRX": 0.20,
    }),

    "traffic_density": Uniform(0.20, 0.50),

    "traffic_mode": Categorical({
        "trigger": 0.20,
        "respawn": 0.35,
        "hybrid": 0.45,
    }),

    "random_lane_width": Bernoulli(0.60),
    "random_lane_num": Bernoulli(0.60),

    "accident_prob": Uniform(0.20, 0.60),
    "static_traffic_object": True,
}
```

---

## 9. MetaDrive environment config

The same termination flags must be used for:

```text
GENERATE
REPLAY
MUTATE
```

Otherwise `U_s` is not comparable across scenarios.

Proposed config:

```yaml
metadrive_env:
  horizon: 1000
  truncate_as_terminate: false

  out_of_route_done: false
  out_of_road_done: false

  on_continuous_line_done: false
  on_broken_line_done: false

  crash_vehicle_done: true
  crash_object_done: true
  crash_human_done: true
```

Rationale:

```text
Crashes terminate the episode.
Off-road, out-of-route, lane-line violations do not terminate.
They are instead captured by the rulebook / reward vector / margins.
```

This allows the curriculum to observe informative violations instead of ending every episode too early.

TODO:

```text
Verify exact flag names against the installed MetaDrive version.
```

---

## 10. ScenarioEnv replay config

When loading stored or mutated scenarios:

```python
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.env_input_policy import EnvInputPolicy

env = ScenarioEnv({
    "data_directory": curriculum_dataset_dir,
    "num_scenarios": 1,
    "start_scenario_index": scenario_index,

    "agent_policy": EnvInputPolicy,
    "reactive_traffic": True,

    "horizon": 1000,
    "truncate_as_terminate": False,

    "out_of_route_done": False,
    "out_of_road_done": False,

    "on_continuous_line_done": False,
    "on_broken_line_done": False,

    "crash_vehicle_done": True,
    "crash_object_done": True,
    "crash_human_done": True,

    "log_level": 50,
})
```

Important:

```text
Do not use ReplayEgoCarPolicy for training.
The ego must be controlled by the current RL agent.
```

Use:

```text
EnvInputPolicy
```

because the agent provides actions.

Use:

```text
reactive_traffic=True
```

so that non-ego traffic can react when ego deviates from logged trajectories.

TODO:

```text
Verify exact ScenarioEnv options and policy class names against the installed MetaDrive version.
```

Current repository note:

```text
In the MetaDrive version currently used by this repository/container,
ScenarioEnv replay does not accept all the flags listed above.

Observed as unsupported at constructor time:
- out_of_road_done
- on_continuous_line_done
- on_broken_line_done

The current implementation therefore filters replay-time ScenarioEnv config
down to the subset of keys actually accepted by the installed version.

Implication:
- this does not block scenario replay or the curriculum architecture;
- it does mean the specification here is a target design, not yet a 1:1
  runtime contract for every listed ScenarioEnv flag;
- this compatibility gap should be revisited explicitly when upgrading
  MetaDrive or when finalizing replay/termination semantics.
```

Additional current repository note:

```text
In the current replay path using ScenarioEnv, some rulebook-enrichment inputs
may be unavailable even when replay itself works correctly.

Observed warnings:
- opposite_carriageway unavailable
- target_region unavailable

Implication:
- this is not a replay crash condition;
- the wrong-way rule may become neutral if opposite_carriageway is missing;
- the goal-progress rule may fall back to target_point-only behavior if
  target_region is missing;
- replay-time rule signals may therefore be semantically weaker than in the
  procedural-generation path.

This should be tracked as a replay/rulebook compatibility gap and revisited
when finalizing replay semantics and rule-aware usefulness.
```

---

## 11. Scenario usefulness

The final usefulness of a scenario is:

```math
U_s = (C_rule_s, LP_alg_s)
```

ordered lexicographically.

### 11.1 Meaning

| Component | Meaning | Depends on |
|---|---|---|
| `C_rule_s` | rulebook-based criticality | raw rulebook margins / reward vector |
| `LP_alg_s` | learning potential | RL algorithm |
| `U_s` | final usefulness | lexicographic pair |

The first component prevents scalarized baselines from hiding safety-critical events.

The second component measures whether the current agent can still learn from the scenario.

### 11.2 Lexicographic order

```math
U_s > U_s'
```

iff:

```math
C_rule_s > C_rule_s'
```

or:

```math
C_rule_s = C_rule_s' and LP_alg_s > LP_alg_s'
```

Therefore:

```text
rulebook criticality dominates learning potential.
```

### 11.3 Rulebook dependency

The exact definition of `C_rule_s` is **not finalized in this curriculum task**.

It depends on the separate rulebook task.

The expected interface is:

```python
def compute_rule_criticality(trajectory, rulebook_cfg) -> C_rule:
    ...
```

The returned value may be:

```text
int in {0, 1, 2, 3}
```

or:

```text
lexicographic vector of violations / near-violations
```

Recommended first implementation after rulebook is finalized:

```text
Crit_s = 3 severe failure
Crit_s = 2 near-violation
Crit_s = 1 safety-relevant
Crit_s = 0 ordinary scenario
```

But the exact rules, margins, thresholds, and order must come from the finalized rulebook.

Open rulebook-dependent items:

```text
- list of rules
- rule ordering
- margin definitions m_t^(k)
- violation definition
- near-violation definition
- thresholds delta_k
- compression into Crit_s if used
- rule_violation_rate
- near_violation_rate
- min_rule_margin
```

---

## 12. Algorithm-dependent learning potential

The learning potential is denoted:

```math
LP_alg_s
```

Do not call this `U_s` directly in implementation, because `U_s` is the full pair:

```math
U_s = (C_rule_s, LP_alg_s)
```

### 12.1 PPO / actor-critic with value function

For PPO or actor-critic methods with value function `V(o)`:

```math
delta_t = r_t + gamma * V(o_{t+1}) - V(o_t)
```

Then compute the GAE-like residual:

```math
A_lambda_t = sum_{k=t}^{T} (gamma * lambda)^(k-t) * delta_k
```

Then:

```math
LP_PPO_s = (1/T) * sum_{t=0}^{T} max(A_lambda_t, 0)
```

This is the positive value loss / positive advantage-style learning potential.

Parameters:

```yaml
ppo_learning_potential:
  gamma: 0.99
  lambda: 0.9
  use_positive_part: true
```

Required inputs:

```text
V(o_t)
V(o_{t+1})
r_t
done flag
```

Here `r_t` is the scalar reward used by the PPO critic.

For scalar baselines:

```math
r_t = r_bar_t = sigma(r_vector_t)
```

where `r_vector_t` is the rulebook reward vector and `sigma` is the scalarization.

### 12.2 TD3 / DDPG-like critic

TD3 has two critics:

```math
Q_1(o,a), Q_2(o,a)
```

Target:

```math
y_t = r_t + gamma * (1-d_t) * min_j Q_target_j(o_{t+1}, pi_target(o_{t+1}))
```

Residual:

```math
delta_TD3_t = y_t - min_j Q_j(o_t, a_t)
```

Learning potential:

```math
LP_TD3_s = (1/T) * sum_{t=0}^{T} abs(delta_TD3_t)
```

Use absolute residual, not positive residual, because the sign of off-policy critic residuals is less interpretable than PPO advantages.

Parameters:

```yaml
td3_learning_potential:
  gamma: 0.99
  residual: absolute
  critic_aggregation: min_q1_q2
```

Required inputs:

```text
Q1, Q2
Q1_target, Q2_target
pi_target
r_t
done flag
```

### 12.3 SAC

SAC uses two critics and an entropy term.

Sample:

```math
a_{t+1} ~ pi(. | o_{t+1})
```

Target:

```math
y_t = r_t + gamma * (1-d_t) * [min_j Q_target_j(o_{t+1}, a_{t+1}) - alpha_ent * log pi(a_{t+1}|o_{t+1})]
```

Residual:

```math
delta_SAC_t = y_t - min_j Q_j(o_t, a_t)
```

Learning potential:

```math
LP_SAC_s = (1/T) * sum_{t=0}^{T} abs(delta_SAC_t)
```

Parameters:

```yaml
sac_learning_potential:
  gamma: 0.99
  residual: absolute
  include_entropy_term: true
  critic_aggregation: min_q1_q2
```

Required inputs:

```text
Q1, Q2
target critics
policy
entropy temperature alpha_ent
log_pi
r_t
done flag
```

### 12.4 Lexicographic RL

If the agent is lexicographic, the reward is vector-valued:

```math
r_vector_t = (r_t^(1), r_t^(2), ..., r_t^(d))
```

and the agent has one value/critic per objective:

```text
V^(1), ..., V^(d)
```

or:

```text
Q^(1), ..., Q^(d)
```

For value-based actor-critic:

```math
delta_t^(k) = r_t^(k) + gamma * V^(k)(o_{t+1}) - V^(k)(o_t)
```

For Q-critic methods:

```math
delta_t^(k) = y_t^(k) - Q^(k)(o_t, a_t)
```

Then:

```math
LP_s^(k) = (1/T) * sum_{t=0}^{T} abs(delta_t^(k))
```

and:

```math
LP_vector_s = (LP_s^(1), ..., LP_s^(d))
```

Do not scalarize this vector.

Rank scenarios lexicographically:

```math
LP_vector_s >_lex LP_vector_s'
```

The full scenario usefulness is still:

```math
U_s = (C_rule_s, LP_vector_s)
```

with `C_rule_s` first.

### 12.5 Distributional RL

For distributional algorithms, do not use a scalar TD-error.

Use the distributional Bellman loss.

For IQN / quantile regression:

```math
LP_dist_s = (1/T) * sum_{t=0}^{T} L_QR_t
```

where `L_QR_t` is the quantile Huber loss between target distribution and predicted distribution.

For C51-like categorical distribution:

```math
LP_dist_s = (1/T) * sum_{t=0}^{T} L_CE_t
```

where `L_CE_t` is cross-entropy or KL between the projected target distribution and the predicted distribution.

Required inputs:

```text
gamma
target return distribution
predicted return distribution
quantiles or atoms
distributional loss
done flag
```

### 12.6 Lexicographic + distributional RL

If the agent is both lexicographic and distributional:

```math
LP_dist_vector_s = (LP_dist_s^1, ..., LP_dist_s^d)
```

where:

```math
LP_dist_s^k = (1/T) * sum_{t=0}^{T} L_dist_t^k
```

Then scenarios are ranked lexicographically using:

```math
U_s = (C_rule_s, LP_dist_vector_s)
```

---

## 13. Normalization for MAB

The buffer and replay do not need absolute scaling of usefulness because they can use ranks.

The MAB needs a scalar feedback:

```math
U_tilde_s in [0, 1]
```

### 13.1 Rank set

Use:

```math
M = |Lambda union W_recent|
```

where:

```text
Lambda = scenario buffer
W_recent = last 100 evaluated scenarios
```

Final parameter:

```yaml
normalization:
  recent_window_size: 100
  rank_set: buffer_plus_recent
```

### 13.2 Scalar usefulness

If `U_s` is scalar after combining `C_rule_s` and `LP_alg_s`, use percentile rank:

```math
U_tilde_s = PercentileRank(U_s)
```

### 13.3 Lexicographic usefulness

If `U_s` is lexicographic, compute:

```math
rank_lex(s)
```

where:

```text
rank = 1 means best / most useful scenario
```

Then:

```math
U_tilde_s = 1 - (rank_lex(s)-1)/(M-1)
```

If `M = 1`:

```math
U_tilde_s = 1
```

Implementation:

```python
def normalize_usefulness_by_rank(record, rank_set):
    M = len(rank_set)
    if M <= 1:
        return 1.0
    rank = compute_rank(record, rank_set)  # rank 1 = best
    return 1.0 - (rank - 1) / (M - 1)
```

---

## 14. Scenario buffer Lambda

The scenario buffer stores scenario records, not RL transitions.

Final parameters:

```yaml
scenario_buffer:
  capacity: 1000
  warmup_size: 100
  replacement: lexicographic_worst
  anti_duplicates: true
  max_children_per_parent: 5
```

### 14.1 Insertion rule

```python
if len(Lambda) < N:
    insert(s)
else:
    worst = lexicographic_argmin(Lambda)
    if U_s > U_worst:
        replace(worst, s)
    else:
        reject(s)
```

### 14.2 Duplicate rejection

Reject if:

```text
same scenario_description_hash
```

Also reject if:

```text
same parent_id
same mutation_type
very similar mutation_params
```

Implementation can start with hash-based duplicate checking only.

### 14.3 No separate priority field

Do not define:

```text
priority_s
```

as a separate metric.

Priority is induced by:

```text
rank(U_s)
```

and converted to probabilities only when sampling.

Allowed cached fields:

```text
rank
U_norm
```

These are not independent metrics.

---

## 15. Replay sampling

When exploiting the buffer, sample scenario `s` using:

```math
P_replay(s) = omega * P_U(s) + (1 - omega) * P_C(s)
```

where:

```math
P_U(s) = rank(s)^(-beta) / sum_x rank(x)^(-beta)
```

and:

```math
P_C(s) = c_s / sum_x c_x
```

with:

```math
c_s = 1 + t - t_last(s)
```

Final parameters:

```yaml
replay_sampling:
  omega: 0.7
  beta: 1.0
  staleness_offset: 1
  rank_one_is_best: true
```

Meaning:

```text
P_U favors highly useful scenarios.
P_C favors scenarios not replayed recently.
omega = 0.7 gives more importance to usefulness than staleness.
```

Implementation:

```python
def compute_replay_probabilities(buffer, t, omega=0.7, beta=1.0):
    ranks = np.array([record.rank for record in buffer], dtype=np.float32)
    p_u = ranks ** (-beta)
    p_u = p_u / p_u.sum()

    staleness = np.array(
        [1 + t - record.last_seen_step for record in buffer],
        dtype=np.float32,
    )
    p_c = staleness / staleness.sum()

    p = omega * p_u + (1 - omega) * p_c
    return p / p.sum()
```

---

## 16. ScenarioRecord schema

Each scenario stored in `Lambda` must have a compact record.

```python
ScenarioRecord = {
    # identity
    "scenario_id": str,
    "source": "generate" | "mutate",
    "parent_id": Optional[str],

    # reproducibility / storage
    "scenario_description_path": str,
    "scenario_description_hash": str,
    "env_config": dict,
    "reset_seed": int,

    # generation
    "generator_arm": Optional[str],

    # mutation
    "mutation_type": Optional[str],
    "mutation_params": Optional[dict],
    "validation_status": str,

    # usefulness
    "C_rule": int | list,       # rulebook-dependent
    "LP_alg": float | list,
    "U": tuple,
    "U_norm": float,
    "rank": int,

    # replay metadata
    "num_seen": int,
    "last_seen_step": int,
    "num_children": int,

    # diagnostics
    "metrics_summary": dict,
}
```

Notes:

```text
C_rule remains rulebook-dependent.
LP_alg is already defined for each algorithm.
U = (C_rule, LP_alg).
U_norm is only for MAB feedback.
rank is a cached ordering field.
```

---

## 17. Diagnostic metrics

Metrics are not used as independent curriculum scores.

They are used for:

```text
logging
debugging
report analysis
explaining why a scenario is useful
```

Minimum metrics:

```python
metrics_summary = {
    "success": bool,
    "route_completion": float,
    "episode_return": float,
    "episode_length": int,

    "collision_vehicle": bool,
    "collision_object": bool,
    "collision_human": bool,
    "out_of_road": bool,
    "out_of_route": bool,
    "stuck": bool,

    "rule_violation_rate": float,
    "near_violation_rate": float,
    "min_rule_margin": float,

    "termination_reason": str,

    "mean_speed": float,
    "num_agents": int,
    "traffic_density": float,
    "generator_config_id": str,
}
```

Rulebook-dependent placeholders:

```text
rule_violation_rate
near_violation_rate
min_rule_margin
```

These must be finalized after the rulebook task.

---

## 18. ScenarioDescription mutation

### 18.1 Core principle

Mutation must modify:

```text
tracks
```

and should not modify:

```text
map_features
metadata structural fields
SDC / ego track
road topology
```

Core rule:

```text
Mutate actors/objects, not the map.
```

### 18.2 Fields allowed for mutation

Allowed:

```text
tracks[object_id]["state"]["position"]
tracks[object_id]["state"]["heading"]
tracks[object_id]["state"]["velocity"]
tracks[object_id]["state"]["valid"]
tracks[object_id]["metadata"], only if needed
metadata["curriculum"], for parent/mutation annotations
```

Not allowed in the core implementation:

```text
map_features
metadata["sdc_id"]
metadata["coordinate"]
metadata["timestep"]
length
version
dynamic_map_states, except future experiments with traffic lights
SDC / ego track
```

### 18.3 Mutation operators

Core mutation operators:

```text
temporal_shift_vehicle
longitudinal_shift_vehicle
remove_actor
shift_static_object
clone_static_object
```

### 18.4 Mutation config

```python
mutation_cfg = {
    "buffer_min_size_for_mutation": 100,
    "max_children_per_parent": 5,
    "max_mutation_attempts": 5,

    "mutation_probs": {
        "temporal_shift_vehicle": 0.40,
        "longitudinal_shift_vehicle": 0.20,
        "remove_actor": 0.15,
        "shift_static_object": 0.15,
        "clone_static_object": 0.10,
    },

    "min_time_shift": -10,
    "max_time_shift": 10,

    "min_long_shift": -3.0,
    "max_long_shift": 3.0,

    "min_dist_from_ego": 10.0,
    "min_dist_between_objects": 3.0,
    "max_lane_distance": 4.0,

    "valid_check": True,
    "min_initial_collision_margin": 0.3,
    "max_speed": 45.0,
    "max_acc": 12.0,
    "max_heading_jump": 1.05,
    "min_sdc_valid_steps": 50,

    "smoke_test_steps": 20,
    "reactive_traffic": True,
}
```

---

## 19. Mutation operators

### 19.1 `temporal_shift_vehicle`

Shift a non-SDC vehicle trajectory forward or backward in time.

```math
track_i(t) <- track_i(t + Delta_t)
```

Purpose:

```text
Change interaction timing without inventing a new trajectory.
Useful for turning ordinary interactions into near-violations.
```

Allowed shift:

```yaml
min_time_shift: -10
max_time_shift: 10
```

Positive shift means delayed actor.

Negative shift means earlier actor.

Pseudo-code:

```python
def temporal_shift_track(track: dict, shift: int) -> dict:
    child = copy.deepcopy(track)
    state = child["state"]

    for key, arr in state.items():
        if not isinstance(arr, np.ndarray):
            continue
        if arr.shape[0] == 0:
            continue

        new_arr = np.zeros_like(arr)

        if shift > 0:
            new_arr[shift:] = arr[:-shift]
            if key == "valid":
                new_arr[:shift] = False
        elif shift < 0:
            k = abs(shift)
            new_arr[:-k] = arr[k:]
            if key == "valid":
                new_arr[-k:] = False
        else:
            new_arr = arr.copy()

        state[key] = new_arr

    return child
```

### 19.2 `longitudinal_shift_vehicle`

Move a non-SDC vehicle slightly along its trajectory direction or lane direction.

Purpose:

```text
Change distance gaps while preserving realistic orientation and topology.
```

Config:

```yaml
min_long_shift: -3.0
max_long_shift: 3.0
```

Must be followed by:

```text
lane proximity check
initial collision check
dynamic plausibility check
```

### 19.3 `remove_actor`

Remove or invalidate a non-SDC actor.

Purpose:

```text
Generate easier variants or isolate causes of failure.
```

Implementation options:

```text
remove track from tracks
or set valid[:] = False
```

Recommended first implementation:

```text
set valid[:] = False
```

because it is less likely to break ScenarioDescription structure.

Do not remove:

```text
SDC / ego
essential map features
all actors
```

### 19.4 `shift_static_object`

Shift a static object slightly.

Purpose:

```text
Vary obstacle position while preserving scenario topology.
```

Must satisfy:

```text
minimum distance from ego
minimum distance from other objects
near-lane / drivable area check
no initial overlap
```

### 19.5 `clone_static_object`

Clone an existing static object to a nearby valid position.

Purpose:

```text
Increase obstacle density using existing object templates.
```

Constraints:

```text
clone only non-SDC static objects
new object_id must be unique
new object must pass lane proximity and collision checks
do not exceed max_children_per_parent
```

---

## 20. ScenarioDescription validation

Validation has two levels:

```text
MetaDrive structural validation
+
custom semantic validation
```

### 20.1 Structural validation

Use MetaDrive utilities where available:

```python
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.scenario import utils as sd_utils

SD.sanity_check(scenario, check_self_type=True, valid_check=True)
SD.is_scenario_file(path)
sd_utils.read_scenario_data(path)
SD.get_number_summary(scenario)
SD.get_object_summary(...)
```

This checks ScenarioDescription formatting and required fields.

If `valid_check=True` is too strict for recorded scenarios, fallback:

```python
SD.sanity_check(scenario, check_self_type=True, valid_check=False)
```

and rely on custom checks.

### 20.2 Validation config

```python
validation_cfg = {
    "valid_check": True,
    "min_collision_margin": 0.3,
    "max_lane_dist": 4.0,
    "dt": 0.1,
    "max_speed": 45.0,
    "max_acc": 12.0,
    "max_heading_jump": 1.05,
    "min_sdc_valid_steps": 50,
    "smoke_test_steps": 20,
}
```

### 20.3 Custom checks

Required custom checks:

```text
check_required_fields
check_track_lengths
check_finite_values
check_sdc
check_valid_mask
check_initial_collisions
check_mutated_objects_near_lanes
check_dynamic_plausibility
check_nontrivial
try_load_with_ScenarioEnv
smoke_test
```

### 20.4 Required fields

```python
def check_required_fields(sd):
    required = [
        "id",
        "version",
        "length",
        "tracks",
        "dynamic_map_states",
        "map_features",
        "metadata",
    ]
    for k in required:
        assert k in sd
```

### 20.5 Track lengths

All temporal arrays must have length:

```math
T = scenario["length"]
```

```python
def check_track_lengths(sd):
    T = sd["length"]
    for obj_id, track in sd["tracks"].items():
        state = track["state"]
        for key, arr in state.items():
            if hasattr(arr, "shape") and len(arr.shape) > 0:
                if arr.shape[0] != T:
                    raise ValueError(
                        f"{obj_id}.{key} has length {arr.shape[0]} != {T}"
                    )
```

### 20.6 Finite values

```python
def check_finite_values(sd):
    for obj_id, track in sd["tracks"].items():
        for key, arr in track["state"].items():
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                if not np.isfinite(arr).all():
                    raise ValueError(f"NaN/Inf in {obj_id}.{key}")
```

### 20.7 SDC present and unchanged

```python
def check_sdc(sd, parent_sd=None):
    sdc_id = sd["metadata"]["sdc_id"]
    assert sdc_id in sd["tracks"]

    if parent_sd is not None:
        parent = parent_sd["tracks"][sdc_id]["state"]
        child = sd["tracks"][sdc_id]["state"]
        assert np.allclose(parent["position"], child["position"])
```

For the curriculum core:

```text
Do not mutate the SDC / ego track.
```

### 20.8 Initial collisions

Simple bounding-circle approximation:

```python
def object_radius(track, t=0):
    state = track["state"]
    length = state.get("length", None)
    width = state.get("width", None)

    if length is None or width is None:
        return 1.5

    return 0.5 * np.sqrt(length[t] ** 2 + width[t] ** 2)


def check_initial_collisions(sd, min_margin=0.3):
    tracks = sd["tracks"]
    ids = list(tracks.keys())

    poses = {}
    radii = {}

    for obj_id in ids:
        tr = tracks[obj_id]
        valid = tr["state"]["valid"]

        if not valid[0]:
            continue

        poses[obj_id] = tr["state"]["position"][0, :2]
        radii[obj_id] = object_radius(tr, 0)

    pose_ids = list(poses.keys())

    for i, a in enumerate(pose_ids):
        for b in pose_ids[i + 1:]:
            d = np.linalg.norm(poses[a] - poses[b])
            if d < radii[a] + radii[b] + min_margin:
                raise ValueError(f"Initial overlap: {a}, {b}")
```

For a later version, replace this with oriented-box intersection.

### 20.9 Lane proximity

Approximate distance to map feature polylines:

```python
def min_distance_to_lane_polylines(xy, map_features):
    best = float("inf")

    for feat in map_features.values():
        if "polyline" not in feat:
            continue

        poly = feat["polyline"][:, :2]
        d = np.linalg.norm(poly - xy[None, :], axis=1).min()
        best = min(best, d)

    return best
```

Rule:

```python
if distance_to_lane > max_lane_dist:
    invalid
```

Default:

```yaml
max_lane_dist: 4.0
```

### 20.10 Dynamic plausibility

```python
def check_dynamic_plausibility(track, cfg):
    state = track["state"]
    pos = state["position"][:, :2]
    valid = state["valid"].astype(bool)
    dt = cfg.get("dt", 0.1)

    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 2:
        return

    p = pos[valid_idx]
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt

    if speed.max() > cfg.get("max_speed", 45.0):
        raise ValueError("Unrealistic speed")

    acc = np.abs(np.diff(speed)) / dt

    if len(acc) > 0 and acc.max() > cfg.get("max_acc", 12.0):
        raise ValueError("Unrealistic acceleration")
```

### 20.11 Nontrivial scenario

```python
def check_nontrivial(sd):
    summary = SD.get_number_summary(sd)

    if summary["num_objects"] < 1:
        raise ValueError("No objects")

    if summary["num_map_features"] < 1:
        raise ValueError("No map features")
```

Optional stricter checks:

```text
at least one object near ego route
at least one dynamic object or obstacle within X meters
```

### 20.12 ScenarioEnv load test

After saving the child scenario into a temporary dataset:

```python
env = ScenarioEnv({
    "data_directory": tmp_dataset_dir,
    "num_scenarios": 1,
    "reactive_traffic": True,
    "agent_policy": EnvInputPolicy,
    "log_level": 50,
})

obs, info = env.reset(seed=0)
```

Then run a short smoke test:

```python
for _ in range(smoke_test_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### 20.13 Final validation function

```python
def validate_mutated_scenario(sd, parent_sd=None, cfg=None):
    cfg = cfg or {}
    report = {"errors": [], "warnings": [], "stats": {}}

    try:
        SD.sanity_check(
            sd,
            check_self_type=True,
            valid_check=cfg.get("valid_check", True),
        )

        check_required_fields(sd)
        check_track_lengths(sd)
        check_finite_values(sd)
        check_sdc(sd, parent_sd=parent_sd)

        check_initial_collisions(
            sd,
            min_margin=cfg.get("min_collision_margin", 0.3),
        )

        check_mutated_objects_near_lanes(sd, cfg)
        check_all_dynamic_tracks_plausible(sd, cfg)
        check_nontrivial(sd)

        report["stats"]["number_summary"] = SD.get_number_summary(sd)

        return True, report

    except Exception as e:
        report["errors"].append(str(e))
        return False, report
```

---

## 21. Technical acceptance tests

These tests must be implemented after the specification is coded.

### 21.1 Procedural generation test

```text
MetaDriveEnv PG rollout
→ collect trajectory
→ compute metrics
→ compute LP_alg
```

### 21.2 ScenarioDescription export test

```text
rollout
→ export ScenarioDescription
→ save to disk
→ SD.is_scenario_file(path)
→ sd_utils.read_scenario_data(path)
→ SD.sanity_check(...)
```

### 21.3 ScenarioEnv reload test

```text
saved ScenarioDescription
→ ScenarioEnv(data_directory=...)
→ reset(seed=index)
→ ego controlled by EnvInputPolicy
→ rollout current agent
```

### 21.4 Reactive traffic test

```text
ScenarioEnv with reactive_traffic=True
→ ego deviates from logged trajectory
→ background traffic should not blindly replay into ego
```

### 21.5 Mutation test

```text
parent ScenarioDescription
→ mutate
→ validate
→ save
→ reload
→ rollout
→ compute U_child
```

### 21.6 Curriculum loop test

```text
warm-up generate until |Lambda| >= 100
then alternate generate/exploit
check MAB updates
check buffer replacement
check replay probabilities
check mutation insertion
```

---

## 22. YAML configuration design

The implementation should be YAML-driven.

Recommended files:

```text
curriculum.yaml
arms.yaml
mab.yaml
mutation.yaml
validation.yaml
scenario_env.yaml
ablation.yaml
```

### 22.1 `curriculum.yaml`

```yaml
curriculum:
  enabled: true

  mode: full_curriculum

  buffer_capacity: 1000
  warmup_buffer_size: 100

  exploit_probability: 0.8
  generate_probability: 0.2

  use_mab: true
  use_scenario_buffer: true
  use_replay: true
  use_mutation: true
  use_staleness: true
  use_rule_criticality: true

  mutation_per_exploit: 2

  recent_window_size: 100
```

### 22.2 `mab.yaml`

```yaml
mab:
  num_arms: 7
  eta: 0.2
  alpha: 0.005

  initial_weight: 1.0

  use_target_mab: true
  target_sync_interval: 5

  weight_clip_min: -5.0
  weight_clip_max: 5.0

  feedback: rank_normalized_usefulness
```

### 22.3 `replay.yaml`

```yaml
replay_sampling:
  omega: 0.7
  beta: 1.0
  staleness_offset: 1
  rank_one_is_best: true
```

### 22.4 `mutation.yaml`

```yaml
mutation:
  buffer_min_size_for_mutation: 100
  max_children_per_parent: 5
  max_mutation_attempts: 5

  mutation_probs:
    temporal_shift_vehicle: 0.40
    longitudinal_shift_vehicle: 0.20
    remove_actor: 0.15
    shift_static_object: 0.15
    clone_static_object: 0.10

  min_time_shift: -10
  max_time_shift: 10

  min_long_shift: -3.0
  max_long_shift: 3.0

  min_dist_from_ego: 10.0
  min_dist_between_objects: 3.0
  max_lane_distance: 4.0

  valid_check: true
  min_initial_collision_margin: 0.3
  max_speed: 45.0
  max_acc: 12.0
  max_heading_jump: 1.05
  min_sdc_valid_steps: 50

  smoke_test_steps: 20
  reactive_traffic: true
```

### 22.5 `scenario_env.yaml`

```yaml
scenario_env:
  horizon: 1000
  truncate_as_terminate: false

  out_of_route_done: false
  out_of_road_done: false

  on_continuous_line_done: false
  on_broken_line_done: false

  crash_vehicle_done: true
  crash_object_done: true
  crash_human_done: true

  reactive_traffic: true
  agent_policy: EnvInputPolicy

  log_level: 50
```

### 22.6 `validation.yaml`

```yaml
validation:
  valid_check: true
  min_collision_margin: 0.3
  max_lane_dist: 4.0
  dt: 0.1
  max_speed: 45.0
  max_acc: 12.0
  max_heading_jump: 1.05
  min_sdc_valid_steps: 50
  smoke_test_steps: 20
```

### 22.7 `ablation.yaml`

```yaml
ablation_modes:
  uniform_pg:
    use_mab: false
    use_scenario_buffer: false
    use_replay: false
    use_mutation: false

  mab_generate_only:
    use_mab: true
    use_scenario_buffer: false
    use_replay: false
    use_mutation: false

  mab_plus_replay:
    use_mab: true
    use_scenario_buffer: true
    use_replay: true
    use_mutation: false

  full_curriculum:
    use_mab: true
    use_scenario_buffer: true
    use_replay: true
    use_mutation: true
```

---

## 23. Ablation strategy

The implementation should allow at least these modes:

```text
uniform_pg
mab_generate_only
mab_plus_replay
full_curriculum
```

### 23.1 `uniform_pg`

Baseline.

```text
No MAB.
No scenario buffer.
No mutation.
MetaDrive PG sampled uniformly or with fixed standard config.
```

### 23.2 `mab_generate_only`

Tests generator-arm adaptation alone.

```text
MAB selects generator arms.
No scenario replay.
No mutation.
```

### 23.3 `mab_plus_replay`

Tests contribution of scenario buffer.

```text
MAB selects generator arms.
Useful scenarios stored in Lambda.
Replay from Lambda using usefulness + staleness.
No mutation.
```

### 23.4 `full_curriculum`

Full method.

```text
MAB + scenario buffer + replay + mutation.
```

---

## 24. Evaluation metrics

For comparing curriculum variants:

```text
success rate
route completion
collision vehicle rate
collision object rate
collision human rate
out-of-road rate
out-of-route rate
rule violation rate
near-violation rate
average scalar return
average vector return if applicable
sample efficiency
generalization on held-out seeds/maps
```

Rulebook-dependent metrics will be finalized after the rulebook task.

---

## 25. Open items

The curriculum architecture is complete except for rulebook-dependent parts.

### 25.1 Open: rulebook definition

Must be completed in the rulebook task:

```text
- final list of rules
- rule order / hierarchy
- reward vector components
- rule margins m_t^(k)
- violation semantics
- near-violation semantics
- thresholds delta_k
- scalarization sigma for scalar baselines
- rulebook logging fields
```

### 25.2 Open: `C_rule_s`

After the rulebook is finalized, define:

```python
def compute_rule_criticality(trajectory, rulebook_cfg):
    ...
```

Potential output:

```text
Crit_s in {0,1,2,3}
```

or:

```text
C_rule_s = (V_s, NV_s)
```

where:

```math
V_s^(k) = max_t max(0, -m_t^(k))
```

and:

```math
NV_s^(k) = max_t [I(0 <= m_t^(k) < delta_k) * (1 - m_t^(k)/delta_k)]
```

### 25.3 Open: rulebook-dependent diagnostics

These fields remain placeholders:

```text
rule_violation_rate
near_violation_rate
min_rule_margin
```

### 25.4 Technical verification TODOs

These are implementation checks, not conceptual open decisions:

```text
- verify block_sequence strings in installed MetaDrive
- verify config flag names in installed MetaDrive
- verify ScenarioDescription export from PG rollout
- verify ScenarioEnv reload
- verify reactive_traffic with ego controlled by current agent
- verify SD.sanity_check strictness with valid_check=True
```

---

## 26. Final implementation checklist

### Core curriculum

```text
[ ] implement generator arms A0–A6
[ ] implement MAB arm selector
[ ] implement target MAB update and sync
[ ] implement MetaDrive config sampler
[ ] implement PG rollout wrapper
[ ] implement ScenarioDescription export/storage
[ ] implement ScenarioRecord
[ ] implement scenario buffer Lambda
[ ] implement rank-based usefulness ordering
[ ] implement replay probability P_replay
[ ] implement mutation operators
[ ] implement mutation validation
[ ] implement ScenarioEnv reload
[ ] implement curriculum loop
```

### Learning potential

```text
[ ] implement LP_PPO
[ ] implement LP_TD3
[ ] implement LP_SAC
[ ] implement LP_lexicographic
[ ] implement LP_distributional
[ ] implement LP_lexicographic_distributional
```

### Rulebook dependency

```text
[ ] finalize rulebook
[ ] implement C_rule
[ ] implement rule margins
[ ] implement near-violation thresholds
[ ] implement rule diagnostics
```

### Testing

```text
[ ] test uniform PG baseline
[ ] test MAB generate-only
[ ] test scenario buffer replay
[ ] test full curriculum
[ ] test mutation validity
[ ] test YAML ablations
```

---

## 27. Final concise summary

The curriculum is a scenario-level automatic curriculum built on MetaDrive procedural generation. A MAB selects one of seven generator arms. Generated scenarios are rolled out with the current agent, exported as ScenarioDescription, evaluated through a rule-aware usefulness

```math
U_s = (C_rule_s, LP_alg_s)
```

and stored in a scenario buffer `Lambda`. The curriculum alternates between generating new scenarios and exploiting the buffer. Exploitation replays useful/stale scenarios and mutates them through controlled transformations of ScenarioDescription tracks. Replay sampling is based on usefulness rank and staleness. Mutation never changes the map or ego trajectory and must pass structural and semantic validation before use. The learning-potential component is algorithm-dependent and supports PPO, TD3, SAC, lexicographic RL, distributional RL, and lexicographic-distributional combinations. The only unresolved part is the exact rulebook-dependent definition of `C_rule_s`, which will be filled after the rulebook task is finalized.
