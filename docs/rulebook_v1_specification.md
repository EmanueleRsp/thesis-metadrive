# Rulebook v1 Specification for MetaDrive RL

## 1. Objective

This document specifies the revised **Rulebook v1** for the MetaDrive autonomous driving reinforcement learning setup.

The previous rulebook used four top-level rules:

1. `vehicle_collision_energy`;
2. `drivable_area`;
3. `wrong_way`;
4. `goal_progress`.

The revised rulebook keeps four top-level rules, following the supervisor's indication that the rulebook should remain compact and composed of justified, atomic rules.

The new priority order is:

\[
\texttt{collision\_severity}
\succ
\texttt{allowed\_driving\_area}
\succ
\texttt{lane\_marking\_compliance}
\succ
\texttt{local\_route\_progress}.
\]

The corresponding rule/reward vector is:

\[
r_t =
\bigl[
r_t^{\text{coll}},
r_t^{\text{allowed}},
r_t^{\text{mark}},
r_t^{\text{prog}}
\bigr].
\]

The design addresses the main issues raised during the meeting:

- `wrong_way` should not remain an independent top-level rule;
- the previous rulebook lacked a true lane keeping / lane boundary rule;
- the previous `goal_progress` was a weak final-target proximity objective;
- a minimum-speed or speed-maximization rule may introduce undesirable bias;
- termination and rule violation must remain conceptually separate;
- rule margins should support reward construction, logging, diagnostics, and future prioritized replay.

---

## 2. Design Rationale

Additional concerns are not added as extra top-level rules. They are placed at the appropriate layer:

| Concern | Handling in Rulebook v1 |
|---|---|
| Collision severity | Top-level rule R1 |
| Drivable / allowed area | Top-level rule R2 |
| Wrong-way | Absorbed into `allowed_driving_area` when possible; otherwise logged as diagnostic |
| Lane keeping | Top-level rule R3 as `lane_marking_compliance` |
| Solid vs dashed lane markings | Subcases inside R3 |
| Goal progress too weak | Replaced by transition-based `local_route_progress` |
| Single checkpoint too myopic | Fallback progress uses multiple weighted checkpoints |
| Speed minimum bias | No top-level speed rule |
| Stagnation | Diagnostic only at first |
| Stop signs / traffic lights | Deferred to Rulebook v2 |
| Comfort / jerk / acceleration smoothness | Outside rulebook; future action smoothing or offline metrics |
| Informative replay | Supported by margins, violations and severities |
| Termination | Separate environment/training configuration |

---

## 3. Scope of Rulebook v1

### Included

- ego vehicle;
- dynamic vehicles;
- static obstacles, if active in the environment;
- physical and semantic allowed driving region;
- lane markings or lane boundaries, if available;
- route progress.

### Excluded

- pedestrians;
- cyclists;
- generic VRUs;
- stop signs;
- traffic lights;
- comfort;
- jerk;
- acceleration smoothness.

VRUs are excluded because the current curriculum does not support them. If they are introduced later, they should be added through an explicit rulebook refinement and not silently treated as ordinary vehicles.

Static obstacles are included in the collision rule if they are active in the environment. If an experiment is explicitly vehicle-only, static obstacles can be disabled in the environment; otherwise they must not be ignored by the rulebook.

---

## 4. Common Rule Output Schema

Each rule must return the same structured output:

```python
{
    "name": str,
    "margin": float,
    "violated": bool,
    "severity": float,
    "available": bool,
    "fallback_used": bool,
    "raw": dict
}
```

For R1, R2 and R3, which are constraint-like rules:

\[
m_i(t)=0
\]

means that the rule is satisfied, while:

\[
m_i(t)<0
\]

means that the rule is violated.

For R4, `local_route_progress`, the margin can be positive:

\[
m_{\text{prog}}(t)>0
\]

means forward progress;

\[
m_{\text{prog}}(t)\approx 0
\]

means no progress or stagnation;

\[
m_{\text{prog}}(t)<0
\]

means regression or movement away from the route.

Therefore, only constraint-like rules have maximum natural margin equal to zero.

---

# 5. Rule 1 — Collision Severity

## 5.1 Name

`collision_severity`

This replaces `vehicle_collision_energy`.

The name is intentionally general because the rule covers all relevant collidable objects currently present in the environment: dynamic vehicles and static obstacles.

## 5.2 Intuition

The ego vehicle should not collide with relevant objects. Collision is the highest-priority rule because it is the most safety-critical event.

The rule should not be purely binary. It should return a severity value so that high-speed collisions are worse than low-speed contacts.

## 5.3 Objects considered

Let \(\mathcal{O}_t\) be the set of relevant collidable objects:

\[
\mathcal{O}_t = \mathcal{V}_t \cup \mathcal{S}_t,
\]

where \(\mathcal{V}_t\) is the set of dynamic vehicles and \(\mathcal{S}_t\) is the set of static obstacles.

VRUs, pedestrians and cyclists are out of scope for Rulebook v1.

The object type must be logged:

```python
raw["collision_object_type"] = "vehicle" | "static_obstacle" | "unknown"
```

## 5.4 Formal condition

The ego footprint should not intersect any relevant object:

\[
G_{[T_1,T_2]}
\left(
\forall i \in \mathcal{O}_t
:
p_0(t) \cap p_i(t) = \emptyset
\right).
\]

## 5.5 Collision set

\[
\mathcal{C}_t =
\left\{
i \in \mathcal{O}_t
:
p_0(t) \cap p_i(t) \neq \emptyset
\right\}.
\]

## 5.6 Severity

For a dynamic object \(i\), the preferred severity is relative kinetic energy:

\[
S_i(t)=
\frac{1}{2}\mu_i
\left\|
v_0(t^-) - v_i(t^-)
\right\|^2.
\]

The reduced mass is:

\[
\mu_i =
\frac{m_0m_i}{m_0+m_i}.
\]

If masses are unavailable or unreliable:

\[
S_i(t)=
\left\|
v_0(t^-) - v_i(t^-)
\right\|^2.
\]

If the pre-impact state is unavailable, use current velocities as fallback.

For a static obstacle, \(v_i(t)=0\), so the simplified severity is:

\[
S_i(t)=\left\|v_0(t^-)\right\|^2.
\]

Fallback, if necessary, is the absolute kinetic-energy delta:

\[
S_i^{\Delta E}(t)=
\left|
E_i(t)-E_i(t-1)
\right|.
\]

The signed energy delta should not be the primary definition.

## 5.7 Margin

\[
m_{\text{coll}}(t)=
\begin{cases}
-\max\limits_{i \in \mathcal{C}_t} S_i(t), & \mathcal{C}_t \neq \emptyset,\\
0, & \mathcal{C}_t = \emptyset.
\end{cases}
\]

The maximum preserves a worst-case semantics: simultaneous collisions are evaluated by the most severe contact.

The logged severity is:

\[
\text{severity}_{\text{coll}}(t)=
\max(0,-m_{\text{coll}}(t)).
\]

## 5.8 Required inputs

Preferred:

- ego footprint;
- object footprints;
- ego velocity;
- object velocity;
- object type;
- previous ego/object state;
- masses, if available.

Fallback:

- collision flag;
- ego speed;
- current object speed, if available;
- static obstacle flag.

---

# 6. Rule 2 — Allowed Driving Area

## 6.1 Name

`allowed_driving_area`

This replaces/refines `drivable_area` and absorbs the old `wrong_way` rule when enough information is available.

## 6.2 Intuition

The previous `drivable_area` rule only checked whether ego stayed on the physically drivable surface.

However, a region can be physically drivable but not allowed for ego. An opposite-direction lane is still road surface, but ego should not drive there.

Therefore, this rule checks whether ego remains inside the **allowed driving region**.

## 6.3 Formal condition

\[
G_{[T_1,T_2]}
\left(
p_0(t) \subseteq \mathcal{R}_{allowed}(t)
\right).
\]

The allowed region is a subset of the physical drivable region:

\[
\mathcal{R}_{allowed}(t)
\subseteq
\mathcal{R}_{driv}(t).
\]

## 6.4 Allowed region

When available, \(\mathcal{R}_{allowed}(t)\) should exclude:

- off-road regions;
- sidewalks;
- non-drivable road regions;
- opposite carriageways;
- wrong-direction lanes;
- route-incompatible lanes.

A practical definition is:

\[
\mathcal{R}_{allowed}(t)=
\mathcal{R}_{driv}(t)
\setminus
\mathcal{R}_{forbidden}(t).
\]

## 6.5 Margin

Define the ego area outside the allowed region:

\[
A_{out}(t)=
\left\|
p_0(t)\setminus\mathcal{R}_{allowed}(t)
\right\|.
\]

Basic margin:

\[
m_{allowed}(t)=-A_{out}(t).
\]

If distance to allowed region is available:

\[
m_{allowed}(t)=
-
\left(
A_{out}(t)+
\lambda_d d(p_0(t),\mathcal{R}_{allowed}(t))^2
\right).
\]

If ego is fully inside the allowed region:

\[
m_{allowed}(t)=0.
\]

If ego is partially or fully outside:

\[
m_{allowed}(t)<0.
\]

Logged severity:

\[
\text{severity}_{allowed}(t)=
\max(0,-m_{allowed}(t)).
\]

## 6.6 Fallback

If the full allowed region cannot be constructed:

\[
\mathcal{R}_{allowed}(t)\approx\mathcal{R}_{driv}(t).
\]

Then the rule behaves like the old physical drivable-area rule and `fallback_used=True`.

If wrong-way information is available but cannot be integrated geometrically, log:

```python
raw["wrong_way_diagnostic"] = {
    "available": bool,
    "overlap_ratio": float,
    "violated": bool
}
```

---

# 7. Rule 3 — Lane Marking Compliance

## 7.1 Name

`lane_marking_compliance`

This is the new lane keeping / lane boundary rule.

It is intentionally narrower than `lane_compliance`, because wrong-way and lane admissibility are already handled by `allowed_driving_area`.

## 7.2 Intuition

Rule 2 checks whether ego is in an allowed driving region.

Rule 3 checks whether ego respects lane markings and lane boundaries inside that allowed region.

It covers:

- avoiding occupation of solid lines;
- avoiding persistent occupation of dashed lines;
- avoiding extended driving across lane boundaries.

## 7.3 Formal condition

\[
G_{[T_1,T_2]}
\left(
p_0(t)\cap\mathcal{B}_{forbidden}(t)=\emptyset
\right).
\]

Here \(\mathcal{B}_{forbidden}(t)\) includes forbidden lane markings, especially solid lane boundaries.

Dashed lines are not instant hard violations; persistent occupancy is penalized.

## 7.4 Solid overlap

\[
O_{solid}(t)=
\left\|
p_0(t)\cap\mathcal{B}_{solid}(t)
\right\|.
\]

## 7.5 Persistent dashed overlap

Instantaneous dashed overlap:

\[
O_{dash}(t)=
\left\|
p_0(t)\cap\mathcal{B}_{dash}(t)
\right\|.
\]

Persistent dashed overlap:

\[
O_{dash}^{persist}(t)=
O_{dash}(t)
\cdot
\mathbb{1}
\left[
N_{dash}(t)\ge N_{min}^{dash}
\right].
\]

Initial configurable default:

\[
N_{min}^{dash}=5.
\]

## 7.6 Margin

\[
m_{mark}(t)=
-
\left(
\alpha_s O_{solid}(t)+
\alpha_d O_{dash}^{persist}(t)
\right).
\]

Initial configurable defaults:

\[
\alpha_s=1.0,
\qquad
\alpha_d=0.25.
\]

with:

\[
\alpha_s>\alpha_d.
\]

If markings are respected:

\[
m_{mark}(t)=0.
\]

If markings are violated:

\[
m_{mark}(t)<0.
\]

Logged severity:

\[
\text{severity}_{mark}(t)=
\max(0,-m_{mark}(t)).
\]

## 7.7 Fallback

If solid/dashed metadata is unavailable, use lane-boundary overlap:

\[
m_{mark}(t)=-O_{boundary}(t).
\]

If no lane-boundary information exists, return the rule as unavailable:

```python
{
    "available": False,
    "fallback_used": False,
    "margin": 0.0,
    "violated": False,
    "severity": 0.0,
    "raw": {}
}
```

---

# 8. Rule 4 — Local Route Progress

## 8.1 Name

`local_route_progress`

This replaces the old `goal_progress`.

## 8.2 Intuition

The old `goal_progress` was proximity to the final target. It was weak because the goal can be far away and the signal can be almost always negative.

The new rule measures local, transition-based progress along the route.

## 8.3 Preferred implementation: route progress coordinate

If MetaDrive exposes a reliable curvilinear route-progress coordinate \(s(t)\), use:

\[
m_{prog}(t)=s(t)-s(t-1).
\]

This means:

\[
m_{prog}(t)>0
\]

forward progress;

\[
m_{prog}(t)\approx0
\]

no progress;

\[
m_{prog}(t)<0
\]

regression or movement away from the route.

This is preferred because it directly measures advancement along the planned route.

## 8.4 Fallback: weighted local checkpoints

If reliable route-progress coordinate is unavailable, use the next \(K\) route checkpoints.

Let \(c_0(t)\) be the ego center and \(q_i(t)\) the \(i\)-th future route checkpoint.

\[
D(t)=
\sum_{i=1}^{K}
w_i
\left\|
c_0(t)-q_i(t)
\right\|.
\]

Then:

\[
m_{prog}(t)=D(t-1)-D(t).
\]

Default:

\[
K=5.
\]

Weights:

\[
w=[0.40,0.25,0.15,0.12,0.08].
\]

The weights must be configurable.

## 8.5 Severity output

Since progress is an objective, not a hard constraint:

\[
\text{severity}_{prog}(t)=
\max(0,-m_{prog}(t)).
\]

Only negative progress is treated as violation-like severity. Stagnation is handled by a diagnostic.

---

# 9. Diagnostics Outside the Rulebook

Diagnostics are logged but are not top-level rulebook entries.

## 9.1 Wrong-way diagnostic

Wrong-way is not a top-level rule.

When available, log:

\[
r_{opp}(t)=
\frac{
\left\|
p_0(t)\cap\mathcal{R}_{opp}(t)
\right\|
}{
\left\|
p_0(t)
\right\|
}.
\]

This verifies whether `allowed_driving_area` captures opposite-lane violations.

## 9.2 Stagnation diagnostic

Stagnation is not part of the top-level rulebook.

\[
\text{stuck}(t)=
\mathbb{1}
\left[
|m_{prog}(t)|<\epsilon_p
\land
v_0(t)<\epsilon_v
\right].
\]

Initial defaults:

\[
\epsilon_p=0.01,
\qquad
\epsilon_v=0.1\text{ m/s}.
\]

Accumulate over time:

\[
N_{stuck}(t)=
\sum_{\tau=t-H}^{t}
\text{stuck}(\tau).
\]

Log stagnation if:

\[
N_{stuck}(t)\ge N_{min}.
\]

Initial default:

\[
N_{min}=20.
\]

Stagnation should initially be logged only. It should not affect the reward vector unless experiments show that local route progress is insufficient.

## 9.3 Speed diagnostic

Log ego speed:

\[
v_0(t).
\]

Speed is not a top-level rule because a minimum-speed rule can introduce unsafe bias.

## 9.4 Comfort / jerk / acceleration diagnostics

Comfort, jerk and acceleration smoothness are outside Rulebook v1.

They can be evaluated offline, for example:

\[
J(t)=
\left\|
a_t-2a_{t-1}+a_{t-2}
\right\|.
\]

Future action smoothing can be represented as:

\[
a'_t=(1-\beta)a_t+\beta a'_{t-1}.
\]

This must not be implemented inside Rulebook v1.

---

# 10. Deferred Rules

## 10.1 Stop-sign compliance

Deferred because it is time-dependent and requires stop-sign detection, distance to stop line, temporal stopping window and restart behavior.

## 10.2 Traffic-light compliance

Deferred because it requires traffic-light state, distance to stop line, and red/yellow/green temporal semantics.

## 10.3 Pedestrian / cyclist / VRU safety

Deferred because the current curriculum does not support VRUs.

## 10.4 Comfort / jerk / acceleration smoothness

Deferred because these are lower-priority driving-quality objectives better handled through action smoothing, trajectory smoothing, auxiliary regularization or offline metrics.

---

# 11. Termination Policy

Rulebook violations do not automatically imply episode termination.

Termination is handled separately by environment/training configuration.

Recommended current base setting:

| Event | Terminates episode? |
|---|---:|
| Collision | Yes |
| Out-of-road / severe outside allowed area | Yes |
| Wrong-way | No |
| Lane marking violation | No |
| Negative progress | No |
| Stagnation | No |

Out-of-road termination is kept in the current base setting because early tests with native MetaDrive reward showed that not terminating out-of-road can introduce excessive noise and make the task significantly harder.

A relaxed non-terminating out-of-road setting can be retested later once the rulebook reward is stable.

---

# 12. Scalarization Notes

After implementing the new margins, the scalar rule reward must be recalibrated.

For each rule:

1. collect active margins;
2. inspect sign and value range;
3. compute percentile-based scale;
4. verify that margins are not always zero;
5. verify that scalar reward does not favor standing still.

The desired qualitative ordering for debugging is:

\[
\text{safe forward driving}
>
\text{standing still}
>
\text{unsafe driving}.
\]

R1, R2 and R3 are constraints. R4 is a progress objective and can be positive.

---

# 13. Logging Requirements

At each step, log:

```python
info["rulebook"] = {
    "collision_severity": {...},
    "allowed_driving_area": {...},
    "lane_marking_compliance": {...},
    "local_route_progress": {...},
    "diagnostics": {
        "wrong_way": {...},
        "stagnation": {...},
        "speed": ...,
        "comfort": {...}
    }
}
```

For each top-level rule, log:

- margin;
- violated;
- severity;
- available;
- fallback_used;
- raw values.

Also log:

- reward vector;
- scalar rule reward;
- native environment reward;
- final training reward;
- termination reason;
- collision object type;
- active fallback mode;
- progress implementation used: `route_coordinate` or `checkpoint_fallback`.

---

# 14. Sanity Tests

## Collision tests

1. No collision: \(m_{coll}=0\).
2. Low-speed collision: \(m_{coll}<0\).
3. High-speed collision: \(|m_{coll}^{high}|>|m_{coll}^{low}|\).
4. Static-obstacle collision is detected if static obstacles are active.

## Allowed-area tests

5. Ego inside allowed area: \(m_{allowed}=0\).
6. Ego outside road: \(m_{allowed}<0\).
7. Ego in opposite lane: violates `allowed_driving_area` if allowed-region geometry exists; otherwise activates `wrong_way_diagnostic`.

## Lane-marking tests

8. Ego not on lane markings: \(m_{mark}=0\).
9. Ego on solid line: \(m_{mark}<0\).
10. Ego briefly on dashed line: no or weak penalty.
11. Ego persistently on dashed line: \(m_{mark}<0\).

## Progress tests

12. Ego advances along route: \(m_{prog}>0\).
13. Ego stationary: \(m_{prog}\approx0\).
14. Ego moves backward or away from route: \(m_{prog}<0\).

## Reward-level tests

15. Safe forward driving should have higher scalar reward than standing still.
16. Standing still should have higher scalar reward than collision or out-of-road.
17. Scalar reward should not be almost always negative in normal forward driving.

---

# 15. Implementation Instructions for Codex

Codex should implement the rulebook incrementally.

Required steps:

1. Inspect the current MetaDrive wrapper and rulebook evaluator.
2. Identify available API/geometry for:
   - ego footprint;
   - dynamic vehicles;
   - static obstacles;
   - drivable area;
   - lane direction;
   - opposite lane;
   - lane markings;
   - lane boundaries;
   - route progress coordinate;
   - future checkpoints.
3. Implement the common rule output schema.
4. Implement `collision_severity`.
5. Implement `allowed_driving_area`.
6. Implement `lane_marking_compliance`.
7. Implement `local_route_progress`.
8. Add diagnostics:
   - wrong-way;
   - stagnation;
   - speed;
   - comfort/jerk placeholder if easy.
9. Add or update logging.
10. Add config flags for fallback modes.
11. Add sanity tests.
12. Do not modify unrelated training code.

---

# 16. Suggested Configuration

```yaml
rulebook:
  enabled: true

  rules:
    collision_severity:
      enabled: true
      include_dynamic_vehicles: true
      include_static_obstacles: true
      include_vrus: false
      use_reduced_mass: false
      fallback_abs_delta_energy: true

    allowed_driving_area:
      enabled: true
      use_allowed_region_if_available: true
      fallback_to_drivable_area: true
      log_wrong_way_diagnostic: true

    lane_marking_compliance:
      enabled: true
      alpha_solid: 1.0
      alpha_dashed: 0.25
      dashed_persistence_steps: 5
      fallback_to_lane_boundary: true

    local_route_progress:
      enabled: true
      prefer_route_progress_coordinate: true
      fallback_to_weighted_checkpoints: true
      checkpoint_k: 5
      checkpoint_weights: [0.40, 0.25, 0.15, 0.12, 0.08]

  diagnostics:
    stagnation:
      enabled: true
      epsilon_progress: 0.01
      epsilon_speed: 0.1
      min_steps: 20

termination:
  collision_done: true
  out_of_road_done: true
  wrong_way_done: false
  lane_marking_done: false
  progress_failure_done: false
  stagnation_done: false
```

---

# 17. Backward Compatibility

Old rule names may remain only as aliases if needed:

| Old name | New handling |
|---|---|
| `vehicle_collision_energy` | alias/deprecated form of `collision_severity` |
| `drivable_area` | fallback physical-area mode of `allowed_driving_area` |
| `wrong_way` | diagnostic only, not top-level rule |
| `goal_progress` | deprecated, replaced by `local_route_progress` |

Old behavior must not silently remain active unless explicitly configured.

---

# 18. Final Summary Table

| Priority | Rule | Replaces | Margin semantics | Main role |
|---:|---|---|---|---|
| 1 | `collision_severity` | `vehicle_collision_energy` | \(0\) if no collision, \(<0\) if collision | Safety |
| 2 | `allowed_driving_area` | `drivable_area` + `wrong_way` | \(0\) if ego is inside allowed region, \(<0\) otherwise | Road admissibility |
| 3 | `lane_marking_compliance` | new lane keeping rule | \(0\) if markings respected, \(<0\) if improperly occupied | Lane discipline |
| 4 | `local_route_progress` | `goal_progress` | \(>0\) progress, \(\approx0\) no progress, \(<0\) regression | Task completion |

Diagnostics outside the top-level rulebook:

| Diagnostic | Purpose |
|---|---|
| `wrong_way_diagnostic` | verify opposite-lane occupation when not integrated in allowed area |
| `stagnation_diagnostic` | detect stationary behavior without adding speed bias |
| `speed` | monitor movement without using speed as a rule |
| `comfort/jerk` | future offline metric or action smoothing module |

Deferred rules:

| Deferred rule | Reason |
|---|---|
| stop signs | time-dependent, Rulebook v2 |
| traffic lights | time-dependent, Rulebook v2 |
| VRUs | not supported by current curriculum |
| comfort/jerk/smoothness | better handled downstream or offline |

---

# 19. Final Decision

Rulebook v1 is closed as:

\[
\boxed{
\texttt{collision\_severity}
\succ
\texttt{allowed\_driving\_area}
\succ
\texttt{lane\_marking\_compliance}
\succ
\texttt{local\_route\_progress}
}
\]

This version is compact, aligned with the supervisor feedback, and suitable for implementation and testing with Codex.
