# ADR-062: The ego LiDAR does not detect traffic-light air walls

- Status: Approved
- Date: 2026-08-09
- Approval evidence: explicit user approval in this conversation. The user
  identified the channel as illegitimate — *"fare detection di un semaforo con
  il lidar identificandolo come un muro non [è] verosimile [...] mi sembrerebbe
  cheating"* — and set the constraint that the fix must not be throwaway code
  (*"vorrei evitare modifiche aggiungendo cose che poi sappiamo già per le run
  finali non ci serviranno"*), then approved (*"Ok"*).
- Affected specification:
  `docs/specifications/observation_lidar_v2.0.1_amendment.md` §1 (adds a known
  limitation to `OBS-LIDAR-V2.0` §5; no change to the observation schema,
  dimensionality or encoder).
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
  (`DEC-RSEC-006`).

## Context

Every MetaDrive traffic light instantiates a ghost physics box of
`0.25 m × lane_width × 1.5 m` at the stop point
(`component/traffic_light/base_traffic_light.py:45`). The box is a ghost node,
so it never blocks the vehicle physically. Its *into*-collide mask is set to
`CollisionGroup.InvisibleWall` when the light is red or yellow and to
`AllOff` when it is green or unknown (`:102-129`).

`InvisibleWall` is inside `CollisionGroup.can_be_lidar_detected()`
(`constants.py:246`), which is the raycast mask of the shared LiDAR sensor
(`component/sensors/lidar.py:27`). The consequence for the
`stacked_lidar_v2` observation arm is a solid obstacle spanning the lane that
appears when the light turns red and vanishes when it turns green — the signal
state leaking into the point cloud as geometry. No real LiDAR reports the phase
of a traffic light, and the returned cloud is untyped distance
(`envs/observations/causal_lidar.py:104`), so it cannot be filtered downstream.

The semantic arm obtains the same information legitimately and explicitly,
through the perception-bounded signal camera (`OBS-V1.3`, `ADR-045`). Leaving
the leak in place would therefore contaminate precisely the comparison between
observation arms that the thesis is built on.

An earlier proposal in this conversation — subclassing `ScenarioTrafficLight` to
stop setting the `InvisibleWall` into-mask — was investigated and **rejected as
incorrect**. Traffic IDM policies detect traffic lights through that same mask:
`IDMPolicy.act` calls `lidar.get_surrounding_objects(...)`
(`policy/idm_policy.py:239`, and `:479` for `TrajectoryIDMPolicy`) and
`lane_change_policy` then tests
`isinstance(surrounding_objects.front_object(), BaseTrafficLight)` (`:346`).
Removing the into-mask would have made all reactive traffic run red lights,
silently degrading scenario realism. It would also have disabled
`vehicle.red_light`, the independent ground truth needed to audit traffic-control
coverage (`ADR-061`).

## Decision

Restrict the **ego-side raycast mask** instead of the object:

```
lidar.mask = CollisionGroup.can_be_lidar_detected() & ~CollisionGroup.InvisibleWall
```

applied to the engine's LiDAR sensor at environment setup.

This is the correct intervention point because only `Lidar.perceive()` consults
`self.mask`. The two mechanisms that must survive use different paths:

- IDM light compliance uses `get_surrounding_objects()`, a broad-phase
  `contactTest` governed by the collision-pair table
  (`(InvisibleWall, LidarBroadDetector, True)`), not by `mask`;
- `vehicle.red_light` / `yellow_light` are set in `_state_check`
  (`component/vehicle/base_vehicle.py:771`) through the chassis contact test,
  governed by `(Vehicle, InvisibleWall, True)`, also not by `mask`.

No patch to the vendored MetaDrive fork is required.

Scope check: `InvisibleWall` is produced by only two components,
`base_traffic_light.py` and `component/pgblock/tollgate.py` (block ID `$`).
No PG profile in `scenarios/pg/profiles.py` uses that token — the token sets are
`S, C, X, T, y, r, R, O` — so traffic lights are the only affected objects in
this repository's scenarios.

Because the change is permanent and the `red_light` probe survives it, the
user's no-throwaway-code constraint is satisfied: the diagnostic counter added
alongside (`red_light == True` while the Rulebook's `signal` sub-rule is
`NOT_APPLICABLE`) remains valid and useful in the final runs.

## Consequences

- The LiDAR arm becomes fully blind to traffic lights. This is an honest
  limitation of a range-sensor-only observation and is recorded as such in the
  `OBS-LIDAR-V2.0` amendment, alongside the pre-existing note that
  `yellow_must_stop` is unreconstructible.
- The comparison between the semantic and LiDAR arms becomes interpretable: the
  semantic arm's advantage at signalised intersections is attributable to its
  signal camera rather than to an accidental geometric artifact.
- Reactive traffic behavior is unchanged; scenario realism is preserved.
- `vehicle.red_light` remains available as an independent, permanent ground
  truth for traffic-control coverage auditing.
- LiDAR observations from prior runs are not comparable for signalised
  scenarios, although the schema and dimensionality are unchanged.

Verification note: that the raycast currently returns the air wall is
**inferred** from the upstream comment (*"add to dynamic world so the lidar can
detect it"*) and the `(InvisibleWall, LidarBroadDetector, True)` pair; it has
not been observed directly. `TEST-RSEC-011` converts this inference into a
verified fact — a raycast against a red light and then a green one — and must
run **before** the mask change is committed, so that the change is known to
remove something real.

Regression tests: `TEST-RSEC-010` (the mask excludes `InvisibleWall` and nothing
else), `TEST-RSEC-011` (no return for a red light; `vehicle.red_light` still
latches; IDM traffic still stops).
