# Specification amendment: LiDAR arm is blind to traffic lights

## Metadata

- Feature: `obs_lidar_traffic_light_blindness`
- Specification ID: `OBS-LIDAR-V2.0.1`
- Version: `2.0.1`
- Status: `APPROVED`
- Date: `2026-08-09`
- Amends: `docs/specifications/observation_lidar_v2.0_specification.md` §5
  (known limitations) only. The observation schema, dimensionality (`D = 6489`),
  frame stacking, masking, tokenization and the `ENC-V1.4` encoder contract are
  unchanged.
- Related ADR: `docs/decisions/ADR-062-ego-lidar-excludes-traffic-light-air-walls.md`
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
- Approval evidence: explicit user approval on 2026-08-09 in the originating
  conversation (*"Ok"*), after the user identified the pre-existing channel as
  illegitimate (*"mi sembrerebbe cheating"*).
- Authoritative: `YES` for §2 below; not authoritative for anything else.

## 1. Context

Every MetaDrive traffic light instantiates a ghost physics box of
`0.25 m × lane_width × 1.5 m` at the stop point, whose *into*-collide mask is
`CollisionGroup.InvisibleWall` when the light is red or yellow and `AllOff` when
it is green or unknown. `InvisibleWall` belongs to
`CollisionGroup.can_be_lidar_detected()`, the raycast mask of the shared LiDAR
sensor.

The `stacked_lidar_v2` observation therefore contained an obstacle spanning the
lane that appeared when a light turned red and vanished when it turned green:
the signal phase leaking into the point cloud as geometry. No physical LiDAR
reports a signal phase, and the returned cloud is untyped distance, so the
channel could not be removed downstream.

`ADR-062` removes it by restricting the ego-side raycast mask, which leaves
reactive-traffic light compliance and the `vehicle.red_light` diagnostic intact
because both use different code paths.

## 2. Amended known limitations

`OBS-LIDAR-V2.0` §5's list of known limitations is extended with:

> **No traffic-control information.** The LiDAR arm receives no observation of
> traffic-light presence, position or phase. Range returns from traffic-light
> air walls are excluded from the ego point cloud. This supersedes and
> generalizes the previous entry stating that `yellow_must_stop` is
> unreconstructible: the arm has no signal information at all, not merely an
> incomplete reconstruction of the yellow commitment.

## 3. Consequence for the arm comparison

The semantic arm (`OBS-V1.3`) observes signal state explicitly and
perception-bounded, through the signal camera of `ADR-045`. Any measured
advantage of the semantic arm at signalised intersections is therefore
attributable to that sensor rather than to an accidental geometric artifact of
the simulator. Reports comparing the two arms must state this limitation
alongside the result.

## 4. Compatibility

The observation schema version is unchanged and existing checkpoints remain
loadable. LiDAR observations recorded before this amendment are not comparable
for signalised scenarios, because the point cloud content changes there.
