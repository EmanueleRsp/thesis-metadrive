# Dataset construction funnel

Generated 2026-09-05T15:11:10.068787+00:00 by `scripts/build_dataset_funnel_report.py`.
Every number below is read from a pipeline artifact whose sha256 is recorded in
`provenance.json`. `catalog/split_report.json` is deliberately NOT a source: it is a
stale `SCENARIONET-INTEGRATION` v1.1 report and disagrees with the frozen v1.2 dataset.

## 1. Funnel

| stage | description | waymo | pg | total | retained_vs_previous_pct | retained_vs_converted_pct |
|---|---|---|---|---|---|---|
| S0 | Converted scenarios in the raw catalog | 54104 | 12750 | 66854 |  | 100.0 |
| S1 | Rulebook-eligible | 16603 | 4130 | 20733 | 31.0 | 31.0 |
| S2 | Driving-mission buildable | 12030 | 4129 | 16159 | 77.9 | 24.2 |
| S3 | Split candidate pool | 6634 | 4129 | 10763 | 66.6 | 16.1 |
| S4 | Frozen selection | 1805 | 1695 | 3500 | 32.5 | 5.2 |

## 2. Exclusion causes, counted per record

The pipeline's own `excluded_by_cause` counts *messages*; a record carrying several
messages is counted several times. The table below counts records, attributing each
record to every category it triggered.

| stage | cause category | waymo | pg | total |
|---|---|---|---|---|
| S1_rulebook | assigned_route_invalid | 25556 | 7273 | 32829 |
| S1_rulebook | traffic_controls_skipped_unbuildable_assigned_route | 25556 | 7273 | 32829 |
| S1_rulebook | signal_state_unknown | 16445 | 0 | 16445 |
| S1_rulebook | invalid_map_feature_geometry | 6872 | 0 | 6872 |
| S1_rulebook | task_route_lane_association_ambiguous_or_unavailable | 5443 | 1347 | 6790 |
| S1_rulebook | movement_key_ambiguous | 706 | 0 | 706 |
| S1_rulebook | invalid_map_feature | 47 | 0 | 47 |
| S2_driving_mission | mission_build_error | 4573 | 1 | 4574 |
| S3_offline_quality | too many dynamic objects | 3056 | 0 | 3056 |
| S3_offline_quality | degenerate SDC route | 2837 | 0 | 2837 |
| S3_offline_quality | possible overpass/elevation artifact | 145 | 0 | 145 |

## 3. Frozen selection by pool, source and arm

| pool | source | A0 | A1 | A2 | A3 | A4 | A5 | total |
|---|---|---|---|---|---|---|---|---|
| train | Waymo | 20 | 126 | 127 | 268 | 367 | 192 | 1100 |
| train | PG | 346 | 240 | 240 | 99 | 0 | 175 | 1100 |
| validation (empirical) | Waymo | 0 | 8 | 49 | 49 | 18 | 26 | 150 |
| validation (empirical) | PG | 87 | 51 | 9 | 1 | 0 | 2 | 150 |
| test (empirical) | Waymo | 1 | 37 | 113 | 126 | 68 | 55 | 400 |
| test (empirical) | PG | 197 | 70 | 23 | 2 | 0 | 8 | 300 |
| test (arm-stratified) | Waymo | 5 | 25 | 25 | 25 | 50 | 25 | 155 |
| test (arm-stratified) | PG | 45 | 25 | 25 | 25 | 0 | 25 | 145 |

## 4. Arm thresholds in force

| parameter | value |
|---|---|
| balanced_source_count | 1100 |
| balanced_sources | True |
| computed_on_split | train |
| dense_traffic_quantile | 0.75 |
| feature_version | v2 |
| low_traffic_quantile | 0.4 |
| relevant_radius_m | 50.0 |
| tau_dense | 19 |
| tau_low | 5 |
| temporal_quantile | 0.9 |
| vertical_tolerance_m | 3.0 |

## 5. Artifacts written

- `tables/funnel.csv`, `tables/exclusion_causes.csv`, `tables/split_source_arm.csv`
- `tables/descriptive_stats.csv`, `tables/panels.csv`, `tables/pg_profile_mixture.csv`
- `tables/arm_thresholds.csv`, `tables/candidate_pool_quality.csv`
- `figures/fig_funnel.svg`, `figures/fig_arm_distribution.svg`, `figures/fig_route_length_cdf.svg`, `figures/fig_agent_density_cdf.svg`, `figures/fig_pg_mixture.svg`
- `provenance.json`
