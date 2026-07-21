# Synthetic Rulebook scenario fixtures

This directory contains small, deterministic MetaDrive `ScenarioDescription`
fixtures used only by automated Rulebook conformance tests. They are not Waymo
data, ScenarioNet records, training data, evaluation data, or curriculum arms.

`manifest.json` identifies every checked-in descriptor and its target transition.
Generate committed `.pkl` files in the repository's primary container, so their
NumPy pickle module names are compatible with test and CI environments:

```bash
docker compose run --rm dev uv run --no-sync env PYTHONPATH=src:tests python tests/generate_rulebook_scenarios.py
```

Do not hand-edit a pickle. Change `tests/rulebook_scenario_fixtures.py`, rerun
the generator, and update the manifest oracle and acceptance tests together.

The initial feasibility set contains `red_light`, `crosswalk_pedestrian`, and
`vehicle_pedestrian_collision`. Every descriptor must pass schema validation,
load through `ScenarioOnlineEnv`, and normalize through the Rulebook static-map
boundary.
