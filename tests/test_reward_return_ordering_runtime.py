"""Reward well-posedness in the live runtime: standstill < partial < completion.

A reward that drives learning must rank three reference behaviours in this
order on the *episode return* the learner actually receives, with the real
termination convention, wrappers and scalarizer in the loop:

1. ``still``: the ego holds the brake from the first step;
2. ``partial``: the logged expert is replayed for half its trajectory, then
   the ego is held still;
3. ``full``: the logged expert is replayed to the end of the scenario.

The check runs on the first scenario of each frozen validation panel, so the
episodes are the ones every screening arm is evaluated on. It is an integration
test: it needs the prepared ScenarioNet runtime and takes minutes, not seconds.

Each ``(arm, panel)`` pair is a separate case. The three behaviours have to be
compared inside one panel, so they stay in one case, but the panels are
independent of each other: splitting them keeps a case near five minutes instead
of ten, lets a parallel run overlap them, and names the failing panel in the
node id instead of inside a joined message.

The ordering is a *necessary* condition, not a sufficient one. MetaDrive's
native reward satisfies it (a replayed expert earns a large positive return)
and still drove arm A of the learnability screening to a standstill, because
under the thesis termination convention its per-step lane-line penalty is
unbounded while a standstill scores exactly zero: every *imperfect* motion in
the neighbourhood of a random policy scores far below standing still. That
property is measured by `scripts/evaluate_constant_action_baseline.py`
(`brake` against `random`), not by this test. See ``docs/open_items.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from thesis_rl.envs.factory import _resolve_agent_policy
from thesis_rl.envs.policies.replay_ego import ReplayEgoStopPolicy

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
PRESETS = {
    "a_native": "presets/learnability/sac_a_native",
    "b_rulebook": "presets/learnability/sac_b_rulebook",
}
PANELS = ("validation_waymo_empirical", "validation_pg")
BEHAVIOURS: dict[str, dict[str, Any]] = {
    "still": {"agent_policy": "env_input_policy"},
    "partial": {"agent_policy": "replay_ego_policy", "replay_ego_stop_fraction": 0.5},
    "full": {"agent_policy": "replay_ego_policy", "replay_ego_stop_fraction": 1.0},
}
BRAKE_ACTION = np.array([0.0, -1.0], dtype=np.float32)
COAST_ACTION = np.zeros(2, dtype=np.float32)


def test_replay_ego_policy_is_resolvable_by_name() -> None:
    assert _resolve_agent_policy("replay_ego_policy") is ReplayEgoStopPolicy
    assert _resolve_agent_policy("replay_ego") is ReplayEgoStopPolicy
    with pytest.raises(ValueError, match="replay_ego_policy"):
        _resolve_agent_policy("teleport_policy")


def _data_root() -> Path:
    return Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet"))


def _require_runtime() -> None:
    root = _data_root()
    if not (root / "frozen" / "scenario_selection_index.json").is_file():
        pytest.skip("requires the frozen ScenarioNet selection index")
    if not (root / "runtime" / "validation" / "dataset_summary.pkl").is_file():
        pytest.skip("requires the prepared canonical ScenarioNet validation runtime")


def _compose(preset: str, run_dir: Path):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(
            config_name=preset,
            overrides=[
                "env.vectorized.enabled=false",
                f"paths.run_dir={run_dir}",
                "experiment.eval_workers=1",
            ],
        )


def _episode_return(env: Any, behaviour: str) -> dict[str, Any]:
    _observation, reset_info = env.reset()
    action = BRAKE_ACTION if behaviour == "still" else COAST_ACTION
    total = 0.0
    steps = 0
    info: dict[str, Any] = {}
    terminated = truncated = False
    while not (terminated or truncated):
        _observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"non-finite reward at step {steps}"
        total += float(reward)
        steps += 1
    return {
        "scenario_uid": reset_info.get("scenario_uid"),
        "return": total,
        "steps": steps,
        "route_completion": float(info.get("route_completion", float("nan"))),
    }


def _returns_by_behaviour(cfg: Any, panel_overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from thesis_rl.runtime.wiring.builders import build_env

    results: dict[str, dict[str, Any]] = {}
    for behaviour, policy_overrides in BEHAVIOURS.items():
        env = build_env(cfg, {**panel_overrides, **policy_overrides})
        try:
            results[behaviour] = _episode_return(env, behaviour)
        finally:
            env.close()
    uids = {result["scenario_uid"] for result in results.values()}
    assert len(uids) == 1, f"behaviours ran on different scenarios: {uids}"
    return results


@pytest.mark.integration
@pytest.mark.parametrize("arm", ["a_native", "b_rulebook"])
@pytest.mark.parametrize("panel_name", PANELS)
def test_reward_return_ordering_on_validation_panels(
    arm: str, panel_name: str, tmp_path: Path, record_property: Any
) -> None:
    _require_runtime()
    from thesis_rl.runtime.evaluation_plan import resolve_scenarionet_evaluation_panels

    cfg = _compose(PRESETS[arm], tmp_path)
    panels = resolve_scenarionet_evaluation_panels(cfg, final=False)
    assert [panel.name for panel in panels] == list(PANELS)
    panel = next(candidate for candidate in panels if candidate.name == panel_name)

    results = _returns_by_behaviour(cfg, panel.env_overrides())
    still, partial, full = (results[key]["return"] for key in ("still", "partial", "full"))
    record_property(f"{arm}:{panel.name}", {key: value["return"] for key, value in results.items()})
    assert (
        results["full"]["route_completion"]
        > results["partial"]["route_completion"]
        > results["still"]["route_completion"]
    ), f"{panel.name}: replay did not produce increasing progress: {results}"
    assert still < partial < full, (
        f"return ordering violated on {panel.name} "
        f"({results['still']['scenario_uid']}): "
        f"still={still:.2f} partial={partial:.2f} full={full:.2f}"
    )
