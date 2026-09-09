from __future__ import annotations

import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from thesis_rl.scenarios.frozen import load_frozen_index


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
FROZEN_INDEX = Path("data/scenarionet/frozen/scenario_selection_index.json")


def _measured_training_horizon_steps() -> tuple[int, list[int]]:
    """Return the longest training episode in control steps, and every episode.

    Measured rather than assumed, which is what the docstring below has always
    claimed. The chain is closed in code and each link is checked here:

    * the frozen index's `length` is the scenario's own `SD.LENGTH`, cross-checked
      against the description by `scripts/validate_frozen_scenarionet_content.py`;
    * the runtime reads the same field
      (`third_party/metadrive/metadrive/manager/scenario_data_manager.py`);
    * the episode truncates at `episode_steps >= scenario_length - 1 + extra`
      (`src/thesis_rl/envs/thesis_scenario_env.py`), and `conf/env/scenarionet.yaml`
      sets `extra_steps_after_scenario: 0` with `horizon: null`, so nothing caps
      an episode earlier.

    An episode is therefore `length - 1` control steps, and the horizon is the
    maximum over the **training** split, because that is the mixture the discount
    has to keep ordered while learning.
    """

    payload = load_frozen_index(FROZEN_INDEX)
    episodes = [
        int(record["length"]) - 1 for record in payload["records"] if record.get("split") == "train"
    ]
    assert episodes, "the frozen index has no training records; the guard would pass vacuously"
    return max(episodes), episodes


def _compose_preset(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name=config_name)


def test_td3_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_sb3")
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch) == [256, 256]


def test_td3_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "latent_query_v2"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"


def test_td3_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"


def test_sac_sb3_algorithm_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=sac_sb3"])
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch) == [256, 256]


def test_sac_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_sac_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "latent_query_v2"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_sac_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_ppo_sb3_algorithm_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=ppo_sb3"])
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert int(cfg.agent.planner.algorithm.n_steps) == 96
    assert int(cfg.agent.planner.algorithm.batch_size) == 63
    assert int(cfg.agent.planner.algorithm.n_epochs) == 10
    assert float(cfg.agent.planner.algorithm.ent_coef) == 0.0
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch.pi) == [256, 256]
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch.vf) == [256, 256]


def test_ppo_algorithm_config_composes_with_sb3_faithful_defaults() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=ppo"])
    assert str(cfg.agent.planner.algorithm.name) == "ppo"
    assert int(cfg.agent.planner.algorithm.n_steps) == 2048
    assert int(cfg.agent.planner.algorithm.batch_size) == 64
    assert int(cfg.agent.planner.algorithm.n_epochs) == 10
    assert float(cfg.agent.planner.algorithm.ent_coef) == 0.0
    assert float(cfg.agent.planner.algorithm.optimizer_eps) == 1e-5


def test_ppo_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"


def test_ppo_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "latent_query_v2"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"


def test_ppo_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"


def test_selection_td3_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/td3_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_sac_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_ppo_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/ppo_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_sac_sb3_native_contract_strict_fast_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_strict_fast")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert bool(cfg.env.config.on_broken_line_done) is False
    assert int(cfg.env.vectorized.num_envs) == 5


def test_selection_sac_sb3_native_contract_relaxed_fast_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_relaxed_fast")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_relaxed"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is False
    assert bool(cfg.env.config.on_broken_line_done) is False
    assert int(cfg.env.vectorized.num_envs) == 5


def test_selection_sac_sb3_native_contract_strict_fast_env4_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_strict_fast_env4")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert int(cfg.env.vectorized.num_envs) == 4


def test_smoke_train_preset_composes() -> None:
    cfg = _compose_preset("presets/test/smoke_train")
    assert str(cfg.run_profile.name) == "smoke"
    assert str(cfg.experiment.name) == "smoke"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"


def _shipped_discount() -> float:
    """The one discount every arm shares, read from the algorithm configs."""

    algorithm_dir = CONF_DIR / "agent" / "planner" / "algorithm"
    observed: set[str] = set()
    for config in sorted(algorithm_dir.glob("*.yaml")):
        text = config.read_text(encoding="utf-8")
        observed.update(
            line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("gamma:")
        )
    assert len(observed) == 1, f"arms do not share one discount: {sorted(observed)}"
    return float(observed.pop())


def test_every_algorithm_shares_one_hierarchy_preserving_discount() -> None:
    """`AC-RB5.1-16` / ADR-081, amending ADR-075.

    **One discount for every arm.** Differing discounts across arms would make a
    difference in results non-attributable to the preference structure under
    test, which is the comparison this thesis exists to make. The shaping
    discount is coupled to it rather than free, because Ng et al.'s
    policy-invariance result for potential-based shaping holds only when the two
    agree.

    ADR-081's second property — that the discount must not invert the hierarchy
    inside an episode — used to be asserted here against a horizon of `199`
    hardcoded under a docstring reading "the horizon is measured, not assumed".
    It was not measured, and `199` is the wrong number: `RULEBOOK-V5.1` §4.6's
    p5/p50/p95 of 197/199/200 is the **1100 Waymo train records only**, while the
    training mixture is half procedurally generated. The horizon is now read from
    the committed index by
    `test_the_hierarchy_preserving_horizon_is_measured_from_the_frozen_index`, and
    the criterion's verdict lands with the approved `gamma = 0.9982` rather than
    here, next to the value that makes it hold (`C49`).

    `gamma = 1` satisfied the second property trivially and failed something
    ADR-075 did not weigh: with two thirds of episodes ending in a bootstrapped
    truncation there is no contraction and the value level is pinned only by the
    terminating minority. See ADR-081.
    """

    algorithm_dir = CONF_DIR / "agent" / "planner" / "algorithm"
    configs = sorted(algorithm_dir.glob("*.yaml"))
    assert configs, "no algorithm configs found; the guard would pass vacuously"

    observed: set[str] = set()
    for config in configs:
        text = config.read_text(encoding="utf-8")
        gammas = [
            line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("gamma:")
        ]
        observed.update(gammas)

        # Ng et al.: potential-based shaping is policy-invariant only when its
        # discount is the MDP's, so this one tracks `gamma` rather than being
        # free to differ.
        shaping = [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("learning_potential_gamma:")
        ]
        assert shaping in ([], gammas), f"{config.name} shaping discount: {shaping} vs {gammas}"

    assert len(observed) == 1, f"arms do not share one discount: {sorted(observed)}"
    gamma = float(observed.pop())
    assert 0.0 < gamma <= 1.0, f"gamma must lie in (0, 1], got {gamma}"
    assert gamma == _shipped_discount()


@pytest.mark.integration
def test_the_hierarchy_preserving_horizon_is_measured_from_the_frozen_index() -> None:
    """`C49`. The horizon ADR-081's criterion is about, measured instead of assumed.

    The criterion is `Delta = ln(a) / -ln(gamma) > L`: a future violation at level
    `k` outranks a present one at `k+1` only while the discount has not damped the
    former below the latter's one-level priority ratio. Equivalently
    `gamma**L >= 1 / a` — whole-episode damping must not fall below the ratio one
    level buys — which is the form that shows the trade, because ADR-081's
    contraction argument wants `gamma**L` small and its hierarchy argument wants
    it at least `1 / a`.

    **`L` is the half this fixture owns, and it is now read rather than asserted.**
    `_measured_training_horizon_steps` documents the chain: the longest training
    episode is 500 control steps, not the 199 that used to be hardcoded here under
    a docstring claiming the horizon was measured. `199` is `RULEBOOK-V5.1` §4.6's
    figure for the 1100 Waymo `train` records, and half the training mixture is
    procedurally generated.

    **The value of `gamma` is not this fixture's to pin, and neither is the
    criterion's verdict.** `gamma = 0.9982` and the A7 architecture were approved
    on 2026-09-09 and land with that change, which is where the verdict assertion
    — `break_even_steps > horizon_steps` — belongs, next to the value that makes
    it true. Asserting a discount here would either duplicate that decision or
    contradict it for as long as the two changes are unmerged, and asserting the
    verdict against a discount this fixture does not own would do both.

    So what is pinned is the requirement, which is a function of `priority_base`
    and the measured horizon alone: at `a = 2.5` over 500 control steps the
    criterion needs `gamma >= 0.998169`. Any change to `priority_base`, or a
    regenerated frozen index whose episodes run longer, moves that number and
    fails here — which is the protection the old formulation was meant to give and
    could not, comparing against a horizon that could not move. Note the margin is
    thin and is a property of the current frozen index rather than of the code:
    one 510-step scenario would raise the requirement above the approved value,
    and `horizon_steps == 500` below is what would say so.
    """

    priority_base = float(OmegaConf.load(CONF_DIR / "scalarization" / "default.yaml").priority_base)
    horizon_steps, episodes = _measured_training_horizon_steps()

    assert horizon_steps == 500
    assert len(episodes) == 2200

    # `conf/env/scenarionet.yaml` must keep letting an episode run to the end of
    # its scenario, or the measured horizon above would not be the real one.
    env_config = OmegaConf.load(CONF_DIR / "env" / "scenarionet.yaml")
    assert env_config.config.horizon is None
    assert int(env_config.episode_control.extra_steps_after_scenario) == 0

    # Exercised so that arms disagreeing on the discount fails here too, without
    # pinning which discount they agree on.
    assert 0.0 < _shipped_discount() <= 1.0

    # The requirement the measured horizon imposes, independent of the configured
    # discount: `gamma**L >= 1 / a` rearranged.
    required_gamma = math.exp(-math.log(priority_base) / horizon_steps)
    assert required_gamma == pytest.approx(0.998169, abs=1.0e-6)
    assert required_gamma**horizon_steps == pytest.approx(1.0 / priority_base)
