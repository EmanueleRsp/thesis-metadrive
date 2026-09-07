from __future__ import annotations

import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


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


def test_every_algorithm_shares_one_hierarchy_preserving_discount() -> None:
    """`AC-RB5.1-16` / ADR-079, amending ADR-075.

    Two properties, and the second is pinned as a *derivation* rather than as a
    literal, so that changing `priority_base` without revisiting `gamma` fails
    here instead of silently inverting the hierarchy mid-episode.

    1. **One discount for every arm.** Differing discounts across arms would make
       a difference in results non-attributable to the preference structure under
       test, which is the comparison this thesis exists to make.

    2. **The discount must not invert the hierarchy inside an episode.** A future
       violation at level `k` still outranks a present one at `k+1` while
       `Delta < ln(a) / -ln(gamma)`. The binding case is one level apart -- L1
       against L2, and identically L2 against L3 -- not the two-level case
       ADR-075 tabulated, whose figures are 2x too generous. The horizon is
       measured, not assumed: RULEBOOK-V5.1 §4.6 reports p5/p50/p95 =
       197/199/200 steps.

    `gamma = 1` satisfied (2) trivially and failed something ADR-075 did not
    weigh: with two thirds of episodes ending in a bootstrapped truncation there
    is no contraction and the value level is pinned only by the terminating
    minority. See ADR-079.
    """

    horizon_steps = 199
    priority_base = float(OmegaConf.load(CONF_DIR / "scalarization" / "default.yaml").priority_base)

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

    if gamma == 1.0:  # pragma: no cover - the undiscounted case needs no bound
        return
    break_even_steps = math.log(priority_base) / -math.log(gamma)
    assert break_even_steps > horizon_steps, (
        f"gamma={gamma} inverts the hierarchy after {break_even_steps:.0f} steps, inside the "
        f"{horizon_steps}-step episode: a violation one level down becomes preferable to a "
        f"higher-level one further away. Raise gamma above "
        f"{math.exp(-math.log(priority_base) / horizon_steps):.5f} or raise priority_base."
    )
