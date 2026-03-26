from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_with_preset(preset_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=[f"preset_test={preset_name}"])


def test_preset_test_baseline_direct() -> None:
    cfg = _compose_with_preset("baseline_direct")

    assert cfg.adapter.name == "direct_action"
    assert cfg.experiment.name == "test_baseline_direct"
    assert bool(cfg.env.policy_mode.enabled) is False


def test_preset_test_neural_adapter() -> None:
    cfg = _compose_with_preset("neural_adapter")

    assert cfg.adapter.name == "neural_adapter"
    assert cfg.experiment.name == "test_neural_adapter"
    assert bool(cfg.env.policy_mode.enabled) is False


def test_preset_test_policy_env_side() -> None:
    cfg = _compose_with_preset("policy_env_side")

    assert cfg.adapter.name == "direct_action"
    assert cfg.experiment.name == "test_policy_env_side"
    assert bool(cfg.env.policy_mode.enabled) is True
    assert str(cfg.env.policy_mode.agent_policy) == "env_input_policy"


def test_preset_test_rulebook_hybrid_direct() -> None:
    cfg = _compose_with_preset("rulebook_hybrid_direct")

    assert cfg.adapter.name == "direct_action"
    assert cfg.reward.name == "hybrid_rulebook"
    assert cfg.rulebook.name == "baseline_v3"
    assert cfg.experiment.name == "test_rulebook_hybrid_direct"


def test_preset_test_rulebook_hybrid_calibration() -> None:
    cfg = _compose_with_preset("rulebook_hybrid_calibration")

    assert cfg.adapter.name == "direct_action"
    assert cfg.reward.name == "hybrid_rulebook"
    assert cfg.rulebook.name == "baseline_v3"
    assert cfg.experiment.name == "test_rulebook_hybrid_calibration"
    assert bool(cfg.reward.include_violation_vector) is True


def test_preset_test_curriculum_auto_smoke() -> None:
    cfg = _compose_with_preset("curriculum_auto_smoke")

    assert cfg.adapter.name == "direct_action"
    assert cfg.experiment.name == "test_curriculum_auto_smoke"
    assert bool(cfg.experiment.curriculum.enabled) is True
    assert str(cfg.experiment.curriculum.mode) == "auto"
    assert len(cfg.experiment.curriculum.stages) == 2


def test_preset_test_curriculum_auto_calibration() -> None:
    cfg = _compose_with_preset("curriculum_auto_calibration")

    assert cfg.adapter.name == "direct_action"
    assert cfg.experiment.name == "test_curriculum_auto_calibration"
    assert bool(cfg.experiment.curriculum.enabled) is True
    assert str(cfg.experiment.curriculum.mode) == "auto"
    assert int(cfg.experiment.curriculum.promotion.consecutive_evals) == 2
