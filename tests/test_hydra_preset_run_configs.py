from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_with_preset(preset_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=[f"preset_run={preset_name}"])


def test_preset_run_td3_rulebook_curriculum_baseline() -> None:
    cfg = _compose_with_preset("td3_rulebook_curriculum_baseline")

    assert cfg.env.name == "metadrive_base"
    assert cfg.agent.name == "td3"
    assert cfg.planner.name == "td3_mlp"
    assert cfg.preprocessor.name == "identity"
    assert cfg.adapter.name == "direct_action"
    assert cfg.reward.name == "hybrid_rulebook"
    assert cfg.rulebook.name == "baseline"
    assert cfg.experiment.name == "rulebook_curriculum_auto"
    assert bool(cfg.experiment.curriculum.enabled) is True
    assert str(cfg.experiment.curriculum.mode) == "auto"
