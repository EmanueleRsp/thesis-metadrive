from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_base_config_composes_with_scalar_default() -> None:
    cfg = _compose("reward=scalar_default", "curriculum=disabled")

    assert cfg.env.name == "metadrive"
    assert cfg.preprocessor.name == "identity"
    assert cfg.adapter.name == "identity"
    assert cfg.reward.name == "scalar_default"
    assert bool(cfg.curriculum.enabled) is False


def test_rulebook_curriculum_config_composes() -> None:
    cfg = _compose("reward=rulebook", "curriculum=stages")

    assert cfg.reward.name == "rulebook"
    assert str(cfg.reward.mode) == "rulebook"
    assert str(cfg.reward.rulebook) == "selection"
    assert bool(cfg.curriculum.enabled) is True
    assert str(cfg.curriculum.mode) == "auto"
    assert len(cfg.curriculum.stages) >= 1
