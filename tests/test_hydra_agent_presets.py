from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_preset(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name=config_name)


def test_td3_none_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_none")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.encoder.type) == "none"
    assert str(cfg.decoder.name) == "mlp_large"
    assert str(cfg.planner.name) == "td3"


def test_sac_lq_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_lq")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.encoder.type) == "lq"
    assert str(cfg.decoder.name) == "mlp_encoded"
    assert str(cfg.planner.name) == "sac"


def test_ppo_lq_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_lq")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.encoder.type) == "lq"
    assert str(cfg.decoder.name) == "mlp_encoded"
    assert str(cfg.planner.name) == "ppo"
