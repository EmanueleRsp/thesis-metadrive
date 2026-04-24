from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_adapter_groups_compose() -> None:
    cfg_identity = _compose("adapter=identity", "reward=scalar_default")
    cfg_neural = _compose("adapter=neural_adapter", "reward=scalar_default")
    cfg_policy = _compose("adapter=policy_adapter", "reward=scalar_default")

    assert cfg_identity.adapter.name == "identity"
    assert cfg_neural.adapter.name == "neural_adapter"
    assert cfg_policy.adapter.name == "policy_adapter"


def test_curriculum_groups_compose() -> None:
    cfg_disabled = _compose("curriculum=disabled", "reward=scalar_default")
    cfg_stages = _compose("curriculum=stages", "reward=scalar_default")

    assert bool(cfg_disabled.curriculum.enabled) is False
    assert bool(cfg_stages.curriculum.enabled) is True
