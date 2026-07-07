from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from thesis_rl.common.paths import default_analysis_root, default_outputs_root


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_default_output_helpers_use_portable_workspace_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OUTPUTS_ROOT", raising=False)
    monkeypatch.delenv("THESIS_OUTPUT_DIR", raising=False)

    assert default_outputs_root() == Path("/workspace/outputs")
    assert default_analysis_root() == Path("/workspace/outputs/analysis")


def test_hydra_paths_default_to_workspace_outputs(monkeypatch) -> None:
    monkeypatch.delenv("OUTPUTS_ROOT", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("SCENARIONET_DATA_ROOT", raising=False)
    monkeypatch.delenv("METADRIVE_DATA_ROOT", raising=False)

    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["reward=monitor_only", "curriculum=disabled"])

    assert str(cfg.paths.outputs_root) == "/workspace/outputs"
    assert str(cfg.paths.analysis_root) == "/workspace/outputs/analysis"
    assert str(cfg.paths.data_root) == "/workspace/data"
    assert str(cfg.paths.scenarionet_data_root) == "/workspace/data/scenarionet"
    assert str(cfg.paths.metadrive_data_root) == "/workspace/data/metadrive"
    assert str(cfg.paths.run_dir).startswith("/workspace/outputs/")


def test_output_helpers_prefer_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OUTPUTS_ROOT", "/tmp/custom-outputs")

    assert default_outputs_root() == Path("/tmp/custom-outputs")
    assert default_analysis_root() == Path("/tmp/custom-outputs/analysis")
