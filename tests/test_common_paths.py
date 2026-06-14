from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from thesis_rl.common.paths import default_analysis_root, default_outputs_root


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_default_output_helpers_use_scratch_user(monkeypatch) -> None:
    monkeypatch.setenv("USER", "alice")

    assert default_outputs_root() == Path("/scratch/alice/thesis-metadrive/outputs")
    assert default_analysis_root() == Path("/scratch/alice/thesis-metadrive/outputs/analysis")


def test_hydra_paths_default_to_scratch(monkeypatch) -> None:
    monkeypatch.setenv("USER", "alice")

    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["reward=monitor_only", "curriculum=disabled"])

    assert str(cfg.paths.outputs_root) == "/scratch/alice/thesis-metadrive/outputs"
    assert str(cfg.paths.analysis_root) == "/scratch/alice/thesis-metadrive/outputs/analysis"
    assert str(cfg.paths.run_dir).startswith("/scratch/alice/thesis-metadrive/outputs/")
