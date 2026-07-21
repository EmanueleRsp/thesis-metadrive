from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_training_monitor_refreshes_every_200_steps_for_all_run_profiles() -> None:
    for profile in ("default", "fast", "long", "medium", "smoke", "thesis", "tune"):
        with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
            cfg = compose(config_name="config", overrides=[f"run_profile={profile}"])
        assert int(cfg.experiment.log_interval) == 200
