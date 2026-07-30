from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from thesis_rl.runtime.wiring.builders import _resolve_planner_cfg

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_off_policy_algorithms_share_buffer_size_on_default_profile() -> None:
    td3_cfg = _compose("agent/planner/algorithm=td3_sb3")
    sac_cfg = _compose("agent/planner/algorithm=sac_sb3")

    assert int(_resolve_planner_cfg(td3_cfg).buffer_size) == 300000
    assert int(_resolve_planner_cfg(sac_cfg).buffer_size) == 300000


def test_fast_profile_overrides_buffer_size_to_24000_for_both_algorithms() -> None:
    for algorithm in ("td3_sb3", "sac_sb3"):
        cfg = _compose("run_profile=fast", f"agent/planner/algorithm={algorithm}")
        resolved = _resolve_planner_cfg(cfg)
        assert int(resolved.buffer_size) == 24000


def test_thesis_profile_keeps_the_unified_300000_buffer() -> None:
    for algorithm in ("td3_sb3", "sac_sb3"):
        cfg = _compose("run_profile=thesis", f"agent/planner/algorithm={algorithm}")
        resolved = _resolve_planner_cfg(cfg)
        assert int(resolved.buffer_size) == 300000
