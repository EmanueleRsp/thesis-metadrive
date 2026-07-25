"""EVAL-PROTOCOL REQ-018 regression: the 6 `conf/presets/td3/*_curr*/*_no_curr.yaml`
curriculum-ablation presets must each declare a distinct `analysis.experiment_group`
so ablation runs are distinguishable in the run registry and separable from the
core `BASELINE-SCALAR-01`/`EXTENSION-ALGORITHM-01` comparison blocks."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

ABLATION_PRESETS = [
    "presets/td3/td3_native_curr",
    "presets/td3/td3_native_no_curr",
    "presets/td3/td3_monitor_only_curr",
    "presets/td3/td3_monitor_only_no_curr",
    "presets/td3/td3_scalar_reward_curr",
    "presets/td3/td3_scalar_reward_no_curr",
]


def _compose_preset(preset: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name=preset, overrides=["run_profile=smoke"])


def test_each_ablation_preset_declares_an_explicit_experiment_group() -> None:
    for preset in ABLATION_PRESETS:
        cfg = _compose_preset(preset)
        assert cfg.analysis.experiment_group, f"{preset} has no experiment_group"
        assert str(cfg.analysis.experiment_group).startswith("ABLATION-CURRICULUM-"), (
            f"{preset}'s experiment_group={cfg.analysis.experiment_group!r} does not "
            "use the ABLATION-CURRICULUM- prefix distinguishing it from "
            "BASELINE-SCALAR-01/EXTENSION-ALGORITHM-01"
        )


def test_all_six_ablation_presets_have_distinct_experiment_groups() -> None:
    groups = [str(_compose_preset(preset).analysis.experiment_group) for preset in ABLATION_PRESETS]
    assert len(groups) == len(set(groups)), f"experiment_group values are not all distinct: {groups}"


def test_experiment_groups_disjoint_from_core_comparison_block_ids() -> None:
    for preset in ABLATION_PRESETS:
        group = str(_compose_preset(preset).analysis.experiment_group)
        assert group not in {"BASELINE-SCALAR-01", "EXTENSION-ALGORITHM-01"}
