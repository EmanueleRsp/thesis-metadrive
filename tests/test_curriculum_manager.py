from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _passing_metrics() -> dict[str, float]:
    return {
        "collision_rate": 0.01,
        "top_rule_violation_rate": 0.01,
        "out_of_road_rate": 0.01,
        "success_rate": 0.9,
        "route_completion": 0.9,
        "success_rate_std": 0.01,
        "collision_rate_std": 0.01,
    }


def test_curriculum_config_parses_from_curriculum_group() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["curriculum=disabled", "reward=scalar_default"])

    curriculum = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)

    assert curriculum.enabled is False
    assert curriculum.mode == "fixed"
    assert len(curriculum.stages) == 0


def test_curriculum_manager_fixed_mode_uses_selected_stage_and_no_promotion() -> None:
    config = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "fixed",
            "fixed_stage": "stage2",
            "stages": [
                {"name": "stage1", "env": {"map": "S"}},
                {"name": "stage2", "env": {"map": 3}},
            ],
        }
    )
    manager = CurriculumManager(config)

    assert manager.get_current_stage().name == "stage2"
    manager.record_train_steps(100000)
    manager.record_eval_metrics(_passing_metrics())
    assert manager.should_promote() is False
    assert manager.promote() is False


def test_curriculum_manager_auto_promotion_requires_warmup_steps_and_consecutive_evals() -> None:
    config = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "auto",
            "stages": [
                {"name": "stage1", "env": {"map": "S"}},
                {"name": "stage2", "env": {"map": 3}},
            ],
            "promotion": {
                "consecutive_evals": 2,
                "warmup_evals": 1,
                "default_min_stage_steps": 10,
            },
        }
    )
    manager = CurriculumManager(config)

    manager.record_train_steps(10)
    manager.record_eval_metrics(_passing_metrics())
    assert manager.should_promote() is False

    manager.record_eval_metrics(_passing_metrics())
    assert manager.should_promote() is True
    assert manager.promote() is True
    assert manager.get_current_stage().name == "stage2"


def test_curriculum_manager_resets_consecutive_counter_on_failed_gate() -> None:
    config = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "auto",
            "stages": [
                {"name": "stage1", "env": {"map": "S"}},
                {"name": "stage2", "env": {"map": 3}},
            ],
            "promotion": {
                "consecutive_evals": 2,
                "warmup_evals": 0,
                "default_min_stage_steps": 0,
                "per_stage": {"stage1": {"min_stage_steps": 0}},
            },
        }
    )
    manager = CurriculumManager(config)

    manager.record_eval_metrics(_passing_metrics())
    assert manager.consecutive_passes == 1

    failed = _passing_metrics()
    failed["collision_rate"] = 0.5
    manager.record_eval_metrics(failed)
    assert manager.consecutive_passes == 0
    assert manager.should_promote() is False


def test_curriculum_manager_eval_env_overrides_train_env() -> None:
    config = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "auto",
            "stages": [
                {
                    "name": "stage4",
                    "env": {"start_seed": 10000, "num_scenarios": 1000},
                    "eval_env": {"start_seed": 20000, "num_scenarios": 300},
                }
            ],
            "promotion": {"default_min_stage_steps": 1},
        }
    )
    manager = CurriculumManager(config)

    train_env = manager.get_env_config(evaluation=False)
    eval_env = manager.get_env_config(evaluation=True)

    assert train_env["start_seed"] == 10000
    assert eval_env["start_seed"] == 20000
    assert eval_env["num_scenarios"] == 300
