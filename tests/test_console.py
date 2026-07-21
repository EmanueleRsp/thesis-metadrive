from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console
from omegaconf import OmegaConf

from thesis_rl.runtime.io import console


def _run_setup_config():
    return OmegaConf.create(
        {
            "paths": {"run_dir": "/tmp/run"},
            "agent": {
                "planner": {
                    "algorithm": {
                        "name": "sac_sb3",
                        "transition_replay": {"enabled": True, "n_steps": 3, "prioritized": True},
                    },
                    "encoder": {"name": "latent_query_v2"},
                    "decoder": {"name": "sac_sb3"},
                }
            },
            "env": {"name": "scenarionet", "vectorized": {"enabled": True, "num_envs": 5}},
            "obs": {"name": "semantic_v2"},
            "reward": {"type": "rulebook", "behavior": "scalar_reward"},
            "rulebook": {"version": "v4.7"},
            "scalarization": {"mode": "bounded_satisfaction_rank"},
            "curriculum": {"enabled": True, "kind": "scenario_acl"},
            "seed": 0,
            "experiment": {
                "total_timesteps": 3000,
                "eval_interval": 4000,
                "eval_episodes": 20,
                "eval_workers": 12,
                "test_workers": 8,
            },
        }
    )


def test_run_setup_displays_requested_configuration_fields(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    cfg = _run_setup_config()
    console.print_run_setup(
        title="Training Run",
        cfg=cfg,
        metadata_path=Path("metadata.yaml"),
        hydra_config_path=Path("config.yaml"),
    )

    rendered = output.getvalue()
    for field, value in (
        ("Run dir", "/tmp/run"),
        ("Algorithm", "sac_sb3"),
        ("Dataset", "ScenarioNet"),
        ("Curriculum", "ACL"),
        ("Rulebook", "v4.7"),
        ("Scalarization function", "bounded_satisfaction_rank"),
        ("Observation", "SemanticState"),
        ("Encoder", "LQ"),
        ("Decoder", "sac_sb3"),
        ("n-steps", "3"),
        ("PER", "on"),
        ("Seed", "0"),
        ("Vectorized training envs", "5"),
        ("Evaluation workers", "12"),
        ("Tests workers", "8"),
    ):
        assert field in rendered
        assert value in rendered

    for removed in (
        "Observation space",
        "Action space",
        "Metadata",
        "Hydra config",
        "Scenario ACL selection",
        "Device",
        "Reward type",
        "Reward behavior",
        "Rulebook config",
    ):
        assert removed not in rendered


def test_run_setup_marks_disabled_features_as_off(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    cfg = _run_setup_config()
    cfg.reward.type = "native"
    cfg.reward.behavior = "off"
    cfg.curriculum.enabled = False
    cfg.env.vectorized.enabled = False
    cfg.agent.planner.algorithm.transition_replay.enabled = False
    cfg.experiment.eval_workers = 1
    cfg.experiment.test_workers = 1
    console.print_run_setup(
        title="Training Run",
        cfg=cfg,
        metadata_path=Path("metadata.yaml"),
        hydra_config_path=Path("config.yaml"),
    )

    rendered = output.getvalue()
    assert "Rulebook" in rendered and "off" in rendered
    assert "Scalarization function" in rendered and "off" in rendered
    assert "Curriculum" in rendered and "off" in rendered
    assert "Vectorized training envs" in rendered and "off" in rendered
    assert "Evaluation workers" in rendered and "off" in rendered
    assert "Tests workers" in rendered and "off" in rendered
    assert "PER" in rendered and "off" in rendered


def test_evaluation_summary_supports_provider_selected_scenarios(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    console.print_evaluation_summary(
        title="Final Evaluation",
        metrics={"mean_reward": 1.0},
        stage="scenario_acl",
        global_step=100,
        episodes=2,
        base_seed=None,
        details_path=Path("events.jsonl"),
    )

    assert "Scenario seeds" in output.getvalue()
    assert "provider-selected" in output.getvalue()


def test_evaluation_summary_formats_sequential_scenarios(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    console.print_evaluation_summary(
        title="Evaluation",
        metrics={},
        stage="baseline",
        global_step=100,
        episodes=3,
        base_seed=42,
        details_path=Path("events.jsonl"),
    )

    assert "Scenario seeds" in output.getvalue()
    assert "42..44" in output.getvalue()
