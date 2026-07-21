from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table


_CONSOLE = Console()


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float, np.floating)):
        value_float = float(value)
        if np.isnan(value_float):
            return "nan"
        return f"{value_float:.{digits}f}"
    return str(value)


def _config_value(cfg: Any, *path: str, default: Any = None) -> Any:
    current = cfg
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _display_name(value: Any, aliases: dict[str, str] | None = None) -> str:
    text = str(value).strip()
    return (aliases or {}).get(text.lower(), text)


def _run_setup_rows(cfg: Any) -> list[tuple[str, str]]:
    reward_behavior = str(_config_value(cfg, "reward", "behavior", default="off"))
    reward_type = str(_config_value(cfg, "reward", "type", default="native"))
    rulebook_enabled = reward_type == "rulebook" or reward_behavior == "monitor_only"
    rulebook_version = str(_config_value(cfg, "rulebook", "version", default="off"))
    rulebook = "off"
    if rulebook_enabled:
        rulebook = "monitor only" if reward_behavior == "monitor_only" else rulebook_version

    scalarization = "off"
    if reward_behavior == "scalar_reward":
        scalarization = str(_config_value(cfg, "scalarization", "mode", default="unknown"))

    curriculum = "off"
    if bool(_config_value(cfg, "curriculum", "enabled", default=False)):
        curriculum_kind = str(_config_value(cfg, "curriculum", "kind", default=""))
        curriculum = {"scenario_acl": "ACL", "staged": "staged"}.get(
            curriculum_kind, curriculum_kind or "on"
        )

    algorithm_cfg = _config_value(cfg, "agent", "planner", "algorithm", default={})
    transition_replay = _config_value(algorithm_cfg, "transition_replay", default={})
    n_steps = _config_value(transition_replay, "n_steps", default=None)
    if n_steps is None:
        n_steps = _config_value(algorithm_cfg, "n_steps", default="off")
    per_enabled = bool(
        _config_value(transition_replay, "enabled", default=False)
        and _config_value(transition_replay, "prioritized", default=False)
    )

    vectorized = bool(_config_value(cfg, "env", "vectorized", "enabled", default=False))
    train_envs = _config_value(cfg, "env", "vectorized", "num_envs", default=1)
    eval_workers = int(_config_value(cfg, "experiment", "eval_workers", default=1))
    test_workers = int(_config_value(cfg, "experiment", "test_workers", default=1))

    observation = _display_name(
        _config_value(cfg, "obs", "name", default="unknown"),
        {"semantic_v2": "SemanticState", "lidar_state": "LidarState"},
    )
    encoder = _display_name(
        _config_value(cfg, "agent", "planner", "encoder", "name", default="unknown"),
        {"latent_query_v2": "LQ"},
    )

    return [
        (
            "Dataset",
            _display_name(
                _config_value(cfg, "env", "dataset", default=None)
                or _config_value(cfg, "env", "name", default="unknown"),
                {"scenarionet": "ScenarioNet", "metadrive": "MetaDrive"},
            ),
        ),
        ("Curriculum", curriculum),
        ("Rulebook", rulebook),
        ("Scalarization function", scalarization),
        ("Observation", observation),
        ("Encoder", encoder),
        (
            "Decoder",
            str(_config_value(cfg, "agent", "planner", "decoder", "name", default="unknown")),
        ),
        ("n-steps", str(n_steps)),
        ("PER", "on" if per_enabled else "off"),
        ("Seed", str(int(_config_value(cfg, "seed", default=0)))),
        ("Vectorized training envs", str(train_envs) if vectorized else "off"),
        ("Evaluation workers", str(eval_workers) if eval_workers > 1 else "off"),
        ("Tests workers", str(test_workers) if test_workers > 1 else "off"),
    ]


def print_evaluation_summary(
    *,
    title: str,
    metrics: dict[str, Any],
    stage: str,
    global_step: int,
    episodes: int,
    base_seed: int | None,
    details_path: Path,
    checkpoint_path: str | None = None,
) -> None:
    if base_seed is None:
        scenario_seed_summary = "provider-selected"
    else:
        seed_end = int(base_seed) + max(int(episodes), 1) - 1
        scenario_seed_summary = f"{int(base_seed)}..{seed_end}"
    table = Table(title=title, expand=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    rows = [
        ("Stage", stage),
        ("Global step", str(int(global_step))),
        ("Episodes", str(int(episodes))),
        ("Scenario seeds", scenario_seed_summary),
        ("Mean reward", _format_metric(metrics.get("mean_reward"))),
        ("Std reward", _format_metric(metrics.get("std_reward"))),
        ("Success rate", _format_metric(metrics.get("success_rate"))),
        ("Collision rate", _format_metric(metrics.get("collision_rate"))),
        ("Out-of-road rate", _format_metric(metrics.get("out_of_road_rate"))),
        ("Route completion", _format_metric(metrics.get("route_completion"))),
        ("Top rule violation", _format_metric(metrics.get("top_rule_violation_rate"))),
        ("Avg error value", _format_metric(metrics.get("avg_error_value"))),
    ]
    if checkpoint_path is not None:
        rows.append(("Checkpoint", checkpoint_path))
    rows.append(("Full metrics", str(details_path)))

    for metric, value in rows:
        table.add_row(metric, value)
    _CONSOLE.print(table)


def print_run_setup(
    *,
    title: str,
    cfg: Any,
    metadata_path: Path,
    hydra_config_path: Path,
    checkpoint_path: str | None = None,
    extra_rows: list[tuple[str, str]] | None = None,
) -> None:
    table = Table(title=title, expand=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    rows = [
        ("Run dir", str(cfg.paths.run_dir)),
        ("Algorithm", str(cfg.agent.planner.algorithm.name)),
        *_run_setup_rows(cfg),
        ("Total timesteps", str(int(cfg.experiment.total_timesteps))),
        ("Eval interval", str(int(cfg.experiment.eval_interval))),
        ("Eval episodes", str(int(cfg.experiment.eval_episodes))),
    ]
    if checkpoint_path is not None:
        rows.insert(1, ("Checkpoint", checkpoint_path))
    if extra_rows:
        rows.extend(extra_rows)

    for field, value in rows:
        table.add_row(field, value)
    _CONSOLE.print(table)
