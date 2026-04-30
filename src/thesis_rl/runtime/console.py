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


def print_evaluation_summary(
    *,
    title: str,
    metrics: dict[str, Any],
    stage: str,
    global_step: int,
    episodes: int,
    base_seed: int,
    details_path: Path,
    checkpoint_path: str | None = None,
) -> None:
    seed_end = int(base_seed) + max(int(episodes), 1) - 1
    table = Table(title=title, expand=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    rows = [
        ("Stage", stage),
        ("Global step", str(int(global_step))),
        ("Episodes", str(int(episodes))),
        ("Scenario seeds", f"{int(base_seed)}..{seed_end}"),
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
        ("Algorithm", str(cfg.planner.name)),
        ("Reward", str(cfg.reward.mode)),
        ("Curriculum", str(cfg.curriculum.name)),
        ("Seed", str(int(cfg.seed))),
        ("Device", str(cfg.device)),
        ("Total timesteps", str(int(cfg.experiment.total_timesteps))),
        ("Eval interval", str(int(cfg.experiment.eval_interval))),
        ("Eval episodes", str(int(cfg.experiment.eval_episodes))),
        ("Metadata", str(metadata_path)),
        ("Hydra config", str(hydra_config_path)),
    ]
    if checkpoint_path is not None:
        rows.insert(1, ("Checkpoint", checkpoint_path))
    if extra_rows:
        rows.extend(extra_rows)

    for field, value in rows:
        table.add_row(field, value)
    _CONSOLE.print(table)
