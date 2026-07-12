"""Explicit factory for provider-driven ScenarioNet environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_rl.envs.thesis_scenario_env import ThesisScenarioEnv


def make_thesis_scenario_env(
    *,
    data_root: str | Path,
    split: str,
    worker_id: int,
    provider: Any,
    config: dict[str, Any],
    catalog: Any | None = None,
) -> ThesisScenarioEnv:
    """Construct one isolated environment over ``runtime/<split>``."""

    env_config = dict(config)
    env_config.setdefault(
        "data_directory", str(Path(data_root).expanduser().resolve() / "runtime" / split)
    )
    env_config["worker_index"] = int(worker_id)
    env_config.setdefault("num_workers", 1)
    return ThesisScenarioEnv(
        env_config,
        scenario_provider=provider,
        catalog=catalog,
        split=split,
        worker_id=worker_id,
    )


__all__ = ["make_thesis_scenario_env"]
