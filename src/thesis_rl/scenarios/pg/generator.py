from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from thesis_rl.scenarios.pg.exporter import export_pg_scenario
from thesis_rl.scenarios.pg.profiles import GenerationSpec, get_pg_profile, validate_native_tokens
from thesis_rl.scenarios.pg.roundabout_priority import roundabout_priority_records
from thesis_rl.scenarios.pg.validation import PGValidationResult
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry


@dataclass(frozen=True, slots=True)
class PGGenerationResult:
    entry: ScenarioCatalogEntry
    validation: PGValidationResult
    generation_manifest: Path
    spec: GenerationSpec


def _available_native_tokens() -> tuple[str, ...]:
    from metadrive.component.algorithm.blocks_prob_dist import (  # type: ignore[import-not-found]
        PGBlockDistConfig,
    )
    from metadrive.utils.registry import get_metadrive_class  # type: ignore[import-not-found]

    return tuple(
        sorted(
            {
                str(getattr(get_metadrive_class(name), "ID"))
                for name in PGBlockDistConfig.all_blocks()
            }
        )
    )


def _build_env(spec: GenerationSpec):
    from metadrive.envs.metadrive_env import MetaDriveEnv  # type: ignore[import-not-found]
    from metadrive.policy.idm_policy import IDMPolicy  # type: ignore[import-not-found]

    map_sequence = "".join(spec.block_sequence)
    return MetaDriveEnv(
        {
            "map": map_sequence,
            "start_seed": spec.seed,
            "num_scenarios": 1,
            "agent_policy": IDMPolicy,
            "use_render": False,
            "log_level": 50,
            "traffic_density": spec.traffic_density,
            "accident_prob": spec.accident_prob,
            "random_lane_num": spec.random_lane_num,
            "random_lane_width": spec.random_lane_width,
            "horizon": spec.max_episode_length,
            "store_map": True,
        }
    )


def _realized_static_obstacle_metadata(env: Any) -> dict[str, Any]:
    """Return static accident-scene objects present in the generated MetaDrive episode."""
    get_objects = getattr(env.engine, "get_objects", None)
    objects = get_objects() if callable(get_objects) else {}
    values = objects.values() if isinstance(objects, dict) else ()
    static_types = tuple(
        sorted(
            {
                type(obj).__name__
                for obj in values
                if any(
                    token in type(obj).__name__.lower()
                    for token in ("trafficwarning", "trafficbarrier", "trafficcone")
                )
            }
        )
    )
    return {"realized": bool(static_types), "object_types": list(static_types)}


def generate_pg_scenario(
    profile: str,
    *,
    seed: int,
    data_root: str | Path,
    output_directory: str | Path | None = None,
    overwrite: bool = False,
    generator_commit: str | None = None,
    exporter_commit: str | None = None,
) -> PGGenerationResult:
    selected = get_pg_profile(profile)
    validate_native_tokens(selected, _available_native_tokens())
    spec = selected.sample(seed=seed)
    env = _build_env(spec)
    map_metadata: dict[str, Any] = {}
    try:
        # IDMPolicy is installed in the environment; export_scenarios still needs
        # a callable policy argument, whose action is ignored by the internal IDM.
        scenarios = env.export_scenarios(
            lambda _observation: np.zeros(2, dtype=np.float32),
            scenario_index=seed,
            max_episode_length=spec.max_episode_length,
            return_done_info=False,
            to_dict=True,
        )
        scenario = scenarios.get(seed)
        if scenario is None:
            raise RuntimeError(f"MetaDrive export did not return scenario seed {seed}")
        map_metadata = env.current_map.get_meta_data()
        map_metadata["static_obstacle"] = _realized_static_obstacle_metadata(env)
        roundabout_priorities = roundabout_priority_records(env.current_map)
        if roundabout_priorities:
            scenario.setdefault("metadata", {}).setdefault("rulebook_vehicle_yield", {})[
                "roundabout_priorities"
            ] = roundabout_priorities
        map_config = env.config.get("map_config", {})
        spec = replace(
            spec,
            lane_num=int(map_config.get("lane_num")) if map_config.get("lane_num") is not None else None,
            lane_width=float(map_config.get("lane_width"))
            if map_config.get("lane_width") is not None
            else None,
        )
    finally:
        env.close()

    root = Path(data_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve() if output_directory else root / "pg" / "database"
    entry, validation, manifest = export_pg_scenario(
        scenario,
        spec=spec,
        realized_map_metadata=map_metadata,
        data_root=root,
        output_directory=output,
        generator_commit=generator_commit,
        exporter_commit=exporter_commit,
        overwrite=overwrite,
    )
    return PGGenerationResult(entry, validation, manifest, spec)
