from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _resolve_agent_policy(policy_name: str):
    name = policy_name.lower()
    if name in {"envinputpolicy", "env_input_policy", "env_input"}:
        from metadrive.policy.env_input_policy import EnvInputPolicy

        return EnvInputPolicy

    if name in {"expertpolicy", "expert_policy", "expert"}:
        from metadrive.policy.expert_policy import ExpertPolicy

        return ExpertPolicy

    if name in {"idmpolicy", "idm_policy", "idm"}:
        from metadrive.policy.idm_policy import IDMPolicy

        return IDMPolicy

    raise ValueError(
        f"Unsupported env policy '{policy_name}'. "
        "Supported values: env_input_policy, expert_policy, idm_policy"
    )


def _configure_agent_observation(
    env_cfg: dict[str, Any],
    observation_cfg: dict[str, Any],
) -> None:
    """Mutate env_cfg to install a selected observation class/config."""
    obs_type = str(observation_cfg.get("type", "lidar_state")).strip().lower()

    if obs_type in {"lidar", "lidar_state", "lidarstateobservation"}:
        # MetaDrive default when `agent_observation` is unset and image_observation=False.
        return

    if obs_type in {"semantic", "semantic_state", "semanticstateobservation"}:
        from thesis_rl.envs.observations.semantic_state import SemanticStateObservation

        semantic_cfg = dict(observation_cfg)
        semantic_cfg.pop("type", None)
        semantic_cfg.pop("name", None)
        if bool(semantic_cfg.get("expose_time_indexed_future_trajectory", False)):
            raise ValueError(
                "Causal observation contract forbids exposing future time-indexed trajectories."
            )
        if bool(semantic_cfg.get("expose_future_signal_phase", False)):
            raise ValueError("Causal observation contract forbids exposing future signal phases.")

        SemanticStateObservation.set_external_config(semantic_cfg)
        env_cfg["agent_observation"] = SemanticStateObservation

        if bool(semantic_cfg.get("disable_lidar_sensors", True)):
            vehicle_cfg = env_cfg.setdefault("vehicle_config", {})
            lidar_cfg = vehicle_cfg.setdefault("lidar", {})
            lidar_cfg["num_lasers"] = 0
            lidar_cfg["num_others"] = 0
            lidar_cfg["distance"] = 0

            side_cfg = vehicle_cfg.setdefault("side_detector", {})
            side_cfg["num_lasers"] = 0
            side_cfg["distance"] = 0

            lane_line_cfg = vehicle_cfg.setdefault("lane_line_detector", {})
            lane_line_cfg["num_lasers"] = 0
            lane_line_cfg["distance"] = 0
        return

    raise ValueError(
        f"Unsupported observation type '{obs_type}'. Supported values: lidar_state, semantic_state"
    )


def _validate_scenarionet_catalog_runtime(catalog: Any, *, split: str, data_directory: str) -> None:
    """Fail early when a catalog and ScenarioNet runtime view do not match."""

    from thesis_rl.scenarios.runtime_database import verify_runtime_mapping

    runtime = Path(data_directory).expanduser().resolve()
    if not (runtime / "dataset_summary.pkl").is_file():
        raise FileNotFoundError(
            "ScenarioNet runtime database is missing dataset_summary.pkl: "
            f"{runtime}. Set env.config.data_directory to the matching runtime view."
        )
    available = set(verify_runtime_mapping(runtime))
    expected = {
        Path(record.relative_path).name
        for record in _runtime_rulebook_records(catalog, split=split)
    }
    missing = sorted(expected.difference(available))
    unexpected = sorted(available.difference(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:3]}")
        raise ValueError(
            "ScenarioNet catalog/runtime mismatch for "
            f"split={split!r} at {runtime}: "
            + ", ".join(details)
            + ". Set env.config.data_directory to the runtime view built from this catalog."
        )


def _runtime_rulebook_records(catalog: Any, *, split: str | None = None) -> tuple[Any, ...]:
    """Return the only catalog records admissible to a v1.1 runtime view."""

    records = catalog.valid_records(split=split)
    non_eligible = [
        record.scenario_uid for record in records if record.rulebook_eligible is not True
    ]
    if non_eligible:
        scope = f" for split={split!r}" if split is not None else ""
        raise ValueError(
            "ScenarioNet runtime catalog requires rulebook_eligible=True"
            f"{scope}; offending records: {sorted(non_eligible)[:5]}"
        )
    return records


def _scenarionet_dataset_root(cfg_env: Any) -> Path | None:
    """Resolve the root owning the frozen catalog and runtime views."""

    configured = getattr(cfg_env, "dataset_root", None)
    value = configured or os.environ.get("SCENARIONET_DATA_ROOT")
    if value in (None, "", "null"):
        return None
    return Path(str(value)).expanduser().resolve()


def make_env(
    cfg_env: Any,
    *,
    scenario_provider: Any | None = None,
    catalog: Any | None = None,
):
    """Create a MetaDrive environment from Hydra env config."""

    # try to import metadrive, and raise a clear error if it's not installed
    try:
        from metadrive import MetaDriveEnv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "metadrive is not installed. Run `uv sync` before training/evaluation."
        ) from exc

    # Convert Hydra config to plain dict for env initialization
    env_cfg = _to_plain_dict(cfg_env.config)

    observation_cfg = _to_plain_dict(getattr(cfg_env, "observation", {}))
    if observation_cfg:
        _configure_agent_observation(env_cfg, observation_cfg)

    # Handle policy mode configuration if enabled
    policy_mode_cfg = _to_plain_dict(getattr(cfg_env, "policy_mode", {}))
    if policy_mode_cfg.get("enabled", False):
        env_cfg["action_check"] = bool(policy_mode_cfg.get("action_check", True))
        policy_name = str(policy_mode_cfg.get("agent_policy", "env_input_policy"))
        agent_policy_cls = _resolve_agent_policy(policy_name)
        env_cfg["agent_policy"] = agent_policy_cls
    elif isinstance(env_cfg.get("agent_policy"), str):
        # Allow direct config override like env_overrides["agent_policy"] = "expert_policy".
        env_cfg["agent_policy"] = _resolve_agent_policy(str(env_cfg["agent_policy"]))

    env_name = str(getattr(cfg_env, "name", "metadrive")).strip().lower()
    env_id = str(getattr(cfg_env, "env_id", "")).strip().lower()
    if env_name == "scenarionet" or env_id == "thesisscenarioenv":
        from thesis_rl.envs.thesis_scenario_env import ThesisScenarioEnv
        from thesis_rl.scenarios.catalog import read_scenario_catalog
        from thesis_rl.scenarios.provider import (
            ArmUniformScenarioProvider,
            FixedSequenceScenarioProvider,
            UniformScenarioProvider,
        )

        split = str(getattr(cfg_env, "split", "train"))
        provider_cfg = _to_plain_dict(getattr(cfg_env, "provider", {}))
        scenario_arm = provider_cfg.get("arm")
        episode_control = _to_plain_dict(getattr(cfg_env, "episode_control", {}))
        if "extra_steps_after_scenario" in episode_control:
            env_cfg["extra_steps_after_scenario"] = int(
                episode_control["extra_steps_after_scenario"]
            )
        dataset_root = _scenarionet_dataset_root(cfg_env)
        if not env_cfg.get("data_directory") and dataset_root is not None:
            env_cfg["data_directory"] = str(dataset_root / "runtime" / split)
        provider_worker_id = int(
            env_cfg.pop("provider_worker_index", env_cfg.get("worker_index", 0))
        )
        provider_worker_count = int(env_cfg.pop("provider_worker_count", 1))
        worker_id = provider_worker_id
        catalog_path = (
            getattr(cfg_env, "catalog_path", None)
            or os.environ.get("SCENARIONET_CATALOG_PATH")
            or (dataset_root / "catalog" / "scenario_catalog.parquet" if dataset_root else None)
        )
        if catalog is None:
            if catalog_path is None:
                raise ValueError(
                    "ScenarioNet requires env.dataset_root or SCENARIONET_DATA_ROOT. "
                    "It must contain catalog/scenario_catalog.parquet and runtime/<split>."
                )
            catalog = read_scenario_catalog(str(catalog_path))
        data_directory = env_cfg.get("data_directory")
        if not data_directory:
            raise ValueError(
                "ScenarioNet requires the runtime view matching its frozen catalog. "
                "Set env.dataset_root or env.config.data_directory."
            )
        _validate_scenarionet_catalog_runtime(
            catalog,
            split=split,
            data_directory=str(data_directory),
        )
        if catalog is not None and int(env_cfg.get("num_scenarios", -1)) <= 0:
            split_records = _runtime_rulebook_records(catalog, split=split)
            if not split_records:
                raise ValueError(f"ScenarioNet catalog has no valid records for split={split!r}")
            # MetaDrive's engine computes its seed modulus before the data
            # manager expands num_scenarios=-1. Resolve it before construction
            # so provider-selected runtime indices are not collapsed to zero.
            env_cfg["num_scenarios"] = len(split_records)
        if scenario_provider is None and catalog is not None:
            provider_kind = str(provider_cfg.get("kind", "uniform")).lower()
            start_index = int(env_cfg.get("start_scenario_index", 0))
            num_scenarios = int(env_cfg.get("num_scenarios", -1))
            records = tuple(
                record
                for record in _runtime_rulebook_records(catalog)
                if (
                    provider_worker_count > 1
                    and record.runtime_index is not None
                    and record.runtime_index % provider_worker_count == provider_worker_id
                )
                or (
                    provider_worker_count <= 1
                    and (
                        num_scenarios <= 0
                        or (
                            record.runtime_index is not None
                            and start_index <= record.runtime_index < start_index + num_scenarios
                        )
                    )
                )
            )
            if not records:
                raise ValueError(
                    "ScenarioNet catalog has no records in the configured worker range: "
                    f"start_scenario_index={start_index}, num_scenarios={num_scenarios}"
                )
            if provider_kind == "uniform":
                probabilities = provider_cfg.get("source_probability", {})
                scenario_provider = UniformScenarioProvider(
                    records,
                    global_seed=int(getattr(cfg_env, "global_seed", 0)),
                    source_probabilities={
                        "waymo": float(probabilities.get("waymo", 0.5)),
                        "pg": float(probabilities.get("pg", 0.5)),
                    },
                    strict=bool(provider_cfg.get("strict", True)),
                    allow_fallback=bool(provider_cfg.get("allow_fallback", False)),
                    default_arm=str(scenario_arm) if scenario_arm is not None else None,
                )
            elif provider_kind == "arm_uniform":
                if scenario_arm is not None:
                    raise ValueError("arm-uniform provider does not accept provider.arm")
                scenario_provider = ArmUniformScenarioProvider(
                    records,
                    global_seed=int(getattr(cfg_env, "global_seed", 0)),
                    strict=bool(provider_cfg.get("strict", True)),
                    allow_fallback=bool(provider_cfg.get("allow_fallback", False)),
                )
            elif provider_kind == "fixed_sequence":
                scenario_provider = FixedSequenceScenarioProvider(
                    tuple(
                        sorted(
                            (record for record in records if record.split == split),
                            key=lambda record: (
                                record.runtime_index is None,
                                record.runtime_index if record.runtime_index is not None else 0,
                            ),
                        )
                    ),
                    repeat=bool(provider_cfg.get("repeat", False)),
                    default_arm=str(scenario_arm) if scenario_arm is not None else None,
                )
            else:
                raise ValueError(f"Unsupported ScenarioNet provider kind: {provider_kind!r}")
        return ThesisScenarioEnv(
            env_cfg,
            scenario_provider=scenario_provider,
            catalog=catalog,
            split=split,
            worker_id=worker_id,
            scenario_arm=str(scenario_arm) if scenario_arm is not None else None,
        )

    return MetaDriveEnv(env_cfg)
