from __future__ import annotations

from typing import Any

from thesis_rl.agent.planners.interfaces.planner import BasePlanner
from thesis_rl.agent.planners.backend_names import planner_backend_family, warn_if_legacy_backend
from thesis_rl.agent.planners.algorithms import (
    PpoPlannerBackend,
    SacPlannerBackend,
    Sb3PpoPlannerBackend,
    Sb3SacPlannerBackend,
    Sb3Td3PlannerBackend,
    Td3PlannerBackend,
)


def build_planner_backend(
    planner_name: str,
    env: Any,
    cfg_planner: Any,
    cfg_encoder: Any | None = None,
    cfg_decoder: Any | None = None,
    cfg_obs: Any | None = None,
    device: str = "auto",
    seed: int | None = None,
) -> BasePlanner:
    name = str(planner_name).lower()
    planner_backend_family(name)
    warn_if_legacy_backend(name)
    if name == "td3":
        return Td3PlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    if name == "td3_sb3":
        return Sb3Td3PlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    if name == "sac":
        return SacPlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    if name == "sac_sb3":
        return Sb3SacPlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    if name == "ppo":
        return PpoPlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    if name == "ppo_sb3":
        return Sb3PpoPlannerBackend.build(
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
            seed=seed,
        )
    raise ValueError(f"Unsupported planner backend: {planner_name}")


def load_planner_backend(
    planner_name: str,
    checkpoint_path: str,
    env: Any,
    cfg_planner: Any,
    cfg_encoder: Any | None = None,
    cfg_decoder: Any | None = None,
    cfg_obs: Any | None = None,
    device: str = "auto",
) -> BasePlanner:
    name = str(planner_name).lower()
    planner_backend_family(name)
    warn_if_legacy_backend(name)
    if name == "td3":
        return Td3PlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    if name == "td3_sb3":
        return Sb3Td3PlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    if name == "sac":
        return SacPlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    if name == "sac_sb3":
        return Sb3SacPlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    if name == "ppo":
        return PpoPlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    if name == "ppo_sb3":
        return Sb3PpoPlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            cfg_planner=cfg_planner,
            cfg_encoder=cfg_encoder,
            cfg_decoder=cfg_decoder,
            cfg_obs=cfg_obs,
            device=device,
        )
    raise ValueError(f"Unsupported planner backend: {planner_name}")
