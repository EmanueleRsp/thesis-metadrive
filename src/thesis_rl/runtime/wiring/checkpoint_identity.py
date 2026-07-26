"""Build the current-run checkpoint manifest for the sidecar compatibility check."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from thesis_rl.contracts.checkpoint_manifest import CheckpointManifest, build_checkpoint_manifest
from thesis_rl.contracts.observation_schema import (
    SemanticObservationSchemaV11,
    SemanticObservationSchemaV12,
)
from thesis_rl.contracts.reward_semantics import build_reward_semantics_identity
from thesis_rl.runtime.io.metadata import get_dependency_versions, get_git_commit

_RAW_TOKEN_COUNT_BY_OBSERVATION_TYPE = {
    "semantic_v2": SemanticObservationSchemaV11.raw_token_count,
    "semantic_v3": SemanticObservationSchemaV12.raw_token_count,
}

_SHARE_FEATURES_EXTRACTOR_BY_BACKEND = {
    "td3_sb3": False,
    "sac_sb3": False,
    "ppo_sb3": True,
}


def build_current_checkpoint_manifest(cfg: DictConfig, env: Any) -> CheckpointManifest:
    """Build a manifest for the checkpoint about to be saved or resumed.

    Pure function of ``cfg`` and an already-constructed ``env``; requires no
    encoder/planner instantiation, so it can be called identically at save
    time (``train_loop.py``) and at load time (``builders.py::load_planner``).
    """

    obs_type = str(cfg.obs.type).strip().lower() if cfg.get("obs") is not None else "lidar_state"
    encoder_cfg = cfg.agent.planner.encoder
    encoder_type = str(encoder_cfg.get("type", "none")).strip().lower()
    backend_name = str(cfg.agent.planner.algorithm.name).strip().lower()

    flat_dim = int(env.observation_space.shape[0])
    features_dim = (
        int(encoder_cfg.get("output_dim", flat_dim)) if encoder_type != "none" else flat_dim
    )

    share_features_extractor: bool | None = None
    ppo_ortho_init: bool | None = None
    if encoder_type != "none":
        share_features_extractor = _SHARE_FEATURES_EXTRACTOR_BY_BACKEND.get(backend_name)
        if backend_name == "ppo_sb3":
            ppo_ortho_init = False

    reward_identity = build_reward_semantics_identity(cfg)
    rulebook_kwargs: dict[str, Any] = {}
    scalarization_kwargs: dict[str, Any] = {}
    if reward_identity is not None:
        rulebook = reward_identity["rulebook"]
        scalarization = reward_identity["scalarization"]
        legacy = scalarization["legacy"]
        rulebook_kwargs = {
            "rulebook_implementation_family": rulebook["implementation_family"],
            "rulebook_specification_id": rulebook["specification_id"],
            "rulebook_version": rulebook["version"],
        }
        scalarization_kwargs = {
            "scalarization_specification_id": scalarization["specification_id"],
            "scalarization_version": scalarization["version"],
            "scalarization_mode": scalarization["mode"],
            "scalarization_vector_schema_id": scalarization["vector_schema_id"],
            "scalarization_priority_base": scalarization["priority_base"],
            "scalarization_sigmoid_sharpness": scalarization["sigmoid_sharpness"],
            "scalarization_numerical_tolerance": scalarization["numerical_tolerance"],
            "native_environment_reward_weight": scalarization["native_environment_reward_weight"],
            "legacy_vector_schema_id": legacy["vector_schema_id"],
            "legacy_rule_scales": legacy["rule_scales"],
            "legacy_scale_source_path": legacy["source_path"],
            "legacy_scale_source_sha256": legacy["source_sha256"],
            "legacy_scale_source_commit": legacy["source_commit"],
        }

    dependency_versions = get_dependency_versions()

    return build_checkpoint_manifest(
        observation_type=obs_type,
        flat_dim=flat_dim,
        raw_token_count=_RAW_TOKEN_COUNT_BY_OBSERVATION_TYPE.get(obs_type),
        encoder_type=encoder_type,
        encoder_config=OmegaConf.to_container(encoder_cfg, resolve=True),
        encoder_architecture_version=str(encoder_cfg.get("architecture_version", "not-applicable")),
        features_dim=features_dim,
        share_features_extractor=share_features_extractor,
        ppo_ortho_init=ppo_ortho_init,
        algorithm=backend_name,
        sb3_version=str(dependency_versions.get("sb3_version") or "unknown"),
        sb3_commit=str(dependency_versions.get("sb3_commit") or "unknown"),
        git_commit=get_git_commit(),
        seed=int(cfg.seed),
        **rulebook_kwargs,
        **scalarization_kwargs,
    )
