"""Checkpoint compatibility manifest for the frozen encoder contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV11


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used by the current contract."""


CHECKPOINT_MANIFEST_VERSION = "1"
CHECKPOINT_MANIFEST_FIELDS = (
    "observation_schema_version",
    "observation_schema_fingerprint",
    "observation_type",
    "flat_dim",
    "raw_token_count",
    "encoder_architecture_version",
    "encoder_type",
    "encoder_config",
    "features_dim",
    "share_features_extractor",
    "ppo_ortho_init",
    "algorithm",
    "sb3_version",
    "sb3_commit",
    "git_commit",
    "seed",
)


@dataclass(frozen=True)
class CheckpointManifest:
    """Immutable, JSON-serializable compatibility metadata."""

    observation_schema_version: str
    observation_schema_fingerprint: str
    observation_type: str
    flat_dim: int
    raw_token_count: int | None
    encoder_architecture_version: str
    encoder_type: str
    encoder_config: dict[str, Any]
    features_dim: int
    share_features_extractor: bool | None
    ppo_ortho_init: bool | None
    algorithm: str
    sb3_version: str
    sb3_commit: str
    git_commit: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied mapping suitable for canonical JSON encoding."""

        payload = {
            "manifest_version": CHECKPOINT_MANIFEST_VERSION,
            "observation_schema_version": self.observation_schema_version,
            "observation_schema_fingerprint": self.observation_schema_fingerprint,
            "observation_type": self.observation_type,
            "flat_dim": self.flat_dim,
            "raw_token_count": self.raw_token_count,
            "encoder_architecture_version": self.encoder_architecture_version,
            "encoder_type": self.encoder_type,
            "encoder_config": self.encoder_config,
            "features_dim": self.features_dim,
            "share_features_extractor": self.share_features_extractor,
            "ppo_ortho_init": self.ppo_ortho_init,
            "algorithm": self.algorithm,
            "sb3_version": self.sb3_version,
            "sb3_commit": self.sb3_commit,
            "git_commit": self.git_commit,
            "seed": self.seed,
        }
        return deepcopy(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckpointManifest":
        """Validate and construct a manifest read from JSON."""

        if not isinstance(payload, Mapping):
            raise CheckpointCompatibilityError("Checkpoint manifest must be a JSON object.")
        if payload.get("manifest_version") != CHECKPOINT_MANIFEST_VERSION:
            raise CheckpointCompatibilityError(
                "Incompatible checkpoint manifest field 'manifest_version': "
                f"checkpoint={payload.get('manifest_version')!r}, "
                f"current={CHECKPOINT_MANIFEST_VERSION!r}."
            )
        missing = [field for field in CHECKPOINT_MANIFEST_FIELDS if field not in payload]
        if missing:
            raise CheckpointCompatibilityError(
                f"Checkpoint manifest is incomplete; missing fields: {', '.join(missing)}."
            )
        try:
            manifest = cls(
                observation_schema_version=str(payload["observation_schema_version"]),
                observation_schema_fingerprint=str(payload["observation_schema_fingerprint"]),
                observation_type=str(payload["observation_type"]),
                flat_dim=int(payload["flat_dim"]),
                raw_token_count=(
                    None if payload["raw_token_count"] is None else int(payload["raw_token_count"])
                ),
                encoder_architecture_version=str(payload["encoder_architecture_version"]),
                encoder_type=str(payload["encoder_type"]),
                encoder_config=deepcopy(dict(payload["encoder_config"])),
                features_dim=int(payload["features_dim"]),
                share_features_extractor=(
                    None
                    if payload["share_features_extractor"] is None
                    else bool(payload["share_features_extractor"])
                ),
                ppo_ortho_init=(
                    None if payload["ppo_ortho_init"] is None else bool(payload["ppo_ortho_init"])
                ),
                algorithm=str(payload["algorithm"]),
                sb3_version=str(payload["sb3_version"]),
                sb3_commit=str(payload["sb3_commit"]),
                git_commit=str(payload["git_commit"]),
                seed=int(payload["seed"]),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"Checkpoint manifest contains invalid values: {exc}"
            ) from exc
        return manifest


def build_checkpoint_manifest(
    *,
    observation_type: str,
    flat_dim: int,
    raw_token_count: int | None,
    encoder_type: str,
    encoder_config: Mapping[str, Any],
    features_dim: int,
    share_features_extractor: bool | None,
    ppo_ortho_init: bool | None,
    algorithm: str,
    sb3_version: str,
    sb3_commit: str,
    git_commit: str,
    seed: int,
    observation_schema_version: str | None = None,
    observation_schema_fingerprint: str | None = None,
    encoder_architecture_version: str = "1.0-final",
) -> CheckpointManifest:
    """Build a manifest and infer the approved semantic schema identity."""

    normalized_type = str(observation_type)
    if normalized_type == "semantic_v2":
        schema = SemanticObservationSchemaV11()
        schema_version = (
            schema.version if observation_schema_version is None else observation_schema_version
        )
        fingerprint = (
            schema.fingerprint_sha256()
            if observation_schema_fingerprint is None
            else observation_schema_fingerprint
        )
    else:
        schema_version = (
            "not-applicable" if observation_schema_version is None else observation_schema_version
        )
        fingerprint = (
            "not-applicable"
            if observation_schema_fingerprint is None
            else observation_schema_fingerprint
        )

    return CheckpointManifest(
        observation_schema_version=str(schema_version),
        observation_schema_fingerprint=str(fingerprint),
        observation_type=normalized_type,
        flat_dim=int(flat_dim),
        raw_token_count=None if raw_token_count is None else int(raw_token_count),
        encoder_architecture_version=str(encoder_architecture_version),
        encoder_type=str(encoder_type),
        encoder_config=deepcopy(dict(encoder_config)),
        features_dim=int(features_dim),
        share_features_extractor=share_features_extractor,
        ppo_ortho_init=ppo_ortho_init,
        algorithm=str(algorithm),
        sb3_version=str(sb3_version),
        sb3_commit=str(sb3_commit),
        git_commit=str(git_commit),
        seed=int(seed),
    )


def assert_checkpoint_compatible(
    checkpoint: CheckpointManifest | Mapping[str, Any],
    current: CheckpointManifest | Mapping[str, Any],
) -> CheckpointManifest:
    """Compare every frozen compatibility field with an informative error."""

    checkpoint_manifest = (
        checkpoint
        if isinstance(checkpoint, CheckpointManifest)
        else CheckpointManifest.from_mapping(checkpoint)
    )
    current_manifest = (
        current
        if isinstance(current, CheckpointManifest)
        else CheckpointManifest.from_mapping(current)
    )
    checkpoint_values = checkpoint_manifest.to_dict()
    current_values = current_manifest.to_dict()
    for field in CHECKPOINT_MANIFEST_FIELDS:
        if checkpoint_values[field] != current_values[field]:
            raise CheckpointCompatibilityError(
                f"Incompatible checkpoint manifest field '{field}': "
                f"checkpoint={checkpoint_values[field]!r}, current={current_values[field]!r}."
            )
    return checkpoint_manifest
