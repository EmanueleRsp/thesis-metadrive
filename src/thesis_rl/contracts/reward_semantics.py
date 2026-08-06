"""Checkpoint sidecar identity for rulebook and scalarization semantics."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


REWARD_SEMANTICS_VERSION = "1"


class RewardSemanticsCompatibilityError(ValueError):
    """Raised before learner loading when reward semantics are incompatible."""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_reward_semantics_identity(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build the exact run identity used by runtime checkpoint sidecars.

    Non-scalar monitor runs do not select a scalarizer and therefore do not
    receive a scalarization sidecar. v2 scalar runs fail closed when their
    identity is absent or differs.
    """

    reward = _mapping(config.get("reward"))
    behavior = str(reward.get("behavior", "")).lower()
    rulebook = _mapping(config.get("rulebook"))
    if behavior != "scalar_reward":
        return None

    scalarization = _mapping(config.get("scalarization"))
    if not scalarization:
        return None
    sigmoid = _mapping(scalarization.get("sigmoid"))
    legacy = _mapping(scalarization.get("legacy"))
    reward_compression = _mapping(scalarization.get("reward_compression"))
    rulebook_version = str(rulebook.get("version", "v1"))
    identity = {
        "sidecar_version": REWARD_SEMANTICS_VERSION,
        "reward_behavior": behavior,
        "rulebook": {
            "implementation_family": str(rulebook.get("implementation_family", rulebook_version)),
            "specification_id": str(rulebook.get("specification_id", "not-applicable")),
            "version": rulebook_version,
        },
        "scalarization": {
            "specification_id": str(scalarization.get("specification_id", "not-applicable")),
            "version": str(scalarization.get("version", "not-applicable")),
            "mode": str(scalarization.get("mode", "not-applicable")),
            "vector_schema_id": scalarization.get("vector_schema_id"),
            "priority_base": scalarization.get("priority_base"),
            "sigmoid_sharpness": scalarization.get("sigmoid_sharpness", sigmoid.get("sharpness")),
            "numerical_tolerance": scalarization.get("numerical_tolerance"),
            "native_environment_reward_weight": scalarization.get(
                "native_environment_reward_weight"
            ),
            "reward_compression_mode": reward_compression.get("mode", "none"),
            "legacy": {
                "vector_schema_id": legacy.get("vector_schema_id"),
                "rule_scales": legacy.get("rule_scales"),
                "source_path": legacy.get("source_path"),
                "source_sha256": legacy.get("source_sha256"),
                "source_commit": legacy.get("source_commit"),
            },
            "extensions": _mapping(scalarization.get("extensions")),
        },
    }
    return deepcopy(identity)


def reward_semantics_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the sidecar path for a checkpoint stem or `.zip` file."""

    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix == ".zip":
        checkpoint = checkpoint.with_suffix("")
    return checkpoint.with_name(f"{checkpoint.name}.reward_semantics.json")


def write_reward_semantics_sidecar(
    checkpoint_path: str | Path,
    identity: Mapping[str, Any],
) -> Path:
    """Atomically write a canonical JSON reward-semantics sidecar."""

    target = reward_semantics_sidecar_path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(identity), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def assert_reward_semantics_compatible(
    checkpoint_path: str | Path,
    expected_identity: Mapping[str, Any] | None,
) -> None:
    """Reject missing or changed scalarization identity before learner loading."""

    if expected_identity is None:
        return
    sidecar = reward_semantics_sidecar_path(checkpoint_path)
    if not sidecar.is_file():
        raise RewardSemanticsCompatibilityError(
            "Checkpoint reward semantics sidecar is missing; the checkpoint can only be used "
            "as transfer initialization in a new run: "
            f"{sidecar}"
        )
    try:
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RewardSemanticsCompatibilityError(
            f"Cannot read checkpoint reward semantics sidecar: {sidecar}"
        ) from exc
    if actual != dict(expected_identity):
        raise RewardSemanticsCompatibilityError(
            "Checkpoint reward semantics mismatch; start a new run or use model-only "
            f"transfer initialization. checkpoint={actual!r}, current={dict(expected_identity)!r}."
        )
