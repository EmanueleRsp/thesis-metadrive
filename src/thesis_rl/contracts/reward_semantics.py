"""Checkpoint sidecar identity for rulebook and scalarization semantics."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


REWARD_SEMANTICS_VERSION = "1"

# Recorded when a configuration declares no value for a rulebook provenance
# field. It is a visible sentinel rather than a substantive guess: the defect
# this replaces (`open_items` C8) was a missing key silently becoming a *wrong*
# family in the run's own provenance, which is worse than an absent one.
UNDECLARED_RULEBOOK_PROVENANCE = "not-declared"

RULEBOOK_PROVENANCE_FIELDS: tuple[str, ...] = (
    "implementation_family",
    "specification_id",
    "version",
)


class RewardSemanticsCompatibilityError(ValueError):
    """Raised before learner loading when reward semantics are incompatible."""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def rulebook_provenance(rulebook: Mapping[str, Any] | None) -> dict[str, str]:
    """Resolve the three rulebook provenance fields under one default.

    Both the run metadata writer and the checkpoint reward-semantics identity
    call this, because the same fact defaulted in two places is a fact that can
    drift -- and did.
    """

    declared = _mapping(rulebook)
    return {
        field: (
            UNDECLARED_RULEBOOK_PROVENANCE
            if declared.get(field) is None
            else str(declared.get(field))
        )
        for field in RULEBOOK_PROVENANCE_FIELDS
    }


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
    identity = {
        "sidecar_version": REWARD_SEMANTICS_VERSION,
        "reward_behavior": behavior,
        "rulebook": rulebook_provenance(rulebook),
        "scalarization": {
            "specification_id": str(scalarization.get("specification_id", "not-applicable")),
            "version": str(scalarization.get("version", "not-applicable")),
            "mode": str(scalarization.get("mode", "not-applicable")),
            "vector_schema_id": scalarization.get("vector_schema_id"),
            "priority_base": scalarization.get("priority_base"),
            # C29. The six-level weights enter the reward formula exactly as
            # `priority_base` does, and until ADR-081 every one of them was
            # frozen, so leaving them out of the identity cost nothing. `sigma`
            # is now a live parameter: without these fields a checkpoint trained
            # at `sigma = 0` would resume against `sigma = 0.30` without a
            # complaint, and the run would carry two different rewards in one
            # set of curves. Absent keys stay `None` rather than defaulting, so
            # a configuration that does not declare them is recorded as not
            # having declared them -- the `C8` rule.
            "severity": scalarization.get("severity"),
            "flat_tie_breaker": scalarization.get("flat_tie_breaker"),
            "progress_weight": scalarization.get("progress_weight"),
            "relaxable_weight": scalarization.get("relaxable_weight"),
            "progress_rate_weight": scalarization.get("progress_rate_weight"),
            "step_dt_s": scalarization.get("step_dt_s"),
            "reference_time_s": scalarization.get("reference_time_s"),
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
