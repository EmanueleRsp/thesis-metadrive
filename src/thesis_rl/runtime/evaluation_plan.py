"""Frozen ScenarioNet multi-panel evaluation-plan resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig

from thesis_rl.scenarios.frozen import load_frozen_index
from thesis_rl.scenarios.panel_manifest import (
    NAMED_PANELS,
    PROFILE_SUBSET_SIZES,
    load_panel_manifest,
)
from thesis_rl.scenarios.runtime_database import sha256_file


FULL_PROFILES = frozenset({"default", "medium", "tune", "long", "thesis"})


@dataclass(frozen=True, slots=True)
class EvaluationPanel:
    """One immutable endpoint selected from a frozen ScenarioNet index."""

    name: str
    role: str
    split: str
    source: str
    scope: str
    episode_count: int
    manifest_path: Path
    panel_hash: str
    parent_panel_hash: str | None
    selection_hash: str

    @property
    def scenario_set(self) -> str:
        """Compatibility label retained by legacy CSV consumers."""
        return self.name

    def env_overrides(self) -> dict[str, Any]:
        """Select this exact UID order through the existing fixed provider."""
        return {
            "split": self.split,
            "provider": {
                "kind": "fixed_sequence",
                "panel_manifest_path": str(self.manifest_path),
                "repeat": False,
            },
        }

    def metadata(self) -> dict[str, str | None]:
        return {
            "panel_name": self.name,
            "evaluation_scope": self.scope,
            "panel_sha256": self.panel_hash,
            "parent_panel_sha256": self.parent_panel_hash,
            "frozen_selection_hash": self.selection_hash,
        }


def is_scenarionet(cfg: DictConfig) -> bool:
    return str(cfg.env.get("name", "")).lower() == "scenarionet"


def evaluation_scope_for_profile(profile: str) -> str:
    if profile in FULL_PROFILES:
        return "full"
    if profile in PROFILE_SUBSET_SIZES:
        return profile
    raise ValueError(f"ScenarioNet profile {profile!r} has no approved evaluation scope")


def resolve_scenarionet_evaluation_panels(
    cfg: DictConfig,
    *,
    final: bool,
) -> tuple[EvaluationPanel, ...]:
    """Resolve and validate the frozen named panel set for the active profile.

    This deliberately has no episode-count argument: official ScenarioNet
    cardinality is a property of the frozen dataset contract. It also validates
    the materialized artifact hash, preventing a stale or hand-edited panel
    file from silently changing an experiment.
    """
    if not is_scenarionet(cfg):
        return ()
    profile = str(cfg.run_profile.name)
    scope = evaluation_scope_for_profile(profile)
    configured_scope = cfg.get("evaluation", {}).get("scenarionet", {}).get("profiles", {}).get(
        profile, {}
    ).get("scope")
    if configured_scope not in (None, scope):
        raise ValueError(
            f"evaluation configuration scope {configured_scope!r} conflicts with profile {profile!r}"
        )
    data_root = Path(str(cfg.paths.scenarionet_data_root)).expanduser().resolve()
    configured_index = cfg.get("evaluation", {}).get("scenarionet", {}).get("frozen_index_path")
    index_path = (
        Path(str(configured_index)).expanduser().resolve()
        if configured_index not in (None, "", "null")
        else data_root / "frozen" / "scenario_selection_index.json"
    )
    index = load_frozen_index(index_path)
    key = "panel_manifests" if scope == "full" else "profile_panel_manifests"
    entries: Mapping[str, Any] = index.get(key, {})
    if scope != "full":
        entries = entries.get(scope, {}) if isinstance(entries, Mapping) else {}
    if not isinstance(entries, Mapping):
        raise ValueError(f"frozen index {key!r} does not contain a panel mapping for {scope!r}")
    names = (
        ("test_waymo_empirical", "test_pg", "test_arm_stratified")
        if final
        else ("validation_waymo_empirical", "validation_pg")
    )
    expected_sizes = (
        {name: int(specification["full_size"]) for name, specification in NAMED_PANELS.items()}
        if scope == "full"
        else PROFILE_SUBSET_SIZES[scope]
    )
    resolved: list[EvaluationPanel] = []
    for name in names:
        item = entries.get(name)
        if not isinstance(item, Mapping) or not isinstance(item.get("manifest"), Mapping):
            raise ValueError(f"frozen index is missing required {scope} panel {name!r}")
        raw = item["manifest"]
        artifact = item.get("artifact")
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("relative_path"), str):
            raise ValueError(f"frozen panel {name!r} has no materialized artifact identity")
        path = data_root / str(artifact["relative_path"])
        manifest = load_panel_manifest(path)
        recorded_file_hash = str(artifact.get("sha256", ""))
        if sha256_file(path) != recorded_file_hash:
            raise ValueError(f"materialized panel {name!r} differs from the frozen file hash")
        if manifest.sha256 != str(raw.get("sha256")):
            raise ValueError(f"materialized panel {name!r} differs from the frozen UID hash")
        if manifest.size != expected_sizes[name]:
            raise ValueError(
                f"panel {name!r} has {manifest.size} episodes, but profile {profile!r} "
                f"requires {expected_sizes[name]}"
            )
        if scope == "full":
            if manifest.scope != "full" or manifest.parent_panel is not None:
                raise ValueError(f"full panel {name!r} has an invalid parent/scope declaration")
        elif (
            manifest.scope != scope
            or manifest.parent_panel != name
            or manifest.parent_sha256 is None
        ):
            raise ValueError(f"diagnostic panel {name!r} does not identify its frozen parent")
        spec = NAMED_PANELS[name]
        if manifest.split != spec["split"] or manifest.source != spec["source"]:
            raise ValueError(f"panel {name!r} does not match its declared endpoint")
        role = (
            "primary"
            if name in {"validation_waymo_empirical", "test_waymo_empirical"}
            else "arm_claim"
            if name == "test_arm_stratified"
            else "secondary"
        )
        resolved.append(
            EvaluationPanel(
                name=name,
                role=role,
                split=manifest.split,
                source=manifest.source,
                scope=scope,
                episode_count=manifest.size,
                manifest_path=path,
                panel_hash=manifest.sha256,
                parent_panel_hash=manifest.parent_sha256,
                selection_hash=str(index["selection_hash"]),
            )
        )
    return tuple(resolved)


__all__ = [
    "EvaluationPanel",
    "FULL_PROFILES",
    "evaluation_scope_for_profile",
    "is_scenarionet",
    "resolve_scenarionet_evaluation_panels",
]
