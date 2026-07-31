"""EVAL-PROTOCOL v1.0 REQ-004/DEC-005: frozen, hashed, deduplicated evaluation
scenario panels.

A "panel" is the exact ordered UID sequence a validation or final-test
evaluation must walk through `FixedSequenceScenarioProvider`. Per `DEC-005`,
this ordered sequence is fixed before official training begins, is identical
across every algorithm/extension/training seed within a comparison block, and
is never redrawn at runtime (`DEC-004`'s no-backfill policy: a data-abort
episode is excluded from aggregates, not replaced).

This module builds such a panel via a deterministic seed-driven **balanced
draw across the six scenario arms** (`A0`..`A5`, see
`thesis_rl.scenarios.arms.ARMS`) and persists it as a versioned JSON artifact
with its SHA-256 content hash, so every run in a comparison block can load and
verify the identical frozen panel rather than re-deriving order at each run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_rl.scenarios.arms import ARMS

PANEL_MANIFEST_SCHEMA_VERSION = "v1"
DRAW_POLICIES = frozenset({"arm_balanced", "empirical"})
PANEL_SOURCES = frozenset({"waymo", "pg", "combined"})


def _uid_sequence_hash(scenario_uids: tuple[str, ...]) -> str:
    """SHA-256 over the ordered, newline-joined UID sequence.

    Order-sensitive by design: `REQ-004` freezes both scenario identity and
    order, so two panels with the same UID set in a different order must
    hash differently.
    """
    payload = "\n".join(scenario_uids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PanelManifest:
    """A frozen, hashed, deduplicated evaluation scenario panel.

    Attributes:
        schema_version: manifest format version.
        split: `"validation"` or `"test"`.
        seed: the deterministic draw seed (not a training seed).
        size: requested panel size (``len(scenario_uids)`` unless truncated
            by insufficient candidates, which is a fatal error at build time,
            never a silent truncation).
        arms: the ordered arm labels the draw balanced across.
        per_arm_counts: number of scenarios drawn from each arm, in `arms`
            order.
        scenario_uids: the frozen, ordered, deduplicated UID sequence.
        sha256: `_uid_sequence_hash(scenario_uids)`, persisted for
            self-consistency verification independent of the JSON file's own
            integrity.
        tracked_subset_uids: the frozen, ordered tracked-subset `scenario_uid`s
            (REQ-014/DEC-014, amended 2026-07-25) computed once at build time
            via feature-diversity greedy selection within each arm's already-
            drawn panel candidates (see `build_balanced_panel`'s
            `tracked_subset_count_per_arm`/`feature_lookup`). Empty for
            manifests built before this field existed or without a catalog
            feature lookup. A subset of `scenario_uids` by construction --
            never a separate draw (`DEC-004`/`DEC-005`: no redraws).
        draw_policy: `"arm_balanced"` (v1.0 behavior, `build_balanced_panel`:
            arm labels drive selection) or `"empirical"`
            (`SCENARIONET-INTEGRATION` v1.2 / `EVAL-PROTOCOL` v1.1
            `REQ-004`, `build_empirical_panel`: a uniform, label-blind draw
            over the panel's source pool; `arms`/`per_arm_counts` are then
            the *observed* distribution, reporting only). Defaults to
            `"arm_balanced"` for manifests built before this field existed.
        source: `"waymo"`, `"pg"`, or `"combined"` -- which source population
            this panel restricts to. Defaults to `"combined"` for manifests
            built before this field existed (the v1.0 balanced panels always
            drew from both sources within each arm).
    """

    schema_version: str
    split: str
    seed: int
    size: int
    arms: tuple[str, ...]
    per_arm_counts: tuple[int, ...]
    scenario_uids: tuple[str, ...]
    sha256: str
    tracked_subset_uids: tuple[str, ...] = ()
    draw_policy: str = "arm_balanced"
    source: str = "combined"
    parent_panel: str | None = None
    parent_sha256: str | None = None
    scope: str = "full"

    def verify_self_consistent(self) -> None:
        """Fail closed if the persisted hash does not match the UID list."""
        expected = _uid_sequence_hash(self.scenario_uids)
        if expected != self.sha256:
            raise ValueError(
                "Panel manifest is internally inconsistent: recorded sha256="
                f"{self.sha256!r} does not match the hash of its own "
                f"scenario_uids ({expected!r}). Refusing to use a corrupted "
                "or hand-edited panel manifest (EVAL-PROTOCOL REQ-004: no "
                "fallback sampling)."
            )
        if len(self.scenario_uids) != len(set(self.scenario_uids)):
            raise ValueError(
                "Panel manifest contains duplicate scenario_uids; REQ-004 "
                "requires a deduplicated panel."
            )
        if sum(self.per_arm_counts) != len(self.scenario_uids):
            raise ValueError(
                "Panel manifest per_arm_counts do not sum to the number of scenario_uids."
            )
        if not set(self.tracked_subset_uids).issubset(set(self.scenario_uids)):
            raise ValueError(
                "Panel manifest tracked_subset_uids must be a subset of "
                "scenario_uids (the tracked subset is a labeled subset of "
                "the frozen panel, never a separate draw)."
            )
        if len(self.tracked_subset_uids) != len(set(self.tracked_subset_uids)):
            raise ValueError("Panel manifest tracked_subset_uids contains duplicates.")
        if self.draw_policy not in DRAW_POLICIES:
            raise ValueError(f"unsupported panel draw_policy: {self.draw_policy!r}")
        if self.source not in PANEL_SOURCES:
            raise ValueError(f"unsupported panel source: {self.source!r}")
        if self.scope not in {"full", "smoke", "fast"}:
            raise ValueError(f"unsupported evaluation scope: {self.scope!r}")
        if (self.parent_panel is None) != (self.parent_sha256 is None):
            raise ValueError(
                "profile-subset manifests must record both parent_panel and parent_sha256"
            )
        if self.scope == "full" and self.parent_panel is not None:
            raise ValueError("a full panel must not declare a parent panel")
        if self.scope != "full" and self.parent_panel is None:
            raise ValueError("a diagnostic subset must declare its frozen parent panel")


_TRACKED_SUBSET_FEATURE_KEYS = (
    "has_intersection",
    "has_merge_or_roundabout",
    "has_route_traffic_light",
    "has_route_stop_sign",
    "has_route_crosswalk",
    "has_vehicle",
    "has_pedestrian",
    "has_cyclist",
    "low_traffic",
    "dense_traffic",
    "vru_interaction",
    "topology_tag",
)


def _feature_tuple(uid: str, feature_lookup: dict[str, dict[str, Any]] | None) -> tuple[Any, ...]:
    if feature_lookup is None:
        return ()
    features = feature_lookup.get(uid, {})
    return tuple(features.get(key) for key in _TRACKED_SUBSET_FEATURE_KEYS)


def _select_diverse_uids(
    candidates: tuple[str, ...],
    *,
    count: int,
    feature_lookup: dict[str, dict[str, Any]] | None,
) -> tuple[str, ...]:
    """Deterministic greedy feature-diversity selection of ``count`` UIDs out
    of ``candidates`` (an arm's already-drawn, order-fixed panel picks).

    Iterates ``candidates`` in their given (already-shuffled, deterministic)
    order and prefers the next UID whose feature-tuple over
    `_TRACKED_SUBSET_FEATURE_KEYS` (`has_intersection`,
    `has_merge_or_roundabout`, `has_route_traffic_light`,
    `has_route_stop_sign`, `has_route_crosswalk`, `has_vehicle`,
    `has_pedestrian`, `has_cyclist`, `low_traffic`, `dense_traffic`,
    `vru_interaction`, `topology_tag`) has not already been selected, so the
    tracked subset covers as many distinct scenario-feature combinations as
    the arm's drawn candidates contain (traffic lights, intersections,
    roundabouts, VRUs, dense/light traffic, simpler/more complex scenarios,
    ...), rather than the arbitrary first ``count`` in shuffle order. Once
    every distinct feature-tuple present among ``candidates`` is covered,
    remaining slots (if ``count`` exceeds the number of distinct
    combinations) are filled from the untaken candidates in the same
    deterministic order. No randomness beyond the already-fixed
    ``candidates`` order; no reach-back into the full catalog. With
    ``feature_lookup=None`` (e.g. records without catalog feature columns),
    falls back to plain prefix order, matching the pre-diversity behavior.
    """
    count = max(0, min(count, len(candidates)))
    if count == 0:
        return ()
    if feature_lookup is None:
        return tuple(candidates[:count])

    seen_tuples: set[tuple[Any, ...]] = set()
    selected: list[str] = []
    remaining: list[str] = []
    for uid in candidates:
        feature_tuple = _feature_tuple(uid, feature_lookup)
        if feature_tuple not in seen_tuples:
            seen_tuples.add(feature_tuple)
            selected.append(uid)
            if len(selected) == count:
                return tuple(selected)
        else:
            remaining.append(uid)
    for uid in remaining:
        selected.append(uid)
        if len(selected) == count:
            break
    return tuple(selected)


def build_balanced_panel(
    records: Any,
    *,
    split: str,
    size: int,
    seed: int,
    arms: tuple[str, ...] = ARMS,
    tracked_subset_count_per_arm: int = 0,
    feature_lookup: dict[str, dict[str, Any]] | None = None,
) -> PanelManifest:
    """Deterministically draw a balanced panel across ``arms``.

    For each arm, in ``arms`` order: candidate scenario UIDs (records whose
    ``primary_arm`` matches) are sorted ascending by ``scenario_uid`` for a
    catalog-order-independent baseline, then deterministically shuffled with
    a seed derived from ``(seed, split, arm_index)`` (never reusing the
    parent seed's raw state across arms, and never depending on training
    seeds), and the first ``per_arm_count`` UIDs are taken. Panel sizes are
    split as evenly as possible: ``size // len(arms)`` per arm, with the
    remainder (``size % len(arms)``) distributed one extra draw to the first
    arms in ``arms`` order. The final panel order concatenates arms in
    ``arms`` order, each arm's picks in shuffle order.

    Fails closed (raises ``ValueError``) if any arm has fewer eligible
    candidates than its allotted count: REQ-004 permits no fallback sampling
    or silent truncation.

    ``tracked_subset_count_per_arm`` (REQ-014/DEC-014, amended 2026-07-25):
    when positive, also computes ``tracked_subset_uids`` -- up to that many
    UIDs per arm, selected via `_select_diverse_uids` out of the arm's own
    just-drawn panel picks (never a separate draw from the full catalog).
    ``feature_lookup`` (``scenario_uid`` -> flat feature dict, typically
    `ScenarioCatalogEntry.features.to_dict()`) drives the diversity
    criterion; without it, the tracked subset falls back to prefix order.
    """
    if int(size) <= 0:
        raise ValueError("Panel size must be positive.")
    if not arms:
        raise ValueError("At least one scenario arm is required.")

    by_arm: dict[str, list[str]] = {arm: [] for arm in arms}
    for record in records:
        arm = getattr(record, "primary_arm", None)
        if arm in by_arm:
            by_arm[arm].append(str(record.scenario_uid))

    n_arms = len(arms)
    base_count = int(size) // n_arms
    remainder = int(size) % n_arms

    per_arm_counts: list[int] = []
    panel_uids: list[str] = []
    tracked_uids: list[str] = []
    for arm_index, arm in enumerate(arms):
        count = base_count + (1 if arm_index < remainder else 0)
        per_arm_counts.append(count)
        candidates = sorted(set(by_arm.get(arm, [])))
        if len(candidates) < count:
            raise ValueError(
                f"Panel build for split={split!r} requires {count} scenarios "
                f"from arm {arm!r} but only {len(candidates)} eligible "
                "candidates exist. No fallback/backfill is permitted "
                "(EVAL-PROTOCOL REQ-004/DEC-004)."
            )
        # Derive a per-arm seed from (seed, split, arm_index) so the draw is
        # deterministic, arm-order-independent in its randomness source, and
        # never coupled to any training seed.
        arm_seed_material = f"{seed}:{split}:{arm_index}:{arm}".encode("utf-8")
        arm_seed = int.from_bytes(hashlib.sha256(arm_seed_material).digest()[:8], "big")
        rng = np.random.default_rng(arm_seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        arm_panel_uids = shuffled[:count]
        panel_uids.extend(arm_panel_uids)
        if tracked_subset_count_per_arm > 0:
            tracked_uids.extend(
                _select_diverse_uids(
                    tuple(arm_panel_uids),
                    count=tracked_subset_count_per_arm,
                    feature_lookup=feature_lookup,
                )
            )

    if len(panel_uids) != len(set(panel_uids)):
        # Cannot happen given per-arm candidates are disjoint by construction
        # (one arm per record), but verified defensively per REQ-004's
        # deduplication invariant.
        raise ValueError("Balanced panel draw produced duplicate scenario_uids.")

    return PanelManifest(
        schema_version=PANEL_MANIFEST_SCHEMA_VERSION,
        split=str(split),
        seed=int(seed),
        size=int(size),
        arms=tuple(arms),
        per_arm_counts=tuple(per_arm_counts),
        scenario_uids=tuple(panel_uids),
        sha256=_uid_sequence_hash(tuple(panel_uids)),
        tracked_subset_uids=tuple(tracked_uids),
    )


def build_empirical_panel(
    records: Any,
    *,
    split: str,
    source: str,
    size: int,
    seed: int,
    tracked_subset_count: int = 0,
    feature_lookup: dict[str, dict[str, Any]] | None = None,
) -> PanelManifest:
    """Deterministically draw a label-blind empirical panel.

    `SCENARIONET-INTEGRATION` v1.2 / `EVAL-PROTOCOL` v1.1 `REQ-004`: unlike
    `build_balanced_panel`, this draw never reads ``primary_arm`` (or any
    other classification field) to decide which scenarios enter the panel.
    Candidate ``scenario_uid``s are deduplicated, sorted ascending for a
    catalog-order-independent baseline, then deterministically shuffled with
    a seed derived from ``(seed, split, source)``, and the first ``size``
    UIDs are taken uniformly. ``arms``/``per_arm_counts`` on the returned
    manifest are the panel's *observed* arm distribution, computed only
    after the draw -- reporting, never a selection criterion
    (`EVAL-PROTOCOL` v1.1 `REQ-007`).

    ``records`` must already be restricted by the caller to the intended
    ``split``/``source``/holdout-pool population (e.g. the empirical Waymo
    test pool, not the arm-stratified one): this function performs no such
    filtering itself, exactly like `build_balanced_panel`.

    Fails closed if fewer than ``size`` candidates are available: no
    fallback sampling or silent truncation (`REQ-004`).
    """
    if int(size) <= 0:
        raise ValueError("Panel size must be positive.")
    candidates = sorted({str(record.scenario_uid) for record in records})
    if len(candidates) < int(size):
        raise ValueError(
            f"Empirical panel build for split={split!r} source={source!r} "
            f"requires {size} scenarios but only {len(candidates)} eligible "
            "candidates exist. No fallback/backfill is permitted "
            "(EVAL-PROTOCOL REQ-004/DEC-004)."
        )

    seed_material = f"{seed}:{split}:{source}".encode("utf-8")
    draw_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(draw_seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    panel_uids = tuple(shuffled[: int(size)])

    arm_by_uid = {str(record.scenario_uid): str(record.primary_arm) for record in records}
    observed_counts = Counter(arm_by_uid[uid] for uid in panel_uids)
    observed_arms = tuple(sorted(observed_counts))
    per_arm_counts = tuple(observed_counts[arm] for arm in observed_arms)

    tracked_uids: tuple[str, ...] = ()
    if tracked_subset_count > 0:
        tracked_uids = _select_diverse_uids(
            panel_uids, count=tracked_subset_count, feature_lookup=feature_lookup
        )

    return PanelManifest(
        schema_version=PANEL_MANIFEST_SCHEMA_VERSION,
        split=str(split),
        seed=int(seed),
        size=int(size),
        arms=observed_arms,
        per_arm_counts=per_arm_counts,
        scenario_uids=panel_uids,
        sha256=_uid_sequence_hash(panel_uids),
        tracked_subset_uids=tracked_uids,
        draw_policy="empirical",
        source=str(source),
    )


def build_profile_subset_panel(
    parent: PanelManifest,
    records: Any,
    *,
    parent_name: str,
    scope: str,
    size: int,
    seed: int,
) -> PanelManifest:
    """Freeze a diagnostic child panel sampled only from ``parent``.

    The selection does not inspect arm labels: labels are read solely after
    selection to preserve the parent panel's reporting metadata.  The fixed
    seed is a profile policy seed, never a learner seed.
    """
    parent.verify_self_consistent()
    if scope not in {"smoke", "fast"}:
        raise ValueError(f"profile subset scope must be smoke or fast, got {scope!r}")
    if not parent_name:
        raise ValueError("profile subset requires a non-empty parent panel name")
    if int(size) <= 0 or int(size) > len(parent.scenario_uids):
        raise ValueError(
            f"profile subset size={size} is outside 1..{len(parent.scenario_uids)}"
        )
    by_uid = {str(record.scenario_uid): record for record in records}
    missing = [uid for uid in parent.scenario_uids if uid not in by_uid]
    if missing:
        raise ValueError(f"parent panel references unknown records: {missing[:5]}")
    material = f"{seed}:{scope}:{parent_name}:{parent.sha256}".encode("utf-8")
    subset_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    shuffled = list(parent.scenario_uids)
    np.random.default_rng(subset_seed).shuffle(shuffled)
    uids = tuple(shuffled[: int(size)])
    counts = Counter(str(by_uid[uid].primary_arm) for uid in uids)
    arms = tuple(sorted(counts))
    parent_tracked = set(parent.tracked_subset_uids)
    return PanelManifest(
        schema_version=PANEL_MANIFEST_SCHEMA_VERSION,
        split=parent.split,
        seed=int(seed),
        size=int(size),
        arms=arms,
        per_arm_counts=tuple(counts[arm] for arm in arms),
        scenario_uids=uids,
        sha256=_uid_sequence_hash(uids),
        tracked_subset_uids=tuple(uid for uid in uids if uid in parent_tracked),
        draw_policy=parent.draw_policy,
        source=parent.source,
        parent_panel=parent_name,
        parent_sha256=parent.sha256,
        scope=scope,
    )


def save_panel_manifest(manifest: PanelManifest, path: str | Path) -> None:
    """Persist ``manifest`` as versioned JSON, verifying self-consistency first."""
    manifest.verify_self_consistent()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "seed": manifest.seed,
        "size": manifest.size,
        "arms": list(manifest.arms),
        "per_arm_counts": list(manifest.per_arm_counts),
        "scenario_uids": list(manifest.scenario_uids),
        "sha256": manifest.sha256,
        "tracked_subset_uids": list(manifest.tracked_subset_uids),
        "draw_policy": manifest.draw_policy,
        "source": manifest.source,
        "parent_panel": manifest.parent_panel,
        "parent_sha256": manifest.parent_sha256,
        "scope": manifest.scope,
    }
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_panel_manifest(path: str | Path) -> PanelManifest:
    """Load and self-verify a persisted panel manifest. Fails closed on any
    structural or hash inconsistency (REQ-004: no fallback sampling)."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Panel manifest not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    required_keys = {
        "schema_version",
        "split",
        "seed",
        "size",
        "arms",
        "per_arm_counts",
        "scenario_uids",
        "sha256",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"Panel manifest {target} is missing required keys: {missing}")
    manifest = PanelManifest(
        schema_version=str(payload["schema_version"]),
        split=str(payload["split"]),
        seed=int(payload["seed"]),
        size=int(payload["size"]),
        arms=tuple(str(a) for a in payload["arms"]),
        per_arm_counts=tuple(int(c) for c in payload["per_arm_counts"]),
        scenario_uids=tuple(str(u) for u in payload["scenario_uids"]),
        sha256=str(payload["sha256"]),
        # Optional key: manifests built before this field existed (or
        # without a catalog feature lookup) have no tracked subset.
        tracked_subset_uids=tuple(str(u) for u in payload.get("tracked_subset_uids", ())),
        # Optional keys: manifests built before SCENARIONET-INTEGRATION v1.2 /
        # EVAL-PROTOCOL v1.1 predate the dual-panel policy and default to the
        # v1.0 arm-balanced, source-combined behavior they always had.
        draw_policy=str(payload.get("draw_policy", "arm_balanced")),
        source=str(payload.get("source", "combined")),
        parent_panel=(
            None if payload.get("parent_panel") in (None, "") else str(payload["parent_panel"])
        ),
        parent_sha256=(
            None if payload.get("parent_sha256") in (None, "") else str(payload["parent_sha256"])
        ),
        scope=str(payload.get("scope", "full")),
    )
    manifest.verify_self_consistent()
    return manifest


def select_tracked_subset_uids(manifest: PanelManifest, *, count: int = 5) -> tuple[str, ...]:
    """DEPRECATED runtime fallback -- prefer ``manifest.tracked_subset_uids``.

    As of the 2026-07-25 feature-diversity revision, the authoritative
    tracked subset is computed once at build time by `build_balanced_panel`
    (feature-diversity greedy selection per arm, with split-specific counts
    -- 4/arm validation, 10/arm test) and persisted on the manifest itself
    as ``tracked_subset_uids``. Callers should read that field directly and
    never recompute a tracked subset at runtime (`DEC-004`/`DEC-005`: no
    redraws). This function is kept only for manifests built before that
    field existed and for the pre-diversity round-robin selection it always
    implemented; it does not use catalog features and is not called by any
    production code path.

    A small, fixed, pre-declared tracked subset of ``scenario_uid``s
    (default 5 per split) rendered as GIFs unconditionally at every
    evaluation (validation and final test alike, same UIDs), for
    training-progression visibility.

    `manifest.scenario_uids` concatenates each arm's picks in `arms` order
    (see `build_balanced_panel`), so naively taking the first ``count`` UIDs
    would draw them all from a single arm. To keep the tracked subset
    visually representative across scenario types, this instead takes one
    UID per arm (the first UID of each arm's block, in `arms` order),
    cycling back to a second UID per arm only if ``count`` exceeds the
    number of arms. Deterministic and reproducible from the manifest alone,
    with no additional randomness and no dependency on any training seed.
    """
    if count <= 0:
        raise ValueError("Tracked-subset count must be positive.")
    if count > len(manifest.scenario_uids):
        raise ValueError(
            f"Tracked-subset count={count} exceeds panel size={len(manifest.scenario_uids)} "
            f"for split={manifest.split!r}."
        )
    arm_blocks: list[tuple[str, ...]] = []
    offset = 0
    for arm_count in manifest.per_arm_counts:
        arm_blocks.append(manifest.scenario_uids[offset : offset + arm_count])
        offset += arm_count

    selected: list[str] = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for block in arm_blocks:
            if round_index < len(block):
                selected.append(block[round_index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        round_index += 1
    return tuple(selected)


def default_panel_manifest_path(split: str, *, data_root: str | Path) -> Path:
    """`data/scenarionet/panels/<split>_panel_manifest_v1.json`, per ExecPlan §7.2."""
    return (
        Path(data_root)
        / "scenarionet"
        / "panels"
        / f"{split}_panel_manifest_{PANEL_MANIFEST_SCHEMA_VERSION}.json"
    )


#: `EVAL-PROTOCOL` v1.1 REQ-004 (§2.1): the five named panels, each with its
#: own split/source/size/draw_policy/role. Superset of the two legacy v1.0
#: panels (validation, test), which remain valid `default_panel_manifest_path`
#: names for backward compatibility with a v1.1-unaware pipeline.
NAMED_PANELS: dict[str, dict[str, str | int]] = {
    "validation_waymo_empirical": {
        "split": "validation",
        "source": "waymo",
        "draw_policy": "empirical",
        "holdout_pool": "empirical",
        "full_size": 150,
    },
    "validation_pg": {
        "split": "validation",
        "source": "pg",
        "draw_policy": "empirical",
        "holdout_pool": "empirical",
        "full_size": 150,
    },
    "test_waymo_empirical": {
        "split": "test",
        "source": "waymo",
        "draw_policy": "empirical",
        "holdout_pool": "empirical",
        "full_size": 400,
    },
    "test_pg": {
        "split": "test",
        "source": "pg",
        "draw_policy": "empirical",
        "holdout_pool": "empirical",
        "full_size": 300,
    },
    "test_arm_stratified": {
        "split": "test",
        "source": "combined",
        "draw_policy": "arm_balanced",
        "holdout_pool": "stratified",
        "full_size": 300,
    },
}


PROFILE_SUBSET_SEEDS: dict[str, int] = {"smoke": 20260731, "fast": 20260732}
PROFILE_SUBSET_SIZES: dict[str, dict[str, int]] = {
    "smoke": {
        "validation_waymo_empirical": 10,
        "validation_pg": 10,
        "test_waymo_empirical": 20,
        "test_pg": 20,
        "test_arm_stratified": 12,
    },
    "fast": {
        "validation_waymo_empirical": 50,
        "validation_pg": 50,
        "test_waymo_empirical": 100,
        "test_pg": 100,
        "test_arm_stratified": 60,
    },
}


def named_panel_manifest_path(name: str, *, data_root: str | Path) -> Path:
    """`data/scenarionet/panels/<name>_panel_manifest_v1.json` for a named v1.1 panel.

    ``name`` must be one of `NAMED_PANELS` (`test_waymo_empirical`,
    `test_pg`, `test_arm_stratified`, `validation_waymo_empirical`,
    `validation_pg`).
    """
    if name not in NAMED_PANELS:
        raise ValueError(f"unknown named panel: {name!r}; expected one of {sorted(NAMED_PANELS)}")
    return (
        Path(data_root)
        / "scenarionet"
        / "panels"
        / f"{name}_panel_manifest_{PANEL_MANIFEST_SCHEMA_VERSION}.json"
    )


def profile_panel_manifest_path(name: str, scope: str, *, data_root: str | Path) -> Path:
    """Path for a frozen `smoke` or `fast` child manifest."""
    if scope not in {"smoke", "fast"}:
        raise ValueError(f"unknown profile subset scope: {scope!r}")
    if name not in NAMED_PANELS:
        raise ValueError(f"unknown named panel: {name!r}")
    return (
        Path(data_root)
        / "scenarionet"
        / "panels"
        / f"{scope}_{name}_panel_manifest_{PANEL_MANIFEST_SCHEMA_VERSION}.json"
    )
