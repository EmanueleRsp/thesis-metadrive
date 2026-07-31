"""EVAL-PROTOCOL v1.0 REQ-004/DEC-005 tests: deterministic balanced panel
draw, freezing/hashing/deduplication, and fail-closed load verification."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.panel_manifest import (
    NAMED_PANELS,
    PanelManifest,
    build_balanced_panel,
    build_empirical_panel,
    build_profile_subset_panel,
    default_panel_manifest_path,
    load_panel_manifest,
    named_panel_manifest_path,
    save_panel_manifest,
    select_tracked_subset_uids,
)
from thesis_rl.scenarios.panel_manifest import _uid_sequence_hash
from thesis_rl.scenarios.records import ScenarioRecord


def _record(index: int, arm: str, *, source: str = "waymo") -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid=f"{source}:v1:{arm}:{index}",
        scenario_id=str(index),
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{index}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=f"log-{index}" if source == "waymo" else None,
        source_scenario_id=str(index) if source == "waymo" else None,
        dataset_version="v1",
        converter_version="converter" if source == "waymo" else None,
        split="validation",  # type: ignore[arg-type]
        runtime_index=index,
        length=100,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id="S",
        primary_arm=arm,
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def _balanced_records(per_arm: int = 20) -> list[ScenarioRecord]:
    records = []
    counter = 0
    for arm in ARMS:
        for _ in range(per_arm):
            records.append(_record(counter, arm))
            counter += 1
    return records


def test_build_balanced_panel_draws_evenly_across_all_six_arms() -> None:
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(records, split="validation", size=60, seed=1234)

    assert manifest.arms == ARMS
    assert manifest.per_arm_counts == (10,) * 6
    assert len(manifest.scenario_uids) == 60
    assert len(set(manifest.scenario_uids)) == 60  # deduplicated

    counts_by_arm = {arm: 0 for arm in ARMS}
    uid_to_arm = {record.scenario_uid: record.primary_arm for record in records}
    for uid in manifest.scenario_uids:
        counts_by_arm[uid_to_arm[uid]] += 1
    assert all(count == 10 for count in counts_by_arm.values())


def test_build_balanced_panel_distributes_remainder_to_first_arms() -> None:
    records = _balanced_records(per_arm=20)
    # 61 / 6 = 10 remainder 1: the first arm in ARMS order gets 11.
    manifest = build_balanced_panel(records, split="validation", size=61, seed=1)
    assert manifest.per_arm_counts == (11, 10, 10, 10, 10, 10)
    assert sum(manifest.per_arm_counts) == 61


def test_build_balanced_panel_is_deterministic_for_the_same_seed() -> None:
    records = _balanced_records(per_arm=20)
    first = build_balanced_panel(records, split="validation", size=60, seed=7)
    second = build_balanced_panel(records, split="validation", size=60, seed=7)
    assert first.scenario_uids == second.scenario_uids
    assert first.sha256 == second.sha256


def test_build_balanced_panel_differs_for_a_different_seed() -> None:
    records = _balanced_records(per_arm=20)
    first = build_balanced_panel(records, split="validation", size=60, seed=1)
    second = build_balanced_panel(records, split="validation", size=60, seed=2)
    assert first.scenario_uids != second.scenario_uids


def test_build_balanced_panel_fails_closed_when_an_arm_is_undersupplied() -> None:
    records = _balanced_records(per_arm=20)
    # Remove all but 2 candidates for one arm.
    scarce_arm = ARMS[0]
    filtered = (
        [r for r in records if r.primary_arm != scarce_arm][:0]
        + [r for r in records if r.primary_arm != scarce_arm]
        + [r for r in records if r.primary_arm == scarce_arm][:2]
    )
    with pytest.raises(ValueError, match="eligible candidates exist"):
        build_balanced_panel(filtered, split="validation", size=60, seed=1)


def test_manifest_hash_is_order_sensitive() -> None:
    manifest = PanelManifest(
        schema_version="v1",
        split="validation",
        seed=1,
        size=2,
        arms=("A0_simple_low_traffic",),
        per_arm_counts=(2,),
        scenario_uids=("waymo:v1:A0:1", "waymo:v1:A0:2"),
        sha256="",
    )
    reordered = PanelManifest(
        schema_version="v1",
        split="validation",
        seed=1,
        size=2,
        arms=("A0_simple_low_traffic",),
        per_arm_counts=(2,),
        scenario_uids=("waymo:v1:A0:2", "waymo:v1:A0:1"),
        sha256="",
    )
    hash_a = build_balanced_panel(
        [_record(1, "A0_simple_low_traffic")], split="validation", size=1, seed=0
    ).sha256
    assert isinstance(hash_a, str) and len(hash_a) == 64
    assert manifest.scenario_uids != reordered.scenario_uids


def test_save_and_load_panel_manifest_round_trips(tmp_path: Path) -> None:
    records = _balanced_records(per_arm=10)
    manifest = build_balanced_panel(records, split="validation", size=30, seed=99)
    path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, path)

    loaded = load_panel_manifest(path)
    assert loaded == manifest


def test_profile_subset_is_deterministic_and_carries_parent_identity() -> None:
    records = _balanced_records(per_arm=20)
    parent = build_balanced_panel(records, split="validation", size=60, seed=99)
    first = build_profile_subset_panel(
        parent, records, parent_name="validation_waymo_empirical", scope="smoke", size=10, seed=20260731
    )
    second = build_profile_subset_panel(
        parent, records, parent_name="validation_waymo_empirical", scope="smoke", size=10, seed=20260731
    )
    assert first == second
    assert first.scope == "smoke"
    assert first.parent_panel == "validation_waymo_empirical"
    assert first.parent_sha256 == parent.sha256
    assert set(first.scenario_uids).issubset(set(parent.scenario_uids))


def test_profile_subset_rejects_learner_style_or_invalid_scope() -> None:
    records = _balanced_records(per_arm=2)
    parent = build_balanced_panel(records, split="validation", size=12, seed=1)
    with pytest.raises(ValueError, match="scope"):
        build_profile_subset_panel(
            parent, records, parent_name="validation_waymo_empirical", scope="full", size=2, seed=7
        )


def test_load_panel_manifest_fails_closed_on_tampered_hash(tmp_path: Path) -> None:
    records = _balanced_records(per_arm=10)
    manifest = build_balanced_panel(records, split="validation", size=30, seed=99)
    path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, path)

    text = path.read_text(encoding="utf-8")
    tampered = text.replace(manifest.sha256, "0" * 64)
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="internally inconsistent"):
        load_panel_manifest(path)


def test_load_panel_manifest_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_panel_manifest("/nonexistent/panel_manifest.json")


def test_default_panel_manifest_path_matches_execplan_convention() -> None:
    path = default_panel_manifest_path("test", data_root="/workspace/data")
    assert path == Path("/workspace/data/scenarionet/panels/test_panel_manifest_v1.json")


def test_select_tracked_subset_uids_defaults_to_five_one_per_arm() -> None:
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(records, split="validation", size=60, seed=42)
    tracked = select_tracked_subset_uids(manifest)
    assert len(tracked) == 5
    # One UID per arm (six arms, five requested): the first UID of each of
    # the first five arm blocks, not the first five UIDs of a single arm.
    offset = 0
    expected = []
    for arm_count in manifest.per_arm_counts[:5]:
        expected.append(manifest.scenario_uids[offset])
        offset += arm_count
    assert tracked == tuple(expected)


def test_select_tracked_subset_uids_is_deterministic_across_calls() -> None:
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(records, split="test", size=90, seed=7)
    first = select_tracked_subset_uids(manifest, count=5)
    second = select_tracked_subset_uids(manifest, count=5)
    assert first == second


def test_select_tracked_subset_uids_rejects_count_larger_than_panel() -> None:
    records = _balanced_records(per_arm=1)
    manifest = build_balanced_panel(records, split="validation", size=6, seed=1)
    with pytest.raises(ValueError, match="exceeds panel size"):
        select_tracked_subset_uids(manifest, count=10)


# --- REQ-014/DEC-014 (amended 2026-07-25): per-arm feature-diversity
# tracked-subset selection, computed once at build time and persisted on
# the manifest as `tracked_subset_uids`. ---


def _feature_lookup_for(
    records: list[ScenarioRecord], *, feature_by_uid: dict[str, dict]
) -> dict[str, dict]:
    # Every record must have a feature entry; unlisted UIDs default to an
    # empty dict (all-None feature tuple), which is still a valid, distinct
    # diversity bucket.
    return {record.scenario_uid: feature_by_uid.get(record.scenario_uid, {}) for record in records}


def test_build_balanced_panel_without_tracked_subset_count_leaves_it_empty() -> None:
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(records, split="validation", size=60, seed=1234)
    assert manifest.tracked_subset_uids == ()


def test_build_balanced_panel_tracked_subset_count_per_arm_without_feature_lookup_uses_prefix_order() -> (
    None
):
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(
        records, split="validation", size=60, seed=1234, tracked_subset_count_per_arm=4
    )
    assert len(manifest.tracked_subset_uids) == 4 * len(ARMS)
    assert set(manifest.tracked_subset_uids).issubset(set(manifest.scenario_uids))
    # No feature_lookup: falls back to the first 4 UIDs of each arm's block.
    offset = 0
    expected: list[str] = []
    for arm_count in manifest.per_arm_counts:
        expected.extend(manifest.scenario_uids[offset : offset + 4])
        offset += arm_count
    assert manifest.tracked_subset_uids == tuple(expected)


def test_build_balanced_panel_tracked_subset_prefers_feature_diversity() -> None:
    records = _balanced_records(per_arm=20)
    arm0_records = [r for r in records if r.primary_arm == ARMS[0]]
    # Two distinct feature buckets among the arm-0 candidates: half have an
    # intersection + traffic light, half don't. A diversity-driven pick of 2
    # should surface both buckets, not two UIDs that happen to share one.
    feature_by_uid = {}
    for index, record in enumerate(arm0_records):
        feature_by_uid[record.scenario_uid] = {
            "has_intersection": bool(index % 2),
            "has_route_traffic_light": bool(index % 2),
        }
    feature_lookup = _feature_lookup_for(records, feature_by_uid=feature_by_uid)

    manifest = build_balanced_panel(
        records,
        split="validation",
        size=60,
        seed=1234,
        tracked_subset_count_per_arm=2,
        feature_lookup=feature_lookup,
    )
    arm0_offset = 0
    arm0_count = manifest.per_arm_counts[0]
    arm0_tracked = manifest.tracked_subset_uids[:2]
    arm0_panel_uids = manifest.scenario_uids[arm0_offset : arm0_offset + arm0_count]
    tracked_features = {
        feature_lookup[uid]["has_intersection"] for uid in arm0_tracked if uid in arm0_panel_uids
    }
    assert tracked_features == {True, False}


def test_build_balanced_panel_tracked_subset_is_deterministic() -> None:
    records = _balanced_records(per_arm=20)
    feature_lookup = _feature_lookup_for(
        records,
        feature_by_uid={
            r.scenario_uid: {"has_intersection": bool(i % 3)} for i, r in enumerate(records)
        },
    )
    manifest_a = build_balanced_panel(
        records,
        split="test",
        size=90,
        seed=99,
        tracked_subset_count_per_arm=3,
        feature_lookup=feature_lookup,
    )
    manifest_b = build_balanced_panel(
        records,
        split="test",
        size=90,
        seed=99,
        tracked_subset_count_per_arm=3,
        feature_lookup=feature_lookup,
    )
    assert manifest_a.tracked_subset_uids == manifest_b.tracked_subset_uids


def test_panel_manifest_rejects_tracked_subset_uids_not_in_scenario_uids() -> None:
    scenario_uids = ("waymo:v1:A0:0", "waymo:v1:A0:1")
    with pytest.raises(ValueError, match="subset of"):
        PanelManifest(
            schema_version="v1",
            split="validation",
            seed=1,
            size=2,
            arms=("A0",),
            per_arm_counts=(2,),
            scenario_uids=scenario_uids,
            sha256=_uid_sequence_hash(scenario_uids),
            tracked_subset_uids=("waymo:v1:A0:99",),
        ).verify_self_consistent()


def test_save_and_load_panel_manifest_roundtrips_tracked_subset_uids(tmp_path: Path) -> None:
    records = _balanced_records(per_arm=20)
    feature_lookup = _feature_lookup_for(
        records, feature_by_uid={r.scenario_uid: {"has_intersection": True} for r in records}
    )
    manifest = build_balanced_panel(
        records,
        split="validation",
        size=60,
        seed=5,
        tracked_subset_count_per_arm=4,
        feature_lookup=feature_lookup,
    )
    path = tmp_path / "panel.json"
    save_panel_manifest(manifest, path)
    loaded = load_panel_manifest(path)
    assert loaded.tracked_subset_uids == manifest.tracked_subset_uids


def test_load_panel_manifest_without_tracked_subset_uids_key_defaults_empty(tmp_path: Path) -> None:
    records = _balanced_records(per_arm=20)
    manifest = build_balanced_panel(records, split="validation", size=60, seed=5)
    path = tmp_path / "panel.json"
    save_panel_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["tracked_subset_uids"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_panel_manifest(path)
    assert loaded.tracked_subset_uids == ()


# --- SCENARIONET-INTEGRATION v1.2 / EVAL-PROTOCOL v1.1 dual-policy panels (M5) ---


def _skewed_records(counts: dict[str, int]) -> list[ScenarioRecord]:
    records = []
    counter = 0
    for arm, count in counts.items():
        for _ in range(count):
            records.append(_record(counter, arm))
            counter += 1
    return records


def test_empirical_panel_is_not_arm_balanced() -> None:
    # TEST-010 (EVAL-PROTOCOL v1.1 REQ-004): drawing from a heavily skewed
    # pool produces a skewed panel, tracking the pool's distribution rather
    # than rebalancing it -- the opposite of build_balanced_panel.
    records = _skewed_records(
        {
            "A0_simple_low_traffic": 90,
            "A1_traffic": 5,
            "A2_junction": 5,
        }
    )
    manifest = build_empirical_panel(records, split="test", source="waymo", size=50, seed=7)
    counts = dict(zip(manifest.arms, manifest.per_arm_counts))
    assert counts.get("A0_simple_low_traffic", 0) > counts.get("A1_traffic", 0)
    assert sum(manifest.per_arm_counts) == 50
    assert manifest.draw_policy == "empirical"
    assert manifest.source == "waymo"


def test_empirical_panel_never_reads_arm_for_selection() -> None:
    # Relabelling every record's arm must not change which UIDs are drawn:
    # the empirical draw shuffles by scenario_uid only.
    records = _skewed_records({"A0_simple_low_traffic": 40, "A1_traffic": 40})
    relabelled = [replace(r, primary_arm="A5_critical_mixed") for r in records]
    manifest_a = build_empirical_panel(records, split="test", source="waymo", size=30, seed=3)
    manifest_b = build_empirical_panel(relabelled, split="test", source="waymo", size=30, seed=3)
    assert manifest_a.scenario_uids == manifest_b.scenario_uids


def test_empirical_panel_fails_closed_when_undersupplied() -> None:
    records = _skewed_records({"A0_simple_low_traffic": 5})
    with pytest.raises(ValueError, match="requires 50 scenarios"):
        build_empirical_panel(records, split="test", source="waymo", size=50, seed=1)


def test_balanced_panel_retains_arm_balanced_draw_policy_default() -> None:
    # TEST-011: existing arm-balanced behavior/manifest fields are unchanged.
    manifest = build_balanced_panel(
        _balanced_records(per_arm=10), split="validation", size=12, seed=0
    )
    assert manifest.draw_policy == "arm_balanced"
    assert manifest.source == "combined"


def test_save_and_load_panel_manifest_roundtrips_draw_policy_and_source(tmp_path: Path) -> None:
    # TEST-012: hash/ordering semantics preserved; new fields round-trip.
    manifest = build_empirical_panel(
        _skewed_records({"A0_simple_low_traffic": 20, "A1_traffic": 20}),
        split="validation",
        source="pg",
        size=15,
        seed=2,
    )
    path = tmp_path / "panel.json"
    save_panel_manifest(manifest, path)
    loaded = load_panel_manifest(path)
    assert loaded == manifest
    assert loaded.draw_policy == "empirical"
    assert loaded.source == "pg"


def test_load_panel_manifest_without_draw_policy_key_defaults_to_arm_balanced_combined(
    tmp_path: Path,
) -> None:
    # A manifest persisted before v1.1 (no draw_policy/source keys) must
    # still load, defaulting to the only behavior it could have had.
    manifest = build_balanced_panel(_balanced_records(per_arm=5), split="test", size=6, seed=0)
    path = tmp_path / "legacy_panel.json"
    save_panel_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["draw_policy"]
    del payload["source"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_panel_manifest(path)
    assert loaded.draw_policy == "arm_balanced"
    assert loaded.source == "combined"


def test_named_panel_manifest_path_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown named panel"):
        named_panel_manifest_path("not_a_real_panel", data_root=tmp_path)


def test_named_panel_manifest_path_matches_convention(tmp_path: Path) -> None:
    for name in NAMED_PANELS:
        path = named_panel_manifest_path(name, data_root=tmp_path)
        assert path == tmp_path / "scenarionet" / "panels" / f"{name}_panel_manifest_v1.json"


def test_panel_manifest_rejects_unsupported_draw_policy() -> None:
    manifest = build_balanced_panel(_balanced_records(per_arm=2), split="test", size=6, seed=0)
    tampered = replace(manifest, draw_policy="bogus")
    with pytest.raises(ValueError, match="unsupported panel draw_policy"):
        tampered.verify_self_consistent()


def test_panel_manifest_rejects_unsupported_source() -> None:
    manifest = build_balanced_panel(_balanced_records(per_arm=2), split="test", size=6, seed=0)
    tampered = replace(manifest, source="bogus")
    with pytest.raises(ValueError, match="unsupported panel source"):
        tampered.verify_self_consistent()
