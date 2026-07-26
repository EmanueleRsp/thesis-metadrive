from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from thesis_rl.scenarios.pg.generator import generate_pg_scenario


@pytest.mark.integration
def test_p0_generation_exports_reloadable_scenario(tmp_path: Path) -> None:
    result = generate_pg_scenario("P0_simple", seed=123, data_root=tmp_path, overwrite=True)

    assert result.validation.status == "valid"
    assert result.entry.record.source == "pg"
    assert result.entry.record.pg_seed == 123
    assert result.entry.record.relative_path.startswith("pg/database/P0_simple/123/")
    assert result.generation_manifest.exists()
    assert result.entry.features.length > 0


def _load_exported_scenario(data_root: Path, profile: str, seed: int) -> dict:
    dataset_dir = data_root / "pg" / "database" / profile / str(seed)
    scenario_files = sorted(
        path
        for path in dataset_dir.glob("*.pkl")
        if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
    )
    assert len(scenario_files) == 1, scenario_files
    with scenario_files[0].open("rb") as handle:
        return pickle.load(handle)


@pytest.mark.integration
def test_roundabout_generation_populates_vehicle_yield_priority_metadata(tmp_path: Path) -> None:
    """Seed 14 of P2_merge_or_roundabout deterministically samples a
    single-block ``("O",)`` roundabout map (verified empirically). The
    exported scenario's ``rulebook_vehicle_yield.roundabout_priorities``
    metadata must be populated from MetaDrive's own live block structure,
    with lane ids that resolve against the exported ``map_features``."""

    result = generate_pg_scenario(
        "P2_merge_or_roundabout", seed=14, data_root=tmp_path, overwrite=True
    )
    assert result.validation.status == "valid"

    scenario = _load_exported_scenario(tmp_path, "P2_merge_or_roundabout", 14)
    records = scenario["metadata"]["rulebook_vehicle_yield"]["roundabout_priorities"]
    assert records
    map_features = scenario["map_features"]
    for record in records:
        assert record["entry_lane_id"] in map_features
        assert record["circulating_lane_id"] in map_features
        assert record["entry_lane_id"] != record["circulating_lane_id"]
