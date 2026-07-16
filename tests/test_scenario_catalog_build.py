from __future__ import annotations

import json
import pickle
from pathlib import Path
import shutil

import pytest

from thesis_rl.cli.scenarios.build_catalog import main as build_catalog_main
from thesis_rl.scenarios.parallel import ordered_process_map
from thesis_rl.scenarios.pg.loader import load_exported_pg_entries
from thesis_rl.scenarios.waymo import load_converted_waymo_entries


def _multiply(value: int) -> int:
    return value * 2


def _pg_scenario(scenario_id: str) -> dict:
    return {
        "id": scenario_id,
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "polygon": [
                    [0.0, -2.0, 0.0],
                    [10.0, -2.0, 0.0],
                    [10.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                ],
                "exit_lanes": [],
            }
        },
        "dynamic_map_states": {},
    }


def _write_pg_database(root: Path, seeds: tuple[int, ...] = (1, 2)) -> Path:
    database = root / "pg" / "database" / "P0_simple"
    for seed in seeds:
        path = database / str(seed) / f"PGMap-{seed}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(_pg_scenario(f"PGMap-{seed}"), handle)
    return root / "pg" / "database"


def test_ordered_process_map_preserves_order_and_reports_progress() -> None:
    progress: list[tuple[int, int]] = []

    result = ordered_process_map(
        (3, 1, 2),
        _multiply,
        workers=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert result == (6, 2, 4)
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_ordered_process_map_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        ordered_process_map((1,), _multiply, workers=0)


def test_waymo_loader_parallel_matches_serial() -> None:
    database = Path("third_party/metadrive/metadrive/assets/waymo")
    serial_entries, serial_groups = load_converted_waymo_entries(database, data_root=Path("."))
    parallel_entries, parallel_groups = load_converted_waymo_entries(
        database,
        data_root=Path("."),
        workers=2,
    )

    assert parallel_entries == serial_entries
    assert parallel_groups == serial_groups


def test_pg_loader_parallel_matches_serial_and_preserves_seed_window(tmp_path: Path) -> None:
    database = _write_pg_database(tmp_path, seeds=(1, 2, 3))
    kwargs = {
        "database_path": database,
        "data_root": tmp_path,
        "split": "train",
        "seed_start": 1,
        "count_per_profile": 2,
    }
    serial = load_exported_pg_entries(workers=1, **kwargs)
    parallel = load_exported_pg_entries(workers=2, **kwargs)

    assert parallel == serial
    assert [entry.record.pg_seed for entry in parallel] == [1, 2]


def test_build_catalog_cli_reports_rich_progress_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    waymo_source = Path("third_party/metadrive/metadrive/assets/waymo")
    waymo_database = tmp_path / "waymo" / "database"
    waymo_database.mkdir(parents=True)
    for path in waymo_source.glob("sd_*.pkl"):
        shutil.copy2(path, waymo_database / path.name)
    pg_database = _write_pg_database(tmp_path)
    output = tmp_path / "catalog" / "raw.parquet"
    groups = tmp_path / "splits" / "groups.json"
    report = tmp_path / "catalog" / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_catalog",
            "--data-root",
            str(tmp_path),
            "--waymo-database",
            str(waymo_database),
            "--pg-database",
            str(pg_database),
            "--waymo-workers",
            "2",
            "--pg-workers",
            "2",
            "--output",
            str(output),
            "--groups-output",
            str(groups),
            "--report-output",
            str(report),
            "--overwrite",
        ],
    )

    assert build_catalog_main() == 0
    captured = capsys.readouterr()
    assert "Loading catalog entries with Waymo workers=2, PG" in captured.err
    assert "workers=2" in captured.err
    assert "Loading Waymo catalog entries" in captured.err
    assert "Loading PG catalog entries" in captured.err
    assert output.is_file()
    assert groups.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["total"] == 5


def test_build_catalog_cli_allows_empty_waymo_candidate_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pg_database = _write_pg_database(tmp_path)
    output = tmp_path / "catalog" / "raw.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_catalog",
            "--data-root",
            str(tmp_path),
            "--waymo-database",
            str(tmp_path / "waymo" / "database"),
            "--pg-database",
            str(pg_database),
            "--output",
            str(output),
            "--allow-empty-waymo",
        ],
    )

    assert build_catalog_main() == 0
    assert json.loads((output.parent / "catalog_report.json").read_text(encoding="utf-8"))[
        "by_source"
    ] == {"pg": 2, "waymo": 0}
