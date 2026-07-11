from __future__ import annotations

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
