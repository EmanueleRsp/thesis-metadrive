from __future__ import annotations

import json
from pathlib import Path

from thesis_rl.cli.scenarios.generate_pg_dataset import main
from thesis_rl.scenarios.pg.report import PGPilotReport


def test_pg_replenishment_can_write_a_separate_report(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "replenishment" / "report.json"
    monkeypatch.setattr(
        "thesis_rl.cli.scenarios.generate_pg_dataset.collect_local_api_inventory",
        lambda _root: {"git": {"metadrive": {"commit": "test-commit"}}},
    )
    monkeypatch.setattr(
        "thesis_rl.cli.scenarios.generate_pg_dataset.run_pg_pilot",
        lambda **kwargs: (
            PGPilotReport(
                requested=5,
                generated=5,
                failed=0,
                by_profile_arm={},
                failures=(),
            ),
            (),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_pg_dataset",
            "--data-root",
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--count",
            "1",
            "--report-output",
            str(report_path),
            "--overwrite",
        ],
    )

    assert main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["generated"] == 5
    assert not (tmp_path / "pg" / "pilot" / "pg_pilot_report.json").exists()
