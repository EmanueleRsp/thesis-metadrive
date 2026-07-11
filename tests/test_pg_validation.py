from __future__ import annotations

from thesis_rl.scenarios.pg.validation import validate_exported_scenario


def test_invalid_export_is_reported_without_exception() -> None:
    result = validate_exported_scenario({"id": "broken", "length": 0})

    assert result.status == "invalid"
    assert result.warnings


def test_bundled_export_validation_can_be_read() -> None:
    import pickle
    from pathlib import Path

    path = sorted(Path("third_party/metadrive/metadrive/assets/waymo").glob("sd_*.pkl"))[0]
    with path.open("rb") as handle:
        scenario = pickle.load(handle)
    result = validate_exported_scenario(scenario)

    assert result.status in {"valid", "warning"}
    assert result.scenario_length == scenario["length"]
