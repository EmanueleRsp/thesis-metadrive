from __future__ import annotations

from metadrive.engine.asset_loader import AssetLoader

from thesis_rl.scenarios.smoke import smoke_test_scenario_env


def test_bundled_waymo_fixture_resets_and_steps() -> None:
    result = smoke_test_scenario_env(
        AssetLoader.file_path("waymo", unix_style=False),
        steps=2,
    )

    assert result["executed_steps"] == 2
    assert result["scenario_length"] > 2
    assert result["action_shape"] == [2]
    assert result["reactive_traffic"] is True


def test_bundled_waymo_fixture_accepts_random_policy() -> None:
    result = smoke_test_scenario_env(
        AssetLoader.file_path("waymo", unix_style=False),
        steps=2,
        policy="random",
    )

    assert result["executed_steps"] == 2
    assert result["policy"] == "random"
