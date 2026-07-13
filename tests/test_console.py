from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from thesis_rl.runtime.io import console


def test_evaluation_summary_supports_provider_selected_scenarios(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    console.print_evaluation_summary(
        title="Final Evaluation",
        metrics={"mean_reward": 1.0},
        stage="scenario_acl",
        global_step=100,
        episodes=2,
        base_seed=None,
        details_path=Path("events.jsonl"),
    )

    assert "Scenario seeds" in output.getvalue()
    assert "provider-selected" in output.getvalue()


def test_evaluation_summary_formats_sequential_scenarios(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(console, "_CONSOLE", Console(file=output, color_system=None))

    console.print_evaluation_summary(
        title="Evaluation",
        metrics={},
        stage="baseline",
        global_step=100,
        episodes=3,
        base_seed=42,
        details_path=Path("events.jsonl"),
    )

    assert "Scenario seeds" in output.getvalue()
    assert "42..44" in output.getvalue()
