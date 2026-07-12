"""Shared Rich output helpers for ScenarioNet command-line tools.

Human-readable progress is sent to stderr so that the JSON reports printed by
the scenario CLIs remain machine-readable on stdout.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


console = Console(stderr=True, highlight=False)


def make_progress() -> Progress:
    """Build a progress display that also degrades cleanly in CI logs."""

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        refresh_per_second=8,
    )


def print_panel(title: str, message: str, *, style: str = "green") -> None:
    """Print a compact stage/result panel to the human-facing stream."""

    console.print(Panel(message, title=title, border_style=style, expand=False))


def print_key_value_table(
    title: str,
    rows: list[tuple[str, Any]],
    *,
    style: str = "cyan",
) -> None:
    """Print a two-column summary table without changing CLI stdout."""

    table = Table(title=title, show_header=False, border_style=style, expand=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(table)


__all__ = ["console", "make_progress", "print_key_value_table", "print_panel"]
