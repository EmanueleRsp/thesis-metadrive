"""Compact Rich progress display for the Waymo converter.

The checked-out ScenarioNet converter is a multiprocessing program and emits
one log line for every worker/file.  The conversion runner parses those lines
into structured events; this module owns the human-facing rendering so the
worker output never reaches the terminal directly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


class ProgressSink(Protocol):
    """Consumer of structured converter progress events."""

    def emit(self, event: dict[str, Any]) -> None:
        ...


class NullProgressSink:
    """No-op sink for callers that do not need a human-facing display."""

    def emit(self, event: dict[str, Any]) -> None:
        del event


_TOTAL_FILES = re.compile(r"(?:Find|read)\s+(\d+)\s+waymo files", re.IGNORECASE)
_WORKER_READ = re.compile(r"Worker\s+(\d+)\s+is reading raw file:\s*(.+)$")
_WORKER_FINISHED = re.compile(r"Worker\s+(\d+)\s+finished read\s+(\d+)\s+files\.")
_WORKER_PERCENT = re.compile(r"Worker\s+(\d+).*?\s(\d+)%")


def parse_converter_line(line: str) -> list[dict[str, Any]]:
    """Translate noisy ScenarioNet output into compact dashboard events."""

    text = line.strip()
    if not text:
        return []
    events: list[dict[str, Any]] = []
    total_match = _TOTAL_FILES.search(text)
    if total_match:
        events.append({"kind": "total", "total_files": int(total_match.group(1))})
    read_match = _WORKER_READ.search(text)
    if read_match:
        events.append(
            {
                "kind": "worker_read",
                "worker": int(read_match.group(1)),
                "file": read_match.group(2).strip().rsplit("/", 1)[-1],
            }
        )
    finished_match = _WORKER_FINISHED.search(text)
    if finished_match:
        events.append(
            {
                "kind": "worker_finished",
                "worker": int(finished_match.group(1)),
                "files": int(finished_match.group(2)),
            }
        )
    percent_match = _WORKER_PERCENT.search(text)
    if percent_match:
        events.append(
            {
                "kind": "worker_percent",
                "worker": int(percent_match.group(1)),
                "percent": int(percent_match.group(2)),
            }
        )
    if not events and not text.startswith(("I tensorflow/", "W tensorflow/")):
        events.append({"kind": "message", "message": text[-240:]})
    return events


@dataclass
class WorkerState:
    worker: int
    status: str = "waiting"
    current_file: str = "-"
    files_completed: int = 0
    percent: int = 0


@dataclass
class WaymoProgressState:
    total_files: int | None = None
    completed_files: int = 0
    workers: dict[int, WorkerState] = field(default_factory=dict)
    last_message: str = "Waiting for converter output"
    started_at: float = field(default_factory=time.monotonic)
    finished: bool = False
    exit_code: int | None = None

    def _worker(self, worker_id: int) -> WorkerState:
        return self.workers.setdefault(worker_id, WorkerState(worker_id))

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "started":
            self.started_at = time.monotonic()
            if event.get("total_files") is not None:
                self.total_files = int(event["total_files"])
            self.last_message = "Converter started"
        elif kind == "total":
            self.total_files = int(event["total_files"])
        elif kind == "worker_read":
            worker = self._worker(int(event["worker"]))
            if worker.current_file != "-" and worker.status == "reading":
                worker.files_completed += 1
            worker.status = "reading"
            worker.current_file = str(event["file"])
            self.completed_files = sum(item.files_completed for item in self.workers.values())
        elif kind == "worker_percent":
            worker = self._worker(int(event["worker"]))
            worker.percent = int(event["percent"])
        elif kind == "worker_finished":
            worker = self._worker(int(event["worker"]))
            worker.files_completed = max(worker.files_completed, int(event["files"]))
            worker.status = "done"
            worker.current_file = "-"
            worker.percent = 100
            self.completed_files = sum(item.files_completed for item in self.workers.values())
        elif kind == "message":
            self.last_message = str(event.get("message", ""))
        elif kind == "finished":
            self.finished = True
            self.exit_code = int(event.get("exit_code", 1))
            self.last_message = "Conversion completed" if self.exit_code == 0 else "Conversion failed"


class WaymoDashboard:
    """Rich Live dashboard backed by :class:`WaymoProgressState`."""

    def __init__(self, state: WaymoProgressState | None = None) -> None:
        self.state = state or WaymoProgressState()

    def render(self) -> Group:
        state = self.state
        elapsed = max(0.0, time.monotonic() - state.started_at)
        rate = state.completed_files / elapsed * 60 if elapsed > 0 else 0.0
        remaining = max(0, (state.total_files or 0) - state.completed_files)
        eta = remaining / (rate / 60) if rate > 0 else 0.0
        eta_text = f"{int(eta // 60)}m {int(eta % 60):02d}s" if eta else "-"

        summary = Table.grid(expand=True)
        summary.add_column(style="bold cyan")
        summary.add_column()
        summary.add_column(style="bold cyan")
        summary.add_column()
        summary.add_row(
            "Files", f"{state.completed_files} / {state.total_files or '?'}",
            "Workers", str(len(state.workers)),
        )
        summary.add_row("Rate", f"{rate:.1f} files/min", "ETA", eta_text)

        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        progress.add_task(
            "Waymo files",
            total=state.total_files,
            completed=min(state.completed_files, state.total_files or state.completed_files),
        )

        workers = Table(expand=True, box=None, show_header=True)
        workers.add_column("Worker", style="cyan", width=8)
        workers.add_column("State", width=10)
        workers.add_column("File")
        workers.add_column("Done", justify="right", width=8)
        for worker_id in sorted(state.workers):
            worker = state.workers[worker_id]
            workers.add_row(
                str(worker.worker), worker.status, worker.current_file, str(worker.files_completed)
            )

        footer = Text(state.last_message, style="dim")
        title = "Waymo conversion" if not state.finished else "Waymo conversion finished"
        style = "green" if state.exit_code == 0 else "red" if state.finished else "cyan"
        return Group(
            Panel(summary, title=title, border_style=style),
            progress,
            Panel(workers, title="Workers", border_style="cyan"),
            footer,
        )


class RichProgressSink:
    """Render all converter events in one in-place Rich dashboard."""

    def __init__(self, *, console: Console | None = None) -> None:
        self.dashboard = WaymoDashboard()
        self.console = console or Console(stderr=True, highlight=False)
        self._live: Live | None = None

    def __enter__(self) -> "RichProgressSink":
        self._live = Live(
            self.dashboard.render(),
            console=self.console,
            refresh_per_second=6,
            transient=False,
        )
        self._live.start()
        return self

    def emit(self, event: dict[str, Any]) -> None:
        self.dashboard.state.apply(event)
        if self._live is not None:
            self._live.update(self.dashboard.render())

    def close(self) -> None:
        if self._live is not None:
            self._live.update(self.dashboard.render())
            self._live.stop()
            self._live = None

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "NullProgressSink",
    "ProgressSink",
    "RichProgressSink",
    "WaymoDashboard",
    "WaymoProgressState",
    "parse_converter_line",
]
