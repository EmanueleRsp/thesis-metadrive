from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_default))
        handle.write("\n")


def setup_file_logger(
    namespace: str,
    name: str,
    log_file: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(f"{namespace}.{name}.{log_file}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_event(events_path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(fields)
    append_jsonl(events_path, payload)
