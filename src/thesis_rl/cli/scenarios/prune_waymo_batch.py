"""Remove converted Waymo scenarios that are absent from a frozen selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.scenarios.frozen import load_frozen_index


def _selected_relative_paths(index_path: str | Path, batch_id: str) -> set[str]:
    payload = load_frozen_index(index_path)
    prefix = f"waymo/database/batches/{batch_id}/"
    selected: set[str] = set()
    for item in payload["source_inventory"]["waymo"]["scenarios"]:
        relative_path = str(item["relative_path"])
        if relative_path.startswith(prefix):
            selected.add(relative_path.removeprefix(prefix))
    if not selected:
        raise ValueError(f"frozen index contains no selected scenarios for batch {batch_id}")
    return selected


def prune_waymo_batch(
    database_path: str | Path,
    *,
    index_path: str | Path,
    batch_id: str,
) -> tuple[int, int]:
    """Keep selected scenario files and remove unselected converted files."""

    database = Path(database_path).expanduser().resolve()
    if not database.is_dir():
        raise FileNotFoundError(f"Waymo staging database does not exist: {database}")
    selected = _selected_relative_paths(index_path, batch_id)
    converted = tuple(sorted(database.rglob("sd_*.pkl")))
    relative = {path: path.relative_to(database).as_posix() for path in converted}
    missing = sorted(path for path in selected if not (database / path).is_file())
    if missing:
        raise FileNotFoundError(
            f"frozen Waymo scenarios are missing from batch {batch_id}: {missing[:5]}"
        )
    removed = 0
    for path, relative_path in relative.items():
        if relative_path not in selected:
            path.unlink()
            removed += 1
    return len(converted) - removed, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    kept, removed = prune_waymo_batch(
        args.database,
        index_path=args.index,
        batch_id=args.batch_id,
    )
    print(json.dumps({"batch_id": args.batch_id, "kept": kept, "removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
