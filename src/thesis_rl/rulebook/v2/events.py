"""Control-step-safe contact onset buffer used by the local simulator hook."""

from __future__ import annotations

from threading import RLock

from thesis_rl.rulebook.v2.types import ContactOnsetRecord


class ContactOnsetBuffer:
    """Thread-safe buffer preserving all contact points and deduplicating actors."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: list[ContactOnsetRecord] = []
        self._active_actor_ids: set[str] = set()

    def clear_control_step(self) -> None:
        with self._lock:
            self._records.clear()
            self._active_actor_ids.clear()

    def record_onset(self, record: ContactOnsetRecord) -> None:
        with self._lock:
            self._active_actor_ids.add(record.actor_id)
            self._records.append(record)

    def drain(self) -> tuple[ContactOnsetRecord, ...]:
        with self._lock:
            records = tuple(self._records)
            self._records.clear()
            self._active_actor_ids.clear()
            return records
