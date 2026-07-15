"""Control-step-safe contact onset buffer used by the local simulator hook."""

from __future__ import annotations

from threading import RLock

from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.types import ContactOnsetRecord


@dataclass(frozen=True, slots=True)
class ZoneTransitionEvents:
    pre_occupied: bool
    post_occupied: bool
    entered: bool
    exited: bool


@dataclass(frozen=True, slots=True)
class ContactSetTransition:
    """Birth/termination partition for two consecutive control snapshots."""

    born_actor_ids: frozenset[str]
    terminated_actor_ids: frozenset[str]
    persistent_actor_ids: frozenset[str]


def derive_contact_transition(
    previous_active_ids: frozenset[str],
    current_active_ids: frozenset[str],
) -> ContactSetTransition:
    """Derive contact births and terminations without relying on crash flags."""

    previous = frozenset(previous_active_ids)
    current = frozenset(current_active_ids)
    if any(
        not isinstance(actor_id, str) or not actor_id
        for actor_id in (*previous, *current)
    ):
        raise ValueError("Active contact IDs must be non-empty")
    return ContactSetTransition(
        born_actor_ids=current - previous,
        terminated_actor_ids=previous - current,
        persistent_actor_ids=previous & current,
    )


def detect_zone_transition(
    *,
    pre_ego_footprint: BaseGeometry,
    post_ego_footprint: BaseGeometry,
    swept_front_bumper: BaseGeometry,
    zone: BaseGeometry,
    pre_front_s_m: float,
    post_front_s_m: float,
    zone_entry_s_m: float,
    epsilon_delta_m: float = 5.0e-2,
) -> ZoneTransitionEvents:
    """Pure pre/post occupancy and swept-front entry event detector."""

    if any(
        geometry.is_empty or not geometry.is_valid
        for geometry in (pre_ego_footprint, post_ego_footprint, swept_front_bumper, zone)
    ):
        raise ValueError("Zone event detector requires valid non-empty geometries")
    if epsilon_delta_m <= 0.0:
        raise ValueError("epsilon_delta_m must be positive")
    pre_occupied = pre_ego_footprint.intersection(zone).area > 0.0
    post_occupied = post_ego_footprint.intersection(zone).area > 0.0
    entered = (
        not pre_occupied
        and pre_front_s_m < zone_entry_s_m - epsilon_delta_m
        and post_front_s_m >= zone_entry_s_m - epsilon_delta_m
        and swept_front_bumper.intersects(zone)
    )
    return ZoneTransitionEvents(
        pre_occupied=pre_occupied,
        post_occupied=post_occupied,
        entered=entered,
        exited=pre_occupied and not post_occupied,
    )


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
