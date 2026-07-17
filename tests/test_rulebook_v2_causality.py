from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.context.live_adapter import LiveSnapshotAdapter, LiveSnapshotSources
from thesis_rl.rulebook.v2.context.static_sources import StaticRecordSources
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot
from shapely.geometry import Polygon


def _actor(actor_id: str) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        ActorClass.VEHICLE,
        (0.0, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        Polygon(((-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5))),
        "lane",
        20.0,
    )


def test_live_snapshot_guard_calls_only_current_step_providers() -> None:
    calls: list[str] = []
    sources = LiveSnapshotSources(
        scenario_id=lambda env: calls.append("scenario_id") or env["scenario_id"],
        step_index=lambda env: calls.append("step_index") or env["step_index"],
        sim_time_s=lambda env: calls.append("sim_time_s") or env["sim_time_s"],
        ego=lambda _env: calls.append("ego") or _actor("ego"),
        actors=lambda _env: calls.append("actors") or (),
        contact_onset_records=lambda _env: calls.append("contact") or (),
        active_contact_ids=lambda _env: calls.append("contacts") or frozenset(),
        signal_states_by_physical_id=lambda _env: calls.append("signals") or {},
    )
    snapshot = LiveSnapshotAdapter(sources).capture(
        {
            "scenario_id": "s",
            "step_index": 1,
            "sim_time_s": 0.1,
            "future_track": object(),
            "future_route": object(),
            "future_maneuver": object(),
        }
    )
    assert snapshot.sim_time_s == 0.1
    assert "future_track" not in calls
    assert "future_route" not in calls
    assert "future_maneuver" not in calls


def test_source_registries_reject_future_track_route_and_maneuver_providers() -> None:
    live = {name: lambda _env: None for name in LiveSnapshotSources.__dataclass_fields__}
    for forbidden in ("future_track", "future_route", "future_maneuver"):
        candidate = dict(live)
        candidate[forbidden] = lambda _env: None
        with pytest.raises(ValueError, match="Unknown"):
            LiveSnapshotSources.from_mapping(candidate)

    static = {name: lambda _record: None for name in StaticRecordSources.__dataclass_fields__}
    static["future_route"] = lambda _record: None
    with pytest.raises(ValueError, match="Unknown"):
        StaticRecordSources.from_mapping(static)
