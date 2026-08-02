from __future__ import annotations

import pytest

from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection


def test_mission_record_hash_is_deterministic_and_round_trips() -> None:
    span = LaneSpan("a", 0.0, 10.0)
    gate = DirectedGate("gate:a", (span,), "a", 10.0)
    record = DrivingMissionRecord(
        "uid", "mission-builder-v1", (MissionSection("section:0", span, (span,), gate),), gate
    )

    assert record.mission_hash == DrivingMissionRecord.from_dict(record.to_dict()).mission_hash
    assert (
        record.mission_hash
        == DrivingMissionRecord(
            "uid", "mission-builder-v1", (MissionSection("section:0", span, (span,), gate),), gate
        ).mission_hash
    )


def test_invalid_mission_record_is_rejected_with_stable_error() -> None:
    with pytest.raises(ValueError, match="end_s_m"):
        LaneSpan("a", 1.0, 0.0)
