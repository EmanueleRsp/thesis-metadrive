"""`REQ-EF-15` / open item `D14`: the corridor diagnostic, reported per episode.

`route_outside_fraction` has been computed on every step since `REQ-EF-15` and
read by nothing. That mattered because `R4` credits arc-length advance of a
nearest-point projection while mission success is a crossing of one frozen finite
gate, and nothing links the two: an ego on a legal same-direction carriageway
beside its route keeps earning progress with the gate uncrossed, and no cost or
termination objects. These tests cover the reporting path that makes the question
answerable from a run instead of from an argument.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.route_adherence_diagnostics import (
    ROUTE_ADHERENCE_EPISODE_COLUMNS,
    route_adherence_episode_fields,
)


def test_eval_episodes_schema_declares_the_columns() -> None:
    schema = CSVRecorder.SCHEMAS["eval_episodes.csv"]

    for column in ROUTE_ADHERENCE_EPISODE_COLUMNS:
        assert column in schema, column
    assert len(schema) == len(set(schema))


def test_the_headline_column_is_the_longest_run() -> None:
    """Guards the one column a reader must not lose.

    A mean cannot separate clipping the inside of four corners from driving
    eighty consecutive steps on the wrong carriageway, and only the second is the
    failure this diagnostic exists to detect. If the column set is ever trimmed,
    this is the entry whose removal breaks the measurement.
    """

    assert "route_fully_outside_max_run" in ROUTE_ADHERENCE_EPISODE_COLUMNS


def test_fields_are_read_from_the_episode_metadata() -> None:
    metadata = {
        "route_outside_evaluated_steps": 120,
        "route_outside_steps": 9,
        "route_fully_outside_steps": 6,
        "route_fully_outside_max_run": 5,
        "mean_route_outside_fraction": 0.04,
        "mean_route_adherence": 0.96,
        "l4_clip_binding_steps": 2,
        "l5_reached_steps": 118,
        "mean_ego_speed_mps": 7.5,
        "scenario_uid": "ignored",
    }

    fields = route_adherence_episode_fields(metadata)

    assert set(fields) == set(ROUTE_ADHERENCE_EPISODE_COLUMNS)
    assert fields["route_fully_outside_max_run"] == 5
    assert fields["mean_route_adherence"] == pytest.approx(0.96)
    assert fields["l4_clip_binding_steps"] == 2
    assert "scenario_uid" not in fields


@pytest.mark.parametrize("metadata", [{}, None, "not a mapping", {"unrelated": 1}])
def test_absent_or_malformed_metadata_yields_empty_cells_not_zeros(metadata: object) -> None:
    """A run without these counters must write a well-formed row of blanks.

    ``None`` and ``0`` are different claims. Zero says the ego never left its
    corridor; blank says nobody looked. A run recorded before these counters
    existed, or an environment stack with no Rulebook wrapper, must produce the
    second.
    """

    fields = route_adherence_episode_fields(metadata)

    assert set(fields) == set(ROUTE_ADHERENCE_EPISODE_COLUMNS)
    assert all(value is None for value in fields.values())


def test_recorder_writes_the_columns(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    recorder.append_row(
        "eval_episodes.csv",
        {
            "episode_id": 1,
            **route_adherence_episode_fields(
                {"route_fully_outside_max_run": 12, "mean_route_adherence": 0.5}
            ),
        },
    )

    rows = list(csv.DictReader((tmp_path / "eval_episodes.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["route_fully_outside_max_run"] == "12"
    assert float(rows[0]["mean_route_adherence"]) == pytest.approx(0.5)
    # Not measured on this row, and therefore blank rather than zero.
    assert rows[0]["route_outside_evaluated_steps"] == ""
