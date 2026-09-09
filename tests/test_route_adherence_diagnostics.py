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


def test_every_eval_episodes_write_site_emits_the_columns() -> None:
    """`D14`. Regression: the official panel path wrote these cells empty.

    The columns were added to the four write sites inside the training and
    intermediate-evaluation loops and not to the fifth, `final_panels.py`, which
    is the one `run_scenarionet_final_panels` uses — and therefore the one
    `cli.evaluate`, the training loop's ScenarioNet final-eval branch, the
    curriculum driver and `evaluate_constant_action_baseline.py` all reach.
    `CSVRecorder.append_row` fills a missing key with `None` without complaint,
    so the schema declared the columns and every official panel row carried
    blanks. `route_fully_outside_max_run` is the statistic that distinguishes
    clipping four corners from eighty consecutive steps on another carriageway,
    so it was reading nothing in exactly the place a decision would consult it.

    Asserted over the **call sites** rather than over the helper, because the
    defect was a forgotten site and a test of the helper cannot see one. Written
    as a source scan for the same reason: reaching every one of these five sites
    at runtime needs a simulator, a trained checkpoint and hours of panel
    evaluation, and a guard that expensive is a guard that does not run.
    """

    source_root = Path(__file__).resolve().parents[1] / "src"
    sites: dict[Path, int] = {}
    emitters: dict[Path, int] = {}
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # An `append_row` whose first argument is this file, i.e. a write site,
        # rather than a schema declaration or a reader naming the same file.
        writes = (
            text.count('append_row(\n            "eval_episodes.csv"')
            + text.count('append_row(\n                "eval_episodes.csv"')
            + text.count('append_row(\n                    "eval_episodes.csv"')
        )
        if not writes:
            continue
        relative = path.relative_to(source_root)
        sites[relative] = writes
        emitters[relative] = text.count("route_adherence_episode_fields(")

    assert sites, "found no eval_episodes.csv write sites; the guard would pass vacuously"
    silent = {
        path: (count, emitters[path]) for path, count in sites.items() if emitters[path] < count
    }
    assert not silent, (
        "these modules append eval_episodes.csv rows without the route-adherence "
        f"columns, so those cells will be silently empty: {silent}"
    )
    # Pinned so that adding a sixth write site is a deliberate act rather than a
    # place these columns can go missing again unnoticed.
    assert sum(sites.values()) == 5, f"write-site count moved: {sites}"
