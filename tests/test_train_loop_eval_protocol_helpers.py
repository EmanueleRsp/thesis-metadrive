"""EVAL-PROTOCOL REQ-006/REQ-012 helper unit tests (train_loop.py)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from thesis_rl.runtime.loops import train_loop
from thesis_rl.runtime.loops.train_loop import (
    _checkpoint_hash,
    _data_abort_coverage_fields,
    _make_tracked_subset_render_gate,
    _tracked_subset_uids_from_resolved_env_cfg,
)


def test_data_abort_coverage_fields_maps_computed_dict() -> None:
    metrics = {
        "data_abort_coverage": {
            "attempted": 4,
            "valid": 3,
            "invalid": 1,
            "invalid_episodes": [{"episode_idx": 1, "scenario_uid": "x", "reason_code": "r"}],
        }
    }
    fields = _data_abort_coverage_fields(metrics)
    assert fields["data_abort_attempted"] == 4
    assert fields["data_abort_valid"] == 3
    assert fields["data_abort_invalid"] == 1
    assert fields["data_abort_coverage"] == 0.75


def test_data_abort_coverage_fields_missing_is_all_none() -> None:
    fields = _data_abort_coverage_fields({})
    assert fields == {
        "data_abort_attempted": None,
        "data_abort_valid": None,
        "data_abort_invalid": None,
        "data_abort_coverage": None,
    }


def test_data_abort_coverage_fields_zero_attempted_no_division_by_zero() -> None:
    fields = _data_abort_coverage_fields(
        {"data_abort_coverage": {"attempted": 0, "valid": 0, "invalid": 0, "invalid_episodes": []}}
    )
    assert fields["data_abort_coverage"] is None


def test_checkpoint_hash_matches_independent_sha256(tmp_path: Path) -> None:
    import hashlib

    stem = tmp_path / "final"
    zip_path = stem.with_suffix(".zip")
    zip_path.write_bytes(b"fixture checkpoint bytes")

    expected = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert _checkpoint_hash(stem) == expected


def test_checkpoint_hash_missing_file_returns_none(tmp_path: Path) -> None:
    assert _checkpoint_hash(tmp_path / "missing") is None


def test_every_evals_csv_append_row_call_wires_data_abort_coverage_fields() -> None:
    """EVAL-PROTOCOL REQ-012 regression: every ``evals.csv`` write (periodic
    validation and final test alike) must persist the data-abort coverage
    fields via ``_data_abort_coverage_fields``. This guards against a call
    site being added/reverted without the helper wired in, which previously
    happened for the three periodic-validation call sites (curriculum-eval
    async, no-curriculum, and staged-curriculum-gate blocks)."""
    source = inspect.getsource(train_loop)
    call_starts = [m.start() for m in re.finditer(r'recorder\.append_row\(\s*\n\s*"evals\.csv",', source)]
    assert len(call_starts) >= 4, (
        f"expected at least 4 'evals.csv' append_row call sites, found {len(call_starts)}"
    )
    next_call_starts = call_starts[1:] + [len(source)]
    for start, next_start in zip(call_starts, next_call_starts):
        block = source[start:next_start]
        assert "_data_abort_coverage_fields(metrics)" in block, (
            "an 'evals.csv' append_row call is missing "
            f"_data_abort_coverage_fields(metrics) wiring:\n{block[:400]}"
        )


# --- REQ-014/DEC-014 (amended 2026-07-25): tracked-subset GIF render
# cadence gate and config-driven tracked-uid resolution. ---


def test_tracked_subset_render_gate_fires_at_first_multiple_of_interval() -> None:
    due = _make_tracked_subset_render_gate(interval=100_000)
    # Default eval_interval=25,000: 4 periodic evals per 100k window.
    assert due(25_000) is False
    assert due(50_000) is False
    assert due(75_000) is False
    assert due(100_000) is True
    assert due(125_000) is False
    assert due(150_000) is False
    assert due(175_000) is False
    assert due(200_000) is True


def test_tracked_subset_render_gate_fires_once_per_window_not_on_every_call_after() -> None:
    due = _make_tracked_subset_render_gate(interval=100_000)
    assert due(100_000) is True
    assert due(100_000) is False  # already rendered this window


def test_tracked_subset_render_gate_handles_non_divisor_eval_interval() -> None:
    # eval_interval=30,000 does not evenly divide 100,000: the gate must
    # still fire once, at the first eval boundary at or after each
    # 100k multiple, rather than silently never firing again.
    due = _make_tracked_subset_render_gate(interval=100_000)
    steps = [30_000, 60_000, 90_000, 120_000, 150_000, 180_000, 210_000]
    fired = [due(s) for s in steps]
    assert fired == [False, False, False, True, False, False, True]


def test_tracked_subset_render_gate_rejects_non_positive_interval() -> None:
    import pytest

    with pytest.raises(ValueError):
        _make_tracked_subset_render_gate(interval=0)


def test_tracked_subset_uids_from_resolved_env_cfg_none_when_no_provider() -> None:
    assert _tracked_subset_uids_from_resolved_env_cfg({}) == ()
    assert _tracked_subset_uids_from_resolved_env_cfg(None) == ()
    assert _tracked_subset_uids_from_resolved_env_cfg({"provider": {}}) == ()


def test_tracked_subset_uids_from_resolved_env_cfg_none_when_manifest_path_unset() -> None:
    resolved = {"provider": {"panel_manifest_path": None}}
    assert _tracked_subset_uids_from_resolved_env_cfg(resolved) == ()
    resolved_empty = {"provider": {"panel_manifest_path": ""}}
    assert _tracked_subset_uids_from_resolved_env_cfg(resolved_empty) == ()


def test_tracked_subset_uids_from_resolved_env_cfg_reads_manifest(tmp_path: Path) -> None:
    from thesis_rl.scenarios.arms import ARMS
    from thesis_rl.scenarios.panel_manifest import build_balanced_panel, save_panel_manifest
    from thesis_rl.scenarios.records import ScenarioRecord

    records = []
    counter = 0
    for arm in ARMS:
        for _ in range(5):
            records.append(
                ScenarioRecord(
                    scenario_uid=f"waymo:v1:{arm}:{counter}",
                    scenario_id=str(counter),
                    source="waymo",  # type: ignore[arg-type]
                    relative_path=f"waymo/database/{counter}.pkl",
                    official_split="training_20s",
                    source_log_id=f"log-{counter}",
                    source_scenario_id=str(counter),
                    dataset_version="v1",
                    converter_version="converter",
                    split="validation",  # type: ignore[arg-type]
                    runtime_index=counter,
                    length=100,
                    pg_profile=None,
                    pg_seed=None,
                    map_id="S",
                    primary_arm=arm,
                    tags=(),
                    signal_reliability="not_applicable",
                    validation_status="valid",
                    validation_warnings=(),
                )
            )
            counter += 1
    manifest = build_balanced_panel(
        records, split="validation", size=len(ARMS) * 3, seed=1, tracked_subset_count_per_arm=2
    )
    manifest_path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, manifest_path)

    resolved = {"provider": {"panel_manifest_path": str(manifest_path)}}
    assert _tracked_subset_uids_from_resolved_env_cfg(resolved) == manifest.tracked_subset_uids

    resolved_nested = {"config": {"provider": {"panel_manifest_path": str(manifest_path)}}}
    assert _tracked_subset_uids_from_resolved_env_cfg(resolved_nested) == manifest.tracked_subset_uids


# --- REQ-014/DEC-014 (amended 2026-07-25): periodic tracked-subset GIF
# render wiring at the periodic-validation `evaluate()` call site. ---


def test_periodic_evaluate_call_site_wires_tracked_subset_artifact_factory() -> None:
    """The periodic-validation `eval_agent.evaluate(...)` call must receive
    `artifact_recorder_factory=periodic_tracked_subset_artifact_factory`,
    computed from `maybe_build_periodic_tracked_subset_recorder_factory`
    gated by `periodic_tracked_subset_render_due`
    (`_make_tracked_subset_render_gate`). This is a regression guard against
    the wiring being present at the manifest-selection call site (post-eval)
    but never threaded into the live rollout itself, which is what actually
    renders GIF pixels."""
    source = inspect.getsource(train_loop)
    assert "maybe_build_periodic_tracked_subset_recorder_factory" in source
    periodic_evaluate_start = source.index("metrics = eval_agent.evaluate(")
    periodic_evaluate_end = source.index("eval_env.close()", periodic_evaluate_start)
    periodic_evaluate_block = source[periodic_evaluate_start:periodic_evaluate_end]
    assert "artifact_recorder_factory=periodic_tracked_subset_artifact_factory" in periodic_evaluate_block


def test_periodic_render_gate_called_exactly_once_per_periodic_eval() -> None:
    """`_make_tracked_subset_render_gate`'s `due(...)` closure has side
    effects (it advances its internal threshold); calling it twice per
    periodic evaluation would silently skip the next due window. Guard
    against a future edit re-introducing a second `tracked_subset_render_due(`
    call inside the periodic-validation branch."""
    source = inspect.getsource(train_loop)
    call_count = source.count("tracked_subset_render_due(")
    assert call_count == 1, (
        f"expected exactly one tracked_subset_render_due(...) call site, found {call_count}"
    )


def test_tracked_subset_render_interval_derives_from_eval_interval_and_factor() -> None:
    """REQ-014/DEC-014 (amended 2026-07-31): the periodic tracked-subset GIF
    cadence must scale with `experiment.eval_interval`, via
    `video.tracked_subset_render_interval_factor` (default 3), instead of a
    fixed timestep constant -- otherwise the cadence silently drifts out of
    proportion whenever a run profile's `eval_interval` changes."""
    source = inspect.getsource(train_loop)
    assert (
        "tracked_subset_render_interval = tracked_subset_render_interval_factor * eval_interval"
        in source
    )
    assert '"tracked_subset_render_interval_factor", 3' in source


def test_final_evaluate_call_site_unaffected_by_periodic_wiring() -> None:
    """Regression: the final-test `eval_agent.evaluate(...)` call must keep
    using `final_eval_artifact_factory` (unconditional full-panel
    rendering), never the periodic tracked-subset factory."""
    source = inspect.getsource(train_loop)
    final_evaluate_start = source.index('progress_description="Test episodes",')
    final_evaluate_block = source[final_evaluate_start : final_evaluate_start + 200]
    assert "artifact_recorder_factory=final_eval_artifact_factory" in final_evaluate_block
    assert "periodic_tracked_subset_artifact_factory" not in final_evaluate_block
