from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from thesis_rl.cli.scenarios.pipeline_config import _validate_payload
from thesis_rl.envs.factory import _runtime_rulebook_records
from thesis_rl.scenarios.catalog import ScenarioCatalog, ScenarioCatalogEntry
from thesis_rl.scenarios.pipeline import (
    assign_catalog_runtime_indices,
    assign_arm_balanced_splits_to_targets,
    assign_source_splits,
    assign_source_splits_to_targets,
    arm_source_balance_diagnostics,
    arm_selection_diagnostics,
    assert_runtime_split_contract,
    balance_arm_distribution,
    group_id_for_entry,
)
from thesis_rl.scenarios.reports import compute_pg_replenishment_report
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


_PIPELINE_CONFIG_VALUES = {
    "SCENARIONET_PG_COUNT": "1",
    "SCENARIONET_PG_SEED_START": "920000",
    "SCENARIONET_PG_WORKERS": "1",
    "SCENARIONET_PG_MAX_COMPOSITION_REPLENISHMENT_BLOCKS": "1",
    "SCENARIONET_PG_REPLENISHMENT_CANDIDATE_BUDGET": "1",
    "SCENARIONET_SPLIT_SEED": "0",
    "SCENARIONET_AUTO_SPLIT": "true",
    "SCENARIONET_RULEBOOK_V2_ENABLED": "true",
    "SCENARIONET_RULEBOOK_V2_WORKERS": "1",
    "SCENARIONET_WAYMO_TRAIN_TARGET": "1",
    "SCENARIONET_WAYMO_VALIDATION_TARGET": "1",
    "SCENARIONET_WAYMO_TEST_TARGET": "1",
    "SCENARIONET_PG_TRAIN_TARGET": "1",
    "SCENARIONET_PG_VALIDATION_TARGET": "1",
    "SCENARIONET_PG_TEST_TARGET": "1",
    "SCENARIONET_CHECK_WORKERS": "1",
    "SCENARIONET_RUN_SIMULATION_CHECK": "false",
    "SCENARIONET_WAYMO_AUTO_EXPAND": "true",
    "SCENARIONET_WAYMO_BATCH_SHARDS": "1",
    "SCENARIONET_WAYMO_MAX_NEW_SHARDS": "1",
    "SCENARIONET_WAYMO_WORKERS": "1",
    "SCENARIONET_WAYMO_KEEP_RAW_BATCHES": "false",
    "SCENARIONET_WAYMO_REQUIRED_A4_VRU": "0",
}


def _run_mocked_pipeline(
    tmp_path: Path,
    *,
    fail_module: str | None = None,
    split_failures: int = 0,
    config_overrides: dict[str, str] | None = None,
    skip_waymo: bool = True,
) -> subprocess.CompletedProcess[str]:
    test_repo = tmp_path / "repo"
    scripts_dir = test_repo / "scripts"
    scripts_dir.mkdir(parents=True)
    pipeline_script = scripts_dir / "prepare_scenarionet_dataset.sh"
    pipeline_script.write_text(
        Path("scripts/prepare_scenarionet_dataset.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    host_data_root = tmp_path / "data" / "scenarionet"
    (host_data_root / "pg" / "database").mkdir(parents=True)
    pilot = host_data_root / "pg" / "pilot" / "pg_pilot_report.json"
    pilot.parent.mkdir(parents=True)
    pilot.write_text("{}\n", encoding="utf-8")
    replenishment_report = host_data_root / "pg" / "replenishment_report.json"
    replenishment_report.write_text(
        json.dumps(
            {
                "hard_count_shortfall": 0,
                "selection_error": (
                    "runtime split target mismatch for pg/train: requested 1, selected 0"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    command_log = tmp_path / "docker-commands.log"
    make_log = tmp_path / "make-commands.log"
    split_state = tmp_path / "split-state"
    (test_repo / ".env").write_text(
        "\n".join(
            (
                f"SCENARIONET_SKIP_WAYMO={'true' if skip_waymo else 'false'}",
                f"SCENARIONET_HOST_DATA_ROOT={host_data_root}",
                f"FAKE_DOCKER_LOG={command_log}",
                f"FAKE_MAKE_LOG={make_log}",
                f"FAKE_SPLIT_STATE={split_state}",
                f"MOCK_SPLIT_FAILURES={split_failures}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    config_values = _PIPELINE_CONFIG_VALUES | (config_overrides or {})
    config_output = "".join(
        f"printf '%s\\t%s\\n' '{key}' '{value}'\n" for key, value in config_values.items()
    )
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_DOCKER_LOG"\n'
        'if [[ " $* " == *" thesis_rl.cli.scenarios.pipeline_config "* ]]; then\n'
        f"{config_output}"
        "  exit 0\n"
        "fi\n"
        'if [[ " $* " == *" python -c "* ]]; then\n'
        "  printf '%s\\t%s\\n' '0' "
        "'runtime split target mismatch for pg/train: requested 1, selected 0'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ " $* " == *" thesis_rl.cli.scenarios.plan_pg_replenishment "* ]]; then\n'
        "  printf '%s\\n' '{\"P5_complex_mixed\": 1}'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ " $* " == *" thesis_rl.cli.scenarios.waymo_pool_status "* ]]; then\n'
        "  printf '%s\\t%s\\n' 'SCENARIONET_WAYMO_POOL_COMPLETE' 'false'\n"
        "  exit 0\n"
        "fi\n"
        "for module in build_catalog filter_rulebook_v2_catalog build_splits "
        "compute_arm_thresholds build_runtime_databases; do\n"
        '  if [[ " $* " == *" thesis_rl.cli.scenarios.${module} "* ]]; then\n'
        '    if [[ "${FAIL_PIPELINE_MODULE:-}" == "$module" ]]; then\n'
        '      echo "synthetic ${module} failure" >&2\n'
        "      exit 17\n"
        "    fi\n"
        '    if [[ "$module" == build_splits && "${MOCK_SPLIT_FAILURES:-0}" -gt 0 ]]; then\n'
        "      split_attempt=0\n"
        '      [[ ! -f "$FAKE_SPLIT_STATE" ]] || split_attempt=$(<"$FAKE_SPLIT_STATE")\n'
        "      split_attempt=$((split_attempt + 1))\n"
        '      printf \'%s\\n\' "$split_attempt" > "$FAKE_SPLIT_STATE"\n'
        "      if (( split_attempt <= MOCK_SPLIT_FAILURES )); then\n"
        '        echo "synthetic split infeasibility" >&2\n'
        "        exit 23\n"
        "      fi\n"
        "    fi\n"
        '    if [[ " $* " != *" --overwrite "* ]]; then\n'
        '      echo "refusing to overwrite existing ${module} artifact" >&2\n'
        "      exit 18\n"
        "    fi\n"
        "  fi\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    fake_make = fake_bin / "make"
    fake_make.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s max_new_shards=%s\\n' \"$*\" "
        '"${WAYMO_MAX_NEW_SHARDS:-unset}" >> "$FAKE_MAKE_LOG"\n',
        encoding="utf-8",
    )
    fake_make.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    if fail_module is not None:
        env["FAIL_PIPELINE_MODULE"] = fail_module
    return subprocess.run(
        ["bash", str(pipeline_script)],
        cwd=test_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pipeline_restart_overwrites_only_derived_stage_outputs(tmp_path: Path) -> None:
    result = _run_mocked_pipeline(tmp_path)

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "docker-commands.log").read_text(encoding="utf-8").splitlines()
    for module in (
        "build_catalog",
        "filter_rulebook_v2_catalog",
        "build_splits",
        "compute_arm_thresholds",
        "build_runtime_databases",
    ):
        invocation = next(line for line in commands if f"thesis_rl.cli.scenarios.{module}" in line)
        assert "--overwrite" in invocation
    assert not any("thesis_rl.cli.scenarios.generate_pg_dataset" in line for line in commands)


def test_pipeline_failure_prints_actionable_summary(tmp_path: Path) -> None:
    result = _run_mocked_pipeline(tmp_path, fail_module="build_catalog")

    assert result.returncode == 17
    assert "ScenarioNet pipeline failure" in result.stderr
    assert "Stage: [3/9] Building catalog and groups" in result.stderr
    assert "Operation: rebuilding the unified candidate catalog and group mapping" in result.stderr
    assert "Failed command:" in result.stderr
    assert "Suggested action:" in result.stderr
    assert "Retry command: make scenarionet-pipeline" in result.stderr


def test_pg_replenishment_does_not_consume_waymo_batch_cap(tmp_path: Path) -> None:
    result = _run_mocked_pipeline(
        tmp_path,
        split_failures=2,
        config_overrides={"SCENARIONET_WAYMO_MAX_NEW_SHARDS": "4"},
        skip_waymo=False,
    )

    assert result.returncode == 0, result.stderr
    make_commands = (tmp_path / "make-commands.log").read_text(encoding="utf-8")
    assert "scenarionet-pg-replenish" in make_commands
    assert "waymo-expand max_new_shards=4" in make_commands


def test_pipeline_rejects_zero_waymo_batch_size_before_arithmetic(tmp_path: Path) -> None:
    result = _run_mocked_pipeline(
        tmp_path,
        config_overrides={"SCENARIONET_WAYMO_BATCH_SHARDS": "0"},
    )

    assert result.returncode == 2
    assert "Reason: Waymo batch size must be a positive integer: 0" in result.stderr
    assert "Operation: validating the resolved feasibility" in result.stderr
    assert "division by 0" not in result.stderr


def test_pipeline_config_rejects_invalid_types_and_ranges(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("conf/scenarios/pipeline_v1.yaml").read_text(encoding="utf-8"))
    payload["waymo"]["batch_shards"] = 0
    with pytest.raises(ValueError, match=r"waymo\.batch_shards must be a positive integer"):
        _validate_payload(payload)

    payload = yaml.safe_load(Path("conf/scenarios/pipeline_v1.yaml").read_text(encoding="utf-8"))
    payload["checks"]["simulation"] = "true"
    invalid_config = tmp_path / "invalid_pipeline.yaml"
    invalid_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "thesis_rl.cli.scenarios.pipeline_config",
            "--config",
            str(invalid_config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "error: checks.simulation must be a boolean" in result.stderr
    assert "Traceback" not in result.stderr


def test_pipeline_pg_report_parser_handles_composition_failure(tmp_path: Path) -> None:
    script = Path("scripts/prepare_scenarionet_dataset.sh").read_text(encoding="utf-8")
    match = re.search(
        r"\n\s+'(import json,sys; p=sys\.argv\[1\].*?)'\s+\\\n",
        script,
        flags=re.DOTALL,
    )
    assert match is not None

    report = tmp_path / "replenishment_report.json"
    report.write_text(
        json.dumps(
            {
                "hard_count_shortfall": 0,
                "selection_error": "runtime split target mismatch for pg/train: requested 1000, selected 550",
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", match.group(1), str(report)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.rstrip("\n") == (
        "0\truntime split target mismatch for pg/train: requested 1000, selected 550"
    )


def test_pipeline_host_data_root_follows_host_data_dir_mount() -> None:
    script = Path("scripts/prepare_scenarionet_dataset.sh").read_text(encoding="utf-8")

    assert 'host_data_dir="${HOST_DATA_DIR:-${repo_root}/data}"' in script
    assert (
        'host_data_root="${SCENARIONET_HOST_DATA_ROOT:-${host_data_dir%/}/scenarionet}"' in script
    )
    assert 'host_data_root="${repo_root}/${host_data_root}"' in script
    assert (
        'host_data_root="${SCENARIONET_HOST_DATA_ROOT:-${repo_root}/data/scenarionet}"'
        not in script
    )


def test_waymo_scripts_normalize_host_mount_paths() -> None:
    prepare_waymo = Path("scripts/prepare_waymo.sh").read_text(encoding="utf-8")
    expand_waymo = Path("scripts/expand_waymo_pool.sh").read_text(encoding="utf-8")

    assert 'host_data_dir="${repo_root}/${host_data_dir}"' in prepare_waymo
    assert 'raw_dir="${repo_root}/${raw_dir}"' in prepare_waymo
    assert 'host_data_dir="${repo_root}/${host_data_dir}"' in expand_waymo
    assert 'raw_root="${repo_root}/${raw_root}"' in expand_waymo


def test_pg_replenishment_uses_cpu_pipeline_service_and_resolved_workers() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split("scenarionet-pg-replenish:", maxsplit=1)[1].split(
        "scenarionet-recatalog:", maxsplit=1
    )[0]

    assert "docker compose run --rm dataset-pipeline" in target
    assert '--workers "$(SCENARIONET_PG_WORKERS)"' in target
    assert "docker compose run --rm dev" not in target


def test_recatalog_never_mutates_source_datasets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split("scenarionet-recatalog:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]

    assert "SCENARIONET_SKIP_PG=true" in target
    assert "SCENARIONET_SKIP_WAYMO=true" in target
    assert "SCENARIONET_OVERWRITE" not in target


def _entry(source: str, index: int) -> ScenarioCatalogEntry:
    scenario_id = f"{source}-{index}"
    record = ScenarioRecord(
        scenario_uid=f"{source}:v1:{index}",
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{scenario_id}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=f"log-{index}" if source == "waymo" else None,
        source_scenario_id=scenario_id if source == "waymo" else None,
        dataset_version="v1",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=10,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        length=10,
        route_length_m=10.0,
        topology_tag="simple",
        has_intersection=False,
        has_merge_or_roundabout=False,
        has_route_traffic_light=False,
        has_route_stop_sign=False,
        has_route_crosswalk=False,
        signal_reliability="not_applicable",
        has_vehicle=False,
        has_pedestrian=False,
        has_cyclist=False,
        relevant_agents_q90=0.0,
        relevant_vehicles_q90=0.0,
        min_vehicle_distance_m=None,
        min_vru_distance_to_route_m=None,
        low_traffic=True,
        dense_traffic=False,
        vru_interaction=False,
    )
    return ScenarioCatalogEntry(record, features)


def test_pipeline_assigns_source_splits_and_runtime_indices() -> None:
    entries = tuple(_entry(source, index) for source in ("waymo", "pg") for index in range(2))
    split = assign_source_splits(
        entries,
        counts={
            "waymo": {"train": 1, "validation": 0, "test": 1},
            "pg": {"train": 1, "validation": 0, "test": 1},
        },
        seed=7,
    )
    assert {entry.record.split for entry in split} == {"train", "test"}

    indexed = assign_catalog_runtime_indices(split)
    for split_name in ("train", "test"):
        indices = sorted(
            entry.record.runtime_index for entry in indexed if entry.record.split == split_name
        )
        assert indices == [0, 1]


def test_waymo_training_shard_is_not_used_as_leakage_group() -> None:
    entry = _entry("waymo", 0)
    shard_record = replace(
        entry.record,
        source_log_id="training_20s.tfrecord-00011-of-01000",
    )
    true_log_record = replace(entry.record, source_log_id="waymo-log-1")

    assert (
        group_id_for_entry(ScenarioCatalogEntry(shard_record, entry.features))
        == "scenario:waymo:v1:0"
    )
    assert (
        group_id_for_entry(ScenarioCatalogEntry(true_log_record, entry.features)) == "waymo-log-1"
    )


def test_pipeline_auto_split_preserves_whole_groups() -> None:
    entries = tuple(_entry(source, index) for source in ("waymo", "pg") for index in range(10))
    split = assign_source_splits_to_targets(
        entries,
        targets={
            "waymo": {"train": 2, "validation": 2, "test": 2},
            "pg": {"train": 2, "validation": 2, "test": 2},
        },
        seed=3,
    )
    for source in ("waymo", "pg"):
        source_entries = [entry for entry in split if entry.record.source == source]
        assert len(source_entries) == 6
        assert {entry.record.split for entry in source_entries} == {"train", "validation", "test"}


def test_pipeline_auto_split_prioritizes_arm_minimums_and_reports_deficits() -> None:
    entries = []
    for source in ("waymo", "pg"):
        for index in range(12):
            entry = _entry(source, index)
            arm = "A2_junction" if index in {0, 1, 2} else "A1_traffic"
            entries.append(
                ScenarioCatalogEntry(replace(entry.record, primary_arm=arm), entry.features)
            )
    minimums = {
        source: {split: {"A2_junction": 1} for split in ("train", "validation", "test")}
        for source in ("waymo", "pg")
    }
    selected = assign_source_splits_to_targets(
        tuple(entries),
        targets={source: {"train": 2, "validation": 2, "test": 2} for source in ("waymo", "pg")},
        arm_minimums=minimums,
        seed=5,
    )
    diagnostics = arm_selection_diagnostics(selected, minimums)

    for source in ("waymo", "pg"):
        for split in ("train", "validation", "test"):
            assert diagnostics[source][split]["A2_junction"] == {
                "minimum": 1,
                "actual": 1,
                "deficit": 0,
            }


def test_pipeline_rejects_arm_minimums_above_source_target() -> None:
    entries = tuple(_entry(source, index) for source in ("waymo", "pg") for index in range(3))
    with pytest.raises(ValueError, match="arm minimums sum"):
        assign_source_splits_to_targets(
            entries,
            targets={
                source: {"train": 1, "validation": 1, "test": 1} for source in ("waymo", "pg")
            },
            arm_minimums={
                "waymo": {
                    "train": {
                        "A1_traffic": 1,
                        "A2_junction": 1,
                    }
                }
            },
            seed=0,
        )


def test_pipeline_excludes_disallowed_signal_reliability() -> None:
    entries = []
    for source in ("waymo", "pg"):
        for index in range(4):
            entry = _entry(source, index)
            reliability = "partial" if source == "waymo" and index == 0 else "complete"
            entries.append(
                ScenarioCatalogEntry(
                    replace(entry.record, signal_reliability=reliability),
                    replace(entry.features, signal_reliability=reliability),
                )
            )
    selected = assign_source_splits_to_targets(
        tuple(entries),
        targets={source: {"train": 1, "validation": 1, "test": 1} for source in ("waymo", "pg")},
        allowed_signal_reliabilities={
            "waymo": ("complete", "not_applicable"),
            "pg": ("complete", "not_applicable"),
        },
        seed=0,
    )

    assert all(entry.record.scenario_uid != "waymo:v1:0" for entry in selected)


def test_pipeline_excludes_invalid_records_before_split_accounting() -> None:
    entries = []
    for source in ("waymo", "pg"):
        for index in range(4):
            entry = _entry(source, index)
            status = "invalid" if index == 0 else "valid"
            entries.append(
                ScenarioCatalogEntry(
                    replace(entry.record, validation_status=status), entry.features
                )
            )

    selected = assign_source_splits_to_targets(
        tuple(entries),
        targets={source: {"train": 1, "validation": 1, "test": 1} for source in ("waymo", "pg")},
        seed=0,
    )

    assert len(selected) == 6
    assert all(entry.record.validation_status == "valid" for entry in selected)


def test_pipeline_excludes_explicitly_rulebook_ineligible_records() -> None:
    entries = []
    for source in ("waymo", "pg"):
        for index in range(4):
            entry = _entry(source, index)
            entries.append(
                ScenarioCatalogEntry(
                    replace(
                        entry.record,
                        rulebook_eligible=False if index == 0 else True,
                    ),
                    entry.features,
                )
            )

    selected = assign_source_splits_to_targets(
        tuple(entries),
        targets={source: {"train": 1, "validation": 1, "test": 1} for source in ("waymo", "pg")},
        seed=0,
    )

    assert len(selected) == 6
    assert all(entry.record.rulebook_eligible is not False for entry in selected)


def test_runtime_catalog_rejects_unverified_rulebook_records() -> None:
    entry = _entry("waymo", 0)
    catalog = ScenarioCatalog((entry,))

    with pytest.raises(ValueError, match="rulebook_eligible=True"):
        _runtime_rulebook_records(catalog, split="train")

    approved_catalog = ScenarioCatalog(
        (ScenarioCatalogEntry(replace(entry.record, rulebook_eligible=True), entry.features),)
    )
    assert _runtime_rulebook_records(approved_catalog, split="train") == approved_catalog.records


def test_arm_balanced_split_targets_split_arm_and_source_halves() -> None:
    entries = []
    index = 0
    for arm in (
        "A0_simple_low_traffic",
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A4_vru",
        "A5_critical_mixed",
    ):
        for source in ("waymo", "pg"):
            base = _entry(source, index)
            entries.append(
                ScenarioCatalogEntry(replace(base.record, primary_arm=arm), base.features)
            )
            index += 1

    selected = assign_arm_balanced_splits_to_targets(
        tuple(entries),
        targets={
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
        seed=0,
    )
    diagnostics = arm_source_balance_diagnostics(
        selected,
        {
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
    )

    assert len(selected) == 12
    for arm in (
        "A0_simple_low_traffic",
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A4_vru",
        "A5_critical_mixed",
    ):
        payload = diagnostics["train"]["arms"][arm]
        assert payload["actual"] == 2
        assert payload["sources"]["waymo"]["actual"] == 1
        assert payload["sources"]["pg"]["actual"] == 1


def test_runtime_split_contract_requires_exact_targets_uniform_arms_and_rulebook() -> None:
    entries = []
    index = 0
    for arm in (
        "A0_simple_low_traffic",
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A4_vru",
        "A5_critical_mixed",
    ):
        for source in ("waymo", "pg"):
            base = _entry(source, index)
            entries.append(
                ScenarioCatalogEntry(
                    replace(
                        base.record,
                        primary_arm=arm,
                        rulebook_eligible=True,
                    ),
                    base.features,
                )
            )
            index += 1

    selected = assign_arm_balanced_splits_to_targets(
        tuple(entries),
        targets={
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
        seed=0,
    )

    assert_runtime_split_contract(
        selected,
        targets={
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
        require_near_uniform_arms=True,
    )

    with pytest.raises(ValueError, match="rulebook_eligible=True"):
        assert_runtime_split_contract(
            (replace(selected[0], record=replace(selected[0].record, rulebook_eligible=None)),),
            targets={
                "waymo": {"train": 1, "validation": 0, "test": 0},
                "pg": {"train": 0, "validation": 0, "test": 0},
            },
            require_near_uniform_arms=False,
        )


def test_runtime_split_contract_rejects_arm_imbalance() -> None:
    entries = tuple(
        ScenarioCatalogEntry(
            replace(_entry("waymo", index).record, rulebook_eligible=True),
            _entry("waymo", index).features,
        )
        for index in range(6)
    )

    with pytest.raises(ValueError, match="seed-derived near-uniform"):
        assert_runtime_split_contract(
            entries,
            targets={
                "waymo": {"train": 6, "validation": 0, "test": 0},
                "pg": {"train": 0, "validation": 0, "test": 0},
            },
            require_near_uniform_arms=True,
        )


def test_runtime_split_contract_rejects_disallowed_signal_reliability() -> None:
    entry = _entry("waymo", 0)
    selected = (
        ScenarioCatalogEntry(
            replace(entry.record, rulebook_eligible=True, signal_reliability="partial"),
            replace(entry.features, signal_reliability="partial"),
        ),
    )

    with pytest.raises(ValueError, match="disallowed signal reliability"):
        assert_runtime_split_contract(
            selected,
            targets={
                "waymo": {"train": 1, "validation": 0, "test": 0},
                "pg": {"train": 0, "validation": 0, "test": 0},
            },
            require_near_uniform_arms=False,
            allowed_signal_reliabilities={"waymo": ("complete",)},
        )


def test_arm_balanced_selector_preserves_exact_source_targets_per_split() -> None:
    entries = []
    index = 0
    for source in ("waymo", "pg"):
        for arm in (
            "A0_simple_low_traffic",
            "A1_traffic",
            "A2_junction",
            "A3_complex_junction",
            "A4_vru",
            "A5_critical_mixed",
        ):
            for _ in range(3):
                base = _entry(source, index)
                entries.append(
                    ScenarioCatalogEntry(
                        replace(base.record, primary_arm=arm, rulebook_eligible=True),
                        base.features,
                    )
                )
                index += 1
    targets = {
        "waymo": {"train": 4, "validation": 1, "test": 1},
        "pg": {"train": 2, "validation": 2, "test": 2},
    }

    selected = assign_arm_balanced_splits_to_targets(tuple(entries), targets=targets, seed=3)

    assert_runtime_split_contract(
        selected,
        targets=targets,
        require_near_uniform_arms=True,
        seed=3,
    )


def test_arm_balanced_selector_solves_sparse_feasible_singleton_pool() -> None:
    entries = []
    index = 0
    for source in ("waymo", "pg"):
        for arm in (
            "A0_simple_low_traffic",
            "A1_traffic",
            "A2_junction",
            "A3_complex_junction",
            "A4_vru",
            "A5_critical_mixed",
        ):
            base = _entry(source, index)
            entries.append(
                ScenarioCatalogEntry(
                    replace(base.record, primary_arm=arm, rulebook_eligible=True),
                    base.features,
                )
            )
            index += 1
    targets = {
        "waymo": {"train": 4, "validation": 2, "test": 0},
        "pg": {"train": 2, "validation": 4, "test": 0},
    }

    selected = assign_arm_balanced_splits_to_targets(tuple(entries), targets=targets, seed=3)

    assert_runtime_split_contract(
        selected,
        targets=targets,
        require_near_uniform_arms=True,
        seed=3,
    )


def test_arm_balanced_selector_scales_beyond_exact_group_threshold() -> None:
    entries = []
    index = 0
    for source in ("waymo", "pg"):
        for arm in (
            "A0_simple_low_traffic",
            "A1_traffic",
            "A2_junction",
            "A3_complex_junction",
            "A4_vru",
            "A5_critical_mixed",
        ):
            for _ in range(2):
                base = _entry(source, index)
                entries.append(
                    ScenarioCatalogEntry(
                        replace(base.record, primary_arm=arm, rulebook_eligible=True),
                        base.features,
                    )
                )
                index += 1
    targets = {
        "waymo": {"train": 6, "validation": 6, "test": 0},
        "pg": {"train": 6, "validation": 6, "test": 0},
    }

    selected = assign_arm_balanced_splits_to_targets(tuple(entries), targets=targets, seed=9)

    assert len(entries) > 18
    assert_runtime_split_contract(
        selected,
        targets=targets,
        require_near_uniform_arms=True,
        seed=9,
    )


def test_pg_replenishment_report_separates_pg_shortage_from_joint_failure() -> None:
    pg = [
        ScenarioCatalogEntry(
            replace(_entry("pg", index).record, rulebook_eligible=True),
            _entry("pg", index).features,
        )
        for index in range(2)
    ]
    report = compute_pg_replenishment_report(
        pg,
        targets={
            "waymo": {"train": 1, "validation": 0, "test": 0},
            "pg": {"train": 3, "validation": 0, "test": 0},
        },
        allowed_signal_reliabilities=("not_applicable",),
        selection_error="Waymo grouped split is infeasible",
    )

    assert report["population_counts"]["runtime_eligible"] == 2
    assert report["hard_count_shortfall"] == 1
    assert report["minimum_additional_runtime_eligible_records"] == 1
    assert report["selection_error"] == "Waymo grouped split is infeasible"


def test_arm_balanced_split_falls_back_to_available_source() -> None:
    entries = []
    index = 0
    for arm in (
        "A0_simple_low_traffic",
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A5_critical_mixed",
    ):
        for source in ("waymo", "pg"):
            base = _entry(source, index)
            entries.append(
                ScenarioCatalogEntry(replace(base.record, primary_arm=arm), base.features)
            )
            index += 1
    for _ in range(2):
        base = _entry("waymo", index)
        entries.append(
            ScenarioCatalogEntry(replace(base.record, primary_arm="A4_vru"), base.features)
        )
        index += 1

    selected = assign_arm_balanced_splits_to_targets(
        tuple(entries),
        targets={
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
        seed=0,
    )
    diagnostics = arm_source_balance_diagnostics(
        selected,
        {
            "waymo": {"train": 6, "validation": 0, "test": 0},
            "pg": {"train": 6, "validation": 0, "test": 0},
        },
    )

    a4 = diagnostics["train"]["arms"]["A4_vru"]
    assert a4["actual"] == 2
    assert a4["sources"]["waymo"]["actual"] == 2
    assert a4["sources"]["pg"]["actual"] == 0
    assert a4["source_compensation_from_equal_share"] == {"waymo": 1.0, "pg": -1.0}


def test_arm_balancing_trims_pg_before_waymo() -> None:
    entries = [
        ScenarioCatalogEntry(
            replace(_entry("waymo", 0).record, primary_arm="A1_traffic"),
            _entry("waymo", 0).features,
        ),
        ScenarioCatalogEntry(
            replace(_entry("pg", 1).record, primary_arm="A1_traffic"),
            _entry("pg", 1).features,
        ),
        ScenarioCatalogEntry(
            replace(_entry("pg", 2).record, primary_arm="A1_traffic"),
            _entry("pg", 2).features,
        ),
    ]
    for index, arm in enumerate(
        (
            "A0_simple_low_traffic",
            "A2_junction",
            "A3_complex_junction",
            "A4_vru",
            "A5_critical_mixed",
        ),
        start=3,
    ):
        base = _entry("waymo", index)
        entries.append(ScenarioCatalogEntry(replace(base.record, primary_arm=arm), base.features))

    balanced, report = balance_arm_distribution(
        tuple(entries),
        target_total=6,
        seed=0,
        prefer_source="waymo",
    )

    a1_entries = [entry for entry in balanced if entry.record.primary_arm == "A1_traffic"]
    assert len(a1_entries) == 1
    assert a1_entries[0].record.source == "waymo"
    assert report["diagnostics"]["A1_traffic"]["removed_by_source"]["pg"] == 2
