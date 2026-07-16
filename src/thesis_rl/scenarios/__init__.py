"""ScenarioNet-backed dataset and environment integration."""

from thesis_rl.scenarios.bootstrap import (
    collect_local_api_inventory,
    create_initial_manifest,
    initialize_bootstrap_artifacts,
)
from thesis_rl.scenarios.arms import (
    ARMS,
    assign_primary_arm,
    classify_catalog_entry,
    derive_scenario_tags,
)
from thesis_rl.scenarios.features import extract_scenario_features
from thesis_rl.scenarios.paths import ScenarioDataPaths
from thesis_rl.scenarios.provider import (
    ArmUniformScenarioProvider,
    FixedSequenceScenarioProvider,
    ScenarioProvider,
    UniformScenarioProvider,
)
from thesis_rl.scenarios.pg.loader import load_exported_pg_entries
from thesis_rl.scenarios.official_checks import (
    build_official_check_command,
    run_official_check,
)
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord
from thesis_rl.scenarios.runtime_database import (
    assign_runtime_indices,
    build_runtime_database,
    sha256_file,
    verify_runtime_mapping,
)
from thesis_rl.scenarios.validation import (
    ScenarioValidationResult,
    validate_records,
    validate_scenario_file,
    validation_summary,
    write_validation_summary,
)
from thesis_rl.scenarios.waymo import (
    WaymoConversionError,
    assign_waymo_internal_splits,
    build_converter_command,
    load_converted_waymo_entries,
    validate_training_20s_source,
    waymo_dependency_status,
    waymo_group_id,
)

__all__ = [
    "collect_local_api_inventory",
    "create_initial_manifest",
    "assign_primary_arm",
    "ARMS",
    "classify_catalog_entry",
    "derive_scenario_tags",
    "extract_scenario_features",
    "initialize_bootstrap_artifacts",
    "ScenarioDataPaths",
    "ScenarioFeatures",
    "ScenarioProvider",
    "ScenarioRecord",
    "ArmUniformScenarioProvider",
    "UniformScenarioProvider",
    "FixedSequenceScenarioProvider",
    "build_official_check_command",
    "run_official_check",
    "load_exported_pg_entries",
    "ScenarioValidationResult",
    "build_runtime_database",
    "assign_runtime_indices",
    "sha256_file",
    "validate_records",
    "validate_scenario_file",
    "validation_summary",
    "write_validation_summary",
    "verify_runtime_mapping",
    "WaymoConversionError",
    "assign_waymo_internal_splits",
    "build_converter_command",
    "load_converted_waymo_entries",
    "validate_training_20s_source",
    "waymo_dependency_status",
    "waymo_group_id",
]
