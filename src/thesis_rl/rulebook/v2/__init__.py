"""Rulebook v2 monitor contracts (monitor wiring follows in later phases)."""

from thesis_rl.rulebook.v2.config import (
    RULEBOOK_V2_VERSION,
    RulebookV2Config,
    load_rulebook_v2_config,
)
from thesis_rl.rulebook.v2.monitor import evaluate_monitor_transition, evaluate_registered_transition
from thesis_rl.rulebook.v2.wrapper import RulebookV2Adapter, RulebookV2MonitorWrapper
from thesis_rl.rulebook.v2.context.live_adapter import (
    LiveSnapshotAdapter,
    LiveSnapshotSources,
    actor_snapshot_from_payload,
    contact_onset_from_payload,
    install_collision_callback_hook,
    wrap_collision_callback,
)
from thesis_rl.rulebook.v2.context.task_route import TaskRouteEligibility, TaskRouteEligibilityIndex, TaskRouteExclusionReport, build_task_route_eligibility_index, build_task_route_exclusion_report
from thesis_rl.rulebook.v2.context.static_sources import StaticRecordAdapter, StaticRecordSources
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.static_adapter import validate_reset_contract
from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.calibration import (
    BrakingTrial,
    calibrate_ego_braking,
    load_calibration_artifact,
    write_calibration_artifact,
)
from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.events import (
    ContactOnsetBuffer,
    ContactSetTransition,
    ZoneTransitionEvents,
    derive_contact_transition,
    detect_zone_transition,
)
from thesis_rl.rulebook.v2.lifecycle import ZoneLifecycleEvaluator, ZoneLifecycleView
from thesis_rl.rulebook.v2.registry import ComponentDefinition, RulebookV2Registry
from thesis_rl.rulebook.v2.types import (
    MACRO_RULE_ORDER,
    ActorClass,
    CacheDelta,
    ComponentStatus,
    EnvSnapshot,
    MacroRule,
    RulebookMemory,
    RulebookResult,
    TaskRouteRecord,
)

__all__ = [
    "ActorClass", "CacheDelta", "ComponentDefinition", "ComponentStatus", "EnvSnapshot",
    "ContactOnsetBuffer", "ContactSetTransition", "ZoneTransitionEvents", "derive_contact_transition", "detect_zone_transition",
    "ZoneLifecycleEvaluator", "ZoneLifecycleView",
    "EvaluationFailure", "MACRO_RULE_ORDER", "MacroRule", "RULEBOOK_V2_VERSION",
    "RulebookEvaluationError", "RulebookMemory", "RulebookResult", "RulebookV2Config",
    "RulebookV2Registry", "TaskRouteRecord", "TaskRouteEligibility", "TaskRouteEligibilityIndex", "TaskRouteExclusionReport", "build_task_route_eligibility_index", "build_task_route_exclusion_report", "StaticRecordAdapter", "StaticRecordSources", "build_waymo_static_adapter_result", "build_pg_static_adapter_result", "validate_reset_contract", "actor_snapshot_from_payload", "load_rulebook_v2_config",
    "aggregate_rulebook_result", "BrakingTrial", "calibrate_ego_braking", "load_calibration_artifact", "write_calibration_artifact", "evaluate_monitor_transition", "evaluate_registered_transition", "RulebookV2Adapter", "RulebookV2MonitorWrapper", "LiveSnapshotAdapter", "LiveSnapshotSources", "contact_onset_from_payload", "install_collision_callback_hook", "wrap_collision_callback",
]
