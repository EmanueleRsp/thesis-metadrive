"""Static task/context contracts for Rulebook v2 adapters."""

from thesis_rl.rulebook.v2.context.task_route import (
    RULEBOOK_V2_ADAPTER_CONTRACT_VERSION,
    TaskRouteEligibility,
    TaskRouteEligibilityIndex,
    build_task_route_eligibility_index,
    TaskRouteExclusionReport,
    build_task_route_exclusion_report,
    build_task_route_record,
    validate_task_route,
)
from thesis_rl.rulebook.v2.context.map_matching import (
    OfflineTrackSample,
    map_match_sdc_track_to_task_route,
)
from thesis_rl.rulebook.v2.context.static_adapter import StaticAdapterResult, normalize_static_records, validate_reset_contract
from thesis_rl.rulebook.v2.context.snapshotter import capture_env_snapshot
from thesis_rl.rulebook.v2.context.live_adapter import (
    LiveSnapshotAdapter,
    LiveSnapshotSources,
    actor_snapshot_from_payload,
    contact_onset_from_payload,
    install_collision_callback_hook,
    wrap_collision_callback,
)
from thesis_rl.rulebook.v2.context.static_sources import StaticRecordAdapter, StaticRecordSources
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result

__all__ = [
    "RULEBOOK_V2_ADAPTER_CONTRACT_VERSION",
    "TaskRouteEligibility",
    "TaskRouteEligibilityIndex",
    "build_task_route_eligibility_index",
    "TaskRouteExclusionReport",
    "build_task_route_exclusion_report",
    "build_task_route_record",
    "validate_task_route",
    "OfflineTrackSample",
    "map_match_sdc_track_to_task_route",
    "StaticAdapterResult",
    "validate_reset_contract",
    "normalize_static_records",
    "capture_env_snapshot",
    "LiveSnapshotAdapter",
    "LiveSnapshotSources",
    "actor_snapshot_from_payload",
    "contact_onset_from_payload",
    "install_collision_callback_hook",
    "wrap_collision_callback",
    "StaticRecordAdapter",
    "StaticRecordSources",
    "build_waymo_static_adapter_result",
    "build_pg_static_adapter_result",
]
