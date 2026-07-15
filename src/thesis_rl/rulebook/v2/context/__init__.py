"""Static task/context contracts for Rulebook v2 adapters."""

from thesis_rl.rulebook.v2.context.task_route import (
    RULEBOOK_V2_ADAPTER_CONTRACT_VERSION,
    TaskRouteEligibility,
    build_task_route_record,
    validate_task_route,
)
from thesis_rl.rulebook.v2.context.map_matching import (
    OfflineTrackSample,
    map_match_sdc_track_to_task_route,
)
from thesis_rl.rulebook.v2.context.static_adapter import StaticAdapterResult, normalize_static_records
from thesis_rl.rulebook.v2.context.snapshotter import capture_env_snapshot
from thesis_rl.rulebook.v2.context.live_adapter import LiveSnapshotAdapter, LiveSnapshotSources

__all__ = [
    "RULEBOOK_V2_ADAPTER_CONTRACT_VERSION",
    "TaskRouteEligibility",
    "build_task_route_record",
    "validate_task_route",
    "OfflineTrackSample",
    "map_match_sdc_track_to_task_route",
    "StaticAdapterResult",
    "normalize_static_records",
    "capture_env_snapshot",
    "LiveSnapshotAdapter",
    "LiveSnapshotSources",
]
