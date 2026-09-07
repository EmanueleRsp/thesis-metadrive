"""ScenarioNet environment with the thesis-specific episode contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
import math
import time
from dataclasses import replace
from typing import Any

from thesis_rl.envs.scene_context import SceneContextAdapter


def scenario_time_limit_reached(
    *, episode_steps: int, scenario_length: int, extra_steps_after_scenario: int
) -> bool:
    """Return whether the exported scenario horizon (plus the configured tail) is met.

    The last step carrying logged data is ``scenario_length - 1``, not
    ``scenario_length``. Both counters are 1-based at this point in the step:
    ``BaseEngine.before_step`` increments ``episode_step`` and
    ``BaseEnv._get_step_return`` increments ``episode_lengths`` before
    ``done_function`` runs, and ``ScenarioTrafficManager.after_step`` -- which
    runs between them -- despawns **every** replayed participant as soon as
    ``episode_step >= current_scenario_length``.

    Stopping at ``scenario_length`` therefore truncated on the one step whose
    observation is built from an emptied, signal-frozen world. That observation
    is the bootstrap state of the truncated transition, so the critic's target
    for the majority of episodes rested on a state that occurs nowhere else in
    the training distribution and is systematically the easiest one in it: no
    actor can violate L1 or L2, and the road ahead is clear. `n`-step replay
    spreads it over the last `n` transitions rather than one.

    ``extra_steps_after_scenario`` keeps its meaning -- steps taken *after* the
    logged data ends, in that emptied world -- and now actually delivers it: at
    the frozen value 0 the episode no longer takes such a step at all, which is
    what `conf/env/scenarionet.yaml` already documented it as doing.
    """

    if episode_steps < 0 or scenario_length <= 0 or extra_steps_after_scenario < 0:
        raise ValueError(
            "episode_steps >= 0, scenario_length > 0 and extra_steps >= 0 are required"
        )
    return episode_steps >= scenario_length - 1 + extra_steps_after_scenario


try:  # keep importing the package possible in lightweight tooling environments
    from metadrive.constants import TerminationState  # type: ignore[import-not-found]
    from metadrive.envs.scenario_env import ScenarioEnv  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in the dedicated container
    ScenarioEnv = object  # type: ignore[assignment,misc]
    TerminationState = type(
        "TerminationState",
        (),
        {
            "SUCCESS": "arrive_dest",
            "OUT_OF_ROAD": "out_of_road",
            "MAX_STEP": "max_step",
            "CRASH": "crash",
            "CRASH_VEHICLE": "crash_vehicle",
            "CRASH_HUMAN": "crash_human",
            "CRASH_OBJECT": "crash_object",
            "CRASH_BUILDING": "crash_building",
            "CRASH_SIDEWALK": "crash_sidewalk",
        },
    )


class ThesisScenarioEnv(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "horizon": None,
                "allowed_more_steps": None,
                "truncate_as_terminate": False,
                "reactive_traffic": True,
                "store_data": False,
                "store_map": False,
                # ADR-058: zero tail; see `conf/env/scenarionet.yaml`.
                "extra_steps_after_scenario": 0,
                # ScenarioEnv treats a reference trajectory shorter than two
                # metres as an immediate success.  That is useful for its
                # replay use case, but corrupts an RL success metric: some
                # converted Waymo SDC tracks are stationary or near-stationary.
                "success_route_completion_threshold": 0.95,
                "minimum_success_route_length_m": 10.0,
                "rulebook_v2_disable_vehicle_yield_for_benchmark": False,
                # Diagnostic only, read by `envs/policies/replay_ego.py` when
                # `agent_policy` is `replay_ego_policy`: fraction of the logged
                # ego trajectory replayed before the ego is held still. 1.0
                # replays the whole log. Ignored by every other agent policy.
                "replay_ego_stop_fraction": 1.0,
                # OBS-V1.2 SS6.2 signal-camera baseline, overridable per ADR-045
                # through `conf/obs/semantic_v3.yaml`. `envs/factory.py`
                # unconditionally injects these three keys whenever the semantic
                # v3 observation is selected, and MetaDrive's `BaseEnv.__init__`
                # updates its default config with `allow_add_new_key=False`, so
                # they must be declared here or environment construction raises
                # `KeyError` before the first reset.
                "semantic_v3_signal_range_m": 80.0,
                "semantic_v3_signal_fov_degrees": 65.0,
                "semantic_v3_signal_camera_height_m": 1.2,
            }
        )
        return config

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        scenario_provider: Any | None = None,
        catalog: Any | None = None,
        split: str = "train",
        worker_id: int = 0,
        scenario_arm: str | None = None,
        scene_context: SceneContextAdapter | None = None,
    ) -> None:
        super().__init__(config)
        self.scenario_provider = scenario_provider
        self.catalog = catalog
        self.split = str(split)
        self.worker_id = int(worker_id)
        self.scenario_arm = scenario_arm
        self.scenario_excluded_uids: set[str] = set()
        self.scene_context = scene_context or SceneContextAdapter()
        self.current_scenario_record: Any | None = None
        self._last_done_info: dict[str, Any] = {}
        self._runtime_stats: dict[str, Any] = {
            "resets": 0,
            "steps": 0,
            "episodes": 0,
            "resets_by_source": Counter(),
            "steps_by_source": Counter(),
            "episodes_by_arm": Counter(),
            "resets_by_source_arm": Counter(),
            "steps_by_source_arm": Counter(),
            "episodes_by_source_arm": Counter(),
            "termination_reasons": Counter(),
        }
        self._runtime_timing_seconds = {"reset": 0.0, "step": 0.0, "observation": 0.0}
        self._active_reset_timing_seconds: dict[str, float] | None = None
        self._last_sampling_metadata: dict[str, Any] = {}
        self._acl_selection: dict[str, Any] | None = None
        self._acl_episode_selection: dict[str, Any] = {}
        self._rulebook_v2_requested = False
        self.rulebook_v2_adapter: Any | None = None
        self._rulebook_v2_brake_mps2: float | None = None
        self._mission_runtime: Any | None = None
        self._mission_pre_snapshot: Any | None = None
        self._mission_snapshotter: Any | None = None
        self._mission_last_update_episode_length: int | None = None
        self._mission_static_result: Any | None = None
        self._mission_episode_cache: Any | None = None

    def setup_engine(self) -> None:
        """Install the thesis-owned source-bounded reactive traffic manager."""

        super().setup_engine()
        if not self.config["no_traffic"]:
            from thesis_rl.envs.scenario_traffic_manager import (
                SourceBoundScenarioTrafficManager,
            )

            self.engine.update_manager("traffic_manager", SourceBoundScenarioTrafficManager())

    def configure_acl_selection(self, selection: Mapping[str, Any]) -> dict[str, Any]:
        """Install parent-owned ACL metadata before the next selective reset."""
        payload = dict(selection)
        if int(payload.get("slot_id", self.worker_id)) != self.worker_id:
            raise ValueError("ACL selection slot_id does not match the ScenarioNet worker.")
        self._acl_selection = payload
        self._acl_episode_selection = payload
        self.scenario_arm = payload.get("arm_name")
        source = payload.get("source")
        if source is not None:
            self.scenario_source = str(source)
        return {"selection_generation": int(payload.get("generation", -1))}

    def quarantine_scenario_uid(self, scenario_uid: str) -> None:
        """Exclude one runtime-invalid scenario for this process/run only."""

        normalized = str(scenario_uid).strip()
        if not normalized:
            raise ValueError("Runtime quarantine requires a non-empty scenario UID.")
        self.scenario_excluded_uids.add(normalized)

    def get_quarantined_scenario_uids(self) -> list[str]:
        """Return the run-local quarantine set for checkpoint persistence."""

        return sorted(self.scenario_excluded_uids)

    def _acl_info(self) -> dict[str, Any]:
        """Expose collection provenance in info only, never in observations."""

        selection = self._acl_episode_selection
        if not selection:
            return {}
        return {
            "acl_slot_id": int(selection.get("slot_id", self.worker_id)),
            "acl_episode_id": int(selection["episode_id"])
            if selection.get("episode_id") is not None
            else None,
            "acl_selection_generation": int(selection.get("generation", -1)),
        }

    def _select_provider_seed(self, force_seed: int | None) -> int | None:
        acl_selection = getattr(self, "_acl_selection", None)
        if acl_selection is not None:
            selection = acl_selection
            self._acl_selection = None
            runtime_index = selection.get("runtime_index")
            if runtime_index is not None:
                force_seed = int(runtime_index)
        if force_seed is not None:
            if self.catalog is not None:
                record = self.catalog.get_by_runtime_index(
                    split=self.split, runtime_index=int(force_seed)
                ).record
                self.current_scenario_record = record
                self._publish_assigned_route_metadata()
            return int(force_seed)
        if self.scenario_provider is None:
            return None
        record = self.scenario_provider.sample(
            split=self.split,
            worker_id=self.worker_id,
            source=getattr(self, "scenario_source", None),
            arm=self.scenario_arm,
            excluded_scenario_uids=self.scenario_excluded_uids,
        )
        sampling_metadata = getattr(self.scenario_provider, "sampling_metadata", None)
        self._last_sampling_metadata = (
            dict(sampling_metadata(worker_id=self.worker_id)) if callable(sampling_metadata) else {}
        )
        if record.runtime_index is None:
            raise ValueError(
                f"scenario provider returned record without runtime_index: {record.scenario_uid}"
            )
        self.current_scenario_record = record
        self._publish_assigned_route_metadata()
        return int(record.runtime_index)

    def _publish_assigned_route_metadata(self) -> None:
        """Expose frozen catalog route metadata to reset-time adapters only."""

        record = self.current_scenario_record
        if record is None:
            return
        # MetaDrive's Config.pop accepts only the key, unlike dict.pop.  Check
        # membership first so this works for both the runtime Config and the
        # plain dictionaries used by lightweight environment tests.
        for field in ("assigned_route_lane_ids", "assigned_route_source"):
            if field in self.config:
                self.config.pop(field)
        lane_ids = tuple(getattr(record, "assigned_route_lane_ids", ()) or ())
        if not lane_ids:
            raise ValueError(
                "Scenario record is missing assigned_route_lane_ids; "
                "exclude it before runtime reset"
            )
        self.config["assigned_route_lane_ids"] = lane_ids
        self.config["assigned_route_source"] = getattr(record, "assigned_route_source", None)

    def _record_reset_phase(self, name: str, seconds: float) -> None:
        if getattr(self, "_active_reset_timing_seconds", None) is not None:
            self._active_reset_timing_seconds[name] = self._active_reset_timing_seconds.get(
                name, 0.0
            ) + float(seconds)

    @contextmanager
    def _profile_existing_engine_reset(self):
        """Measure manager phases for an already initialized MetaDrive engine."""

        engine = getattr(self, "engine", None)
        managers = getattr(engine, "_managers", None)
        if engine is None or not isinstance(managers, Mapping):
            yield
            return

        sentinel = object()
        restored: list[tuple[object, str, object]] = []
        wrapped: set[tuple[int, str]] = set()

        def wrap(target: object, label: str, method_name: str) -> None:
            key = (id(target), method_name)
            method = getattr(target, method_name, None)
            if key in wrapped or not callable(method):
                return
            wrapped.add(key)
            original_attribute = getattr(target, "__dict__", {}).get(method_name, sentinel)

            def measured(
                *args: object,
                _method: Any = method,
                _label: str = label,
                _method_name: str = method_name,
                **kwargs: object,
            ) -> object:
                started = time.perf_counter()
                try:
                    return _method(*args, **kwargs)
                finally:
                    self._record_reset_phase(
                        f"engine_{_method_name}.{_label}", time.perf_counter() - started
                    )

            setattr(target, method_name, measured)
            restored.append((target, method_name, original_attribute))

        for manager_name, manager in managers.items():
            for method_name in ("before_reset", "reset", "after_reset"):
                wrap(manager, str(manager_name), method_name)
        terrain = getattr(engine, "terrain", None)
        if terrain is not None:
            for method_name in ("before_reset", "reset", "after_reset"):
                wrap(terrain, "terrain", method_name)
        try:
            yield
        finally:
            for target, method_name, original_attribute in reversed(restored):
                if original_attribute is sentinel:
                    delattr(target, method_name)
                else:
                    setattr(target, method_name, original_attribute)

    def _reset_global_seed(self, force_seed=None):
        started = time.perf_counter()
        try:
            provider_seed = self._select_provider_seed(force_seed)
            if provider_seed is not None:
                self.seed(provider_seed)
                return
            super()._reset_global_seed(force_seed)
        finally:
            self._record_reset_phase("scenario_selection", time.perf_counter() - started)

    def _inject_assigned_route_metadata_into_scenario(self) -> None:
        """Attach frozen catalog route metadata without reading SDC samples."""

        record = self.current_scenario_record
        if record is None:
            return
        data_manager = getattr(getattr(self, "engine", None), "data_manager", None)
        if data_manager is None:
            return
        scenario = getattr(data_manager, "current_scenario", None)
        if not isinstance(scenario, dict):
            return
        metadata = scenario.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Scenario metadata must be mutable before route injection")
        metadata["assigned_route_lane_ids"] = list(
            getattr(record, "assigned_route_lane_ids", ()) or ()
        )
        metadata["assigned_route_source"] = getattr(record, "assigned_route_source", None)

    @staticmethod
    def _build_static_adapter_result(scenario: Mapping[str, Any], record: Any) -> Any:
        source = str(getattr(record, "source", "")).lower()
        if source == "pg":
            from thesis_rl.rulebook.v2.context.pg_static_adapter import (
                build_pg_static_adapter_result,
            )

            return build_pg_static_adapter_result(
                scenario, scenario_uid=str(getattr(record, "scenario_uid"))
            )
        elif source == "waymo":
            from thesis_rl.rulebook.v2.context.waymo_static_adapter import (
                build_waymo_static_adapter_result,
            )

            return build_waymo_static_adapter_result(
                scenario, scenario_uid=str(getattr(record, "scenario_uid"))
            )
        raise ValueError(f"Unsupported scenario source for causal route: {source!r}")

    def _build_causal_frame_builder(
        self, config: Mapping[str, Any], route_lanes: tuple[Any, ...] = ()
    ) -> Any:
        from thesis_rl.envs.observations.assigned_route import (
            AssignedRouteWaypointAdapter,
            MapRouteNavigationObservation22,
        )
        from thesis_rl.envs.observations.causal_lidar import CausalLidarFrameBuilder
        from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper

        # The mission's canonical route (not the legacy assigned-route polyline) is
        # the sole ego route-station authority shared with R4/completion/semantic
        # observation, per DRIVING-MISSION-V1.1 §3/§5.
        route = self._mission_runtime.route
        navigation = MapRouteNavigationObservation22(
            AssignedRouteWaypointAdapter(
                route,
                num_waypoints=10,
                spacing_m=5.0,
                mission_provider=lambda: self._mission_runtime,
            )
        )
        observation_cfg = config.get("observation", {})
        if not isinstance(observation_cfg, Mapping):
            observation_cfg = {}
        ray_cfg = observation_cfg.get("ray_noise_wrapper", {})
        if not isinstance(ray_cfg, Mapping):
            ray_cfg = {}
        return CausalLidarFrameBuilder(
            navigation,
            RayNoiseWrapper(
                sigma_normalized=float(ray_cfg.get("sigma_normalized", 0.001)),
                dropout_prob=float(ray_cfg.get("dropout_prob", 0.0)),
                enabled=bool(ray_cfg.get("enabled", True)),
            ),
            route_lanes=route_lanes,
        )

    def _install_causal_observation_builder(self) -> None:
        record = self.current_scenario_record
        if record is None:
            return
        observations = getattr(getattr(self, "agent_manager", None), "observations", {})
        frame_setters = [
            setter
            for setter in (
                getattr(observation, "set_frame_builder", None)
                for observation in observations.values()
            )
            if callable(setter)
        ]
        batch_setters = [
            setter
            for setter in (
                getattr(observation, "set_batch_builder", None)
                for observation in observations.values()
            )
            if callable(setter)
        ]
        if not frame_setters and not batch_setters:
            return
        engine = getattr(self, "engine", None)
        data_manager = getattr(engine, "data_manager", None)
        scenario = getattr(data_manager, "current_scenario", None)
        if not isinstance(scenario, Mapping):
            raise RuntimeError("Causal observation requires the loaded scenario mapping")
        if frame_setters:
            # OBS-LIDAR-V2.0.2 needs the route lanes for the posted-limit feature,
            # resolved the same way the semantic path below resolves them: the
            # live Rulebook cache when it exists, since it carries the
            # source-to-simulator elevation translation, and the static adapter
            # result otherwise.
            rulebook_adapter = getattr(self, "rulebook_v2_adapter", None)
            episode_cache = getattr(rulebook_adapter, "initial_cache", None)
            frame_route_lanes = (
                episode_cache.route_lanes
                if episode_cache is not None
                else self._build_static_adapter_result(scenario, record).route_lanes
            )
            builder = self._build_causal_frame_builder(self.config, frame_route_lanes)
            for setter in frame_setters:
                setter(builder)
        self._causal_semantic_builders = []
        if batch_setters:
            from thesis_rl.envs.observations.causal_semantic import (
                CausalSemanticBatchBuilder,
                PerceptionBoundedSemanticBatchBuilder,
            )
            from thesis_rl.envs.observations.semantic_state_v3 import SemanticStateObservationV3

            static_result = self._build_static_adapter_result(scenario, record)
            # The mission's canonical route is the sole ego route-station authority
            # shared with R4/completion, per DRIVING-MISSION-V1.1 §3/§5; it never
            # independently reprojects the ego. Lane geometry lookups (route_lanes)
            # are unaffected -- the live Rulebook cache still applies the
            # source-to-simulator elevation datum translation for those.
            route = self._mission_runtime.route
            rulebook_adapter = getattr(self, "rulebook_v2_adapter", None)
            episode_cache = getattr(rulebook_adapter, "initial_cache", None)
            route_lanes = (
                episode_cache.route_lanes
                if episode_cache is not None
                else static_result.route_lanes
            )
            builder_type = (
                PerceptionBoundedSemanticBatchBuilder
                if any(
                    isinstance(observation, SemanticStateObservationV3)
                    for observation in observations.values()
                )
                else CausalSemanticBatchBuilder
            )
            builder_kwargs = dict(
                route=route,
                route_lanes=route_lanes,
                context_provider=lambda: self.causal_scene_context,
                history_length=5,
                control_timestep_s=0.1,
                dynamic_radius_m=50.0,
                static_radius_m=50.0,
                control_radius_m=80.0,
                prediction_horizon_s=3.0,
                vertical_tolerance_m=3.0,
                signal_range_m=float(self.config.get("semantic_v3_signal_range_m", 80.0)),
                signal_fov_degrees=float(self.config.get("semantic_v3_signal_fov_degrees", 65.0)),
                signal_camera_height_m=float(
                    self.config.get("semantic_v3_signal_camera_height_m", 1.2)
                ),
            )
            if builder_type is PerceptionBoundedSemanticBatchBuilder:
                builder_kwargs["brake_mps2"] = self._rulebook_v2_brake_mps2
            semantic_builder = builder_type(**builder_kwargs)
            self._causal_semantic_builders.append(semantic_builder)
            for setter in batch_setters:
                setter(semantic_builder)

    def _prepare_initial_causal_context(self) -> None:
        """Publish the reset context before MetaDrive requests observation."""

        adapter = getattr(self, "rulebook_v2_adapter", None)
        if adapter is None:
            return
        snapshot = adapter.snapshotter(self)
        from thesis_rl.contracts.causal_scene_context import CausalSceneContext

        context = CausalSceneContext(
            adapter.initial_cache, snapshot, adapter.initial_memory, self._mission_runtime.route
        )
        self.causal_scene_context = context
        for builder in getattr(self, "_causal_semantic_builders", ()):
            builder.reset()
            builder.commit_context(context)

    def _install_mission_runtime(self) -> None:
        """Install the mandatory frozen mission before any observation or reward."""

        from thesis_rl.mission.runtime import MissionRuntime
        from thesis_rl.mission.types import DrivingMissionRecord
        from thesis_rl.rulebook.v2.context.metadrive_live import live_ego_snapshot
        from thesis_rl.rulebook.v2.types import EnvSnapshot

        scenario = getattr(getattr(self.engine, "data_manager", None), "current_scenario", None)
        record = self.current_scenario_record
        if not isinstance(scenario, Mapping) or record is None:
            raise RuntimeError(
                "Driving mission runtime requires a loaded scenario and catalog record"
            )
        mission_payload = getattr(record, "driving_mission", None)
        if not isinstance(mission_payload, dict):
            raise ValueError("Scenario record is missing the required frozen driving_mission")
        static_result = self._build_static_adapter_result(scenario, record)
        decision_repeat = int(self.config.get("decision_repeat", 1))
        physics_dt = float(self.config.get("physics_world_step_size", 0.02))

        def capture(_env: Any) -> EnvSnapshot:
            return EnvSnapshot(
                scenario_id=str(record.scenario_uid),
                step_index=int(self.episode_step),
                sim_time_s=float(self.episode_step * decision_repeat * physics_dt),
                ego=live_ego_snapshot(self),
                actors=(),
                contact_onset_records=(),
                active_contact_ids=frozenset(),
                signal_states_by_physical_id={},
            )

        initial_snapshot = capture(self)
        from thesis_rl.rulebook.v2.transition import (
            align_episode_cache_to_live_elevation,
            build_episode_cache,
        )

        cache = align_episode_cache_to_live_elevation(
            build_episode_cache(static_result), initial_snapshot
        )
        self._mission_static_result = static_result
        self._mission_episode_cache = cache
        # `ScenarioDataManager` loads every scenario with `centralize=True`, so
        # the live map and ego are translated by the SDC's first raw position
        # while the frozen mission is built from the raw source file. MetaDrive
        # records the inverse translation here precisely so raw geometry can be
        # mapped back onto the live frame; it is absent only when MetaDrive
        # skipped centralization, in which case the two frames already coincide.
        raw_origin = scenario.get("metadata", {}).get("old_origin_in_current_coordinate")
        origin_offset_xy = (
            (0.0, 0.0) if raw_origin is None else (float(raw_origin[0]), float(raw_origin[1]))
        )
        self._mission_runtime = MissionRuntime(
            DrivingMissionRecord.from_dict(mission_payload),
            cache.route_lanes,
            initial_snapshot,
            origin_offset_xy=origin_offset_xy,
        )
        initial_snapshot = replace(
            initial_snapshot, mission_snapshot=self._mission_runtime.snapshot
        )

        def capture_with_mission(_env: Any) -> EnvSnapshot:
            snapshot = capture(_env)
            runtime = self._mission_runtime
            return replace(snapshot, mission_snapshot=None if runtime is None else runtime.snapshot)

        self._mission_pre_snapshot = initial_snapshot
        self._mission_snapshotter = capture_with_mission
        self._mission_last_update_episode_length = None

    def _install_rulebook_v2_adapter(self) -> None:
        """Install the source-bound Rulebook adapter after ScenarioEnv reset."""

        if not self._rulebook_v2_requested:
            return
        from metadrive.engine.core.collision_callback import collision_callback
        from metadrive.utils.utils import get_object_from_node
        import hashlib
        import json
        from pathlib import Path

        from thesis_rl.rulebook.v2.context.metadrive_live import (
            MetaDriveContactRecorder,
            live_actor_snapshots,
            live_ego_snapshot,
            live_signal_states_by_physical_id,
        )
        from thesis_rl.rulebook.v2.context.live_adapter import (
            LiveSnapshotAdapter,
            LiveSnapshotSources,
            install_collision_callback_hook,
        )
        from thesis_rl.rulebook.v2.calibration import load_calibration_artifact
        from thesis_rl.rulebook.v2.transition import (
            RulebookTransitionConfig,
            initial_memory_for_snapshot,
            transition_evaluator_factory,
        )
        from thesis_rl.rulebook.v2.wrapper import RulebookV2Adapter
        from thesis_rl.rulebook.v2.types import EnvSnapshot

        scenario = getattr(getattr(self.engine, "data_manager", None), "current_scenario", None)
        record = self.current_scenario_record
        if not isinstance(scenario, Mapping) or record is None:
            raise RuntimeError("Rulebook v2 adapter requires a loaded scenario and catalog record")
        static_result = self._mission_static_result
        cache = self._mission_episode_cache
        if static_result is None or cache is None:
            raise RuntimeError("Driving mission static cache is unavailable before Rulebook setup")
        recorder = MetaDriveContactRecorder(self, object_from_node=get_object_from_node)
        dynamic_world = self.engine.physics_world.dynamic_world
        install_collision_callback_hook(
            dynamic_world,
            original_callback=collision_callback,
            observer=recorder.observe,
        )
        contact_cache: dict[str, object] = {"step": None, "value": ((), frozenset())}

        def contact_state(_env: Any) -> tuple[tuple[Any, ...], frozenset[str]]:
            step = int(self.episode_step)
            if contact_cache["step"] != step:
                contact_cache["value"] = recorder.snapshot_contact_state()
                contact_cache["step"] = step
            return contact_cache["value"]  # type: ignore[return-value]

        decision_repeat = int(self.config.get("decision_repeat", 1))
        physics_dt = float(self.config.get("physics_world_step_size", 0.02))
        snapshotter = LiveSnapshotAdapter(
            LiveSnapshotSources(
                scenario_id=lambda _env: str(record.scenario_uid),
                step_index=lambda env: int(env.episode_step),
                sim_time_s=lambda env: float(env.episode_step * decision_repeat * physics_dt),
                ego=live_ego_snapshot,
                actors=live_actor_snapshots,
                contact_onset_records=lambda env: tuple(contact_state(env)[0]),
                active_contact_ids=lambda env: contact_state(env)[1],
                signal_states_by_physical_id=live_signal_states_by_physical_id,
            )
        )
        initial_snapshot = snapshotter.capture(self)
        if self._mission_runtime is None:
            raise RuntimeError("Rulebook v2 adapter requires the installed driving mission runtime")
        initial_snapshot = replace(
            initial_snapshot, mission_snapshot=self._mission_runtime.snapshot
        )

        def snapshot_with_mission(env: Any) -> EnvSnapshot:
            snapshot = snapshotter.capture(env)
            return replace(snapshot, mission_snapshot=self._mission_runtime.snapshot)

        calibration = None
        data_directory = Path(str(self.config.get("data_directory", "")))
        data_root = data_directory.parent.parent
        configured_calibration = self.config.get("rulebook_v2_calibration_path")
        if configured_calibration:
            calibration_path = Path(str(configured_calibration))
        else:
            calibration_path = data_root / "rulebook_v2" / "calibration_b_e.json"
        if calibration_path.is_file():
            configured_ego = self.config.get("rulebook_v2_ego_config_path")
            ego_config_path = (
                Path(str(configured_ego))
                if configured_ego
                else data_root / "rulebook_v2" / "ego_config.json"
            )
            if not ego_config_path.is_file():
                raise ValueError(
                    "Rulebook RSS calibration exists but its canonical ego config is missing"
                )
            expected_config_hash = hashlib.sha256(
                json.dumps(
                    json.loads(ego_config_path.read_text(encoding="utf-8")),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            calibration = load_calibration_artifact(
                calibration_path,
                expected_config_hash=expected_config_hash,
            )
        transition_config = RulebookTransitionConfig(
            rss_calibration=calibration,
            expected_config_hash="" if calibration is None else calibration.config_hash,
            disable_vehicle_yield_for_benchmark=bool(
                self.config.get("rulebook_v2_disable_vehicle_yield_for_benchmark", False)
            ),
        )
        self._rulebook_v2_brake_mps2 = (
            None if calibration is None else calibration.ego_min_brake_mps2
        )
        self.rulebook_v2_adapter = RulebookV2Adapter(
            snapshotter=snapshot_with_mission,
            transition_evaluator=transition_evaluator_factory(transition_config),
            initial_memory=initial_memory_for_snapshot(initial_snapshot, cache),
            initial_cache=cache,
            mission_route=self._mission_runtime.route,
        )
        # `done_function` must read the same actor-complete snapshot the Rulebook
        # reads. `_install_mission_runtime` installs an actor-less snapshotter so
        # the mission tracker can run without the Rulebook; leaving it in place
        # here made `contact_onset_records` and `actors` permanently empty in
        # `done_function`, so `_collision_is_not_at_fault` returned False on every
        # step and ADR-071's truncation never fired (audit 2026-09-06, A1).
        self._mission_snapshotter = snapshot_with_mission
        self._mission_pre_snapshot = initial_snapshot

    def make_rulebook_v2_adapter(self) -> Any:
        """Return the adapter prepared by the immediately preceding reset."""

        if self.rulebook_v2_adapter is None:
            raise RuntimeError("Rulebook v2 adapter is unavailable before ScenarioEnv reset")
        return self.rulebook_v2_adapter

    def _on_causal_context_committed(self, context: object) -> None:
        for builder in getattr(self, "_causal_semantic_builders", ()):
            builder.commit_context(context)

    def _refresh_causal_observation(self, previous_observation: object) -> object:
        """Rebuild only observation payloads after the causal commit boundary."""

        started = time.perf_counter()
        observations = getattr(getattr(self, "agent_manager", None), "observations", {})
        agents = getattr(self, "agents", {})
        if not observations or not agents:
            return previous_observation
        rebuilt = {}
        for vehicle_id, vehicle in agents.items():
            observation = observations.get(vehicle_id)
            if observation is not None and callable(
                getattr(observation, "set_batch_builder", None)
            ):
                rebuilt[vehicle_id] = observation.observe(vehicle)
        if not rebuilt:
            return previous_observation
        self._runtime_timing_seconds["observation"] += time.perf_counter() - started
        return rebuilt if getattr(self, "is_multi_agent", False) else next(iter(rebuilt.values()))

    def _get_reset_return(self, reset_info):
        # ScenarioDataManager must complete its reset before the current
        # scenario is accessed. Injecting route metadata from
        # _reset_global_seed would populate the manager before its
        # before_reset hook and leave two scenarios cached across episodes.
        started = time.perf_counter()
        self._inject_assigned_route_metadata_into_scenario()
        self._record_reset_phase("route_metadata", time.perf_counter() - started)
        started = time.perf_counter()
        self._install_mission_runtime()
        self._record_reset_phase("driving_mission", time.perf_counter() - started)
        started = time.perf_counter()
        self._install_rulebook_v2_adapter()
        self._record_reset_phase("rulebook_adapter", time.perf_counter() - started)
        started = time.perf_counter()
        self._install_causal_observation_builder()
        self._record_reset_phase("causal_observation_builder", time.perf_counter() - started)
        started = time.perf_counter()
        self._prepare_initial_causal_context()
        self._record_reset_phase("initial_causal_context", time.perf_counter() - started)
        started = time.perf_counter()
        result = super()._get_reset_return(reset_info)
        self._record_reset_phase("base_reset_observation", time.perf_counter() - started)
        return result

    @staticmethod
    def _recompute_terminated(done_info: Mapping[str, Any]) -> bool:
        return any(
            bool(done_info.get(key, False))
            for key in (
                TerminationState.SUCCESS,
                TerminationState.CRASH,
                TerminationState.CRASH_HUMAN,
                TerminationState.CRASH_VEHICLE,
                TerminationState.CRASH_OBJECT,
                TerminationState.CRASH_BUILDING,
                TerminationState.CRASH_SIDEWALK,
                TerminationState.OUT_OF_ROAD,
            )
        )

    def _scenario_length(self) -> int:
        return int(self.engine.data_manager.current_scenario_length)

    @staticmethod
    def _normalise_route_completion(value: Any) -> tuple[float | None, float | None]:
        """Return ``(bounded, raw)`` for a ScenarioEnv route-completion value.

        TrajectoryNavigation projects the vehicle onto an unbounded reference
        line, so a vehicle past the final point legitimately yields a raw value
        above one.  Metrics and rewards, however, use route *completion* and
        must remain in [0, 1].
        """

        try:
            raw = float(value)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(raw):
            return None, None
        return min(1.0, max(0.0, raw)), raw

    def _is_thesis_success(self, vehicle: Any) -> bool:
        """Use only the frozen mission tracker as task-success authority."""

        del vehicle
        runtime = self._mission_runtime
        if runtime is None:
            raise RuntimeError("Driving mission runtime is unavailable before success evaluation")
        return bool(runtime.snapshot.mission_success)

    def _attach_route_metrics(self, info: dict[str, Any]) -> None:
        vehicle = self.scene_context.get_ego_vehicle(self)
        navigation = getattr(vehicle, "navigation", None)
        completion, raw_completion = self._normalise_route_completion(
            info.get("route_completion", getattr(navigation, "route_completion", None))
        )
        runtime = self._mission_runtime
        if runtime is None:
            raise RuntimeError("Driving mission runtime is unavailable before route metrics")
        snapshot = runtime.snapshot
        info["route_completion"] = snapshot.route_completion
        info["mission_completion_instant"] = snapshot.instantaneous_completion
        info["mission_completion_max"] = snapshot.maximum_completion
        info["mission_s_m"] = snapshot.s_m
        info["mission_delta_s_m"] = snapshot.delta_s_m
        info["mission_remaining_distance_m"] = snapshot.remaining_distance_m
        info["mission_pending_gate_index"] = snapshot.pending_gate_index
        info["mission_hash"] = snapshot.mission_hash
        info["mission_reachable"] = snapshot.reachable
        info["mission_success"] = snapshot.mission_success
        info["mission_unreachable"] = snapshot.mission_unreachable
        info["mission_reason"] = snapshot.reason
        if completion is not None:
            info["native_route_completion"] = completion
        if raw_completion is not None and raw_completion != completion:
            info["native_raw_route_completion"] = raw_completion
        reference_trajectory = getattr(navigation, "reference_trajectory", None)
        route_length = getattr(reference_trajectory, "length", None)
        try:
            route_length_value = float(route_length)
        except (TypeError, ValueError):
            route_length_value = None
        if route_length_value is not None and math.isfinite(route_length_value):
            info["native_reference_route_length_m"] = route_length_value

    def reward_function(self, vehicle_id: str):
        """Remove ScenarioEnv's terminal bonus for a degenerate native route.

        Mission success/termination is tracker-owned (``done_function`` reads
        only ``mission_snapshot.mission_success``), but the reward signal is
        not: MetaDrive's native ``reward_function`` overwrites ``reward`` with
        the flat ``success_reward`` config value on every step for which its
        own ``_is_arrive_destination`` holds, independent of the thesis
        mission. That native predicate is permanently true for the rest of an
        episode once ``vehicle.navigation.reference_trajectory.length < 2``
        (MetaDrive's own live-built navigation object, structurally
        independent of the mission's route). When the native check disagrees
        with the mission tracker, substitute the dense per-step reward so the
        agent is never trained on a spurious flat bonus.
        """

        reward, step_info = super().reward_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        native_success = bool(self._is_arrive_destination(vehicle))
        thesis_success = self._is_thesis_success(None)
        if native_success and not thesis_success:
            # ScenarioEnv stores the dense pre-terminal reward before it
            # replaces it with ``success_reward``. Restore that value instead
            # of training the policy to exploit a stationary SDC snippet.
            reward = float(step_info.get("step_reward", reward))
            step_info["success_reward_suppressed"] = True
        step_info["thesis_success"] = thesis_success
        return reward, step_info

    def _is_out_of_road(self, vehicle: Any) -> bool:
        """Expose the thesis physical-boundary predicate to native call paths."""

        scene_context = getattr(self, "scene_context", None)
        if scene_context is None:
            return bool(super()._is_out_of_road(vehicle))
        return scene_context.is_physically_out_of_road(self, vehicle)

    def done_function(self, vehicle_id: str):
        runtime = self._mission_runtime
        snapshotter = self._mission_snapshotter
        if runtime is None or not callable(snapshotter) or self._mission_pre_snapshot is None:
            raise RuntimeError("Driving mission runtime is unavailable before done evaluation")
        post_snapshot = snapshotter(self)
        # The causal pre-state of this transition, captured before the commit
        # below replaces it. The at-fault classification must read it, exactly
        # as `evaluate_collision_impact` does (audit 2026-09-06, A6).
        pre_snapshot = self._mission_pre_snapshot
        episode_length = int(self.episode_lengths[vehicle_id])
        if (
            episode_length <= 0
            or getattr(self, "_mission_last_update_episode_length", None) == episode_length
        ):
            # MetaDrive evaluates done during reset and may invoke it once per
            # active agent. BaseEnv increments episode_lengths immediately
            # before done_function only for a committed env.step(), making it
            # the causal discriminator when episode_step is not advanced by a
            # source-specific ScenarioEnv path.
            mission_snapshot = runtime.snapshot
        else:
            mission_snapshot = runtime.update(self._mission_pre_snapshot, post_snapshot)
            # The physical post-state is the causal pre-state for the next
            # transition, while the mission tracker has just committed the
            # new ordered-gate state. Keep both in one immutable snapshot.
            self._mission_pre_snapshot = replace(post_snapshot, mission_snapshot=mission_snapshot)
            self._mission_last_update_episode_length = episode_length
        done, done_info = super().done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        line_only = self.scene_context.is_on_continuous_line(vehicle)
        physical_out = self.scene_context.is_physically_out_of_road(self, vehicle)

        # ScenarioEnv's predicate also includes line flags and (in this local
        # version) negative route completion. Neither is a thesis terminal
        # condition unless the native physical-boundary primitives agree.
        if not physical_out:
            done_info[TerminationState.OUT_OF_ROAD] = False
            # ScenarioEnv maps ROAD_EDGE_BOUNDARY and actual sidewalk contact
            # to the same CRASH_SIDEWALK flag. The adapter has already
            # classified boundary-only contact as non-physical, so it must not
            # re-enter the aggregate termination predicate here.
            native_crash = bool(done_info.get(TerminationState.CRASH, False))
            native_crash_sidewalk = bool(done_info.get(TerminationState.CRASH_SIDEWALK, False))
            done_info[TerminationState.CRASH_SIDEWALK] = False
            collision_without_sidewalk = any(
                bool(done_info.get(key, False))
                for key in (
                    TerminationState.CRASH_VEHICLE,
                    TerminationState.CRASH_OBJECT,
                    TerminationState.CRASH_BUILDING,
                    TerminationState.CRASH_HUMAN,
                    TerminationState.CRASH_SIDEWALK,
                )
            )
            done_info[TerminationState.CRASH] = collision_without_sidewalk or (
                native_crash and not native_crash_sidewalk
            )
            done = self._recompute_terminated(done_info)
        if physical_out:
            done_info[TerminationState.OUT_OF_ROAD] = True
            done = True

        # MetaDrive ScenarioEnv declares every route shorter than 2 m a
        # success. The thesis additionally rejects routes shorter than 10 m,
        # which are too short to provide a meaningful RL episode.
        done_info[TerminationState.SUCCESS] = mission_snapshot.mission_success
        done_info["mission_unreachable"] = False
        done = self._recompute_terminated(done_info)

        # Do not use the native truthy allowed_more_steps branch: zero is a
        # meaningful value in the thesis contract.
        if not done and scenario_time_limit_reached(
            episode_steps=int(self.episode_lengths[vehicle_id]),
            scenario_length=self._scenario_length(),
            extra_steps_after_scenario=int(self.config.get("extra_steps_after_scenario", 0)),
        ):
            done_info[TerminationState.MAX_STEP] = True
            done = False

        # ADR-071, the load-bearing half. A collision the ego is not to blame for
        # truncates instead of terminating, and R1 charges nothing for it.
        #
        # Zeroing the cost while keeping termination would be *worse* than the
        # status quo: the agent would still be punished, through the zero
        # bootstrap, and would additionally have learned that provoking one is
        # free. Truncation closes that by construction -- a truncated episode
        # returns the agent its own expected continuation value, i.e. exactly
        # what it would have obtained by continuing to drive, so provoking the
        # impact buys nothing. This is partial-episode bootstrapping (Pardo,
        # Tavakoli, Levdik & Kormushev, ICML 2018) and it is the
        # termination/truncation distinction this repository already commits to.
        not_at_fault_only = self._collision_is_not_at_fault(post_snapshot, pre_snapshot)
        done_info["not_at_fault_collision"] = bool(not_at_fault_only)
        if not_at_fault_only:
            for key in (
                TerminationState.CRASH,
                TerminationState.CRASH_VEHICLE,
                TerminationState.CRASH_HUMAN,
                TerminationState.CRASH_OBJECT,
                TerminationState.CRASH_BUILDING,
            ):
                done_info[key] = False
            done = self._recompute_terminated(done_info)
            if not done:
                # Truncation, not "keep driving": the physics contact happened
                # and the episode cannot continue meaningfully, so the episode
                # ends while the value target bootstraps from `V(s)`.
                done_info[TerminationState.MAX_STEP] = True

        done_info["crossed_continuous_line"] = bool(line_only)
        done_info["physical_out_of_road"] = bool(physical_out)
        done_info.update(self.scene_context.get_physical_road_diagnostics(self, vehicle))
        done_info["termination_reason"] = self.scene_context.get_termination_reason(
            self, vehicle, done_info
        )
        self._last_done_info = dict(done_info)
        return done, done_info

    def _collision_is_not_at_fault(self, post_snapshot: Any, pre_snapshot: Any) -> bool:
        """Whether this step's contacts exist and are all not the ego's fault.

        Uses the Rulebook's own classifier, never a second implementation, on the
        Rulebook's own inputs: the **pre**-transition ego and the **pre**-transition
        actor records, exactly as `evaluate_collision_impact` reads them. The
        reward and the episode contract therefore classify the same state, so
        "charged" and "terminated" cannot drift apart. Reading the post-state
        actors here (as an earlier revision did) let the two disagree whenever the
        other agent crossed the stopped threshold or the ego's rear half-plane
        within the control step.

        An actor that has no pre-state record but is present in the post-state
        appeared during this control step; R1 charges nothing for it (REQ-EF-13),
        so it does not make the contact at fault here either. An actor present in
        neither snapshot is an instrumentation gap and resolves to ``False``.

        Returns ``False`` when there is no contact at all, when any contact is at
        fault, and when the inputs are not resolvable -- the conservative
        direction in every case, because an undeterminable state must not earn
        the softer ending.
        """

        from thesis_rl.rulebook.v2.components.collision_fault import (
            classify_contact,
            is_at_fault,
        )
        from thesis_rl.rulebook.v2.transition import ego_within_single_lane

        onsets = getattr(post_snapshot, "contact_onset_records", ())
        if not onsets:
            return False
        if pre_snapshot is None:
            return False
        pre_actors_by_id = {actor.actor_id: actor for actor in getattr(pre_snapshot, "actors", ())}
        post_actor_ids = {actor.actor_id for actor in getattr(post_snapshot, "actors", ())}
        adapter = getattr(self, "rulebook_v2_adapter", None)
        cache = getattr(adapter, "initial_cache", None)
        route_lanes = getattr(cache, "route_lanes", ()) if cache is not None else ()
        within_single_lane = ego_within_single_lane(pre_snapshot.ego.footprint, route_lanes)
        for record in onsets:
            actor = pre_actors_by_id.get(record.actor_id)
            if actor is None:
                if record.actor_id in post_actor_ids:
                    # Appeared during the step: R1 = 0 by causal attribution.
                    continue
                return False
            fault = classify_contact(pre_ego=pre_snapshot.ego, actor=actor)
            if is_at_fault(fault, ego_within_single_lane=within_single_lane):
                return False
        return True

    def _scenario_metadata(self) -> dict[str, Any]:
        record = self.current_scenario_record
        vehicle = self.scene_context.get_ego_vehicle(self)
        dimensions = self.scene_context.get_ego_dimensions(vehicle) if vehicle is not None else None
        data_manager = getattr(self.engine, "data_manager", None)
        scenario_id = getattr(data_manager, "current_scenario_id", None)
        record_scenario_id = getattr(record, "scenario_id", None)
        record_uid = getattr(record, "scenario_uid", record_scenario_id)
        record_arm = getattr(record, "primary_arm", getattr(record, "scenario_arm", None))
        payload: dict[str, Any] = {
            "scenario_id": str(scenario_id) if scenario_id is not None else None,
            "scenario_length": int(data_manager.current_scenario_length)
            if data_manager is not None
            else None,
            "scenario_uid": record_uid,
            "scenario_source": getattr(record, "source", None),
            "scenario_arm": record_arm,
            "source": getattr(record, "source", None),
            "arm": record_arm,
            "ego_length": dimensions[0] if dimensions else None,
            "ego_width": dimensions[1] if dimensions else None,
            "assigned_route_lane_ids": list(getattr(record, "assigned_route_lane_ids", ()) or ()),
            "assigned_route_source": getattr(record, "assigned_route_source", None),
        }
        expected_id = str(record_scenario_id) if record_scenario_id is not None else None
        loaded_id = str(scenario_id) if scenario_id is not None else None
        pg_runtime_id_match = (
            expected_id is not None
            and loaded_id is not None
            and expected_id.startswith("PGMap-")
            and loaded_id == expected_id.removeprefix("PGMap-")
        )
        if (
            record is not None
            and scenario_id is not None
            and loaded_id != expected_id
            and not pg_runtime_id_match
        ):
            raise RuntimeError(
                "ScenarioNet catalog/runtime mismatch: "
                f"expected scenario_id={record.scenario_id!r}, loaded={scenario_id!r}"
            )
        return payload

    def reset(self, seed: int | None = None, **kwargs):
        started = time.perf_counter()
        self._active_reset_timing_seconds = {}
        try:
            with self._profile_existing_engine_reset():
                observation, info = super().reset(seed=seed, **kwargs)
        finally:
            self._record_reset_phase("total", time.perf_counter() - started)
            reset_timing = dict(self._active_reset_timing_seconds)
            self._active_reset_timing_seconds = None
        info = dict(info)
        info["_thesis_reset_timing_seconds"] = reset_timing
        metadata = self._scenario_metadata()
        info.update(metadata)
        info.update(self._last_sampling_metadata)
        info.update(self._acl_info())
        source = str(metadata.get("source") or "unknown")
        arm = str(metadata.get("arm") or "unknown")
        self._runtime_stats["resets"] += 1
        self._runtime_stats["resets_by_source"][source] += 1
        self._runtime_stats["resets_by_source_arm"][(source, arm)] += 1
        self._runtime_timing_seconds["reset"] += time.perf_counter() - started
        return observation, info

    def step(self, action):
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        self._attach_route_metrics(info)
        metadata = self._scenario_metadata()
        info.update(metadata)
        info.update(self._last_sampling_metadata)
        info.update(self._acl_info())
        # Publish the final Gymnasium boundary explicitly so every outer
        # wrapper and artifact recorder observes the same flags as the return
        # tuple, including custom ScenarioNet truncation.
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["termination_reason"] = self._last_done_info.get("termination_reason")
        info["crossed_continuous_line"] = bool(
            self._last_done_info.get("crossed_continuous_line", False)
        )
        source = str(metadata.get("source") or "unknown")
        arm = str(metadata.get("arm") or "unknown")
        self._runtime_stats["steps"] += 1
        self._runtime_stats["steps_by_source"][source] += 1
        self._runtime_stats["steps_by_source_arm"][(source, arm)] += 1
        if terminated or truncated:
            reason = str(
                info.get("termination_reason") or ("truncated" if truncated else "terminated")
            )
            self._runtime_stats["episodes"] += 1
            self._runtime_stats["episodes_by_arm"][arm] += 1
            self._runtime_stats["episodes_by_source_arm"][(source, arm)] += 1
            self._runtime_stats["termination_reasons"][reason] += 1
        self._runtime_timing_seconds["step"] += time.perf_counter() - started
        return observation, reward, terminated, truncated, info

    def get_runtime_stats(self) -> dict[str, Any]:
        """Return JSON-safe counters for parent-process/run-level aggregation."""

        result = {
            "resets": int(self._runtime_stats["resets"]),
            "steps": int(self._runtime_stats["steps"]),
            "episodes": int(self._runtime_stats["episodes"]),
            "resets_by_source": dict(self._runtime_stats["resets_by_source"]),
            "steps_by_source": dict(self._runtime_stats["steps_by_source"]),
            "episodes_by_arm": dict(self._runtime_stats["episodes_by_arm"]),
            "termination_reasons": dict(self._runtime_stats["termination_reasons"]),
            "timing_seconds": {
                name: float(value) for name, value in self._runtime_timing_seconds.items()
            },
        }
        for key in ("resets_by_source_arm", "steps_by_source_arm", "episodes_by_source_arm"):
            result[key] = {
                source: {
                    arm: int(count)
                    for (counter_source, arm), count in self._runtime_stats[key].items()
                    if counter_source == source
                }
                for source in sorted({source for source, _arm in self._runtime_stats[key]})
            }
        return result


__all__ = ["SceneContextAdapter", "ThesisScenarioEnv", "scenario_time_limit_reached"]
