"""ScenarioNet environment with the thesis-specific episode contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
import math
import time
from typing import Any

from thesis_rl.envs.scene_context import SceneContextAdapter


def scenario_time_limit_reached(
    *, episode_steps: int, scenario_length: int, extra_steps_after_scenario: int
) -> bool:
    """Return whether the exported scenario horizon (plus the configured tail) is met."""

    if episode_steps < 0 or scenario_length <= 0 or extra_steps_after_scenario < 0:
        raise ValueError(
            "episode_steps >= 0, scenario_length > 0 and extra_steps >= 0 are required"
        )
    return episode_steps >= scenario_length + extra_steps_after_scenario


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
                "extra_steps_after_scenario": 50,
                # ScenarioEnv treats a reference trajectory shorter than two
                # metres as an immediate success.  That is useful for its
                # replay use case, but corrupts an RL success metric: some
                # converted Waymo SDC tracks are stationary or near-stationary.
                "success_route_completion_threshold": 0.95,
                "minimum_success_route_length_m": 10.0,
                "rulebook_v2_disable_vehicle_yield_for_benchmark": False,
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
        if self._active_reset_timing_seconds is not None:
            self._active_reset_timing_seconds[name] = (
                self._active_reset_timing_seconds.get(name, 0.0) + float(seconds)
            )

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

    @staticmethod
    def _build_causal_frame_builder(
        scenario: Mapping[str, Any], record: Any, config: Mapping[str, Any]
    ) -> Any:
        from thesis_rl.envs.observations.assigned_route import (
            AssignedRouteWaypointAdapter,
            MapRouteNavigationObservation22,
        )
        from thesis_rl.envs.observations.causal_lidar import CausalLidarFrameBuilder
        from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper

        static_result = ThesisScenarioEnv._build_static_adapter_result(scenario, record)
        route = static_result.assigned_route_polyline
        navigation = MapRouteNavigationObservation22(
            AssignedRouteWaypointAdapter(route, num_waypoints=10, spacing_m=5.0)
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
            builder = self._build_causal_frame_builder(scenario, record, self.config)
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
            # The live Rulebook cache may apply a source-to-simulator elevation
            # datum translation at reset. Reuse that committed geometry for
            # semantic observations so the causal route identity is exact in
            # 2.5D, rather than comparing a pre-alignment static copy.
            rulebook_adapter = getattr(self, "rulebook_v2_adapter", None)
            episode_cache = getattr(rulebook_adapter, "initial_cache", None)
            route = (
                episode_cache.route_polyline
                if episode_cache is not None
                else static_result.assigned_route_polyline
            )
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

        context = CausalSceneContext(adapter.initial_cache, snapshot, adapter.initial_memory)
        self.causal_scene_context = context
        for builder in getattr(self, "_causal_semantic_builders", ()):
            builder.reset()
            builder.commit_context(context)

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
            align_episode_cache_to_live_elevation,
            build_episode_cache,
            initial_memory_for_snapshot,
            transition_evaluator_factory,
        )
        from thesis_rl.rulebook.v2.wrapper import RulebookV2Adapter

        scenario = getattr(getattr(self.engine, "data_manager", None), "current_scenario", None)
        record = self.current_scenario_record
        if not isinstance(scenario, Mapping) or record is None:
            raise RuntimeError("Rulebook v2 adapter requires a loaded scenario and catalog record")
        static_result = self._build_static_adapter_result(scenario, record)
        cache = build_episode_cache(static_result)
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
        cache = align_episode_cache_to_live_elevation(cache, initial_snapshot)
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
            snapshotter=snapshotter.capture,
            transition_evaluator=transition_evaluator_factory(transition_config),
            initial_memory=initial_memory_for_snapshot(initial_snapshot, cache),
            initial_cache=cache,
        )

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
        """Reject ScenarioEnv's short/static-trajectory success shortcut."""

        navigation = getattr(vehicle, "navigation", None)
        completion, _raw = self._normalise_route_completion(
            getattr(navigation, "route_completion", None)
        )
        if completion is None:
            return False

        reference_trajectory = getattr(navigation, "reference_trajectory", None)
        route_length = getattr(reference_trajectory, "length", None)
        try:
            route_length_value = float(route_length) if route_length is not None else None
        except (TypeError, ValueError):
            route_length_value = None
        if route_length_value is not None and (
            not math.isfinite(route_length_value)
            or route_length_value < float(self.config.get("minimum_success_route_length_m", 10.0))
        ):
            return False

        return completion >= float(self.config.get("success_route_completion_threshold", 0.95))

    def _attach_route_metrics(self, info: dict[str, Any]) -> None:
        vehicle = self.scene_context.get_ego_vehicle(self)
        navigation = getattr(vehicle, "navigation", None)
        completion, raw_completion = self._normalise_route_completion(
            info.get("route_completion", getattr(navigation, "route_completion", None))
        )
        if completion is not None:
            info["route_completion"] = completion
        if raw_completion is not None and raw_completion != completion:
            info["raw_route_completion"] = raw_completion
        reference_trajectory = getattr(navigation, "reference_trajectory", None)
        route_length = getattr(reference_trajectory, "length", None)
        try:
            route_length_value = float(route_length)
        except (TypeError, ValueError):
            route_length_value = None
        if route_length_value is not None and math.isfinite(route_length_value):
            info["reference_route_length_m"] = route_length_value

    def reward_function(self, vehicle_id: str):
        """Remove ScenarioEnv's terminal bonus for a degenerate Waymo route."""

        reward, step_info = super().reward_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        native_success = bool(self._is_arrive_destination(vehicle))
        thesis_success = self._is_thesis_success(vehicle)
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
        done_info[TerminationState.SUCCESS] = self._is_thesis_success(vehicle)
        done = self._recompute_terminated(done_info)

        # Do not use the native truthy allowed_more_steps branch: zero is a
        # meaningful value in the thesis contract.
        if not done and scenario_time_limit_reached(
            episode_steps=int(self.episode_lengths[vehicle_id]),
            scenario_length=self._scenario_length(),
            extra_steps_after_scenario=int(self.config.get("extra_steps_after_scenario", 50)),
        ):
            done_info[TerminationState.MAX_STEP] = True
            done = False

        done_info["crossed_continuous_line"] = bool(line_only)
        done_info["physical_out_of_road"] = bool(physical_out)
        done_info.update(self.scene_context.get_physical_road_diagnostics(self, vehicle))
        done_info["termination_reason"] = self.scene_context.get_termination_reason(
            self, vehicle, done_info
        )
        self._last_done_info = dict(done_info)
        return done, done_info

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
