"""ScenarioNet environment with the thesis-specific episode contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
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
        self._last_sampling_metadata: dict[str, Any] = {}

    def _select_provider_seed(self, force_seed: int | None) -> int | None:
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
        self.config.pop("assigned_route_lane_ids", None)
        self.config.pop("assigned_route_source", None)
        lane_ids = tuple(getattr(record, "assigned_route_lane_ids", ()) or ())
        if not lane_ids:
            raise ValueError(
                "Scenario record is missing assigned_route_lane_ids; "
                "exclude it before runtime reset"
            )
        self.config["assigned_route_lane_ids"] = lane_ids
        self.config["assigned_route_source"] = getattr(record, "assigned_route_source", None)

    def _reset_global_seed(self, force_seed=None):
        provider_seed = self._select_provider_seed(force_seed)
        if provider_seed is not None:
            self.seed(provider_seed)
            self._inject_assigned_route_metadata_into_scenario()
            return
        super()._reset_global_seed(force_seed)
        self._inject_assigned_route_metadata_into_scenario()

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
            from thesis_rl.envs.observations.causal_semantic import CausalSemanticBatchBuilder

            static_result = self._build_static_adapter_result(scenario, record)
            semantic_builder = CausalSemanticBatchBuilder(
                route=static_result.assigned_route_polyline,
                route_lanes=static_result.route_lanes,
                context_provider=lambda: self.causal_scene_context,
                history_length=5,
                control_timestep_s=0.1,
                dynamic_radius_m=50.0,
                static_radius_m=50.0,
                control_radius_m=80.0,
                prediction_horizon_s=3.0,
                vertical_tolerance_m=3.0,
            )
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

    def _on_causal_context_committed(self, context: object) -> None:
        for builder in getattr(self, "_causal_semantic_builders", ()):
            builder.commit_context(context)

    def _refresh_causal_observation(self, previous_observation: object) -> object:
        """Rebuild only observation payloads after the causal commit boundary."""

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
        return rebuilt if getattr(self, "is_multi_agent", False) else next(iter(rebuilt.values()))

    def _get_reset_return(self, reset_info):
        self._install_causal_observation_builder()
        self._prepare_initial_causal_context()
        return super()._get_reset_return(reset_info)

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
        observation, info = super().reset(seed=seed, **kwargs)
        info = dict(info)
        metadata = self._scenario_metadata()
        info.update(metadata)
        info.update(self._last_sampling_metadata)
        source = str(metadata.get("source") or "unknown")
        arm = str(metadata.get("arm") or "unknown")
        self._runtime_stats["resets"] += 1
        self._runtime_stats["resets_by_source"][source] += 1
        self._runtime_stats["resets_by_source_arm"][(source, arm)] += 1
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        self._attach_route_metrics(info)
        metadata = self._scenario_metadata()
        info.update(metadata)
        info.update(self._last_sampling_metadata)
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
