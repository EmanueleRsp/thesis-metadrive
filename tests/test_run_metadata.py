from __future__ import annotations

import yaml
from omegaconf import OmegaConf

from thesis_rl.runtime.io.metadata import save_run_metadata


def test_run_metadata_records_effective_algorithm_and_transition_replay(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "metadata-test",
            "agent": {
                "planner": {
                    "algorithm": {
                        "name": "td3_sb3",
                        "transition_replay": {
                            "enabled": True,
                            "n_steps": 3,
                            "prioritized": True,
                            "persistence": {"enabled": True, "trigger": "final_or_manual"},
                        },
                    }
                }
            },
            "env": {"name": "metadrive"},
            "reward": {"type": "rulebook", "behavior": "scalar_reward"},
            "scalarization": {"specification_id": "SCAL-V1.0", "mode": "bounded_satisfaction_rank"},
            "curriculum": {"name": "scenario_acl_scenarionet"},
            "analysis": {"experiment_group": "metadata-test", "include_in_comparison": False},
            "seed": 42,
            "experiment": {"total_timesteps": 2000, "eval_interval": 1000, "eval_episodes": 2},
        }
    )

    path = save_run_metadata(cfg, tmp_path)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert metadata["algorithm"] == "td3_sb3"
    assert metadata["transition_replay"]["n_steps"] == 3
    assert metadata["transition_replay"]["prioritized"] is True
    assert metadata["transition_replay"]["persistence"]["enabled"] is True


def test_run_metadata_records_eval_protocol_reproducibility_fields(tmp_path) -> None:
    """EVAL-PROTOCOL REQ-015: dirty flag, dependency versions, and additional
    specification identities must be present in every run's metadata."""

    cfg = OmegaConf.create(
        {
            "name": "metadata-repro-test",
            "agent": {"planner": {"algorithm": {"name": "sac_sb3"}}},
            "env": {"name": "metadrive"},
            "reward": {"type": "rulebook", "behavior": "scalar_reward"},
            "seed": 7,
            "experiment": {"total_timesteps": 2000},
        }
    )

    path = save_run_metadata(cfg, tmp_path)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "dirty" in metadata["git"]
    assert isinstance(metadata["git"]["dirty"], (bool, type(None)))
    assert metadata["dependencies"]["torch_version"]
    assert "sb3_version" in metadata["dependencies"]
    assert "sb3_commit" in metadata["dependencies"]
    assert "metadrive_version" in metadata["dependencies"]
    assert "scenarionet_version" in metadata["dependencies"]
    assert metadata["observation"]["specification_id"] == "OBS-V1.2"
    assert metadata["encoder"]["specification_id"] == "ENC-V1.2"
    assert metadata["acl"]["specification_id"] == "ACL-SN-EMA-001"
    assert metadata["transition_replay"]["specification_id"] == "TRANSITION-REPLAY"
    assert metadata["evaluation_protocol"]["specification_id"] == "EVAL-PROTOCOL"
    assert metadata["evaluation_protocol"]["version"] == "1.0"


def test_run_metadata_records_panel_manifest_hash(tmp_path) -> None:
    """EVAL-PROTOCOL REQ-015/REQ-004: the frozen validation/test panel
    manifest referenced by ``env.provider.panel_manifest_path`` must be
    snapshotted and hashed into run_metadata.yaml, same as the other
    ScenarioNet definition artifacts."""

    manifest_path = tmp_path / "validation_panel_manifest_v1.json"
    manifest_path.write_text('{"split": "validation", "scenario_uids": ["a", "b"]}', encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "name": "panel-manifest-test",
            "agent": {"planner": {"algorithm": {"name": "td3_sb3"}}},
            "env": {
                "name": "scenarionet",
                "split": "validation",
                "provider": {"kind": "fixed_sequence", "panel_manifest_path": str(manifest_path)},
            },
            "reward": {"type": "rulebook", "behavior": "scalar_reward"},
            "seed": 3,
            "experiment": {"total_timesteps": 2000},
        }
    )

    artifacts_dir = tmp_path / "run"
    path = save_run_metadata(cfg, artifacts_dir)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))

    snapshots = metadata["scenarionet"]["artifact_snapshots"]
    assert "panel_manifest_validation_panel_manifest_v1" in snapshots
    assert snapshots["panel_manifest_validation_panel_manifest_v1_sha256"]


def test_update_run_metadata_records_disposition(tmp_path) -> None:
    """EVAL-PROTOCOL REQ-011: disposition defaults are additive-patchable and
    are never set implicitly by save_run_metadata."""

    cfg = OmegaConf.create({"name": "disposition-test", "seed": 1})
    path = save_run_metadata(cfg, tmp_path)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert metadata.get("disposition") is None

    from thesis_rl.runtime.io.metadata import update_run_metadata

    update_run_metadata(
        tmp_path,
        {
            "disposition": "non_convergent",
            "disposition_rationale": "Completed but low success rate; not an infra failure.",
        },
    )
    updated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert updated["disposition"] == "non_convergent"
    assert updated["disposition_rationale"]
    # Unrelated fields already written must survive the additive patch.
    assert updated["algorithm"] == metadata["algorithm"]
