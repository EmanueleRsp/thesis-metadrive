# Documentation

Start with the repository [README.md](../README.md) for the canonical clone,
setup, build, and run workflow.

This file is only a lightweight map of the `docs/` folder.

## Setup And Validation

- [setup/environment_setup.md](setup/environment_setup.md): `.env` usage,
  host/container path conventions, and the new-machine bring-up checklist
- [setup/validation_commands.md](setup/validation_commands.md): smoke checks
  and validation commands
- [setup/comparison_run_commands.md](setup/comparison_run_commands.md):
  current Docker run commands for comparison workflows

## Operational Workflows

- [workflows/algorithm_selection_playbook.md](workflows/algorithm_selection_playbook.md):
  algorithm selection and qualification flow
- [workflows/analysis_pipeline.md](workflows/analysis_pipeline.md): analysis
  outputs and pipeline conventions
- [workflows/thesis_experiment_roadmap.md](workflows/thesis_experiment_roadmap.md):
  current experiment roadmap

## Architecture And Design

- [architecture/architecture.md](architecture/architecture.md)
- [architecture/sb3_fork_migration_plan.md](architecture/sb3_fork_migration_plan.md)
- [architecture/initial_design_decisions.md](architecture/initial_design_decisions.md)
- [architecture/metadrive_assumptions.md](architecture/metadrive_assumptions.md)

## Reference Material

- [specs/algorithm_comparison_protocol.md](specs/algorithm_comparison_protocol.md)
- [specs/csv_evaluation_objectives.md](specs/csv_evaluation_objectives.md)
- [specs/rulebook_v1_specification.md](specs/rulebook_v1_specification.md)
- [workflows/rulebook_v1_calibration.md](workflows/rulebook_v1_calibration.md)
- [specs/live_eval_video_protocol.md](specs/live_eval_video_protocol.md)
- [specs/curriculum_learning_specification.md](specs/curriculum_learning_specification.md)
- [specs/scenario_acl_implementation_plan.md](specs/scenario_acl_implementation_plan.md)

## Archive

- [archive/README.md](archive/README.md)
- [archive/plans/hydra_config_strategy.md](archive/plans/hydra_config_strategy.md)
- [archive/plans/implementation_plan.md](archive/plans/implementation_plan.md)
- [archive/plans/open_questions.md](archive/plans/open_questions.md)
- [archive/plans/ppo_sb3_porting_plan.md](archive/plans/ppo_sb3_porting_plan.md)
- [archive/plans/sac_sb3_porting_plan.md](archive/plans/sac_sb3_porting_plan.md)
- [archive/plans/td3_sb3_porting_plan.md](archive/plans/td3_sb3_porting_plan.md)

Archived documents are kept for context and historical decisions. They are not
the preferred entry point for current implementation work.

## Source Of Truth

If a document conflicts with the root `README.md`, the current code, or the
portable `third_party/` plus `.env` plus `/workspace/{outputs,data}` layout,
the document should be updated.
