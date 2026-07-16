# Documentation

Start with the repository [README.md](../README.md) for clone, setup, build, and
run instructions. Start specification-driven work with the
[authority index](project_index.md) and [engineering workflow](engineering_workflow.md).

## Specification-Driven Development

- [project_index.md](project_index.md): authoritative versions, implementation
  status, candidates, and missing documents
- [engineering_workflow.md](engineering_workflow.md): lifecycle, approval gates,
  protected tests, and completion criteria
- [specifications/](specifications/): user-approved scientific and functional
  contracts
- [implementation/](implementation/): living ExecPlans and implementation
  trackers
- [protocols/](protocols/): evaluation, reporting, and operational protocols
- [decisions/](decisions/): architecture decision records
- [templates/specification_template.md](templates/specification_template.md)
- [templates/adr_template.md](templates/adr_template.md)
- [templates/feature_implementation_request.md](templates/feature_implementation_request.md)
- [templates/session_continuation_request.md](templates/session_continuation_request.md)

## Authoritative Specifications

- [Rulebook v2, version 4.6](specifications/rulebook_v2_spec.md)
- [Automatic curriculum learning](specifications/curriculum_learning_specification.md)
- [ScenarioNet integration v1](specifications/scenarionet_integration_spec_v1.md)

The semantic observation v1.1 specification has not yet been supplied. The
authority index is definitive when this summary and the registry differ.

## Implementation Plans

- [Rulebook v2](implementation/rulebook_v2_implementation_plan.md)
- [Scenario ACL](implementation/scenario_acl_implementation_plan.md)
- [ScenarioNet integration](implementation/scenarionet_integration_implementation_plan.md)
- [Documentation restructure](implementation/repository_documentation_restructure_exec_plan.md)
- [Repository quality commands](implementation/repository_quality_commands_exec_plan.md)

## Protocols

- [Algorithm comparison](protocols/algorithm_comparison_protocol.md)
- [CSV evaluation objectives](protocols/csv_evaluation_objectives.md)
- [Live evaluation video](protocols/live_eval_video_protocol.md)

These protocols remain candidates unless `project_index.md` explicitly records
approval and an authoritative version.

## Setup And Validation

- [Environment setup](setup/environment_setup.md)
- [Validation commands](setup/validation_commands.md)
- [Comparison run commands](setup/comparison_run_commands.md)
- [ScenarioNet/Waymo conversion](setup/scenarionet_waymo_conversion.md)

## Operational Workflows

- [Algorithm selection](workflows/algorithm_selection_playbook.md)
- [Analysis pipeline](workflows/analysis_pipeline.md)
- [Thesis experiment roadmap](workflows/thesis_experiment_roadmap.md)
- [Scenario ACL parallelization](workflows/scenario_acl_parallelization.md)
- [Rulebook v1 calibration](workflows/rulebook_v1_calibration.md)

## Architecture And Archive

- [Current architecture](architecture/architecture.md)
- [SB3 fork migration](architecture/sb3_fork_migration_plan.md)
- [Initial design decisions](architecture/initial_design_decisions.md)
- [MetaDrive assumptions](architecture/metadrive_assumptions.md)
- [Archive index](archive/README.md)

Archived material is historical context, not a current source of authority.

## Source Of Truth

Scientific behavior follows explicit user approvals, the specification selected
in `project_index.md`, and applicable approved ADRs. Code, configuration, and
pinned dependencies provide implementation evidence but do not silently override
approved scientific requirements. Resolve conflicts in the relevant ExecPlan.
