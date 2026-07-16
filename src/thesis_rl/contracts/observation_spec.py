"""Backward-compatible import for the approved semantic v1.1 schema."""

from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV11

# Kept only to avoid breaking import paths owned by historical callers. New
# code must use SemanticObservationSchemaV11 by its explicit name.
ObservationSpec = SemanticObservationSchemaV11

__all__ = ["ObservationSpec"]
