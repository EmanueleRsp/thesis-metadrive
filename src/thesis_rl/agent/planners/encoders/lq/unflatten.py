"""Compatibility accessors backed by the sole v1.1 observation schema."""

from thesis_rl.contracts.observation_schema import (
    SemanticObservationSchemaV11,
    SemanticObservationTensorBatch,
)

StructuredObservation = SemanticObservationTensorBatch


def unflatten_observation(obs_flat, spec: SemanticObservationSchemaV11 | None = None):
    schema = spec or SemanticObservationSchemaV11()
    return schema.unflatten_torch(obs_flat)


__all__ = ["StructuredObservation", "unflatten_observation"]
