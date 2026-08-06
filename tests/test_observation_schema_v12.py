from __future__ import annotations

import numpy as np
import torch

from thesis_rl.contracts.observation_schema import (
    SemanticObservationBatchV12,
    SemanticObservationSchemaV12,
)


def _batch() -> SemanticObservationBatchV12:
    schema = SemanticObservationSchemaV12()
    value = 0.0
    arrays: dict[str, np.ndarray] = {}
    for name, shape in schema.group_shapes.items():
        size = int(np.prod(shape))
        arrays[name] = np.arange(value, value + size, dtype=np.float32).reshape(shape)
        value += size
    return SemanticObservationBatchV12(**arrays)


def test_v12_schema_has_normative_dimensions_and_order() -> None:
    schema = SemanticObservationSchemaV12()

    assert schema.version == "1.2-perception-bounded-mission-route-v1"
    assert schema.flat_dim == 3009
    assert schema.raw_token_count == 143
    assert tuple(schema.group_shapes) == (
        "ego_history",
        "ego_history_mask",
        "ego_current",
        "route",
        "route_mask",
        "dynamic",
        "dynamic_mask",
        "static",
        "static_mask",
        "lane_road",
        "controls",
        "controls_mask",
        "interactions",
        "interactions_mask",
        "context_history",
        "context_history_mask",
        "signal_onset_state",
    )


def test_v12_schema_numpy_round_trip_preserves_all_groups() -> None:
    schema = SemanticObservationSchemaV12()
    batch = _batch()

    flat = schema.flatten_numpy(batch)
    restored = schema.unflatten_numpy(flat)

    assert flat.shape == (3009,)
    for name in schema.group_shapes:
        np.testing.assert_array_equal(getattr(restored, name), getattr(batch, name))


def test_v12_schema_torch_unflatten_preserves_autograd_dtype_and_device() -> None:
    schema = SemanticObservationSchemaV12()
    flat = torch.arange(2 * schema.flat_dim, dtype=torch.float32).reshape(2, -1)
    flat.requires_grad_()

    structured = schema.unflatten_torch(flat)
    loss = structured.dynamic.sum() + structured.context_history.sum()
    loss.backward()

    assert structured.context_history.dtype is torch.float32
    assert structured.context_history.device == flat.device
    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()


def test_v12_schema_fingerprint_is_deterministic_and_sensitive_to_contract() -> None:
    schema = SemanticObservationSchemaV12()

    assert schema.fingerprint_sha256() == SemanticObservationSchemaV12().fingerprint_sha256()
    changed = schema.canonical_dict()
    changed["flat_order"] = list(reversed(changed["flat_order"]))

    assert schema.fingerprint_sha256_from_canonical_dict(changed) != schema.fingerprint_sha256()
