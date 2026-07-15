"""DEC-007 canonical geometry normalization and stable IDs."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Iterator

import shapely
from shapely.geometry.base import BaseGeometry


PRECISION_GRID_M = 1.0e-3


class CanonicalGeometryError(ValueError):
    """Geometry violates the frozen Rulebook v2 canonicalization contract."""


def _coordinate_tuples(geometry: BaseGeometry) -> Iterator[tuple[float, ...]]:
    geometry_type = geometry.geom_type
    if geometry_type == "Polygon":
        yield from geometry.exterior.coords
        for interior in geometry.interiors:
            yield from interior.coords
        return
    if geometry_type.startswith("Multi") or geometry_type == "GeometryCollection":
        for member in geometry.geoms:
            yield from _coordinate_tuples(member)
        return
    if hasattr(geometry, "coords"):
        yield from geometry.coords


def _validate_geometry(geometry: BaseGeometry) -> None:
    if geometry.is_empty:
        raise CanonicalGeometryError("Geometry must not be empty.")
    if not geometry.is_valid:
        raise CanonicalGeometryError("Geometry is invalid before precision snapping.")
    for coordinate in _coordinate_tuples(geometry):
        if len(coordinate) < 2 or not all(isfinite(float(value)) for value in coordinate[:2]):
            raise CanonicalGeometryError("Geometry XY coordinates must be finite.")


def _validate_noncollapsed(geometry: BaseGeometry) -> None:
    if geometry.is_empty:
        raise CanonicalGeometryError("Geometry collapsed to empty after precision snapping.")
    if not geometry.is_valid:
        raise CanonicalGeometryError("Geometry is invalid after precision snapping.")
    if geometry.geom_type in {"Polygon", "MultiPolygon"} and geometry.area <= 0.0:
        raise CanonicalGeometryError("Polygonal geometry collapsed after precision snapping.")
    if geometry.geom_type in {"LineString", "MultiLineString"} and geometry.length <= 0.0:
        raise CanonicalGeometryError("Linear geometry collapsed after precision snapping.")


def canonicalize_geometry(geometry: BaseGeometry, *, precision_grid_m: float = PRECISION_GRID_M) -> BaseGeometry:
    """Snap, validate and normalize a geometry under DEC-007.

    No repair such as ``make_valid`` is applied: invalid input is an eligibility
    error, never a source-dependent fallback.
    """

    if not isfinite(precision_grid_m) or precision_grid_m <= 0.0:
        raise ValueError("precision_grid_m must be finite and positive")
    _validate_geometry(geometry)
    snapped = shapely.set_precision(geometry, precision_grid_m, mode="valid_output")
    _validate_noncollapsed(snapped)
    normalized = shapely.normalize(snapped)
    _validate_noncollapsed(normalized)
    return normalized


def canonical_geometry_wkb(geometry: BaseGeometry, *, precision_grid_m: float = PRECISION_GRID_M) -> bytes:
    """Return the frozen 2D big-endian, no-SRID canonical WKB representation."""

    normalized = canonicalize_geometry(geometry, precision_grid_m=precision_grid_m)
    return shapely.to_wkb(
        normalized,
        output_dimension=2,
        byte_order=0,
        include_srid=False,
        flavor="extended",
    )


def stable_geometry_id(
    *,
    scenario_id: str,
    namespace: str,
    feature_type: str,
    geometry: BaseGeometry,
    precision_grid_m: float = PRECISION_GRID_M,
) -> str:
    """Build the SHA-256 synthetic ID payload specified by DEC-007."""

    if not scenario_id or not namespace or not feature_type:
        raise ValueError("scenario_id, namespace and feature_type must be non-empty")
    payload = {
        "scenario_id": scenario_id,
        "namespace": namespace,
        "feature_type": feature_type,
        "canonical_wkb_hex": canonical_geometry_wkb(
            geometry, precision_grid_m=precision_grid_m
        ).hex(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
