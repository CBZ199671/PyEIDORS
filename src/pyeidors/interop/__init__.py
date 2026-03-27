"""Interop helpers for exchanging native geometries with EIDORS."""

from .geometry_exchange import (
    STANDARD_INTEROP_FORMAT,
    build_boundary_edges,
    build_electrode_arrays,
    build_mesh_from_exchange_mat,
    export_forward_csv,
    load_forward_csv,
    save_exchange_mat,
    validate_exchange_payload,
)

__all__ = [
    "STANDARD_INTEROP_FORMAT",
    "build_boundary_edges",
    "build_electrode_arrays",
    "build_mesh_from_exchange_mat",
    "export_forward_csv",
    "load_forward_csv",
    "save_exchange_mat",
    "validate_exchange_payload",
]
