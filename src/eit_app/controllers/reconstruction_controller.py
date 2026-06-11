"""Runs EIT reconstruction in a background QThread.

Accepts reference/target frame pairs, builds MeasurementDataset,
and calls pyeidors EITSystem for difference reconstruction.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import importlib
import io
import json
import logging
import math
from functools import lru_cache
import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from eit_app.models.forward_model_config import (
    drive_mode_for_mesh_dimension,
    electrode_level_fractions_for_rings,
    parse_complex_scalar,
    parse_complex_scalar_list,
)
from eit_app.models.frame_model import FrameData
from pyeidors.cache.keys import update_digest_with_array_payload
from pyeidors.runtime_paths import pyeidors_cache_path, resolve_pyeidors_cache_dir
from pyeidors.utils.numeric_ops import (
    all_finite_values,
    any_not_equal_values,
    has_nonzero_imaginary,
    min_alpha_for_value_floor,
)

log = logging.getLogger(__name__)

_SYSTEM_CACHE_LOCK = threading.Lock()
_SYSTEM_CACHE_MAX_ITEMS = 4
_SYSTEM_CACHE_MAX_BYTES = 512 * 1024 * 1024
_SYSTEM_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_SYSTEM_CACHE_SIZES: dict[tuple[Any, ...], int] = {}

_FAST_CONTEXT_CACHE_LOCK = threading.Lock()
_FAST_CONTEXT_CACHE_MAX_ITEMS = 4
_FAST_CONTEXT_CACHE_MAX_BYTES = 512 * 1024 * 1024
_FAST_CONTEXT_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_FAST_CONTEXT_CACHE_SIZES: dict[tuple[Any, ...], int] = {}
LINEARIZED_SINGLE_STEP_AUTO_MAX_MEASUREMENTS = 512
_SINGLE_STEP_SIGNATURE_SCHEMA_VERSION = "single_step_signature_schema_v1"
_SINGLE_STEP_JACOBIAN_CALCULATOR = "EidorsJacobianAdapter"
_SINGLE_STEP_JACOBIAN_MATH_CONVENTION = "eidors_adapter_difference_dv_dsigma_v4"
_SINGLE_STEP_PROJECTION_MATH_CONVENTION = "difference_projection_weights_v3"
_SINGLE_STEP_OPERATOR_MATH_CONVENTION = "noser_jtj_lambda_diag_jtj_v1"
_SINGLE_STEP_CACHED_ALGORITHM_VERSION = "eidors_noser_single_step_v5"
_ONE_STEP_RM_SIGNATURE_SCHEMA_VERSION = "one_step_rm_signature_schema_v2"
_ONE_STEP_RM_JACOBIAN_BUILD_CONVENTION = "dense_eidors_adapter_jacobian_v3"
_ONE_STEP_RM_PRIOR_MATH_CONVENTION = "singular_graph_prior_param_form_hp2_rtr_v5"
_ONE_STEP_RM_ALGORITHM_VERSION = "one_step_rm_auto_build_dense_jacobian_v7"
_ONE_STEP_RM_CONTENT_CONTRACT = "one_step_rm_hdf5_dense_fit_jacobian_contract_v1"
_RM_ARTIFACT_CACHE_LOCK = threading.Lock()
_RM_ARTIFACT_CACHE_MAX_ITEMS = 4
_RM_ARTIFACT_CACHE_MAX_BYTES = 512 * 1024 * 1024
_RM_ARTIFACT_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_RM_FIT_JACOBIAN_CACHE_LOCK = threading.Lock()
_RM_FIT_JACOBIAN_CACHE_MAX_ITEMS = 2
_RM_FIT_JACOBIAN_CACHE_MAX_BYTES = 512 * 1024 * 1024
_RM_FIT_JACOBIAN_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_RM_FIT_JACOBIAN_CACHE_SIZES: dict[str, int] = {}
_RM_ARTIFACT_META_KEYS = (
    "rm_artifact_path",
    "dual_model_rm_path",
    "greit_rm_path",
    "greit_common_config_artifact_path",
    "greit_common_config_path",
    "common_greit_rm_path",
    "reconstruction_matrix_path",
)
_PRODUCTION_RM_ROUTE_TASKS = {
    "noser_rm": "T100",
    "pseudo3d_noser_rm": "T100",
    "laplace_rm": "T101",
    "curvature_rm": "T101",
    "greit": "T105",
    "greit3d_rm": "T105",
}
_ONE_STEP_RM_ROUTE_REGULARIZATION = {
    "noser_rm": "noser",
    "pseudo3d_noser_rm": "noser",
    "laplace_rm": "laplace",
    "curvature_rm": "curvature",
}
_AUTO_BUILD_RM_ROUTES = frozenset(_ONE_STEP_RM_ROUTE_REGULARIZATION)
_GREIT_REGISTRY_ROUTES = frozenset({"greit", "greit_rm", "greit2d_rm", "greit3d_rm"})
_EIT_SYSTEM_DIFFERENCE_PRESETS = frozenset(
    {"eidors_demo3d_tv", "eidors_one_step_noser", "sphere_multistep_noser"}
)
_DEFAULT_EIT_SYSTEM_DIFFERENCE_PRESET = "eidors_one_step_noser"


def build_difference_vector(*args, **kwargs):
    from pyeidors.data.difference import build_difference_vector as _build

    return _build(*args, **kwargs)


def effective_pattern_layout_for_3d_mesh(*args, **kwargs):
    from pyeidors.electrodes.layout import (
        effective_pattern_layout_for_3d_mesh as _resolve,
    )

    return _resolve(*args, **kwargs)


def probe_petsc_cuda_runtime() -> dict[str, Any]:
    from pyeidors.perf.capabilities import probe_petsc_cuda_runtime as _probe

    return _probe()


def resolve_3d_cuda_forward_solver_policy(*args, **kwargs) -> dict[str, Any]:
    from pyeidors.perf.forward_solver_policy import (
        resolve_3d_cuda_forward_solver_policy as _resolve,
    )

    return _resolve(*args, **kwargs)


def resolve_3d_cuda_mat_solve_policy(*args, **kwargs) -> dict[str, Any]:
    from pyeidors.perf.forward_solver_policy import (
        resolve_3d_cuda_mat_solve_policy as _resolve,
    )

    return _resolve(*args, **kwargs)


def resolve_greit_common_config_artifact_path_from_meta(*args, **kwargs):
    from pyeidors.inverse.greit_warmup import (
        resolve_greit_common_config_artifact_path_from_meta as _resolve,
    )

    return _resolve(*args, **kwargs)


def precompute_greit_common_config(*args, **kwargs):
    from pyeidors.inverse.greit_warmup import precompute_greit_common_config as _warmup

    return _warmup(*args, **kwargs)


def greit_artifact_signature_payload(*args, **kwargs):
    from pyeidors.inverse.greit_registry import greit_artifact_signature_payload as _sig

    return _sig(*args, **kwargs)


def resolve_or_build_greit_artifact(*args, **kwargs):
    from pyeidors.inverse.greit_registry import resolve_or_build_greit_artifact as _res

    return _res(*args, **kwargs)


def _total_electrodes_from_meta(meta: dict[str, Any]) -> int:
    return max(int(meta.get("n_elec", 16)), 1) * max(int(meta.get("n_rings", 1)), 1)


def _request_measurement_count(req: ReconstructionRequest) -> int:
    try:
        return int(req.reference_frame.to_measurement_vector(req.use_part).size)
    except Exception:
        return 0


def _rm_artifact_measurement_mismatch_message(
    *,
    path: Path,
    artifact_columns: int,
    request_measurements: int,
) -> str:
    return (
        "RM artifact measurement dimension "
        f"{int(artifact_columns)} does not match request measurement dimension "
        f"{int(request_measurements)} for {path}. Rebuild/select an artifact for "
        "the current stimulation/measurement protocol."
    )


def _validate_rm_artifact_measurement_dimension(
    artifact: dict[str, Any],
    *,
    path: Path,
    expected_n_measurements: int | None,
) -> None:
    if expected_n_measurements is None or int(expected_n_measurements) <= 0:
        return
    rm_shape = _rm_artifact_matrix_shape(artifact)
    artifact_columns = int(rm_shape[1])
    request_measurements = int(expected_n_measurements)
    if artifact_columns != request_measurements:
        raise ValueError(
            _rm_artifact_measurement_mismatch_message(
                path=path,
                artifact_columns=artifact_columns,
                request_measurements=request_measurements,
            )
        )


def _public_runtime_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value for key, value in meta.items() if not str(key).startswith("_")
    }


def _normalize_rm_dtype_name(value: Any, default: str = "float64") -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        dtype = np.dtype(default)
    if dtype == np.dtype(np.float32):
        return "float32"
    if dtype == np.dtype(np.complex64):
        return "complex64"
    if dtype == np.dtype(np.complex128):
        return "complex128"
    return "float64"


def _display_node_coords_array(values: Any) -> np.ndarray:
    coords = np.asarray(values)
    if np.iscomplexobj(coords):
        coords = np.real(coords)
    if np.issubdtype(np.asarray(coords).dtype, np.floating):
        return np.asarray(coords)
    return np.asarray(coords, dtype=np.float32)


def _display_cell_connectivity_array(values: Any) -> np.ndarray:
    cells = np.asarray(values)
    if np.issubdtype(cells.dtype, np.integer) and cells.dtype == np.dtype(np.int32):
        return cells
    return np.asarray(cells, dtype=np.int32)


def _triangulate_pseudo3d_cells(
    cell_connectivity: Any,
    conductivity: Any,
) -> tuple[np.ndarray, np.ndarray]:
    cells = _display_cell_connectivity_array(cell_connectivity)
    sigma = np.asarray(conductivity).reshape(-1)
    if cells.ndim != 2 or cells.shape[0] == 0:
        raise ValueError("pseudo-3D extrusion requires a non-empty 2D cell array.")
    if cells.shape[1] == 3:
        return cells, sigma
    if cells.shape[1] != 4:
        raise ValueError(
            "pseudo-3D extrusion supports triangular or quadrilateral 2D cells, "
            f"got {cells.shape[1]} vertices per cell."
        )
    triangles = np.empty((cells.shape[0] * 2, 3), dtype=np.int32)
    triangles[0::2, 0] = cells[:, 0]
    triangles[0::2, 1] = cells[:, 1]
    triangles[0::2, 2] = cells[:, 2]
    triangles[1::2, 0] = cells[:, 0]
    triangles[1::2, 1] = cells[:, 2]
    triangles[1::2, 2] = cells[:, 3]
    if sigma.size == cells.shape[0]:
        sigma = np.repeat(sigma, 2)
    return triangles, sigma


def _pseudo3d_display_layers(meta: dict[str, Any]) -> int:
    raw = meta.get("pseudo3d_display_layers", meta.get("pseudo3d_layers", 5))
    try:
        layers = int(raw)
    except (TypeError, ValueError):
        layers = 5
    return max(layers, 2)


def _pseudo3d_display_height(meta: dict[str, Any]) -> float:
    radius = 1.0
    try:
        radius = max(float(meta.get("radius", 1.0)), 1.0e-9)
    except (TypeError, ValueError):
        radius = 1.0
    for key in ("pseudo3d_display_height", "height", "mesh_height"):
        value = meta.get(key)
        if value in (None, ""):
            continue
        try:
            height = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(height) and height > 0.0:
            return height
    return 2.0 * radius


def _pseudo3d_layer_count(meta: dict[str, Any]) -> int:
    raw = meta.get(
        "pseudo3d_layer_count",
        meta.get("pseudo3d_source_n_rings", meta.get("n_rings", 1)),
    )
    try:
        layers = int(raw)
    except (TypeError, ValueError):
        layers = 1
    return max(layers, 1)


def _pseudo3d_source_layer_z_values(
    *,
    meta: dict[str, Any],
    n_layers: int,
    height: float,
    z_center: float,
    dtype: np.dtype,
) -> np.ndarray:
    levels_raw = meta.get("electrode_level_fractions")
    fractions: list[float] | None = None
    if isinstance(levels_raw, str):
        try:
            fractions = [
                float(part)
                for part in levels_raw.replace(";", ",").split(",")
                if part.strip()
            ]
        except ValueError:
            fractions = None
    elif isinstance(levels_raw, (list, tuple, np.ndarray)):
        try:
            fractions = [float(value) for value in levels_raw]
        except (TypeError, ValueError):
            fractions = None
    if fractions is None or len(fractions) != int(n_layers):
        fractions = list(electrode_level_fractions_for_rings(int(n_layers)))
    if len(fractions) != int(n_layers):
        fractions = np.linspace(0.0, 1.0, int(n_layers)).tolist()
    fraction_arr = np.asarray(fractions, dtype=dtype).reshape(-1)
    return np.asarray(z_center + (fraction_arr - 0.5) * height, dtype=dtype)


def _interpolate_layer_values_along_z(
    *,
    source_z: np.ndarray,
    display_z: np.ndarray,
    layer_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(layer_values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("pseudo-3D interpolation requires layer-major values.")
    if values.shape[0] == 1:
        return np.repeat(values, display_z.size, axis=0)

    z_src = np.asarray(source_z, dtype=np.float64).reshape(-1)
    z_dst = np.asarray(display_z, dtype=np.float64).reshape(-1)
    if z_src.size != values.shape[0]:
        raise ValueError("pseudo-3D layer z count does not match layer values.")
    order = np.argsort(z_src)
    z_src = z_src[order]
    values = values[order]
    if float(np.ptp(z_src)) <= 0.0:
        z_src = np.linspace(-0.5, 0.5, values.shape[0], dtype=np.float64)

    out = np.empty((z_dst.size, values.shape[1]), dtype=values.dtype)
    if np.iscomplexobj(values):
        for col in range(values.shape[1]):
            out[:, col] = np.interp(z_dst, z_src, values[:, col].real) + 1j * np.interp(
                z_dst,
                z_src,
                values[:, col].imag,
            )
    else:
        for col in range(values.shape[1]):
            out[:, col] = np.interp(z_dst, z_src, values[:, col])
    return out


def _extrude_2d_result_to_pseudo3d(
    *,
    conductivity: Any,
    node_coords: Any,
    cell_connectivity: Any,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coords_raw = np.asarray(node_coords)
    coord_dtype = (
        np.float32
        if np.issubdtype(coords_raw.dtype, np.floating)
        and coords_raw.dtype.itemsize <= 4
        else np.float64
    )
    coords2 = np.asarray(coords_raw, dtype=coord_dtype)
    if coords2.ndim != 2 or coords2.shape[0] == 0 or coords2.shape[1] < 2:
        raise ValueError("pseudo-3D extrusion requires 2D node coordinates.")
    triangles, sigma2 = _triangulate_pseudo3d_cells(cell_connectivity, conductivity)
    n_nodes = int(coords2.shape[0])
    if np.any(triangles < 0) or np.any(triangles >= n_nodes):
        raise ValueError("pseudo-3D extrusion received out-of-range cell indices.")

    layers = _pseudo3d_display_layers(meta)
    slabs = layers - 1
    height = _pseudo3d_display_height(meta)
    try:
        z_center = float(meta.get("z_center", 0.0) or 0.0)
    except (TypeError, ValueError):
        z_center = 0.0
    z_values = np.linspace(
        z_center - 0.5 * height,
        z_center + 0.5 * height,
        layers,
        dtype=coord_dtype,
    )
    coords3 = np.empty((layers * n_nodes, 3), dtype=coord_dtype)
    for layer_idx, z_value in enumerate(z_values):
        start = layer_idx * n_nodes
        stop = start + n_nodes
        coords3[start:stop, 0] = coords2[:, 0]
        coords3[start:stop, 1] = coords2[:, 1]
        coords3[start:stop, 2] = z_value

    n_tri = int(triangles.shape[0])
    tets = np.empty((slabs * n_tri * 3, 4), dtype=np.int32)
    cursor = 0
    for slab_idx in range(slabs):
        lower = slab_idx * n_nodes
        upper = (slab_idx + 1) * n_nodes
        a0 = triangles[:, 0] + lower
        b0 = triangles[:, 1] + lower
        c0 = triangles[:, 2] + lower
        a1 = triangles[:, 0] + upper
        b1 = triangles[:, 1] + upper
        c1 = triangles[:, 2] + upper
        block = tets[cursor : cursor + n_tri * 3]
        block[0::3, 0] = a0
        block[0::3, 1] = b0
        block[0::3, 2] = c0
        block[0::3, 3] = a1
        block[1::3, 0] = b0
        block[1::3, 1] = b1
        block[1::3, 2] = c1
        block[1::3, 3] = a1
        block[2::3, 0] = b0
        block[2::3, 1] = c0
        block[2::3, 2] = c1
        block[2::3, 3] = a1
        cursor += n_tri * 3

    if sigma2.size == n_tri:
        sigma3 = np.repeat(np.tile(sigma2, slabs), 3)
    elif sigma2.size == n_nodes:
        sigma3 = np.tile(sigma2, layers)
    else:
        sigma3 = np.asarray(sigma2)

    extrusion_meta = {
        "pseudo3d_extruded": True,
        "pseudo3d_display_layers": int(layers),
        "pseudo3d_display_height": float(height),
        "pseudo3d_source_cell_count": int(n_tri),
        "pseudo3d_source_node_count": int(n_nodes),
        "pseudo3d_tetra_cell_count": int(tets.shape[0]),
        "pseudo3d_node_count": int(coords3.shape[0]),
    }
    return sigma3, coords3, tets, extrusion_meta


def _extrude_layered_2d_results_to_pseudo3d(
    *,
    conductivity_layers: Any,
    node_coords: Any,
    cell_connectivity: Any,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coords_raw = np.asarray(node_coords)
    coord_dtype = (
        np.float32
        if np.issubdtype(coords_raw.dtype, np.floating)
        and coords_raw.dtype.itemsize <= 4
        else np.float64
    )
    coords2 = np.asarray(coords_raw, dtype=coord_dtype)
    if coords2.ndim != 2 or coords2.shape[0] == 0 or coords2.shape[1] < 2:
        raise ValueError("pseudo-3D layered extrusion requires 2D node coordinates.")

    cells = _display_cell_connectivity_array(cell_connectivity)
    if cells.ndim != 2 or cells.shape[0] == 0:
        raise ValueError(
            "pseudo-3D layered extrusion requires a non-empty 2D cell array."
        )
    if cells.shape[1] == 3:
        triangles = cells
    elif cells.shape[1] == 4:
        triangles = np.empty((cells.shape[0] * 2, 3), dtype=np.int32)
        triangles[0::2, 0] = cells[:, 0]
        triangles[0::2, 1] = cells[:, 1]
        triangles[0::2, 2] = cells[:, 2]
        triangles[1::2, 0] = cells[:, 0]
        triangles[1::2, 1] = cells[:, 2]
        triangles[1::2, 2] = cells[:, 3]
    else:
        raise ValueError(
            "pseudo-3D layered extrusion supports triangular or quadrilateral 2D "
            f"cells, got {cells.shape[1]} vertices per cell."
        )

    n_nodes = int(coords2.shape[0])
    if np.any(triangles < 0) or np.any(triangles >= n_nodes):
        raise ValueError(
            "pseudo-3D layered extrusion received out-of-range cell indices."
        )

    raw_layers = np.asarray(conductivity_layers)
    if raw_layers.ndim == 1:
        raw_layers = raw_layers.reshape(1, -1)
    elif raw_layers.ndim > 2:
        raw_layers = raw_layers.reshape(raw_layers.shape[0], -1)
    if raw_layers.ndim != 2 or raw_layers.shape[0] < 2:
        raise ValueError("pseudo-3D layered extrusion requires at least two 2D layers.")

    n_source_layers = int(raw_layers.shape[0])
    n_cells = int(cells.shape[0])
    n_tri = int(triangles.shape[0])
    value_count = int(raw_layers.shape[1])
    nodal_values = False
    if value_count == n_nodes:
        layer_values = raw_layers
        nodal_values = True
    elif value_count == n_tri:
        layer_values = raw_layers
    elif cells.shape[1] == 4 and value_count == n_cells:
        layer_values = np.repeat(raw_layers, 2, axis=1)
    elif cells.shape[1] == 3 and value_count == n_cells:
        layer_values = raw_layers
    else:
        raise ValueError(
            "pseudo-3D layered conductivity size mismatch: expected per-node, "
            f"per-cell, or per-triangle values; got {value_count}."
        )

    display_layers = max(_pseudo3d_display_layers(meta), n_source_layers)
    slabs = display_layers - 1
    height = _pseudo3d_display_height(meta)
    try:
        z_center = float(meta.get("z_center", 0.0) or 0.0)
    except (TypeError, ValueError):
        z_center = 0.0
    z_values = np.linspace(
        z_center - 0.5 * height,
        z_center + 0.5 * height,
        display_layers,
        dtype=coord_dtype,
    )
    source_z = _pseudo3d_source_layer_z_values(
        meta=meta,
        n_layers=n_source_layers,
        height=height,
        z_center=z_center,
        dtype=np.dtype(coord_dtype),
    )

    coords3 = np.empty((display_layers * n_nodes, 3), dtype=coord_dtype)
    for layer_idx, z_value in enumerate(z_values):
        start = layer_idx * n_nodes
        stop = start + n_nodes
        coords3[start:stop, 0] = coords2[:, 0]
        coords3[start:stop, 1] = coords2[:, 1]
        coords3[start:stop, 2] = z_value

    tets = np.empty((slabs * n_tri * 3, 4), dtype=np.int32)
    cursor = 0
    for slab_idx in range(slabs):
        lower = slab_idx * n_nodes
        upper = (slab_idx + 1) * n_nodes
        a0 = triangles[:, 0] + lower
        b0 = triangles[:, 1] + lower
        c0 = triangles[:, 2] + lower
        a1 = triangles[:, 0] + upper
        b1 = triangles[:, 1] + upper
        c1 = triangles[:, 2] + upper
        block = tets[cursor : cursor + n_tri * 3]
        block[0::3, 0] = a0
        block[0::3, 1] = b0
        block[0::3, 2] = c0
        block[0::3, 3] = a1
        block[1::3, 0] = b0
        block[1::3, 1] = b1
        block[1::3, 2] = c1
        block[1::3, 3] = a1
        block[2::3, 0] = b0
        block[2::3, 1] = c0
        block[2::3, 2] = c1
        block[2::3, 3] = a1
        cursor += n_tri * 3

    sigma_at_display_layers = _interpolate_layer_values_along_z(
        source_z=source_z,
        display_z=z_values,
        layer_values=np.asarray(layer_values),
    )
    if nodal_values:
        sigma3 = sigma_at_display_layers.reshape(-1)
    else:
        slab_sigma = 0.5 * (
            sigma_at_display_layers[:-1, :] + sigma_at_display_layers[1:, :]
        )
        sigma3 = np.repeat(slab_sigma.reshape(-1), 3)

    extrusion_meta = {
        "pseudo3d_extruded": True,
        "pseudo3d_layered_extruded": True,
        "pseudo3d_interpolation": "linear_z_between_2d_layers",
        "pseudo3d_display_layers": int(display_layers),
        "pseudo3d_display_height": float(height),
        "pseudo3d_source_layer_count": int(n_source_layers),
        "pseudo3d_source_layer_z": [float(value) for value in source_z],
        "pseudo3d_display_layer_z": [float(value) for value in z_values],
        "pseudo3d_source_cell_count": int(n_tri),
        "pseudo3d_source_node_count": int(n_nodes),
        "pseudo3d_tetra_cell_count": int(tets.shape[0]),
        "pseudo3d_node_count": int(coords3.shape[0]),
    }
    return sigma3, coords3, tets, extrusion_meta


def _maybe_apply_pseudo3d_result(result: ReconstructionResult) -> ReconstructionResult:
    meta = dict(result.metadata or {})
    if not _flag_enabled(meta.get("pseudo3d_output", False)):
        return result
    if result.error_msg:
        return result
    try:
        if _flag_enabled(meta.get("pseudo3d_layered_output", False)):
            sigma3, coords3, cells3, extrusion_meta = (
                _extrude_layered_2d_results_to_pseudo3d(
                    conductivity_layers=result.conductivity,
                    node_coords=result.node_coords,
                    cell_connectivity=result.cell_connectivity,
                    meta=meta,
                )
            )
        else:
            sigma3, coords3, cells3, extrusion_meta = _extrude_2d_result_to_pseudo3d(
                conductivity=result.conductivity,
                node_coords=result.node_coords,
                cell_connectivity=result.cell_connectivity,
                meta=meta,
            )
    except Exception as exc:
        updated_meta = dict(meta)
        updated_meta["pseudo3d_extrusion_error"] = str(exc)
        return ReconstructionResult(
            conductivity=result.conductivity,
            node_coords=result.node_coords,
            cell_connectivity=result.cell_connectivity,
            measured=result.measured,
            simulated=result.simulated,
            error_msg=result.error_msg,
            metadata=updated_meta,
        )
    meta.update(extrusion_meta)
    meta["mesh_dimension"] = 3
    meta["conductivity_display_mode"] = str(
        meta.get("conductivity_display_mode", "absolute_sigma")
    )
    return ReconstructionResult(
        conductivity=sigma3,
        node_coords=coords3,
        cell_connectivity=cells3,
        measured=result.measured,
        simulated=result.simulated,
        error_msg=result.error_msg,
        metadata=meta,
    )


def _pseudo3d_layer_measurement_indices(
    meta: dict[str, Any],
    *,
    expected_measurements: int | None = None,
) -> list[np.ndarray]:
    source_n_elec = max(
        int(meta.get("pseudo3d_source_n_elec", meta.get("n_elec", 16))), 1
    )
    source_n_rings = max(
        int(meta.get("pseudo3d_source_n_rings", meta.get("n_rings", 1))),
        1,
    )
    if source_n_rings < 2:
        raise ValueError(
            "pseudo-3D layered reconstruction requires at least two rings."
        )

    source_geometry = dict(meta.get("pseudo3d_source_geometry") or {})
    from pyeidors.data.structures import PatternConfig
    from pyeidors.electrodes.patterns import StimMeasPatternManager

    pattern_config = PatternConfig(
        n_elec=source_n_elec,
        n_rings=source_n_rings,
        stim_pattern=meta.get("stim_pattern", "{ad}"),
        meas_pattern=meta.get("meas_pattern", "{ad}"),
        electrode_layout=str(
            source_geometry.get(
                "electrode_layout",
                meta.get("electrode_layout", "ring_major"),
            )
        ),
        measurement_protocol=str(
            source_geometry.get(
                "measurement_protocol",
                meta.get("measurement_protocol", "eidors_full_3d"),
            )
        ),
        custom_stim_matrix=meta.get("custom_stim_matrix"),
        custom_meas_matrices=meta.get("custom_meas_matrices"),
        drive_mode=str(meta.get("drive_mode", "line_current_density")),
        drive_value=float(meta.get("drive_value", 1.0) or 1.0),
        geometry_scale_to_m=float(meta.get("geometry_scale_to_m", 1.0) or 1.0),
        electrode_length_m_override=meta.get("electrode_length_m_override"),
        use_meas_current=_flag_enabled(meta.get("use_meas_current", False)),
        use_meas_current_next=int(meta.get("use_meas_current_next", 0) or 0),
        rotate_meas=_flag_enabled(meta.get("rotate_meas", True)),
        stim_direction=str(meta.get("stim_direction", "ccw")),
        meas_direction=str(meta.get("meas_direction", "ccw")),
        stim_first_positive=_flag_enabled(meta.get("stim_first_positive", False)),
    )
    manager = StimMeasPatternManager(pattern_config)
    total = int(manager.n_meas_total)
    if expected_measurements is not None and int(expected_measurements) != total:
        raise ValueError(
            "pseudo-3D source measurement count mismatch: "
            f"expected {total}, got {int(expected_measurements)}."
        )

    layer_indices: list[list[int]] = [[] for _ in range(source_n_rings)]
    for stim_idx, (start_idx, stim_row, meas_mat) in enumerate(
        zip(manager.meas_start_indices, manager.stim_matrix, manager.meas_matrices)
    ):
        del stim_idx
        stim_electrodes = np.flatnonzero(np.abs(stim_row) > 0.0)
        stim_rings = {int(electrode // source_n_elec) for electrode in stim_electrodes}
        if len(stim_rings) != 1:
            continue
        ring = next(iter(stim_rings))
        if ring < 0 or ring >= source_n_rings:
            continue
        for row_idx, meas_row in enumerate(np.asarray(meas_mat)):
            meas_electrodes = np.flatnonzero(np.abs(meas_row) > 0.0)
            meas_rings = {
                int(electrode // source_n_elec) for electrode in meas_electrodes
            }
            if meas_rings == {ring}:
                layer_indices[ring].append(int(start_idx) + int(row_idx))

    arrays = [np.asarray(indices, dtype=np.int64) for indices in layer_indices]
    missing = [idx + 1 for idx, indices in enumerate(arrays) if indices.size == 0]
    if missing:
        raise ValueError(
            "pseudo-3D layered reconstruction could not find same-ring "
            f"measurements for layer(s): {missing}."
        )
    return arrays


def _subset_frame_data_for_indices(
    frame: FrameData,
    indices: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> FrameData:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    real = np.asarray(frame.real)
    imag = np.asarray(frame.imag)
    if idx.size == 0:
        raise ValueError("pseudo-3D layer measurement subset is empty.")
    if real.size <= int(np.max(idx)) or imag.size <= int(np.max(idx)):
        raise ValueError("pseudo-3D layer measurement index exceeds frame length.")
    return FrameData(
        real=np.asarray(real[idx]).copy(),
        imag=np.asarray(imag[idx]).copy(),
        timestamp=frame.timestamp,
        frame_index=frame.frame_index,
        metadata={**dict(frame.metadata or {}), **metadata},
    )


def _pseudo3d_layer_request_metadata(
    meta: dict[str, Any],
    *,
    layer_index: int,
    indices: np.ndarray,
) -> dict[str, Any]:
    source_n_elec = max(
        int(meta.get("pseudo3d_source_n_elec", meta.get("n_elec", 16))), 1
    )
    layer_meta = dict(meta)
    for key in _RM_ARTIFACT_META_KEYS:
        layer_meta.pop(key, None)
    layer_meta.update(
        {
            "pseudo3d_output": False,
            "pseudo3d_layered_output": False,
            "pseudo3d_parent_output": True,
            "pseudo3d_layer_index": int(layer_index),
            "pseudo3d_layer_source_ring": int(layer_index),
            "pseudo3d_layer_measurement_indices": [
                int(value) for value in np.asarray(indices).reshape(-1)
            ],
            "mesh_dimension": 2,
            "n_elec": int(source_n_elec),
            "n_rings": 1,
            "electrode_layout": "ring_major",
            "measurement_protocol": "eidors_full_3d",
            "drive_mode": "line_current_density",
            "drive_value": 1.0,
            "petsc_device": "cpu",
            "device": "cpu",
            "rm_device": "cpu",
            "forward_backend": "dolfinx",
            "forward_solver_preset": "auto",
            "forward_mat_solve": "off",
            "acceleration_profile": "default",
            "reconstruction_runtime": "single_step_cached",
        }
    )
    return layer_meta


def _run_pseudo3d_layered_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    meta = dict(req.metadata or {})
    full_count = _request_measurement_count(req)
    layer_indices = _pseudo3d_layer_measurement_indices(
        meta,
        expected_measurements=full_count if full_count > 0 else None,
    )
    layer_results: list[ReconstructionResult] = []
    layer_conductivities: list[np.ndarray] = []
    base_coords: np.ndarray | None = None
    base_cells: np.ndarray | None = None
    source_n_elec = max(
        int(meta.get("pseudo3d_source_n_elec", meta.get("n_elec", 16))), 1
    )

    for layer_index, indices in enumerate(layer_indices):
        emit(
            "Running pseudo-3D layer "
            f"{layer_index + 1}/{len(layer_indices)} 2D reconstruction..."
        )
        layer_meta = _pseudo3d_layer_request_metadata(
            meta,
            layer_index=layer_index,
            indices=indices,
        )
        frame_meta = {
            "pseudo3d_layer_index": int(layer_index),
            "pseudo3d_parent_measurement_count": int(full_count),
        }
        layer_req = ReconstructionRequest(
            reference_frame=_subset_frame_data_for_indices(
                req.reference_frame,
                indices,
                metadata=frame_meta,
            ),
            target_frame=_subset_frame_data_for_indices(
                req.target_frame,
                indices,
                metadata=frame_meta,
            ),
            use_part=req.use_part,
            method=req.method,
            regularization_alpha=req.regularization_alpha,
            max_iterations=req.max_iterations,
            mesh_dimension=2,
            mesh_refinement=req.mesh_refinement,
            metadata=layer_meta,
        )
        layer_result = _run_single_step_cached_request(
            layer_req,
            progress_cb=progress_cb,
        )
        if layer_result.error_msg:
            error_meta = dict(meta)
            error_meta.update(
                {
                    "pseudo3d_layered_execution_error_layer": int(layer_index),
                    "pseudo3d_layered_execution_error": layer_result.error_msg,
                }
            )
            return ReconstructionResult(
                conductivity=np.array([]),
                node_coords=np.array([]),
                cell_connectivity=np.array([]),
                error_msg=f"Pseudo-3D layer {layer_index + 1} failed: "
                f"{layer_result.error_msg}",
                metadata=error_meta,
            )
        layer_results.append(layer_result)
        layer_conductivities.append(np.asarray(layer_result.conductivity).reshape(-1))
        coords = np.asarray(layer_result.node_coords)
        cells = np.asarray(layer_result.cell_connectivity)
        if base_coords is None:
            base_coords = coords
            base_cells = cells
        elif (
            base_coords.shape != coords.shape
            or base_cells is None
            or base_cells.shape != cells.shape
            or not np.array_equal(base_cells, cells)
            or not np.allclose(base_coords, coords, equal_nan=True)
        ):
            raise ValueError("pseudo-3D layer inverse meshes do not match.")

    if base_coords is None or base_cells is None or not layer_conductivities:
        raise ValueError("pseudo-3D layered reconstruction produced no layer results.")

    conductivity_lengths = {values.size for values in layer_conductivities}
    if len(conductivity_lengths) != 1:
        raise ValueError("pseudo-3D layer conductivity vector lengths do not match.")
    conductivity_layers = np.stack(layer_conductivities, axis=0)
    ref_vec = np.asarray(req.reference_frame.to_measurement_vector(req.use_part))
    tgt_vec = np.asarray(req.target_frame.to_measurement_vector(req.use_part))
    measured_diff = build_difference_vector(
        tgt_vec,
        ref_vec,
        mode=str(meta.get("difference_mode", "raw")),
        orientation=str(meta.get("difference_orientation", "target_minus_reference")),
    )
    combined_meta = dict(meta)
    combined_meta.update(
        {
            "pseudo3d_layered_executed": True,
            "pseudo3d_layer_count": int(len(layer_indices)),
            "pseudo3d_layer_n_elec": int(source_n_elec),
            "pseudo3d_layer_measurement_counts": [
                int(indices.size) for indices in layer_indices
            ],
            "pseudo3d_layer_measurement_indices": [
                [int(value) for value in indices] for indices in layer_indices
            ],
            "pseudo3d_layer_result_count": int(len(layer_results)),
            "pseudo3d_voltage_fit_scope": "full_source_measured_layer_local_inverse",
        }
    )
    emit("Interpolating pseudo-3D layer reconstructions...")
    return _maybe_apply_pseudo3d_result(
        ReconstructionResult(
            conductivity=conductivity_layers,
            node_coords=base_coords,
            cell_connectivity=base_cells,
            measured=measured_diff,
            simulated=None,
            metadata=combined_meta,
        )
    )


def _rm_dtype_name_from_meta(meta: dict[str, Any]) -> str:
    for key in (
        "rm_dtype",
        "rm_matmul_dtype",
        "compute_dtype",
        "compute_precision",
        "precision",
    ):
        value = meta.get(key)
        if value not in (None, ""):
            return _normalize_rm_dtype_name(value)
    return "float64"


def _parse_bytes_limit(value: Any, *, default: int) -> int:
    if value is None or value == "" or isinstance(value, bool):
        return int(default)
    if isinstance(value, (int, float)):
        return int(max(0.0, float(value)))
    text = str(value).strip().lower()
    if not text:
        return int(default)
    units = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    number = text
    unit = ""
    for suffix in sorted(units, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            unit = suffix
            break
    try:
        return int(max(0.0, float(number)) * units[unit])
    except (KeyError, ValueError):
        return int(default)


def _runtime_bytes_limit(
    meta: dict[str, Any],
    *,
    keys: tuple[str, ...],
    env_key: str,
    default: int,
) -> int:
    for key in keys:
        if key in meta:
            return _parse_bytes_limit(meta.get(key), default=default)
    return _parse_bytes_limit(os.environ.get(env_key), default=default)


def _rm_fit_jacobian_max_bytes(meta: dict[str, Any]) -> int:
    return _runtime_bytes_limit(
        meta,
        keys=("rm_fit_jacobian_max_bytes", "rm_fit_jacobian_cache_max_bytes"),
        env_key="EIT_APP_RM_FIT_JACOBIAN_MAX_BYTES",
        default=_RM_FIT_JACOBIAN_CACHE_MAX_BYTES,
    )


def _rm_artifact_process_cache_max_bytes(meta: dict[str, Any]) -> int:
    return _runtime_bytes_limit(
        meta,
        keys=("rm_artifact_process_cache_max_bytes", "rm_artifact_cache_max_bytes"),
        env_key="EIT_APP_RM_ARTIFACT_PROCESS_CACHE_MAX_BYTES",
        default=_RM_ARTIFACT_CACHE_MAX_BYTES,
    )


def _rm_device_resident_max_bytes(meta: dict[str, Any]) -> int:
    return _runtime_bytes_limit(
        meta,
        keys=(
            "rm_device_resident_max_bytes",
            "rm_cuda_resident_max_bytes",
            "rm_gpu_resident_max_bytes",
        ),
        env_key="EIT_APP_RM_DEVICE_RESIDENT_MAX_BYTES",
        default=_RM_ARTIFACT_CACHE_MAX_BYTES,
    )


def _reconstruction_system_cache_max_bytes(meta: dict[str, Any] | None = None) -> int:
    return _runtime_bytes_limit(
        meta or {},
        keys=(
            "reconstruction_system_cache_max_bytes",
            "system_cache_max_bytes",
        ),
        env_key="EIT_APP_RECONSTRUCTION_SYSTEM_CACHE_MAX_BYTES",
        default=_SYSTEM_CACHE_MAX_BYTES,
    )


def _single_step_context_cache_max_bytes(meta: dict[str, Any] | None = None) -> int:
    return _runtime_bytes_limit(
        meta or {},
        keys=(
            "single_step_context_cache_max_bytes",
            "fast_context_cache_max_bytes",
            "reconstruction_fast_context_cache_max_bytes",
        ),
        env_key="EIT_APP_SINGLE_STEP_CONTEXT_CACHE_MAX_BYTES",
        default=_FAST_CONTEXT_CACHE_MAX_BYTES,
    )


def _array_like_nbytes(value: Any) -> int | None:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            return None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        count = 1
        for dim in tuple(shape):
            count *= int(dim)
        return int(count * np.dtype(dtype).itemsize)
    except (TypeError, ValueError):
        return None


def _array_like_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in tuple(shape))
    except (TypeError, ValueError):
        return None


def _array_payload_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate resident bytes for array-like payloads without deep object walks."""

    if value is None:
        return 0
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    nbytes = _array_like_nbytes(value)
    if nbytes is not None:
        return int(max(0, nbytes))
    if isinstance(value, dict):
        return sum(_array_payload_nbytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_array_payload_nbytes(v, seen) for v in value)

    total = 0
    for attr in ("data", "indices", "indptr"):
        part = getattr(value, attr, None)
        if part is not None and part is not value:
            total += _array_payload_nbytes(part, seen)

    for attr in (
        "grad_u_all",
        "adjoint_gradients",
        "_adjoint_blocks",
        "cell_areas",
        "sigma_values",
        "u_all",
        "meas_matrices",
        "projection_weights",
        "reg_diag",
        "inv_reg_diag",
        "precond_diag",
        "lu",
        "piv",
    ):
        part = getattr(value, attr, None)
        if part is not None:
            total += _array_payload_nbytes(part, seen)
    return int(total)


def _mesh_payload_nbytes(mesh: Any, seen: set[int] | None = None) -> int:
    if mesh is None:
        return 0
    if seen is None:
        seen = set()
    mesh_id = id(mesh)
    if mesh_id in seen:
        return 0
    seen.add(mesh_id)

    total = 0
    coordinates = getattr(mesh, "coordinates", None)
    if callable(coordinates):
        with contextlib.suppress(Exception):
            total += _array_payload_nbytes(coordinates(), seen)
    cells = getattr(mesh, "cells", None)
    if callable(cells):
        with contextlib.suppress(Exception):
            total += _array_payload_nbytes(cells(), seen)

    geometry = getattr(mesh, "geometry", None)
    total += _array_payload_nbytes(getattr(geometry, "x", None), seen)
    topology = getattr(mesh, "topology", None)
    total += _array_payload_nbytes(getattr(topology, "connectivity", None), seen)

    for attr in (
        "cell_tags",
        "facet_tags",
        "electrode_vertices",
        "_electrode_vertices",
    ):
        total += _array_payload_nbytes(getattr(mesh, attr, None), seen)
    return int(total)


def _pattern_manager_payload_nbytes(pattern_manager: Any, seen: set[int]) -> int:
    total = 0
    for attr in (
        "stim_matrix",
        "stim_patterns",
        "meas_matrix",
        "meas_matrices",
        "n_meas_per_stim",
        "inj_electrodes",
        "inj_weights",
        "_electrode_lengths_m",
    ):
        total += _array_payload_nbytes(getattr(pattern_manager, attr, None), seen)
    return int(total)


def _forward_model_payload_nbytes(fwd_model: Any, seen: set[int] | None = None) -> int:
    if fwd_model is None:
        return 0
    if seen is None:
        seen = set()
    model_id = id(fwd_model)
    if model_id in seen:
        return 0
    seen.add(model_id)

    total = 0
    total += _mesh_payload_nbytes(getattr(fwd_model, "mesh", None), seen)
    total += _mesh_payload_nbytes(getattr(fwd_model, "eit_mesh", None), seen)
    total += _pattern_manager_payload_nbytes(
        getattr(fwd_model, "pattern_manager", None), seen
    )
    for attr in (
        "z",
        "electrode_lengths_m",
        "electrode_areas_m2",
        "electrode_boundary_measures",
        "cell_volumes",
        "cell_areas",
    ):
        total += _array_payload_nbytes(getattr(fwd_model, attr, None), seen)
    return int(total)


def _reconstruction_system_cache_entry_bytes(system: Any) -> int:
    seen: set[int] = set()
    total = 0
    total += _mesh_payload_nbytes(getattr(system, "mesh", None), seen)
    total += _forward_model_payload_nbytes(getattr(system, "fwd_model", None), seen)
    for attr in ("reconstructor", "absolute_reconstructor", "difference_reconstructor"):
        total += _array_payload_nbytes(getattr(system, attr, None), seen)
    return int(total)


def _fast_context_cache_entry_bytes(ctx: Any) -> int:
    seen: set[int] = set()
    if not isinstance(ctx, dict):
        return _array_payload_nbytes(ctx, seen)
    light_ctx = {
        key: value
        for key, value in ctx.items()
        if key not in {"mesh", "fwd_model", "img_bg"}
    }
    total = _array_payload_nbytes(light_ctx, seen)
    total += _mesh_payload_nbytes(ctx.get("mesh"), seen)
    total += _forward_model_payload_nbytes(ctx.get("fwd_model"), seen)
    return int(total)


def _ordered_cache_total_bytes(
    cache: OrderedDict[Any, Any],
    sizes: dict[Any, int],
) -> int:
    return sum(int(max(0, sizes.get(key, 0))) for key in cache)


def _put_bounded_ordered_cache(
    cache: OrderedDict[Any, Any],
    sizes: dict[Any, int],
    key: Any,
    value: Any,
    *,
    entry_bytes: int,
    max_items: int,
    max_bytes: int,
) -> bool:
    cache.pop(key, None)
    sizes.pop(key, None)
    entry_bytes = int(max(0, entry_bytes))
    max_bytes = int(max(0, max_bytes))
    if max_bytes <= 0 or entry_bytes > max_bytes:
        return False
    cache[key] = value
    cache.move_to_end(key)
    sizes[key] = entry_bytes
    while len(cache) > int(max_items):
        old_key, _ = cache.popitem(last=False)
        sizes.pop(old_key, None)
    while _ordered_cache_total_bytes(cache, sizes) > max_bytes:
        if not cache:
            break
        oldest_key = next(iter(cache))
        if oldest_key == key and len(cache) == 1:
            break
        old_key, _ = cache.popitem(last=False)
        sizes.pop(old_key, None)
    return key in cache


def _rm_artifact_matrix_shape(artifact: dict[str, Any]) -> tuple[int, int]:
    for key in ("rm", "rm_lazy_dataset"):
        value = artifact.get(key)
        if value is None:
            continue
        shape = _array_like_shape(value)
        if shape is not None:
            if len(shape) != 2 or 0 in shape:
                raise ValueError(
                    f"RM artifact matrix must be non-empty 2D, got {shape}."
                )
            return (int(shape[0]), int(shape[1]))
    raw_shape = _parse_int_shape((artifact.get("metadata") or {}).get("rm_shape"))
    if len(raw_shape) == 2 and 0 not in raw_shape:
        return (int(raw_shape[0]), int(raw_shape[1]))
    raise ValueError("RM artifact is missing matrix shape metadata.")


def _rm_artifact_matrix_nbytes(artifact: dict[str, Any]) -> int:
    for key in ("rm", "rm_lazy_dataset"):
        value = artifact.get(key)
        if value is None:
            continue
        nbytes = _array_like_nbytes(value)
        if nbytes is not None:
            return int(nbytes)
    rows, cols = _rm_artifact_matrix_shape(artifact)
    dtype = np.dtype(artifact.get("rm_dtype", "float64"))
    return int(rows * cols * dtype.itemsize)


@dataclass(frozen=True)
class _RMFitJacobianReadResult:
    array: np.ndarray | None
    status: str
    nbytes: int | None = None


def _rm_fit_jacobian_cache_key(path: Path, signature: Any) -> str:
    signature_text = str(signature or "").strip()
    if signature_text:
        return signature_text
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _fit_jacobian_array(
    value: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ):
        arr = arr.astype(np.float64, copy=False)
    if arr.ndim != 2 or 0 in arr.shape or not all_finite_values(arr):
        return None
    if expected_shape is not None and tuple(int(v) for v in arr.shape) != tuple(
        int(v) for v in expected_shape
    ):
        return None
    return np.ascontiguousarray(arr)


def _real_sigma_update_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    dtype = (
        np.float32
        if np.issubdtype(np.asarray(arr).dtype, np.floating)
        and np.asarray(arr).dtype.itemsize <= 4
        else np.float64
    )
    return np.ascontiguousarray(np.asarray(arr, dtype=dtype).reshape(-1))


def _put_rm_fit_jacobian_cache(key: str, jacobian: np.ndarray) -> str:
    arr = _fit_jacobian_array(jacobian)
    if arr is None:
        return "invalid"
    if int(arr.nbytes) > _RM_FIT_JACOBIAN_CACHE_MAX_BYTES:
        return "too_large"
    with _RM_FIT_JACOBIAN_CACHE_LOCK:
        stored = _put_bounded_ordered_cache(
            _RM_FIT_JACOBIAN_CACHE,
            _RM_FIT_JACOBIAN_CACHE_SIZES,
            key,
            arr,
            entry_bytes=int(arr.nbytes),
            max_items=_RM_FIT_JACOBIAN_CACHE_MAX_ITEMS,
            max_bytes=_RM_FIT_JACOBIAN_CACHE_MAX_BYTES,
        )
    return "stored" if stored else "too_large"


def _get_rm_fit_jacobian_cache(
    key: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    with _RM_FIT_JACOBIAN_CACHE_LOCK:
        cached = _RM_FIT_JACOBIAN_CACHE.get(key)
        if cached is None:
            return None
        arr = _fit_jacobian_array(cached, expected_shape=expected_shape)
        if arr is None:
            _RM_FIT_JACOBIAN_CACHE.pop(key, None)
            _RM_FIT_JACOBIAN_CACHE_SIZES.pop(key, None)
            return None
        _RM_FIT_JACOBIAN_CACHE.move_to_end(key)
        return arr


def _read_rm_artifact_fit_jacobian(
    path: Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    return _read_rm_artifact_fit_jacobian_result(
        path,
        expected_shape=expected_shape,
        max_bytes=_RM_FIT_JACOBIAN_CACHE_MAX_BYTES,
    ).array


def _read_rm_artifact_fit_jacobian_result(
    path: Path,
    *,
    expected_shape: tuple[int, int] | None = None,
    max_bytes: int,
) -> _RMFitJacobianReadResult:
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        return _RMFitJacobianReadResult(None, "unsupported_format")
    try:
        from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

        artifact = read_hdf5_artifact(
            path,
            lazy=True,
            verify_checksums=False,
        )
        raw = artifact.arrays.get("jacobian")
        if raw is None:
            raw = artifact.arrays.get("fit_jacobian")
        if raw is None:
            metadata = dict(artifact.metadata or {})
            if _fit_jacobian_metadata_declares_too_large(metadata):
                return _RMFitJacobianReadResult(
                    None,
                    "too_large",
                    _fit_jacobian_nbytes_from_metadata(metadata),
                )
            return _RMFitJacobianReadResult(None, "missing")
        raw_shape = _array_like_shape(raw)
        if expected_shape is not None and raw_shape is not None:
            if tuple(raw_shape) != tuple(int(v) for v in expected_shape):
                return _RMFitJacobianReadResult(
                    None,
                    "shape_mismatch",
                    _array_like_nbytes(raw),
                )
        raw_nbytes = _array_like_nbytes(raw)
        if raw_nbytes is not None and raw_nbytes > int(max(0, max_bytes)):
            return _RMFitJacobianReadResult(None, "too_large", raw_nbytes)
        arr = _fit_jacobian_array(raw, expected_shape=expected_shape)
        if arr is None:
            return _RMFitJacobianReadResult(None, "invalid", raw_nbytes)
        return _RMFitJacobianReadResult(arr, "hit", int(arr.nbytes))
    except Exception as exc:
        log.debug("Could not restore RM fit Jacobian from %s: %s", path, exc)
        return _RMFitJacobianReadResult(None, "error")


def _fit_jacobian_metadata_declares_too_large(metadata: dict[str, Any]) -> bool:
    persisted = metadata.get("fit_jacobian_persisted")
    if isinstance(persisted, bool):
        persisted_false = not persisted
    else:
        persisted_false = str(persisted).strip().lower() in {"0", "false", "no", "off"}
    reason = (
        str(
            metadata.get("fit_jacobian_persist_skip_reason")
            or metadata.get("rm_fit_jacobian_persist_skip_reason")
            or ""
        )
        .strip()
        .lower()
    )
    return bool(persisted_false and reason in {"too_large", "entry_too_large"})


def _fit_jacobian_nbytes_from_metadata(metadata: dict[str, Any]) -> int | None:
    for key in ("rm_fit_jacobian_bytes", "fit_jacobian_bytes"):
        try:
            return int(metadata[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _stash_rm_fit_jacobian(
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    path: Path,
    signature: Any,
    jacobian: Any,
    status_prefix: str,
) -> np.ndarray | None:
    arr = _fit_jacobian_array(jacobian)
    if arr is None:
        runtime.meta["rm_fit_jacobian_cache_status"] = f"{status_prefix}_invalid"
        return None
    runtime.meta["_inmem_jacobian"] = arr
    cache_key = _rm_fit_jacobian_cache_key(path, signature)
    cache_status = _put_rm_fit_jacobian_cache(cache_key, arr)
    runtime.meta["rm_fit_jacobian_cache_status"] = f"{status_prefix}_{cache_status}"
    return arr


def _restore_rm_fit_jacobian(
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    path: Path,
    signature: Any,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    current = runtime.meta.get("_inmem_jacobian")
    if current is not None:
        arr = _fit_jacobian_array(current, expected_shape=expected_shape)
        if arr is not None:
            runtime.meta["_inmem_jacobian"] = arr
            runtime.meta.setdefault("rm_fit_jacobian_cache_status", "runtime_hit")
            return arr
        runtime.meta.pop("_inmem_jacobian", None)

    cache_key = _rm_fit_jacobian_cache_key(path, signature)
    cached = _get_rm_fit_jacobian_cache(cache_key, expected_shape=expected_shape)
    if cached is not None:
        runtime.meta["_inmem_jacobian"] = cached
        runtime.meta["rm_fit_jacobian_cache_status"] = "process_hit"
        return cached

    max_bytes = _rm_fit_jacobian_max_bytes(runtime.meta)
    runtime.meta["rm_fit_jacobian_max_bytes"] = int(max_bytes)
    read_result = _read_rm_artifact_fit_jacobian_result(
        path,
        expected_shape=expected_shape,
        max_bytes=max_bytes,
    )
    if read_result.nbytes is not None:
        runtime.meta["rm_fit_jacobian_bytes"] = int(read_result.nbytes)
    artifact_jacobian = read_result.array
    if artifact_jacobian is not None:
        runtime.meta["_inmem_jacobian"] = artifact_jacobian
        cache_status = _put_rm_fit_jacobian_cache(cache_key, artifact_jacobian)
        runtime.meta["rm_fit_jacobian_cache_status"] = f"artifact_hit_{cache_status}"
        return artifact_jacobian

    if read_result.status == "too_large":
        runtime.meta["rm_fit_jacobian_cache_status"] = "artifact_too_large"
        runtime.meta["rm_fit_jacobian_available_but_skipped"] = True
    elif read_result.status == "missing":
        runtime.meta["rm_fit_jacobian_cache_status"] = "miss"
    else:
        runtime.meta["rm_fit_jacobian_cache_status"] = f"artifact_{read_result.status}"
    return None


def _contact_impedance_array(value: Any, default: complex = 0.01 + 0.0j) -> np.ndarray:
    if value is None or value == "":
        return np.asarray([default], dtype=np.complex128)
    if isinstance(value, str):
        try:
            parsed = parse_complex_scalar_list(value)
        except ValueError:
            try:
                parsed = [parse_complex_scalar(value)]
            except ValueError:
                parsed = [default]
        arr = np.asarray(parsed, dtype=np.complex128).reshape(-1)
    else:
        try:
            arr = np.asarray(value, dtype=np.complex128).reshape(-1)
        except (TypeError, ValueError):
            arr = np.asarray([default], dtype=np.complex128)
    if arr.size == 0:
        arr = np.asarray([default], dtype=np.complex128)
    if not np.iscomplexobj(arr) or not has_nonzero_imaginary(
        arr, tol=_COMPLEX_ZERO_TOL
    ):
        return np.asarray(np.real(arr), dtype=float).reshape(-1)
    return arr


def _contact_impedance_scalar(value: Any, default: float = 0.01) -> float | complex:
    arr = _contact_impedance_array(value, complex(default, 0.0)).reshape(-1)
    if arr.size == 0:
        return float(default)
    scalar = arr[0]
    if np.iscomplexobj(arr) and abs(complex(scalar).imag) > _COMPLEX_ZERO_TOL:
        return complex(scalar)
    return float(np.real(scalar))


def _contact_impedance_vector_from_meta(
    meta: dict[str, Any], *, total_electrodes: int
) -> np.ndarray:
    raw = meta.get("contact_impedance", 0.01)
    total = max(int(total_electrodes), 1)
    if raw is None or raw == "":
        return np.full(total, 0.01, dtype=float)
    arr = _contact_impedance_array(raw).reshape(-1)
    dtype = np.complex128 if np.iscomplexobj(arr) else float
    if arr.size == 1:
        return np.full(total, arr[0], dtype=dtype)
    if arr.size == total:
        return arr.astype(dtype, copy=False)
    if arr.size > 0 and total % arr.size == 0:
        out = np.empty(total, dtype=dtype)
        for repeat_idx in range(total // arr.size):
            start = repeat_idx * arr.size
            out[start : start + arr.size] = arr
        return out
    if arr.size != total:
        raise ValueError(
            "contact_impedance length mismatch: "
            f"expected {total} or a divisor of it, got {arr.size}."
        )
    return arr.astype(dtype, copy=False)


def _single_step_semantic_signature(meta: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(meta.get("single_step_signature_schema_version", "")),
        str(meta.get("single_step_jacobian_calculator", "")),
        str(meta.get("single_step_jacobian_math_convention", "")),
        str(meta.get("single_step_projection_math_convention", "")),
        str(meta.get("single_step_operator_math_convention", "")),
    )


_DEFAULT_SINGLE_STEP_SIGMA_FLOOR = 1.0e-6
_SINGLE_STEP_SIGMA_STEP_MARGIN = 1.0e-9


def _single_step_sigma_floor(meta: dict[str, Any]) -> float:
    raw = meta.get(
        "sigma_floor", meta.get("conductivity_floor", _DEFAULT_SINGLE_STEP_SIGMA_FLOOR)
    )
    try:
        floor = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_SINGLE_STEP_SIGMA_FLOOR
    if not np.isfinite(floor) or floor <= 0.0:
        return _DEFAULT_SINGLE_STEP_SIGMA_FLOOR
    return floor


def _limit_single_step_alpha_for_sigma_floor(
    sigma_bg: np.ndarray,
    delta_sigma: np.ndarray,
    alpha: float,
    *,
    sigma_floor: float,
) -> float:
    try:
        requested = float(alpha)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(requested) or requested <= 0.0:
        return 0.0

    sigma = _real_sigma_update_array(sigma_bg)
    delta = np.asarray(_real_sigma_update_array(delta_sigma), dtype=sigma.dtype)
    if sigma.shape != delta.shape:
        raise ValueError(
            f"sigma_bg and delta_sigma shape mismatch: {sigma.shape} != {delta.shape}"
        )
    if sigma.size == 0 or not all_finite_values(sigma) or not all_finite_values(delta):
        return 0.0

    max_alpha = min_alpha_for_value_floor(sigma, delta, float(sigma_floor))
    if not np.isfinite(max_alpha):
        return requested

    if not np.isfinite(max_alpha) or max_alpha <= 0.0:
        return 0.0
    interior_alpha = max_alpha * (1.0 - _SINGLE_STEP_SIGMA_STEP_MARGIN)
    if not np.isfinite(interior_alpha) or interior_alpha <= 0.0:
        interior_alpha = np.nextafter(max_alpha, 0.0)
    return max(0.0, min(requested, float(interior_alpha)))


def _constrain_single_step_sigma_update(
    sigma_bg: np.ndarray,
    delta_sigma: np.ndarray,
    alpha: float,
    *,
    sigma_floor: float,
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    limited_alpha = _limit_single_step_alpha_for_sigma_floor(
        sigma_bg,
        delta_sigma,
        alpha,
        sigma_floor=sigma_floor,
    )
    sigma = _real_sigma_update_array(sigma_bg)
    delta = np.asarray(_real_sigma_update_array(delta_sigma), dtype=sigma.dtype)
    if sigma.shape != delta.shape:
        raise ValueError(
            f"sigma_bg and delta_sigma shape mismatch: {sigma.shape} != {delta.shape}"
        )
    if not all_finite_values(sigma) or not all_finite_values(delta):
        raise RuntimeError(
            "single-step conductivity update contains non-finite values."
        )

    raw_sigma_est = np.empty_like(sigma)
    np.multiply(delta, float(limited_alpha), out=raw_sigma_est)
    raw_sigma_est += sigma
    if not all_finite_values(raw_sigma_est):
        raise RuntimeError("single-step conductivity estimate is non-finite.")

    floor_value = np.asarray(
        np.nextafter(float(sigma_floor), np.inf), dtype=sigma.dtype
    )
    sigma_est = np.maximum(raw_sigma_est, floor_value)
    floor_applied = any_not_equal_values(sigma_est, raw_sigma_est)
    display_delta = np.empty_like(sigma_est)
    np.subtract(sigma_est, sigma, out=display_delta)
    return float(limited_alpha), display_delta, sigma_est, floor_applied


def _meta_requests_complex_admittivity(meta: dict[str, Any]) -> bool:
    hints = " ".join(
        str(meta.get(key, ""))
        for key in (
            "eit_value_mode",
            "complex_measurement_mode",
            "complex_reconstruction_dispatch",
            "compute_dtype",
            "rm_dtype",
            "rm_matmul_dtype",
        )
    ).lower()
    if "complex" in hints:
        return True
    for key in ("background_sigma", "background_conductivity", "contact_impedance"):
        value = meta.get(key)
        if value in (None, ""):
            continue
        if abs(_complex_scalar_from_value(value).imag) > _COMPLEX_ZERO_TOL:
            return True
    return False


def _resolve_reconstruction_runtime(
    meta: dict[str, Any], *, mesh_dim: int
) -> dict[str, Any]:
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()

    def _auto(key: str, default: str) -> str:
        raw = str(meta.get(key, "") or "").strip().lower()
        return default if raw in {"", "auto"} else raw

    requested_profile = _auto("acceleration_profile", "default")
    mesh_family = _auto("mesh_family", "tetra")
    forward_backend = _auto("forward_backend", "dolfinx")
    potential_order = max(1, int(meta.get("potential_order", 1) or 1))
    complex_admittivity = _meta_requests_complex_admittivity(meta)
    if complex_admittivity and forward_backend == "cuda_structured":
        forward_backend = "dolfinx"
    if potential_order != 1 and forward_backend == "cuda_structured":
        raise ValueError(
            "potential_order > 1 requires the DOLFINx forward backend; "
            "cuda_structured currently supports only P1."
        )
    wants_gpu_request = gui_profile == "gpu" or requested_profile in {
        "gpu3d",
        "gpu3d_fused",
    }
    wants_structured_gpu = (
        int(mesh_dim) == 3
        and mesh_family == "hex"
        and potential_order == 1
        and not complex_admittivity
        and (wants_gpu_request or forward_backend == "cuda_structured")
    )
    wants_3d_cuda = int(mesh_dim) == 3 and (
        wants_gpu_request or forward_backend == "cuda_structured"
    )

    acceleration_profile = requested_profile
    if wants_3d_cuda and acceleration_profile == "default":
        acceleration_profile = "gpu3d"
    if int(mesh_dim) != 3 and acceleration_profile in {"gpu3d", "gpu3d_fused"}:
        acceleration_profile = "default"

    if wants_structured_gpu and forward_backend == "dolfinx":
        forward_backend = "cuda_structured"
    elif not wants_structured_gpu and forward_backend == "cuda_structured":
        forward_backend = "dolfinx"

    petsc_device = _auto("petsc_device", "cuda" if wants_3d_cuda else "cpu")
    capability: dict[str, Any] = {}
    if int(mesh_dim) == 3 and petsc_device == "cuda":
        try:
            capability = dict(probe_petsc_cuda_runtime())
        except Exception as exc:
            capability = {"errors": {"forward_solver_policy": str(exc)}}
    solver_policy = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset=_auto("forward_solver_preset", "auto"),
        mesh_dim=int(mesh_dim),
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        capability=capability,
        prefer_amgx=True,
    )
    mat_solve_policy = resolve_3d_cuda_mat_solve_policy(
        requested_mat_solve=_auto(
            "forward_mat_solve", "auto" if int(mesh_dim) == 3 else "off"
        ),
        mesh_dim=int(mesh_dim),
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        solver_preset=solver_policy["forward_solver_preset_effective"],
    )

    return {
        "solver_mode": _auto("solver_mode", "fast" if int(mesh_dim) == 3 else "strict"),
        "line_search_mode": _auto(
            "line_search_mode", "fast" if int(mesh_dim) == 3 else "full"
        ),
        "linear_solver": _auto("linear_solver", "auto"),
        "preconditioner": _auto("preconditioner", "auto"),
        "fast_linear_path": _auto("fast_linear_path", "auto"),
        "forward_solver_preset": str(solver_policy["forward_solver_preset_effective"]),
        "forward_solver_preset_requested": str(
            solver_policy["forward_solver_preset_requested"]
        ),
        "forward_solver_policy_reason": str(
            solver_policy["forward_solver_policy_reason"]
        ),
        "forward_solver_policy_warning": str(
            solver_policy["forward_solver_policy_warning"]
        ),
        "petsc_amgx_available": bool(solver_policy["petsc_amgx_available"]),
        "petsc_hypre_available": bool(solver_policy["petsc_hypre_available"]),
        "petsc_hypre_cuda_blacklisted": bool(
            solver_policy["petsc_hypre_cuda_blacklisted"]
        ),
        "forward_mat_solve": str(
            mat_solve_policy["forward_mat_solve_effective_policy"]
        ),
        "forward_mat_solve_requested": str(
            mat_solve_policy["forward_mat_solve_requested"]
        ),
        "forward_mat_solve_policy_reason": str(
            mat_solve_policy["forward_mat_solve_policy_reason"]
        ),
        "forward_mat_solve_policy_warning": str(
            mat_solve_policy["forward_mat_solve_policy_warning"]
        ),
        "petsc_device": petsc_device,
        "device": _auto("device", "cuda" if wants_3d_cuda else "auto"),
        "forward_backend": forward_backend,
        "mesh_family": mesh_family,
        "geometry_version": _auto("geometry_version", "geomv2"),
        "potential_order": potential_order,
        "acceleration_profile": acceleration_profile,
    }


def clear_reconstruction_system_cache() -> None:
    """Clear the in-process EITSystem cache used by realtime reconstruction."""
    with _SYSTEM_CACHE_LOCK:
        _SYSTEM_CACHE.clear()
        _SYSTEM_CACHE_SIZES.clear()
    with _FAST_CONTEXT_CACHE_LOCK:
        _FAST_CONTEXT_CACHE.clear()
        _FAST_CONTEXT_CACHE_SIZES.clear()


@dataclass
class ReconstructionRequest:
    """Input for a reconstruction job."""

    reference_frame: FrameData
    target_frame: FrameData
    use_part: str = "real"
    method: str = "gn-difference"
    regularization_alpha: float = 1.0
    max_iterations: int = 10
    mesh_dimension: int = 2
    mesh_refinement: float = 4.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Output from a reconstruction job."""

    conductivity: np.ndarray  # element-wise conductivity
    node_coords: np.ndarray  # (n_nodes, 2 or 3)
    cell_connectivity: np.ndarray  # (n_cells, verts_per_cell)
    measured: np.ndarray | None = None
    simulated: np.ndarray | None = None
    error_msg: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SingleStepCachedRuntimeConfig:
    meta: dict[str, Any]
    mesh_dim: int
    refinement: int
    lam: float
    background_sigma: float | complex
    contact_impedance: float | complex
    mesh_height: float
    electrode_height_ratio: float
    z_center: float
    cache_key: tuple[Any, ...]


_COMPLEX_ZERO_TOL = 1.0e-12


def _complex_scalar_from_value(value: Any, default: complex = 0.0 + 0.0j) -> complex:
    if value is None:
        return complex(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return complex(default)
        try:
            return complex(parse_complex_scalar(text))
        except ValueError:
            return complex(default)
    try:
        arr = np.asarray(value, dtype=np.complex128).reshape(-1)
    except (TypeError, ValueError):
        try:
            return complex(value)
        except (TypeError, ValueError):
            return complex(default)
    if arr.size == 0:
        return complex(default)
    return complex(arr[0])


def _real_if_close_scalar(
    value: Any, *, tol: float = _COMPLEX_ZERO_TOL
) -> float | complex:
    scalar = complex(value)
    if abs(scalar.imag) <= tol:
        return float(scalar.real)
    return scalar


def _background_scalar_from_meta(meta: dict[str, Any]) -> float | complex:
    return _real_if_close_scalar(
        _complex_scalar_from_value(
            meta.get("background_sigma", meta.get("background_conductivity", 1.0)),
            1.0 + 0.0j,
        )
    )


def _scalar_for_array_dtype(value: Any, dtype: np.dtype) -> float | complex:
    scalar = _real_if_close_scalar(value)
    if np.issubdtype(dtype, np.complexfloating):
        return np.asarray(complex(scalar), dtype=dtype).item()
    return float(complex(scalar).real)


def _is_complex_measurement_request(req: ReconstructionRequest) -> bool:
    if str(getattr(req, "use_part", "real")).strip().lower() == "complex":
        return True
    metadata = dict(getattr(req, "metadata", {}) or {})
    mode = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "eit_value_mode",
            "complex_measurement_mode",
            "complex_reconstruction_dispatch",
        )
    ).lower()
    return "complex" in mode and "split" not in mode


def _is_native_complex_reconstruction_request(
    req: ReconstructionRequest,
    ref_vec: np.ndarray,
    tgt_vec: np.ndarray,
) -> bool:
    return bool(
        np.iscomplexobj(ref_vec)
        or np.iscomplexobj(tgt_vec)
        or _is_complex_measurement_request(req)
    )


def _eit_system_difference_preset_for_full_gn(
    requested_preset: Any,
    *,
    native_complex: bool,
) -> str:
    requested = (
        str(requested_preset or _DEFAULT_EIT_SYSTEM_DIFFERENCE_PRESET).strip().lower()
    )
    if requested in _EIT_SYSTEM_DIFFERENCE_PRESETS:
        return requested
    if native_complex:
        return _DEFAULT_EIT_SYSTEM_DIFFERENCE_PRESET
    return requested


def _native_complex_dtype_from_request(
    req: ReconstructionRequest,
    ref_vec: np.ndarray,
    tgt_vec: np.ndarray,
) -> np.dtype:
    meta = dict(getattr(req, "metadata", {}) or {})
    dtype_hint = str(
        meta.get("compute_dtype", meta.get("dtype", meta.get("precision", "")))
    ).lower()
    if "complex64" in dtype_hint:
        return np.dtype(np.complex64)
    if "complex128" in dtype_hint:
        return np.dtype(np.complex128)
    raw_dtype = np.result_type(np.asarray(ref_vec).dtype, np.asarray(tgt_vec).dtype)
    if raw_dtype == np.complex64:
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


def _regularization_for_native_complex(reconstructor, n_param: int):
    matrix = getattr(reconstructor, "R_matrix", None)
    if matrix is not None:
        return matrix
    diag = getattr(reconstructor, "R_diag", None)
    if diag is not None:
        arr = np.asarray(diag, dtype=np.float64).reshape(-1)
        if arr.size == int(n_param):
            return arr
    return None


def _run_native_complex_linearized_difference(
    *,
    req: ReconstructionRequest,
    system,
    ref_vec: np.ndarray,
    tgt_vec: np.ndarray,
    meta: dict[str, Any],
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    from pyeidors.data.structures import EITImage
    from pyeidors.inverse.solvers.gauss_newton_linear_system import (
        solve_native_complex_normal_step,
    )
    from dolfinx import fem

    emit("Building native complex Jacobian...")
    background = _complex_scalar_from_value(
        meta.get("background_sigma", meta.get("background_conductivity", 1.0)),
        1.0 + 0.0j,
    )
    complex_dtype = _native_complex_dtype_from_request(req, ref_vec, tgt_vec)
    n_elements = int(fem.Function(system.fwd_model.V_sigma).x.array.size)
    sigma_bg = np.full(n_elements, background, dtype=complex_dtype)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=system.fwd_model)
    jacobian = system.reconstructor.jacobian_calculator.calculate_from_image(img_bg)
    if bool(getattr(system.reconstructor, "negate_jacobian", True)):
        jacobian = -np.asarray(jacobian)
    else:
        jacobian = np.asarray(jacobian)

    difference_mode = str(meta.get("difference_mode", "raw"))
    difference_orientation = str(
        meta.get("difference_orientation", "target_minus_reference")
    )
    measured_diff = build_difference_vector(
        tgt_vec,
        ref_vec,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    residual = -np.asarray(measured_diff, dtype=np.result_type(measured_diff, jacobian))
    lambda_eff = float(
        meta.get(
            "difference_lambda",
            meta.get("lambda_eff", req.regularization_alpha),
        )
    )
    try:
        system.reconstructor.ensure_regularization_ready()
    except Exception:
        log.debug("Native complex path using identity regularization", exc_info=True)
    regularization = _regularization_for_native_complex(
        system.reconstructor,
        int(jacobian.shape[1]),
    )

    emit("Solving native complex normal equation...")
    delta_sigma, solve_meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=lambda_eff,
        regularization=regularization,
    )
    simulated_diff = np.asarray(jacobian @ delta_sigma)
    mesh = system.mesh
    coords = mesh.coordinates()
    cells = mesh.cells()
    result_meta = dict(meta)
    result_meta.update(
        {
            "eit_value_mode": "complex_admittance",
            "complex_reconstruction_mode": "native_complex_linearized_gn",
            "complex_reconstruction_approximation": "single_step_linearized",
            "complex_reconstruction_dispatch": "native_complex",
            "reconstruction_runtime": "native_complex_linearized",
            "native_complex_dtype": str(complex_dtype),
            "conductivity_display_mode": "absolute_sigma",
            "complex_background_admittance": background,
            "native_complex_solver": solve_meta,
        }
    )
    emit("Native complex reconstruction complete")
    return ReconstructionResult(
        conductivity=np.asarray(background + delta_sigma),
        node_coords=coords,
        cell_connectivity=cells,
        measured=measured_diff,
        simulated=simulated_diff,
        metadata=result_meta,
    )


class _ReconstructionWorker(QObject):
    """Runs reconstruction in a background thread."""

    finished = Signal(object)  # ReconstructionResult
    progress = Signal(str)  # status messages
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._request: ReconstructionRequest | None = None
        self._eit_system = None  # lazy import pyeidors
        self._cancel_requested = False
        self._backend_profile: str | None = None

    def cancel(self) -> None:
        self._cancel_requested = True
        profile = self._backend_profile
        if profile:
            try:
                from eit_app.backend_worker_pool import stop_persistent_backend_worker

                stop_persistent_backend_worker(repo=_repo_root(), profile=profile)
            except Exception:
                log.debug(
                    "Failed to stop persistent backend worker during cancellation",
                    exc_info=True,
                )

    @Slot()
    def run(self) -> None:
        req = self._request
        if req is None:
            self.error.emit("No reconstruction request set")
            return
        if self._cancel_requested:
            return

        def _progress(message: str) -> None:
            if not self._cancel_requested:
                self.progress.emit(message)

        try:
            from eit_app.backend_routing import select_reconstruction_backend_route

            route = select_reconstruction_backend_route(req)
            if route.external:
                self._backend_profile = route.profile
                try:
                    result = execute_reconstruction_request_in_backend(
                        req,
                        profile=route.profile,
                        route_reason=route.reason,
                        progress_cb=_progress,
                        cancelled=lambda: self._cancel_requested,
                    )
                finally:
                    self._backend_profile = None
            else:
                result = run_reconstruction_request(req, progress_cb=_progress)
                result.metadata = {
                    **dict(result.metadata or {}),
                    "backend_worker_profile": route.profile,
                    "backend_worker_route_reason": route.reason,
                    "backend_worker_process_isolated": False,
                }
        except Exception as exc:
            if self._cancel_requested:
                return
            log.exception("Reconstruction worker failed")
            result = ReconstructionResult(
                conductivity=np.array([]),
                node_coords=np.array([]),
                cell_connectivity=np.array([]),
                error_msg=str(exc),
                metadata=dict(getattr(req, "metadata", {}) or {}),
            )
        if self._cancel_requested:
            return
        if result.error_msg:
            self.error.emit(result.error_msg)
        self.finished.emit(result)


def _get_cached_system(cache_key: tuple[Any, ...]):
    with _SYSTEM_CACHE_LOCK:
        system = _SYSTEM_CACHE.get(cache_key)
        if system is None:
            return None
        _SYSTEM_CACHE.move_to_end(cache_key)
        return system


def _put_cached_system(cache_key: tuple[Any, ...], system: Any) -> None:
    max_bytes = _parse_bytes_limit(
        getattr(system, "_reconstruction_system_cache_max_bytes", None),
        default=_reconstruction_system_cache_max_bytes(),
    )
    entry_bytes = _reconstruction_system_cache_entry_bytes(system)
    with _SYSTEM_CACHE_LOCK:
        stored = _put_bounded_ordered_cache(
            _SYSTEM_CACHE,
            _SYSTEM_CACHE_SIZES,
            cache_key,
            system,
            entry_bytes=entry_bytes,
            max_items=_SYSTEM_CACHE_MAX_ITEMS,
            max_bytes=max_bytes,
        )
    if not stored:
        log.info(
            "Skipped reconstruction system process cache entry: bytes=%s max_bytes=%s",
            entry_bytes,
            max_bytes,
        )


def _get_cached_fast_context(cache_key: tuple[Any, ...]):
    with _FAST_CONTEXT_CACHE_LOCK:
        ctx = _FAST_CONTEXT_CACHE.get(cache_key)
        if ctx is None:
            return None
        _FAST_CONTEXT_CACHE.move_to_end(cache_key)
        return ctx


def _put_cached_fast_context(cache_key: tuple[Any, ...], ctx: Any) -> None:
    ctx_meta = ctx if isinstance(ctx, dict) else {}
    max_bytes = _single_step_context_cache_max_bytes(ctx_meta)
    entry_bytes = _fast_context_cache_entry_bytes(ctx)
    if isinstance(ctx, dict):
        ctx["single_step_context_process_cache_bytes"] = int(entry_bytes)
        ctx["single_step_context_process_cache_max_bytes"] = int(max_bytes)
    with _FAST_CONTEXT_CACHE_LOCK:
        stored = _put_bounded_ordered_cache(
            _FAST_CONTEXT_CACHE,
            _FAST_CONTEXT_CACHE_SIZES,
            cache_key,
            ctx,
            entry_bytes=entry_bytes,
            max_items=_FAST_CONTEXT_CACHE_MAX_ITEMS,
            max_bytes=max_bytes,
        )
    if isinstance(ctx, dict):
        ctx["single_step_context_process_cache_stored"] = bool(stored)
        ctx["single_step_context_process_cache_skip_reason"] = (
            "" if stored else "entry_too_large"
        )
    if not stored:
        log.info(
            "Skipped single-step context process cache entry: bytes=%s max_bytes=%s",
            entry_bytes,
            max_bytes,
        )


def _quiet_call(fn: Callable[[], Any]) -> Any:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = fn()
    captured = sink.getvalue().strip()
    if captured:
        log.debug("Suppressed realtime reconstruction output:\n%s", captured)
    return result


def _recover_nix_runtime_site_packages(missing_name: str) -> tuple[str, ...]:
    """Best-effort recovery for nix-provided Python runtime packages.

    This keeps the GUI realtime reconstruction path working even when the app
    was launched with `PYTHONPATH=src`, which accidentally drops the nix
    shell's FEniCSx-related site-packages.
    """
    missing = str(missing_name or "").strip().lower()
    if missing not in {"ufl", "dolfinx", "mpi4py", "petsc4py", "ffcx", "basix"}:
        return ()
    nix_store = Path("/nix/store")
    if not nix_store.exists():
        return ()

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    patterns = (
        f"/nix/store/*-{pyver}-fenics-dolfinx-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-ufl-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-basix-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-ffcx-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-mpi4py-*/lib/{pyver}/site-packages",
        f"/nix/store/*-petsc-*/lib/{pyver}/site-packages",
        f"/nix/store/*-slepc-*/lib/{pyver}/site-packages",
    )
    discovered: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if candidate in seen or not os.path.isdir(candidate):
                continue
            seen.add(candidate)
            discovered.append(candidate)
    added: list[str] = []
    for candidate in reversed(discovered):
        if candidate in sys.path:
            continue
        sys.path.insert(0, candidate)
        added.append(candidate)
    if added:
        current = os.environ.get("PYTHONPATH", "")
        prefix = os.pathsep.join(added)
        os.environ["PYTHONPATH"] = (
            prefix if not current else f"{prefix}{os.pathsep}{current}"
        )
        log.info(
            "Recovered nix runtime site-packages for realtime reconstruction (%s): %s",
            missing,
            added,
        )
    return tuple(added)


@lru_cache(maxsize=1)
def _load_gn_difference_runner_module():
    """Load the packaged realtime GN helper module."""
    module_name = "pyeidors.realtime.gn_difference_runner"

    for _attempt in range(4):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = str(getattr(exc, "name", "") or "")
            recovered = _recover_nix_runtime_site_packages(missing_name)
            if recovered:
                importlib.invalidate_caches()
                sys.modules.pop(module_name, None)
                continue
            raise
    raise ModuleNotFoundError(f"Unable to import {module_name}")


def _compute_effective_refinement(
    radius: float,
    mesh_refinement: float,
    *,
    mesh_size: float | None = None,
) -> int:
    """Resolve the optimized-mesh refinement used by reconstruction.

    Hardware reconstruction passes the historical integer refinement control
    (4, 8, ...).  The Simulation tab passes a physical mesh_size such as 0.1.
    Treating that mesh_size as ``1 / mesh_size`` and then applying the legacy
    conversion inflates 0.1 to ref20, which makes simulation inverse appear to
    hang while loading/building an unnecessarily dense cache mesh.
    """

    radius_f = max(float(radius), 1e-9)
    size_f = None
    if mesh_size is not None:
        try:
            size_f = float(mesh_size)
        except (TypeError, ValueError):
            size_f = None
    if size_f is not None and np.isfinite(size_f) and size_f > 0.0:
        return max(2, int(round(radius_f / max(size_f, 1e-6) / 2.0)))

    try:
        refinement_f = float(mesh_refinement)
    except (TypeError, ValueError):
        refinement_f = 4.0
    if np.isfinite(refinement_f) and 0.0 < refinement_f < 1.0:
        return max(2, int(round(radius_f / max(refinement_f, 1e-6) / 2.0)))

    mesh_size_f = max(0.02, 0.25 / max(1, int(refinement_f)))
    return max(2, int(round(radius_f / max(mesh_size_f, 1e-6) / 2.0)))


def default_rm_inverse_mesh_size(
    requested_mesh_size: float,
    radius: float,
    *,
    mesh_dimension: int = 3,
) -> float:
    """Return the default coarse inverse mesh size for auto-built RM routes."""

    try:
        requested = float(requested_mesh_size)
    except (TypeError, ValueError):
        requested = 0.1
    if not np.isfinite(requested) or requested <= 0.0:
        requested = 0.1
    radius_f = max(float(radius), 1.0e-9)
    if int(mesh_dimension) == 3:
        # A one-step 3D RM reconstruction has far fewer voltage
        # measurements than cell parameters.  Auto-refining the inverse
        # mesh makes NOSER chase boundary artifacts, so keep the hidden
        # default deliberately coarse unless the user explicitly overrides
        # ``rm_inverse_mesh_size``.
        target = radius_f / 3.0
        return float(max(requested, max(target, 1.0e-6)))
    else:
        target = radius_f / 10.0
    return float(min(requested, max(target, 1.0e-6)))


def _resolve_drive_mode(
    meta: dict[str, Any],
    *,
    mesh_dim: int,
    default: str = "total_current",
) -> str:
    raw_mode = meta.get("drive_mode", default)
    mode = drive_mode_for_mesh_dimension(raw_mode, mesh_dim)
    return mode or default


def _resolve_drive_value(
    meta: dict[str, Any],
    *,
    default: float = 1.0e-5,
) -> float:
    raw_value = meta.get("drive_value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value) and value > 0.0:
        return value

    raw_stim_uA = meta.get("stim_amp_uA")
    try:
        stim_uA = float(raw_stim_uA)
    except (TypeError, ValueError):
        stim_uA = float("nan")
    if np.isfinite(stim_uA) and stim_uA > 0.0:
        return stim_uA * 1.0e-6

    return float(default)


def _first_present_meta_value(meta: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _warm_greit_common_config_artifact_from_meta(meta: dict[str, Any]) -> Path | None:
    if not _flag_enabled(meta.get("greit_common_config_auto_warm", False)):
        return None
    if not _flag_enabled(meta.get("greit_fixture_auto_warm_allowed", False)):
        meta["greit_common_config_unavailable_reason"] = (
            "GREIT production route requires a registered EIDORS-parity artifact; "
            "deterministic fixture auto-warm is disabled."
        )
        return None
    config_id = _first_present_meta_value(
        meta,
        "greit_common_config",
        "greit_common_config_id",
        "common_greit_config",
        "common_config",
    )
    if config_id is None:
        return None
    artifact_dir = _first_present_meta_value(
        meta,
        "greit_common_config_dir",
        "greit_common_artifact_dir",
        "common_greit_artifact_dir",
    )
    warmup = precompute_greit_common_config(
        config_id,
        artifact_dir=artifact_dir,
        overwrite=False,
        prepare_online=False,
    )
    meta["greit_common_config"] = warmup.config.config_id
    meta["greit_common_config_artifact_path"] = str(warmup.artifact_path)
    meta["greit_common_config_dir"] = str(warmup.artifact_path.parent)
    meta["rm_artifact_auto_built"] = bool(warmup.built)
    meta["rm_artifact_cache_status"] = "built" if warmup.built else "disk_hit"
    return warmup.artifact_path


def _resolve_rm_artifact_path(meta: dict[str, Any]) -> Path | None:
    for key in _RM_ARTIFACT_META_KEYS:
        raw = meta.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            return path
        if not path.is_absolute():
            cache_relative = resolve_pyeidors_cache_dir(path)
            if cache_relative != path and cache_relative.exists():
                return cache_relative
            runtime_relative = pyeidors_cache_path(path)
            if runtime_relative.exists():
                return runtime_relative
            repo_relative = Path(__file__).resolve().parents[3] / path
            if repo_relative.exists():
                return repo_relative
        raise FileNotFoundError(f"RM artifact path does not exist: {text}")
    try:
        common_path = resolve_greit_common_config_artifact_path_from_meta(meta)
    except FileNotFoundError as exc:
        common_path = _warm_greit_common_config_artifact_from_meta(meta)
        if common_path is None:
            meta["rm_artifact_unavailable_reason"] = str(
                meta.get("greit_common_config_unavailable_reason") or exc
            )
            return None
    if common_path is not None:
        return common_path
    return None


def _json_signature_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_signature_value(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return _json_signature_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_signature_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_signature_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_json_digest(value: Any) -> str:
    payload = json.dumps(
        _json_signature_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_relative_or_absolute_path(path: Any, *, default: str) -> Path:
    raw = str(path or default).strip() or default
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    cache_relative = resolve_pyeidors_cache_dir(candidate)
    if cache_relative != candidate:
        return cache_relative
    return pyeidors_cache_path(candidate)


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _meta_value_present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _simulation_inverse_route(meta: dict[str, Any]) -> str:
    return str(meta.get("simulation_inverse_route", "")).strip().lower()


def _one_step_rm_regularization(meta: dict[str, Any]) -> str:
    route = _simulation_inverse_route(meta)
    regularization = _ONE_STEP_RM_ROUTE_REGULARIZATION.get(route)
    if regularization:
        return regularization
    return str(meta.get("rm_regularization", "")).strip().lower()


def _is_noser_rm_route(meta: dict[str, Any]) -> bool:
    return _simulation_inverse_route(meta) == "noser_rm"


def _is_auto_built_one_step_rm_route(meta: dict[str, Any]) -> bool:
    return _simulation_inverse_route(meta) in _AUTO_BUILD_RM_ROUTES


def _should_auto_build_rm_artifact(meta: dict[str, Any]) -> bool:
    route = _simulation_inverse_route(meta)
    return route in _AUTO_BUILD_RM_ROUTES and _flag_enabled(
        meta.get("rm_auto_build", False)
    )


def _single_step_context_cache_scope(meta: dict[str, Any]) -> str:
    raw = str(meta.get("single_step_context_cache_scope", "both") or "both")
    scope = raw.strip().lower()
    if scope in {"memory", "runtime", "session"}:
        return "process"
    if scope in {"process", "disk", "both", "off"}:
        return scope
    return "both"


def _should_resolve_greit_registry(meta: dict[str, Any]) -> bool:
    return _simulation_inverse_route(meta) in _GREIT_REGISTRY_ROUTES and (
        _flag_enabled(meta.get("greit_registry_auto_resolve", False))
        or _flag_enabled(meta.get("rm_auto_build", False))
        or bool(meta.get("greit_registry_signature"))
    )


def _channel_mask_from_meta(
    meta: dict[str, Any],
    *,
    n_measurements: int,
) -> np.ndarray | None:
    from pyeidors.data.channels import normalize_bad_channel_mask

    for key in ("channel_mask", "bad_channel_mask"):
        if key in meta and _meta_value_present(meta[key]):
            return normalize_bad_channel_mask(
                meta[key],
                n_measurements=n_measurements,
            )
    for key in ("bad_channels", "bad_channel_indices"):
        if key in meta and _meta_value_present(meta[key]):
            return normalize_bad_channel_mask(
                meta[key],
                n_measurements=n_measurements,
            )
    return None


def _measurement_weights_from_meta(
    meta: dict[str, Any],
    *,
    n_measurements: int,
) -> np.ndarray | None:
    _ = n_measurements
    for key in ("measurement_weights", "noise_precision", "W"):
        if key in meta and _meta_value_present(meta[key]):
            weights = np.asarray(meta[key], dtype=np.float64)
            return np.ascontiguousarray(weights, dtype=np.float64)
    return None


def _greit_registry_config_from_runtime(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
) -> dict[str, Any]:
    meta = runtime.meta
    n_meas = max(_request_measurement_count(req), 1)
    channel_mask = _channel_mask_from_meta(meta, n_measurements=n_meas)
    weights = _measurement_weights_from_meta(meta, n_measurements=n_meas)
    config = dict(meta.get("greit_registry_config") or {})
    config.update(
        {
            "mesh_dimension": int(runtime.mesh_dim),
            "mesh_refinement": meta.get("mesh_size", req.mesh_refinement),
            "mesh_family": meta.get("mesh_family"),
            "geometry_version": meta.get("geometry_version"),
            "n_elec": int(meta.get("n_elec", 16)),
            "n_rings": int(meta.get("n_rings", 1)),
            "n_layers": int(meta.get("n_layers", meta.get("n_rings", 1))),
            "radius": float(meta.get("radius", 1.0)),
            "height": float(meta.get("height", meta.get("mesh_height", 1.0))),
            "electrode_length_m_override": meta.get("electrode_length_m_override"),
            "electrode_area_m2_override": meta.get("electrode_area_m2_override"),
            "electrode_height_ratio": float(
                meta.get("electrode_height_ratio", runtime.electrode_height_ratio)
            ),
            "electrode_level_fractions": meta.get(
                "electrode_level_fractions", (0.25, 0.75)
            ),
            "electrode_layout": str(meta.get("electrode_layout", "ring_major")),
            "measurement_protocol": str(
                meta.get("measurement_protocol", "eidors_full_3d")
            ),
            "stim_pattern": str(meta.get("stim_pattern", "{ad}")),
            "meas_pattern": str(meta.get("meas_pattern", "{ad}")),
            "rotate_meas": bool(meta.get("rotate_meas", True)),
            "use_meas_current": bool(meta.get("use_meas_current", False)),
            "use_meas_current_next": int(meta.get("use_meas_current_next", 0)),
            "stim_direction": str(meta.get("stim_direction", "ccw")),
            "meas_direction": str(meta.get("meas_direction", "ccw")),
            "stim_first_positive": bool(meta.get("stim_first_positive", False)),
            "custom_stim_matrix": meta.get("custom_stim_matrix"),
            "custom_meas_matrices": meta.get("custom_meas_matrices"),
            "measurement_count": n_meas,
            "channel_order": np.arange(n_meas, dtype=np.int64),
            "bad_channel_mask": channel_mask,
            "measurement_weights": weights,
            "background_conductivity": float(complex(runtime.background_sigma).real),
            "contact_impedance": runtime.contact_impedance,
            "normalize_measurements": str(
                meta.get("difference_mode", "normalized")
            ).lower()
            == "normalized",
            "builder_backend": str(meta.get("greit_builder_backend", "native")),
            "builder_semantic_version": str(
                meta.get("greit_builder_semantic_version", "")
            ),
        }
    )
    for key in (
        "imgsz",
        "greit_imgsz",
        "xvec",
        "yvec",
        "zvec",
        "greit_xvec",
        "greit_yvec",
        "greit_zvec",
        "downsample",
        "greit_downsample",
        "target_distribution",
        "distr",
        "target_size",
        "greit_target_size",
        "target_radius",
        "greit_target_radius",
        "target_contrast",
        "greit_target_contrast",
        "desired_solution_fn",
        "desired_solution_params",
        "greit_desired_options",
        "noise_covar",
        "weight_strategy",
        "greit_weight_strategy",
        "weight",
        "greit_weight",
        "noise_figure",
        "greit_noise_figure",
        "image_SNR",
        "image_snr",
        "training_mode",
        "artifact_schema",
        "greit_use_cached_rm",
        "greit_rebuild_rm",
    ):
        if key in meta and key not in config:
            config[key] = meta[key]
    return config


def _greit_registry_dir_from_meta(meta: dict[str, Any]) -> Path:
    return _cache_relative_or_absolute_path(
        meta.get("greit_registry_dir", meta.get("rm_artifact_dir")),
        default=".pyeidors_cache/greit_artifacts",
    )


def _array_pair_hash(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.asarray(array)
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        update_digest_with_array_payload(digest, arr)
    return digest.hexdigest()


def _one_step_rm_form(
    meta: dict[str, Any], regularization_type: str | None = None
) -> str:
    regularization = (
        str(regularization_type or _one_step_rm_regularization(meta) or "noser")
        .strip()
        .lower()
    )
    if regularization in {"laplace", "curvature", "graph_ltl", "tv_irls"}:
        return "param"
    requested = str(meta.get("rm_form", "")).strip().lower()
    if requested in {"param", "measurement"}:
        return requested
    return "measurement"


def _planned_one_step_rm_signature(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
) -> tuple[str, dict[str, Any]]:
    from pyeidors.inverse.reconstruction_matrix import rm_signature_payload

    meta = runtime.meta
    n_meas = max(_request_measurement_count(req), 1)
    channel_mask = _channel_mask_from_meta(meta, n_measurements=n_meas)
    weights = _measurement_weights_from_meta(meta, n_measurements=n_meas)
    hp = math.sqrt(max(float(runtime.lam), 0.0))
    regularization_type = _one_step_rm_regularization(meta) or "noser"
    rm_form = _one_step_rm_form(meta, regularization_type)
    rm_dtype_name = _rm_dtype_name_from_meta(meta)
    graph_weight = str(meta.get("rm_graph_weight", "unit")).strip().lower() or "unit"
    forward_mesh_payload = {
        "mesh_dimension": runtime.mesh_dim,
        "forward_mesh_size": meta.get("mesh_size", req.mesh_refinement),
        "mesh_family": meta.get("mesh_family"),
        "geometry_version": meta.get("geometry_version"),
        "radius": meta.get("radius"),
        "height": meta.get("height", meta.get("mesh_height")),
        "electrode_coverage": meta.get("electrode_coverage"),
        "electrode_length_m_override": meta.get("electrode_length_m_override"),
        "electrode_area_m2_override": meta.get("electrode_area_m2_override"),
        "electrode_height_ratio": meta.get("electrode_height_ratio"),
    }
    inverse_mesh_payload = {
        "mesh_dimension": runtime.mesh_dim,
        "inverse_mesh_size": meta.get("rm_inverse_mesh_size"),
        "effective_refinement": runtime.refinement,
        "mesh_family": meta.get("mesh_family"),
        "geometry_version": meta.get("geometry_version"),
        "radius": meta.get("radius"),
        "height": meta.get("height", meta.get("mesh_height")),
        "electrode_coverage": meta.get("electrode_coverage"),
        "electrode_length_m_override": meta.get("electrode_length_m_override"),
        "electrode_area_m2_override": meta.get("electrode_area_m2_override"),
        "electrode_height_ratio": meta.get("electrode_height_ratio"),
    }
    inverse_mesh_hash = _stable_json_digest(inverse_mesh_payload)
    payload = rm_signature_payload(
        forward_mesh_hash=_stable_json_digest(forward_mesh_payload),
        inverse_mesh_hash=inverse_mesh_hash,
        coarse2fine_hash=_stable_json_digest(
            {
                "projection": "identity-current-inverse-mesh",
                "inverse_mesh_hash": inverse_mesh_hash,
            }
        ),
        electrode_geometry={
            "n_elec": int(meta.get("n_elec", 16)),
            "n_rings": int(meta.get("n_rings", 1)),
            "electrode_layout": str(meta.get("electrode_layout", "ring_major")),
            "electrode_coverage": float(meta.get("electrode_coverage", 0.5)),
            "electrode_length_m_override": meta.get("electrode_length_m_override"),
            "electrode_area_m2_override": meta.get("electrode_area_m2_override"),
            "electrode_height_ratio": float(
                meta.get("electrode_height_ratio", runtime.electrode_height_ratio)
            ),
        },
        stim_meas_protocol={
            "n_measurements": n_meas,
            "points_per_frame": int(meta.get("points_per_frame", n_meas) or n_meas),
            "measurement_protocol": str(
                meta.get("measurement_protocol", "eidors_full_3d")
            ),
            "stim_pattern": str(meta.get("stim_pattern", "{ad}")),
            "meas_pattern": str(meta.get("meas_pattern", "{ad}")),
            "rotate_meas": bool(meta.get("rotate_meas", True)),
            "use_meas_current": bool(meta.get("use_meas_current", False)),
            "use_meas_current_next": int(meta.get("use_meas_current_next", 0)),
            "stim_direction": str(meta.get("stim_direction", "ccw")),
            "meas_direction": str(meta.get("meas_direction", "ccw")),
            "stim_first_positive": bool(meta.get("stim_first_positive", False)),
            "drive_mode": str(meta.get("drive_mode", "")),
            "drive_value": float(meta.get("drive_value", 1.0e-5)),
            "use_part": str(req.use_part),
            "custom_stim_matrix": meta.get("custom_stim_matrix"),
            "custom_meas_matrices": meta.get("custom_meas_matrices"),
        },
        background={
            "sigma0": _json_signature_value(runtime.background_sigma),
            "z0": _json_signature_value(runtime.contact_impedance),
        },
        difference_mode=str(meta.get("difference_mode", "raw")),
        bad_channel_mask=channel_mask,
        noise_covariance=weights,
        regularization_type=regularization_type,
        hyperparameters={
            "hp": hp,
            "hp_squared": hp * hp,
            "lambda_eff": float(runtime.lam),
            "form": rm_form,
            "singular_prior_form_policy": "param_for_graph_laplace_curvature_v1"
            if regularization_type in {"laplace", "curvature", "graph_ltl"}
            else None,
            "rm_signature_schema_version": str(
                meta.get("one_step_rm_signature_schema_version", "")
            ),
            "rm_jacobian_calculator": str(
                meta.get("single_step_jacobian_calculator", "")
            ),
            "rm_jacobian_math_convention": str(
                meta.get("single_step_jacobian_math_convention", "")
            ),
            "rm_projection_math_convention": str(
                meta.get("single_step_projection_math_convention", "")
            ),
            "rm_jacobian_build_representation": str(
                meta.get(
                    "rm_build_jacobian_representation",
                    meta.get("jacobian_representation", "dense"),
                )
            ),
            "rm_jacobian_build_convention": str(
                meta.get("one_step_rm_jacobian_build_convention", "")
            ),
            "rm_jacobian_source_cache_scope": _single_step_context_cache_scope(meta),
            "rm_prior_math_convention": str(
                meta.get("one_step_rm_prior_math_convention", "")
            ),
            "rm_algorithm_version": str(meta.get("one_step_rm_algorithm_version", "")),
            "rm_content_contract": str(meta.get("one_step_rm_content_contract", "")),
            "rm_dtype": rm_dtype_name,
            "noser_exponent": 0.5 if regularization_type == "noser" else None,
            "prior_operator": {
                "laplace": "eidors_prior_laplace_graph_x2",
                "curvature": "eidors_prior_laplace_squared",
                "graph_ltl": "eidors_prior_laplace_squared",
            }.get(regularization_type),
            "graph_weight": graph_weight
            if regularization_type in {"laplace", "curvature", "graph_ltl"}
            else None,
        },
    )
    return _stable_json_digest(payload), payload


def _planned_noser_rm_signature(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
) -> tuple[str, dict[str, Any]]:
    """Backward-compatible alias for tests and diagnostics."""

    return _planned_one_step_rm_signature(req, runtime)


def _planned_one_step_rm_artifact_path(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
) -> tuple[Path, str, dict[str, Any]]:
    signature, payload = _planned_one_step_rm_signature(req, runtime)
    artifact_dir = _cache_relative_or_absolute_path(
        runtime.meta.get("rm_artifact_dir"),
        default=".pyeidors_cache/gui_rm",
    )
    route = _simulation_inverse_route(runtime.meta) or "one_step_rm"
    return artifact_dir / f"{route}_{signature[:24]}.h5", signature, payload


def _ensure_auto_built_one_step_rm_artifact(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    emit: Callable[[str], None],
) -> Path | None:
    if not _should_auto_build_rm_artifact(runtime.meta):
        return None
    if not _is_auto_built_one_step_rm_route(runtime.meta):
        return None
    route = _simulation_inverse_route(runtime.meta)
    regularization_type = _one_step_rm_regularization(runtime.meta) or "noser"
    rm_form = _one_step_rm_form(runtime.meta, regularization_type)
    rm_dtype_name = _rm_dtype_name_from_meta(runtime.meta)
    rm_dtype = np.dtype(rm_dtype_name)
    runtime.meta["rm_dtype"] = rm_dtype_name
    runtime.meta["rm_matmul_dtype"] = rm_dtype_name
    artifact_path, signature, signature_payload = _planned_one_step_rm_artifact_path(
        req,
        runtime,
    )
    runtime.meta["rm_signature"] = signature
    runtime.meta["rm_signature_payload"] = signature_payload
    if artifact_path.exists():
        fit_jacobian = _restore_rm_fit_jacobian(
            runtime,
            path=artifact_path,
            signature=signature,
        )
        if fit_jacobian is not None or bool(
            runtime.meta.get("rm_fit_jacobian_available_but_skipped", False)
        ):
            runtime.meta["rm_artifact_path"] = str(artifact_path)
            runtime.meta["dual_model_rm_path"] = str(artifact_path)
            runtime.meta["rm_artifact_auto_built"] = False
            runtime.meta["rm_artifact_cache_status"] = "disk_hit"
            message = f"Using cached {regularization_type.upper()} RM artifact..."
            log.info(message)
            emit(message)
            return artifact_path
        runtime.meta["rm_artifact_cache_status"] = "disk_fit_jacobian_miss"
        message = (
            f"Cached {regularization_type.upper()} RM artifact lacks voltage-fit "
            "Jacobian; rebuilding..."
        )
        log.info(message)
        emit(message)

    diff_runner = _load_gn_difference_runner_module()
    message = f"Building {regularization_type.upper()} RM artifact..."
    log.info(message)
    emit(message)
    ctx = _ensure_single_step_cached_context(
        runtime,
        emit=emit,
        build_shared_context=diff_runner.build_shared_context,
        rm_build_only=True,
    )
    jacobian = np.asarray(ctx["J"], dtype=rm_dtype)
    if jacobian.ndim != 2 or 0 in jacobian.shape:
        raise ValueError(
            f"{route} RM build requires a non-empty dense J, got {jacobian.shape}."
        )
    n_meas = int(jacobian.shape[0])
    channel_mask = _channel_mask_from_meta(runtime.meta, n_measurements=n_meas)
    weights = _measurement_weights_from_meta(runtime.meta, n_measurements=n_meas)
    hp = math.sqrt(max(float(runtime.lam), 0.0))

    node_coords = _display_node_coords_array(ctx["display_node_coords"])
    cell_connectivity = _display_cell_connectivity_array(
        ctx["display_cell_connectivity"]
    )
    regularization = None
    graph_weight = (
        str(runtime.meta.get("rm_graph_weight", "unit")).strip().lower() or "unit"
    )
    if regularization_type in {"laplace", "curvature", "graph_ltl"}:
        from pyeidors.inverse import CellMesh
        from pyeidors.inverse.prior import graph_laplacian, graph_ltl_prior

        inverse_mesh = CellMesh(
            coordinates=node_coords,
            cells=cell_connectivity,
            name=f"{route}-inverse-mesh",
        )
        if regularization_type == "laplace":
            regularization = graph_laplacian(inverse_mesh, weight=graph_weight)
        else:
            regularization = graph_ltl_prior(inverse_mesh, weight=graph_weight)

    from pyeidors.inverse.reconstruction_matrix import (
        build_one_step_rm,
        write_rm_artifact,
    )

    rm = build_one_step_rm(
        jacobian,
        regularization=regularization,
        lambda_=hp,
        mode=regularization_type,
        form=rm_form,
        channel_mask=channel_mask,
        measurement_weights=weights,
        dtype=rm_dtype_name,
        return_metadata=True,
    )
    mesh_hash = _array_pair_hash(node_coords, cell_connectivity)
    fit_jacobian_max_bytes = _rm_fit_jacobian_max_bytes(runtime.meta)
    fit_jacobian_bytes = int(jacobian.nbytes)
    persist_fit_jacobian = fit_jacobian_bytes <= int(max(0, fit_jacobian_max_bytes))
    metadata = {
        "algorithm": f"one-step-{regularization_type}",
        "rm_build_route": route,
        "rm_signature": signature,
        "rm_signature_payload": signature_payload,
        "forward_mesh_hash": mesh_hash,
        "inverse_mesh_hash": mesh_hash,
        "coarse2fine_hash": signature_payload["coarse2fine_hash"],
        "difference_mode": str(runtime.meta.get("difference_mode", "raw")),
        "difference_orientation": str(
            runtime.meta.get("difference_orientation", "target_minus_reference")
        ),
        "rm_build_jacobian_representation": str(
            runtime.meta.get("rm_build_jacobian_representation", "dense")
        ),
        "rm_jacobian_source_cache_scope": str(
            _single_step_context_cache_scope(runtime.meta)
        ),
        "rm_form": rm_form,
        "rm_dtype": rm_dtype_name,
        "rm_build_dtype": rm_dtype_name,
        "singular_prior_form_policy": "param_for_graph_laplace_curvature_v1"
        if regularization_type in {"laplace", "curvature", "graph_ltl"}
        else "",
        "one_step_rm_signature_schema_version": str(
            runtime.meta.get("one_step_rm_signature_schema_version", "")
        ),
        "one_step_rm_jacobian_build_convention": str(
            runtime.meta.get("one_step_rm_jacobian_build_convention", "")
        ),
        "one_step_rm_prior_math_convention": str(
            runtime.meta.get("one_step_rm_prior_math_convention", "")
        ),
        "one_step_rm_algorithm_version": str(
            runtime.meta.get("one_step_rm_algorithm_version", "")
        ),
        "one_step_rm_content_contract": str(
            runtime.meta.get("one_step_rm_content_contract", "")
        ),
        "fit_jacobian_persisted": bool(persist_fit_jacobian),
        "rm_fit_jacobian_bytes": fit_jacobian_bytes,
        "rm_fit_jacobian_max_bytes": int(fit_jacobian_max_bytes),
        "fit_jacobian_persist_skip_reason": "" if persist_fit_jacobian else "too_large",
        "rm_output_display_mode": str(
            runtime.meta.get("rm_output_display_mode", "absolute_sigma")
        ),
        "lambda_eff": float(runtime.lam),
        "hp": hp,
        "hp_squared": hp * hp,
        "n_elec": int(runtime.meta.get("n_elec", 16)),
        "n_rings": int(runtime.meta.get("n_rings", 1)),
        "mesh_dimension": int(runtime.mesh_dim),
        "effective_refinement": int(runtime.refinement),
        "inverse_mesh_size": runtime.meta.get("rm_inverse_mesh_size"),
        "rm_graph_weight": graph_weight
        if regularization_type in {"laplace", "curvature", "graph_ltl"}
        else "",
        "online_hot_path": "rm_matmul",
        **dict(rm.metadata),
    }
    write_rm_artifact(
        artifact_path,
        rm.rm,
        metadata=metadata,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        channel_mask=channel_mask,
        measurement_weights=weights,
        jacobian=jacobian if persist_fit_jacobian else None,
    )
    runtime.meta["rm_artifact_path"] = str(artifact_path)
    runtime.meta["dual_model_rm_path"] = str(artifact_path)
    runtime.meta["rm_artifact_auto_built"] = True
    runtime.meta["rm_artifact_cache_status"] = "built"
    runtime.meta["rm_fit_jacobian_bytes"] = fit_jacobian_bytes
    runtime.meta["rm_fit_jacobian_max_bytes"] = int(fit_jacobian_max_bytes)
    if persist_fit_jacobian:
        _stash_rm_fit_jacobian(
            runtime,
            path=artifact_path,
            signature=signature,
            jacobian=jacobian,
            status_prefix="built",
        )
    else:
        runtime.meta["rm_fit_jacobian_cache_status"] = "built_too_large"
        runtime.meta["rm_fit_jacobian_available_but_skipped"] = True
    return artifact_path


def _ensure_greit_registry_artifact(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    emit: Callable[[str], None],
) -> Path | None:
    if not _should_resolve_greit_registry(runtime.meta):
        return None
    config = _greit_registry_config_from_runtime(req, runtime)
    payload = greit_artifact_signature_payload(config)
    signature = _stable_json_digest(payload)
    runtime.meta["greit_registry_signature"] = signature
    runtime.meta["greit_registry_signature_payload"] = payload
    runtime.meta["rm_signature"] = signature
    registry_dir = _greit_registry_dir_from_meta(runtime.meta)
    auto_build = _flag_enabled(runtime.meta.get("rm_auto_build", False))

    def _builder(
        build_config: dict[str, Any],
        _payload: dict[str, Any],
        artifact_path: Path,
    ):
        diff_runner = _load_gn_difference_runner_module()
        emit("Building native GREIT registry artifact...")
        ctx = _ensure_single_step_cached_context(
            runtime,
            emit=emit,
            build_shared_context=diff_runner.build_shared_context,
        )
        from pyeidors.inverse.greit_registry import build_native_greit_artifact

        return build_native_greit_artifact(
            build_config,
            fwd_model=ctx["fwd_model"],
            artifact_path=artifact_path,
            signature=signature,
            signature_payload=payload,
        )

    try:
        lookup = resolve_or_build_greit_artifact(
            config,
            registry_dir=registry_dir,
            auto_build=auto_build,
            builder=_builder if auto_build else None,
            prepare_online=False,
        )
    except FileNotFoundError as exc:
        runtime.meta["greit_registry_unavailable_reason"] = str(exc)
        runtime.meta["rm_artifact_unavailable_reason"] = str(exc)
        return None
    runtime.meta["greit_registry_dir"] = str(registry_dir)
    runtime.meta["greit_registry_manifest_path"] = str(lookup.manifest_path)
    runtime.meta["greit_registry_signature"] = lookup.signature
    runtime.meta["greit_registry_cache_status"] = lookup.cache_status
    runtime.meta["greit_builder_backend"] = lookup.backend
    runtime.meta["rm_artifact_path"] = str(lookup.artifact_path)
    runtime.meta["greit_rm_path"] = str(lookup.artifact_path)
    runtime.meta["rm_artifact_auto_built"] = bool(lookup.built)
    runtime.meta["rm_artifact_cache_status"] = lookup.cache_status
    emit(
        "Using GREIT registry artifact..."
        if not lookup.built
        else "Built GREIT registry artifact."
    )
    return lookup.artifact_path


def _single_step_rm_route_requires_artifact(meta: dict[str, Any]) -> bool:
    route = str(meta.get("simulation_inverse_route", "")).strip().lower()
    return bool(meta.get("rm_route_requires_artifact", False)) or (
        route in _PRODUCTION_RM_ROUTE_TASKS
    )


def _missing_rm_artifact_result(
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    emit: Callable[[str], None],
) -> ReconstructionResult:
    meta = runtime.meta
    route = str(meta.get("simulation_inverse_route", "")).strip().lower() or "rm"
    task = str(
        meta.get("rm_route_pending_task")
        or _PRODUCTION_RM_ROUTE_TASKS.get(route, "T100/T101/T102")
    )
    message = (
        f"{route} requires a precomputed RM/GREIT artifact before reconstruction. "
        f"Build or attach the artifact in {task}, or use debug_fine_mesh_noser for "
        "the current fine-mesh dense baseline."
    )
    mismatch_reason = str(
        meta.get("rm_artifact_unavailable_reason")
        or meta.get("greit_common_config_unavailable_reason")
        or ""
    ).strip()
    if mismatch_reason:
        message = f"{message} {mismatch_reason}"
    emit(message)
    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta.get("n_elec", 0) or 0),
            "reconstruction_runtime": "single_step_cached",
            "single_step_operator_space": "rm",
            "online_hot_path": "rm_matmul",
            "rm_artifact_missing": True,
            "rm_artifact_required": True,
            "rm_route_pending_task": task,
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "solver_diagnostics": {
                "path": "single_step_cached_rm_missing_artifact",
                "strict_solver_backend_effective": "rm",
                "runtime": {
                    "online_hot_path": "rm_matmul",
                    "single_step_operator_space": "rm",
                    "forward_solve_count": 0,
                    "adjoint_solve_count": 0,
                    "jacobian_rebuild_count": 0,
                    "ksp_solve_count": 0,
                    "rm_artifact_missing": True,
                },
            },
        }
    )
    verts_per_cell = 4 if runtime.mesh_dim == 3 else 3
    return ReconstructionResult(
        conductivity=np.asarray([], dtype=np.float64),
        node_coords=np.empty((0, runtime.mesh_dim), dtype=np.float64),
        cell_connectivity=np.empty((0, verts_per_cell), dtype=np.int32),
        error_msg=message,
        metadata=result_meta,
    )


def _parse_int_shape(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        for sep in ("x", "X", ",", " "):
            if sep in raw:
                parts = [
                    part
                    for part in raw.replace("x", sep).replace("X", sep).split(sep)
                    if part
                ]
                break
        else:
            parts = [raw]
        try:
            return tuple(int(part) for part in parts if str(part).strip())
        except ValueError:
            return ()
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(int(v) for v in arr if int(v) > 0)


def _rm_shape_from_meta(meta: dict[str, Any]) -> tuple[int, ...]:
    for key in ("rm_voxel_shape", "inverse_voxel_shape", "coarse_shape", "voxel_shape"):
        shape = _parse_int_shape(meta.get(key))
        if shape:
            return shape
    return ()


def _optional_lazy_artifact_array(
    arrays: dict[str, Any],
    key: str,
    *,
    dtype: Any,
) -> np.ndarray | None:
    raw = arrays.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=dtype)
    if arr.size == 0:
        return None
    return arr


def _optional_lazy_artifact_handle(arrays: dict[str, Any], key: str) -> Any | None:
    raw = arrays.get(key)
    if raw is None:
        return None
    size = getattr(raw, "size", None)
    if size is not None:
        try:
            if int(size) == 0:
                return None
        except (TypeError, ValueError):
            pass
    return raw


def _load_hdf5_rm_artifact_lightweight(
    path: Path, meta: dict[str, Any]
) -> dict[str, Any]:
    from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

    artifact = read_hdf5_artifact(path, lazy=True, verify_checksums=False)
    arrays = dict(artifact.arrays)
    rm = arrays.get("rm")
    if rm is None:
        rm = arrays.get("RM")
    if rm is None:
        raise ValueError(f"RM artifact is missing 'rm': {path}")
    artifact_meta = dict(artifact.metadata)
    voxel_shape = _parse_int_shape(arrays.get("voxel_shape")) or _rm_shape_from_meta(
        meta
    )
    greit_y = _optional_lazy_artifact_handle(arrays, "y")
    if greit_y is None:
        greit_y = _optional_lazy_artifact_handle(arrays, "Y")
    greit_d = _optional_lazy_artifact_handle(arrays, "d")
    if greit_d is None:
        greit_d = _optional_lazy_artifact_handle(arrays, "D")
    node_coords = _optional_lazy_artifact_array(arrays, "node_coords", dtype=np.float64)
    cell_connectivity = _optional_lazy_artifact_array(
        arrays, "cell_connectivity", dtype=np.int32
    )
    rec_model = None
    if node_coords is None or cell_connectivity is None:
        rec_model = _optional_lazy_artifact_array(arrays, "rec_model", dtype=np.float64)
    return {
        "path": str(path),
        "rm": None,
        "rm_lazy_dataset": rm,
        "rm_dtype": str(np.dtype(getattr(rm, "dtype", np.float64))),
        "metadata": artifact_meta,
        "voxel_shape": tuple(int(v) for v in voxel_shape),
        "node_coords": node_coords,
        "cell_connectivity": cell_connectivity,
        "channel_mask": _optional_lazy_artifact_array(
            arrays, "channel_mask", dtype=bool
        ),
        "measurement_weights": _optional_lazy_artifact_array(
            arrays, "measurement_weights", dtype=np.float64
        ),
        "rec_model": rec_model,
        "greit_y": greit_y,
        "greit_d": greit_d,
        "schema": artifact.schema,
    }


def _load_rm_artifact(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        from pyeidors.inverse.reconstruction_matrix import (
            load_rm_artifact as _load_hdf5_rm_artifact,
        )

        artifact = _load_hdf5_rm_artifact(path)
        rm = artifact.rm
        artifact_meta = dict(artifact.metadata)
        voxel_shape = tuple(
            int(v) for v in artifact.voxel_shape
        ) or _rm_shape_from_meta(meta)
        node_coords = artifact.node_coords
        cell_connectivity = artifact.cell_connectivity
    elif suffix == ".npy":
        from pyeidors.inverse.reconstruction_matrix import (
            load_rm_artifact as _load_legacy_rm_artifact,
        )

        artifact = _load_legacy_rm_artifact(path)
        rm = artifact.rm
        artifact_meta = dict(artifact.metadata)
        voxel_shape = _rm_shape_from_meta(meta)
        node_coords = None
        cell_connectivity = None
    elif suffix == ".npz":
        from pyeidors.inverse.reconstruction_matrix import (
            load_rm_artifact as _load_legacy_rm_artifact,
        )

        artifact = _load_legacy_rm_artifact(path)
        rm = artifact.rm
        artifact_meta = dict(artifact.metadata)
        voxel_shape = tuple(
            int(v) for v in artifact.voxel_shape
        ) or _rm_shape_from_meta(meta)
        node_coords = artifact.node_coords
        cell_connectivity = artifact.cell_connectivity
    else:
        raise ValueError(
            f"Unsupported RM artifact suffix {suffix!r}; expected .h5, .npz, or .npy."
        )
    if rm.ndim != 2 or 0 in rm.shape:
        raise ValueError(f"RM artifact matrix must be non-empty 2D, got {rm.shape}.")
    rm_array = np.asarray(rm)
    if rm_array.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ):
        rm_array = rm_array.astype(np.float64, copy=False)
    return {
        "path": str(path),
        "rm": np.ascontiguousarray(rm_array),
        "metadata": artifact_meta,
        "voxel_shape": tuple(int(v) for v in voxel_shape),
        "node_coords": node_coords,
        "cell_connectivity": cell_connectivity,
        "channel_mask": artifact.channel_mask,
        "measurement_weights": artifact.measurement_weights,
        "rec_model": artifact.rec_model,
        "greit_y": artifact.greit_y,
        "greit_d": artifact.greit_d,
        "schema": artifact.schema,
    }


def _rm_torch_compile_mode(meta: dict[str, Any]) -> str:
    raw = str(
        meta.get(
            "rm_torch_compile",
            meta.get(
                "rm_matmul_compile",
                os.environ.get("EIT_APP_RM_TORCH_COMPILE", "off"),
            ),
        )
        or "off"
    )
    value = raw.strip().lower().replace("-", "_")
    aliases = {
        "1": "force",
        "true": "force",
        "yes": "force",
        "on": "force",
        "force": "force",
        "always": "force",
        "auto": "auto",
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "never": "off",
    }
    return aliases.get(value, "off")


def _rm_artifact_cache_key(
    path: Path, *, device: str, dtype: str, compile_mode: str = "off"
) -> tuple[Any, ...]:
    stat = path.stat()
    return (
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(device).strip().lower(),
        str(dtype).strip().lower(),
        str(compile_mode).strip().lower(),
    )


def _rm_artifact_cache_entry_bytes(entry: dict[str, Any]) -> int:
    total = 0
    for key in (
        "rm",
        "node_coords",
        "cell_connectivity",
        "channel_mask",
        "measurement_weights",
        "rec_model",
        "greit_y",
        "greit_d",
    ):
        value = entry.get(key)
        nbytes = _array_like_nbytes(value)
        if nbytes is not None:
            total += int(nbytes)
    return int(total)


def _rm_artifact_cache_total_bytes() -> int:
    return sum(
        _rm_artifact_cache_entry_bytes(entry) for entry in _RM_ARTIFACT_CACHE.values()
    )


def _store_rm_artifact_process_cache(
    key: tuple[Any, ...],
    artifact: dict[str, Any],
    *,
    max_bytes: int,
) -> bool:
    entry_bytes = _rm_artifact_cache_entry_bytes(artifact)
    artifact["rm_artifact_process_cache_bytes"] = int(entry_bytes)
    artifact["rm_artifact_process_cache_max_bytes"] = int(max_bytes)
    if entry_bytes > int(max(0, max_bytes)):
        artifact["rm_artifact_process_cache_stored"] = False
        artifact["rm_artifact_process_cache_skip_reason"] = "entry_too_large"
        return False
    with _RM_ARTIFACT_CACHE_LOCK:
        _RM_ARTIFACT_CACHE[key] = dict(artifact)
        _RM_ARTIFACT_CACHE.move_to_end(key)
        while len(_RM_ARTIFACT_CACHE) > _RM_ARTIFACT_CACHE_MAX_ITEMS:
            _RM_ARTIFACT_CACHE.popitem(last=False)
        while _rm_artifact_cache_total_bytes() > int(max(0, max_bytes)):
            if not _RM_ARTIFACT_CACHE:
                break
            oldest_key = next(iter(_RM_ARTIFACT_CACHE))
            if oldest_key == key and len(_RM_ARTIFACT_CACHE) == 1:
                break
            _RM_ARTIFACT_CACHE.popitem(last=False)
        stored = key in _RM_ARTIFACT_CACHE
    artifact["rm_artifact_process_cache_stored"] = bool(stored)
    artifact["rm_artifact_process_cache_skip_reason"] = "" if stored else "evicted"
    return bool(stored)


def _rm_artifact_array_for_shape(artifact: dict[str, Any]) -> Any:
    rm = artifact.get("rm")
    if rm is not None:
        return rm
    lazy = artifact.get("rm_lazy_dataset")
    if lazy is not None:
        return lazy
    raise ValueError("RM artifact is missing matrix payload.")


def _rm_streaming_mode(meta: dict[str, Any]) -> str:
    raw = str(
        meta.get(
            "rm_streaming_matmul",
            os.environ.get("EIT_APP_RM_STREAMING_MATMUL", "auto"),
        )
        or "auto"
    )
    value = raw.strip().lower().replace("-", "_")
    aliases = {
        "1": "force",
        "true": "force",
        "yes": "force",
        "on": "force",
        "force": "force",
        "always": "force",
        "auto": "auto",
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "never": "off",
    }
    return aliases.get(value, "auto")


def _rm_streaming_chunk_bytes(meta: dict[str, Any]) -> int:
    return _runtime_bytes_limit(
        meta,
        keys=("rm_streaming_chunk_bytes", "rm_hdf5_streaming_chunk_bytes"),
        env_key="EIT_APP_RM_STREAMING_CHUNK_BYTES",
        default=8 * 1024 * 1024,
    )


def _rm_streaming_rows_per_chunk(
    dataset: Any,
    *,
    rows: int,
    cols: int,
    dtype: np.dtype[Any],
    chunk_bytes: int,
) -> int:
    row_bytes = max(1, int(cols) * max(1, int(dtype.itemsize)))
    budget_rows = max(1, int(max(1, chunk_bytes) // row_bytes))
    budget_rows = min(max(1, int(rows)), budget_rows)
    dataset_chunks = getattr(dataset, "chunks", None)
    if dataset_chunks is None:
        return int(budget_rows)
    try:
        chunk_rows = int(dataset_chunks[0])
        chunk_cols = int(dataset_chunks[1])
    except (TypeError, ValueError, IndexError):
        return int(budget_rows)
    if chunk_rows <= 0 or chunk_cols != int(cols):
        return int(budget_rows)
    if chunk_rows * row_bytes > int(max(1, chunk_bytes)):
        return int(budget_rows)
    chunk_multiple = max(1, budget_rows // chunk_rows)
    return int(min(max(1, int(rows)), chunk_multiple * chunk_rows))


def _iter_rm_row_blocks(
    dataset: Any,
    *,
    rows: int,
    cols: int,
    rows_per_chunk: int,
    dtype: np.dtype[Any],
) -> tuple[str, Any]:
    info = getattr(dataset, "info", None)
    path = getattr(info, "path", None)
    name = getattr(info, "name", None)
    if path is not None and name:
        import h5py

        def _single_open_blocks():
            with h5py.File(path, "r") as handle:
                source = handle["arrays"][str(name)]
                for start in range(0, int(rows), int(rows_per_chunk)):
                    stop = min(start + int(rows_per_chunk), int(rows))
                    yield start, np.asarray(source[start:stop, :], dtype=dtype)

        return "single_open", _single_open_blocks()

    def _fallback_blocks():
        for start in range(0, int(rows), int(rows_per_chunk)):
            stop = min(start + int(rows_per_chunk), int(rows))
            yield start, np.asarray(dataset[start:stop, :], dtype=dtype)

    return "getitem_per_chunk", _fallback_blocks()


def _should_stream_hdf5_rm_artifact(
    artifact: dict[str, Any],
    meta: dict[str, Any],
    *,
    device: str,
    max_cache_bytes: int,
) -> bool:
    return _hdf5_rm_streaming_decision(
        artifact,
        meta,
        device=device,
        max_cache_bytes=max_cache_bytes,
    )[0]


def _hdf5_rm_streaming_decision(
    artifact: dict[str, Any],
    meta: dict[str, Any],
    *,
    device: str,
    max_cache_bytes: int,
) -> tuple[bool, str]:
    if artifact.get("rm_lazy_dataset") is None:
        return False, "not_hdf5_lazy"
    mode = _rm_streaming_mode(meta)
    if mode == "off":
        return False, "disabled"
    matrix_nbytes = _rm_artifact_matrix_nbytes(artifact)
    if mode == "force":
        return True, "forced"
    requested_device = str(device or "auto").strip().lower()
    if requested_device in {"cuda", "gpu", "torch-cuda"}:
        device_resident_max_bytes = _rm_device_resident_max_bytes(meta)
        if matrix_nbytes > int(max(0, device_resident_max_bytes)):
            return True, "cuda_resident_budget_exceeded"
        return False, "cuda_resident_preferred"
    if matrix_nbytes > int(max(0, max_cache_bytes)):
        return True, "process_cache_budget_exceeded"
    return False, "within_process_cache_budget"


def _load_cached_rm_artifact(
    path: Path,
    meta: dict[str, Any],
    *,
    device: str,
    dtype: str,
    expected_n_measurements: int | None = None,
) -> dict[str, Any]:
    compile_mode = _rm_torch_compile_mode(meta)
    key = _rm_artifact_cache_key(
        path,
        device=device,
        dtype=dtype,
        compile_mode=compile_mode,
    )
    max_cache_bytes = _rm_artifact_process_cache_max_bytes(meta)
    with _RM_ARTIFACT_CACHE_LOCK:
        cached = _RM_ARTIFACT_CACHE.get(key)
        if cached is not None:
            cached_bytes = _rm_artifact_cache_entry_bytes(cached)
            if cached_bytes > int(max(0, max_cache_bytes)):
                _RM_ARTIFACT_CACHE.pop(key, None)
                cached = None
        if cached is not None:
            _validate_rm_artifact_measurement_dimension(
                cached,
                path=path,
                expected_n_measurements=expected_n_measurements,
            )
            _RM_ARTIFACT_CACHE.move_to_end(key)
            result = dict(cached)
            result["rm_artifact_cache_hit"] = True
            result["rm_artifact_cache_key"] = key
            result["rm_artifact_process_cache_bytes"] = int(
                _rm_artifact_cache_entry_bytes(cached)
            )
            result["rm_artifact_process_cache_max_bytes"] = int(max_cache_bytes)
            result["rm_artifact_process_cache_stored"] = True
            return result

    if path.suffix.lower() in {".h5", ".hdf5"}:
        lightweight = _load_hdf5_rm_artifact_lightweight(path, meta)
        _validate_rm_artifact_measurement_dimension(
            lightweight,
            path=path,
            expected_n_measurements=expected_n_measurements,
        )
        should_stream, streaming_decision = _hdf5_rm_streaming_decision(
            lightweight,
            meta,
            device=device,
            max_cache_bytes=max_cache_bytes,
        )
        if should_stream:
            lightweight["rm_artifact_cache_hit"] = False
            lightweight["rm_artifact_cache_key"] = key
            lightweight["rm_streaming"] = True
            lightweight["rm_streaming_backend"] = "hdf5_chunked_cpu"
            lightweight["rm_streaming_decision"] = streaming_decision
            lightweight["rm_device_resident_max_bytes"] = int(
                _rm_device_resident_max_bytes(meta)
            )
            lightweight["rm_artifact_process_cache_bytes"] = int(
                _rm_artifact_matrix_nbytes(lightweight)
            )
            lightweight["rm_artifact_process_cache_max_bytes"] = int(max_cache_bytes)
            lightweight["rm_artifact_process_cache_stored"] = False
            lightweight["rm_artifact_process_cache_skip_reason"] = "streaming_hdf5"
            return lightweight

    artifact = _load_rm_artifact(path, meta)
    rm_dtype = np.dtype(_normalize_rm_dtype_name(dtype))
    artifact["rm"] = np.ascontiguousarray(artifact["rm"], dtype=rm_dtype)
    _validate_rm_artifact_measurement_dimension(
        artifact,
        path=path,
        expected_n_measurements=expected_n_measurements,
    )
    from pyeidors.perf.gpu_kernels import prepare_rm_matmul

    artifact["rm_handle"] = prepare_rm_matmul(
        artifact["rm"],
        device=device,
        dtype=dtype,
        cache_key=str(path),
        compile_mode=compile_mode,
    )
    _store_rm_artifact_process_cache(key, artifact, max_bytes=max_cache_bytes)
    result = dict(artifact)
    result["rm_artifact_cache_hit"] = False
    result["rm_artifact_cache_key"] = key
    result["rm_streaming_decision"] = "device_resident_prepared"
    result["rm_device_resident_max_bytes"] = int(_rm_device_resident_max_bytes(meta))
    return result


def _voxel_bounds_from_meta(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_bounds = meta.get("rm_voxel_bounds", meta.get("inverse_bounds"))
    try:
        bounds = np.asarray(raw_bounds, dtype=np.float64)
        if (
            bounds.shape == (2, 3)
            and all_finite_values(bounds)
            and np.all(bounds[1] > bounds[0])
        ):
            return bounds[0], bounds[1]
    except (TypeError, ValueError):
        pass
    radius = float(meta.get("radius", 1.0) or 1.0)
    height = float(
        meta.get("mesh_height", meta.get("height", 2.0 * radius)) or (2.0 * radius)
    )
    lower = np.asarray([-radius, -radius, -0.5 * height], dtype=np.float64)
    upper = np.asarray([radius, radius, 0.5 * height], dtype=np.float64)
    return lower, upper


def _voxel_grid_geometry(
    shape: tuple[int, ...],
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(shape) != 3 or any(int(v) <= 0 for v in shape):
        return None
    nx, ny, nz = (int(v) for v in shape)
    lower, upper = _voxel_bounds_from_meta(meta)
    axes = [np.linspace(lower[axis], upper[axis], shape[axis] + 1) for axis in range(3)]
    coords = np.asarray(
        [[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]],
        dtype=np.float64,
    )

    def node(ix: int, iy: int, iz: int) -> int:
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    cells: list[list[int]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                cells.append(
                    [
                        node(ix, iy, iz),
                        node(ix + 1, iy, iz),
                        node(ix + 1, iy + 1, iz),
                        node(ix, iy + 1, iz),
                        node(ix, iy, iz + 1),
                        node(ix + 1, iy, iz + 1),
                        node(ix + 1, iy + 1, iz + 1),
                        node(ix, iy + 1, iz + 1),
                    ]
                )
    return coords, np.asarray(cells, dtype=np.int32)


def _center_cloud_hexa_geometry(
    centers: np.ndarray,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    centers_raw = np.asarray(centers)
    coord_dtype = (
        np.float32
        if np.issubdtype(centers_raw.dtype, np.floating)
        and centers_raw.dtype.itemsize <= 4
        else np.float64
    )
    centers = np.ascontiguousarray(centers_raw, dtype=coord_dtype)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        return None
    if not all_finite_values(centers):
        return None
    axis_spacing: list[float] = []
    for axis in range(3):
        unique = np.unique(np.round(centers[:, axis], decimals=12))
        diffs = np.diff(unique)
        if diffs.size:
            axis_spacing.append(float(np.median(diffs)))
        else:
            axis_spacing.append(float("nan"))
    if not any(np.isfinite(value) and value > 0.0 for value in axis_spacing):
        radius = float(meta.get("radius", 1.0) or 1.0)
        fallback = radius / max(round(centers.shape[0] ** (1.0 / 3.0)), 1)
        axis_spacing = [fallback, fallback, fallback]
    finite_spacing = [
        value for value in axis_spacing if np.isfinite(value) and value > 0.0
    ]
    fallback_spacing = float(min(finite_spacing)) if finite_spacing else 1.0
    half_axes = np.asarray(
        [
            0.45 * (value if np.isfinite(value) and value > 0.0 else fallback_spacing)
            for value in axis_spacing
        ],
        dtype=coord_dtype,
    )
    half_axes = np.maximum(half_axes, np.finfo(coord_dtype).eps)
    offsets = np.asarray(
        [
            [-half_axes[0], -half_axes[1], -half_axes[2]],
            [half_axes[0], -half_axes[1], -half_axes[2]],
            [half_axes[0], half_axes[1], -half_axes[2]],
            [-half_axes[0], half_axes[1], -half_axes[2]],
            [-half_axes[0], -half_axes[1], half_axes[2]],
            [half_axes[0], -half_axes[1], half_axes[2]],
            [half_axes[0], half_axes[1], half_axes[2]],
            [-half_axes[0], half_axes[1], half_axes[2]],
        ],
        dtype=coord_dtype,
    )
    coords = np.empty((centers.shape[0] * 8, 3), dtype=coord_dtype)
    for offset_idx, offset in enumerate(offsets):
        np.add(centers, offset, out=coords[offset_idx::8])
    cells = np.arange(centers.shape[0] * 8, dtype=np.int32).reshape(-1, 8)
    return coords, cells


def _center_cloud_quad_geometry(
    centers: np.ndarray,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    centers_raw = np.asarray(centers)
    coord_dtype = (
        np.float32
        if np.issubdtype(centers_raw.dtype, np.floating)
        and centers_raw.dtype.itemsize <= 4
        else np.float64
    )
    centers = np.ascontiguousarray(centers_raw, dtype=coord_dtype)
    if centers.ndim != 2 or centers.shape[1] < 2 or centers.shape[0] == 0:
        return None
    centers_xy = centers[:, :2]
    if not all_finite_values(centers_xy):
        return None
    axis_spacing: list[float] = []
    for axis in range(2):
        unique = np.unique(np.round(centers_xy[:, axis], decimals=12))
        diffs = np.diff(unique)
        if diffs.size:
            axis_spacing.append(float(np.median(diffs)))
        else:
            axis_spacing.append(float("nan"))
    if not any(np.isfinite(value) and value > 0.0 for value in axis_spacing):
        radius = float(meta.get("radius", 1.0) or 1.0)
        fallback = radius / max(round(centers_xy.shape[0] ** 0.5), 1)
        axis_spacing = [fallback, fallback]
    finite_spacing = [
        value for value in axis_spacing if np.isfinite(value) and value > 0.0
    ]
    fallback_spacing = float(min(finite_spacing)) if finite_spacing else 1.0
    half_axes = np.asarray(
        [
            0.45 * (value if np.isfinite(value) and value > 0.0 else fallback_spacing)
            for value in axis_spacing
        ],
        dtype=coord_dtype,
    )
    half_axes = np.maximum(half_axes, np.finfo(coord_dtype).eps)
    offsets = np.asarray(
        [
            [-half_axes[0], -half_axes[1]],
            [half_axes[0], -half_axes[1]],
            [half_axes[0], half_axes[1]],
            [-half_axes[0], half_axes[1]],
        ],
        dtype=coord_dtype,
    )
    coords = np.empty((centers_xy.shape[0] * 4, 2), dtype=coord_dtype)
    for offset_idx, offset in enumerate(offsets):
        np.add(centers_xy, offset, out=coords[offset_idx::4])
    cells = np.arange(centers_xy.shape[0] * 4, dtype=np.int32).reshape(-1, 4)
    return coords, cells


def _greit_rec_model_geometry(
    rec_model: Any,
    *,
    n_parameters: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    raw_input = np.asarray(rec_model)
    raw_dtype = (
        np.float32
        if np.issubdtype(raw_input.dtype, np.floating) and raw_input.dtype.itemsize <= 4
        else np.float64
    )
    raw = np.ascontiguousarray(raw_input, dtype=raw_dtype)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.shape[0] == 0:
        return None
    if raw.shape[0] == int(n_parameters):
        centers = raw
    elif int(n_parameters) > 0 and raw.shape[0] % int(n_parameters) == 0:
        centers = raw.reshape(int(n_parameters), -1, raw.shape[1]).mean(axis=1)
    else:
        return None
    if int(meta.get("mesh_dimension", meta.get("dim", 3)) or 3) == 2:
        return _center_cloud_quad_geometry(centers, meta)
    if centers.shape[1] == 2:
        padded = np.zeros((centers.shape[0], 3), dtype=centers.dtype)
        padded[:, :2] = centers
        centers = padded
    elif centers.shape[1] > 3:
        centers = centers[:, :3]
    return _center_cloud_hexa_geometry(centers, meta)


def _rm_artifact_geometry(
    artifact: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    coords = artifact.get("node_coords")
    cells = artifact.get("cell_connectivity")
    if coords is not None and cells is not None:
        return _display_node_coords_array(coords), _display_cell_connectivity_array(
            cells
        )
    rec_model = artifact.get("rec_model")
    if rec_model is not None:
        generated_from_rec = _greit_rec_model_geometry(
            rec_model,
            n_parameters=int(_rm_artifact_matrix_shape(artifact)[0]),
            meta=meta,
        )
        if generated_from_rec is not None:
            meta["rm_geometry_source"] = (
                "greit_rec_model_centers_2d"
                if generated_from_rec[0].shape[1] == 2
                else "greit_rec_model_centers"
            )
            return generated_from_rec
    generated = _voxel_grid_geometry(tuple(artifact.get("voxel_shape", ())), meta)
    if generated is not None:
        meta["rm_geometry_source"] = "voxel_shape_full_grid"
        return generated
    raise ValueError(
        "RM artifact hot path requires node/cell geometry or a 3D voxel_shape."
    )


def _greit_artifact_unavailable_reason(
    artifact: dict[str, Any],
    meta: dict[str, Any],
    *,
    expected_n_measurements: int,
) -> str:
    route = _simulation_inverse_route(meta)
    if route not in _GREIT_REGISTRY_ROUTES:
        return ""
    artifact_meta = dict(artifact.get("metadata", {}) or {})
    if bool(artifact_meta.get("fixture_only", False)):
        return (
            "GREIT artifact is a deterministic fixture, not an official EIDORS "
            f"parity artifact; refusing production {route}."
        )
    if not bool(artifact_meta.get("eidors_parity", False)):
        return (
            "GREIT artifact metadata does not declare eidors_parity=true; refusing "
            f"production {route}."
        )
    expected_signature = str(meta.get("greit_registry_signature") or "").strip()
    if expected_signature:
        actual_signature = str(
            artifact_meta.get("greit_registry_signature") or ""
        ).strip()
        if actual_signature != expected_signature:
            return (
                "GREIT artifact registry signature mismatch; refusing production "
                f"{route}."
            )
    rm_shape = _rm_artifact_matrix_shape(artifact)
    if int(rm_shape[1]) != int(expected_n_measurements):
        return _rm_artifact_measurement_mismatch_message(
            path=Path(str(artifact.get("path", "")) or "<artifact>"),
            artifact_columns=int(rm_shape[1]),
            request_measurements=int(expected_n_measurements),
        )
    return ""


def _greit_training_space_fit(
    artifact: dict[str, Any],
    delta_conductivity: np.ndarray,
    *,
    n_measurements: int,
) -> np.ndarray | None:
    y = artifact.get("greit_y")
    d = artifact.get("greit_d")
    if y is None or d is None:
        return None
    y_matrix = np.asarray(y, dtype=np.float64)
    d_matrix = np.asarray(d, dtype=np.float64)
    delta = np.asarray(delta_conductivity, dtype=np.float64).reshape(-1)
    if (
        y_matrix.ndim != 2
        or d_matrix.ndim != 2
        or y_matrix.shape[0] != int(n_measurements)
        or d_matrix.shape[0] != delta.size
        or y_matrix.shape[1] != d_matrix.shape[1]
    ):
        return None
    try:
        coeff = np.linalg.pinv(d_matrix) @ delta
        fitted = np.asarray(y_matrix @ coeff, dtype=np.float64).reshape(-1)
    except np.linalg.LinAlgError:
        return None
    if fitted.size != int(n_measurements) or not all_finite_values(fitted):
        return None
    return fitted


def _stream_hdf5_rm_matmul(
    artifact: dict[str, Any],
    delta_v: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
    device_requested: str,
    dtype: str,
    meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    dataset = artifact.get("rm_lazy_dataset")
    if dataset is None:
        raise ValueError("Streaming RM matmul requires an HDF5 lazy RM dataset.")
    np_dtype = np.dtype(_normalize_rm_dtype_name(dtype))
    rows, cols = _rm_artifact_matrix_shape(artifact)

    from pyeidors.data.channels import apply_measurement_contract_to_vector

    measurement, contract = apply_measurement_contract_to_vector(
        delta_v,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    measurement = np.asarray(measurement, dtype=np_dtype).reshape(-1)
    if int(measurement.size) != int(cols):
        raise ValueError(
            f"delta_v measurement dimension {measurement.size} does not match "
            f"RM columns {cols}."
        )
    chunk_bytes = _rm_streaming_chunk_bytes(meta)
    rows_per_chunk = _rm_streaming_rows_per_chunk(
        dataset,
        rows=int(rows),
        cols=int(cols),
        dtype=np_dtype,
        chunk_bytes=chunk_bytes,
    )
    dataset_chunks = getattr(dataset, "chunks", None)
    values = np.empty(int(rows), dtype=np_dtype)
    chunks = 0
    file_open_mode, row_blocks = _iter_rm_row_blocks(
        dataset,
        rows=int(rows),
        cols=int(cols),
        rows_per_chunk=int(rows_per_chunk),
        dtype=np_dtype,
    )
    for start, block in row_blocks:
        stop = min(int(start) + int(rows_per_chunk), int(rows))
        if block.ndim != 2 or block.shape[1] != int(cols):
            raise ValueError(
                f"RM HDF5 chunk has invalid shape {block.shape}; expected (*, {cols})."
            )
        values[start:stop] = block @ measurement
        chunks += 1
    if not all_finite_values(values):
        raise FloatingPointError("Streaming RM matmul produced non-finite values.")
    metadata = {
        "backend": "hdf5_chunked",
        "device_requested": str(device_requested or "auto"),
        "device_effective": "cpu",
        "fallback_reason": "streaming_hdf5_rm",
        "batched": False,
        "n_frames": 1,
        "rm_shape": (int(rows), int(cols)),
        "delta_v_shape": (1, int(cols)),
        "output_shape": tuple(int(v) for v in values.shape),
        "rm_dtype": _normalize_rm_dtype_name(dtype),
        "rm_persistent": False,
        "rm_tensor_reused": False,
        "rm_prepare_mode": "streaming_hdf5",
        "rm_matrix_resident": "hdf5",
        "rm_cache_key": str(artifact.get("path", "")),
        "host_device_transfer": "none",
        "online_hot_path": "rm_hdf5_streaming_matmul",
        "rm_streaming": True,
        "rm_streaming_chunks": int(chunks),
        "rm_streaming_chunk_bytes": int(chunk_bytes),
        "rm_streaming_rows_per_chunk": int(rows_per_chunk),
        "rm_hdf5_file_open_mode": str(file_open_mode),
        "rm_hdf5_dataset_chunks": (
            tuple(int(v) for v in dataset_chunks) if dataset_chunks else None
        ),
        "measurement_weight_kind": str(getattr(contract, "weight_kind", "")),
        "bad_channel_count": int(getattr(contract, "bad_channel_count", 0)),
    }
    return values, metadata


def _try_run_cached_rm_request(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult | None:
    path = _resolve_rm_artifact_path(runtime.meta)
    if path is None:
        return None

    if progress_cb is not None:
        progress_cb("Loading cached reconstruction matrix...")

    device = str(runtime.meta.get("rm_device", runtime.meta.get("device", "auto")))
    dtype = _rm_dtype_name_from_meta(runtime.meta)
    runtime.meta["rm_dtype"] = dtype
    runtime.meta["rm_matmul_dtype"] = dtype
    np_dtype = np.dtype(dtype)
    ref_vec = np.asarray(
        req.reference_frame.to_measurement_vector(req.use_part), dtype=np_dtype
    )
    tgt_vec = np.asarray(
        req.target_frame.to_measurement_vector(req.use_part), dtype=np_dtype
    )
    artifact = _load_cached_rm_artifact(
        path,
        runtime.meta,
        device=device,
        dtype=dtype,
        expected_n_measurements=int(ref_vec.size),
    )
    greit_unavailable = _greit_artifact_unavailable_reason(
        artifact,
        runtime.meta,
        expected_n_measurements=int(ref_vec.size),
    )
    if greit_unavailable:
        runtime.meta["rm_artifact_unavailable_reason"] = greit_unavailable
        return _missing_rm_artifact_result(
            runtime,
            emit=progress_cb if progress_cb is not None else (lambda _msg: None),
        )
    node_coords, cell_connectivity = _rm_artifact_geometry(artifact, runtime.meta)
    difference_mode = str(runtime.meta.get("difference_mode", "raw"))
    difference_orientation = str(
        runtime.meta.get("difference_orientation", "target_minus_reference")
    )
    dv = np.asarray(
        build_difference_vector(
            tgt_vec,
            ref_vec,
            mode=difference_mode,
            orientation=difference_orientation,
        ),
        dtype=np_dtype,
    )
    if bool(artifact.get("rm_streaming", False)):
        delta_values, rm_matmul_metadata = _stream_hdf5_rm_matmul(
            artifact,
            dv,
            channel_mask=artifact.get("channel_mask"),
            measurement_weights=artifact.get("measurement_weights"),
            device_requested=device,
            dtype=dtype,
            meta=runtime.meta,
        )
    else:
        from pyeidors.inverse.reconstruction_matrix import reconstruct_difference_batch

        rm_result = reconstruct_difference_batch(
            artifact.get("rm_handle", artifact["rm"]),
            dv,
            normalize=False,
            channel_mask=artifact.get("channel_mask"),
            measurement_weights=artifact.get("measurement_weights"),
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
        delta_values = rm_result.values
        rm_matmul_metadata = dict(rm_result.metadata)
    delta_conductivity = np.asarray(delta_values, dtype=np_dtype).reshape(-1)
    output_display_mode = str(runtime.meta.get("rm_output_display_mode", "")).lower()
    if output_display_mode == "absolute_sigma":
        conductivity = delta_conductivity + _scalar_for_array_dtype(
            runtime.background_sigma,
            np_dtype,
        )
    else:
        conductivity = delta_conductivity

    # Simulated boundary-voltage diff = J @ delta_sigma.  Auto-built RM
    # artifacts persist an optional fit Jacobian and also keep a small
    # process cache keyed by the RM semantic signature, so warm runs keep
    # the GUI voltage-fit overlay without carrying J in result metadata.
    simulated_dv: np.ndarray | None = None
    artifact_meta = dict(artifact.get("metadata", {}) or {})
    rm_signature = runtime.meta.get("rm_signature") or artifact_meta.get("rm_signature")
    inmem_jacobian = _restore_rm_fit_jacobian(
        runtime,
        path=path,
        signature=rm_signature,
        expected_shape=(int(dv.size), int(delta_conductivity.size)),
    )
    if inmem_jacobian is not None:
        try:
            jac = np.asarray(inmem_jacobian, dtype=np_dtype)
            if (
                jac.ndim == 2
                and jac.shape[1] == delta_conductivity.size
                and jac.shape[0] == int(dv.size)
            ):
                simulated_dv = np.asarray(jac @ delta_conductivity, dtype=np_dtype)
                if not all_finite_values(simulated_dv):
                    simulated_dv = None
        except Exception:
            simulated_dv = None
    if (
        simulated_dv is None
        and _simulation_inverse_route(runtime.meta) in _GREIT_REGISTRY_ROUTES
    ):
        simulated_dv = _greit_training_space_fit(
            artifact,
            delta_conductivity,
            n_measurements=int(dv.size),
        )
        if simulated_dv is not None:
            runtime.meta["rm_fit_source"] = "greit_training_space_projection"
    result_meta = _public_runtime_metadata(runtime.meta)
    result_meta.update(
        {
            "n_elec": int(runtime.meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "single_step_operator_space": "rm",
            "online_hot_path": str(
                rm_matmul_metadata.get("online_hot_path", "rm_matmul")
            ),
            "rm_artifact_path": str(path),
            "rm_shape": tuple(int(v) for v in _rm_artifact_matrix_shape(artifact)),
            "rm_nbytes": int(_rm_artifact_matrix_nbytes(artifact)),
            "rm_streaming": bool(artifact.get("rm_streaming", False)),
            "rm_streaming_decision": str(artifact.get("rm_streaming_decision", "")),
            "rm_device_resident_max_bytes": int(
                artifact.get("rm_device_resident_max_bytes", 0) or 0
            ),
            "rm_artifact_process_cache_bytes": int(
                artifact.get("rm_artifact_process_cache_bytes", 0) or 0
            ),
            "rm_artifact_process_cache_max_bytes": int(
                artifact.get("rm_artifact_process_cache_max_bytes", 0) or 0
            ),
            "rm_artifact_process_cache_stored": bool(
                artifact.get("rm_artifact_process_cache_stored", False)
            ),
            "rm_artifact_process_cache_skip_reason": str(
                artifact.get("rm_artifact_process_cache_skip_reason", "")
            ),
            "rm_voxel_shape": tuple(int(v) for v in artifact.get("voxel_shape", ())),
            "rm_dtype": str(rm_matmul_metadata.get("rm_dtype", dtype)),
            "rm_matmul_compile_mode": str(
                rm_matmul_metadata.get("rm_matmul_compile_mode", "")
            ),
            "rm_matmul_compile_status": str(
                rm_matmul_metadata.get("rm_matmul_compile_status", "")
            ),
            "rm_matmul_compiled": bool(
                rm_matmul_metadata.get("rm_matmul_compiled", False)
            ),
            "rm_artifact_cache_hit": bool(artifact.get("rm_artifact_cache_hit", False)),
            "rm_output_display_mode": output_display_mode or "delta_sigma",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "solver_diagnostics": {
                "path": "single_step_cached_rm",
                "strict_solver_backend_effective": "rm",
                "runtime": {
                    "online_hot_path": str(
                        rm_matmul_metadata.get("online_hot_path", "rm_matmul")
                    ),
                    "single_step_operator_space": "rm",
                    "forward_solve_count": 0,
                    "adjoint_solve_count": 0,
                    "jacobian_rebuild_count": 0,
                    "ksp_solve_count": 0,
                    "device_requested": str(
                        rm_matmul_metadata.get("device_requested", device)
                    ),
                    "device_effective": str(
                        rm_matmul_metadata.get("device_effective", "")
                    ),
                    "rm_dtype": str(rm_matmul_metadata.get("rm_dtype", dtype)),
                    "rm_matmul_compile_mode": str(
                        rm_matmul_metadata.get("rm_matmul_compile_mode", "")
                    ),
                    "rm_matmul_compile_status": str(
                        rm_matmul_metadata.get("rm_matmul_compile_status", "")
                    ),
                    "rm_matmul_compiled": bool(
                        rm_matmul_metadata.get("rm_matmul_compiled", False)
                    ),
                    "rm_persistent": bool(
                        rm_matmul_metadata.get("rm_persistent", False)
                    ),
                    "rm_tensor_reused": bool(
                        rm_matmul_metadata.get("rm_tensor_reused", False)
                    ),
                    "rm_prepare_mode": str(
                        rm_matmul_metadata.get("rm_prepare_mode", "")
                    ),
                    "host_device_transfer": str(
                        rm_matmul_metadata.get("host_device_transfer", "")
                    ),
                    "rm_streaming": bool(artifact.get("rm_streaming", False)),
                    "rm_streaming_decision": str(
                        artifact.get("rm_streaming_decision", "")
                    ),
                    "rm_device_resident_max_bytes": int(
                        artifact.get("rm_device_resident_max_bytes", 0) or 0
                    ),
                    "rm_artifact_cache_hit": bool(
                        artifact.get("rm_artifact_cache_hit", False)
                    ),
                    "rm_shape": tuple(
                        int(v) for v in _rm_artifact_matrix_shape(artifact)
                    ),
                    "rm_nbytes": int(_rm_artifact_matrix_nbytes(artifact)),
                    "rm_artifact_process_cache_bytes": int(
                        artifact.get("rm_artifact_process_cache_bytes", 0) or 0
                    ),
                    "rm_artifact_process_cache_max_bytes": int(
                        artifact.get("rm_artifact_process_cache_max_bytes", 0) or 0
                    ),
                    "rm_artifact_process_cache_stored": bool(
                        artifact.get("rm_artifact_process_cache_stored", False)
                    ),
                    "rm_artifact_process_cache_skip_reason": str(
                        artifact.get("rm_artifact_process_cache_skip_reason", "")
                    ),
                    "rm_artifact_path": str(path),
                },
                "cache_lookups": {
                    "rm_artifact": {
                        "hit": True,
                        "layer": (
                            "process"
                            if bool(artifact.get("rm_artifact_cache_hit", False))
                            else "artifact"
                        ),
                        "process_cache_hit": bool(
                            artifact.get("rm_artifact_cache_hit", False)
                        ),
                        "artifact": "reconstruction_matrix",
                        "key": str(path),
                    }
                },
                "rm_metadata": dict(artifact.get("metadata", {}) or {}),
                "rm_matmul": dict(rm_matmul_metadata),
            },
        }
    )
    if progress_cb is not None:
        progress_cb("Reconstruction complete")
    return ReconstructionResult(
        conductivity=conductivity,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        measured=dv,
        simulated=simulated_dv,
        metadata=result_meta,
    )


def _prepare_single_step_cached_runtime(
    req: ReconstructionRequest,
) -> _SingleStepCachedRuntimeConfig:
    meta = dict(req.metadata)
    meta.setdefault("n_elec", 16)
    meta.setdefault("n_rings", 1)
    meta.setdefault("electrode_layout", "ring_major")
    meta.setdefault("measurement_protocol", "eidors_full_3d")
    meta.setdefault("custom_stim_matrix", None)
    meta.setdefault("custom_meas_matrices", None)
    meta.setdefault("stim_pattern", "{ad}")
    meta.setdefault("meas_pattern", "{ad}")
    meta.setdefault("rotate_meas", True)
    meta.setdefault("use_meas_current", False)
    meta.setdefault("use_meas_current_next", 0)
    meta.setdefault("stim_direction", "ccw")
    meta.setdefault("meas_direction", "ccw")
    meta.setdefault("stim_first_positive", False)
    meta.setdefault("radius", 1.0)
    meta.setdefault("geometry_scale_to_m", 1.0)
    meta.setdefault("electrode_length_m_override", None)
    meta.setdefault("electrode_area_m2_override", None)
    meta.setdefault("electrode_coverage", 0.5)
    meta.setdefault("mesh_dir", "eit_meshes")
    meta.setdefault("difference_mode", "raw")
    meta.setdefault("difference_orientation", "target_minus_reference")
    meta.setdefault("step_size_calib", True)
    meta.setdefault("step_size_min", 1.0e-6)
    meta.setdefault("step_size_max", 1.0)
    meta.setdefault("step_size_maxiter", 64)
    meta["sigma_floor"] = _single_step_sigma_floor(meta)
    meta.setdefault("solver_mode", "auto")
    meta.setdefault("linear_solver", "auto")
    meta.setdefault("preconditioner", "auto")
    meta.setdefault("fast_linear_path", "auto")
    meta.setdefault("forward_solver_preset", "auto")
    meta.setdefault("forward_mat_solve", "auto")
    meta.setdefault("petsc_device", "auto")
    meta.setdefault("device", "auto")
    rm_dtype_name = _rm_dtype_name_from_meta(meta)
    meta["rm_dtype"] = rm_dtype_name
    meta["rm_matmul_dtype"] = rm_dtype_name
    meta.setdefault("jacobian_representation", "auto")
    meta.setdefault("linearized_solver_strategy", "auto")
    meta.setdefault("linearized_maxiter", 0)
    meta.setdefault("lazy_preconditioner_mode", "auto")
    meta.setdefault("lazy_diag_batch_max_measurements", 512)
    meta.setdefault("forward_backend", "dolfinx")
    meta.setdefault("mesh_family", "tetra")
    meta.setdefault("geometry_version", "geomv2")
    meta.setdefault("potential_order", 1)
    meta.setdefault("acceleration_profile", "default")
    meta.setdefault(
        "single_step_signature_schema_version",
        _SINGLE_STEP_SIGNATURE_SCHEMA_VERSION,
    )
    meta.setdefault("single_step_jacobian_calculator", _SINGLE_STEP_JACOBIAN_CALCULATOR)
    meta.setdefault(
        "single_step_jacobian_math_convention",
        _SINGLE_STEP_JACOBIAN_MATH_CONVENTION,
    )
    meta.setdefault(
        "single_step_projection_math_convention",
        _SINGLE_STEP_PROJECTION_MATH_CONVENTION,
    )
    meta.setdefault(
        "single_step_operator_math_convention",
        _SINGLE_STEP_OPERATOR_MATH_CONVENTION,
    )
    meta.setdefault(
        "single_step_algorithm_version",
        _SINGLE_STEP_CACHED_ALGORITHM_VERSION,
    )
    meta.setdefault(
        "one_step_rm_signature_schema_version",
        _ONE_STEP_RM_SIGNATURE_SCHEMA_VERSION,
    )
    meta.setdefault(
        "one_step_rm_jacobian_build_convention",
        _ONE_STEP_RM_JACOBIAN_BUILD_CONVENTION,
    )
    meta.setdefault(
        "one_step_rm_prior_math_convention",
        _ONE_STEP_RM_PRIOR_MATH_CONVENTION,
    )
    meta.setdefault("one_step_rm_algorithm_version", _ONE_STEP_RM_ALGORITHM_VERSION)
    meta.setdefault("one_step_rm_content_contract", _ONE_STEP_RM_CONTENT_CONTRACT)
    mesh_dim = int(meta.get("mesh_dimension", req.mesh_dimension))
    meta["mesh_dimension"] = mesh_dim
    meta["drive_mode"] = _resolve_drive_mode(meta, mesh_dim=mesh_dim)
    meta["drive_value"] = _resolve_drive_value(meta)
    runtime_options = _resolve_reconstruction_runtime(meta, mesh_dim=mesh_dim)
    meta.update(runtime_options)
    jac_repr = (
        str(meta.get("jacobian_representation", "auto") or "auto").strip().lower()
    )
    jac_repr = jac_repr.replace("_", "-")
    if jac_repr in {"", "auto"}:
        measurement_count = _request_measurement_count(req)
        use_linearized_auto = (
            mesh_dim == 3
            and meta["solver_mode"] == "fast"
            and 0 < measurement_count <= LINEARIZED_SINGLE_STEP_AUTO_MAX_MEASUREMENTS
        )
        jac_repr = "linearized" if use_linearized_auto else "dense"
        meta["jacobian_representation_reason"] = (
            "auto_small_3d_fast" if use_linearized_auto else "auto_dense_large_or_non3d"
        )
    elif jac_repr in {"jacobian-linearization", "operator"}:
        jac_repr = "linearized"
        meta["jacobian_representation_reason"] = "explicit_linearized"
    elif jac_repr in {"lazy", "lazy-adjoint", "matrix-free", "matrixfree"}:
        jac_repr = "lazy"
        meta["jacobian_representation_reason"] = "explicit_lazy"
    elif jac_repr not in {"dense", "linearized", "lazy"}:
        raise ValueError(
            "jacobian_representation must be auto|dense|linearized|lazy, "
            f"got {meta.get('jacobian_representation')!r}."
        )
    else:
        meta["jacobian_representation_reason"] = f"explicit_{jac_repr}"
    if _is_auto_built_one_step_rm_route(meta):
        meta["rm_build_jacobian_representation"] = "dense"
        meta.setdefault("single_step_context_cache_scope", "process")
        if jac_repr != "dense":
            meta["rm_build_jacobian_representation_requested"] = jac_repr
            meta["rm_build_jacobian_representation_reason"] = (
                f"rm_auto_build_requires_dense_from_{jac_repr}"
            )
            meta["jacobian_representation_reason"] = (
                f"rm_auto_build_requires_dense_from_{jac_repr}"
            )
            jac_repr = "dense"
    meta["jacobian_representation"] = jac_repr
    radius = float(meta.get("radius", 1.0))
    mesh_size_for_runtime = meta.get("mesh_size")
    if _is_auto_built_one_step_rm_route(meta):
        try:
            requested_mesh_size = float(meta.get("mesh_size", req.mesh_refinement))
        except (TypeError, ValueError):
            requested_mesh_size = float(req.mesh_refinement)
        inverse_mesh_size = meta.get("rm_inverse_mesh_size")
        if inverse_mesh_size in (None, ""):
            inverse_mesh_size = default_rm_inverse_mesh_size(
                requested_mesh_size,
                radius,
                mesh_dimension=int(meta.get("mesh_dimension", req.mesh_dimension)),
            )
            meta["rm_inverse_mesh_size"] = inverse_mesh_size
        mesh_size_for_runtime = inverse_mesh_size
    meta["effective_inverse_mesh_size"] = mesh_size_for_runtime
    refinement = _compute_effective_refinement(
        radius,
        req.mesh_refinement,
        mesh_size=mesh_size_for_runtime,
    )
    raw_lam = meta.get("difference_lambda")
    try:
        lam = float(raw_lam)
    except (TypeError, ValueError):
        lam = float("nan")
    if not np.isfinite(lam) or lam <= 0.0:
        try:
            lam = float(req.regularization_alpha)
        except (TypeError, ValueError):
            lam = float("nan")
    if not np.isfinite(lam) or lam <= 0.0:
        lam = 1.0e-2
    meta["difference_lambda"] = lam
    background_sigma = _background_scalar_from_meta(meta)
    contact_impedance = _contact_impedance_scalar(meta.get("contact_impedance", 0.01))
    mesh_height = float(meta.get("mesh_height", meta.get("height", 1.0)))
    electrode_height_ratio = float(meta.get("electrode_height_ratio", 0.2))
    z_center = float(meta.get("z_center", 0.0))
    cache_key = (
        _single_step_semantic_signature(meta),
        str(meta.get("single_step_algorithm_version")),
        int(meta["n_elec"]),
        int(meta.get("n_rings", 1)),
        mesh_dim,
        refinement,
        radius,
        mesh_height,
        electrode_height_ratio,
        repr(meta.get("electrode_level_fractions", (0.25, 0.75))),
        z_center,
        lam,
        background_sigma,
        contact_impedance,
        float(meta.get("geometry_scale_to_m", 1.0)),
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        repr(meta.get("electrode_area_m2_override")),
        str(meta.get("difference_mode", "raw")),
        str(meta.get("difference_orientation", "target_minus_reference")),
        str(meta.get("stim_pattern", "{ad}")),
        str(meta.get("meas_pattern", "{ad}")),
        str(meta.get("electrode_layout", "ring_major")),
        str(meta.get("measurement_protocol", "eidors_full_3d")),
        repr(meta.get("custom_stim_matrix")),
        repr(meta.get("custom_meas_matrices")),
        bool(meta.get("rotate_meas", True)),
        bool(meta.get("use_meas_current", False)),
        int(meta.get("use_meas_current_next", 0)),
        str(meta.get("stim_direction", "ccw")),
        str(meta.get("meas_direction", "ccw")),
        bool(meta.get("stim_first_positive", False)),
        str(meta.get("drive_mode", "total_current")),
        float(meta.get("drive_value", 1.0e-5)),
        str(meta.get("mesh_dir", "eit_meshes")),
        repr(meta.get("rm_inverse_mesh_size")),
        _single_step_context_cache_scope(meta),
        str(req.use_part),
        str(meta.get("solver_mode", "auto")),
        str(meta.get("linear_solver", "auto")),
        str(meta.get("preconditioner", "auto")),
        str(meta.get("fast_linear_path", "auto")),
        str(meta.get("jacobian_representation", "dense")),
        str(meta.get("linearized_solver_strategy", "auto")),
        int(meta.get("linearized_maxiter", 0)),
        str(meta.get("lazy_preconditioner_mode", "auto")),
        int(meta.get("lazy_diag_batch_max_measurements", 512)),
        str(meta.get("forward_solver_preset", "auto")),
        str(meta.get("forward_mat_solve", "auto")),
        str(meta.get("petsc_device", "auto")),
        str(meta.get("device", "auto")),
        str(meta.get("rm_dtype", "float64")),
        str(meta.get("forward_backend", "dolfinx")),
        str(meta.get("mesh_family", "tetra")),
        str(meta.get("geometry_version", "geomv2")),
        int(meta.get("potential_order", 1)),
        str(meta.get("acceleration_profile", "default")),
    )
    return _SingleStepCachedRuntimeConfig(
        meta=meta,
        mesh_dim=mesh_dim,
        refinement=refinement,
        lam=lam,
        background_sigma=background_sigma,
        contact_impedance=contact_impedance,
        mesh_height=mesh_height,
        electrode_height_ratio=electrode_height_ratio,
        z_center=z_center,
        cache_key=cache_key,
    )


def get_single_step_cached_cache_key(req: ReconstructionRequest) -> tuple[Any, ...]:
    """Return the effective cache key for a single-step cached request."""
    return _prepare_single_step_cached_runtime(req).cache_key


def _has_explicit_rm_artifact_meta(meta: dict[str, Any]) -> bool:
    return any(_meta_value_present(meta.get(key)) for key in _RM_ARTIFACT_META_KEYS)


def _can_dispatch_single_step_cached(
    req: ReconstructionRequest,
    *,
    method_lc: str,
    runtime_path: str,
) -> bool:
    if method_lc != "gn-difference" or runtime_path != "single_step_cached":
        return False
    if str(req.use_part).strip().lower() == "real":
        return True
    if not _is_complex_measurement_request(req):
        return False
    meta = dict(req.metadata or {})
    return bool(
        _single_step_rm_route_requires_artifact(meta)
        or _should_auto_build_rm_artifact(meta)
        or _should_resolve_greit_registry(meta)
        or _has_explicit_rm_artifact_meta(meta)
    )


def _cache_hit_summary(
    cache_lookups: dict[str, Any],
) -> tuple[bool | None, dict[str, bool]]:
    hits: dict[str, bool] = {}
    for key, value in cache_lookups.items():
        if not isinstance(value, dict):
            continue
        layer = str(value.get("layer", "")).strip().lower()
        if layer == "disabled":
            continue
        if "hit" in value:
            hits[key] = bool(value.get("hit"))
    if not hits:
        return None, hits
    return all(hits.values()), hits


def _single_step_runtime_diagnostics(ctx: dict[str, Any]) -> dict[str, Any]:
    cache_lookups = dict(ctx.get("cache_lookups", {}))
    cache_hit, cache_hits = _cache_hit_summary(cache_lookups)
    petsc_info = dict(ctx.get("petsc_backend_info", {}))
    return {
        "mesh_family": str(ctx.get("mesh_family", "")),
        "potential_order": int(
            petsc_info.get("potential_order", ctx.get("potential_order", 1)) or 1
        ),
        "forward_backend": str(ctx.get("forward_backend", "")),
        "forward_backend_effective": str(
            petsc_info.get("forward_backend_effective", ctx.get("forward_backend", ""))
        ),
        "solver_preset": str(petsc_info.get("solver_preset", "")),
        "forward_solver_preset": str(
            petsc_info.get("solver_preset", ctx.get("forward_solver_preset", ""))
        ),
        "forward_solver_policy_reason": str(
            petsc_info.get(
                "forward_solver_policy_reason",
                ctx.get("forward_solver_policy_reason", ""),
            )
        ),
        "forward_solver_policy_warning": str(
            petsc_info.get(
                "forward_solver_policy_warning",
                ctx.get("forward_solver_policy_warning", ""),
            )
        ),
        "petsc_device_requested": str(
            petsc_info.get("petsc_device_requested", ctx.get("petsc_device", ""))
        ),
        "petsc_device_effective": str(petsc_info.get("petsc_device_effective", "")),
        "petsc_amgx_available": bool(
            petsc_info.get(
                "petsc_amgx_available", ctx.get("petsc_amgx_available", False)
            )
        ),
        "petsc_hypre_available": bool(
            petsc_info.get(
                "petsc_hypre_available",
                ctx.get("petsc_hypre_available", False),
            )
        ),
        "petsc_hypre_cuda_blacklisted": bool(
            petsc_info.get(
                "petsc_hypre_cuda_blacklisted",
                ctx.get("petsc_hypre_cuda_blacklisted", False),
            )
        ),
        "forward_mat_solve_effective": str(
            petsc_info.get("forward_mat_solve_effective", "")
        ),
        "forward_mat_solve_policy_reason": str(
            petsc_info.get(
                "forward_mat_solve_policy_reason",
                ctx.get("forward_mat_solve_policy_reason", ""),
            )
        ),
        "torch_device": str(ctx.get("torch_device", "")),
        "device_requested": str(ctx.get("device_requested", "")),
        "device_effective": str(ctx.get("device_effective", "")),
        "jacobian_representation": str(ctx.get("jacobian_representation", "")),
        "jacobian_representation_reason": str(
            ctx.get("jacobian_representation_reason", "")
        ),
        "linearized_solver_strategy": str(ctx.get("linearized_solver_strategy", "")),
        "linearized_maxiter": ctx.get("linearized_maxiter"),
        "lazy_preconditioner_mode": str(ctx.get("lazy_preconditioner_mode", "")),
        "mesh_cache_hit": ctx.get("mesh_cache_hit"),
        "mesh_cache_layer": ctx.get("mesh_cache_layer"),
        "mesh_cache_name": ctx.get("mesh_cache_name"),
        "single_step_context_process_cache_bytes": int(
            ctx.get("single_step_context_process_cache_bytes", 0) or 0
        ),
        "single_step_context_process_cache_max_bytes": int(
            ctx.get("single_step_context_process_cache_max_bytes", 0) or 0
        ),
        "single_step_context_process_cache_stored": bool(
            ctx.get("single_step_context_process_cache_stored", False)
        ),
        "single_step_context_process_cache_skip_reason": str(
            ctx.get("single_step_context_process_cache_skip_reason", "")
        ),
        "cache_hit": cache_hit,
        "cache_hits": cache_hits,
    }


def _single_step_cached_solver_diagnostics(
    ctx: dict[str, Any],
    *,
    strict_backend: str,
) -> dict[str, Any]:
    return {
        "path": "single_step_cached",
        "strict_solver_backend_effective": strict_backend,
        "runtime": _single_step_runtime_diagnostics(ctx),
        "cache_lookups": dict(ctx.get("cache_lookups", {})),
        "cache_build_seconds": dict(ctx.get("cache_build_seconds", {})),
        "context_build_seconds": ctx.get("context_build_seconds"),
        "cache_miss_reasons": dict(ctx.get("cache_miss_reasons", {})),
        "cache_stats": (
            ctx["cache_manager"].stats() if ctx.get("cache_manager") is not None else {}
        ),
    }


def _single_step_operator_space(
    operator_bundle: dict[str, Any],
    dv: np.ndarray,
    *,
    measurement_backend: str,
) -> str:
    """Return whether a cached single-step operator solves measurement or parameter space."""
    dv_len = int(np.asarray(dv).reshape(-1).shape[0])
    a_shape = tuple(int(dim) for dim in np.shape(operator_bundle.get("A")))
    if len(a_shape) >= 2 and a_shape[-2] == dv_len and a_shape[-1] == dv_len:
        return "measurement"
    if (
        str(operator_bundle.get("strict_solver_backend_effective", ""))
        == measurement_backend
    ):
        return "measurement"
    if str(operator_bundle.get("mode", "")).strip().lower() == "fast":
        return "measurement"
    return "parameter"


def _ensure_single_step_cached_context(
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    emit: Callable[[str], None],
    build_shared_context: Callable[..., Any],
    rm_build_only: bool = False,
) -> dict[str, Any]:
    meta = runtime.meta
    context_cache_key = (
        (*runtime.cache_key, "rm_build_only") if rm_build_only else runtime.cache_key
    )
    ctx = _get_cached_fast_context(context_cache_key)
    built_context = False
    if ctx is None:
        emit("Building cached single-step context...")
        ctx = _quiet_call(
            lambda: build_shared_context(
                mesh_dir=str(meta.get("mesh_dir", "eit_meshes")),
                mesh_name=None,
                mesh_dim=runtime.mesh_dim,
                mesh_height=runtime.mesh_height,
                electrode_height_ratio=runtime.electrode_height_ratio,
                z_center=runtime.z_center,
                electrode_level_fractions=meta.get(
                    "electrode_level_fractions", (0.25, 0.75)
                ),
                refinement=runtime.refinement,
                n_elec=int(meta["n_elec"]),
                radius=float(meta.get("radius", 1.0)),
                drive_mode=str(meta["drive_mode"]),
                drive_value=float(meta["drive_value"]),
                contact_impedance=runtime.contact_impedance,
                electrode_length_m_override=meta.get("electrode_length_m_override"),
                electrode_coverage=float(meta.get("electrode_coverage", 0.5)),
                geometry_scale_to_m=float(meta.get("geometry_scale_to_m", 1.0)),
                n_rings=int(meta.get("n_rings", 1)),
                electrode_layout=str(meta.get("electrode_layout", "ring_major")),
                measurement_protocol=str(
                    meta.get("measurement_protocol", "eidors_full_3d")
                ),
                custom_stim_matrix=meta.get("custom_stim_matrix"),
                custom_meas_matrices=meta.get("custom_meas_matrices"),
                stim_pattern=str(meta.get("stim_pattern", "{ad}")),
                meas_pattern=str(meta.get("meas_pattern", "{ad}")),
                rotate_meas=bool(meta.get("rotate_meas", True)),
                use_meas_current=bool(meta.get("use_meas_current", False)),
                use_meas_current_next=int(meta.get("use_meas_current_next", 0)),
                stim_direction=str(meta.get("stim_direction", "ccw")),
                meas_direction=str(meta.get("meas_direction", "ccw")),
                stim_first_positive=bool(meta.get("stim_first_positive", False)),
                difference_mode=str(meta.get("difference_mode", "raw")),
                difference_orientation=str(
                    meta.get("difference_orientation", "target_minus_reference")
                ),
                background_sigma=runtime.background_sigma,
                lam=runtime.lam,
                cache_scope=_single_step_context_cache_scope(meta),
                solver_mode=str(meta.get("solver_mode", "strict")),
                linear_solver=str(meta.get("linear_solver", "auto")),
                preconditioner=str(meta.get("preconditioner", "auto")),
                rom_mode="off",
                lowrank_mode="off",
                forward_solver_preset=str(meta.get("forward_solver_preset", "auto")),
                forward_mat_solve=str(meta.get("forward_mat_solve", "off")),
                petsc_device=str(meta.get("petsc_device", "auto")),
                device=str(meta.get("device", "auto")),
                jacobian_representation=str(
                    meta.get("jacobian_representation", "dense")
                ),
                linearized_solver_strategy=str(
                    meta.get("linearized_solver_strategy", "auto")
                ),
                linearized_maxiter=int(meta.get("linearized_maxiter", 0)),
                lazy_preconditioner_mode=str(
                    meta.get("lazy_preconditioner_mode", "auto")
                ),
                lazy_diag_batch_max_measurements=int(
                    meta.get("lazy_diag_batch_max_measurements", 512)
                ),
                forward_backend=str(meta.get("forward_backend", "dolfinx")),
                mesh_family=str(meta.get("mesh_family", "tetra")),
                geometry_version=str(meta.get("geometry_version", "geomv2")),
                potential_order=int(meta.get("potential_order", 1)),
                single_step_signature_schema_version=str(
                    meta.get("single_step_signature_schema_version")
                ),
                single_step_jacobian_calculator=str(
                    meta.get("single_step_jacobian_calculator")
                ),
                single_step_jacobian_math_convention=str(
                    meta.get("single_step_jacobian_math_convention")
                ),
                single_step_projection_math_convention=str(
                    meta.get("single_step_projection_math_convention")
                ),
                single_step_operator_math_convention=str(
                    meta.get("single_step_operator_math_convention")
                ),
                single_step_algorithm_version=str(
                    meta.get("single_step_algorithm_version")
                ),
                scalar_dtype=str(meta.get("rm_dtype", "")),
                rm_build_only=bool(rm_build_only),
            )
        )
        built_context = True
    else:
        emit("Reusing cached single-step context...")

    mesh = ctx["mesh"]
    if "display_node_coords" not in ctx:
        ctx["display_node_coords"] = _display_node_coords_array(mesh.coordinates())
    if "display_cell_connectivity" not in ctx:
        ctx["display_cell_connectivity"] = _display_cell_connectivity_array(
            mesh.cells()
        )
    ctx.setdefault(
        "jacobian_representation",
        str(meta.get("jacobian_representation", "dense")),
    )
    ctx["jacobian_representation_reason"] = str(
        meta.get("jacobian_representation_reason", "")
    )
    if built_context:
        ctx["single_step_context_cache_max_bytes"] = (
            _single_step_context_cache_max_bytes(meta)
        )
        _put_cached_fast_context(context_cache_key, ctx)
    return ctx


def _run_full_gn_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request via the legacy full GN runtime."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    emit("Loading PyEIDORS...")
    from pyeidors import EITSystem
    from pyeidors.data import MeasurementDataset, PatternConfig
    from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh

    ref_vec = req.reference_frame.to_measurement_vector(req.use_part)
    tgt_vec = req.target_frame.to_measurement_vector(req.use_part)
    native_complex_request = _is_native_complex_reconstruction_request(
        req,
        np.asarray(ref_vec),
        np.asarray(tgt_vec),
    )

    meta = dict(req.metadata)
    meta.setdefault("n_elec", 16)
    meta.setdefault("n_rings", 1)
    meta.setdefault("electrode_layout", "ring_major")
    meta.setdefault("measurement_protocol", "eidors_full_3d")
    meta.setdefault("custom_stim_matrix", None)
    meta.setdefault("custom_meas_matrices", None)
    meta.setdefault("stim_pattern", "{ad}")
    meta.setdefault("meas_pattern", "{ad}")
    meta.setdefault("rotate_meas", True)
    meta.setdefault("use_meas_current", False)
    meta.setdefault("use_meas_current_next", 0)
    meta.setdefault("stim_direction", "ccw")
    meta.setdefault("meas_direction", "ccw")
    meta.setdefault("stim_first_positive", False)
    meta.setdefault("geometry_scale_to_m", 1.0)
    meta.setdefault("radius", 1.0)
    meta.setdefault("electrode_coverage", 0.5)
    meta.setdefault("electrode_length_m_override", None)
    meta.setdefault("electrode_area_m2_override", None)
    meta.setdefault("contact_impedance", 0.01)
    meta.setdefault("difference_mode", "raw")
    meta.setdefault("difference_orientation", "target_minus_reference")
    meta.setdefault("difference_preset", "eidors_one_step_noser")
    meta.setdefault("absolute_preset", "eidors_abs_gn")
    meta.setdefault("hyperparameter", None)
    meta.setdefault("solver_mode", "auto")
    meta.setdefault("line_search_mode", "auto")
    meta.setdefault("linear_solver", "auto")
    meta.setdefault("preconditioner", "auto")
    meta.setdefault("fast_linear_path", "auto")
    meta.setdefault("forward_solver_preset", "auto")
    meta.setdefault("forward_mat_solve", "auto")
    meta.setdefault("petsc_device", "auto")
    meta.setdefault("device", "auto")
    meta.setdefault("linearized_solver_strategy", "auto")
    meta.setdefault("linearized_maxiter", 0)
    meta.setdefault("lazy_preconditioner_mode", "auto")
    meta.setdefault("lazy_diag_batch_max_measurements", 512)
    meta.setdefault("forward_backend", "dolfinx")
    meta.setdefault("mesh_family", "tetra")
    meta.setdefault("geometry_version", "geomv2")
    meta.setdefault("potential_order", 1)
    meta.setdefault("acceleration_profile", "default")
    meta["drive_mode"] = _resolve_drive_mode(meta, mesh_dim=int(req.mesh_dimension))
    meta["drive_value"] = _resolve_drive_value(meta)
    runtime_options = _resolve_reconstruction_runtime(
        meta, mesh_dim=int(req.mesh_dimension)
    )
    meta.update(runtime_options)
    requested_difference_preset = str(
        meta.get("difference_preset", _DEFAULT_EIT_SYSTEM_DIFFERENCE_PRESET)
    )
    system_difference_preset = _eit_system_difference_preset_for_full_gn(
        requested_difference_preset,
        native_complex=native_complex_request,
    )
    if system_difference_preset != requested_difference_preset.strip().lower():
        meta.setdefault("difference_preset_requested", requested_difference_preset)
        meta["difference_preset_effective"] = system_difference_preset

    emit("Building measurement datasets...")
    data_type = (
        req.use_part if req.use_part in {"real", "imag", "mag", "complex"} else "real"
    )
    ref_ds = MeasurementDataset.from_metadata(
        measurements=ref_vec.reshape(1, -1),
        metadata=meta,
        data_type=data_type,
    )
    tgt_ds = MeasurementDataset.from_metadata(
        measurements=tgt_vec.reshape(1, -1),
        metadata=meta,
        data_type=data_type,
    )
    ref_eit = ref_ds.to_eit_data(frame_index=0)
    tgt_eit = tgt_ds.to_eit_data(frame_index=0)

    emit("Setting up EIT system...")
    radius = float(meta.get("radius", 1.0))
    refinement = _compute_effective_refinement(
        radius,
        req.mesh_refinement,
        mesh_size=meta.get("mesh_size"),
    )
    cache_key = (
        int(meta["n_elec"]),
        int(meta.get("n_rings", 1)),
        str(meta.get("electrode_layout", "ring_major")),
        str(meta.get("measurement_protocol", "eidors_full_3d")),
        repr(meta.get("custom_stim_matrix")),
        repr(meta.get("custom_meas_matrices")),
        str(meta["stim_pattern"]),
        str(meta["meas_pattern"]),
        bool(meta.get("rotate_meas", True)),
        bool(meta.get("use_meas_current", False)),
        int(meta.get("use_meas_current_next", 0)),
        str(meta.get("stim_direction", "ccw")),
        str(meta.get("meas_direction", "ccw")),
        bool(meta.get("stim_first_positive", False)),
        str(meta["drive_mode"]),
        float(meta["drive_value"]),
        float(meta["geometry_scale_to_m"]),
        int(req.mesh_dimension),
        int(refinement),
        float(meta.get("radius", 1.0)),
        float(meta.get("mesh_height", meta.get("height", 1.0))),
        float(meta.get("electrode_height_ratio", 0.2)),
        float(meta.get("z_center", 0.0)),
        repr(meta.get("electrode_level_fractions", (0.25, 0.75))),
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        repr(meta.get("electrode_area_m2_override")),
        _contact_impedance_scalar(meta.get("contact_impedance", 0.01)),
        float(req.regularization_alpha),
        repr(meta.get("hyperparameter")),
        system_difference_preset,
        str(meta.get("absolute_preset", "eidors_abs_gn")),
        str(meta["difference_mode"]),
        str(meta["difference_orientation"]),
        str(meta.get("solver_mode", "auto")),
        str(meta.get("line_search_mode", "auto")),
        str(meta.get("linear_solver", "auto")),
        str(meta.get("preconditioner", "auto")),
        str(meta.get("fast_linear_path", "auto")),
        str(meta.get("linearized_solver_strategy", "auto")),
        int(meta.get("linearized_maxiter", 0)),
        str(meta.get("lazy_preconditioner_mode", "auto")),
        int(meta.get("lazy_diag_batch_max_measurements", 512)),
        str(meta.get("forward_solver_preset", "auto")),
        str(meta.get("forward_mat_solve", "auto")),
        str(meta.get("petsc_device", "auto")),
        str(meta.get("device", "auto")),
        str(meta.get("forward_backend", "dolfinx")),
        str(meta.get("mesh_family", "tetra")),
        str(meta.get("geometry_version", "geomv2")),
        str(meta.get("acceleration_profile", "default")),
    )
    total_electrodes = _total_electrodes_from_meta(meta)
    system = _get_cached_system(cache_key)
    if system is None:
        pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
            mesh_tdim=req.mesh_dimension,
            n_elec=int(meta["n_elec"]),
            n_rings=int(meta.get("n_rings", 1)),
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
        )
        pattern_config = PatternConfig(
            n_elec=pattern_n_elec,
            n_rings=pattern_n_rings,
            stim_pattern=meta["stim_pattern"],
            meas_pattern=meta["meas_pattern"],
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
            measurement_protocol=str(
                meta.get("measurement_protocol", "eidors_full_3d")
            ),
            custom_stim_matrix=meta.get("custom_stim_matrix"),
            custom_meas_matrices=meta.get("custom_meas_matrices"),
            drive_mode=meta["drive_mode"],
            drive_value=meta["drive_value"],
            geometry_scale_to_m=meta["geometry_scale_to_m"],
            electrode_length_m_override=meta.get("electrode_length_m_override"),
            use_meas_current=bool(meta.get("use_meas_current", False)),
            use_meas_current_next=int(meta.get("use_meas_current_next", 0)),
            rotate_meas=bool(meta.get("rotate_meas", True)),
            stim_direction=str(meta.get("stim_direction", "ccw")),
            meas_direction=str(meta.get("meas_direction", "ccw")),
            stim_first_positive=bool(meta.get("stim_first_positive", False)),
        )
        hyperparameter = meta.get("hyperparameter")
        if hyperparameter in (None, ""):
            hyperparameter = None
        else:
            hyperparameter = float(hyperparameter)
        system = EITSystem(
            n_elec=total_electrodes,
            pattern_config=pattern_config,
            regularization_alpha=req.regularization_alpha,
            hyperparameter=hyperparameter,
            difference_mode=meta["difference_mode"],
            difference_orientation=meta["difference_orientation"],
            difference_preset=system_difference_preset,
            absolute_preset=str(meta.get("absolute_preset", "eidors_abs_gn")),
            contact_impedance=_contact_impedance_vector_from_meta(
                meta,
                total_electrodes=total_electrodes,
            ),
            solver_mode=str(meta.get("solver_mode", "strict")),
            line_search_mode=str(meta.get("line_search_mode", "full")),
            linear_solver=str(meta.get("linear_solver", "auto")),
            preconditioner=str(meta.get("preconditioner", "auto")),
            fast_linear_path=str(meta.get("fast_linear_path", "auto")),
            linear_backend_config={
                "solver_preset": str(meta.get("forward_solver_preset", "auto")),
                "mat_solve_mode": str(meta.get("forward_mat_solve", "off")),
                "petsc_device": str(meta.get("petsc_device", "auto")),
            },
            petsc_device=str(meta.get("petsc_device", "auto")),
            device=str(meta.get("device", "auto")),
            forward_backend=str(meta.get("forward_backend", "dolfinx")),
            mesh_family=str(meta.get("mesh_family", "tetra")),
            potential_order=int(meta.get("potential_order", 1)),
            acceleration_profile=str(meta.get("acceleration_profile", "default")),
        )
        mesh = load_or_create_mesh(
            mesh_dir=str(meta.get("mesh_dir", "eit_meshes")),
            n_elec=total_electrodes,
            dimension=int(req.mesh_dimension),
            radius=radius,
            refinement=refinement,
            electrode_coverage=float(meta.get("electrode_coverage", 0.5)),
            height=float(meta.get("mesh_height", meta.get("height", 1.0))),
            electrode_height_ratio=float(meta.get("electrode_height_ratio", 0.2)),
            electrode_level_fractions=meta.get(
                "electrode_level_fractions", (0.25, 0.75)
            ),
            z_center=float(meta.get("z_center", 0.0)),
            mesh_family=str(meta.get("mesh_family", "tetra")),
            geometry_version=str(meta.get("geometry_version", "geomv2")),
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
        )
        system.setup(mesh=mesh)
        setattr(
            system,
            "_reconstruction_system_cache_max_bytes",
            _reconstruction_system_cache_max_bytes(meta),
        )
        _put_cached_system(cache_key, system)
    else:
        emit("Reusing cached reconstruction system...")

    method = req.method.strip().lower()
    if method == "gn-absolute" and getattr(system, "reconstructor", None) is not None:
        max_iterations = max(1, int(req.max_iterations))
        system.reconstructor.max_iterations = max_iterations
        meta["max_iterations_requested"] = max_iterations
        meta["max_iterations_effective"] = max_iterations

    if method == "gn-difference" and native_complex_request:
        return _run_native_complex_linearized_difference(
            req=req,
            system=system,
            ref_vec=np.asarray(ref_vec),
            tgt_vec=np.asarray(tgt_vec),
            meta=meta,
            progress_cb=progress_cb,
        )

    emit("Running reconstruction...")
    if method == "gn-absolute":
        recon = system.absolute_reconstruct(measurement_data=tgt_eit)
    elif method == "sparse-bayes-absolute":
        from pyeidors.inverse.workflows.sparse_bayesian import (
            perform_sparse_absolute_reconstruction,
        )

        recon = perform_sparse_absolute_reconstruction(
            eit_system=system,
            measurement_data=tgt_eit,
        )
    elif method == "sparse-bayes-difference" or method == "sparse-bayes":
        from pyeidors.inverse.workflows.sparse_bayesian import (
            perform_sparse_difference_reconstruction,
        )

        recon = perform_sparse_difference_reconstruction(
            eit_system=system,
            measurement_data=tgt_eit,
            reference_data=ref_eit,
        )
    else:
        # default: gn-difference (single-step Gauss-Newton)
        recon = system.difference_reconstruct(
            measurement_data=tgt_eit,
            reference_data=ref_eit,
        )

    mesh = system.mesh
    coords = mesh.coordinates()
    cells = mesh.cells()

    emit("Reconstruction complete")
    result_meta = dict(meta)
    result_meta["reconstruction_runtime"] = "full_gn"
    diagnostics = getattr(recon, "metadata", {}).get("solver_diagnostics")
    if diagnostics is not None:
        result_meta["solver_diagnostics"] = diagnostics
    return ReconstructionResult(
        conductivity=(
            recon.conductivity if hasattr(recon, "conductivity") else np.asarray([])
        ),
        node_coords=coords,
        cell_connectivity=cells,
        measured=getattr(recon, "measured", None),
        simulated=getattr(recon, "simulated", None),
        metadata=result_meta,
    )


def _run_single_step_cached_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request via the cached single-step realtime path."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    runtime = _prepare_single_step_cached_runtime(req)
    if _should_resolve_greit_registry(runtime.meta):
        _ensure_greit_registry_artifact(req, runtime, emit=emit)
    rm_result = _try_run_cached_rm_request(req, runtime, progress_cb=progress_cb)
    if rm_result is not None:
        return rm_result
    if _should_auto_build_rm_artifact(runtime.meta):
        _ensure_auto_built_one_step_rm_artifact(req, runtime, emit=emit)
        rm_result = _try_run_cached_rm_request(req, runtime, progress_cb=progress_cb)
        if rm_result is not None:
            return rm_result
    if _single_step_rm_route_requires_artifact(runtime.meta):
        return _missing_rm_artifact_result(runtime, emit=emit)

    diff_runner = _load_gn_difference_runner_module()
    STRICT_SOLVER_BACKEND_MEASUREMENT = diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT
    _calibrate_step_size = diff_runner._calibrate_step_size
    _measurement_space_delta = diff_runner._measurement_space_delta
    _solve_linear_from_bundle = diff_runner._solve_linear_from_bundle
    _solve_linearized_delta = getattr(diff_runner, "_solve_linearized_delta", None)
    meta = runtime.meta
    ctx = _ensure_single_step_cached_context(
        runtime,
        emit=emit,
        build_shared_context=diff_runner.build_shared_context,
    )
    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
        }
    )

    if bool(meta.get("warmup_only", False)):
        result_meta["cache_warmup_only"] = True
        result_meta["solver_diagnostics"] = _single_step_cached_solver_diagnostics(
            ctx,
            strict_backend="warmup_only",
        )
        emit("Realtime reconstruction context ready")
        return ReconstructionResult(
            conductivity=np.asarray([], dtype=np.float64),
            node_coords=ctx["display_node_coords"],
            cell_connectivity=ctx["display_cell_connectivity"],
            metadata=result_meta,
        )

    from pyeidors.data.structures import EITImage
    from pyeidors.utils.numeric_ops import safe_dot

    ref_vec = np.asarray(req.reference_frame.to_measurement_vector(req.use_part))
    tgt_vec = np.asarray(req.target_frame.to_measurement_vector(req.use_part))

    emit("Running cached single-step reconstruction...")
    difference_mode = str(meta.get("difference_mode", "raw"))
    difference_orientation = str(
        meta.get("difference_orientation", "target_minus_reference")
    )
    dv = build_difference_vector(
        tgt_vec,
        ref_vec,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    operator_bundle = ctx["operator_bundle"]
    strict_backend = str(
        operator_bundle.get(
            "strict_solver_backend_effective",
            "dense-param",
        )
    )
    if str(operator_bundle.get("jacobian_representation", "dense")) in {
        "linearized",
        "lazy",
    }:
        if _solve_linearized_delta is None:
            raise RuntimeError("linearized single-step runtime is unavailable.")
        operator_space = "linearized"
        delta_sigma = _solve_linearized_delta(operator_bundle=operator_bundle, rhs=dv)
    else:
        operator_space = _single_step_operator_space(
            operator_bundle,
            dv,
            measurement_backend=STRICT_SOLVER_BACKEND_MEASUREMENT,
        )
    if operator_space == "measurement":
        delta_sigma = _measurement_space_delta(operator_bundle=operator_bundle, rhs=dv)
    elif operator_space != "linearized":
        rhs = np.asarray(
            safe_dot(operator_bundle["Jt"], dv, "eit_app.fast_recon.Jt_dv")
        )
        delta_sigma = _solve_linear_from_bundle(operator_bundle, rhs)

    sigma_floor = _single_step_sigma_floor(meta)
    alpha = 1.0
    if bool(meta.get("step_size_calib", True)):
        try:
            alpha = float(
                _calibrate_step_size(
                    fwd_model=ctx["fwd_model"],
                    sigma_bg=ctx["sigma_bg"],
                    delta_sigma=delta_sigma,
                    dv=dv,
                    base_meas=ctx["base_meas"],
                    step_size_min=float(meta.get("step_size_min", 1.0e-6)),
                    step_size_max=float(meta.get("step_size_max", 1.0)),
                    step_size_maxiter=int(meta.get("step_size_maxiter", 64)),
                    difference_mode=difference_mode,
                    difference_orientation=difference_orientation,
                    sigma_floor=sigma_floor,
                )
            )
        except Exception as exc:
            log.debug("Realtime step-size calibration failed: %s", exc)
            alpha = 1.0
        if not np.isfinite(alpha) or alpha <= 0.0:
            alpha = 1.0

    alpha_requested = float(alpha)
    alpha, _display_delta, sigma_est, sigma_floor_applied = (
        _constrain_single_step_sigma_update(
            ctx["sigma_bg"],
            delta_sigma,
            alpha,
            sigma_floor=sigma_floor,
        )
    )
    img_est = EITImage(elem_data=sigma_est, fwd_model=ctx["fwd_model"])
    pred_vi, _ = ctx["fwd_model"].fwd_solve(img_est)
    pred_diff = build_difference_vector(
        pred_vi.meas,
        ctx["base_meas"],
        mode=difference_mode,
        orientation=difference_orientation,
    )

    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "conductivity_display_mode": "absolute_sigma",
            "step_size_alpha": alpha,
            "step_size_alpha_requested": alpha_requested,
            "step_size_alpha_limited": bool(alpha < alpha_requested),
            "sigma_floor": sigma_floor,
            "sigma_floor_applied": sigma_floor_applied,
            "single_step_operator_space": operator_space,
            "solver_diagnostics": _single_step_cached_solver_diagnostics(
                ctx,
                strict_backend=strict_backend,
            ),
        }
    )

    emit("Reconstruction complete")
    return ReconstructionResult(
        conductivity=sigma_est,
        node_coords=ctx["display_node_coords"],
        cell_connectivity=ctx["display_cell_connectivity"],
        measured=dv,
        simulated=pred_diff,
        metadata=result_meta,
    )


def run_reconstruction_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request synchronously using the realtime app pipeline."""
    try:
        runtime_path = (
            str((req.metadata or {}).get("reconstruction_runtime", "")).strip().lower()
        )
        method_lc = req.method.strip().lower()
        log.info(
            "[recon-dispatch] method=%r use_part=%r runtime_path=%r source=%r",
            method_lc,
            req.use_part,
            runtime_path,
            (req.metadata or {}).get("request_source"),
        )
        if _flag_enabled((req.metadata or {}).get("pseudo3d_layered_output", False)):
            log.info("[recon-dispatch] -> pseudo3d_layered")
            return _run_pseudo3d_layered_request(req, progress_cb=progress_cb)
        if _can_dispatch_single_step_cached(
            req,
            method_lc=method_lc,
            runtime_path=runtime_path,
        ):
            log.info("[recon-dispatch] -> single_step_cached (fast path)")
            return _maybe_apply_pseudo3d_result(
                _run_single_step_cached_request(req, progress_cb=progress_cb)
            )
        log.info("[recon-dispatch] -> full_gn (iterative path)")
        return _maybe_apply_pseudo3d_result(
            _run_full_gn_request(req, progress_cb=progress_cb)
        )

    except Exception as exc:
        log.exception("Reconstruction failed")
        return ReconstructionResult(
            conductivity=np.array([]),
            node_coords=np.array([]),
            cell_connectivity=np.array([]),
            error_msg=str(exc),
            metadata=dict(getattr(req, "metadata", {}) or {}),
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def execute_reconstruction_request_in_backend(
    req: ReconstructionRequest,
    *,
    profile: str,
    route_reason: str,
    progress_cb: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ReconstructionResult:
    """Run a reconstruction request in a profile-isolated backend process."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def check_cancelled() -> bool:
        return bool(cancelled is not None and cancelled())

    repo = _repo_root()
    profile_name = str(profile or "default").strip() or "default"
    with tempfile.TemporaryDirectory(prefix="pyeidors-gui-backend-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "reconstruction_request.h5"
        output_path = tmp_dir / "reconstruction_result.h5"
        from eit_app.backend_worker_protocol import (
            read_reconstruction_result,
            write_reconstruction_request,
        )
        from eit_app.backend_worker_runtime import (
            backend_worker_command,
            clean_profile_command_env,
            backend_worker_env,
            backend_worker_profile_lock,
        )

        write_reconstruction_request(input_path, req)
        from eit_app.backend_worker_pool import (
            BackendWorkerTransportError,
            persistent_backend_workers_enabled,
            run_persistent_backend_worker_request,
        )

        if persistent_backend_workers_enabled():
            try:
                worker_meta = run_persistent_backend_worker_request(
                    repo=repo,
                    profile=profile_name,
                    command="reconstruct",
                    input_path=input_path,
                    output_path=output_path,
                    progress_cb=progress_cb,
                )
                result = read_reconstruction_result(output_path)
                result.metadata = {
                    **dict(result.metadata or {}),
                    "backend_worker_profile": profile_name,
                    "backend_worker_route_reason": route_reason,
                    "backend_worker_process_isolated": True,
                    "backend_worker_persistent": True,
                    "backend_worker_launch_mode": worker_meta.launch_mode,
                    "backend_worker_cache_home": str(worker_meta.cache_home),
                    "backend_worker_pid": worker_meta.pid,
                    "backend_worker_reused_process": worker_meta.reused_process,
                    "backend_worker_stale_jit_locks_removed": (
                        worker_meta.stale_jit_locks_removed
                    ),
                    "backend_worker_rss_bytes": getattr(worker_meta, "rss_bytes", 0),
                    "backend_worker_rss_limit_bytes": getattr(
                        worker_meta, "rss_limit_bytes", 0
                    ),
                    "backend_worker_recycled_after_request": (
                        getattr(worker_meta, "recycled_after_request", False)
                    ),
                    "backend_worker_recycle_reason": getattr(
                        worker_meta, "recycle_reason", ""
                    ),
                    "backend_worker_primed_runtime": getattr(
                        worker_meta, "primed_runtime", False
                    ),
                    "backend_worker_prime_command": getattr(
                        worker_meta, "prime_command", ""
                    ),
                    "backend_worker_prime_duration_ms": getattr(
                        worker_meta, "prime_duration_ms", 0.0
                    ),
                }
                emit("Backend reconstruction complete.")
                return result
            except BackendWorkerTransportError as exc:
                if check_cancelled():
                    raise InterruptedError from exc
                emit(
                    "Persistent backend worker unavailable; "
                    f"falling back to one-shot worker: {exc}"
                )
        if check_cancelled():
            raise InterruptedError

        cmd, launch_mode = backend_worker_command(
            profile=profile_name,
            worker_args=[
                "reconstruct",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )
        emit(
            "Dispatching reconstruction to backend "
            f"profile={profile_name} via {launch_mode} ({route_reason})..."
        )
        with backend_worker_profile_lock(repo, profile_name):
            env, cache = backend_worker_env(repo=repo, profile=profile_name)
            if launch_mode == "profile_command":
                clean_profile_command_env(env)
            if cache.removed_stale_jit_locks:
                emit(
                    "Cleaned backend JIT cache: "
                    f"{len(cache.removed_stale_jit_locks)} stale lock file(s)."
                )
            emit(f"Backend cache: {cache.xdg_cache_home}")
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        if proc.stderr:
            for line in proc.stderr.splitlines():
                line = line.strip()
                if line:
                    emit(line)
        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"Backend worker profile={profile_name} failed with exit "
                f"{proc.returncode}: {details}"
            )
        result = read_reconstruction_result(output_path)
        result.metadata = {
            **dict(result.metadata or {}),
            "backend_worker_profile": profile_name,
            "backend_worker_route_reason": route_reason,
            "backend_worker_process_isolated": True,
            "backend_worker_persistent": False,
            "backend_worker_launch_mode": launch_mode,
            "backend_worker_cache_home": str(cache.xdg_cache_home),
            "backend_worker_stale_jit_locks_removed": len(
                cache.removed_stale_jit_locks
            ),
        }
        emit("Backend reconstruction complete.")
        return result


class ReconstructionController(QObject):
    """GUI-facing controller for EIT reconstruction.

    Signals:
        reconstruction_done: Emitted with ReconstructionResult.
        progress: Emitted with status strings during reconstruction.
        error: Emitted on errors.
    """

    reconstruction_done = Signal(object)  # ReconstructionResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ReconstructionWorker | None = None
        self._busy = False
        self._shutting_down = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def reconstruct(self, request: ReconstructionRequest) -> bool:
        """Submit a reconstruction request. Runs in a background thread."""
        if self._busy:
            self.error.emit("Reconstruction already in progress")
            return False

        self._shutting_down = False
        self._busy = True
        self._thread = QThread()
        self._worker = _ReconstructionWorker()
        self._worker._request = request
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)

        self._thread.start()
        return True

    def _on_finished(self, result: ReconstructionResult) -> None:
        self._busy = False
        self._stop_worker_thread(force=False, grace_ms=5000)
        if not self._shutting_down:
            self.reconstruction_done.emit(result)

    def shutdown(self) -> None:
        self._shutting_down = True
        self._stop_worker_thread(force=True, grace_ms=3000)
        self._busy = False

    def _stop_worker_thread(self, *, force: bool, grace_ms: int) -> None:
        thread = self._thread
        worker = self._worker
        if worker is not None:
            worker.cancel()
        if thread is not None:
            if thread.isRunning():
                thread.requestInterruption()
                thread.quit()
                if not thread.wait(grace_ms):
                    log.warning(
                        "Reconstruction thread did not stop within %d ms%s",
                        grace_ms,
                        "; terminating" if force else "",
                    )
                    if force:
                        thread.terminate()
                        thread.wait(3000)
            if not thread.isRunning():
                thread.deleteLater()
                self._thread = None
        if worker is not None and self._thread is None:
            worker.deleteLater()
            self._worker = None
