"""Forward problem solver running in a background QThread."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from eit_app.models.precision import compute_dtype
from eit_app.models.forward_model_config import ForwardModelConfig
from eit_app.models.simulation_state import InhomogeneitySpec

log = logging.getLogger(__name__)


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


_HEX_SAMPLE_AXIS = np.asarray([0.125, 0.375, 0.625, 0.875], dtype=np.float64)
_HEX_SAMPLE_GRID = np.asarray(
    [
        [x, y, z]
        for z in _HEX_SAMPLE_AXIS
        for y in _HEX_SAMPLE_AXIS
        for x in _HEX_SAMPLE_AXIS
    ],
    dtype=np.float64,
)


def _build_hex_sample_weights(sample_grid: np.ndarray) -> np.ndarray:
    u = sample_grid[:, 0]
    v = sample_grid[:, 1]
    w = sample_grid[:, 2]
    weights = np.empty((sample_grid.shape[0], 8), dtype=np.float64)
    one_minus_u = 1.0 - u
    one_minus_v = 1.0 - v
    one_minus_w = 1.0 - w
    weights[:, 0] = one_minus_u * one_minus_v * one_minus_w
    weights[:, 1] = u * one_minus_v * one_minus_w
    weights[:, 2] = u * v * one_minus_w
    weights[:, 3] = one_minus_u * v * one_minus_w
    weights[:, 4] = one_minus_u * one_minus_v * w
    weights[:, 5] = u * one_minus_v * w
    weights[:, 6] = u * v * w
    weights[:, 7] = one_minus_u * v * w
    return np.ascontiguousarray(weights, dtype=np.float64)


_HEX_SAMPLE_WEIGHTS = _build_hex_sample_weights(_HEX_SAMPLE_GRID)
_TET_SAMPLE_BARYCENTRIC = np.asarray(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.55, 0.15, 0.15, 0.15],
        [0.15, 0.55, 0.15, 0.15],
        [0.15, 0.15, 0.55, 0.15],
        [0.15, 0.15, 0.15, 0.55],
        [0.35, 0.35, 0.15, 0.15],
        [0.35, 0.15, 0.35, 0.15],
        [0.35, 0.15, 0.15, 0.35],
        [0.15, 0.35, 0.35, 0.15],
        [0.15, 0.35, 0.15, 0.35],
        [0.15, 0.15, 0.35, 0.35],
    ],
    dtype=np.float64,
)
_VOLUME_FRACTION_CHUNK_CELLS = 8192


@dataclass
class ForwardSolverRequest:
    """Input parameters for a single forward solve."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    n_electrodes: int = 16
    background_conductivity: float | complex = 1.0
    inhomogeneities: list[InhomogeneitySpec] = field(default_factory=list)
    noise_level: float = 0.0
    forward_model_config: dict[str, Any] = field(default_factory=dict)


def _cell_volume_sample_points(cell_vertices: np.ndarray) -> np.ndarray | None:
    """Return deterministic interior sample points for supported 3D cells."""
    vertices_raw = np.asarray(cell_vertices)
    if np.iscomplexobj(vertices_raw):
        vertices_raw = np.real(vertices_raw)
    if np.issubdtype(vertices_raw.dtype, np.floating):
        vertices = np.asarray(
            vertices_raw,
            dtype=np.result_type(vertices_raw.dtype, np.float32),
        )
    else:
        vertices = np.asarray(vertices_raw, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[1] not in {4, 8} or vertices.shape[2] < 3:
        return None
    vertices = vertices[:, :, :3]
    if vertices.shape[1] == 4:
        weights = np.asarray(_TET_SAMPLE_BARYCENTRIC, dtype=vertices.dtype)
        return np.einsum("sv,nvd->nsd", weights, vertices)

    weights = np.asarray(_HEX_SAMPLE_WEIGHTS, dtype=vertices.dtype)
    return np.einsum("sv,nvd->nsd", weights, vertices)


def _volume_fraction_sample_weights(vertices_per_cell: int) -> np.ndarray | None:
    if vertices_per_cell == 4:
        return _TET_SAMPLE_BARYCENTRIC
    if vertices_per_cell == 8:
        return _HEX_SAMPLE_WEIGHTS
    return None


def _apply_volume_fraction(
    values: np.ndarray,
    sample_points: np.ndarray | None,
    inside: np.ndarray,
    conductivity: float | complex,
) -> bool:
    """Blend cell conductivity by sampled inclusion volume fraction."""
    if sample_points is None:
        return False
    inside = np.asarray(inside, dtype=bool)
    if inside.ndim != 2 or inside.shape[:2] != sample_points.shape[:2]:
        return False
    fraction_dtype = np.result_type(sample_points.dtype, np.float32)
    fractions = inside.mean(axis=1, dtype=fraction_dtype)
    active = fractions > 0.0
    if np.any(active):
        conductivity_value = np.asarray(conductivity).reshape(()).item()
        values[active] = values[active] + fractions[active] * (
            conductivity_value - values[active]
        )
    return True


def _apply_volume_fraction_streaming(
    values: np.ndarray,
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
    inside_fn,
    conductivity: float | complex,
    *,
    chunk_size: int = _VOLUME_FRACTION_CHUNK_CELLS,
) -> bool:
    """Blend sampled cell fractions in chunks without expanding all vertices."""
    if node_coords is None or cell_connectivity is None:
        return False
    coords = np.asarray(node_coords)
    if np.iscomplexobj(coords):
        coords = np.real(coords)
    if np.issubdtype(np.asarray(coords).dtype, np.floating):
        coords = np.asarray(coords, dtype=np.result_type(coords.dtype, np.float32))
    else:
        coords = np.asarray(coords, dtype=np.float32)
    cells = np.asarray(cell_connectivity)
    if (
        coords.ndim != 2
        or coords.shape[1] < 3
        or cells.ndim != 2
        or cells.shape[0] != values.shape[0]
    ):
        return False
    if cells.shape[1] not in {4, 8}:
        return False
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= coords.shape[0]):
        return False

    coords_xyz = coords[:, :3]
    weights = _volume_fraction_sample_weights(cells.shape[1])
    if weights is None:
        return False
    fraction_dtype = np.result_type(coords_xyz.dtype, np.float32)
    conductivity_value = np.asarray(conductivity).reshape(()).item()
    step = max(1, int(chunk_size))
    for start in range(0, cells.shape[0], step):
        stop = min(start + step, cells.shape[0])
        chunk_cells = cells[start:stop]
        n_chunk = stop - start
        inside_counts = np.zeros(n_chunk, dtype=fraction_dtype)
        sample_points = np.empty((n_chunk, 3), dtype=coords_xyz.dtype)
        vertex_points = np.empty((n_chunk, 3), dtype=coords_xyz.dtype)
        for sample_weights in weights:
            sample_points.fill(0.0)
            for vertex_index, weight in enumerate(sample_weights):
                sample_weight = float(weight)
                if sample_weight == 0.0:
                    continue
                np.take(
                    coords_xyz,
                    chunk_cells[:, vertex_index],
                    axis=0,
                    out=vertex_points,
                )
                np.multiply(vertex_points, sample_weight, out=vertex_points)
                np.add(sample_points, vertex_points, out=sample_points)

            inside = np.asarray(inside_fn(sample_points[:, None, :]), dtype=bool)
            if inside.shape == (n_chunk,):
                inside_counts += inside
            elif inside.shape == (n_chunk, 1):
                inside_counts += inside[:, 0]
            else:
                return False

        fractions = inside_counts / fraction_dtype.type(weights.shape[0])
        active = fractions > 0.0
        if np.any(active):
            chunk_values = values[start:stop]
            chunk_values[active] = chunk_values[active] + fractions[active] * (
                conductivity_value - chunk_values[active]
            )
    return True


def _positive_min_radius(*radii: float) -> float:
    values = [abs(float(value)) for value in radii if abs(float(value)) > 0.0]
    return min(values) if values else 0.0


@dataclass
class ForwardSolverResult:
    """Output of a forward solve."""

    boundary_voltages: np.ndarray
    ground_truth_conductivity: np.ndarray
    node_coords: np.ndarray
    cell_connectivity: np.ndarray
    n_elements: int
    n_measurements: int
    homogeneous_voltages: np.ndarray | None = None
    forward_model_config: dict[str, Any] = field(default_factory=dict)
    error_msg: str | None = None


def _paint_axis_aligned_box_centers(
    values: np.ndarray,
    centers: np.ndarray,
    axes: tuple[tuple[float, float], ...],
    conductivity: float | complex,
    *,
    chunk_size: int = 65536,
) -> None:
    centers_arr = np.asarray(centers)
    if centers_arr.ndim != 2 or centers_arr.shape[0] == 0 or not axes:
        return
    n_values = min(int(values.shape[0]), int(centers_arr.shape[0]))
    if n_values <= 0:
        return
    block_size = max(1, min(int(chunk_size), n_values))
    work_dtype = (
        centers_arr.dtype
        if np.issubdtype(centers_arr.dtype, np.floating)
        else np.dtype(np.float64)
    )
    axis_work = np.empty(block_size, dtype=work_dtype)
    mask_work = np.empty(block_size, dtype=bool)
    axis_mask_work = np.empty(block_size, dtype=bool)
    usable_axes = tuple(
        (axis, float(center), float(radius))
        for axis, (center, radius) in enumerate(axes)
        if axis < centers_arr.shape[1]
    )
    if not usable_axes:
        return
    for start in range(0, n_values, block_size):
        stop = min(start + block_size, n_values)
        chunk_len = stop - start
        mask_chunk = mask_work[:chunk_len]
        axis_mask_chunk = axis_mask_work[:chunk_len]
        axis_work_chunk = axis_work[:chunk_len]
        first_axis = True
        for axis, center, radius in usable_axes:
            axis_values = centers_arr[start:stop, axis]
            np.subtract(axis_values, center, out=axis_work_chunk)
            np.abs(axis_work_chunk, out=axis_work_chunk)
            np.less(axis_work_chunk, radius, out=axis_mask_chunk)
            if first_axis:
                mask_chunk[...] = axis_mask_chunk
                first_axis = False
            else:
                np.logical_and(mask_chunk, axis_mask_chunk, out=mask_chunk)
        _paint_values_where(values[start:stop], mask_chunk, conductivity)


def _paint_shape(
    values: np.ndarray,
    centers: np.ndarray,
    spec: InhomogeneitySpec,
    *,
    mesh_dimension: int = 2,
    cell_vertices: np.ndarray | None = None,
    node_coords: np.ndarray | None = None,
    cell_connectivity: np.ndarray | None = None,
) -> None:
    """Paint a single inhomogeneity shape onto element-centered values."""
    if centers.size == 0:
        return

    cx, cy = spec.center_x, spec.center_y
    rx = abs(float(spec.size_x))
    ry = abs(float(spec.size_y))
    rz = abs(float(getattr(spec, "size_z", spec.size_x)))
    if rx <= 0:
        return
    if ry <= 0:
        ry = rx
    if rz <= 0:
        rz = rx

    is_3d = int(mesh_dimension) == 3 and centers.shape[1] >= 3
    sphere_radius = _positive_min_radius(rx, ry, rz) if is_3d else rx
    sample_points = None
    if is_3d and cell_vertices is not None:
        vertices = np.asarray(cell_vertices)
        if vertices.shape[0] == values.shape[0]:
            sample_points = _cell_volume_sample_points(vertices)

    if spec.shape == "circle":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            if _apply_volume_fraction_streaming(
                values,
                node_coords,
                cell_connectivity,
                lambda samples: (
                    (
                        (samples[:, :, 0] - cx) ** 2
                        + (samples[:, :, 1] - cy) ** 2
                        + (samples[:, :, 2] - cz) ** 2
                    )
                    <= sphere_radius**2
                ),
                spec.conductivity,
            ):
                return
            if sample_points is not None:
                dist2_samples = (
                    (sample_points[:, :, 0] - cx) ** 2
                    + (sample_points[:, :, 1] - cy) ** 2
                    + (sample_points[:, :, 2] - cz) ** 2
                )
                if _apply_volume_fraction(
                    values,
                    sample_points,
                    dist2_samples <= sphere_radius**2,
                    spec.conductivity,
                ):
                    return
            dist2 = (
                (centers[:, 0] - cx) ** 2
                + (centers[:, 1] - cy) ** 2
                + (centers[:, 2] - cz) ** 2
            )
            _paint_values_where(values, dist2 < sphere_radius**2, spec.conductivity)
            return
        dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        _paint_values_where(values, dist2 < rx**2, spec.conductivity)

    elif spec.shape == "ellipse":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            if _apply_volume_fraction_streaming(
                values,
                node_coords,
                cell_connectivity,
                lambda samples: (
                    (
                        ((samples[:, :, 0] - cx) / rx) ** 2
                        + ((samples[:, :, 1] - cy) / ry) ** 2
                        + ((samples[:, :, 2] - cz) / rz) ** 2
                    )
                    <= 1.0
                ),
                spec.conductivity,
            ):
                return
            if sample_points is not None:
                norm_samples = (
                    ((sample_points[:, :, 0] - cx) / rx) ** 2
                    + ((sample_points[:, :, 1] - cy) / ry) ** 2
                    + ((sample_points[:, :, 2] - cz) / rz) ** 2
                )
                if _apply_volume_fraction(
                    values,
                    sample_points,
                    norm_samples <= 1.0,
                    spec.conductivity,
                ):
                    return
            norm = (
                ((centers[:, 0] - cx) / rx) ** 2
                + ((centers[:, 1] - cy) / ry) ** 2
                + ((centers[:, 2] - cz) / rz) ** 2
            )
            _paint_values_where(values, norm < 1.0, spec.conductivity)
            return
        norm = ((centers[:, 0] - cx) / rx) ** 2 + ((centers[:, 1] - cy) / ry) ** 2
        _paint_values_where(values, norm < 1.0, spec.conductivity)

    elif spec.shape == "rectangle":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            if _apply_volume_fraction_streaming(
                values,
                node_coords,
                cell_connectivity,
                lambda samples: (
                    (np.abs(samples[:, :, 0] - cx) <= rx)
                    & (np.abs(samples[:, :, 1] - cy) <= ry)
                    & (np.abs(samples[:, :, 2] - cz) <= rz)
                ),
                spec.conductivity,
            ):
                return
            if sample_points is not None:
                mask_samples = (
                    (np.abs(sample_points[:, :, 0] - cx) <= rx)
                    & (np.abs(sample_points[:, :, 1] - cy) <= ry)
                    & (np.abs(sample_points[:, :, 2] - cz) <= rz)
                )
                if _apply_volume_fraction(
                    values, sample_points, mask_samples, spec.conductivity
                ):
                    return
            _paint_axis_aligned_box_centers(
                values,
                centers,
                ((cx, rx), (cy, ry), (cz, rz)),
                spec.conductivity,
            )
            return
        else:
            _paint_axis_aligned_box_centers(
                values,
                centers,
                ((cx, rx), (cy, ry)),
                spec.conductivity,
            )

    else:
        log.warning("Unknown shape %r, falling back to circle", spec.shape)
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            if _apply_volume_fraction_streaming(
                values,
                node_coords,
                cell_connectivity,
                lambda samples: (
                    (
                        (samples[:, :, 0] - cx) ** 2
                        + (samples[:, :, 1] - cy) ** 2
                        + (samples[:, :, 2] - cz) ** 2
                    )
                    <= sphere_radius**2
                ),
                spec.conductivity,
            ):
                return
            if sample_points is not None:
                dist2_samples = (
                    (sample_points[:, :, 0] - cx) ** 2
                    + (sample_points[:, :, 1] - cy) ** 2
                    + (sample_points[:, :, 2] - cz) ** 2
                )
                if _apply_volume_fraction(
                    values,
                    sample_points,
                    dist2_samples <= sphere_radius**2,
                    spec.conductivity,
                ):
                    return
            dist2 = (
                (centers[:, 0] - cx) ** 2
                + (centers[:, 1] - cy) ** 2
                + (centers[:, 2] - cz) ** 2
            )
            _paint_values_where(values, dist2 < sphere_radius**2, spec.conductivity)
            return
        dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        _paint_values_where(values, dist2 < rx**2, spec.conductivity)


def _paint_values_where(
    values: np.ndarray,
    mask: np.ndarray,
    conductivity: float | complex,
) -> None:
    np.copyto(values, conductivity, where=mask)


def _total_electrode_count(forward_cfg: ForwardModelConfig) -> int:
    return max(int(forward_cfg.n_elec), 1) * max(int(forward_cfg.n_rings), 1)


def _contact_impedance_vector(value: Any, *, total_electrodes: int) -> np.ndarray:
    total = max(int(total_electrodes), 1)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return np.full(total, 0.01, dtype=float)
    arr = np.asarray(value).reshape(-1)
    if arr.dtype.kind in {"O", "S", "U"}:
        arr = np.asarray([complex(item) for item in arr], dtype=np.complex128)
    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    arr = np.asarray(arr, dtype=dtype).reshape(-1)
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


def _conductivity_dtype(
    background: complex | float,
    inhomogeneities: list[InhomogeneitySpec],
) -> np.dtype:
    values = [np.asarray(background).dtype]
    values.extend(np.asarray(spec.conductivity).dtype for spec in inhomogeneities)
    return np.dtype(np.result_type(np.float64, *values))


def _gui_numeric_array(values: Any, *, real_dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return arr
    return np.asarray(arr, dtype=real_dtype)


def _forward_measurement_values(
    values: Any,
    *,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if float(noise_level) <= 0.0:
        return arr
    out = np.array(arr, copy=True)
    generator = rng if rng is not None else np.random.default_rng()
    noise_std = float(noise_level) * float(np.std(out))
    out += noise_std * generator.standard_normal(out.shape)
    return out


def _forward_result_passthrough_metadata(mapping: dict[str, Any]) -> dict[str, Any]:
    """Keep GUI/run provenance that ForwardModelConfig intentionally ignores."""

    allowed_prefixes = ("simulation_", "inhomogeneities_")
    return {
        str(key): value
        for key, value in dict(mapping or {}).items()
        if str(key).startswith(allowed_prefixes)
    }


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _finish_timing_phase(
    timings_ms: dict[str, float],
    phase_order: list[str],
    phase_name: str,
    started: float,
) -> None:
    timings_ms[str(phase_name)] = _elapsed_ms(started)
    phase_order.append(str(phase_name))


def _forward_timing_metadata(
    *,
    timings_ms: dict[str, float],
    phase_order: list[str],
    total_started: float,
    mesh_dimension: int,
) -> dict[str, Any]:
    timings = {key: float(value) for key, value in timings_ms.items()}
    total_ms = _elapsed_ms(total_started)
    timings["total"] = total_ms
    return {
        "forward_timing_schema": "eit_app_forward_timing_v1",
        "forward_timing_ms": timings,
        "forward_timing_phase_order": [*phase_order, "total"],
        "forward_timing_total_ms": total_ms,
        "forward_timing_mesh_dimension": int(mesh_dimension),
    }


def _forward_config_from_request(req: ForwardSolverRequest) -> ForwardModelConfig:
    return ForwardModelConfig.from_mapping(
        req.forward_model_config
        or {
            "mesh_dimension": req.mesh_dimension,
            "mesh_refinement": req.mesh_refinement,
            "n_elec": req.n_electrodes,
            "background_conductivity": req.background_conductivity,
            "noise_level": req.noise_level,
        }
    )


def _pattern_and_electrode_count(
    forward_cfg: ForwardModelConfig,
) -> tuple[Any, int]:
    from pyeidors.data.structures import PatternConfig
    from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh

    total_electrodes = _total_electrode_count(forward_cfg)
    pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
        mesh_tdim=forward_cfg.mesh_dimension,
        n_elec=forward_cfg.n_elec,
        n_rings=forward_cfg.n_rings,
        electrode_layout=forward_cfg.electrode_layout,
    )
    pattern = PatternConfig(
        n_elec=pattern_n_elec,
        n_rings=pattern_n_rings,
        stim_pattern=forward_cfg.stim_pattern,
        meas_pattern=forward_cfg.meas_pattern,
        electrode_layout=forward_cfg.electrode_layout,
        measurement_protocol=forward_cfg.measurement_protocol,
        custom_stim_matrix=forward_cfg.custom_stim_matrix,
        custom_meas_matrices=forward_cfg.custom_meas_matrices,
        drive_mode=forward_cfg.drive_mode,
        drive_value=forward_cfg.drive_value,
        geometry_scale_to_m=forward_cfg.geometry_scale_to_m,
        electrode_length_m_override=forward_cfg.electrode_length_m_override,
        use_meas_current=forward_cfg.use_meas_current,
        use_meas_current_next=forward_cfg.use_meas_current_next,
        rotate_meas=forward_cfg.rotate_meas,
        stim_direction=forward_cfg.stim_direction,
        meas_direction=forward_cfg.meas_direction,
        stim_first_positive=forward_cfg.stim_first_positive,
    )
    return pattern, total_electrodes


def _create_forward_system(
    *,
    forward_cfg: ForwardModelConfig,
    runtime: dict[str, Any],
    pattern: Any,
    total_electrodes: int,
) -> Any:
    from pyeidors import EITSystem

    return EITSystem(
        n_elec=total_electrodes,
        pattern_config=pattern,
        contact_impedance=_contact_impedance_vector(
            forward_cfg.contact_impedance,
            total_electrodes=total_electrodes,
        ),
        base_conductivity=forward_cfg.background_conductivity,
        solver_mode=runtime["solver_mode"],
        line_search_mode=runtime["line_search_mode"],
        linear_solver=runtime["linear_solver"],
        preconditioner=runtime["preconditioner"],
        fast_linear_path=runtime["fast_linear_path"],
        linear_backend_config={
            "solver_preset": runtime["forward_solver_preset"],
            "mat_solve_mode": runtime["forward_mat_solve"],
            "petsc_device": runtime["petsc_device"],
        },
        petsc_device=runtime["petsc_device"],
        device=runtime["device"],
        forward_backend=runtime["forward_backend"],
        mesh_family=runtime["mesh_family"],
        potential_order=forward_cfg.potential_order,
        acceleration_profile=runtime["acceleration_profile"],
    )


def _configure_forward_system_from_request(
    req: ForwardSolverRequest,
    *,
    timings_ms: dict[str, float],
    phase_order: list[str],
) -> tuple[ForwardModelConfig, dict[str, Any], Any]:
    configure_started = time.perf_counter()

    phase_started = time.perf_counter()
    forward_cfg = _forward_config_from_request(req)
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "configure.forward_config",
        phase_started,
    )

    phase_started = time.perf_counter()
    pattern, total_electrodes = _pattern_and_electrode_count(forward_cfg)
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "configure.pattern",
        phase_started,
    )

    phase_started = time.perf_counter()
    runtime = _resolve_forward_runtime(forward_cfg)
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "configure.runtime",
        phase_started,
    )

    phase_started = time.perf_counter()
    system = _create_forward_system(
        forward_cfg=forward_cfg,
        runtime=runtime,
        pattern=pattern,
        total_electrodes=total_electrodes,
    )
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "configure.system_object",
        phase_started,
    )
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "configure_system",
        configure_started,
    )
    return forward_cfg, runtime, system


def _forward_mesh_geometry_arrays(
    mesh: Any,
    *,
    mesh_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract GUI forward geometry arrays with one connectivity traversal."""

    dim = max(1, int(mesh_dimension))
    node_coords = np.asarray(mesh.geometry.x)[:, :dim].copy()
    topology = mesh.topology
    tdim = int(topology.dim)
    create_connectivity = getattr(topology, "create_connectivity", None)
    if callable(create_connectivity):
        create_connectivity(tdim, 0)
    connectivity = topology.connectivity(tdim, 0)
    index_map = topology.index_map(tdim)
    n_cells = int(index_map.size_local if index_map is not None else 0)
    if connectivity is None or n_cells <= 0:
        empty_cells = np.empty((0, 0), dtype=np.int32)
        empty_centers = np.empty((0, dim), dtype=node_coords.dtype)
        return empty_centers, node_coords, empty_cells, 0

    flat = getattr(connectivity, "array", None)
    offsets = getattr(connectivity, "offsets", None)
    cell_connectivity: np.ndarray | None = None
    if flat is not None and offsets is not None:
        offset_arr = np.asarray(offsets)
        if offset_arr.size >= n_cells + 1:
            widths = np.diff(offset_arr[: n_cells + 1])
            if widths.size and np.all(widths == widths[0]):
                vertices_per_cell = int(widths[0])
                start = int(offset_arr[0])
                stop = int(offset_arr[n_cells])
                if (
                    vertices_per_cell >= 0
                    and stop - start == n_cells * vertices_per_cell
                ):
                    flat_arr = np.asarray(flat, dtype=np.int32)
                    cell_connectivity = np.array(
                        flat_arr[start:stop].reshape(n_cells, vertices_per_cell),
                        dtype=np.int32,
                        copy=True,
                    )
    if cell_connectivity is None:
        first = np.asarray(connectivity.links(0), dtype=np.int32)
        cell_connectivity = np.empty((n_cells, first.size), dtype=np.int32)
        cell_connectivity[0] = first
        for cell_idx in range(1, n_cells):
            cell_connectivity[cell_idx] = np.asarray(
                connectivity.links(cell_idx),
                dtype=np.int32,
            )

    centers = np.zeros((n_cells, dim), dtype=node_coords.dtype)
    if cell_connectivity.shape[1]:
        work = np.empty(n_cells, dtype=node_coords.dtype)
        for local_vertex in range(cell_connectivity.shape[1]):
            indices = cell_connectivity[:, local_vertex]
            for axis in range(dim):
                np.take(node_coords[:, axis], indices, out=work)
                centers[:, axis] += work
        centers /= float(cell_connectivity.shape[1])
    return centers, node_coords, cell_connectivity, n_cells


def _setup_generated_forward_system(
    system: Any,
    *,
    forward_cfg: ForwardModelConfig,
    runtime: dict[str, Any],
) -> None:
    system.setup(
        mesh_source="generated",
        dimension=forward_cfg.mesh_dimension,
        mesh_size=forward_cfg.mesh_refinement,
        radius=forward_cfg.radius,
        height=forward_cfg.height,
        electrode_coverage=forward_cfg.electrode_coverage,
        electrode_height_ratio=forward_cfg.electrode_height_ratio,
        electrode_level_fractions=forward_cfg.electrode_level_fractions,
        z_center=forward_cfg.z_center,
        mesh_family=runtime["mesh_family"],
        geometry_version=forward_cfg.geometry_version,
        electrode_layout=forward_cfg.electrode_layout,
        initialize_inverse=False,
    )


def _resolve_forward_runtime(forward_cfg: ForwardModelConfig) -> dict[str, Any]:
    mesh_dim = int(forward_cfg.mesh_dimension)
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()

    def _auto(value: str, default: str) -> str:
        raw = str(value or "").strip().lower()
        return default if raw in {"", "auto"} else raw

    requested_profile = _auto(forward_cfg.acceleration_profile, "default")
    mesh_family = _auto(forward_cfg.mesh_family, "tetra")
    forward_backend = _auto(forward_cfg.forward_backend, "dolfinx")
    potential_order = max(1, int(getattr(forward_cfg, "potential_order", 1)))
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
        mesh_dim == 3
        and mesh_family == "hex"
        and potential_order == 1
        and (wants_gpu_request or forward_backend == "cuda_structured")
    )
    wants_3d_cuda = mesh_dim == 3 and (
        wants_gpu_request or forward_backend == "cuda_structured"
    )

    acceleration_profile = requested_profile
    if wants_3d_cuda and acceleration_profile == "default":
        acceleration_profile = "gpu3d"
    if mesh_dim != 3 and acceleration_profile in {"gpu3d", "gpu3d_fused"}:
        acceleration_profile = "default"

    if wants_structured_gpu and forward_backend == "dolfinx":
        forward_backend = "cuda_structured"
    elif not wants_structured_gpu and forward_backend == "cuda_structured":
        # The structured CUDA backend is deliberately hex-only.  Keep tetra
        # on the stable generic DOLFINx path so forward and inverse use the
        # same CEM/Jacobian convention.
        forward_backend = "dolfinx"

    petsc_device = _auto(forward_cfg.petsc_device, "cuda" if wants_3d_cuda else "cpu")
    capability: dict[str, Any] = {}
    if mesh_dim == 3 and petsc_device == "cuda":
        try:
            capability = dict(probe_petsc_cuda_runtime())
        except Exception as exc:
            capability = {"errors": {"forward_solver_policy": str(exc)}}
    probe_cache = (
        dict(capability.get("probe_cache") or {})
        if isinstance(capability.get("probe_cache"), dict)
        else {}
    )
    solver_policy = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset=_auto(forward_cfg.forward_solver_preset, "auto"),
        mesh_dim=mesh_dim,
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        capability=capability,
        prefer_amgx=True,
    )
    mat_solve_policy = resolve_3d_cuda_mat_solve_policy(
        requested_mat_solve=_auto(
            forward_cfg.forward_mat_solve, "auto" if mesh_dim == 3 else "off"
        ),
        mesh_dim=mesh_dim,
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        solver_preset=solver_policy["forward_solver_preset_effective"],
    )

    return {
        "solver_mode": _auto(
            forward_cfg.solver_mode, "fast" if mesh_dim == 3 else "strict"
        ),
        "line_search_mode": _auto(
            forward_cfg.line_search_mode, "fast" if mesh_dim == 3 else "full"
        ),
        "linear_solver": _auto(forward_cfg.linear_solver, "auto"),
        "preconditioner": _auto(forward_cfg.preconditioner, "auto"),
        "fast_linear_path": _auto(forward_cfg.fast_linear_path, "auto"),
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
        "petsc_cuda_probe_cache": probe_cache,
        "petsc_cuda_probe_cache_hit": bool(probe_cache.get("hit", False)),
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
        "device": _auto(forward_cfg.device, "cuda" if wants_3d_cuda else "auto"),
        "forward_backend": forward_backend,
        "mesh_family": mesh_family,
        "potential_order": potential_order,
        "acceleration_profile": acceleration_profile,
    }


def _forward_runtime_diagnostics(system: Any) -> dict[str, Any]:
    """Return the small runtime summary the GUI should surface to users."""
    fwd_model = getattr(system, "fwd_model", None)
    backend_diag = (
        fwd_model.get_backend_diagnostics()
        if fwd_model is not None and hasattr(fwd_model, "get_backend_diagnostics")
        else {}
    )
    mesh = getattr(system, "mesh", None)
    return {
        "mesh_family": str(getattr(mesh, "mesh_family", "") or ""),
        "potential_order": int(
            backend_diag.get(
                "potential_order", getattr(fwd_model, "potential_order", 1)
            )
            if fwd_model is not None
            else 1
        ),
        "forward_backend": str(getattr(system, "forward_backend", "") or ""),
        "forward_backend_effective": str(
            backend_diag.get(
                "forward_backend_effective",
                getattr(fwd_model, "forward_backend", "")
                if fwd_model is not None
                else "",
            )
        ),
        "solver_preset": str(backend_diag.get("solver_preset", "")),
        "forward_solver_preset": str(backend_diag.get("solver_preset", "")),
        "forward_solver_policy_reason": str(
            backend_diag.get("forward_solver_policy_reason", "")
        ),
        "forward_solver_policy_warning": str(
            backend_diag.get("forward_solver_policy_warning", "")
        ),
        "petsc_device_requested": str(backend_diag.get("petsc_device_requested", "")),
        "petsc_device_effective": str(backend_diag.get("petsc_device_effective", "")),
        "petsc_amgx_available": bool(backend_diag.get("petsc_amgx_available", False)),
        "petsc_hypre_available": bool(backend_diag.get("petsc_hypre_available", False)),
        "petsc_hypre_cuda_blacklisted": bool(
            backend_diag.get("petsc_hypre_cuda_blacklisted", False)
        ),
        "forward_mat_solve_effective": str(
            backend_diag.get("forward_mat_solve_effective", "")
        ),
        "forward_mat_solve_policy_reason": str(
            backend_diag.get("forward_mat_solve_policy_reason", "")
        ),
        "forward_ksp_session": dict(backend_diag.get("forward_ksp_session", {}) or {}),
        "torch_device": str(getattr(system, "device", "") or ""),
        "mesh_cache_hit": getattr(mesh, "_pyeidors_mesh_cache_hit", None),
        "mesh_cache_layer": getattr(mesh, "_pyeidors_mesh_cache_layer", None),
        "mesh_cache_name": getattr(mesh, "_pyeidors_mesh_cache_name", None),
    }


def _profile_lock_already_held() -> bool:
    return os.getenv("EIT_APP_BACKEND_PROFILE_LOCK_HELD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def execute_forward_request(
    req: ForwardSolverRequest,
    *,
    progress_cb: Any | None = None,
    cancelled: Any | None = None,
) -> ForwardSolverResult:
    from eit_app.backend_worker_runtime import backend_worker_profile_lock

    if _profile_lock_already_held():
        return _execute_forward_request_unlocked(
            req,
            progress_cb=progress_cb,
            cancelled=cancelled,
        )
    repo = _repo_root()
    profile_name = (
        os.getenv("EIT_APP_GUI_RUNTIME_PROFILE", "default").strip() or "default"
    )
    with backend_worker_profile_lock(repo, profile_name):
        return _execute_forward_request_unlocked(
            req,
            progress_cb=progress_cb,
            cancelled=cancelled,
        )


def _execute_forward_request_unlocked(
    req: ForwardSolverRequest,
    *,
    progress_cb: Any | None = None,
    cancelled: Any | None = None,
) -> ForwardSolverResult:
    """Execute the forward solve body in the current Python runtime.

    This pure function is intentionally free of Qt objects so it can run either
    inside the GUI process or inside a profile-isolated backend worker process.
    """

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def check_cancelled() -> bool:
        return bool(cancelled is not None and cancelled())

    total_started = time.perf_counter()
    timings_ms: dict[str, float] = {}
    phase_order: list[str] = []

    phase_started = time.perf_counter()
    from eit_app.backend_worker_runtime import prepare_inprocess_backend_runtime

    runtime_cache = prepare_inprocess_backend_runtime(repo=_repo_root())
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "prepare_runtime_cache",
        phase_started,
    )
    emit("Initializing EIT system...")
    phase_started = time.perf_counter()
    import pyeidors.core_system as _core_system  # noqa: F401
    import pyeidors.forward.eit_forward_model as _forward_model_module  # noqa: F401

    _finish_timing_phase(
        timings_ms,
        phase_order,
        "import_solver_modules",
        phase_started,
    )

    forward_cfg, runtime, system = _configure_forward_system_from_request(
        req,
        timings_ms=timings_ms,
        phase_order=phase_order,
    )
    if check_cancelled():
        raise InterruptedError("Forward solve cancelled.")

    emit("Generating mesh...")
    phase_started = time.perf_counter()
    _setup_generated_forward_system(
        system,
        forward_cfg=forward_cfg,
        runtime=runtime,
    )
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "setup_mesh_and_forward_model",
        phase_started,
    )
    if check_cancelled():
        raise InterruptedError("Forward solve cancelled.")

    emit("Building conductivity distribution...")
    phase_started = time.perf_counter()
    mesh = system.mesh if system.mesh is not None else system.fwd_model.mesh
    centers, node_coords, cell_connectivity, n_cells = _forward_mesh_geometry_arrays(
        mesh,
        mesh_dimension=forward_cfg.mesh_dimension,
    )
    sigma = np.full(
        len(centers),
        forward_cfg.background_conductivity,
        dtype=_conductivity_dtype(
            forward_cfg.background_conductivity,
            req.inhomogeneities,
        ),
    )
    for spec in req.inhomogeneities:
        _paint_shape(
            sigma,
            centers,
            spec,
            mesh_dimension=forward_cfg.mesh_dimension,
            node_coords=node_coords,
            cell_connectivity=cell_connectivity,
        )
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "build_conductivity_distribution",
        phase_started,
    )

    emit("Running forward solve...")
    phase_started = time.perf_counter()
    data = system.forward_solve(sigma)
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "solve_target_forward",
        phase_started,
    )
    if check_cancelled():
        raise InterruptedError("Forward solve cancelled.")

    emit("Computing homogeneous reference...")
    phase_started = time.perf_counter()
    sigma_homog = np.full_like(sigma, forward_cfg.background_conductivity)
    data_homog = system.forward_solve(sigma_homog)
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "solve_homogeneous_forward",
        phase_started,
    )
    if check_cancelled():
        raise InterruptedError("Forward solve cancelled.")

    phase_started = time.perf_counter()
    voltages = _forward_measurement_values(
        data.meas,
        noise_level=forward_cfg.noise_level,
    )
    homog_voltages = _forward_measurement_values(data_homog.meas)

    out_dtype = compute_dtype()
    boundary_voltages_out = _gui_numeric_array(voltages, real_dtype=out_dtype)
    ground_truth_out = _gui_numeric_array(sigma, real_dtype=out_dtype)
    homogeneous_voltages_out = _gui_numeric_array(
        homog_voltages,
        real_dtype=out_dtype,
    )
    runtime_diagnostics = _forward_runtime_diagnostics(system)
    forward_model_config = {
        **forward_cfg.to_mapping(),
        **_forward_result_passthrough_metadata(req.forward_model_config),
        **runtime,
        "backend_inprocess_cache_home": str(runtime_cache.xdg_cache_home),
        "backend_inprocess_stale_jit_locks_removed": len(
            runtime_cache.removed_stale_jit_locks
        ),
        "runtime_diagnostics": runtime_diagnostics,
    }
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "pack_forward_result",
        phase_started,
    )
    forward_model_config.update(
        _forward_timing_metadata(
            timings_ms=timings_ms,
            phase_order=phase_order,
            total_started=total_started,
            mesh_dimension=forward_cfg.mesh_dimension,
        )
    )
    result = ForwardSolverResult(
        boundary_voltages=boundary_voltages_out,
        ground_truth_conductivity=ground_truth_out,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        n_elements=n_cells,
        n_measurements=len(voltages),
        homogeneous_voltages=homogeneous_voltages_out,
        forward_model_config=forward_model_config,
    )
    emit("Forward solve complete.")
    return result


def prime_forward_setup_request(
    req: ForwardSolverRequest,
    *,
    progress_cb: Any | None = None,
    cancelled: Any | None = None,
) -> dict[str, Any]:
    """Warm mesh + forward-model static setup without running a solve."""

    from eit_app.backend_worker_runtime import backend_worker_profile_lock

    if _profile_lock_already_held():
        return _prime_forward_setup_request_unlocked(
            req,
            progress_cb=progress_cb,
            cancelled=cancelled,
        )
    repo = _repo_root()
    profile_name = (
        os.getenv("EIT_APP_GUI_RUNTIME_PROFILE", "default").strip() or "default"
    )
    with backend_worker_profile_lock(repo, profile_name):
        return _prime_forward_setup_request_unlocked(
            req,
            progress_cb=progress_cb,
            cancelled=cancelled,
        )


def _prime_forward_setup_request_unlocked(
    req: ForwardSolverRequest,
    *,
    progress_cb: Any | None = None,
    cancelled: Any | None = None,
) -> dict[str, Any]:
    """Warm mesh + forward-model static setup with profile lock already held."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def check_cancelled() -> bool:
        return bool(cancelled is not None and cancelled())

    total_started = time.perf_counter()
    timings_ms: dict[str, float] = {}
    phase_order: list[str] = []

    phase_started = time.perf_counter()
    from eit_app.backend_worker_runtime import prepare_inprocess_backend_runtime

    runtime_cache = prepare_inprocess_backend_runtime(repo=_repo_root())
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "prepare_runtime_cache",
        phase_started,
    )

    emit("Priming forward setup...")
    forward_cfg, runtime, system = _configure_forward_system_from_request(
        req,
        timings_ms=timings_ms,
        phase_order=phase_order,
    )
    if check_cancelled():
        raise InterruptedError("Forward setup prime cancelled.")

    emit("Generating mesh and forward model for setup prime...")
    phase_started = time.perf_counter()
    _setup_generated_forward_system(
        system,
        forward_cfg=forward_cfg,
        runtime=runtime,
    )
    _finish_timing_phase(
        timings_ms,
        phase_order,
        "setup_mesh_and_forward_model",
        phase_started,
    )
    if check_cancelled():
        raise InterruptedError("Forward setup prime cancelled.")

    mesh = system.mesh
    n_nodes = int(len(mesh.geometry.x)) if mesh is not None else 0
    n_cells = 0
    if mesh is not None:
        n_cells = int(mesh.topology.index_map(mesh.topology.dim).size_local)
    metadata = {
        **runtime,
        "forward_setup_prime": True,
        "backend_inprocess_cache_home": str(runtime_cache.xdg_cache_home),
        "backend_inprocess_stale_jit_locks_removed": len(
            runtime_cache.removed_stale_jit_locks
        ),
        "runtime_diagnostics": _forward_runtime_diagnostics(system),
        "n_nodes": n_nodes,
        "n_elements": n_cells,
    }
    metadata.update(
        _forward_timing_metadata(
            timings_ms=timings_ms,
            phase_order=phase_order,
            total_started=total_started,
            mesh_dimension=forward_cfg.mesh_dimension,
        )
    )
    emit("Forward setup prime complete.")
    return metadata


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def execute_forward_request_in_backend(
    req: ForwardSolverRequest,
    *,
    profile: str,
    route_reason: str,
    progress_cb: Any | None = None,
    cancelled: Any | None = None,
) -> ForwardSolverResult:
    """Run a forward request in a profile-isolated backend process."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def check_cancelled() -> bool:
        return bool(cancelled is not None and cancelled())

    repo = _repo_root()
    profile_name = str(profile or "default").strip() or "default"
    with tempfile.TemporaryDirectory(prefix="pyeidors-gui-backend-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "forward_request.h5"
        output_path = tmp_dir / "forward_result.h5"
        from eit_app.backend_worker_protocol import (
            read_forward_result,
            write_forward_request,
        )
        from eit_app.backend_worker_runtime import (
            backend_worker_command,
            backend_worker_env,
            backend_worker_profile_lock,
        )

        request_write_started = time.perf_counter()
        write_forward_request(input_path, req)
        backend_worker_request_write_ms = _elapsed_ms(request_write_started)
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
                    command="forward",
                    input_path=input_path,
                    output_path=output_path,
                    progress_cb=progress_cb,
                )
                result_read_started = time.perf_counter()
                result = read_forward_result(output_path)
                backend_worker_result_read_ms = _elapsed_ms(result_read_started)
                result.forward_model_config = {
                    **dict(result.forward_model_config or {}),
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
                    "backend_worker_request_write_ms": (
                        backend_worker_request_write_ms
                    ),
                    "backend_worker_request_duration_ms": getattr(
                        worker_meta, "request_duration_ms", 0.0
                    ),
                    "backend_worker_result_read_ms": backend_worker_result_read_ms,
                }
                emit("Backend forward solve complete.")
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
                "forward",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )
        emit(
            "Dispatching forward solve to backend "
            f"profile={profile_name} via {launch_mode} ({route_reason})..."
        )
        with backend_worker_profile_lock(repo, profile_name):
            env, cache = backend_worker_env(repo=repo, profile=profile_name)
            env["EIT_APP_BACKEND_PROFILE_LOCK_HELD"] = "1"
            if cache.removed_stale_jit_locks:
                emit(
                    "Cleaned backend JIT cache: "
                    f"{len(cache.removed_stale_jit_locks)} stale lock file(s)."
                )
                emit(f"Backend cache: {cache.xdg_cache_home}")
            subprocess_started = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            backend_worker_subprocess_duration_ms = _elapsed_ms(subprocess_started)
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
        result_read_started = time.perf_counter()
        result = read_forward_result(output_path)
        backend_worker_result_read_ms = _elapsed_ms(result_read_started)
        result.forward_model_config = {
            **dict(result.forward_model_config or {}),
            "backend_worker_profile": profile_name,
            "backend_worker_route_reason": route_reason,
            "backend_worker_process_isolated": True,
            "backend_worker_persistent": False,
            "backend_worker_launch_mode": launch_mode,
            "backend_worker_cache_home": str(cache.xdg_cache_home),
            "backend_worker_stale_jit_locks_removed": len(
                cache.removed_stale_jit_locks
            ),
            "backend_worker_request_write_ms": backend_worker_request_write_ms,
            "backend_worker_subprocess_duration_ms": (
                backend_worker_subprocess_duration_ms
            ),
            "backend_worker_result_read_ms": backend_worker_result_read_ms,
        }
        emit("Backend forward solve complete.")
        return result


class _ForwardSolverWorker(QObject):
    finished = Signal(object)  # ForwardSolverResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, request: ForwardSolverRequest) -> None:
        super().__init__()
        self._request = request
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

    def _cancelled(self) -> bool:
        thread = QThread.currentThread()
        return bool(
            self._cancel_requested
            or (thread is not None and thread.isInterruptionRequested())
        )

    def run(self) -> None:
        req = self._request
        try:
            from eit_app.backend_routing import select_forward_backend_route

            route = select_forward_backend_route(req)
            if route.external:
                self._backend_profile = route.profile
                try:
                    result = execute_forward_request_in_backend(
                        req,
                        profile=route.profile,
                        route_reason=route.reason,
                        progress_cb=self.progress.emit,
                        cancelled=self._cancelled,
                    )
                finally:
                    self._backend_profile = None
            else:
                result = execute_forward_request(
                    req,
                    progress_cb=self.progress.emit,
                    cancelled=self._cancelled,
                )
                result.forward_model_config = {
                    **dict(result.forward_model_config or {}),
                    "backend_worker_profile": route.profile,
                    "backend_worker_route_reason": route.reason,
                    "backend_worker_process_isolated": False,
                }
            self.finished.emit(result)

        except InterruptedError:
            return
        except Exception as exc:
            if self._cancelled():
                return
            log.exception("Forward solver failed")
            self.error.emit(str(exc))
            self.finished.emit(
                ForwardSolverResult(
                    boundary_voltages=np.array([]),
                    ground_truth_conductivity=np.array([]),
                    node_coords=np.array([]),
                    cell_connectivity=np.array([]),
                    n_elements=0,
                    n_measurements=0,
                    error_msg=str(exc),
                )
            )


class ForwardSolverController(QObject):
    """Manages forward problem solving in a background thread."""

    forward_done = Signal(object)  # ForwardSolverResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ForwardSolverWorker | None = None

    @property
    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def solve(self, request: ForwardSolverRequest) -> bool:
        """Start a forward solve in a background thread."""
        if self.is_busy:
            self.error.emit("A forward solve is already running.")
            return False

        self._thread = QThread()
        self._worker = _ForwardSolverWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()
        return True

    def _on_finished(self, result: ForwardSolverResult) -> None:
        self.forward_done.emit(result)
        self._cleanup()

    def _cleanup(self) -> None:
        self._stop_worker_thread(force=False, grace_ms=3000)

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
                        "Forward solver thread did not stop within %d ms%s",
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

    def shutdown(self) -> None:
        """Stop any running worker and clean up."""
        self._stop_worker_thread(force=True, grace_ms=3000)
