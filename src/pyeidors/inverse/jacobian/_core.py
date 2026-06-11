"""Shared building blocks for the EIT Jacobian calculators.

Path C progressive fusion (T75):

* Stage 1 extracted the FEM geometry handles, field-gradient interpolation
  and measurement-to-current pattern construction that
  ``DirectJacobianCalculator`` and ``EidorsJacobianAdapter`` historically
  duplicated.
* Stage 2 adds the pure-numpy assembly /
  electrode-to-measurement mapping / block-size calibration helpers
  consumed by the direct calculator. ``DirectJacobianCalculator`` is now a
  thin façade that owns the cache integration, the runtime device tracking
  and the CUDA assembly orchestration; everything else flows through this
  module.

Nothing in this module changes the V73 Jacobian sign contract or the
production GN runtime. Both calculators keep their current public
attributes (``mesh``, ``V``, ``V_sigma``, ``gdim``, ``Q_DG``, ``DG0``,
``cell_areas``) and the existing characterisation tests in
``tests/unit/test_jacobian_direct_adjoint_parity.py`` continue to gate
the contract.
"""

from __future__ import annotations

from time import perf_counter
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc


class JacobianGeometry(NamedTuple):
    """Bundled FEM geometry handles required by the Jacobian assembly paths."""

    mesh: object
    V: fem.FunctionSpace
    V_sigma: fem.FunctionSpace
    gdim: int
    Q_DG: fem.FunctionSpace
    DG0: fem.FunctionSpace
    cell_areas: np.ndarray
    linear_cell_dofs: np.ndarray | None = None
    linear_gradient_weights: np.ndarray | None = None


def build_jacobian_geometry(fwd_model) -> JacobianGeometry:
    """Construct the function spaces and per-cell volume vector once."""

    mesh = fwd_model.mesh
    gdim = int(mesh.geometry.dim)
    Q_DG = fem.functionspace(mesh, ("DG", 0, (gdim,)))
    DG0 = fem.functionspace(mesh, ("DG", 0))

    test_v = ufl.TestFunction(DG0)
    areas_vec = fem_petsc.assemble_vector(fem.form(ufl.conj(test_v) * ufl.dx))
    areas_vec.assemble()
    cell_areas = np.real_if_close(np.asarray(areas_vec.array), tol=1000).real.astype(
        float,
        copy=False,
    )
    linear_cell_dofs, linear_gradient_weights = _build_linear_gradient_cache(
        fwd_model.V,
        DG0,
        gdim,
        int(cell_areas.size),
    )

    return JacobianGeometry(
        mesh=mesh,
        V=fwd_model.V,
        V_sigma=fwd_model.V_sigma,
        gdim=gdim,
        Q_DG=Q_DG,
        DG0=DG0,
        cell_areas=cell_areas,
        linear_cell_dofs=linear_cell_dofs,
        linear_gradient_weights=linear_gradient_weights,
    )


def _build_linear_gradient_cache(
    V: fem.FunctionSpace,
    DG0: fem.FunctionSpace,
    gdim: int,
    n_cells: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Precompute exact per-cell P1 simplex gradient weights when possible."""

    if n_cells <= 0:
        return None, None

    try:
        dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)[:, :gdim]
        dofmap = V.dofmap
        dg0_dofmap = DG0.dofmap
    except (AttributeError, TypeError, RuntimeError):
        return None, None

    expected_dofs = int(gdim) + 1
    cell_dofs = np.empty((int(n_cells), expected_dofs), dtype=np.int64)
    gradient_weights = np.empty((int(n_cells), int(gdim), expected_dofs), dtype=float)
    seen_dg0_rows = np.zeros(int(n_cells), dtype=bool)
    for cell in range(int(n_cells)):
        try:
            dofs = np.asarray(dofmap.cell_dofs(cell), dtype=np.int64)
            dg0_dofs = np.asarray(dg0_dofmap.cell_dofs(cell), dtype=np.int64)
        except (AttributeError, TypeError, RuntimeError, IndexError):
            return None, None
        if dofs.size != expected_dofs:
            return None, None
        if dg0_dofs.size != 1:
            return None, None
        dg0_row = int(dg0_dofs[0])
        if dg0_row < 0 or dg0_row >= n_cells or seen_dg0_rows[dg0_row]:
            return None, None

        coords = dof_coords[dofs, :]
        affine_matrix = np.empty((dofs.size, expected_dofs), dtype=float)
        affine_matrix[:, 0] = 1.0
        affine_matrix[:, 1:] = coords
        if np.linalg.matrix_rank(affine_matrix) < expected_dofs:
            return None, None

        cell_dofs[dg0_row, :] = dofs
        gradient_weights[dg0_row, :, :] = np.linalg.pinv(affine_matrix)[1:, :]
        seen_dg0_rows[dg0_row] = True

    if not bool(np.all(seen_dg0_rows)):
        return None, None
    return cell_dofs, gradient_weights


def _complex_preserving_dtype(values) -> np.dtype:
    if isinstance(values, (np.dtype, type)):
        iterable = (values,)
    else:
        try:
            iterable = tuple(values)
        except TypeError:
            iterable = (values,)
    dtypes = [
        np.dtype(value)
        if isinstance(value, (np.dtype, type))
        else np.asarray(value).dtype
        for value in iterable
    ]
    complex_dtypes = [
        dtype for dtype in dtypes if np.issubdtype(dtype, np.complexfloating)
    ]
    if complex_dtypes:
        if any(dtype != np.dtype(np.complex64) for dtype in complex_dtypes):
            return np.dtype(np.complex128)
        return np.dtype(np.complex64)
    return np.dtype(np.float64)


def _compute_linear_cell_gradients(
    field: np.ndarray,
    cell_dofs: np.ndarray,
    gradient_weights: np.ndarray,
) -> np.ndarray:
    """Apply precomputed P1 simplex gradient weights to one nodal field."""

    values = np.asarray(field).reshape(-1)
    if values.size <= int(np.max(cell_dofs)):
        raise ValueError(
            "Field vector is shorter than the local P1 cell dof map required "
            "for Jacobian gradient assembly."
        )
    cell_values = values[cell_dofs]
    gradients = np.einsum("cgd,cd->cg", gradient_weights, cell_values, optimize=True)
    return np.asarray(gradients, dtype=_complex_preserving_dtype(values.dtype))


def compute_field_gradients(
    field_solutions: Iterable[np.ndarray],
    geometry: JacobianGeometry,
) -> list[np.ndarray]:
    """Project nodal fields onto the per-cell DG-0 vector gradient space."""

    linear_cell_dofs = getattr(geometry, "linear_cell_dofs", None)
    linear_gradient_weights = getattr(geometry, "linear_gradient_weights", None)
    if linear_cell_dofs is not None and linear_gradient_weights is not None:
        return [
            _compute_linear_cell_gradients(
                field,
                linear_cell_dofs,
                linear_gradient_weights,
            )
            for field in field_solutions
        ]

    interpolation_points = geometry.Q_DG.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()

    gradients: list[np.ndarray] = []
    for field in field_solutions:
        u_fun = fem.Function(geometry.V)
        u_fun.x.array[:] = field

        grad_expr = fem.Expression(ufl.grad(u_fun), interpolation_points)
        grad_u = fem.Function(geometry.Q_DG)
        grad_u.interpolate(grad_expr)
        gradients.append(grad_u.x.array.reshape(-1, geometry.gdim))

    return gradients


def measurement_to_current_patterns(fwd_model) -> np.ndarray:
    """Build the ``meas_pattern.T`` block driven on each electrode for the adjoint solve."""

    n_meas = fwd_model.pattern_manager.n_meas_total
    n_elec = fwd_model.n_elec

    current_patterns = np.zeros((n_elec, n_meas), dtype=float)

    meas_idx = 0
    for stim_idx in range(fwd_model.pattern_manager.n_stim):
        meas_matrix = fwd_model.pattern_manager.meas_matrices[stim_idx]
        n_meas_this_stim = meas_matrix.shape[0]
        current_patterns[:, meas_idx : meas_idx + n_meas_this_stim] = meas_matrix.T
        meas_idx += n_meas_this_stim

    return current_patterns


def assemble_jacobian_efficient_numpy(
    *,
    grad_u_all: Sequence[np.ndarray],
    adjoint_gradients: Sequence[np.ndarray],
    cell_areas: np.ndarray,
    n_meas_per_stim: Sequence[int],
    block_size: int,
) -> tuple[np.ndarray, float]:
    """Vectorised CPU assembly of the EIDORS-style efficient Jacobian.

    Mirrors the historical ``DirectJacobianCalculator._assemble_jacobian_efficient``
    NumPy branch. Returns ``(jacobian, elapsed_seconds)`` so the caller can
    record the assembly-only timing alongside its own wall-clock metrics.
    """

    n_measurements = int(len(adjoint_gradients))
    n_elements = int(len(cell_areas))
    block = int(max(1, min(block_size, n_elements or 1)))

    value_dtype = _complex_preserving_dtype((*grad_u_all, *adjoint_gradients))
    jacobian = np.zeros((n_measurements, n_elements), dtype=value_dtype)
    cell_areas_arr = np.asarray(cell_areas, dtype=float)

    t0 = perf_counter()
    meas_idx = 0
    for stim_idx, grad_u in enumerate(grad_u_all):
        n_meas_this_stim = int(n_meas_per_stim[stim_idx])
        adjoint_block = np.asarray(
            adjoint_gradients[meas_idx : meas_idx + n_meas_this_stim],
            dtype=value_dtype,
        )
        grad_u_arr = np.asarray(grad_u, dtype=value_dtype)
        for start in range(0, n_elements, block):
            end = min(start + block, n_elements)
            sensitivity_block = np.einsum(
                "eg,meg->me",
                grad_u_arr[start:end, :],
                adjoint_block[:, start:end, :],
                optimize=True,
            )
            jacobian[meas_idx : meas_idx + n_meas_this_stim, start:end] = (
                sensitivity_block * cell_areas_arr[None, start:end]
            )
        meas_idx += n_meas_this_stim
    elapsed = float(perf_counter() - t0)
    return jacobian, elapsed


def assemble_jacobian_traditional(
    grad_u_all: Sequence[np.ndarray],
    grad_bu_all: Sequence[np.ndarray],
    cell_areas: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Assemble the per-electrode Jacobian via the all-pairs gradient product."""

    value_dtype = _complex_preserving_dtype((*grad_u_all, *grad_bu_all))
    cell_areas_arr = np.asarray(cell_areas, dtype=float)
    t0 = perf_counter()
    n_u = len(grad_u_all)
    n_bu = len(grad_bu_all)
    jacobian = np.empty((n_u * n_bu, cell_areas_arr.size), dtype=value_dtype)
    row = 0
    for grad_u in grad_u_all:
        for grad_bu in grad_bu_all:
            sensitivity = np.sum(grad_bu * grad_u, axis=1) * cell_areas_arr
            jacobian[row, :] = sensitivity
            row += 1
    elapsed = float(perf_counter() - t0)
    return np.ascontiguousarray(jacobian, dtype=value_dtype), elapsed


def convert_electrode_to_measurement_jacobian(
    electrode_jacobian: np.ndarray,
    *,
    n_stim: int,
    n_elec: int,
    meas_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Project the per-electrode traditional Jacobian onto the measurement basis."""

    row_counts: list[int] = []
    result_dtype = np.asarray(electrode_jacobian).dtype
    for stim_idx in range(int(n_stim)):
        meas_matrix = meas_matrices[stim_idx]
        row_counts.append(int(meas_matrix.shape[0]))
        result_dtype = np.result_type(result_dtype, np.asarray(meas_matrix).dtype)
    n_rows = int(sum(row_counts))
    n_cols = int(electrode_jacobian.shape[1])
    measurement_jacobian = np.empty((n_rows, n_cols), dtype=result_dtype)
    row_start = 0
    for stim_idx in range(int(n_stim)):
        elec_start = stim_idx * int(n_elec)
        elec_end = (stim_idx + 1) * int(n_elec)
        electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]
        meas_matrix = meas_matrices[stim_idx]
        row_end = row_start + row_counts[stim_idx]
        measurement_jacobian[row_start:row_end, :] = (
            meas_matrix @ electrode_jac_for_stim
        )
        row_start = row_end
    return np.ascontiguousarray(measurement_jacobian, dtype=result_dtype)


def calibrate_block_size_once(
    *,
    grad_u_all: Sequence[np.ndarray],
    adjoint_gradients: Sequence[np.ndarray],
    n_elements: int,
    candidates: Sequence[int],
    sample_meas_count: int,
) -> int:
    """Time the candidate block sizes on a small slice and return the winner."""

    candidate_tuple = tuple(int(c) for c in candidates) if candidates else ()
    fallback_max = candidate_tuple[-1] if candidate_tuple else 256

    if not grad_u_all or not adjoint_gradients:
        return int(min(int(n_elements), fallback_max))

    sample_grad_u = np.asarray(grad_u_all[0])
    if sample_grad_u.ndim != 2 or sample_grad_u.shape[0] == 0:
        return int(min(int(n_elements), fallback_max))

    local_meas = int(sample_meas_count)
    sample_adjoint = np.asarray(adjoint_gradients[:local_meas])
    if sample_adjoint.ndim != 3 or sample_adjoint.shape[1] == 0:
        return int(min(int(n_elements), fallback_max))

    n_sample_elem = int(min(sample_grad_u.shape[0], 2048))
    sample_grad_u = sample_grad_u[:n_sample_elem, :]
    sample_adjoint = sample_adjoint[:, :n_sample_elem, :]

    bounded = sorted(
        {max(16, min(int(candidate), n_sample_elem)) for candidate in candidate_tuple}
    )
    if not bounded:
        return int(min(int(n_elements), 256))

    best_size = bounded[0]
    best_elapsed = float("inf")
    for candidate in bounded:
        t0 = perf_counter()
        for start in range(0, n_sample_elem, candidate):
            end = min(start + candidate, n_sample_elem)
            _ = np.einsum(
                "eg,meg->me",
                sample_grad_u[start:end, :],
                sample_adjoint[:, start:end, :],
                optimize=True,
            )
        elapsed = perf_counter() - t0
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_size = candidate
    return int(max(1, min(best_size, int(n_elements))))
