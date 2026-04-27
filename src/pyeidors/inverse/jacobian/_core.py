"""Shared building blocks for the EIT Jacobian calculators.

Path C progressive fusion (T75):

* Stage 1 extracted the FEM geometry handles, field-gradient interpolation
  and measurement-to-current pattern construction that
  ``DirectJacobianCalculator`` and ``EidorsJacobianAdapter`` historically
  duplicated.
* Stage 2 adds the shared adjoint-fields solve and the pure-numpy
  assembly / electrode-to-measurement mapping / block-size calibration
  helpers consumed by the direct calculator. ``DirectJacobianCalculator``
  is now a thin façade that owns the cache integration, the runtime
  device tracking and the CUDA assembly orchestration; everything else
  flows through this module.

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


def build_jacobian_geometry(fwd_model) -> JacobianGeometry:
    """Construct the function spaces and per-cell volume vector once."""

    mesh = fwd_model.mesh
    gdim = int(mesh.geometry.dim)
    Q_DG = fem.functionspace(mesh, ("DG", 0, (gdim,)))
    DG0 = fem.functionspace(mesh, ("DG", 0))

    test_v = ufl.TestFunction(DG0)
    areas_vec = fem_petsc.assemble_vector(fem.form(test_v * ufl.dx))
    areas_vec.assemble()
    cell_areas = np.asarray(areas_vec.array, dtype=float)

    return JacobianGeometry(
        mesh=mesh,
        V=fwd_model.V,
        V_sigma=fwd_model.V_sigma,
        gdim=gdim,
        Q_DG=Q_DG,
        DG0=DG0,
        cell_areas=cell_areas,
    )


def compute_field_gradients(
    field_solutions: Iterable[np.ndarray],
    geometry: JacobianGeometry,
) -> list[np.ndarray]:
    """Project nodal fields onto the per-cell DG-0 vector gradient space."""

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


def compute_adjoint_fields_efficient(
    fwd_model,
    sigma,
    geometry: JacobianGeometry,
) -> list[np.ndarray]:
    """Solve the adjoint problem for measurement-current patterns and return per-meas gradients."""

    adjoint_patterns = measurement_to_current_patterns(fwd_model)
    adjoint_fields, _ = fwd_model.forward_solve(sigma, adjoint_patterns)
    return compute_field_gradients(adjoint_fields, geometry)


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

    jacobian = np.zeros((n_measurements, n_elements), dtype=float)
    cell_areas_arr = np.asarray(cell_areas, dtype=float)

    t0 = perf_counter()
    meas_idx = 0
    for stim_idx, grad_u in enumerate(grad_u_all):
        n_meas_this_stim = int(n_meas_per_stim[stim_idx])
        adjoint_block = np.asarray(
            adjoint_gradients[meas_idx : meas_idx + n_meas_this_stim],
            dtype=float,
        )
        grad_u_arr = np.asarray(grad_u, dtype=float)
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

    cell_areas_arr = np.asarray(cell_areas, dtype=float)
    t0 = perf_counter()
    jacobian_blocks: list[np.ndarray] = []
    for grad_u in grad_u_all:
        derivatives = []
        for grad_bu in grad_bu_all:
            sensitivity = np.sum(grad_bu * grad_u, axis=1) * cell_areas_arr
            derivatives.append(sensitivity)
        jacobian_blocks.append(np.array(derivatives))
    elapsed = float(perf_counter() - t0)
    return np.vstack(jacobian_blocks), elapsed


def convert_electrode_to_measurement_jacobian(
    electrode_jacobian: np.ndarray,
    *,
    n_stim: int,
    n_elec: int,
    meas_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Project the per-electrode traditional Jacobian onto the measurement basis."""

    measurement_jacobian_blocks: list[np.ndarray] = []
    for stim_idx in range(int(n_stim)):
        elec_start = stim_idx * int(n_elec)
        elec_end = (stim_idx + 1) * int(n_elec)
        electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]
        meas_matrix = meas_matrices[stim_idx]
        measurement_jacobian_blocks.append(meas_matrix @ electrode_jac_for_stim)
    return np.vstack(measurement_jacobian_blocks)


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

    sample_grad_u = np.asarray(grad_u_all[0], dtype=float)
    if sample_grad_u.ndim != 2 or sample_grad_u.shape[0] == 0:
        return int(min(int(n_elements), fallback_max))

    local_meas = int(sample_meas_count)
    sample_adjoint = np.asarray(adjoint_gradients[:local_meas], dtype=float)
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
