"""Shared building blocks for the EIT Jacobian calculators.

Stage 1 of the Path C progressive fusion (T75): factor out the
geometry setup, field-gradient interpolation and measurement-to-current
pattern construction that ``DirectJacobianCalculator`` and
``EidorsStyleAdjointJacobian`` historically duplicated. The two
classes still own their own assembly + sign convention; subsequent
stages will migrate the assembly paths and ``linearize_lazy`` plumbing
in here so the Adjoint class can shrink to a sign-flip adapter.

Nothing in this module changes the V73 Jacobian sign contract or the
production GN runtime. Both calculators keep their current public
attributes (``mesh``, ``V``, ``V_sigma``, ``gdim``, ``Q_DG``, ``DG0``,
``cell_areas``) and the existing characterisation tests in
``tests/unit/test_jacobian_direct_adjoint_parity.py`` continue to gate
the contract.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

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
        current_patterns[
            :, meas_idx : meas_idx + n_meas_this_stim
        ] = meas_matrix.T
        meas_idx += n_meas_this_stim

    return current_patterns
