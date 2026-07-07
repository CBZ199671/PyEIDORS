"""Small real-ish GN smokes for the linearized Jacobian runtime path."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import build_eit_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    create_cylinder_3d_eit_mesh,
)
from pyeidors.inverse.regularization.smoothness import TikhonovRegularization
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.inverse.solvers.gauss_newton import GaussNewtonReconstructor


def _make_tagged_unit_square(*, n_elec: int = 4):
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    fdim = mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]

    centroids = np.zeros((boundary_facets.size, 2), dtype=np.float64)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)

    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1e-10
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])

    tags = (
        np.floor(np.clip(t, 0.0, 4.0 - eps) / (4.0 / float(n_elec))).astype(np.int32)
        + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(n_elec)}
    return build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=1.0,
    )


def _run_linearized_smoke(eit_mesh, *, n_elec: int) -> dict[str, object]:
    pattern = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    fwd = EITForwardModel(
        n_elec=n_elec,
        pattern_config=pattern,
        z=np.full(n_elec, 1e-5, dtype=np.float64),
        mesh=eit_mesh,
        linear_backend="scipy",
    )

    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))
    reconstructor = GaussNewtonReconstructor(
        fwd_model=fwd,
        regularization=TikhonovRegularization(fwd, alpha=1.0),
        max_iterations=1,
        min_iterations=1,
        regularization_param=0.1,
        solver_mode="fast",
        line_search_mode="fast",
        fast_linear_path="pcg",
        preconditioner="diag",
        verbose=False,
        clip_values=None,
        min_step=1.0,
        negate_jacobian=False,
    )

    output = reconstructor.reconstruct(
        measured,
        initial_conductivity=base.copy(),
        jacobian_method="linearized",
    )
    backend_info = output.diagnostics["backend_info"]

    assert output.iterations == 1
    assert np.isfinite(output.final_residual)
    assert np.all(np.isfinite(output.conductivity.x.array))
    assert backend_info["jacobian_representation"] == "jacobian_linearization"
    assert backend_info["dense_jacobian_materialized"] is False
    assert backend_info["startup_cache_lookup"]["reason"] == "operator_jacobian"
    assert "pcg" in str(backend_info["fast_solver_path"])
    return backend_info


def test_gn_linearized_jacobian_smoke_2d_real_fem():
    backend_info = _run_linearized_smoke(_make_tagged_unit_square(n_elec=4), n_elec=4)
    assert backend_info["jacobian_shape"][1] > 0


def test_lazy_adjoint_linearization_matches_finite_difference_and_transpose_2d_real_fem():
    clear_process_forward_setup_cache()
    n_elec = 4
    pattern = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    fwd = EITForwardModel(
        n_elec=n_elec,
        pattern_config=pattern,
        z=np.full(n_elec, 1e-5, dtype=np.float64),
        mesh=_make_tagged_unit_square(n_elec=n_elec),
        linear_backend="scipy",
    )
    sigma = fem.Function(fwd.V_sigma)
    sigma.x.array[:] = 1.0
    jac_calc = EidorsJacobianAdapter(fwd)

    lazy = jac_calc.linearize_lazy(
        sigma,
        diag_exact_max_measurements=10_000,
        diag_chunk_size=2,
    )
    vector = np.linspace(0.1, 0.9, lazy.shape[1], dtype=np.float64) * 1.0e4
    residual = np.linspace(-0.2, 0.3, lazy.shape[0], dtype=np.float64)

    base_data, _ = fwd.fwd_solve(
        EITImage(elem_data=np.ones(lazy.shape[1]), fwd_model=fwd)
    )
    # The direction vector is intentionally large enough to stress the
    # matrix-free path, so the finite-difference scale must be small enough to
    # stay in the infinitesimal regime.  A 1e-4 step perturbs sigma by up to
    # 0.9 here and measures nonlinear response rather than the Jacobian action.
    eps = (
        1.0e-5 if np.asarray(base_data.meas).dtype == np.dtype(np.complex64) else 1.0e-6
    )
    perturbed_data, _ = fwd.fwd_solve(
        EITImage(elem_data=np.ones(lazy.shape[1]) + eps * vector, fwd_model=fwd)
    )
    finite_difference = (perturbed_data.meas - base_data.meas) / eps
    jv = lazy.matvec(vector)
    finite_difference_tol = (
        {"rtol": 5e-2, "atol": 1e-5}
        if np.asarray(finite_difference).dtype == np.dtype(np.complex64)
        else {"rtol": 2e-3, "atol": 2e-6}
    )
    np.testing.assert_allclose(jv, finite_difference, **finite_difference_tol)
    lhs = np.real_if_close(np.dot(jv, residual))
    rhs = np.real_if_close(np.dot(vector, lazy.rmatvec(residual)))
    np.testing.assert_allclose(
        float(np.real(lhs)),
        float(np.real(rhs)),
        rtol=1e-8,
        atol=1e-10,
    )

    eye = np.eye(lazy.shape[1], dtype=np.float64)
    lazy_dense = np.column_stack(
        [lazy.matvec(eye[:, idx]) for idx in range(lazy.shape[1])]
    )
    np.testing.assert_allclose(
        lazy.hessian_diag(),
        np.sum(lazy_dense * lazy_dense, axis=0),
        rtol=1e-6,
        atol=1e-8,
    )
    assert lazy.last_diag_info["mode"] == "lazy_chunked_exact"

    approx_diag = lazy.hessian_diag(diag_mode="approx")
    assert approx_diag.shape == (lazy.shape[1],)
    assert np.all(np.isfinite(approx_diag))
    assert lazy.last_diag_info["mode"] == "lazy_approx"

    sampled_diag = lazy.hessian_diag(
        diag_mode="batch_noser",
        diag_batch_max_measurements=2,
    )
    assert sampled_diag.shape == (lazy.shape[1],)
    assert np.all(np.isfinite(sampled_diag))
    assert lazy.last_diag_info["mode"] == "lazy_batch_noser"
    assert lazy.last_diag_info["sampled_measurements"] <= lazy.shape[0]


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_gn_linearized_jacobian_smoke_3d_real_fem(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=4,
        radius=0.1,
        height=0.08,
        refinement=1,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="tfpx006_linearized_3d",
    )
    backend_info = _run_linearized_smoke(mesh, n_elec=4)
    assert backend_info["jacobian_shape"][1] > 0
