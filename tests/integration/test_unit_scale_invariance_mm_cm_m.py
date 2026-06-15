"""Integration test: mm/cm/m representations should produce consistent voltages."""

from __future__ import annotations

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import build_eit_mesh
from pyeidors.forward.complex_support import petsc_scalar_dtype
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.utils.numeric_ops import real_array_if_zero_imaginary


def _build_square_eit_mesh(side_length: float):
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 18, 18)
    mesh.geometry.x[:, : mesh.geometry.dim] *= float(side_length)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=float)
    for i, facet in enumerate(boundary_facets):
        centroids[i, :] = coords[f2v.links(int(facet))].mean(axis=0)

    scale = float(side_length)
    x = np.round(centroids[:, 0] / scale, decimals=7)
    y = np.round(centroids[:, 1] / scale, decimals=7)
    eps = 1e-10
    t = np.zeros_like(x)
    xmin, ymin = 0.0, 0.0
    xmax, ymax = 1.0, 1.0
    left = np.isclose(x, xmin, atol=eps)
    top = (~left) & np.isclose(y, ymax, atol=eps)
    right = (~left) & (~top) & np.isclose(x, xmax, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, ymin, atol=eps)
    t[left] = (y[left] - ymin) / (ymax - ymin)
    t[top] = 1.0 + (x[top] - xmin) / (xmax - xmin)
    t[right] = 2.0 + (ymax - y[right]) / (ymax - ymin)
    t[bottom] = 3.0 + (xmax - x[bottom]) / (xmax - xmin)
    seg = 4.0 / 16
    bin_t = np.clip(t + 1e-12, 0.0, 4.0 - 1e-12)
    tags = (np.floor(bin_t / seg).astype(np.int32) + 2).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{i + 1}": i + 2 for i in range(16)}
    return build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=float(side_length),
    )


def _solve_voltage(side_length: float, geometry_scale_to_m: float) -> np.ndarray:
    mesh = _build_square_eit_mesh(side_length)
    config = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=5e-5,
        geometry_scale_to_m=geometry_scale_to_m,
        use_meas_current=False,
        rotate_meas=True,
    )
    z = np.full(16, 1e-5, dtype=float)
    model = EITForwardModel(n_elec=16, pattern_config=config, z=z, mesh=mesh)
    n_elem = int(
        model.V_sigma.dofmap.index_map.size_local * model.V_sigma.dofmap.index_map_bs
    )
    sigma = np.ones(n_elem, dtype=float)
    data, _ = model.fwd_solve(EITImage(elem_data=sigma, fwd_model=model))
    return real_array_if_zero_imaginary(data.meas, name="forward measurements")


def test_same_object_mm_cm_m_invariance():
    # Same physical object: side length 0.2 m represented in three coordinate systems.
    v_m = _solve_voltage(side_length=0.2, geometry_scale_to_m=1.0)
    v_cm = _solve_voltage(side_length=20.0, geometry_scale_to_m=0.01)
    v_mm = _solve_voltage(side_length=200.0, geometry_scale_to_m=0.001)

    amp_m = float(np.max(np.abs(v_m)))
    amp_cm = float(np.max(np.abs(v_cm)))
    amp_mm = float(np.max(np.abs(v_mm)))
    ratio_cm = amp_cm / amp_m
    ratio_mm = amp_mm / amp_m

    rel_l2_cm = float(np.linalg.norm(v_cm - v_m) / np.linalg.norm(v_m))
    rel_l2_mm = float(np.linalg.norm(v_mm - v_m) / np.linalg.norm(v_m))
    rel_l2_tol = (
        1e-4
        if petsc_scalar_dtype() in {np.dtype(np.float32), np.dtype(np.complex64)}
        else 1e-6
    )

    assert 0.99 <= ratio_cm <= 1.01, (
        f"cm/m amplitude ratio out of range: {ratio_cm:.12f}; "
        f"amp_m={amp_m:.6e}, amp_cm={amp_cm:.6e}"
    )
    assert 0.99 <= ratio_mm <= 1.01, (
        f"mm/m amplitude ratio out of range: {ratio_mm:.12f}; "
        f"amp_m={amp_m:.6e}, amp_mm={amp_mm:.6e}"
    )
    assert rel_l2_cm <= rel_l2_tol, (
        f"cm/m vector relative L2 too large: {rel_l2_cm:.3e}"
    )
    assert rel_l2_mm <= rel_l2_tol, (
        f"mm/m vector relative L2 too large: {rel_l2_mm:.3e}"
    )
