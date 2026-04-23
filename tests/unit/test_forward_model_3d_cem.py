"""Smoke tests for 3D CEM forward model assembly/solve."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

import pyeidors.forward.eit_forward_model as forward_module
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    create_cylinder_3d_eit_mesh,
)
from pyeidors.inverse.dual_mesh import DualMesh, VoxelGrid


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_forward_model_3d_solves_and_returns_finite_measurements(tmp_path):
    clear_process_forward_setup_cache()
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="cyl3d_forward",
    )

    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    fwd = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=np.full(16, 1e-5, dtype=float),
        mesh=mesh,
        linear_backend="scipy",
    )

    sigma = fem.Function(fwd.V_sigma)
    sigma.x.array[:] = 1.0
    u_all, U_all = fwd.forward_solve(sigma)
    assert len(u_all) == fwd.pattern_manager.n_stim
    assert U_all.shape == (fwd.pattern_manager.n_stim, 16)
    assert np.all(np.isfinite(U_all))

    img_ref = EITImage(elem_data=np.full_like(sigma.x.array, 1.0), fwd_model=fwd)
    ref_data, _ = fwd.fwd_solve(img_ref)
    assert ref_data.meas.shape[0] == fwd.pattern_manager.n_meas_total
    assert np.all(np.isfinite(ref_data.meas))

    coords = mesh.coordinates()
    lower = coords.min(axis=0) - 1e-9
    upper = coords.max(axis=0) + 1e-9
    coarse = VoxelGrid.from_bounds(
        lower,
        upper,
        shape=(2, 2, 2),
        name="real-cem-coarse-inverse-voxels",
    )
    dual = DualMesh(fine_mesh=mesh, coarse_mesh=coarse)
    assert dual.n_fine_cells == sigma.x.array.size
    assert dual.n_fine_cells > dual.n_coarse_cells

    occupancy = np.asarray(dual.coarse2fine.sum(axis=0)).reshape(-1)
    active = np.flatnonzero(occupancy > 0.0)
    assert active.size > 0
    coarse_centers = coarse.cell_centers()
    active_idx = int(active[np.argmax(coarse_centers[active, 0])])
    coarse_delta = np.zeros(coarse.num_cells(), dtype=float)
    coarse_delta[active_idx] = 0.8
    fine_delta = dual.project_to_fine(coarse_delta)
    restricted = dual.restrict_to_coarse(fine_delta)
    assert np.count_nonzero(fine_delta) == int(occupancy[active_idx])
    assert restricted[active_idx] == pytest.approx(coarse_delta[active_idx])

    sigma_perturbed = np.full_like(sigma.x.array, 1.0)
    sigma_perturbed[:] += fine_delta
    img_tgt = EITImage(elem_data=sigma_perturbed, fwd_model=fwd)
    tgt_data, _ = fwd.fwd_solve(img_tgt)
    assert np.all(np.isfinite(tgt_data.meas))
    assert not np.allclose(ref_data.meas, tgt_data.meas)


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_forward_model_3d_reuses_static_setup_bundle(tmp_path):
    clear_process_forward_setup_cache()
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=2,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="cyl3d_forward_bundle",
    )

    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    z = np.full(16, 1e-5, dtype=float)
    fwd1 = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=z,
        mesh=mesh,
        linear_backend="scipy",
    )
    fwd2 = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=z,
        mesh=mesh,
        linear_backend="scipy",
    )

    lookup1 = fwd1.get_backend_diagnostics().get("static_setup_lookup", {})
    lookup2 = fwd2.get_backend_diagnostics().get("static_setup_lookup", {})
    assert lookup1.get("hit") is False
    assert lookup2.get("hit") is True
    assert lookup2.get("layer") == "process"
    assert fwd1.V is fwd2.V
    assert fwd1.V_sigma is fwd2.V_sigma
    assert fwd1.pattern_manager is not fwd2.pattern_manager
    assert np.array_equal(
        fwd1.pattern_manager.stim_matrix, fwd2.pattern_manager.stim_matrix
    )
    assert np.array_equal(
        fwd1.pattern_manager._meas_projection, fwd2.pattern_manager._meas_projection
    )
    assert np.array_equal(
        fwd1.pattern_manager.meas_selector, fwd2.pattern_manager.meas_selector
    )
    assert fwd1.M is fwd2.M


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
@pytest.mark.skipif(forward_module.PETSc is None, reason="petsc4py not available")
def test_forward_model_3d_petsc_gamg_smoke_records_multi_rhs_diagnostics(tmp_path):
    clear_process_forward_setup_cache()
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=8,
        radius=0.2,
        height=0.18,
        refinement=2,
        electrode_coverage=0.45,
        output_dir=str(tmp_path),
        mesh_name="cyl3d_petsc_gamg",
    )

    pattern = PatternConfig(
        n_elec=8,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    fwd = EITForwardModel(
        n_elec=8,
        pattern_config=pattern,
        z=np.full(8, 1e-5, dtype=float),
        mesh=mesh,
        linear_backend="petsc",
        backend_config={"solver_preset": "3d_gamg", "mat_solve_mode": "auto"},
        performance_mode="aggressive",
    )

    sigma = fem.Function(fwd.V_sigma)
    sigma.x.array[:] = 1.0
    try:
        u_all, U_all = fwd.forward_solve(sigma)
    except RuntimeError as exc:
        if "gamg" in str(exc).lower() and "unknown" in str(exc).lower():
            pytest.skip(f"PETSc GAMG unavailable in this runtime: {exc}")
        raise

    assert len(u_all) == fwd.pattern_manager.n_stim
    assert U_all.shape == (fwd.pattern_manager.n_stim, 8)
    assert np.all(np.isfinite(U_all))

    diag = fwd.get_backend_diagnostics()
    assert diag["solver_preset"] == "3d_gamg"
    assert diag["ksp_type"] == "fgmres"
    assert diag["pc_type"] == "gamg"
    assert diag["pc_gamg_type"] == "agg"
    assert diag["forward_rhs_count"] == fwd.pattern_manager.n_stim
    assert str(diag["forward_factor_backend"]).startswith("petsc-ksp")
    assert str(diag["pc_type"]).lower() != "lu"
    assert int(diag["forward_ksp_setup_count"]) >= 1
    assert diag["forward_reuse_preconditioner_requested"] is True
    assert diag["forward_reuse_preconditioner_applied"] in {True, False, None}

    effective = diag["forward_mat_solve_effective"]
    assert effective in {"matsolve", "vec-loop"}
    if effective == "matsolve":
        assert diag["forward_ksp_mat_solve_count"] == 1
        assert diag["forward_ksp_solve_count"] == 0
        assert diag["forward_ksp_converged"] is not False
    else:
        assert diag["forward_ksp_mat_solve_count"] == 0
        solve_count = int(diag["forward_ksp_solve_count"])
        assert 1 <= solve_count <= fwd.pattern_manager.n_stim
        if diag.get("forward_ksp_converged") is False:
            assert diag.get("fallback_reason") or diag.get(
                "forward_mat_solve_fallback_reason"
            )
        else:
            assert solve_count == fwd.pattern_manager.n_stim
