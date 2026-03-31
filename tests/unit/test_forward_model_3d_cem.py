"""Smoke tests for 3D CEM forward model assembly/solve."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    create_cylinder_3d_eit_mesh,
)


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_forward_model_3d_solves_and_returns_finite_measurements(tmp_path):
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

    sigma_perturbed = np.full_like(sigma.x.array, 1.0)
    sigma_perturbed[: max(1, sigma_perturbed.size // 20)] = 1.8
    img_tgt = EITImage(elem_data=sigma_perturbed, fwd_model=fwd)
    tgt_data, _ = fwd.fwd_solve(img_tgt)
    assert np.all(np.isfinite(tgt_data.meas))
    assert not np.allclose(ref_data.meas, tgt_data.meas)
