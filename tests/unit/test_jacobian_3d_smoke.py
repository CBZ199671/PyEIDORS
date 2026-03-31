"""Smoke tests for 3D Jacobian calculation."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    create_cylinder_3d_eit_mesh,
)
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_direct_jacobian_3d_shape_and_finite(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="cyl3d_jac",
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
    calc = DirectJacobianCalculator(fwd)

    sigma = fem.Function(fwd.V_sigma)
    sigma.x.array[:] = 1.0
    jacobian = calc.calculate(sigma, method="efficient")

    assert jacobian.shape[0] == fwd.pattern_manager.n_meas_total
    assert jacobian.shape[1] == sigma.x.array.size
    assert np.all(np.isfinite(jacobian))
