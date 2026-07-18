"""3D same-mesh parity gate for classic and Robin CEM formulations."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import EITForwardModel, RobinTransconductanceForwardModel
from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    create_cylinder_3d_eit_mesh,
)


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_3d_robin_transconductance_matches_classic_cem(tmp_path) -> None:
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=8,
        radius=0.2,
        height=0.18,
        refinement=1,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="robin_cem_3d_parity",
    )
    pattern = PatternConfig(
        n_elec=8,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    impedance = np.full(8, 1e-2, dtype=float)
    classic = EITForwardModel(
        n_elec=8,
        pattern_config=pattern,
        z=impedance,
        mesh=mesh,
        linear_backend="scipy",
    )
    robin = RobinTransconductanceForwardModel(
        n_elec=8,
        pattern_config=pattern,
        z=impedance,
        mesh=mesh,
        linear_backend="scipy",
    )
    sigma_classic = fem.Function(classic.V_sigma)
    sigma_robin = fem.Function(robin.V_sigma)
    sigma_classic.x.array[:] = 0.25
    sigma_robin.x.array[:] = 0.25

    potential_classic, voltage_classic = classic.forward_solve(sigma_classic)
    potential_robin, voltage_robin = robin.forward_solve(sigma_robin)
    real_dtype = np.empty(1, dtype=voltage_classic.dtype).real.dtype
    rtol = 2e-5 if real_dtype == np.dtype(np.float32) else 1e-10

    voltage_relative_l2 = float(
        np.linalg.norm(voltage_robin - voltage_classic)
        / max(float(np.linalg.norm(voltage_classic)), 1.0)
    )
    potential_robin_matrix = np.column_stack(potential_robin)
    potential_classic_matrix = np.column_stack(potential_classic)
    potential_relative_l2 = float(
        np.linalg.norm(potential_robin_matrix - potential_classic_matrix)
        / max(float(np.linalg.norm(potential_classic_matrix)), 1.0)
    )
    assert voltage_relative_l2 <= rtol
    assert potential_relative_l2 <= rtol
    assert np.all(np.isfinite(voltage_robin))
    assert np.all(np.isfinite(potential_robin_matrix))
    diagnostics = robin.get_backend_diagnostics()
    assert diagnostics["robin_transconductance_rank"] == 7
    assert (
        diagnostics["robin_current_balance_residual"]
        <= diagnostics["robin_current_balance_tolerance"]
    )
