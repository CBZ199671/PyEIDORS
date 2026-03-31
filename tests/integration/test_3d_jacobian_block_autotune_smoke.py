"""Smoke test for 3D Jacobian block-size auto-tuning diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_3d_jacobian_block_autotune_smoke(tmp_path: Path):
    mesh = load_or_create_mesh(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="jacobian_block_tune_smoke",
        n_elec=8,
        dimension=3,
        radius=0.12,
        height=0.12,
        refinement=1,
        electrode_height_ratio=0.2,
        z_center=0.0,
        electrode_coverage=0.5,
    )
    pattern = PatternConfig(
        n_elec=8,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    system = EITSystem(
        n_elec=8,
        pattern_config=pattern,
        contact_impedance=np.full(8, 1e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        solver_mode="fast",
        linear_solver="auto",
        jacobian_update_every=2,
        jacobian_reuse_tol=1e-3,
        line_search_mode="fast",
        jacobian_block_tune="auto",
        jacobian_block_size=0,
        jacobian_block_candidates=(32, 64, 128),
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = 1
    system.reconstructor.verbose = False

    baseline = system.create_homogeneous_image(conductivity=1.0)
    baseline_data = system.forward_solve(baseline)
    sigma = np.asarray(baseline.elem_data, dtype=float).copy()
    sigma[: max(1, sigma.size // 8)] = 1.3
    target = EITImage(elem_data=sigma, fwd_model=system.fwd_model)
    target_data = system.forward_solve(target)

    out = system.inverse_solve(data=target_data, reference_data=baseline_data)
    backend = out.diagnostics.get("backend_info", {})
    assert isinstance(backend, dict)
    tuning = backend.get("jacobian_block_tune", {})
    assert isinstance(tuning, dict)
    assert int(tuning.get("selected_block_size", 0)) > 0
    assert str(tuning.get("tune_source", "")) in {
        "compute",
        "process",
        "disk",
        "small-problem",
        "disabled",
        "fixed",
        "unset",
    }
