"""Quick fused-vs-strict parity smoke for 3D absolute GN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_absolute_3d_fused_vs_strict_quick(tmp_path: Path):
    mesh = load_or_create_mesh(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="abs_fused_vs_strict_quick",
        n_elec=8,
        dimension=3,
        radius=0.1,
        height=0.1,
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

    strict = EITSystem(
        n_elec=8,
        pattern_config=pattern,
        contact_impedance=np.full(8, 1e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        solver_mode="strict",
        line_search_mode="full",
    )
    strict.setup(mesh=mesh)
    strict.reconstructor.max_iterations = 2
    strict.reconstructor.verbose = False

    fused = EITSystem(
        n_elec=8,
        pattern_config=pattern,
        contact_impedance=np.full(8, 1e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        solver_mode="fast",
        line_search_mode="fast",
        preconditioner="auto",
        fast_linear_path="auto",
        rom_mode="on",
        rom_rank_global=24,
        rom_rank_adaptive=12,
        inexact_mode="on",
        lowrank_mode="on",
        lowrank_rank=12,
        lowrank_method="tsvd",
    )
    fused.setup(mesh=mesh)
    fused.reconstructor.max_iterations = 2
    fused.reconstructor.verbose = False

    baseline = strict.create_homogeneous_image(conductivity=1.0)
    baseline_data = strict.forward_solve(baseline)
    phantom_sigma = np.asarray(baseline.elem_data, dtype=float).copy()
    phantom_sigma[: max(1, phantom_sigma.size // 12)] = 1.35
    target = EITImage(elem_data=phantom_sigma, fwd_model=strict.fwd_model)
    target_data = strict.forward_solve(target)

    strict_out = strict.inverse_solve(data=target_data, reference_data=baseline_data)
    fused_out = fused.inverse_solve(data=target_data, reference_data=baseline_data)

    strict_arr = function_get_array(strict_out.conductivity).copy()
    fused_arr = function_get_array(fused_out.conductivity).copy()
    rmse = float(np.sqrt(np.mean((strict_arr - fused_arr) ** 2)))
    assert np.isfinite(rmse)
    assert rmse <= 2.5
    assert np.isfinite(float(fused_out.final_residual))
    assert float(fused_out.final_residual) <= float(strict_out.final_residual) * 5.0

    backend = fused_out.diagnostics.get("backend_info", {})
    assert isinstance(backend, dict)
    assert backend.get("solver_mode") == "fast"
    assert bool(backend.get("fast_solver_path"))
    assert "degrade_stage_counts" in backend
