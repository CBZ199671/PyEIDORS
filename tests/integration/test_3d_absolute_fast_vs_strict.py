"""3D absolute GN parity between strict and fast solver modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf.capabilities import detect_performance_capabilities


def _build_system(mesh, *, solver_mode: str, preconditioner: str = "auto") -> EITSystem:
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
        solver_mode=solver_mode,
        linear_solver="auto",
        jacobian_update_every=2,
        jacobian_reuse_tol=1e-3,
        line_search_mode="fast" if solver_mode == "fast" else "full",
        preconditioner=preconditioner,
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = 1
    system.reconstructor.verbose = False
    return system


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_absolute_3d_fast_vs_strict(tmp_path: Path):
    mesh = load_or_create_mesh(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="abs_fast_vs_strict",
        n_elec=8,
        dimension=3,
        radius=0.12,
        height=0.12,
        refinement=1,
        electrode_height_ratio=0.2,
        z_center=0.0,
        electrode_coverage=0.5,
    )
    caps = detect_performance_capabilities()
    fast_preconditioner = "cholmod" if caps.get("cholmod", False) else "auto"
    strict_system = _build_system(mesh, solver_mode="strict")
    fast_system = _build_system(
        mesh,
        solver_mode="fast",
        preconditioner=fast_preconditioner,
    )

    baseline = strict_system.create_homogeneous_image(conductivity=1.0)
    baseline_data = strict_system.forward_solve(baseline)
    phantom_sigma = np.asarray(baseline.elem_data, dtype=float).copy()
    phantom_sigma[: max(1, phantom_sigma.size // 10)] = 1.25
    target = EITImage(elem_data=phantom_sigma, fwd_model=strict_system.fwd_model)
    target_data = strict_system.forward_solve(target)

    strict = strict_system.inverse_solve(data=target_data, reference_data=baseline_data)
    fast = fast_system.inverse_solve(data=target_data, reference_data=baseline_data)

    strict_arr = function_get_array(strict.conductivity).copy()
    fast_arr = function_get_array(fast.conductivity).copy()
    rmse = float(np.sqrt(np.mean((strict_arr - fast_arr) ** 2)))
    assert rmse <= 1e-6
    backend = fast.diagnostics.get("backend_info", {})
    assert isinstance(backend, dict)
    fast_path = str(backend.get("fast_solver_path", ""))
    fallback_reason = str(backend.get("fallback_reason", ""))
    selected_path = str(backend.get("fast_linear_path_selected", ""))
    assert fast_path or fallback_reason
    assert selected_path in {"woodbury", "pcg", "cholmod-direct", "strict", "fused", ""}
    if caps.get("cholmod", False) and selected_path == "pcg":
        assert ("cholmod" in fast_path.lower()) or bool(fallback_reason)
