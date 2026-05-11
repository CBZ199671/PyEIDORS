"""2D PETSc-CUDA EIDORS-style Jacobian regression gate."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

try:  # pragma: no cover - optional in non-CUDA shells
    import torch
except Exception:  # pragma: no cover
    torch = None

from pyeidors import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter


pytestmark = [pytest.mark.slow, pytest.mark.fenicsx, pytest.mark.gpu]


def _require_cuda_gate() -> None:
    if os.environ.get("PYEIDORS_RUN_CUDA_2D_JACOBIAN") != "1":
        pytest.skip("set PYEIDORS_RUN_CUDA_2D_JACOBIAN=1 to run this CUDA gate")
    if not GMSH_AVAILABLE:
        pytest.skip("requires gmsh python bindings")
    if torch is None or not torch.cuda.is_available():
        pytest.skip("requires torch.cuda")


def _make_2d_cuda_system(tmp_path: Path) -> EITSystem:
    mesh = create_eit_mesh(
        n_elec=8,
        radius=0.18,
        refinement=3,
        electrode_coverage=0.5,
        output_dir=str(tmp_path / "mesh"),
        mesh_name="cuda_2d_jacobian_circle",
    )
    pattern = PatternConfig(
        n_elec=8,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        measurement_protocol="eidors_full_3d",
        drive_mode="total_current",
        drive_value=1.0,
        rotate_meas=True,
        use_meas_current=False,
        stim_first_positive=False,
    )
    system = EITSystem(
        n_elec=8,
        pattern_config=pattern,
        contact_impedance=np.full(8, 1.0e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_scope="off",
        petsc_device="cuda",
        device="cuda",
        linear_backend_config={"petsc_device": "cuda"},
    )
    system.setup(mesh=mesh)
    return system


def test_2d_cuda_eidors_jacobian_uses_cuda_forward_without_cell_list_error(
    tmp_path: Path,
):
    _require_cuda_gate()
    system = _make_2d_cuda_system(tmp_path)
    backend = getattr(system.fwd_model, "_petsc_backend_info", {})
    assert backend.get("petsc_device_effective") == "cuda"

    background = system.create_homogeneous_image(conductivity=1.0)
    jac_calc = EidorsJacobianAdapter(
        system.fwd_model,
        use_torch=True,
        device="cuda",
        torch_dtype="float64",
        torch_batch_all=True,
    )

    jacobian = jac_calc.calculate_from_image(background)

    assert jacobian.shape == (
        system.fwd_model.pattern_manager.n_meas_total,
        system.fwd_model.V_sigma.dofmap.index_map.size_local,
    )
    assert np.isfinite(jacobian).all()
    assert float(np.linalg.norm(jacobian)) > 0.0
