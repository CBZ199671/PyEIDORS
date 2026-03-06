"""CUDA parity checks for 2D/3D forward, Jacobian and GN workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from dolfinx import fem

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array, function_set_array
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf.capabilities import probe_petsc_cuda_runtime


CUDA_AVAILABLE = bool(probe_petsc_cuda_runtime().get("petsc_cuda", False)) and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA-enabled PETSc and torch.cuda")


def _pattern(n_elec: int, mesh_dim: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized" if mesh_dim == 2 else "total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )


def _make_system(mesh, *, mesh_dim: int, petsc_device: str, device: str) -> EITSystem:
    n_elec = 16 if mesh_dim == 2 else 8
    system = EITSystem(
        n_elec=n_elec,
        pattern_config=_pattern(n_elec, mesh_dim),
        contact_impedance=np.full(n_elec, 1e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        solver_mode="fast" if mesh_dim == 3 else "strict",
        line_search_mode="fast" if mesh_dim == 3 else "full",
        jacobian_update_every=1,
        jacobian_reuse_tol=0.0,
        petsc_device=petsc_device,
        device=device,
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = 1
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system


def _make_phantom(system: EITSystem) -> EITImage:
    baseline = system.create_homogeneous_image(conductivity=1.0)
    sigma = np.asarray(baseline.elem_data, dtype=float).copy()
    sigma[: max(1, sigma.size // 10)] = 1.2
    return EITImage(elem_data=sigma, fwd_model=system.fwd_model)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    denom = np.linalg.norm(np.asarray(left, dtype=float)) + 1e-12
    return float(np.linalg.norm(diff) / denom)


@pytest.fixture(scope="module")
def mesh_3d(tmp_path_factory: pytest.TempPathFactory):
    if not GMSH_AVAILABLE:
        pytest.skip("requires gmsh python bindings for 3D CUDA parity")
    tmp_path = tmp_path_factory.mktemp("cuda3d")
    return load_or_create_mesh(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="cuda_parity_3d",
        n_elec=8,
        dimension=3,
        radius=0.12,
        height=0.12,
        refinement=1,
        electrode_height_ratio=0.2,
        z_center=0.0,
        electrode_coverage=0.5,
    )


@pytest.mark.parametrize("mesh_dim", [2, 3])
def test_forward_cpu_vs_cuda_parity(eit_mesh, mesh_3d, mesh_dim: int):
    mesh = eit_mesh if mesh_dim == 2 else mesh_3d
    cpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cuda", device="cuda")
    phantom = _make_phantom(cpu)

    cpu_data = cpu.forward_solve(phantom)
    gpu_data = gpu.forward_solve(phantom)

    assert np.allclose(cpu_data.meas, gpu_data.meas, rtol=1e-6, atol=1e-9)
    assert gpu.fwd_model._petsc_backend_info.get("petsc_device_effective") == "cuda"
    assert gpu.reconstructor.device_effective == "cuda"


@pytest.mark.parametrize("mesh_dim", [2, 3])
def test_jacobian_cpu_vs_cuda_parity(eit_mesh, mesh_3d, mesh_dim: int):
    mesh = eit_mesh if mesh_dim == 2 else mesh_3d
    cpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cuda", device="cuda")

    cpu_sigma = fem.Function(cpu.fwd_model.V_sigma)
    gpu_sigma = fem.Function(gpu.fwd_model.V_sigma)
    function_set_array(cpu_sigma, np.asarray(cpu.create_homogeneous_image(conductivity=1.0).elem_data, dtype=float))
    function_set_array(gpu_sigma, np.asarray(gpu.create_homogeneous_image(conductivity=1.0).elem_data, dtype=float))

    cpu_jac = cpu.reconstructor.jacobian_calculator.calculate(cpu_sigma, method="efficient")
    gpu_jac = gpu.reconstructor.jacobian_calculator.calculate(gpu_sigma, method="efficient")
    rel_fro = np.linalg.norm(cpu_jac - gpu_jac) / (np.linalg.norm(cpu_jac) + 1e-12)
    assert rel_fro <= 1e-5


@pytest.mark.parametrize("mode", ["difference", "absolute"])
@pytest.mark.parametrize("mesh_dim", [2, 3])
def test_inverse_cpu_vs_cuda_parity(eit_mesh, mesh_3d, mesh_dim: int, mode: str):
    mesh = eit_mesh if mesh_dim == 2 else mesh_3d
    cpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, mesh_dim=mesh_dim, petsc_device="cuda", device="cuda")

    baseline = cpu.create_homogeneous_image(conductivity=1.0)
    phantom = _make_phantom(cpu)
    baseline_data = cpu.forward_solve(baseline)
    target_data = cpu.forward_solve(phantom)

    if mode == "difference":
        cpu_out = cpu.difference_reconstruct(target_data, baseline_data)
        gpu_out = gpu.difference_reconstruct(target_data, baseline_data)
    else:
        cpu_out = cpu.absolute_reconstruct(target_data, baseline_image=baseline)
        gpu_out = gpu.absolute_reconstruct(target_data, baseline_image=baseline)

    cpu_sigma = np.asarray(cpu_out.conductivity, dtype=float)
    gpu_sigma = np.asarray(gpu_out.conductivity, dtype=float)
    rmse = float(np.sqrt(np.mean((cpu_sigma - gpu_sigma) ** 2)))
    rel_l2 = _relative_l2(cpu_sigma, gpu_sigma)

    assert rel_l2 <= 5e-5
    assert rmse <= 1e-6
