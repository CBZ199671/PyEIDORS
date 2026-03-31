"""3D hex cuda_structured parity gates against the 3D CPU/dolfinx reference path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:  # pragma: no cover - optional in lean environments
    import torch
except Exception:  # pragma: no cover
    torch = None

from dolfinx import fem

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_set_array
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


CUDA_STRUCTURED_AVAILABLE = bool(torch is not None and torch.cuda.is_available() and GMSH_AVAILABLE)
pytestmark = pytest.mark.skipif(
    not CUDA_STRUCTURED_AVAILABLE,
    reason="requires gmsh python bindings and torch.cuda for cuda_structured parity",
)


def _pattern() -> PatternConfig:
    return PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )


def _make_system(mesh, *, forward_backend: str, petsc_device: str, device: str) -> EITSystem:
    system = EITSystem(
        n_elec=16,
        pattern_config=_pattern(),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        solver_mode="fast",
        line_search_mode="fast",
        jacobian_update_every=1,
        jacobian_reuse_tol=0.0,
        forward_backend=forward_backend,
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d3",
        petsc_device=petsc_device,
        device=device,
        linear_backend_config={"petsc_device": petsc_device},
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = 1
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system


def _clone_image(system: EITSystem, elem_data: np.ndarray) -> EITImage:
    return EITImage(elem_data=np.asarray(elem_data, dtype=np.float64).copy(), fwd_model=system.fwd_model)


def _make_phantom(system: EITSystem) -> EITImage:
    baseline = system.create_homogeneous_image(conductivity=1.0)
    sigma = np.asarray(baseline.elem_data, dtype=np.float64).copy()
    sigma[: max(1, sigma.size // 10)] = 1.2
    return _clone_image(system, sigma)


@pytest.fixture(scope="module")
def mesh_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cuda_structured_meshes")


def _load_mesh(mesh_root: Path, refinement: int):
    return load_or_create_mesh(
        mesh_dir=str(mesh_root),
        mesh_name=f"cuda_structured_ref{refinement}_cfhex_geomv2_g3d3",
        n_elec=16,
        dimension=3,
        radius=0.18,
        height=0.16,
        refinement=int(refinement),
        electrode_height_ratio=0.2,
        z_center=0.0,
        electrode_coverage=0.5,
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d3",
    )


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    denom = np.linalg.norm(np.asarray(left, dtype=float)) + 1e-12
    return float(np.linalg.norm(diff) / denom)


@pytest.mark.parametrize("refinement", [1, 2, 3])
def test_forward_dolfinx_vs_cuda_structured_parity(mesh_root: Path, refinement: int):
    mesh = _load_mesh(mesh_root, refinement)
    cpu = _make_system(mesh, forward_backend="dolfinx", petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, forward_backend="cuda_structured", petsc_device="cuda", device="cuda")
    phantom_cpu = _make_phantom(cpu)
    phantom_gpu = _clone_image(gpu, phantom_cpu.elem_data)

    baseline_cpu = cpu.create_homogeneous_image(conductivity=1.0)
    baseline_gpu = gpu.create_homogeneous_image(conductivity=1.0)
    cpu_hom = cpu.forward_solve(baseline_cpu)
    gpu_hom = gpu.forward_solve(baseline_gpu)
    cpu_phantom = cpu.forward_solve(phantom_cpu)
    gpu_phantom = gpu.forward_solve(phantom_gpu)

    assert _relative_l2(cpu_hom.meas, gpu_hom.meas) <= 1e-6
    assert _relative_l2(cpu_phantom.meas, gpu_phantom.meas) <= 1e-6


@pytest.mark.parametrize("refinement", [1, 2])
def test_jacobian_dolfinx_vs_cuda_structured_parity(mesh_root: Path, refinement: int):
    mesh = _load_mesh(mesh_root, refinement)
    cpu = _make_system(mesh, forward_backend="dolfinx", petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, forward_backend="cuda_structured", petsc_device="cuda", device="cuda")

    sigma_values = np.asarray(cpu.create_homogeneous_image(conductivity=1.0).elem_data, dtype=np.float64)
    cpu_sigma = fem.Function(cpu.fwd_model.V_sigma)
    gpu_sigma = fem.Function(gpu.fwd_model.V_sigma)
    function_set_array(cpu_sigma, sigma_values)
    function_set_array(gpu_sigma, sigma_values)

    cpu_u_all, _ = cpu.fwd_model.forward_solve(cpu_sigma)
    gpu_u_all, _ = gpu.fwd_model.forward_solve(gpu_sigma)
    cpu_grad = cpu.reconstructor.jacobian_calculator._compute_field_gradients(cpu_u_all)
    gpu_grad = gpu.reconstructor.jacobian_calculator._compute_field_gradients(gpu_u_all)
    assert len(cpu_grad) == len(gpu_grad)
    assert cpu_grad[0].shape == gpu_grad[0].shape

    cpu_jac = cpu.reconstructor.jacobian_calculator.calculate(cpu_sigma, method="efficient")
    gpu_jac = gpu.reconstructor.jacobian_calculator.calculate(gpu_sigma, method="efficient")
    rel_fro = np.linalg.norm(cpu_jac - gpu_jac) / (np.linalg.norm(cpu_jac) + 1e-12)
    assert rel_fro <= 1e-5


@pytest.mark.parametrize("mode", ["difference", "absolute"])
@pytest.mark.parametrize("refinement", [1, 2])
def test_inverse_dolfinx_vs_cuda_structured_parity(mesh_root: Path, refinement: int, mode: str):
    mesh = _load_mesh(mesh_root, refinement)
    cpu = _make_system(mesh, forward_backend="dolfinx", petsc_device="cpu", device="cpu")
    gpu = _make_system(mesh, forward_backend="cuda_structured", petsc_device="cuda", device="cuda")

    baseline_cpu = cpu.create_homogeneous_image(conductivity=1.0)
    baseline_gpu = gpu.create_homogeneous_image(conductivity=1.0)
    phantom_cpu = _make_phantom(cpu)
    baseline_data = cpu.forward_solve(baseline_cpu)
    target_data = cpu.forward_solve(phantom_cpu)

    if mode == "difference":
        cpu_out = cpu.difference_reconstruct(target_data, baseline_data)
        gpu_out = gpu.difference_reconstruct(target_data, baseline_data)
    else:
        cpu_out = cpu.absolute_reconstruct(target_data, baseline_image=baseline_cpu)
        gpu_out = gpu.absolute_reconstruct(target_data, baseline_image=baseline_cpu)

    cpu_sigma = np.asarray(cpu_out.conductivity, dtype=float)
    gpu_sigma = np.asarray(gpu_out.conductivity, dtype=float)
    rmse = float(np.sqrt(np.mean((cpu_sigma - gpu_sigma) ** 2)))
    rel_l2 = _relative_l2(cpu_sigma, gpu_sigma)

    assert rel_l2 <= 5e-5
    assert rmse <= 1.25e-6
