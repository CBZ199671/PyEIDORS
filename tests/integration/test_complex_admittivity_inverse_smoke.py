"""FEniCSx-native complex-admittivity Jacobian/inverse smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import petsc_scalar_dtype, petsc_scalar_is_complex
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian import DirectJacobianCalculator
from pyeidors.inverse.solvers.gauss_newton_linear_system import (
    solve_native_complex_normal_step,
)
from pyeidors.perf.capabilities import probe_petsc_cuda_runtime

pytestmark = pytest.mark.fenicsx


def _well_conditioned_columns(jacobian: np.ndarray, *, n_cols: int = 2) -> np.ndarray:
    best_cols: np.ndarray | None = None
    best_cond = float("inf")
    for start in range(0, max(1, jacobian.shape[1] - int(n_cols) + 1)):
        cols = np.arange(start, start + int(n_cols))
        block = jacobian[:, cols]
        singular_values = np.linalg.svd(block, compute_uv=False)
        if singular_values.size < int(n_cols) or singular_values[-1] <= 1.0e-14:
            continue
        cond = float(singular_values[0] / singular_values[-1])
        if cond < best_cond:
            best_cond = cond
            best_cols = cols
    if best_cols is not None and best_cond < 1.0e6:
        return best_cols
    norms = np.linalg.norm(jacobian, axis=0)
    return np.asarray([int(np.argmax(norms))], dtype=np.int64)


def test_complex_admittivity_jacobian_feeds_native_complex_normal_step(eit_mesh):
    if not petsc_scalar_is_complex():
        pytest.skip(
            "requires nix develop .#complex or .#complex64 PETSc/DOLFINx runtime"
        )

    dtype = petsc_scalar_dtype()
    pattern = PatternConfig(
        n_elec=4,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    model = EITForwardModel(
        n_elec=4,
        pattern_config=pattern,
        z=np.full(4, 1.0e-3 + 2.0e-4j, dtype=dtype),
        mesh=eit_mesh,
        linear_backend="petsc",
    )

    gamma = fem.Function(model.V_sigma)
    gamma.x.array[:] = np.full(gamma.x.array.shape, 1.0 + 0.25j, dtype=dtype)

    jacobian = DirectJacobianCalculator(model, runtime_device="cpu").calculate(gamma)

    assert np.iscomplexobj(jacobian)
    assert np.dtype(jacobian.dtype) == dtype
    assert jacobian.shape[0] == model.pattern_manager.n_meas_total
    assert jacobian.shape[1] == gamma.x.array.size
    assert np.max(np.abs(np.imag(jacobian))) > 1.0e-12

    column_idx = _well_conditioned_columns(jacobian, n_cols=min(2, jacobian.shape[0]))
    jacobian_small = jacobian[:, column_idx]
    true_delta = np.linspace(
        1.0e-3 + 2.0e-4j,
        4.0e-3 + 8.0e-4j,
        jacobian_small.shape[1],
        dtype=dtype,
    )
    residual = -(jacobian_small @ true_delta)

    delta, meta = solve_native_complex_normal_step(
        jacobian=jacobian_small,
        residual=residual,
        lambda_eff=0.0,
        regularization=np.eye(jacobian_small.shape[1], dtype=np.float64),
    )

    assert meta["native_complex_linear_algebra"] is True
    assert meta["transpose"] == "hermitian_conjugate"
    assert np.dtype(delta.dtype) == dtype
    np.testing.assert_allclose(delta, true_delta, rtol=1e-4, atol=1e-7)


@pytest.mark.gpu
def test_complex_admittivity_cuda_jacobian_preserves_complex_dtype(eit_mesh):
    if not petsc_scalar_is_complex():
        pytest.skip("requires nix develop .#complex-cuda or .#complex64-cuda")
    capability = probe_petsc_cuda_runtime()
    if not bool(capability.get("petsc_cuda", False)):
        pytest.skip("requires PETSc CUDA Mat/Vec support in the active runtime")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional GPU runtime
        pytest.skip(f"requires torch CUDA runtime: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("requires torch.cuda for CUDA Jacobian contraction")

    dtype = petsc_scalar_dtype()
    pattern = PatternConfig(
        n_elec=4,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    model = EITForwardModel(
        n_elec=4,
        pattern_config=pattern,
        z=np.full(4, 1.0e-3 + 2.0e-4j, dtype=dtype),
        mesh=eit_mesh,
        linear_backend="petsc",
        backend_config={
            "petsc_device": "cuda",
            "ksp_type": "gmres",
            "pc_type": "none",
            "mat_solve_mode": "off",
            "rtol": 1.0e-9,
            "atol": 1.0e-11,
            "max_it": 2000,
        },
        forward_backend="dolfinx",
    )
    gamma = fem.Function(model.V_sigma)
    gamma.x.array[:] = np.full(gamma.x.array.shape, 1.0 + 0.25j, dtype=dtype)

    calculator = DirectJacobianCalculator(model, runtime_device="cuda")
    calculator.set_runtime_device("cuda", "cuda", torch_device="cuda")
    jacobian = calculator.calculate(gamma)
    info = calculator.block_tuning_info()
    diag = model.get_backend_diagnostics()

    assert np.iscomplexobj(jacobian)
    assert np.dtype(jacobian.dtype) == dtype
    assert info["jacobian_block_backend"] == "torch-cuda"
    assert diag["petsc_scalar_is_complex"] is True
    assert diag["petsc_device_effective"] == "cuda"
