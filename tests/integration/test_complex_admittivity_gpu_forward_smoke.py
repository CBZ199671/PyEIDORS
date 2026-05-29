"""PETSc-CUDA smoke for FEniCSx-native complex-admittivity CEM."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import petsc_scalar_dtype, petsc_scalar_is_complex
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.perf.capabilities import probe_petsc_cuda_runtime

pytestmark = [pytest.mark.fenicsx, pytest.mark.gpu]


def test_complex_admittivity_cem_forward_uses_petsc_cuda(eit_mesh):
    if not petsc_scalar_is_complex():
        pytest.skip("requires nix develop .#complex-cuda or .#complex64-cuda")
    capability = probe_petsc_cuda_runtime()
    if not bool(capability.get("petsc_cuda", False)):
        pytest.skip("requires PETSc CUDA Mat/Vec support in the active runtime")

    dtype = petsc_scalar_dtype()
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    model = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=np.full(16, 1.0e-3 + 2.0e-4j, dtype=dtype),
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

    _potential, electrode_voltages = model.forward_solve(gamma)
    diag = model.get_backend_diagnostics()

    assert np.iscomplexobj(electrode_voltages)
    assert np.dtype(electrode_voltages.dtype) == dtype
    assert np.max(np.abs(np.imag(electrode_voltages))) > 1.0e-10
    assert diag["petsc_scalar_is_complex"] is True
    assert diag["petsc_device_effective"] == "cuda"
    assert diag["forward_backend_effective"] == "dolfinx"
