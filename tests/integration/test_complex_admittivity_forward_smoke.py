"""FEniCSx-native complex-admittivity CEM smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import petsc_scalar_dtype, petsc_scalar_is_complex
from pyeidors.forward.eit_forward_model import EITForwardModel

pytestmark = pytest.mark.fenicsx


def test_complex_admittivity_cem_forward_preserves_voltage_phase(eit_mesh):
    if not petsc_scalar_is_complex():
        pytest.skip("requires nix develop .#complex PETSc/DOLFINx runtime")

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
    )

    gamma = fem.Function(model.V_sigma)
    gamma.x.array[:] = np.full(gamma.x.array.shape, 1.0 + 0.25j, dtype=dtype)

    _potential, electrode_voltages = model.forward_solve(gamma)

    assert np.iscomplexobj(electrode_voltages)
    assert np.dtype(electrode_voltages.dtype) == dtype
    assert np.max(np.abs(np.imag(electrode_voltages))) > 1.0e-10
    assert model.get_backend_diagnostics()["petsc_scalar_is_complex"] is True
