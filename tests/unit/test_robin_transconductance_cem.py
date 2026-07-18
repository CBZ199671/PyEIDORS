"""Parity and contract tests for Robin-transconductance CEM."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.forward import EITForwardModel, RobinTransconductanceForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.forward.robin_transconductance import (
    normalize_cem_formulation,
    zero_sum_helmert_basis,
)


def _pattern() -> PatternConfig:
    return PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )


def _tolerances(dtype: np.dtype) -> tuple[float, float]:
    real_dtype = np.empty(1, dtype=np.dtype(dtype)).real.dtype
    if real_dtype == np.dtype(np.float32):
        return 2e-5, 2e-6
    return 1e-10, 1e-11


def test_zero_sum_helmert_basis_is_deterministic_orthonormal() -> None:
    basis = zero_sum_helmert_basis(8, dtype=np.complex64)
    repeated = zero_sum_helmert_basis(8, dtype=np.complex64)

    np.testing.assert_array_equal(basis, repeated)
    np.testing.assert_allclose(np.sum(basis, axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=2e-7)

    with pytest.raises(ValueError, match="at least 2 electrodes"):
        zero_sum_helmert_basis(1)


def test_cem_formulation_normalization_preserves_classic_default() -> None:
    assert normalize_cem_formulation(None) == "classic"
    assert normalize_cem_formulation("classic_cem") == "classic"
    assert normalize_cem_formulation("robin") == "robin_transconductance"
    assert (
        normalize_cem_formulation("robin-transconductance") == "robin_transconductance"
    )
    with pytest.raises(ValueError, match="Unknown cem_formulation"):
        normalize_cem_formulation("point-electrode")


def test_robin_transconductance_matches_classic_cem_on_same_mesh(eit_mesh) -> None:
    clear_process_forward_setup_cache()
    impedance = np.full(16, 1e-2, dtype=float)
    classic = EITForwardModel(
        n_elec=16,
        pattern_config=_pattern(),
        z=impedance,
        mesh=eit_mesh,
        linear_backend="scipy",
    )
    robin = RobinTransconductanceForwardModel(
        n_elec=16,
        pattern_config=_pattern(),
        z=impedance,
        mesh=eit_mesh,
        linear_backend="scipy",
    )
    sigma_classic = fem.Function(classic.V_sigma)
    sigma_robin = fem.Function(robin.V_sigma)
    sigma_values = np.linspace(0.8, 1.2, sigma_classic.x.array.size)
    sigma_classic.x.array[:] = sigma_values
    sigma_robin.x.array[:] = sigma_values

    u_classic, voltage_classic = classic.forward_solve(sigma_classic)
    u_robin, voltage_robin = robin.forward_solve(sigma_robin)
    rtol, atol = _tolerances(voltage_classic.dtype)

    np.testing.assert_allclose(
        voltage_robin,
        voltage_classic,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.column_stack(u_robin),
        np.column_stack(u_classic),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        robin.pattern_manager.apply_meas_pattern(voltage_robin),
        classic.pattern_manager.apply_meas_pattern(voltage_classic),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.sum(voltage_robin, axis=1),
        0.0,
        atol=max(atol, 2e-6),
    )

    diagnostics = robin.get_backend_diagnostics()
    assert diagnostics["cem_formulation_effective"] == "robin_transconductance"
    assert diagnostics["robin_transconductance_rank"] == 15
    assert diagnostics["robin_transconductance_condition_number"] > 0.0
    assert diagnostics["robin_response_residual"] <= max(rtol, 1e-10)
    assert diagnostics["robin_reduced_solve_residual"] <= max(rtol, 1e-10)
    assert diagnostics["robin_transconductance_cache_hit"] is False

    _, repeated_voltage = robin.forward_solve(sigma_robin)
    np.testing.assert_array_equal(repeated_voltage, voltage_robin)
    assert robin.get_backend_diagnostics()["robin_transconductance_cache_hit"] is True


def test_robin_transconductance_rejects_imbalanced_current(eit_mesh) -> None:
    robin = RobinTransconductanceForwardModel(
        n_elec=16,
        pattern_config=_pattern(),
        z=np.full(16, 1e-2, dtype=float),
        mesh=eit_mesh,
        linear_backend="scipy",
    )
    sigma = fem.Function(robin.V_sigma)
    sigma.x.array[:] = 1.0
    imbalanced = np.zeros((1, 16), dtype=float)
    imbalanced[0, 0] = 1.0

    with pytest.raises(ValueError, match="requires balanced current patterns"):
        robin.forward_solve(sigma, imbalanced)


def test_robin_transconductance_petsc_reuses_one_basis_setup(eit_mesh) -> None:
    scipy_model = RobinTransconductanceForwardModel(
        n_elec=16,
        pattern_config=_pattern(),
        z=np.full(16, 1e-2, dtype=float),
        mesh=eit_mesh,
        linear_backend="scipy",
    )
    petsc_model = RobinTransconductanceForwardModel(
        n_elec=16,
        pattern_config=_pattern(),
        z=np.full(16, 1e-2, dtype=float),
        mesh=eit_mesh,
        linear_backend="petsc",
        backend_config={
            "solver_preset": "custom",
            "ksp_type": "gmres",
            "pc_type": "jacobi",
            "rtol": 1e-12,
            "atol": 1e-14,
            "max_it": 4000,
            "petsc_device": "cpu",
        },
    )
    sigma_scipy = fem.Function(scipy_model.V_sigma)
    sigma_petsc = fem.Function(petsc_model.V_sigma)
    sigma_scipy.x.array[:] = 1.0
    sigma_petsc.x.array[:] = 1.0

    potential_scipy, voltage_scipy = scipy_model.forward_solve(sigma_scipy)
    potential_petsc, voltage_petsc = petsc_model.forward_solve(sigma_petsc)
    rtol, atol = _tolerances(voltage_scipy.dtype)
    requested_rtol = float(petsc_model.backend_config.rtol)
    iterative_rtol = max(rtol, 64.0 * requested_rtol)
    voltage_atol = max(
        atol,
        64.0 * requested_rtol * float(np.max(np.abs(voltage_scipy))),
    )
    potential_scipy_matrix = np.column_stack(potential_scipy)
    potential_atol = max(
        atol,
        64.0 * requested_rtol * float(np.max(np.abs(potential_scipy_matrix))),
    )
    np.testing.assert_allclose(
        voltage_petsc,
        voltage_scipy,
        rtol=iterative_rtol,
        atol=voltage_atol,
    )
    np.testing.assert_allclose(
        np.column_stack(potential_petsc),
        potential_scipy_matrix,
        rtol=iterative_rtol,
        atol=potential_atol,
    )

    diagnostics = petsc_model.get_backend_diagnostics()
    assert str(diagnostics["robin_transconductance_backend"]).startswith("petsc"), {
        key: diagnostics.get(key)
        for key in (
            "fallback_reason",
            "solver_preset",
            "ksp_type",
            "pc_type",
            "pc_factor_mat_solver_type",
            "petsc_device_effective",
            "petsc_mat_type",
        )
    }
    assert diagnostics["forward_ksp_setup_count"] == 1
    assert diagnostics["forward_ksp_solve_count"] == 15
    assert not diagnostics.get("fallback_reason")


def test_eit_system_selects_robin_without_changing_default(eit_mesh) -> None:
    classic_system = EITSystem(
        n_elec=16,
        pattern_config=_pattern(),
        contact_impedance=np.full(16, 1e-2, dtype=float),
        linear_backend="scipy",
        cache_scope="off",
    )
    classic_system.setup(mesh=eit_mesh, initialize_inverse=False)
    assert type(classic_system.fwd_model) is EITForwardModel
    assert classic_system.cem_formulation == "classic"

    robin_system = EITSystem(
        n_elec=16,
        pattern_config=_pattern(),
        contact_impedance=np.full(16, 1e-2, dtype=float),
        cem_formulation="robin_transconductance",
        linear_backend="scipy",
        cache_scope="off",
    )
    robin_system.setup(mesh=eit_mesh, initialize_inverse=False)
    assert isinstance(robin_system.fwd_model, RobinTransconductanceForwardModel)
    assert (
        robin_system.fwd_model.get_backend_diagnostics()["cem_formulation_effective"]
        == "robin_transconductance"
    )
