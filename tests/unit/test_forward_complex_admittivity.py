"""Complex-admittivity forward-model guardrails."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.forward.complex_support as complex_support
import pyeidors.forward.cuda_structured_backend as cuda_structured_module
import pyeidors.forward.eit_forward_model as forward_module
import pyeidors.forward.process_setup_cache as process_cache_module
from pyeidors.core_system import EITSystem
from pyeidors.data.difference import build_difference_frames, build_difference_vector
from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.forward.eit_forward_model import (
    EITForwardModel,
    _coerce_scalar_array,
)
from pyeidors.forward.process_setup_cache import build_process_forward_setup_key


def test_complex_support_reports_active_petsc_scalar(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        complex_support,
        "PETSc",
        SimpleNamespace(ScalarType=np.complex128),
    )

    assert complex_support.petsc_scalar_dtype() == np.dtype(np.complex128)
    assert complex_support.petsc_scalar_is_complex() is True
    complex_support.require_complex_scalar_support()

    monkeypatch.setattr(
        complex_support,
        "PETSc",
        SimpleNamespace(ScalarType=np.float64),
    )
    with pytest.raises(RuntimeError, match=r"nix develop .#complex") as excinfo:
        complex_support.require_complex_scalar_support()
    assert ".#complex64-cuda" in str(excinfo.value)


def test_scalar_coercion_preserves_phase_and_rejects_real_runtime_loss():
    values = np.array([1.0 + 0.25j, 2.0 - 0.5j])

    complex_values = _coerce_scalar_array(
        values,
        np.dtype(np.complex128),
        name="gamma",
    )
    assert complex_values.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(complex_values.imag, [0.25, -0.5])

    with pytest.raises(RuntimeError, match=r"active PETSc/DOLFINx scalar dtype"):
        _coerce_scalar_array(values, np.dtype(np.float64), name="gamma")


def test_process_setup_key_separates_real_and_complex_contact_impedance():
    kwargs = {
        "mesh_file": "/tmp/mesh.xdmf",
        "n_elec": 4,
        "pattern_config": PatternConfig(n_elec=4),
        "mesh_content_hash": None,
    }

    real_key = build_process_forward_setup_key(
        **kwargs,
        z=np.full(4, 0.01, dtype=np.float64),
        scalar_dtype=np.float64,
    )
    complex_key = build_process_forward_setup_key(
        **kwargs,
        z=np.full(4, 0.01 + 0.002j, dtype=np.complex128),
        scalar_dtype=np.complex128,
    )

    assert real_key != complex_key
    with pytest.raises(RuntimeError, match="complex contact impedance"):
        build_process_forward_setup_key(
            **kwargs,
            z=np.full(4, 0.01 + 0.002j, dtype=np.complex128),
            scalar_dtype=np.float64,
        )


def test_v475_forward_complex_guards_use_shared_bounded_imag_scan() -> None:
    forward_source = inspect.getsource(forward_module._has_nonzero_imaginary)
    setup_source = inspect.getsource(
        process_cache_module.build_process_forward_setup_key
    )

    assert "_array_has_nonzero_imaginary(values)" in forward_source
    assert "has_nonzero_imaginary(z_array)" in setup_source
    assert "np.any(np.abs(np.imag" not in forward_source
    assert "np.any(np.abs(np.imag" not in setup_source


def test_v476_forward_scalar_coercion_uses_bounded_finite_scan() -> None:
    source = inspect.getsource(forward_module._coerce_scalar_array)

    assert "all_finite_values(out)" in source
    assert "np.all(np.isfinite(out))" not in source
    assert "np.isfinite(out).all()" not in source


def test_v476_cuda_structured_diag_guard_uses_bounded_finite_scan_source() -> None:
    source = inspect.getsource(
        cuda_structured_module.CudaStructuredForwardBackend._build_sigma_state
    )

    assert "all_finite_values(diag)" in source
    assert "np.all(np.isfinite(diag))" not in source
    assert "np.isfinite(diag).all()" not in source


def test_measurement_projection_preserves_complex_dtype_and_phase():
    manager = StimMeasPatternManager(
        PatternConfig(
            n_elec=4,
            stim_pattern="{ad}",
            meas_pattern="{ad}",
            drive_mode="line_current_density",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
        ),
        mesh_tdim=2,
    )
    electrode_voltages = np.array(
        [
            [1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j],
            [2.0 + 0.4j, 3.0 + 0.3j, 4.0 + 0.2j, 5.0 + 0.1j],
            [3.0 + 0.2j, 4.0 + 0.1j, 5.0 + 0.4j, 6.0 + 0.3j],
            [4.0 + 0.3j, 5.0 + 0.4j, 6.0 + 0.1j, 7.0 + 0.2j],
        ],
        dtype=np.complex64,
    )

    measured = manager.apply_meas_pattern(electrode_voltages)

    assert measured.dtype == np.dtype(np.complex64)
    assert np.max(np.abs(np.imag(measured))) > 0.0


def test_forward_model_complex_inputs_keep_complex_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    model = EITForwardModel.__new__(EITForwardModel)
    model.scalar_dtype = np.dtype(np.complex128)
    model.is_complex = True
    model.n_elec = 2
    model.V_sigma = object()
    model.pattern_manager = SimpleNamespace(
        stim_matrix=np.array([[1.0, -1.0]], dtype=float),
        n_stim=1,
        n_meas_total=2,
        apply_meas_pattern=lambda U: U.sum(axis=1),
    )

    created = []

    class _FakeFemFunction:
        def __init__(self, _space):
            self.x = SimpleNamespace(array=np.zeros(2, dtype=np.complex128))
            created.append(self)

    monkeypatch.setattr(forward_module.fem, "Function", _FakeFemFunction)
    model.forward_solve = lambda _sigma: (
        (np.array([1.0 + 1.0j], dtype=np.complex128),),
        np.array([[1.0 + 0.2j, 2.0 + 0.4j]], dtype=np.complex128),
    )

    data, electrode_voltages = EITForwardModel.fwd_solve(
        model,
        SimpleNamespace(
            get_conductivity=lambda: np.array(
                [1.0 + 0.1j, 1.2 + 0.3j],
                dtype=np.complex128,
            )
        ),
    )

    assert created
    np.testing.assert_allclose(created[0].x.array.imag, [0.1, 0.3])
    assert electrode_voltages.dtype == np.dtype(np.complex128)
    assert data.meas.dtype == np.dtype(np.complex128)
    assert data.type == "complex_simulated"


def test_forward_model_real_runtime_rejects_complex_current_patterns():
    model = EITForwardModel.__new__(EITForwardModel)
    model.scalar_dtype = np.dtype(np.float64)
    model.is_complex = False
    model.n_elec = 2
    model.pattern_manager = SimpleNamespace(
        stim_matrix=np.array([[1.0, -1.0]], dtype=float)
    )

    with pytest.raises(RuntimeError, match="current_patterns contains complex values"):
        model._resolve_pattern_matrix(np.array([[1.0 + 0.1j, -1.0]], dtype=complex))


def test_system_configuration_preserves_complex_admittivity_values():
    system = EITSystem(
        n_elec=4,
        pattern_config=PatternConfig(n_elec=4),
        contact_impedance=np.full(4, 1.0e-3 + 2.0e-4j, dtype=np.complex128),
        base_conductivity=1.0 + 0.25j,
        jacobian_background_conductivity=1.0 + 0.1j,
        cache_scope="off",
    )

    assert np.iscomplexobj(system.contact_impedance)
    assert system.contact_impedance[0] == pytest.approx(1.0e-3 + 2.0e-4j)
    assert system.base_conductivity == pytest.approx(1.0 + 0.25j)
    assert system.jacobian_background_conductivity == pytest.approx(1.0 + 0.1j)


def test_difference_measurements_preserve_complex_phase():
    target = np.array([1.0 + 0.2j, 2.0 - 0.1j], dtype=np.complex128)
    reference = np.array([0.5 + 0.1j, 1.0 + 0.2j], dtype=np.complex128)

    raw = build_difference_vector(target, reference)
    normalized = build_difference_vector(target, reference, mode="normalized")
    frames = build_difference_frames(
        target.reshape(1, -1),
        reference.reshape(1, -1),
        mode="normalized",
    )

    assert raw.dtype == np.dtype(np.complex128)
    assert normalized.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(raw, target - reference)
    np.testing.assert_allclose(normalized, (target - reference) / reference)
    np.testing.assert_allclose(frames[0], normalized)


def test_cuda_guidance_names_complex_cuda_profiles():
    guidance = EITForwardModel._actionable_cuda_guidance()

    assert ".#cuda" in guidance
    assert ".#complex-cuda" in guidance
    assert ".#complex64-cuda" in guidance
