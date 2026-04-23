"""Additional branch coverage for absolute/difference workflow wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.workflows import absolute as absolute_workflow
from pyeidors.inverse.workflows import difference as difference_workflow


def _stub_system(*, initialized: bool = True):
    baseline = SimpleNamespace(elem_data=np.array([1.0, 1.1], dtype=float))
    fwd_model = SimpleNamespace(
        fwd_solve=lambda _img: (
            SimpleNamespace(meas=np.array([0.7, 0.8], dtype=float)),
            {"ok": True},
        ),
    )
    reconstruction = SimpleNamespace(
        diagnostics={"solver": "stub"},
        simulated_measurement=np.array([0.2, 0.3], dtype=float),
    )
    return SimpleNamespace(
        _is_initialized=initialized,
        fwd_model=fwd_model,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        create_homogeneous_image=lambda: baseline,
        inverse_solve=lambda **kwargs: reconstruction,
    )


def test_absolute_workflow_init_guard_and_metadata_merge(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(RuntimeError, match="not initialized"):
        absolute_workflow.perform_absolute_reconstruction(
            _stub_system(initialized=False),
            measurement_data=SimpleNamespace(meas=np.array([1.0], dtype=float)),
        )

    monkeypatch.setattr(
        absolute_workflow,
        "resolve_reconstruction_output",
        lambda reconstruction, _fwd_model: (
            SimpleNamespace(elem_data=np.array([1.2, 1.3], dtype=float)),
            np.array([1.2, 1.3], dtype=float),
            [0.4, 0.1],
            [0.2, 0.05],
        ),
    )
    monkeypatch.setattr(
        absolute_workflow,
        "compute_residuals",
        lambda measured, simulated: (measured - simulated, 0.0, 0.0, 0.0),
    )

    measured = SimpleNamespace(meas=np.array([1.0, 1.1], dtype=float))
    result = absolute_workflow.perform_absolute_reconstruction(
        _stub_system(),
        measurement_data=measured,
        metadata={"case": "absolute"},
    )
    assert result.mode == "absolute"
    assert result.metadata["case"] == "absolute"
    assert result.metadata["solver_diagnostics"]["solver"] == "stub"
    np.testing.assert_allclose(
        result.metadata["baseline_used"], np.array([1.0, 1.1], dtype=float)
    )


def test_difference_workflow_init_guard_and_metadata_merge(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(RuntimeError, match="not initialized"):
        difference_workflow.perform_difference_reconstruction(
            _stub_system(initialized=False),
            measurement_data=SimpleNamespace(meas=np.array([1.0], dtype=float)),
            reference_data=SimpleNamespace(meas=np.array([0.9], dtype=float)),
        )

    monkeypatch.setattr(
        difference_workflow,
        "resolve_reconstruction_output",
        lambda reconstruction, _fwd_model: (
            SimpleNamespace(elem_data=np.array([1.2, 1.3], dtype=float)),
            np.array([1.2, 1.3], dtype=float),
            [0.3, 0.1],
            [0.15, 0.02],
        ),
    )
    monkeypatch.setattr(
        difference_workflow,
        "difference_measurement",
        lambda _meas, ref, **kwargs: SimpleNamespace(
            meas=np.array([0.4, 0.5], dtype=float),
            reference_meas=ref.meas.copy(),
            difference_mode=kwargs["mode"],
            difference_orientation=kwargs["orientation"],
        ),
    )
    monkeypatch.setattr(
        difference_workflow,
        "compute_residuals",
        lambda measured, simulated: (measured - simulated, 0.0, 0.0, 0.0),
    )

    measured = SimpleNamespace(meas=np.array([1.4, 1.5], dtype=float))
    reference = SimpleNamespace(meas=np.array([1.0, 1.0], dtype=float))
    result = difference_workflow.perform_difference_reconstruction(
        _stub_system(),
        measurement_data=measured,
        reference_data=reference,
        metadata={"case": "difference"},
    )
    assert result.mode == "difference"
    assert result.metadata["case"] == "difference"
    np.testing.assert_allclose(result.metadata["reference_measured"], reference.meas)
