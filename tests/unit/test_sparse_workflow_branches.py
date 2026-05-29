"""Additional branch coverage for sparse Bayesian workflow wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.contracts import SolverOutput
from pyeidors.inverse.workflows import sparse_bayesian as sparse_workflow_module


class _FactoryReconstructor:
    def __init__(self, *, output):
        self._output = output

    def reconstruct(self, **kwargs):
        _ = kwargs
        return self._output


def _make_stub_system(*, initialized: bool = True):
    baseline = SimpleNamespace(elem_data=np.array([1.0, 1.5], dtype=float))
    simulated = np.array([0.2, 0.4], dtype=float)
    fwd_model = SimpleNamespace(
        fwd_solve=lambda _img: (
            SimpleNamespace(meas=simulated.copy()),
            {"source": "fallback"},
        ),
    )
    return SimpleNamespace(
        _is_initialized=initialized,
        fwd_model=fwd_model,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        create_homogeneous_image=lambda: baseline,
    )


def test_sparse_absolute_requires_initialized_system():
    with pytest.raises(RuntimeError, match="must be initialised"):
        sparse_workflow_module.perform_sparse_absolute_reconstruction(
            eit_system=_make_stub_system(initialized=False),
            measurement_data=SimpleNamespace(meas=np.array([1.0], dtype=float)),
            reconstructor=_FactoryReconstructor(output="bad"),
        )


def test_sparse_absolute_type_guard():
    with pytest.raises(TypeError, match="must return SolverOutput"):
        sparse_workflow_module.perform_sparse_absolute_reconstruction(
            eit_system=_make_stub_system(),
            measurement_data=SimpleNamespace(meas=np.array([1.0], dtype=float)),
            reconstructor=_FactoryReconstructor(output="not-a-solver-output"),
        )


def test_sparse_absolute_uses_solver_simulated_without_forward_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    output = SolverOutput(
        conductivity=np.array([1.1, 1.2], dtype=float),
        simulated_measurement=np.array([0.6, 0.7], dtype=float),
        likelihood_noise_std=1e-3,
        prior_scale=2e-2,
        metadata={},
        iterations=1,
        converged=True,
        final_residual=0.1,
        final_relative_change=0.01,
    )
    fwd_model = SimpleNamespace(
        fwd_solve=lambda _img: (_ for _ in ()).throw(
            AssertionError("solver_output.simulated_measurement must skip fwd_solve")
        )
    )
    eit_system = SimpleNamespace(
        _is_initialized=True,
        fwd_model=fwd_model,
        create_homogeneous_image=lambda: SimpleNamespace(
            elem_data=np.array([1.0, 1.5], dtype=float)
        ),
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "resolve_reconstruction_output",
        lambda solver_output, _fwd_model: (
            SimpleNamespace(elem_data=np.array([1.1, 1.2], dtype=float)),
            np.array([1.1, 1.2], dtype=float),
            [0.5],
            [0.1],
        ),
    )

    result = sparse_workflow_module.perform_sparse_absolute_reconstruction(
        eit_system=eit_system,
        measurement_data=SimpleNamespace(meas=np.array([0.5, 0.6], dtype=float)),
        reconstructor=_FactoryReconstructor(output=output),
    )

    np.testing.assert_allclose(result.simulated, np.array([0.6, 0.7], dtype=float))


def test_sparse_difference_requires_initialized_system():
    with pytest.raises(RuntimeError, match="must be initialised"):
        sparse_workflow_module.perform_sparse_difference_reconstruction(
            eit_system=_make_stub_system(initialized=False),
            measurement_data=SimpleNamespace(meas=np.array([1.0], dtype=float)),
            reference_data=SimpleNamespace(meas=np.array([0.9], dtype=float)),
            reconstructor=_FactoryReconstructor(output="bad"),
        )


def test_sparse_absolute_uses_factory_fallback_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    output = SolverOutput(
        conductivity=np.array([1.1, 1.2], dtype=float),
        simulated_measurement=None,
        likelihood_noise_std=1e-3,
        prior_scale=2e-2,
        metadata={"solver_meta": "absolute", "shared": "solver"},
        iterations=2,
        converged=True,
        final_residual=0.1,
        final_relative_change=0.01,
    )
    created: dict[str, object] = {}

    def _factory(*, eit_system, config):
        created["eit_system"] = eit_system
        created["config"] = config
        return _FactoryReconstructor(output=output)

    monkeypatch.setattr(sparse_workflow_module, "SparseBayesianReconstructor", _factory)
    monkeypatch.setattr(
        sparse_workflow_module,
        "resolve_reconstruction_output",
        lambda solver_output, _fwd_model: (
            SimpleNamespace(elem_data=np.array([1.1, 1.2], dtype=float)),
            np.array([1.1, 1.2], dtype=float),
            [0.5, 0.1],
            [0.2, 0.05],
        ),
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "compute_residuals",
        lambda measured, simulated: (measured - simulated, 0.0, 0.0, 0.0),
    )

    eit_system = _make_stub_system()
    config = object()
    measurement_data = SimpleNamespace(meas=np.array([0.9, 1.0], dtype=float))
    result = sparse_workflow_module.perform_sparse_absolute_reconstruction(
        eit_system=eit_system,
        measurement_data=measurement_data,
        reconstructor=None,
        config=config,
        metadata={"user_meta": "kept", "shared": "user"},
    )

    assert created["eit_system"] is eit_system
    assert created["config"] is config
    np.testing.assert_allclose(result.simulated, np.array([0.2, 0.4], dtype=float))
    np.testing.assert_allclose(result.measured, measurement_data.meas)
    assert result.metadata["user_meta"] == "kept"
    assert result.metadata["solver_meta"] == "absolute"
    assert result.metadata["solver"] == "sparse_bayesian"
    assert result.metadata["shared"] == "solver"
    assert np.shares_memory(
        result.metadata["baseline_used"],
        eit_system.create_homogeneous_image().elem_data,
    )
    np.testing.assert_allclose(result.residual, np.array([0.7, 0.6], dtype=float))


def test_sparse_difference_type_guard_and_projection_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    eit_system = _make_stub_system()
    measurement_data = SimpleNamespace(meas=np.array([1.4, 1.8], dtype=float))
    reference_data = SimpleNamespace(meas=np.array([1.0, 1.1], dtype=float))

    with pytest.raises(TypeError, match="must return SolverOutput"):
        sparse_workflow_module.perform_sparse_difference_reconstruction(
            eit_system=eit_system,
            measurement_data=measurement_data,
            reference_data=reference_data,
            reconstructor=_FactoryReconstructor(output="not-a-solver-output"),
        )

    output = SolverOutput(
        conductivity=np.array([1.3, 1.4], dtype=float),
        simulated_measurement=None,
        likelihood_noise_std=2e-3,
        prior_scale=3e-2,
        metadata={"solver_meta": "difference", "shared": "solver"},
        iterations=1,
        converged=True,
        final_residual=0.05,
        final_relative_change=0.005,
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "resolve_reconstruction_output",
        lambda solver_output, _fwd_model: (
            SimpleNamespace(elem_data=np.array([1.3, 1.4], dtype=float)),
            np.array([1.3, 1.4], dtype=float),
            [0.3],
            [0.1],
        ),
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "difference_measurement",
        lambda _meas, _ref, mode, orientation: SimpleNamespace(
            meas=np.array([0.4, 0.7], dtype=float),
            reference_meas=reference_data.meas.copy(),
            difference_mode=mode,
            difference_orientation=orientation,
        ),
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "project_measurement_vector",
        lambda simulated, **kwargs: simulated - kwargs["reference_meas"],
    )
    monkeypatch.setattr(
        sparse_workflow_module,
        "compute_residuals",
        lambda measured, simulated: (measured - simulated, 0.0, 0.0, 0.0),
    )

    result = sparse_workflow_module.perform_sparse_difference_reconstruction(
        eit_system=eit_system,
        measurement_data=measurement_data,
        reference_data=reference_data,
        baseline_image=None,
        reconstructor=_FactoryReconstructor(output=output),
        metadata={"user_meta": "diff", "shared": "user"},
    )

    np.testing.assert_allclose(result.measured, np.array([0.4, 0.7], dtype=float))
    np.testing.assert_allclose(result.simulated, np.array([-0.8, -0.7], dtype=float))
    np.testing.assert_allclose(
        result.metadata["reference_measured"], reference_data.meas
    )
    assert np.shares_memory(result.metadata["reference_measured"], reference_data.meas)
    assert result.metadata["solver_meta"] == "difference"
    assert result.metadata["user_meta"] == "diff"
    assert result.metadata["shared"] == "solver"
    np.testing.assert_allclose(result.residual, np.array([1.2, 1.4], dtype=float))
