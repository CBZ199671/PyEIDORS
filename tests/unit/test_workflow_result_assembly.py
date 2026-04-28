"""Contracts for shared workflow result assembly helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyeidors.inverse.workflows.base import (
    build_reconstruction_result,
    merge_workflow_metadata,
    resolve_difference_vectors,
)


def test_merge_workflow_metadata_keeps_later_precedence() -> None:
    merged = merge_workflow_metadata(
        {"source": "base", "shared": "base"},
        None,
        {"user": "kept", "shared": "user"},
        {"solver_meta": "kept", "shared": "solver"},
    )

    assert merged == {
        "source": "base",
        "user": "kept",
        "solver_meta": "kept",
        "shared": "solver",
    }


def test_build_reconstruction_result_uses_injected_residual_fn() -> None:
    measured = np.array([1.0, 2.0], dtype=float)
    simulated = np.array([0.25, 0.5], dtype=float)

    result = build_reconstruction_result(
        mode="absolute",
        conductivity_values=np.array([1.1, 1.2], dtype=float),
        conductivity_image=SimpleNamespace(elem_data=np.array([1.1, 1.2])),
        measured_vector=measured,
        simulated_vector=simulated,
        residual_history=[0.3],
        sigma_change_history=[0.1],
        metadata={"case": "injected"},
        residual_fn=lambda m, s: (m - s, 0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(result.residual, np.array([0.75, 1.5], dtype=float))
    assert result.metadata == {"case": "injected"}
    assert result.residual_history == [0.3]
    assert result.sigma_change_history == [0.1]


def test_resolve_difference_vectors_keeps_preprojected_simulated_vector() -> None:
    measurement = SimpleNamespace(meas=np.array([1.4, 1.8], dtype=float))
    reference = SimpleNamespace(meas=np.array([1.0, 1.1], dtype=float))
    simulated = np.array([0.2, 0.3], dtype=float)

    def _difference_fn(_meas, _ref, mode, orientation):
        return SimpleNamespace(
            meas=np.array([0.4, 0.7], dtype=float),
            reference_meas=reference.meas.copy(),
            difference_mode=mode,
            difference_orientation=orientation,
        )

    def _project_unexpected(*_args, **_kwargs):
        raise AssertionError("difference-space simulated vector must not project")

    measured, resolved, diff_data = resolve_difference_vectors(
        measurement_data=measurement,
        reference_data=reference,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        simulated_vector=simulated,
        simulated_measurement_space="difference",
        difference_fn=_difference_fn,
        project_fn=_project_unexpected,
    )

    np.testing.assert_allclose(measured, np.array([0.4, 0.7], dtype=float))
    np.testing.assert_allclose(resolved, simulated)
    assert diff_data.difference_mode == "normalized"


def test_resolve_difference_vectors_projects_raw_simulated_vector() -> None:
    measurement = SimpleNamespace(meas=np.array([1.4, 1.8], dtype=float))
    reference = SimpleNamespace(meas=np.array([1.0, 1.1], dtype=float))
    simulated_raw = np.array([0.2, 0.4], dtype=float)

    def _difference_fn(_meas, _ref, mode, orientation):
        return SimpleNamespace(
            meas=np.array([0.4, 0.7], dtype=float),
            reference_meas=reference.meas.copy(),
            difference_mode=mode,
            difference_orientation=orientation,
        )

    def _project_fn(simulated, **kwargs):
        assert kwargs["measurement_type"] == "difference"
        assert kwargs["difference_mode"] == "raw"
        return simulated - kwargs["reference_meas"]

    measured, resolved, _ = resolve_difference_vectors(
        measurement_data=measurement,
        reference_data=reference,
        difference_mode="raw",
        difference_orientation="target_minus_reference",
        simulated_vector=simulated_raw,
        simulated_measurement_space="raw",
        difference_fn=_difference_fn,
        project_fn=_project_fn,
    )

    np.testing.assert_allclose(measured, np.array([0.4, 0.7], dtype=float))
    np.testing.assert_allclose(resolved, np.array([-0.8, -0.7], dtype=float))
