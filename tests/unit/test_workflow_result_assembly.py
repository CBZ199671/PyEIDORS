"""Contracts for shared workflow result assembly helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyeidors.inverse.workflows.base import (
    build_reconstruction_result,
    merge_workflow_metadata,
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
