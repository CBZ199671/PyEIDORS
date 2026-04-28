"""Absolute imaging workflow wrapper."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...data.structures import EITData, EITImage
from .base import (
    build_reconstruction_result,
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
    merge_workflow_metadata,
    require_initialized,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...core_system import EITSystem


def perform_absolute_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    baseline_image: EITImage | None = None,
    initial_image: EITImage | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReconstructionResult:
    """Perform absolute imaging.

    Args:
        eit_system: Initialized `EITSystem` instance (setup() must have been called).
        measurement_data: Current frame measurement data (`EITData`), typically from real or simulated data.
        baseline_image: Homogeneous conductivity image, used for forward baseline and displaying difference.
                        If not provided, `create_homogeneous_image()` is used automatically.
        initial_image: Initial guess for solver, defaults to `baseline_image`.
        metadata: Additional info (frame index, frequency, etc.), stored as-is in result.
    """

    require_initialized(
        eit_system,
        message="EITSystem not initialized, please call setup() first.",
    )

    if baseline_image is None:
        baseline_image = eit_system.create_homogeneous_image()

    baseline_elem = baseline_image.elem_data.copy()
    initial_guess = (
        initial_image.elem_data if initial_image is not None else baseline_elem
    )

    reconstruction = eit_system.inverse_solve(
        data=measurement_data,
        reference_data=None,
        initial_guess=initial_guess,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(reconstruction, eit_system.fwd_model)
    )

    simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)

    measured_vector = measurement_data.meas
    simulated_vector = simulated_data.meas
    result_metadata = merge_workflow_metadata(
        {
            "display_values": conductivity_values,
            "baseline_used": baseline_elem,
            "solver_diagnostics": reconstruction.diagnostics,
        },
        metadata,
    )

    return build_reconstruction_result(
        mode="absolute",
        conductivity_values=conductivity_values,
        conductivity_image=conductivity_image,
        measured_vector=measured_vector,
        simulated_vector=simulated_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
        residual_fn=compute_residuals,
    )
