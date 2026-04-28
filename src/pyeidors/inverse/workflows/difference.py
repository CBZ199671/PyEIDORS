"""Difference imaging workflow wrapper."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...core_system_helpers import difference_measurement
from ...data.difference import project_measurement_vector
from ...data.structures import EITData, EITImage
from .base import (
    build_reconstruction_result,
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
    merge_workflow_metadata,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...core_system import EITSystem


def perform_difference_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    reference_data: EITData,
    initial_image: EITImage | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReconstructionResult:
    """Perform difference imaging.

    Args:
        eit_system: Initialized `EITSystem` (setup() must have been called).
        measurement_data: Target frame `EITData`.
        reference_data: Reference frame `EITData`.
        initial_image: Initial conductivity image (optional).
        metadata: Additional info (frame index, etc.).
    """

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem not initialized, please call setup() first.")

    initial_guess = initial_image.elem_data if initial_image is not None else None

    reconstruction = eit_system.inverse_solve(
        data=measurement_data,
        reference_data=reference_data,
        initial_guess=initial_guess,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(reconstruction, eit_system.fwd_model)
    )

    simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)
    diff_data = difference_measurement(
        measurement_data,
        reference_data,
        mode=eit_system.difference_mode,
        orientation=eit_system.difference_orientation,
    )
    measured_vector = diff_data.meas
    simulated_vector = (
        reconstruction.simulated_measurement
        if reconstruction.simulated_measurement is not None
        else project_measurement_vector(
            simulated_data.meas,
            measurement_type="difference",
            reference_meas=diff_data.reference_meas,
            difference_mode=diff_data.difference_mode,
            difference_orientation=diff_data.difference_orientation,
        )
    )
    result_metadata = merge_workflow_metadata(
        {
            "reference_measured": reference_data.meas,
            "display_values": conductivity_values,
            "solver_diagnostics": reconstruction.diagnostics,
        },
        metadata,
    )

    return build_reconstruction_result(
        mode="difference",
        conductivity_values=conductivity_values,
        conductivity_image=conductivity_image,
        measured_vector=measured_vector,
        simulated_vector=simulated_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
        residual_fn=compute_residuals,
    )
