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
    require_initialized,
    resolve_difference_vectors,
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

    require_initialized(
        eit_system,
        message="EITSystem not initialized, please call setup() first.",
    )

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
    if reconstruction.simulated_measurement is not None:
        simulated_source = reconstruction.simulated_measurement
        simulated_space = "difference"
    else:
        simulated_source = simulated_data.meas
        simulated_space = "raw"
    measured_vector, simulated_vector, _ = resolve_difference_vectors(
        measurement_data=measurement_data,
        reference_data=reference_data,
        difference_mode=eit_system.difference_mode,
        difference_orientation=eit_system.difference_orientation,
        simulated_vector=simulated_source,
        simulated_measurement_space=simulated_space,
        difference_fn=difference_measurement,
        project_fn=project_measurement_vector,
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
