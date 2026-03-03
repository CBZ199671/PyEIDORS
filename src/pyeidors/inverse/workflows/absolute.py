"""Absolute imaging workflow wrapper."""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from typing import TYPE_CHECKING

from ...data.structures import EITData, EITImage
from .base import (
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...core_system import EITSystem


def perform_absolute_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    baseline_image: Optional[EITImage] = None,
    initial_image: Optional[EITImage] = None,
    metadata: Optional[Dict[str, Any]] = None,
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

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem not initialized, please call setup() first.")

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
    residual_vector, l2_error, rel_error, mse = compute_residuals(
        measured_vector, simulated_vector
    )

    result_metadata: Dict[str, Any] = {
        "display_values": conductivity_values,
        "baseline_used": baseline_elem,
    }
    if metadata:
        result_metadata.update(metadata)

    return ReconstructionResult(
        mode="absolute",
        conductivity=conductivity_values,
        conductivity_image=conductivity_image,
        measured=measured_vector,
        simulated=simulated_vector,
        residual=residual_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
    )
