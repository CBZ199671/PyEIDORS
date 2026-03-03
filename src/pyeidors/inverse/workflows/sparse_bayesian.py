"""Sparse Bayesian imaging workflows."""

from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np

from .base import (
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
)
from ..solvers.sparse_bayesian import (
    SparseBayesianConfig,
    SparseBayesianReconstructor,
)
from ...data.structures import EITData, EITImage

try:  # pragma: no cover - optional import guard for type checking
    from ...core_system import EITSystem
except ImportError:  # pragma: no cover
    EITSystem = Any  # type: ignore


def _ensure_reconstructor(
    eit_system: "EITSystem",
    reconstructor: Optional[SparseBayesianReconstructor],
    config: Optional[SparseBayesianConfig],
) -> SparseBayesianReconstructor:
    if reconstructor is not None:
        return reconstructor
    return SparseBayesianReconstructor(
        eit_system=eit_system,
        config=config,
    )


def perform_sparse_absolute_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    baseline_image: Optional[EITImage] = None,
    reconstructor: Optional[SparseBayesianReconstructor] = None,
    config: Optional[SparseBayesianConfig] = None,
    noise_std: Optional[float] = None,
    prior_scale: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReconstructionResult:
    """Execute sparse Bayesian absolute imaging."""

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem must be initialised before reconstruction.")

    baseline_image = baseline_image or eit_system.create_homogeneous_image()
    solver = _ensure_reconstructor(eit_system, reconstructor, config)

    solver_output = solver.reconstruct(
        measurement_data=measurement_data,
        baseline_image=baseline_image,
        noise_std=noise_std,
        prior_scale=prior_scale,
        metadata=metadata,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(solver_output, eit_system.fwd_model)
    )

    simulated_vector = solver_output.get("simulated_measurement")
    if simulated_vector is None:
        simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)
        simulated_vector = simulated_data.meas

    measured_vector = measurement_data.meas
    residual_vector, _, _, _ = compute_residuals(measured_vector, simulated_vector)

    result_metadata: Dict[str, Any] = {
        "baseline_used": baseline_image.elem_data.copy(),
        "display_values": conductivity_values,
        "solver": "sparse_bayesian",
        "likelihood_noise_std": solver_output.get("likelihood_noise_std"),
        "prior_scale": solver_output.get("prior_scale"),
    }
    if metadata:
        result_metadata.update(metadata)
    if "metadata" in solver_output:
        result_metadata.update(solver_output["metadata"])

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


def perform_sparse_difference_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    reference_data: EITData,
    baseline_image: Optional[EITImage] = None,
    reconstructor: Optional[SparseBayesianReconstructor] = None,
    config: Optional[SparseBayesianConfig] = None,
    noise_std: Optional[float] = None,
    prior_scale: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReconstructionResult:
    """Execute sparse Bayesian difference imaging."""

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem must be initialised before reconstruction.")

    baseline_image = baseline_image or eit_system.create_homogeneous_image()
    solver = _ensure_reconstructor(eit_system, reconstructor, config)

    solver_output = solver.reconstruct(
        measurement_data=measurement_data,
        baseline_image=baseline_image,
        reference_data=reference_data,
        noise_std=noise_std,
        prior_scale=prior_scale,
        metadata=metadata,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(solver_output, eit_system.fwd_model)
    )

    simulated_vector = solver_output.get("simulated_measurement")
    if simulated_vector is None:
        simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)
        simulated_vector = simulated_data.meas

    measured_vector = measurement_data.meas - reference_data.meas
    predicted_vector = simulated_vector - reference_data.meas
    residual_vector, _, _, _ = compute_residuals(measured_vector, predicted_vector)

    result_metadata: Dict[str, Any] = {
        "reference_measured": reference_data.meas.copy(),
        "display_values": conductivity_values,
        "solver": "sparse_bayesian",
        "likelihood_noise_std": solver_output.get("likelihood_noise_std"),
        "prior_scale": solver_output.get("prior_scale"),
    }
    if metadata:
        result_metadata.update(metadata)
    if "metadata" in solver_output:
        result_metadata.update(solver_output["metadata"])

    return ReconstructionResult(
        mode="difference",
        conductivity=conductivity_values,
        conductivity_image=conductivity_image,
        measured=measured_vector,
        simulated=predicted_vector,
        residual=residual_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
    )
