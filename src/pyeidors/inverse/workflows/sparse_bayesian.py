"""Sparse Bayesian imaging workflows."""

from __future__ import annotations

from typing import Any

from ...core_system_helpers import difference_measurement
from ...data.difference import project_measurement_vector
from ...data.structures import EITData, EITImage
from ..contracts import SolverOutput
from ..solvers.sparse_bayesian_engine import (
    SparseBayesianConfig,
    SparseBayesianReconstructor,
)
from .base import (
    build_reconstruction_result,
    ReconstructionResult,
    compute_residuals,
    merge_workflow_metadata,
    resolve_reconstruction_output,
)

try:  # pragma: no cover - optional import guard for type checking
    from ...core_system import EITSystem
except ImportError:  # pragma: no cover
    EITSystem = Any  # type: ignore


def _ensure_reconstructor(
    eit_system: "EITSystem",
    reconstructor: SparseBayesianReconstructor | None,
    config: SparseBayesianConfig | None,
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
    baseline_image: EITImage | None = None,
    reconstructor: SparseBayesianReconstructor | None = None,
    config: SparseBayesianConfig | None = None,
    noise_std: float | None = None,
    prior_scale: float | None = None,
    metadata: dict[str, Any] | None = None,
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
    if not isinstance(solver_output, SolverOutput):
        raise TypeError(
            "SparseBayesianReconstructor must return SolverOutput. "
            f"Received {type(solver_output).__name__}."
        )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(solver_output, eit_system.fwd_model)
    )

    simulated_vector = solver_output.simulated_measurement
    if simulated_vector is None:
        simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)
        simulated_vector = simulated_data.meas

    measured_vector = measurement_data.meas
    result_metadata = merge_workflow_metadata(
        {
            "baseline_used": baseline_image.elem_data.copy(),
            "display_values": conductivity_values,
            "solver": "sparse_bayesian",
            "likelihood_noise_std": solver_output.likelihood_noise_std,
            "prior_scale": solver_output.prior_scale,
        },
        metadata,
        solver_output.metadata,
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


def perform_sparse_difference_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    reference_data: EITData,
    baseline_image: EITImage | None = None,
    reconstructor: SparseBayesianReconstructor | None = None,
    config: SparseBayesianConfig | None = None,
    noise_std: float | None = None,
    prior_scale: float | None = None,
    metadata: dict[str, Any] | None = None,
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
    if not isinstance(solver_output, SolverOutput):
        raise TypeError(
            "SparseBayesianReconstructor must return SolverOutput. "
            f"Received {type(solver_output).__name__}."
        )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(solver_output, eit_system.fwd_model)
    )

    simulated_vector = solver_output.simulated_measurement
    if simulated_vector is None:
        simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)
        simulated_vector = simulated_data.meas

    diff_data = difference_measurement(
        measurement_data,
        reference_data,
        mode=eit_system.difference_mode,
        orientation=eit_system.difference_orientation,
    )
    measured_vector = diff_data.meas
    predicted_vector = project_measurement_vector(
        simulated_vector,
        measurement_type="difference",
        reference_meas=diff_data.reference_meas,
        difference_mode=diff_data.difference_mode,
        difference_orientation=diff_data.difference_orientation,
    )
    result_metadata = merge_workflow_metadata(
        {
            "reference_measured": reference_data.meas.copy(),
            "display_values": conductivity_values,
            "solver": "sparse_bayesian",
            "likelihood_noise_std": solver_output.likelihood_noise_std,
            "prior_scale": solver_output.prior_scale,
        },
        metadata,
        solver_output.metadata,
    )

    return build_reconstruction_result(
        mode="difference",
        conductivity_values=conductivity_values,
        conductivity_image=conductivity_image,
        measured_vector=measured_vector,
        simulated_vector=predicted_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
        residual_fn=compute_residuals,
    )
