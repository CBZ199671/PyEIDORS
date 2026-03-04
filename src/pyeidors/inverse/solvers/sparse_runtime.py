"""Runtime reconstruction flow for sparse Bayesian solver."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from dolfinx import fem

from ...data.structures import EITData, EITImage
from ...femx import function_set_array
from ..contracts import SolverOutput


def run_sparse_reconstruction(
    reconstructor,
    measurement_data: EITData,
    baseline_image: Optional[EITImage] = None,
    reference_data: Optional[EITData] = None,
    initial_conductivity: float = 1.0,
    noise_std: Optional[float] = None,
    prior_scale: Optional[float] = None,
    clip_values: Optional[Tuple[float, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SolverOutput:
    """Compute one sparse Bayesian reconstruction step."""
    mode = "difference" if reference_data is not None else "absolute"
    clip_bounds = clip_values if clip_values is not None else reconstructor.config.clip_values

    baseline_image = baseline_image or reconstructor._create_homogeneous_image(initial_conductivity)
    baseline_values = baseline_image.elem_data.copy()

    baseline_meas = reconstructor._forward_measurement(baseline_values)
    jacobian = reconstructor._prepare_jacobian(baseline_values)

    target_vector = np.asarray(measurement_data.meas, dtype=float).ravel()
    if mode == "difference":
        reference_vector = np.asarray(reference_data.meas, dtype=float).ravel()
        data_vector = target_vector - reference_vector
        baseline_meas = reference_vector
    else:
        data_vector = target_vector - baseline_meas

    noise_sigma = noise_std or reconstructor._estimate_noise_level(data_vector)
    prior_scale = prior_scale or reconstructor.config.prior_scale
    map_delta = reconstructor._solve_sparse_map(jacobian, data_vector, noise_sigma, prior_scale)

    conductivity_values = baseline_values + map_delta
    if clip_bounds is not None:
        conductivity_values = np.clip(conductivity_values, clip_bounds[0], clip_bounds[1])

    conductivity_function = fem.Function(reconstructor.fwd_model.V_sigma)
    function_set_array(conductivity_function, conductivity_values)

    simulated_vector = reconstructor._forward_measurement(conductivity_values)
    predicted_vector = simulated_vector - baseline_meas
    residual_vector = predicted_vector - data_vector

    merged_metadata: Dict[str, Any] = {"mode": mode}
    if metadata:
        merged_metadata.update(metadata)

    diagnostics: Dict[str, Any] = {
        "delta_sigma": map_delta,
        "baseline_conductivity": baseline_values,
        "posterior_map": map_delta,
        "jacobian": jacobian,
        "observed_data": data_vector,
        "predicted_data": predicted_vector,
        "residual_vector": residual_vector,
        "target_measurement": target_vector,
        "clip_bounds": clip_bounds,
    }
    cache_stats = {}
    if getattr(reconstructor.eit_system, "cache_manager", None) is not None:
        cache_stats = reconstructor.eit_system.cache_manager.stats()
    diagnostics.update(
        {
            "cache_hits": cache_stats.get("total_hits", 0),
            "cache_misses": cache_stats.get("total_misses", 0),
            "cache_stats": cache_stats,
            "backend_info": {
                "linear_backend": getattr(reconstructor.fwd_model, "linear_backend", "unknown"),
                "performance_mode": getattr(reconstructor.eit_system, "performance_mode", "aggressive"),
            },
        }
    )

    return SolverOutput(
        conductivity=conductivity_function,
        residual_history=None,
        sigma_change_history=None,
        iterations=1,
        converged=True,
        final_residual=float(np.linalg.norm(residual_vector)),
        final_relative_change=float(np.linalg.norm(map_delta) / (np.linalg.norm(conductivity_values) + 1e-12)),
        simulated_measurement=simulated_vector,
        baseline_measurement=baseline_meas,
        likelihood_noise_std=noise_sigma,
        prior_scale=prior_scale,
        metadata=merged_metadata,
        diagnostics=diagnostics,
    )
