"""Step-size helpers for the Gauss-Newton runtime."""

from __future__ import annotations

import numpy as np
import torch
from dolfinx import fem
from scipy.optimize import minimize_scalar

from ...data.structures import EITImage
from .gauss_newton_linear_system import _require_scalar_finite
from .gauss_newton_measurement_space import _project_simulated_measurements


def _difference_step_size_objective(
    reconstructor,
    *,
    prior_sigma: np.ndarray,
    delta_sigma: np.ndarray,
    measured_vector: np.ndarray,
    alpha: float,
) -> float:
    sigma = np.asarray(prior_sigma + float(alpha) * delta_sigma, dtype=np.float64)
    clip_values = getattr(reconstructor, "clip_values", None)
    if clip_values is not None:
        sigma = np.clip(sigma, clip_values[0], clip_values[1])
    image = EITImage(elem_data=sigma, fwd_model=reconstructor.fwd_model)
    simulated, _ = reconstructor.fwd_model.fwd_solve(image)
    simulated_vector = _project_simulated_measurements(reconstructor, simulated.meas)
    residual = np.asarray(simulated_vector, dtype=np.float64) - measured_vector
    return float(np.dot(residual, residual))


def _apply_difference_step_size(
    reconstructor,
    *,
    sigma_final: np.ndarray,
    measured_vector: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    mode = (
        str(getattr(reconstructor, "difference_step_size_mode", "off")).strip().lower()
    )
    info: dict[str, object] = {
        "mode": mode,
        "applied": False,
        "value": 1.0,
        "objective": None,
    }
    if getattr(reconstructor, "_measurement_space_type", "real") != "difference":
        info["reason"] = "real_measurement_space"
        return np.asarray(sigma_final, dtype=np.float64), info
    if mode == "off":
        info["reason"] = "disabled"
        return np.asarray(sigma_final, dtype=np.float64), info
    if str(getattr(reconstructor, "active_preset_name", "")).strip().lower() not in {
        "eidors_one_step_noser",
        "eidors_demo3d_tv",
    }:
        info["reason"] = "preset_not_one_step"
        return np.asarray(sigma_final, dtype=np.float64), info

    prior_raw = getattr(reconstructor, "_prior_data", None)
    if prior_raw is None:
        info["reason"] = "missing_prior"
        return np.asarray(sigma_final, dtype=np.float64), info
    prior_sigma = np.asarray(prior_raw, dtype=np.float64).reshape(-1)
    if prior_sigma.shape[0] != sigma_final.shape[0]:
        info["reason"] = "missing_prior"
        return np.asarray(sigma_final, dtype=np.float64), info

    delta_sigma = np.asarray(sigma_final - prior_sigma, dtype=np.float64)
    if np.linalg.norm(delta_sigma) <= 1e-18:
        info["reason"] = "zero_delta"
        return np.asarray(sigma_final, dtype=np.float64), info

    if mode == "fixed":
        alpha = float(
            1.0
            if reconstructor.difference_step_size_value is None
            else reconstructor.difference_step_size_value
        )
        objective = _difference_step_size_objective(
            reconstructor,
            prior_sigma=prior_sigma,
            delta_sigma=delta_sigma,
            measured_vector=measured_vector,
            alpha=alpha,
        )
        info["applied"] = True
        info["value"] = alpha
        info["objective"] = objective
        return np.asarray(prior_sigma + alpha * delta_sigma, dtype=np.float64), info

    bounds = tuple(
        float(v)
        for v in getattr(reconstructor, "difference_step_size_bounds", (0.0, 4.0))
    )
    options = {"xatol": 1e-3, "maxiter": 32}
    options.update(
        dict(getattr(reconstructor, "difference_step_size_fmin_options", {}) or {})
    )
    info["bounds"] = [float(bounds[0]), float(bounds[1])]
    calls = {"count": 0}

    def _objective(alpha: float) -> float:
        calls["count"] += 1
        return _difference_step_size_objective(
            reconstructor,
            prior_sigma=prior_sigma,
            delta_sigma=delta_sigma,
            measured_vector=measured_vector,
            alpha=float(alpha),
        )

    try:
        result = minimize_scalar(
            _objective,
            bounds=bounds,
            method="bounded",
            options=options,
        )
        alpha = float(result.x)
        objective = float(result.fun)
        info["success"] = bool(result.success)
        info["message"] = str(result.message)
    except Exception as exc:
        info["reason"] = f"optimization_failed:{type(exc).__name__}"
        info["eval_count"] = int(calls["count"])
        return np.asarray(sigma_final, dtype=np.float64), info

    info["eval_count"] = int(calls["count"])
    info["applied"] = True
    info["value"] = alpha
    info["objective"] = objective
    return np.asarray(prior_sigma + alpha * delta_sigma, dtype=np.float64), info


def _select_step_size(
    reconstructor,
    iteration: int,
    sigma_current: fem.Function,
    delta_sigma_torch: torch.Tensor,
    meas_torch: torch.Tensor,
    residual_norm_weighted: float,
    prior_torch: torch.Tensor,
    lambda_eff: float,
) -> float:
    if (
        getattr(reconstructor, "_measurement_space_type", "real") == "difference"
        and str(getattr(reconstructor, "active_preset_name", "")).strip().lower()
        in {"eidors_one_step_noser", "eidors_demo3d_tv"}
        and int(getattr(reconstructor, "max_iterations", 1)) <= 1
    ):
        return 1.0

    if reconstructor.solver_mode == "fast" and reconstructor.line_search_mode == "fast":
        quick_step = min(float(reconstructor.max_step), 1.0)
        if reconstructor.min_step is not None:
            quick_step = max(float(reconstructor.min_step), quick_step)
        _require_scalar_finite("optimal_step_size", quick_step, iteration)
        return quick_step

    if reconstructor.step_schedule is not None and iteration < len(
        reconstructor.step_schedule
    ):
        optimal_step_size = float(reconstructor.step_schedule[iteration])
    else:
        optimal_step_size = reconstructor._line_search_torch(
            sigma_current,
            delta_sigma_torch,
            meas_torch,
            residual_norm_weighted,
            reconstructor._meas_weight_sqrt,
            prior_torch=prior_torch,
            lambda_eff=lambda_eff,
        )
        if (
            reconstructor.min_step is not None
            and optimal_step_size < reconstructor.min_step
        ):
            optimal_step_size = reconstructor.min_step
    _require_scalar_finite("optimal_step_size", optimal_step_size, iteration)
    return optimal_step_size
