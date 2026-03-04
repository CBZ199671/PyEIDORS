"""Runtime iteration helpers for the Gauss-Newton solver."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from dolfinx import fem

from ...data.structures import EITImage
from ...femx import function_get_array, function_set_array
from ..contracts import SolverOutput
from .gauss_newton_weights import build_weight_reference


def ensure_measurement_weights(reconstructor, sigma_function: fem.Function) -> None:
    """Compute measurement weights based on baseline forward solution."""
    strategy = reconstructor.measurement_weight_strategy
    if not reconstructor.use_measurement_weights or strategy == "none":
        reconstructor._meas_weight_sqrt = None
        reconstructor._baseline_measurement = None
        return

    img = EITImage(elem_data=function_get_array(sigma_function), fwd_model=reconstructor.fwd_model)
    baseline_data, _ = reconstructor.fwd_model.fwd_solve(img)
    baseline_vector = baseline_data.meas.astype(np.float64)
    reconstructor._baseline_measurement = baseline_vector.copy()

    reference_vector = build_weight_reference(
        strategy=strategy,
        baseline_vector=baseline_vector,
        measured_vector=reconstructor._measured_vector,
        floor=reconstructor.weight_floor,
    )

    weights = reference_vector ** 2
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, reconstructor.weight_floor)
    median = np.median(weights)
    if median > 0:
        weights = weights / median

    reconstructor._meas_weight_sqrt = torch.from_numpy(np.sqrt(weights)).to(
        reconstructor.device,
        dtype=reconstructor._torch_dtype,
    )
    if reconstructor.verbose:
        finite_weights = weights[np.isfinite(weights)]
        w_min = finite_weights.min() if finite_weights.size else float("nan")
        w_max = finite_weights.max() if finite_weights.size else float("nan")
        w_med = np.median(finite_weights) if finite_weights.size else float("nan")
        print(
            f"[INFO] measurement weights ({strategy}): min={w_min:.3e}, med={w_med:.3e}, max={w_max:.3e}"
        )


def run_reconstruction(
    reconstructor,
    measured_data: Union[object, np.ndarray],
    initial_conductivity: float = 1.0,
    jacobian_method: str = "efficient",
    prior_data: Optional[np.ndarray] = None,
    record_conductivity_history: bool = False,
    conductivity_history_stride: int = 1,
) -> SolverOutput:
    """Execute Gauss-Newton iterations and return typed solver output."""
    reconstructor._meas_weight_sqrt = None
    reconstructor._baseline_measurement = None

    if hasattr(measured_data, "meas"):
        meas_vector = measured_data.meas
    else:
        meas_vector = measured_data.flatten()

    if len(meas_vector) != reconstructor.n_measurements:
        raise ValueError(
            f"Measurement data length mismatch: {len(meas_vector)} vs {reconstructor.n_measurements}"
        )

    meas_torch = torch.from_numpy(meas_vector).to(reconstructor.device, dtype=reconstructor._torch_dtype)
    reconstructor._measured_vector = meas_vector.copy()

    if reconstructor.R_torch is None:
        R_np = reconstructor.regularization.get_regularization_matrix()
        reconstructor.R_torch = torch.from_numpy(R_np).to(
            reconstructor.device,
            dtype=reconstructor._torch_dtype,
        )

    meas_norm = torch.norm(meas_torch).item()
    meas_max = torch.max(torch.abs(meas_torch)).item()
    meas_weighted_norm = None

    if initial_conductivity is None:
        initial_conductivity = 1.0
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    if np.isscalar(initial_conductivity):
        function_set_array(
            sigma_current,
            np.full(reconstructor.n_elements, float(initial_conductivity), dtype=float),
        )
    else:
        function_set_array(sigma_current, np.asarray(initial_conductivity).flatten())
    reconstructor._ensure_measurement_weights(sigma_current)

    if reconstructor._meas_weight_sqrt is not None:
        meas_weighted_norm = torch.norm(meas_torch * reconstructor._meas_weight_sqrt).item()

    if prior_data is not None:
        reconstructor._prior_data = np.asarray(prior_data).flatten()
    elif np.isscalar(initial_conductivity):
        reconstructor._prior_data = np.full(reconstructor.n_elements, initial_conductivity)
    else:
        reconstructor._prior_data = np.asarray(initial_conductivity).flatten()
    prior_torch = torch.from_numpy(reconstructor._prior_data).to(
        reconstructor.device,
        dtype=reconstructor._torch_dtype,
    )

    residual_history = []
    sigma_change_history = []
    iteration_logs = []
    conductivity_history = []
    history_stride = max(1, int(conductivity_history_stride))
    if record_conductivity_history:
        conductivity_history.append(function_get_array(sigma_current).copy())

    consecutive_rollbacks = 0
    max_consecutive_rollbacks = 5

    if reconstructor.verbose:
        print(f"[INFO] lambda={reconstructor.regularization_param:.3e}")
        print("\nStarting modular Gauss-Newton reconstruction...")
        print(f"Using Jacobian method: {jacobian_method}")

    prev_residual = None
    relative_change = float("inf")

    with reconstructor._progress(total=reconstructor.max_iterations) as pbar:
        for iteration in range(reconstructor.max_iterations):
            sigma_array = function_get_array(sigma_current)
            img_current = EITImage(elem_data=sigma_array, fwd_model=reconstructor.fwd_model)
            data_simulated, _ = reconstructor.fwd_model.fwd_solve(img_current)

            data_sim_torch = torch.from_numpy(data_simulated.meas).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            residual_torch = data_sim_torch - meas_torch
            if reconstructor._meas_weight_sqrt is not None:
                weighted_residual_torch = residual_torch * reconstructor._meas_weight_sqrt
            else:
                weighted_residual_torch = residual_torch

            residual_norm_weighted = torch.norm(weighted_residual_torch).item()
            residual_norm = torch.norm(residual_torch).item()
            residual_max = torch.max(torch.abs(residual_torch)).item()

            sigma_vec_torch = torch.from_numpy(sigma_array).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            de_current = sigma_vec_torch - prior_torch
            meas_misfit = 0.5 * torch.dot(weighted_residual_torch, weighted_residual_torch).item()
            lambda_eff = reconstructor.regularization_param
            RtR_de = torch.mv(reconstructor.R_torch, de_current)
            prior_misfit = 0.5 * lambda_eff * torch.dot(de_current, RtR_de).item()
            total_objective = meas_misfit + prior_misfit

            residual_history.append(residual_norm)
            res_drop = None if prev_residual is None else prev_residual - residual_norm

            measurement_jacobian_np = reconstructor.jacobian_calculator.calculate(
                sigma_current,
                method=jacobian_method,
            )
            if reconstructor.negate_jacobian:
                measurement_jacobian_np = -measurement_jacobian_np
            J_torch = torch.from_numpy(measurement_jacobian_np).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            if reconstructor._meas_weight_sqrt is not None:
                J_weighted = J_torch * reconstructor._meas_weight_sqrt.unsqueeze(1)
            else:
                J_weighted = J_torch

            JTJ = torch.mm(J_weighted.t(), J_weighted)
            JTr = torch.mv(J_weighted.t(), weighted_residual_torch)

            sigma_current_torch = torch.from_numpy(sigma_array).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            de_torch = sigma_current_torch - prior_torch
            A = JTJ + lambda_eff * reconstructor.R_torch
            if reconstructor.use_prior_term:
                RtR_de = torch.mv(reconstructor.R_torch, de_torch)
                b = -(JTr + lambda_eff * RtR_de)
            else:
                b = -JTr

            pred_norm = torch.norm(data_sim_torch).item()
            pred_max = torch.max(torch.abs(data_sim_torch)).item()
            jtr_norm = torch.norm(JTr).item()
            rel_residual = residual_norm / (meas_norm + 1e-12)
            rel_residual_weighted = (
                residual_norm_weighted / (meas_weighted_norm + 1e-12)
                if meas_weighted_norm
                else None
            )

            try:
                delta_sigma_torch = torch.linalg.solve(A, b)
            except RuntimeError:
                A_regularized = JTJ + (reconstructor.regularization_param * 10) * reconstructor.R_torch
                delta_sigma_torch = torch.linalg.solve(A_regularized, b)
            delta_norm = torch.norm(delta_sigma_torch).item()

            if reconstructor.step_schedule is not None and iteration < len(reconstructor.step_schedule):
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
                if reconstructor.min_step is not None and optimal_step_size < reconstructor.min_step:
                    optimal_step_size = reconstructor.min_step

            sigma_old_values = sigma_array.copy()
            sigma_array[:] += optimal_step_size * delta_sigma_torch.cpu().numpy()

            if reconstructor.clip_values is not None:
                function_set_array(
                    sigma_current,
                    np.clip(sigma_array, reconstructor.clip_values[0], reconstructor.clip_values[1]),
                )
                sigma_array = function_get_array(sigma_current)

            sigma_new_torch = torch.from_numpy(sigma_array).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            sigma_old_torch = torch.from_numpy(sigma_old_values).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )

            sigma_change = torch.norm(sigma_new_torch - sigma_old_torch).item()
            relative_change = sigma_change / (torch.norm(sigma_new_torch).item() + 1e-12)

            if prev_residual is not None and residual_norm > prev_residual:
                consecutive_rollbacks += 1
                if reconstructor.verbose:
                    print(
                        f"[WARN] residual increased ({residual_norm:.3e} > {prev_residual:.3e}), "
                        f"rolling back step ({consecutive_rollbacks}/{max_consecutive_rollbacks})"
                    )
                function_set_array(sigma_current, sigma_old_values)
                residual_history[-1] = prev_residual
                sigma_change_history[-1] = 0.0

                if consecutive_rollbacks >= max_consecutive_rollbacks:
                    if reconstructor.verbose:
                        print(f"[STOP] {max_consecutive_rollbacks} consecutive rollbacks, terminating early")
                    break
                continue

            consecutive_rollbacks = 0
            sigma_change_history.append(relative_change)
            if record_conductivity_history and (iteration + 1) % history_stride == 0:
                conductivity_history.append(function_get_array(sigma_current).copy())

            iteration_logs.append(
                {
                    "iteration": iteration,
                    "residual": residual_norm,
                    "residual_weighted": residual_norm_weighted,
                    "relative_residual": rel_residual,
                    "relative_residual_weighted": rel_residual_weighted,
                    "residual_max": residual_max,
                    "meas_norm": meas_norm,
                    "pred_norm": pred_norm,
                    "meas_max": meas_max,
                    "pred_max": pred_max,
                    "JTr_norm": jtr_norm,
                    "delta_norm": delta_norm,
                    "step": optimal_step_size,
                    "lambda_eff": lambda_eff,
                    "relative_change": relative_change,
                    "res_drop": res_drop,
                    "meas_misfit": meas_misfit,
                    "prior_misfit": prior_misfit,
                    "total_objective": total_objective,
                }
            )
            prev_residual = residual_norm

            if relative_change < reconstructor.convergence_tol and iteration + 1 >= reconstructor.min_iterations:
                if reconstructor.verbose:
                    print(
                        f"\nConverged! Iteration {iteration}, relative change: {relative_change:.2e}"
                    )
                break

            if reconstructor.verbose:
                pbar.set_postfix_str(
                    f"residual={residual_norm:.2e}, step={optimal_step_size:.3f}, Δσ={relative_change:.2e}"
                )
                pbar.update(1)

    cache_stats = {}
    if getattr(reconstructor, "cache_manager", None) is not None:
        cache_stats = reconstructor.cache_manager.stats()
    backend_info = {
        "linear_backend": getattr(reconstructor.fwd_model, "linear_backend", "unknown"),
        "performance_mode": getattr(reconstructor, "performance_mode", "aggressive"),
        "forward_cache_lookup": getattr(reconstructor.fwd_model, "_last_cache_lookup", {}),
        "jacobian_cache_lookup": getattr(reconstructor.jacobian_calculator, "_last_cache_lookup", {}),
    }

    results = SolverOutput(
        conductivity=sigma_current,
        residual_history=residual_history,
        sigma_change_history=sigma_change_history,
        iterations=len(residual_history),
        converged=relative_change < reconstructor.convergence_tol,
        final_residual=residual_history[-1],
        final_relative_change=relative_change,
        jacobian_method=jacobian_method,
        regularization_type=type(reconstructor.regularization).__name__,
        iteration_logs=iteration_logs,
        conductivity_history=conductivity_history if record_conductivity_history else None,
        baseline_measurement=(
            reconstructor._baseline_measurement.copy()
            if reconstructor._baseline_measurement is not None
            else None
        ),
        measurement_weight=(
            reconstructor._meas_weight_sqrt.detach().cpu().numpy() ** 2
            if reconstructor._meas_weight_sqrt is not None
            else None
        ),
        diagnostics={
            "cache_hits": cache_stats.get("total_hits", 0),
            "cache_misses": cache_stats.get("total_misses", 0),
            "cache_stats": cache_stats,
            "backend_info": backend_info,
        },
    )

    if reconstructor.verbose:
        print("\nReconstruction complete:")
        print(f"  Iterations: {results.iterations}")
        print(f"  Final residual: {results.final_residual:.2e}")
        print(f"  Jacobian method: {jacobian_method}")
        print(f"  Regularization type: {results.regularization_type}")

    return results
