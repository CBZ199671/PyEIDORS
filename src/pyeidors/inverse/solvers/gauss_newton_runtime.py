"""Runtime iteration helpers for the Gauss-Newton solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from dolfinx import fem

from ...data.structures import EITImage
from ...femx import function_get_array, function_set_array
from ..contracts import SolverOutput
from .gauss_newton_weights import build_weight_reference


@dataclass(slots=True)
class _IterationLog:
    iteration: int
    residual: float
    residual_weighted: float
    relative_residual: float
    relative_residual_weighted: float | None
    residual_max: float
    meas_norm: float
    pred_norm: float
    meas_max: float
    pred_max: float
    jtr_norm: float
    delta_norm: float
    step: float
    lambda_eff: float
    relative_change: float
    res_drop: float | None
    meas_misfit: float
    prior_misfit: float
    total_objective: float

    def to_payload(self) -> dict[str, float | int | None]:
        return {
            "iteration": self.iteration,
            "residual": self.residual,
            "residual_weighted": self.residual_weighted,
            "relative_residual": self.relative_residual,
            "relative_residual_weighted": self.relative_residual_weighted,
            "residual_max": self.residual_max,
            "meas_norm": self.meas_norm,
            "pred_norm": self.pred_norm,
            "meas_max": self.meas_max,
            "pred_max": self.pred_max,
            "JTr_norm": self.jtr_norm,
            "delta_norm": self.delta_norm,
            "step": self.step,
            "lambda_eff": self.lambda_eff,
            "relative_change": self.relative_change,
            "res_drop": self.res_drop,
            "meas_misfit": self.meas_misfit,
            "prior_misfit": self.prior_misfit,
            "total_objective": self.total_objective,
        }


def _to_runtime_tensor(reconstructor, values) -> torch.Tensor:
    return torch.as_tensor(
        values,
        device=reconstructor.device,
        dtype=reconstructor._torch_dtype,
    )


def _to_runtime_tensor_cached(
    reconstructor,
    name: str,
    values,
) -> torch.Tensor:
    """Reuse per-iteration tensor buffers to reduce repeated allocations."""
    source = _to_runtime_tensor(reconstructor, values)
    cache = getattr(reconstructor, "_runtime_tensor_cache", None)
    if cache is None:
        cache = {}
        reconstructor._runtime_tensor_cache = cache
    target = cache.get(name)
    if target is None or tuple(target.shape) != tuple(source.shape):
        target = source.clone()
        cache[name] = target
        return target
    target.copy_(source)
    return target


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

    reconstructor._meas_weight_sqrt = _to_runtime_tensor(
        reconstructor,
        np.sqrt(weights),
    )
    if reconstructor.verbose:
        finite_weights = weights[np.isfinite(weights)]
        w_min = finite_weights.min() if finite_weights.size else float("nan")
        w_max = finite_weights.max() if finite_weights.size else float("nan")
        w_med = np.median(finite_weights) if finite_weights.size else float("nan")
        print(
            f"[INFO] measurement weights ({strategy}): min={w_min:.3e}, med={w_med:.3e}, max={w_max:.3e}"
        )


def _finite_summary(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "finite_count=0"
    return (
        f"finite_count={finite.size} "
        f"min={float(finite.min()):.6e} "
        f"max={float(finite.max()):.6e} "
        f"l2={float(np.linalg.norm(finite)):.6e}"
    )


def _require_finite(name: str, values, iteration: int | None = None) -> None:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if np.isfinite(arr).all():
        return
    iter_tag = "init" if iteration is None else str(iteration)
    raise FloatingPointError(
        f"Non-finite values detected in {name} at iteration={iter_tag}. "
        f"{_finite_summary(arr.astype(np.float64, copy=False))}"
    )


def _require_scalar_finite(name: str, value: float, iteration: int | None = None) -> None:
    if np.isfinite(float(value)):
        return
    iter_tag = "init" if iteration is None else str(iteration)
    raise FloatingPointError(
        f"Non-finite scalar detected in {name} at iteration={iter_tag}: {value!r}"
    )


def _extract_measured_vector(measured_data) -> np.ndarray:
    if hasattr(measured_data, "meas"):
        return measured_data.meas
    return measured_data.flatten()


def _init_sigma_function(
    reconstructor,
    initial_conductivity,
) -> tuple[fem.Function, float | np.ndarray]:
    if initial_conductivity is None:
        initial_conductivity = 1.0
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    if np.isscalar(initial_conductivity):
        function_set_array(
            sigma_current,
            np.full(reconstructor.n_elements, float(initial_conductivity), dtype=float),
        )
    else:
        function_set_array(
            sigma_current, np.asarray(initial_conductivity).flatten()
        )
    return sigma_current, initial_conductivity


def _prepare_prior(
    reconstructor,
    prior_data: Optional[np.ndarray],
    initial_conductivity: float | np.ndarray,
) -> torch.Tensor:
    if prior_data is not None:
        reconstructor._prior_data = np.asarray(prior_data).flatten()
    elif np.isscalar(initial_conductivity):
        reconstructor._prior_data = np.full(reconstructor.n_elements, initial_conductivity)
    else:
        reconstructor._prior_data = np.asarray(initial_conductivity).flatten()
    return _to_runtime_tensor(reconstructor, reconstructor._prior_data)


def _compute_residuals(
    reconstructor,
    simulated_meas: np.ndarray,
    meas_torch: torch.Tensor,
    iteration: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
]:
    data_sim_torch = _to_runtime_tensor_cached(
        reconstructor, "simulated_meas", simulated_meas
    )
    residual_torch = data_sim_torch - meas_torch
    if reconstructor._meas_weight_sqrt is not None:
        weighted_residual_torch = residual_torch * reconstructor._meas_weight_sqrt
    else:
        weighted_residual_torch = residual_torch

    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
    residual_norm = torch.norm(residual_torch).item()
    residual_max = torch.max(torch.abs(residual_torch)).item()
    _require_scalar_finite("residual_norm_weighted", residual_norm_weighted, iteration)
    _require_scalar_finite("residual_norm", residual_norm, iteration)
    _require_scalar_finite("residual_max", residual_max, iteration)
    return (
        data_sim_torch,
        residual_torch,
        weighted_residual_torch,
        residual_norm_weighted,
        residual_norm,
        residual_max,
    )


def _compute_objective(
    reconstructor,
    weighted_residual_torch: torch.Tensor,
    de_current: torch.Tensor,
    lambda_eff: float,
    iteration: int,
) -> tuple[float, float, float, torch.Tensor]:
    meas_misfit = 0.5 * torch.dot(weighted_residual_torch, weighted_residual_torch).item()
    RtR_de = torch.mv(reconstructor.R_torch, de_current)
    prior_misfit = 0.5 * lambda_eff * torch.dot(de_current, RtR_de).item()
    total_objective = meas_misfit + prior_misfit
    _require_scalar_finite("meas_misfit", meas_misfit, iteration)
    _require_scalar_finite("prior_misfit", prior_misfit, iteration)
    _require_scalar_finite("total_objective", total_objective, iteration)
    return meas_misfit, prior_misfit, total_objective, RtR_de


def _build_linear_system(
    reconstructor,
    JTJ: torch.Tensor,
    JTr: torch.Tensor,
    de_torch: torch.Tensor,
    lambda_eff: float,
    iteration: int,
    RtR_de: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = JTJ + lambda_eff * reconstructor.R_torch
    if reconstructor.use_prior_term:
        if RtR_de is None:
            RtR_de = torch.mv(reconstructor.R_torch, de_torch)
        b = -(JTr + lambda_eff * RtR_de)
    else:
        b = -JTr
    _require_finite("A", A, iteration)
    _require_finite("b", b, iteration)
    return A, b


def _solve_linear_system(
    reconstructor,
    A: torch.Tensor,
    b: torch.Tensor,
    JTJ: torch.Tensor,
    iteration: int,
) -> tuple[torch.Tensor, float]:
    try:
        delta_sigma_torch = torch.linalg.solve(A, b)
    except RuntimeError:
        A_regularized = JTJ + (reconstructor.regularization_param * 10) * reconstructor.R_torch
        _require_finite("A_regularized", A_regularized, iteration)
        delta_sigma_torch = torch.linalg.solve(A_regularized, b)
    _require_finite("delta_sigma_torch", delta_sigma_torch, iteration)
    delta_norm = torch.norm(delta_sigma_torch).item()
    _require_scalar_finite("delta_norm", delta_norm, iteration)
    return delta_sigma_torch, delta_norm


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
    _require_scalar_finite("optimal_step_size", optimal_step_size, iteration)
    return optimal_step_size


def _maybe_rollback(
    reconstructor,
    sigma_current: fem.Function,
    sigma_old_values: np.ndarray,
    residual_norm: float,
    prev_residual: Optional[float],
    residual_history: list[float],
    sigma_change_history: list[float],
    consecutive_rollbacks: int,
    max_consecutive_rollbacks: int,
) -> tuple[bool, bool, int]:
    if prev_residual is None or residual_norm <= prev_residual:
        return False, False, consecutive_rollbacks

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
        return True, True, consecutive_rollbacks

    return True, False, consecutive_rollbacks


def _record_iteration_log(
    iteration_logs: list[_IterationLog],
    *,
    iteration: int,
    residual_norm: float,
    residual_norm_weighted: float,
    rel_residual: float,
    rel_residual_weighted: float | None,
    residual_max: float,
    meas_norm: float,
    pred_norm: float,
    meas_max: float,
    pred_max: float,
    jtr_norm: float,
    delta_norm: float,
    optimal_step_size: float,
    lambda_eff: float,
    relative_change: float,
    res_drop: float | None,
    meas_misfit: float,
    prior_misfit: float,
    total_objective: float,
) -> None:
    iteration_logs.append(
        _IterationLog(
            iteration=iteration,
            residual=residual_norm,
            residual_weighted=residual_norm_weighted,
            relative_residual=rel_residual,
            relative_residual_weighted=rel_residual_weighted,
            residual_max=residual_max,
            meas_norm=meas_norm,
            pred_norm=pred_norm,
            meas_max=meas_max,
            pred_max=pred_max,
            jtr_norm=jtr_norm,
            delta_norm=delta_norm,
            step=optimal_step_size,
            lambda_eff=lambda_eff,
            relative_change=relative_change,
            res_drop=res_drop,
            meas_misfit=meas_misfit,
            prior_misfit=prior_misfit,
            total_objective=total_objective,
        )
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

    meas_vector = _extract_measured_vector(measured_data)

    if len(meas_vector) != reconstructor.n_measurements:
        raise ValueError(
            f"Measurement data length mismatch: {len(meas_vector)} vs {reconstructor.n_measurements}"
        )
    _require_finite("measured_data", meas_vector)

    meas_torch = _to_runtime_tensor(reconstructor, meas_vector)
    reconstructor._measured_vector = meas_vector.copy()
    reconstructor.ensure_regularization_ready()

    meas_norm = torch.norm(meas_torch).item()
    meas_max = torch.max(torch.abs(meas_torch)).item()
    _require_scalar_finite("meas_norm", meas_norm)
    _require_scalar_finite("meas_max", meas_max)
    meas_weighted_norm = None

    sigma_current, initial_conductivity = _init_sigma_function(
        reconstructor, initial_conductivity
    )
    reconstructor._ensure_measurement_weights(sigma_current)

    if reconstructor._meas_weight_sqrt is not None:
        meas_weighted_norm = torch.norm(meas_torch * reconstructor._meas_weight_sqrt).item()
        _require_scalar_finite("meas_weighted_norm", meas_weighted_norm)

    prior_torch = _prepare_prior(reconstructor, prior_data, initial_conductivity)

    residual_history = []
    sigma_change_history = []
    iteration_logs: list[_IterationLog] = []
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
    reconstructor._runtime_tensor_cache = {}

    with reconstructor._progress(total=reconstructor.max_iterations) as pbar:
        for iteration in range(reconstructor.max_iterations):
            sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array", sigma_array, iteration)
            img_current = EITImage(elem_data=sigma_array, fwd_model=reconstructor.fwd_model)
            data_simulated, _ = reconstructor.fwd_model.fwd_solve(img_current)
            _require_finite("data_simulated.meas", data_simulated.meas, iteration)

            lambda_eff = reconstructor.regularization_param
            (
                data_sim_torch,
                residual_torch,
                weighted_residual_torch,
                residual_norm_weighted,
                residual_norm,
                residual_max,
            ) = _compute_residuals(
                reconstructor,
                data_simulated.meas,
                meas_torch,
                iteration,
            )

            sigma_vec_torch = _to_runtime_tensor_cached(
                reconstructor,
                "sigma_vector",
                sigma_array,
            )
            de_current = sigma_vec_torch - prior_torch
            meas_misfit, prior_misfit, total_objective, RtR_de = _compute_objective(
                reconstructor,
                weighted_residual_torch,
                de_current,
                lambda_eff,
                iteration,
            )

            residual_history.append(residual_norm)
            res_drop = None if prev_residual is None else prev_residual - residual_norm

            measurement_jacobian_np = reconstructor.jacobian_calculator.calculate(
                sigma_current,
                method=jacobian_method,
            )
            if reconstructor.negate_jacobian:
                measurement_jacobian_np = -measurement_jacobian_np
            _require_finite("measurement_jacobian_np", measurement_jacobian_np, iteration)
            J_torch = _to_runtime_tensor_cached(
                reconstructor,
                "measurement_jacobian",
                measurement_jacobian_np,
            )
            if reconstructor._meas_weight_sqrt is not None:
                J_weighted = J_torch * reconstructor._meas_weight_sqrt.unsqueeze(1)
            else:
                J_weighted = J_torch

            JTJ = torch.mm(J_weighted.t(), J_weighted)
            JTr = torch.mv(J_weighted.t(), weighted_residual_torch)

            de_torch = de_current
            A, b = _build_linear_system(
                reconstructor,
                JTJ,
                JTr,
                de_torch,
                lambda_eff,
                iteration,
                RtR_de=RtR_de,
            )

            pred_norm = torch.norm(data_sim_torch).item()
            pred_max = torch.max(torch.abs(data_sim_torch)).item()
            jtr_norm = torch.norm(JTr).item()
            rel_residual = residual_norm / (meas_norm + 1e-12)
            rel_residual_weighted = (
                residual_norm_weighted / (meas_weighted_norm + 1e-12)
                if meas_weighted_norm
                else None
            )
            _require_scalar_finite("pred_norm", pred_norm, iteration)
            _require_scalar_finite("pred_max", pred_max, iteration)
            _require_scalar_finite("jtr_norm", jtr_norm, iteration)
            _require_scalar_finite("rel_residual", rel_residual, iteration)
            if rel_residual_weighted is not None:
                _require_scalar_finite("rel_residual_weighted", rel_residual_weighted, iteration)

            delta_sigma_torch, delta_norm = _solve_linear_system(
                reconstructor,
                A,
                b,
                JTJ,
                iteration,
            )

            optimal_step_size = _select_step_size(
                reconstructor,
                iteration,
                sigma_current,
                delta_sigma_torch,
                meas_torch,
                residual_norm_weighted,
                prior_torch,
                lambda_eff,
            )

            needs_snapshot = prev_residual is not None or reconstructor.clip_values is not None
            sigma_old_values = sigma_array.copy() if needs_snapshot else None
            delta_sigma_np = delta_sigma_torch.detach().cpu().numpy()
            sigma_array[:] += optimal_step_size * delta_sigma_np
            _require_finite("sigma_array_updated", sigma_array, iteration)

            if reconstructor.clip_values is not None:
                function_set_array(
                    sigma_current,
                    np.clip(sigma_array, reconstructor.clip_values[0], reconstructor.clip_values[1]),
                )
                sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array_clipped", sigma_array, iteration)

            sigma_new_norm = float(np.linalg.norm(sigma_array))
            if sigma_old_values is None:
                sigma_change = abs(optimal_step_size) * delta_norm
            else:
                sigma_change = float(np.linalg.norm(sigma_array - sigma_old_values))
            relative_change = sigma_change / (sigma_new_norm + 1e-12)
            _require_scalar_finite("sigma_change", sigma_change, iteration)
            _require_scalar_finite("relative_change", relative_change, iteration)

            rolled_back, should_stop, consecutive_rollbacks = _maybe_rollback(
                reconstructor,
                sigma_current,
                sigma_old_values if sigma_old_values is not None else sigma_array,
                residual_norm,
                prev_residual,
                residual_history,
                sigma_change_history,
                consecutive_rollbacks,
                max_consecutive_rollbacks,
            )
            if rolled_back:
                if should_stop:
                    break
                continue

            consecutive_rollbacks = 0
            sigma_change_history.append(relative_change)
            if record_conductivity_history and (iteration + 1) % history_stride == 0:
                conductivity_history.append(function_get_array(sigma_current).copy())

            _record_iteration_log(
                iteration_logs,
                iteration=iteration,
                residual_norm=residual_norm,
                residual_norm_weighted=residual_norm_weighted,
                rel_residual=rel_residual,
                rel_residual_weighted=rel_residual_weighted,
                residual_max=residual_max,
                meas_norm=meas_norm,
                pred_norm=pred_norm,
                meas_max=meas_max,
                pred_max=pred_max,
                jtr_norm=jtr_norm,
                delta_norm=delta_norm,
                optimal_step_size=optimal_step_size,
                lambda_eff=lambda_eff,
                relative_change=relative_change,
                res_drop=res_drop,
                meas_misfit=meas_misfit,
                prior_misfit=prior_misfit,
                total_objective=total_objective,
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
        iteration_logs=[item.to_payload() for item in iteration_logs],
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
