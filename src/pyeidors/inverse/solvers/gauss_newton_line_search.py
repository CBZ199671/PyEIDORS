"""Line-search helpers for Gauss-Newton reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from ...data.difference import project_measurement_vector
from ...data.structures import EITImage
from ...femx import function_get_array
from ...utils.numeric_ops import real_array_if_zero_imaginary


def _max_machine_epsilon_alpha(
    x: np.ndarray,
    dx: np.ndarray,
    *,
    eps_machine: float,
    chunk_size: int = 65_536,
) -> float:
    """Scan the lower alpha guard without materialising ``abs(x) / abs(dx)``."""
    x_arr, dx_arr = np.broadcast_arrays(np.asarray(x), np.asarray(dx))
    x_flat = x_arr.reshape(-1)
    dx_flat = dx_arr.reshape(-1)
    if x_flat.size == 0:
        return 0.0

    block_size = max(1, min(int(chunk_size), x_flat.size))
    x_abs = np.empty(block_size, dtype=np.float64)
    dx_abs = np.empty(block_size, dtype=np.float64)
    alpha_work = np.empty(block_size, dtype=np.float64)
    finite_mask = np.empty(block_size, dtype=bool)
    max_alpha = 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for start in range(0, x_flat.size, block_size):
            stop = min(start + block_size, x_flat.size)
            n_values = stop - start
            x_abs_chunk = x_abs[:n_values]
            dx_abs_chunk = dx_abs[:n_values]
            alpha_chunk = alpha_work[:n_values]
            finite_chunk = finite_mask[:n_values]

            np.abs(x_flat[start:stop], out=x_abs_chunk)
            np.abs(dx_flat[start:stop], out=dx_abs_chunk)
            np.divide(x_abs_chunk, dx_abs_chunk, out=alpha_chunk)
            np.multiply(alpha_chunk, eps_machine, out=alpha_chunk)
            np.isfinite(alpha_chunk, out=finite_chunk)
            np.logical_not(finite_chunk, out=finite_chunk)
            np.copyto(alpha_chunk, 0.0, where=finite_chunk)

            chunk_max = float(np.max(alpha_chunk))
            if chunk_max > max_alpha:
                max_alpha = chunk_max

    return max_alpha


def _min_stable_upper_alpha(
    x: np.ndarray,
    dx: np.ndarray,
    *,
    realmax: float,
    chunk_size: int = 65_536,
) -> float:
    """Scan upper alpha limits without materialising positive/negative arrays."""
    x_arr, dx_arr = np.broadcast_arrays(np.asarray(x), np.asarray(dx))
    x_flat = x_arr.reshape(-1)
    dx_flat = dx_arr.reshape(-1)
    if x_flat.size == 0:
        return np.inf

    block_size = max(1, min(int(chunk_size), x_flat.size))
    alpha_work = np.empty(block_size, dtype=np.float64)
    limit_mask = np.empty(block_size, dtype=bool)
    best_pos = np.inf
    best_neg = np.inf
    upper_limit = float(realmax)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for start in range(0, x_flat.size, block_size):
            stop = min(start + block_size, x_flat.size)
            n_values = stop - start
            x_chunk = x_flat[start:stop]
            dx_chunk = dx_flat[start:stop]
            alpha_chunk = alpha_work[:n_values]
            mask_chunk = limit_mask[:n_values]

            np.subtract(upper_limit, x_chunk, out=alpha_chunk)
            np.divide(alpha_chunk, dx_chunk, out=alpha_chunk)
            np.less_equal(dx_chunk, 0, out=mask_chunk)
            np.copyto(alpha_chunk, np.inf, where=mask_chunk)
            np.isfinite(alpha_chunk, out=mask_chunk)
            np.logical_not(mask_chunk, out=mask_chunk)
            np.copyto(alpha_chunk, np.inf, where=mask_chunk)
            best_pos = min(best_pos, float(np.min(alpha_chunk)))

            np.subtract(-upper_limit, x_chunk, out=alpha_chunk)
            np.divide(alpha_chunk, dx_chunk, out=alpha_chunk)
            np.greater_equal(dx_chunk, 0, out=mask_chunk)
            np.copyto(alpha_chunk, np.inf, where=mask_chunk)
            np.isfinite(alpha_chunk, out=mask_chunk)
            np.logical_not(mask_chunk, out=mask_chunk)
            np.copyto(alpha_chunk, np.inf, where=mask_chunk)
            best_neg = min(best_neg, float(np.min(alpha_chunk)))

    return min(best_pos, best_neg)


def line_search_torch(
    reconstructor,
    sigma_current,
    delta_sigma_torch,
    meas_target_torch,
    current_weighted_residual,
    weight_vector=None,
    prior_torch=None,
    lambda_eff=None,
    retry: int = 0,
) -> float:
    """EIDORS-style line search on full objective (measurement + prior)."""
    delta_sigma_np = delta_sigma_torch.cpu().numpy()
    current_residual = float(current_weighted_residual)
    x = function_get_array(sigma_current).copy()

    if (
        not hasattr(reconstructor, "_line_search_perturb")
        or reconstructor._line_search_perturb is None
    ):
        base_perturb = np.array([0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1])
        reconstructor._line_search_perturb = base_perturb * reconstructor.max_step

    perturb = calc_perturb_limits(reconstructor, x, delta_sigma_np)
    mlist = np.full(len(perturb), np.nan)
    baseline_objective = current_residual**2 * 0.5

    for i, alpha in enumerate(perturb):
        if i == 0:
            mlist[i] = baseline_objective
            continue

        sigma_test_np = x + alpha * delta_sigma_np
        if reconstructor.clip_values is not None and not np.iscomplexobj(sigma_test_np):
            sigma_test_np = np.clip(
                sigma_test_np,
                reconstructor.clip_values[0],
                reconstructor.clip_values[1],
            )

        img_test = EITImage(elem_data=sigma_test_np, fwd_model=reconstructor.fwd_model)
        try:
            data_test, _ = reconstructor.fwd_model.fwd_solve(img_test)
        except Exception:
            mlist[i] = np.inf
            continue

        data_test_values = data_test.meas
        if reconstructor._torch_dtype in {torch.float16, torch.float32, torch.float64}:
            data_test_values = real_array_if_zero_imaginary(
                data_test_values, name="line-search simulated measurements"
            )
        data_test_torch = torch.from_numpy(np.asarray(data_test_values)).to(
            reconstructor.device,
            dtype=reconstructor._torch_dtype,
        )
        data_test_projected = project_measurement_vector(
            data_test_torch.detach().cpu().numpy(),
            measurement_type=getattr(reconstructor, "_measurement_space_type", "real"),
            reference_meas=getattr(reconstructor, "_difference_reference_meas", None),
            difference_mode=getattr(
                reconstructor,
                "_difference_mode_effective",
                reconstructor.difference_mode,
            ),
            difference_orientation=getattr(
                reconstructor,
                "_difference_orientation_effective",
                reconstructor.difference_orientation,
            ),
        )
        if reconstructor._torch_dtype in {torch.float16, torch.float32, torch.float64}:
            data_test_projected = real_array_if_zero_imaginary(
                data_test_projected, name="line-search projected measurements"
            )
        data_test_projected_torch = torch.from_numpy(data_test_projected).to(
            reconstructor.device,
            dtype=reconstructor._torch_dtype,
        )
        dv_torch = data_test_projected_torch - meas_target_torch
        weighted_dv = (
            dv_torch * weight_vector if weight_vector is not None else dv_torch
        )

        if weighted_dv.is_complex():
            meas_misfit = 0.5 * torch.vdot(weighted_dv, weighted_dv).real.item()
        else:
            meas_misfit = 0.5 * torch.dot(weighted_dv, weighted_dv).item()
        prior_misfit = 0.0
        if (
            reconstructor.use_prior_term
            and prior_torch is not None
            and lambda_eff is not None
        ):
            sigma_test_values = sigma_test_np
            if reconstructor._torch_dtype in {
                torch.float16,
                torch.float32,
                torch.float64,
            }:
                sigma_test_values = real_array_if_zero_imaginary(
                    sigma_test_values, name="line-search trial conductivity"
                )
            sigma_test_torch = torch.from_numpy(np.asarray(sigma_test_values)).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            de_torch = sigma_test_torch - prior_torch
            RtR_de = torch.mv(reconstructor.R_torch, de_torch)
            if de_torch.is_complex() or RtR_de.is_complex():
                prior_misfit = (
                    0.5 * lambda_eff * torch.vdot(de_torch, RtR_de).real.item()
                )
            else:
                prior_misfit = 0.5 * lambda_eff * torch.dot(de_torch, RtR_de).item()

        total_objective = meas_misfit + prior_misfit
        if np.isnan(total_objective) or np.isinf(total_objective):
            mlist[i] = np.inf
        else:
            mlist[i] = total_objective

        if baseline_objective > 0 and mlist[i] / baseline_objective > 1e10:
            break

    best_idx, valid_count, last_valid_idx = _finite_metric_summary(mlist)
    if best_idx < 0:
        chosen_step = 0.0
    else:
        chosen_step = float(perturb[best_idx])

    update_perturb_eidors_style(
        reconstructor,
        chosen_step,
        perturb,
        mlist,
        None,
        valid_count=valid_count,
        last_valid_idx=last_valid_idx,
    )

    if chosen_step == 0 and retry < 5:
        return line_search_torch(
            reconstructor,
            sigma_current,
            delta_sigma_torch,
            meas_target_torch,
            current_weighted_residual,
            weight_vector,
            prior_torch,
            lambda_eff,
            retry=retry + 1,
        )
    return chosen_step


def calc_perturb_limits(reconstructor, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Compute numerically stable line-search samples for ``alpha``."""
    perturb = reconstructor._line_search_perturb.copy()
    if perturb[0] != 0:
        expanded = np.empty(perturb.size + 1, dtype=perturb.dtype)
        expanded[0] = 0.0
        expanded[1:] = perturb
        perturb = expanded

    if np.iscomplexobj(x) or np.iscomplexobj(dx):
        return np.asarray(np.clip(perturb, 0.0, 1.0), dtype=np.float64)

    eps_machine = np.finfo(np.float64).eps
    realmax = np.finfo(np.float64).max / 2

    max_alpha = _min_stable_upper_alpha(x, dx, realmax=realmax)
    min_alpha = _max_machine_epsilon_alpha(x, dx, eps_machine=eps_machine)

    max_alpha = min(max_alpha, 1.0)

    if perturb[-1] > max_alpha or (len(perturb) > 1 and perturb[1] < min_alpha):
        p_nonzero = perturb[perturb > eps_machine]
        if len(p_nonzero) == 0:
            return np.array([0, max_alpha / 4, max_alpha / 2, max_alpha])

        log_p = np.log10(p_nonzero)
        log_max = np.log10(max_alpha) if max_alpha > eps_machine else -10
        log_min = np.log10(min_alpha) if min_alpha > eps_machine else -100

        p_range = log_p[-1] - log_p[0] if len(log_p) > 1 else 1
        target_range = log_max - log_min

        if p_range > target_range and target_range > 0:
            log_p = log_p * (target_range / p_range)

        if log_p[-1] > log_max:
            log_p = log_p - (log_p[-1] - log_max)
        elif log_p[0] < log_min:
            log_p = log_p + (log_min - log_p[0])

        log_values = 10**log_p
        perturb = np.empty(log_values.size + 1, dtype=np.float64)
        perturb[0] = 0.0
        perturb[1:] = log_values

    return perturb


def _finite_metric_summary(values: np.ndarray) -> tuple[int, int, int]:
    """Return best finite index, finite count, and last finite index."""
    best_idx = -1
    best_value = np.inf
    valid_count = 0
    last_valid_idx = -1
    for idx, raw_value in enumerate(np.asarray(values)):
        value = float(raw_value)
        if not np.isfinite(value):
            continue
        valid_count += 1
        last_valid_idx = idx
        if value < best_value:
            best_value = value
            best_idx = idx
    return best_idx, valid_count, last_valid_idx


def _valid_metric_count_last(
    values: np.ndarray,
    valid_idx: np.ndarray | None,
    *,
    valid_count: int | None,
    last_valid_idx: int | None,
) -> tuple[int, int]:
    if valid_count is not None and last_valid_idx is not None:
        return int(valid_count), int(last_valid_idx)
    if valid_idx is None:
        _, scanned_count, scanned_last = _finite_metric_summary(values)
        return scanned_count, scanned_last
    if len(valid_idx) == 0:
        return 0, -1
    return len(valid_idx), int(valid_idx[-1])


def _valid_metrics_all_similar(
    values: np.ndarray,
    valid_idx: np.ndarray | None,
    baseline_objective: float,
    dtol: float,
    *,
    valid_count: int,
) -> bool:
    if valid_count <= 0 or baseline_objective <= 0:
        return False
    threshold = -10 * dtol
    if valid_idx is None:
        iterator = enumerate(np.asarray(values))
    else:
        iterator = ((int(idx), values[int(idx)]) for idx in valid_idx)
    for _, raw_value in iterator:
        value = float(raw_value)
        if not np.isfinite(value) or value / baseline_objective - 1 <= threshold:
            return False
    return True


def update_perturb_eidors_style(
    reconstructor,
    chosen_step: float,
    perturb: np.ndarray,
    mlist: np.ndarray,
    valid_idx: np.ndarray | None,
    *,
    valid_count: int | None = None,
    last_valid_idx: int | None = None,
) -> None:
    """Adapt line-search sample schedule using EIDORS heuristic."""
    dtol = reconstructor.convergence_tol
    goodi_count, goodi_last = _valid_metric_count_last(
        mlist,
        valid_idx,
        valid_count=valid_count,
        last_valid_idx=last_valid_idx,
    )

    if chosen_step == 0:
        if goodi_count > 1 and mlist[0] * 1.05 < mlist[goodi_last]:
            reconstructor._line_search_perturb = reconstructor._line_search_perturb / 10
        elif perturb[-1] > 1.0 - 1e-9:
            pass
        elif perturb[-1] * 10 > 1.0 - 1e-9:
            reconstructor._line_search_perturb = (
                reconstructor._line_search_perturb / perturb[-1]
            )
        else:
            reconstructor._line_search_perturb = reconstructor._line_search_perturb * 10
    else:
        baseline_objective = float(mlist[0])
        all_similar = _valid_metrics_all_similar(
            mlist,
            valid_idx,
            baseline_objective,
            dtol,
            valid_count=goodi_count,
        )
        if all_similar and perturb[-1] * 10 < 1.0 + 1e-9:
            reconstructor._line_search_perturb = reconstructor._line_search_perturb * 10
        elif chosen_step > 0 and perturb[-1] > 0:
            scale = (chosen_step / perturb[-1]) * 2
            new_perturb = reconstructor._line_search_perturb * scale
            if new_perturb[-1] > 1.0 - 1e-9:
                new_perturb = new_perturb / new_perturb[-1]
            reconstructor._line_search_perturb = new_perturb

    jiggle = np.exp(np.random.randn(len(reconstructor._line_search_perturb)) * 0.01)
    reconstructor._line_search_perturb = reconstructor._line_search_perturb * jiggle
    if reconstructor._line_search_perturb[-1] > 1.0 - 1e-9:
        reconstructor._line_search_perturb = (
            reconstructor._line_search_perturb / reconstructor._line_search_perturb[-1]
        )
