"""Line-search helpers for Gauss-Newton reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from ...data.difference import project_measurement_vector
from ...data.structures import EITImage
from ...femx import function_get_array


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
        if reconstructor.clip_values is not None:
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

        data_test_torch = torch.from_numpy(data_test.meas).to(
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
        data_test_projected_torch = torch.from_numpy(data_test_projected).to(
            reconstructor.device,
            dtype=reconstructor._torch_dtype,
        )
        dv_torch = data_test_projected_torch - meas_target_torch
        weighted_dv = (
            dv_torch * weight_vector if weight_vector is not None else dv_torch
        )

        meas_misfit = 0.5 * torch.dot(weighted_dv, weighted_dv).item()
        prior_misfit = 0.0
        if (
            reconstructor.use_prior_term
            and prior_torch is not None
            and lambda_eff is not None
        ):
            sigma_test_torch = torch.from_numpy(sigma_test_np).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            de_torch = sigma_test_torch - prior_torch
            RtR_de = torch.mv(reconstructor.R_torch, de_torch)
            prior_misfit = 0.5 * lambda_eff * torch.dot(de_torch, RtR_de).item()

        total_objective = meas_misfit + prior_misfit
        if np.isnan(total_objective) or np.isinf(total_objective):
            mlist[i] = np.inf
        else:
            mlist[i] = total_objective

        if baseline_objective > 0 and mlist[i] / baseline_objective > 1e10:
            break

    valid_idx = np.where(np.isfinite(mlist))[0]
    if len(valid_idx) == 0:
        chosen_step = 0.0
    else:
        best_idx = valid_idx[np.argmin(mlist[valid_idx])]
        chosen_step = float(perturb[best_idx])

    update_perturb_eidors_style(reconstructor, chosen_step, perturb, mlist, valid_idx)

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
        perturb = np.concatenate([[0], perturb])

    eps_machine = np.finfo(np.float64).eps
    realmax = np.finfo(np.float64).max / 2

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        au_pos = (realmax - x) / dx
        au_pos[dx <= 0] = np.inf
        au_pos[~np.isfinite(au_pos)] = np.inf

        au_neg = (-realmax - x) / dx
        au_neg[dx >= 0] = np.inf
        au_neg[~np.isfinite(au_neg)] = np.inf

        max_alpha = min(np.min(au_pos), np.min(au_neg))

    with np.errstate(divide="ignore", invalid="ignore"):
        al = eps_machine * np.abs(x) / np.abs(dx)
        al[~np.isfinite(al)] = 0
        min_alpha = np.max(al) if len(al) > 0 else 0

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

        perturb = np.concatenate([[0], 10**log_p])

    return perturb


def update_perturb_eidors_style(
    reconstructor,
    chosen_step: float,
    perturb: np.ndarray,
    mlist: np.ndarray,
    valid_idx: np.ndarray,
) -> None:
    """Adapt line-search sample schedule using EIDORS heuristic."""
    dtol = reconstructor.convergence_tol
    goodi = valid_idx

    if chosen_step == 0:
        if len(goodi) > 1 and mlist[0] * 1.05 < mlist[goodi[-1]]:
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
        all_similar = (
            len(goodi) > 0
            and baseline_objective > 0
            and np.all(mlist[goodi] / baseline_objective - 1 > -10 * dtol)
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
