"""Linearized optimizer kernels for sparse Bayesian reconstruction."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...utils.numeric_ops import safe_dot
from .sparse_projection import estimate_lipschitz_constant


def _resolve_gpu_context(config):
    if not config.use_gpu:
        return None
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None

    gpu_dtype = str(config.gpu_dtype).lower()
    if gpu_dtype == "float64":
        dtype = torch.float64
    elif gpu_dtype in {"float16", "half"}:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return torch, torch.device("cuda"), dtype


def solve_fista(
    linear_matrix: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
    warm_start: Optional[np.ndarray],
    config,
) -> np.ndarray:
    """Solve l1-regularized least squares with FISTA."""
    A = linear_matrix / max(noise_sigma, 1e-9)
    b = data_vector / max(noise_sigma, 1e-9)
    lambda_reg = 1.0 / max(prior_scale, 1e-12)

    n = A.shape[1]
    x = warm_start.copy() if warm_start is not None else np.zeros(n, dtype=float)
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz_constant(A)

    gpu_ctx = _resolve_gpu_context(config)
    if gpu_ctx is not None:
        torch, device, dtype = gpu_ctx
        compute_dtype = torch.float32 if dtype == torch.float16 else dtype
        A_t = torch.as_tensor(A, device=device, dtype=compute_dtype)
        b_t = torch.as_tensor(b, device=device, dtype=compute_dtype)
        x_t = torch.as_tensor(x, device=device, dtype=compute_dtype)
        y_t = x_t.clone()
        t_scalar = torch.tensor(1.0, device=device, dtype=compute_dtype)
        L_t = torch.tensor(L, device=device, dtype=compute_dtype)
        lam_over_L = torch.tensor(lambda_reg, device=device, dtype=compute_dtype) / L_t

        for _ in range(config.linear_max_iterations):
            grad = torch.matmul(A_t.T, torch.matmul(A_t, y_t) - b_t)
            z = y_t - grad / L_t
            x_new = torch.sign(z) * torch.clamp(torch.abs(z) - lam_over_L, min=0.0)

            if torch.norm(x_new - x_t) <= config.linear_tolerance * (torch.norm(x_t) + 1e-12):
                x_t = x_new
                break

            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_scalar * t_scalar)) / 2.0
            y_t = x_new + ((t_scalar - 1.0) / t_new) * (x_new - x_t)
            x_t, t_scalar = x_new, t_new

        return x_t.detach().cpu().double().numpy()

    for _ in range(config.linear_max_iterations):
        residual = safe_dot(A, y, "solve_fista.residual") - b
        grad = safe_dot(A.T, residual, "solve_fista.gradient")
        z = y - grad / L
        x_new = np.sign(z) * np.maximum(np.abs(z) - lambda_reg / L, 0.0)

        if np.linalg.norm(x_new - x) <= config.linear_tolerance * (np.linalg.norm(x) + 1e-12):
            x = x_new
            break

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

    return x


def solve_irls(
    linear_matrix: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
    warm_start: Optional[np.ndarray],
    config,
) -> np.ndarray:
    """Solve smoothed-l1 least squares with IRLS."""
    A = linear_matrix / max(noise_sigma, 1e-9)
    b = data_vector / max(noise_sigma, 1e-9)
    lambda_reg = 1.0 / max(prior_scale, 1e-12)

    n = A.shape[1]
    x = warm_start.copy() if warm_start is not None else np.zeros(n, dtype=float)

    gpu_ctx = _resolve_gpu_context(config)
    if gpu_ctx is not None:
        torch, device, dtype = gpu_ctx
        compute_dtype = torch.float32 if dtype == torch.float16 else dtype
        A_t = torch.as_tensor(A, device=device, dtype=compute_dtype)
        b_t = torch.as_tensor(b, device=device, dtype=compute_dtype)
        x_t = torch.as_tensor(x, device=device, dtype=compute_dtype)

        AtA = torch.matmul(A_t.T, A_t)
        Atb = torch.matmul(A_t.T, b_t)

        for _ in range(config.linear_max_iterations):
            weights = torch.rsqrt(x_t * x_t + config.smoothing_beta)
            M = AtA.clone()
            M.diagonal().add_(lambda_reg * weights)
            rhs = Atb
            try:
                x_new = torch.linalg.solve(M, rhs)
            except RuntimeError:
                x_new = torch.linalg.lstsq(M, rhs).solution

            if torch.norm(x_new - x_t) <= config.linear_tolerance * (torch.norm(x_t) + 1e-12):
                x_t = x_new
                break
            x_t = x_new

        return x_t.detach().cpu().double().numpy()

    ata = safe_dot(A.T, A, "solve_irls.ata")
    atb = safe_dot(A.T, b, "solve_irls.atb")
    for _ in range(config.linear_max_iterations):
        weights = 1.0 / np.sqrt(x * x + config.smoothing_beta)
        M = ata + lambda_reg * np.diag(weights)
        rhs = atb
        try:
            x_new = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            x_new = np.linalg.lstsq(M, rhs, rcond=None)[0]

        if np.linalg.norm(x_new - x) <= config.linear_tolerance * (np.linalg.norm(x) + 1e-12):
            x = x_new
            break
        x = x_new

    return x
