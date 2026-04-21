"""Matrix-free IRGNM / Levenberg-Marquardt step helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

from .gauss_newton_runtime import _solve_linear_system_fast


@dataclass(frozen=True)
class MatrixFreeGNStepResult:
    """One matrix-free GN-family step plus solver metadata."""

    delta: np.ndarray
    delta_norm: float
    jtr_norm: float
    metadata: MappingProxyType

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.delta, dtype=dtype)


def solve_matrix_free_gn_step(
    jacobian: Any,
    residual: Any,
    *,
    current: Any | None = None,
    prior: Any | None = None,
    regularization: Any | None = None,
    alpha: float = 1.0e-2,
    damping: float = 0.0,
    method: str = "irgnm",
    measurement_weights: Any | None = None,
    matrix_free_ksp_backend: str = "scipy",
    preconditioner: str = "diag",
    fast_linear_path: str = "pcg",
    performance_mode: str = "aggressive",
    iteration: int = 0,
) -> MatrixFreeGNStepResult:
    """Solve one matrix-free IRGNM / LM correction step.

    The wrapper builds the normal-equation action
    ``J.T W J + alpha R + damping I`` but delegates the actual Krylov solve to
    the existing fast GN backend. ``measurement_weights`` is the diagonal
    precision ``W``; full covariance is intentionally left to a later phase.
    """

    resolved_method = str(method).strip().lower()
    if resolved_method not in {"irgnm", "lm", "levenberg-marquardt"}:
        raise ValueError("method must be one of: 'irgnm', 'lm'.")
    if resolved_method == "levenberg-marquardt":
        resolved_method = "lm"
    alpha_value = _nonnegative_float(alpha, name="alpha")
    damping_value = _nonnegative_float(damping, name="damping")

    residual_vec = _as_vector(residual, name="residual")
    n_measurements = residual_vec.size
    n_parameters = _infer_n_parameters(jacobian)
    current_vec = _optional_vector(current, n_parameters=n_parameters, name="current")
    prior_vec = _optional_vector(prior, n_parameters=n_parameters, name="prior")
    current_minus_prior = current_vec - prior_vec

    reg_base = _as_regularization(regularization, n_parameters=n_parameters)
    reg_eff = _effective_regularization(
        reg_base,
        alpha=alpha_value,
        damping=damping_value if resolved_method == "lm" else 0.0,
    )
    rhs_current = _effective_current_for_existing_solver(
        reg_base,
        reg_eff,
        current_minus_prior,
        alpha=alpha_value,
    )

    sqrt_weights, weight_kind = _sqrt_measurement_weights(
        measurement_weights,
        n_measurements=n_measurements,
    )
    weighted_residual = (
        residual_vec if sqrt_weights is None else sqrt_weights * residual_vec
    )

    reconstructor = SimpleNamespace(
        R_matrix=reg_eff,
        R_diag=np.asarray(reg_eff.diagonal(), dtype=np.float64),
        use_prior_term=True,
        performance_mode=str(performance_mode),
        linear_solver="auto",
        preconditioner=str(preconditioner),
        fast_linear_path=str(fast_linear_path),
        matrix_free_ksp_backend=str(matrix_free_ksp_backend),
        cholmod_max_n=50000,
        cholmod_max_memory_gib=4.0,
    )

    delta, delta_norm, jtr_norm = _solve_linear_system_fast(
        reconstructor,
        J_weighted_np=jacobian,
        weighted_residual_np=weighted_residual,
        de_current_np=rhs_current,
        lambda_eff=1.0,
        iteration=int(iteration),
        measurement_weight_np=sqrt_weights,
    )
    metadata = dict(getattr(reconstructor, "_last_fast_linear_meta", {}))
    metadata.update(
        {
            "gn_family_method": resolved_method,
            "alpha": float(alpha_value),
            "damping": float(damping_value if resolved_method == "lm" else 0.0),
            "measurement_weight_kind": weight_kind,
            "regularization_shape": [int(n_parameters), int(n_parameters)],
        }
    )
    return MatrixFreeGNStepResult(
        delta=np.asarray(delta, dtype=np.float64).reshape(-1),
        delta_norm=float(delta_norm),
        jtr_norm=float(jtr_norm),
        metadata=MappingProxyType(metadata),
    )


def _nonnegative_float(value: float, *, name: str) -> float:
    out = float(value)
    if out < 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


def _as_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _optional_vector(values: Any | None, *, n_parameters: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(int(n_parameters), dtype=np.float64)
    vector = _as_vector(values, name=name)
    if vector.size != int(n_parameters):
        raise ValueError(f"{name} length {vector.size} does not match {n_parameters}.")
    return vector


def _infer_n_parameters(jacobian: Any) -> int:
    shape = getattr(jacobian, "shape", None)
    if shape is not None and len(shape) == 2:
        n_parameters = int(shape[1])
        if n_parameters > 0:
            return n_parameters
    n_parameters = getattr(jacobian, "n_coarse_cells", None)
    if n_parameters is not None and int(n_parameters) > 0:
        return int(n_parameters)
    array = np.asarray(jacobian)
    if array.ndim == 2 and array.shape[1] > 0:
        return int(array.shape[1])
    raise ValueError("Cannot infer matrix-free parameter dimension from jacobian.")


def _as_regularization(
    regularization: Any | None, *, n_parameters: int
) -> sparse.csr_matrix:
    n = int(n_parameters)
    if regularization is None:
        return sparse.eye(n, format="csr", dtype=np.float64)
    if sparse.issparse(regularization):
        matrix = regularization.tocsr().astype(np.float64)
    else:
        array = np.asarray(regularization, dtype=np.float64)
        if array.ndim == 1:
            if array.size != n:
                raise ValueError("regularization diagonal length mismatch.")
            matrix = sparse.diags(array, 0, shape=(n, n), format="csr")
        else:
            matrix = sparse.csr_matrix(array, dtype=np.float64)
    if matrix.shape != (n, n):
        raise ValueError(f"regularization shape {matrix.shape} != {(n, n)}.")
    if not np.isfinite(matrix.data).all():
        raise FloatingPointError("regularization contains non-finite values.")
    return matrix


def _effective_regularization(
    base: sparse.csr_matrix,
    *,
    alpha: float,
    damping: float,
) -> sparse.csr_matrix:
    n = int(base.shape[0])
    matrix = (float(alpha) * base).tocsr()
    if damping:
        matrix = matrix + sparse.eye(n, format="csr", dtype=np.float64) * float(damping)
    if matrix.nnz == 0:
        matrix = sparse.eye(n, format="csr", dtype=np.float64) * 1.0e-12
    return matrix.tocsr()


def _effective_current_for_existing_solver(
    base: sparse.csr_matrix,
    effective: sparse.csr_matrix,
    current_minus_prior: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Map ``alpha R m`` onto existing solver's ``R_eff @ de_current`` slot."""

    target = np.asarray(
        float(alpha) * (base @ current_minus_prior), dtype=np.float64
    ).reshape(-1)
    if np.linalg.norm(target) <= np.finfo(np.float64).eps:
        return np.zeros_like(target)
    try:
        return np.asarray(
            sparse.linalg.spsolve(effective.tocsc(), target), dtype=np.float64
        )
    except Exception:
        return np.asarray(
            np.linalg.pinv(effective.toarray()) @ target, dtype=np.float64
        )


def _sqrt_measurement_weights(
    measurement_weights: Any | None,
    *,
    n_measurements: int,
) -> tuple[np.ndarray | None, str]:
    if measurement_weights is None:
        return None, "identity"
    weights = np.asarray(measurement_weights, dtype=np.float64)
    if weights.ndim == 1:
        diag = weights.reshape(-1)
        kind = "diagonal"
    elif weights.ndim == 2 and weights.shape == (n_measurements, n_measurements):
        if not np.allclose(weights, np.diag(np.diag(weights)), rtol=1e-12, atol=1e-14):
            raise NotImplementedError(
                "Full measurement covariance is phase-2+; pass diagonal precision weights."
            )
        diag = np.diag(weights)
        kind = "diagonal-matrix"
    else:
        raise ValueError(
            "measurement_weights must be a length-n vector or n-by-n diagonal matrix."
        )
    if diag.size != int(n_measurements):
        raise ValueError(
            f"measurement_weights length {diag.size} does not match {n_measurements}."
        )
    if not np.isfinite(diag).all():
        raise FloatingPointError("measurement_weights contain non-finite values.")
    if np.any(diag < 0.0):
        raise ValueError("measurement_weights must be non-negative precision weights.")
    return np.sqrt(diag).astype(np.float64), kind


__all__ = ["MatrixFreeGNStepResult", "solve_matrix_free_gn_step"]
