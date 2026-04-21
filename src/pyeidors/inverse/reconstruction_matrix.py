"""Online reconstruction-matrix helpers for difference EIT."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data.difference import normalize_time_difference
from pyeidors.utils.numeric_ops import safe_dot


@dataclass(frozen=True)
class OneStepRMResult:
    """One-step reconstruction matrix plus build metadata."""

    rm: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, int]:
        return self.rm.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.rm, dtype=dtype)


def _as_measurement_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector measurement array.")
    vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _as_jacobian(jacobian: Any) -> np.ndarray:
    if sparse.issparse(jacobian):
        matrix = np.asarray(jacobian.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("J must be a 2D measurement-by-parameter matrix.")
    if 0 in matrix.shape:
        raise ValueError("J must be non-empty.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("J contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _as_regularization_matrix(regularization: Any, *, n_parameters: int) -> np.ndarray:
    if regularization is None:
        return np.eye(n_parameters, dtype=np.float64)
    if sparse.issparse(regularization):
        matrix = np.asarray(regularization.toarray(), dtype=np.float64)
    else:
        array = np.asarray(regularization, dtype=np.float64)
        matrix = np.diag(array) if array.ndim == 1 else array
    if matrix.shape != (n_parameters, n_parameters):
        raise ValueError(
            "regularization must have shape "
            f"{(n_parameters, n_parameters)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("regularization contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _noser_regularization(
    jacobian: np.ndarray,
    *,
    floor: float,
    exponent: float,
) -> np.ndarray:
    if floor < 0.0:
        raise ValueError("noser_floor must be non-negative.")
    if exponent <= 0.0:
        raise ValueError("noser_exponent must be positive.")
    diag = np.sum(jacobian * jacobian, axis=0)
    diag = np.maximum(diag, float(floor))
    if exponent != 1.0:
        diag = diag ** float(exponent)
    return np.diag(diag)


def build_one_step_rm(
    J: Any,
    regularization: Any = None,
    lambda_: float = 1e-2,
    *,
    mode: str = "tikhonov",
    form: str = "param",
    noser_floor: float = 1e-12,
    noser_exponent: float = 1.0,
    return_metadata: bool = False,
) -> np.ndarray | OneStepRMResult:
    """Build a one-step GN/NOSER/Laplace reconstruction matrix.

    T16 deliberately implements the parameter-space form
    ``RM = (J.T @ J + lambda_**2 R)^-1 @ J.T``. The measurement-space
    ``P J.T (J P J.T + lambda_**2 Rn)^-1`` path is T17 and therefore
    raises until that task lands.
    """

    resolved_form = str(form).strip().lower()
    if resolved_form != "param":
        raise NotImplementedError(
            "build_one_step_rm(form='measurement') is reserved for T17."
        )
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"tikhonov", "noser", "laplace"}:
        raise ValueError("mode must be one of: 'tikhonov', 'noser', 'laplace'.")
    lam = float(lambda_)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda_ must be finite and non-negative.")

    jac = _as_jacobian(J)
    _, n_parameters = jac.shape
    if resolved_mode == "noser":
        reg = _noser_regularization(
            jac,
            floor=float(noser_floor),
            exponent=float(noser_exponent),
        )
        regularization_source = "diag_jtj"
    elif resolved_mode == "laplace":
        if regularization is None:
            raise ValueError(
                "mode='laplace' requires a graph-Laplacian regularization."
            )
        reg = _as_regularization_matrix(regularization, n_parameters=n_parameters)
        regularization_source = "provided_laplace"
    else:
        reg = _as_regularization_matrix(regularization, n_parameters=n_parameters)
        regularization_source = "identity" if regularization is None else "provided"

    lhs = np.asarray(jac.T @ jac + (lam * lam) * reg, dtype=np.float64)
    rhs = jac.T
    try:
        rm = np.linalg.solve(lhs, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        rm = np.linalg.pinv(lhs) @ rhs
        solver = "pinv"
    rm = np.asarray(rm, dtype=np.float64)
    if not np.isfinite(rm).all():
        raise FloatingPointError("one-step RM contains non-finite values.")
    try:
        condition_estimate = float(np.linalg.cond(lhs))
    except np.linalg.LinAlgError:
        condition_estimate = float("inf")

    metadata = MappingProxyType(
        {
            "mode": resolved_mode,
            "form": resolved_form,
            "lambda": lam,
            "n_measurements": int(jac.shape[0]),
            "n_parameters": int(jac.shape[1]),
            "regularization_source": regularization_source,
            "regularization_nnz": int(np.count_nonzero(reg)),
            "condition_estimate": condition_estimate,
            "solver": solver,
            "rm_shape": tuple(int(v) for v in rm.shape),
        }
    )
    if return_metadata:
        return OneStepRMResult(rm=rm, metadata=metadata)
    return rm


def _matvec(rm: Any, vector: np.ndarray) -> np.ndarray:
    if sparse.issparse(rm):
        matrix = rm.tocsr()
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(matrix @ vector, dtype=np.float64)
    else:
        matrix = np.asarray(rm, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(
            safe_dot(matrix, vector, "reconstruction_matrix.apply"), dtype=np.float64
        )
    if not np.isfinite(out).all():
        raise FloatingPointError("RM application produced non-finite values.")
    return out.reshape(-1)


def reconstruct_difference(
    rm: Any,
    dv,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
) -> np.ndarray:
    """Apply a precomputed reconstruction matrix to one difference frame.

    If ``normalize`` is true and ``v_ref`` is provided, ``dv`` is interpreted
    as target voltages ``v_t`` and first converted with
    :func:`normalize_time_difference`. Otherwise ``dv`` is treated as an
    already-projected measurement vector. The hot path is deliberately just
    ``RM @ dv_projected``; RM construction belongs to later T16/T17 tasks.
    """

    if normalize and v_ref is not None:
        measurement = normalize_time_difference(dv, v_ref, floor=floor)
    else:
        measurement = _as_measurement_vector(dv, name="dv")
    return _matvec(rm, measurement)


__all__ = ["OneStepRMResult", "build_one_step_rm", "reconstruct_difference"]
