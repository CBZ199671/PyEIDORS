"""Online reconstruction-matrix helpers for difference EIT."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
)
from pyeidors.data.difference import normalize_time_difference
from pyeidors.perf.gpu_kernels import RMMatmulResult, rm_matmul
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


def _as_measurement_frames(values: Any, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        frames = array.reshape(1, -1)
        was_vector = True
    elif array.ndim == 2:
        frames = array
        was_vector = False
    else:
        raise ValueError(f"{name} must be a 1D vector or 2D frame batch.")
    if frames.shape[0] == 0 or frames.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(frames).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(frames, dtype=np.float64), was_vector


def _reference_frames(
    reference: Any, *, n_frames: int, n_measurements: int
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim == 1:
        if ref.size != n_measurements:
            raise ValueError(
                f"v_ref length {ref.size} does not match {n_measurements} measurements."
            )
        return np.broadcast_to(ref.reshape(1, -1), (n_frames, n_measurements)).copy()
    if ref.ndim == 2:
        if ref.shape != (n_frames, n_measurements):
            raise ValueError(
                f"v_ref shape {ref.shape} does not match {(n_frames, n_measurements)}."
            )
        return np.ascontiguousarray(ref, dtype=np.float64)
    raise ValueError("v_ref must be a 1D reference vector or 2D frame batch.")


def _normalize_time_difference_frames(
    targets: np.ndarray,
    reference: Any,
    *,
    floor: float | None,
) -> np.ndarray:
    refs = _reference_frames(
        reference,
        n_frames=targets.shape[0],
        n_measurements=targets.shape[1],
    )
    rows = [
        normalize_time_difference(targets[idx], refs[idx], floor=floor)
        for idx in range(targets.shape[0])
    ]
    return np.vstack(rows)


def _apply_measurement_contract_to_frames(
    frames: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> np.ndarray:
    rows = [
        apply_measurement_contract_to_vector(
            frames[idx],
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
        )[0]
        for idx in range(frames.shape[0])
    ]
    return np.vstack(rows)


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


def _as_measurement_regularization(
    measurement_regularization: Any,
    *,
    n_measurements: int,
) -> tuple[np.ndarray, str]:
    if measurement_regularization is None:
        return np.eye(n_measurements, dtype=np.float64), "identity"
    if sparse.issparse(measurement_regularization):
        matrix = np.asarray(measurement_regularization.toarray(), dtype=np.float64)
    else:
        array = np.asarray(measurement_regularization, dtype=np.float64)
        matrix = np.diag(array) if array.ndim == 1 else array
    if matrix.shape != (n_measurements, n_measurements):
        raise ValueError(
            "measurement_regularization must have shape "
            f"{(n_measurements, n_measurements)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError(
            "measurement_regularization contains non-finite values."
        )
    return np.ascontiguousarray(matrix, dtype=np.float64), "provided"


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


def _regularization_for_mode(
    jacobian: np.ndarray,
    regularization: Any,
    *,
    mode: str,
    noser_floor: float,
    noser_exponent: float,
) -> tuple[np.ndarray, str]:
    n_parameters = int(jacobian.shape[1])
    if mode == "noser":
        return (
            _noser_regularization(
                jacobian,
                floor=float(noser_floor),
                exponent=float(noser_exponent),
            ),
            "diag_jtj",
        )
    if mode == "laplace":
        if regularization is None:
            raise ValueError(
                "mode='laplace' requires a graph-Laplacian regularization."
            )
        return (
            _as_regularization_matrix(regularization, n_parameters=n_parameters),
            "provided_laplace",
        )
    return (
        _as_regularization_matrix(regularization, n_parameters=n_parameters),
        "identity" if regularization is None else "provided",
    )


def _solve_or_pinv(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        return np.linalg.solve(lhs, rhs), "solve"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs, "pinv"


def build_one_step_rm(
    J: Any,
    regularization: Any = None,
    lambda_: float = 1e-2,
    *,
    mode: str = "tikhonov",
    form: str = "param",
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    measurement_regularization: Any = None,
    noser_floor: float = 1e-12,
    noser_exponent: float = 1.0,
    return_metadata: bool = False,
) -> np.ndarray | OneStepRMResult:
    """Build a one-step GN/NOSER/Laplace reconstruction matrix.

    ``form="param"`` uses ``RM = (J.T @ J + lambda_**2 R)^-1 @ J.T``.
    ``form="measurement"`` uses
    ``RM = P J.T (J P J.T + lambda_**2 Rn)^-1`` with ``P≈R^-1`` and
    identity ``Rn`` by default.

    ``channel_mask`` uses the data-channel contract where ``True`` marks a
    bad channel. ``measurement_weights`` is the symmetric precision matrix
    ``W`` from ``J.T @ W @ J``; diagonal vectors are accepted. The returned
    RM expects online residuals passed through the same contract.
    """

    resolved_form = str(form).strip().lower()
    if resolved_form not in {"param", "measurement"}:
        raise ValueError("form must be one of: 'param', 'measurement'.")
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"tikhonov", "noser", "laplace"}:
        raise ValueError("mode must be one of: 'tikhonov', 'noser', 'laplace'.")
    lam = float(lambda_)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda_ must be finite and non-negative.")

    jac_raw = _as_jacobian(J)
    jac, measurement_contract = apply_measurement_contract_to_jacobian(
        jac_raw,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    reg, regularization_source = _regularization_for_mode(
        jac,
        regularization,
        mode=resolved_mode,
        noser_floor=float(noser_floor),
        noser_exponent=float(noser_exponent),
    )

    if resolved_form == "measurement":
        rn, rn_source = _as_measurement_regularization(
            measurement_regularization,
            n_measurements=int(jac.shape[0]),
        )
        p_jt, prior_inverse_solver = _solve_or_pinv(reg, jac.T)
        lhs = np.asarray(jac @ p_jt + (lam * lam) * rn, dtype=np.float64)
        rm_t, solver = _solve_or_pinv(lhs.T, p_jt.T)
        rm = rm_t.T
        inversion_dimension = "measurement"
    else:
        rn_source = "unused"
        prior_inverse_solver = "unused"
        lhs = np.asarray(jac.T @ jac + (lam * lam) * reg, dtype=np.float64)
        rm, solver = _solve_or_pinv(lhs, jac.T)
        inversion_dimension = "parameter"
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
            "bad_channel_count": int(measurement_contract.bad_channel_count),
            "measurement_weight_kind": measurement_contract.weight_kind,
            "expects_measurement_contract": True,
            "inversion_dimension": inversion_dimension,
            "regularization_source": regularization_source,
            "regularization_nnz": int(np.count_nonzero(reg)),
            "measurement_regularization_source": rn_source,
            "condition_estimate": condition_estimate,
            "solver": solver,
            "prior_inverse_solver": prior_inverse_solver,
            "system_shape": tuple(int(v) for v in lhs.shape),
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
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    device: str = "cpu",
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
    measurement, _ = apply_measurement_contract_to_vector(
        measurement,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    return np.asarray(rm_matmul(rm, measurement, device=device), dtype=np.float64)


def reconstruct_difference_batch(
    rm: Any,
    frames,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    device: str = "auto",
    return_metadata: bool = False,
) -> np.ndarray | RMMatmulResult:
    """Apply a precomputed RM to one or more online difference frames."""

    frame_batch, was_vector = _as_measurement_frames(frames, name="frames")
    if normalize and v_ref is not None:
        measurement_batch = _normalize_time_difference_frames(
            frame_batch,
            v_ref,
            floor=floor,
        )
    else:
        measurement_batch = frame_batch
    measurement_batch = _apply_measurement_contract_to_frames(
        measurement_batch,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    payload: np.ndarray
    if was_vector:
        payload = measurement_batch.reshape(-1)
    else:
        payload = measurement_batch
    return rm_matmul(
        rm,
        payload,
        device=device,
        return_metadata=return_metadata,
    )


__all__ = [
    "OneStepRMResult",
    "build_one_step_rm",
    "reconstruct_difference",
    "reconstruct_difference_batch",
]
