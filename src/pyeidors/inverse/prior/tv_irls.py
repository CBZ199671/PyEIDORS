"""TV-IRLS priors built on the generic RtR contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
)

from .laplace import graph_difference_operator
from .rtr import RtRPrior, as_rtr_prior


@dataclass(frozen=True)
class TVIRLSResult:
    """TV-IRLS reconstruction result plus per-outer-iteration metadata."""

    values: np.ndarray
    metadata: MappingProxyType
    history: tuple[MappingProxyType, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def tv_irls_prior_from_state(
    mesh_or_difference: Any,
    state: Any,
    *,
    beta: float = 1.0e-6,
    beta_floor: float = 1.0e-12,
    graph_weight: str = "unit",
    iteration: int = 0,
    name: str = "tv_irls",
) -> RtRPrior:
    """Build ``L.T @ diag(1/sqrt((Lx)^2 + beta)) @ L`` as an RtR prior."""

    effective_beta = _effective_beta(beta, beta_floor)
    difference = _difference_operator(mesh_or_difference, graph_weight=graph_weight)
    n_parameters = int(difference.shape[1])
    state_vec = _state_vector(state, n_parameters=n_parameters)
    gradient = np.asarray(difference @ state_vec, dtype=np.float64).reshape(-1)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("TV-IRLS graph gradient contains non-finite values.")
    weights = 1.0 / np.sqrt(
        np.maximum(gradient * gradient + effective_beta, beta_floor)
    )
    if not np.isfinite(weights).all():
        raise FloatingPointError("TV-IRLS weights contain non-finite values.")
    matrix = (
        difference.T @ sparse.diags(weights, 0, format="csr") @ difference
    ).tocsr()
    state_signature = _digest_array(state_vec)
    signature_hint = f"tv_irls:{state_signature}:beta={effective_beta:.16e}"
    return as_rtr_prior(
        matrix,
        n_parameters=n_parameters,
        name=name,
        metadata={
            "prior_family": "tv_irls",
            "regularization_source": "tv_irls_graph_difference_operator",
            "signature_hint": signature_hint,
            "state_signature": state_signature,
            "stale_rm_token": signature_hint,
            "irls_iteration": int(iteration),
            "beta": float(beta),
            "beta_floor": float(beta_floor),
            "effective_beta": float(effective_beta),
            "graph_weight": str(graph_weight).strip().lower(),
            "difference_operator_shape": tuple(int(v) for v in difference.shape),
            "weight_min": float(np.min(weights)) if weights.size else 0.0,
            "weight_max": float(np.max(weights)) if weights.size else 0.0,
            "weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        },
    )


def solve_tv_irls_frame(
    jacobian: Any,
    measurement: Any,
    mesh_or_difference: Any,
    *,
    lambda_: float = 1.0e-2,
    initial: Any | None = None,
    beta: float = 1.0e-6,
    beta_floor: float = 1.0e-12,
    graph_weight: str = "unit",
    max_outer_iterations: int = 5,
    tolerance: float = 1.0e-5,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    monotone: bool = True,
) -> TVIRLSResult:
    """Run a small framewise TV-IRLS inverse solve using RM rebuilds."""

    from pyeidors.inverse.reconstruction_matrix import (
        build_one_step_rm,
        reconstruct_difference,
    )

    lam = _nonnegative_finite(lambda_, name="lambda_")
    max_iter = int(max_outer_iterations)
    if max_iter <= 0:
        raise ValueError("max_outer_iterations must be positive.")
    tol = _nonnegative_finite(tolerance, name="tolerance")
    effective_beta = _effective_beta(beta, beta_floor)
    difference = _difference_operator(mesh_or_difference, graph_weight=graph_weight)
    n_parameters = int(difference.shape[1])
    measurement_vec = _measurement_vector(measurement)
    if initial is None:
        seed_rm = build_one_step_rm(
            jacobian,
            lambda_=lam,
            mode="tikhonov",
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
        )
        current = reconstruct_difference(
            seed_rm,
            measurement_vec,
            normalize=False,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            device="cpu",
        )
        initial_source = "tikhonov_rm"
    else:
        current = _state_vector(initial, n_parameters=n_parameters)
        initial_source = "provided"

    current = _state_vector(current, n_parameters=n_parameters)
    current_objective = tv_irls_objective(
        jacobian,
        measurement_vec,
        current,
        difference,
        lambda_=lam,
        beta=effective_beta,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    objective_history = [float(current_objective)]
    history: list[MappingProxyType] = []
    stopped_reason = "max_outer_iterations"
    final_prior_metadata: Mapping[str, Any] = {}

    for iteration in range(max_iter):
        prior = tv_irls_prior_from_state(
            difference,
            current,
            beta=effective_beta,
            beta_floor=beta_floor,
            graph_weight=graph_weight,
            iteration=iteration,
        )
        rm = build_one_step_rm(
            jacobian,
            regularization=prior,
            lambda_=lam,
            mode="tv_irls",
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            return_metadata=True,
        )
        candidate = reconstruct_difference(
            rm.rm,
            measurement_vec,
            normalize=False,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            device="cpu",
        )
        candidate = _state_vector(candidate, n_parameters=n_parameters)
        accepted, accepted_objective, step_fraction = _accept_monotone_candidate(
            current,
            candidate,
            objective_current=current_objective,
            jacobian=jacobian,
            measurement=measurement_vec,
            difference=difference,
            lambda_=lam,
            beta=effective_beta,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            monotone=bool(monotone),
        )
        relative_change = float(
            np.linalg.norm(accepted - current) / max(np.linalg.norm(current), 1.0e-12)
        )
        final_prior_metadata = dict(prior.metadata)
        history_item = MappingProxyType(
            {
                "iteration": int(iteration),
                "objective_before": float(current_objective),
                "objective_candidate": float(
                    tv_irls_objective(
                        jacobian,
                        measurement_vec,
                        candidate,
                        difference,
                        lambda_=lam,
                        beta=effective_beta,
                        channel_mask=channel_mask,
                        measurement_weights=measurement_weights,
                    )
                ),
                "objective": float(accepted_objective),
                "relative_change": relative_change,
                "step_fraction": float(step_fraction),
                "RtR_signature_hash": prior.signature_hash,
                "state_signature": str(prior.metadata["state_signature"]),
                "stale_rm_token": str(prior.metadata["stale_rm_token"]),
                "rm_signature_hash": str(rm.metadata["RtR_signature_hash"]),
            }
        )
        history.append(history_item)
        current = accepted
        current_objective = float(accepted_objective)
        objective_history.append(float(current_objective))
        if relative_change <= max(tol, 1.0e-15):
            stopped_reason = "relative_change_tolerance"
            break

    meta = MappingProxyType(
        {
            "method": "tv-irls",
            "regularization_type": "tv_irls",
            "lambda": float(lam),
            "beta": float(beta),
            "beta_floor": float(beta_floor),
            "effective_beta": float(effective_beta),
            "graph_weight": str(graph_weight).strip().lower(),
            "max_outer_iterations": int(max_iter),
            "iterations": int(len(history)),
            "stopped_reason": stopped_reason,
            "initial_source": initial_source,
            "monotone": bool(monotone),
            "objective_history": tuple(float(v) for v in objective_history),
            "objective_monotone": _is_nonincreasing(objective_history),
            "RtR_signature_hash_history": tuple(
                str(item["RtR_signature_hash"]) for item in history
            ),
            "stale_rm_token_history": tuple(
                str(item["stale_rm_token"]) for item in history
            ),
            "final_RtR_signature_hash": str(history[-1]["RtR_signature_hash"])
            if history
            else "",
            "final_prior_metadata": dict(final_prior_metadata),
            "tv_pdhg_postprocess_separate": True,
        }
    )
    return TVIRLSResult(
        values=np.ascontiguousarray(current, dtype=np.float64),
        metadata=meta,
        history=tuple(history),
    )


def solve_tv_irls_batch(
    jacobian: Any,
    frames: Any,
    mesh_or_difference: Any,
    *,
    lambda_: float = 1.0e-2,
    initial: Any | None = None,
    beta: float = 1.0e-6,
    beta_floor: float = 1.0e-12,
    graph_weight: str = "unit",
    max_outer_iterations: int = 5,
    tolerance: float = 1.0e-5,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    monotone: bool = True,
) -> TVIRLSResult:
    """Apply :func:`solve_tv_irls_frame` independently to one or more frames."""

    frame_batch, was_vector = _frame_batch(frames)
    initial_batch = _initial_batch(initial, n_frames=frame_batch.shape[0])
    results = [
        solve_tv_irls_frame(
            jacobian,
            frame,
            mesh_or_difference,
            lambda_=lambda_,
            initial=None if initial_batch is None else initial_batch[idx],
            beta=beta,
            beta_floor=beta_floor,
            graph_weight=graph_weight,
            max_outer_iterations=max_outer_iterations,
            tolerance=tolerance,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            monotone=monotone,
        )
        for idx, frame in enumerate(frame_batch)
    ]
    values = np.vstack(
        [np.asarray(result.values, dtype=np.float64) for result in results]
    )
    if was_vector:
        values = values.reshape(-1)
    metadata = MappingProxyType(
        {
            "method": "tv-irls-batch",
            "n_frames": int(frame_batch.shape[0]),
            "frame_iterations": tuple(
                int(result.metadata["iterations"]) for result in results
            ),
            "objective_monotone_all": all(
                bool(result.metadata["objective_monotone"]) for result in results
            ),
            "final_RtR_signature_hashes": tuple(
                str(result.metadata["final_RtR_signature_hash"]) for result in results
            ),
            "stale_rm_token_history": tuple(
                tuple(str(token) for token in result.metadata["stale_rm_token_history"])
                for result in results
            ),
            "frame_metadata": tuple(dict(result.metadata) for result in results),
            "tv_pdhg_postprocess_separate": True,
        }
    )
    return TVIRLSResult(
        values=np.ascontiguousarray(values, dtype=np.float64),
        metadata=metadata,
        history=tuple(item for result in results for item in result.history),
    )


def tv_irls_objective(
    jacobian: Any,
    measurement: Any,
    state: Any,
    mesh_or_difference: Any,
    *,
    lambda_: float,
    beta: float,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
) -> float:
    """Weighted data misfit plus smoothed TV penalty for one frame."""

    lam = _nonnegative_finite(lambda_, name="lambda_")
    effective_beta = _effective_beta(beta, beta)
    difference = _difference_operator(mesh_or_difference, graph_weight="unit")
    state_vec = _state_vector(state, n_parameters=int(difference.shape[1]))
    jac, _ = apply_measurement_contract_to_jacobian(
        jacobian,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    meas, _ = apply_measurement_contract_to_vector(
        _measurement_vector(measurement),
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    residual = np.asarray(jac @ state_vec - meas, dtype=np.float64).reshape(-1)
    gradient = np.asarray(difference @ state_vec, dtype=np.float64).reshape(-1)
    tv = np.sum(np.sqrt(gradient * gradient + effective_beta))
    value = 0.5 * float(np.dot(residual, residual)) + (lam * lam) * float(tv)
    if not np.isfinite(value):
        raise FloatingPointError("TV-IRLS objective is non-finite.")
    return float(value)


def _accept_monotone_candidate(
    current: np.ndarray,
    candidate: np.ndarray,
    *,
    objective_current: float,
    jacobian: Any,
    measurement: np.ndarray,
    difference: sparse.spmatrix,
    lambda_: float,
    beta: float,
    channel_mask: Any | None,
    measurement_weights: Any | None,
    monotone: bool,
) -> tuple[np.ndarray, float, float]:
    candidate_objective = tv_irls_objective(
        jacobian,
        measurement,
        candidate,
        difference,
        lambda_=lambda_,
        beta=beta,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    slack = 1.0e-12 * max(1.0, abs(float(objective_current)))
    if (not monotone) or candidate_objective <= objective_current + slack:
        return candidate, float(candidate_objective), 1.0
    for fraction in (0.5, 0.25, 0.125, 0.0625, 0.0):
        blended = current + float(fraction) * (candidate - current)
        objective = tv_irls_objective(
            jacobian,
            measurement,
            blended,
            difference,
            lambda_=lambda_,
            beta=beta,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
        )
        if objective <= objective_current + slack:
            return np.asarray(blended, dtype=np.float64), float(objective), fraction
    return current.copy(), float(objective_current), 0.0


def _difference_operator(
    mesh_or_difference: Any, *, graph_weight: str
) -> sparse.csr_matrix:
    if sparse.issparse(mesh_or_difference):
        difference = mesh_or_difference.tocsr().astype(np.float64)
    else:
        difference = graph_difference_operator(mesh_or_difference, weight=graph_weight)
    if difference.ndim != 2 or difference.shape[1] == 0:
        raise ValueError("TV-IRLS difference operator must be non-empty 2D.")
    if difference.nnz and not np.isfinite(difference.data).all():
        raise FloatingPointError(
            "TV-IRLS difference operator contains non-finite data."
        )
    return difference


def _effective_beta(beta: float, beta_floor: float) -> float:
    beta_value = _positive_finite(beta, name="beta")
    floor_value = _positive_finite(beta_floor, name="beta_floor")
    return float(max(beta_value, floor_value))


def _positive_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return out


def _nonnegative_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


def _state_vector(value: Any, *, n_parameters: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != int(n_parameters):
        raise ValueError(
            f"TV-IRLS state length {vector.size} does not match {n_parameters}."
        )
    if not np.isfinite(vector).all():
        raise FloatingPointError("TV-IRLS state contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _measurement_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("measurement must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError("measurement contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _frame_batch(value: Any) -> tuple[np.ndarray, bool]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("frames must be a non-empty 1D or 2D array.")
        if not np.isfinite(arr).all():
            raise FloatingPointError("frames contain non-finite values.")
        return np.ascontiguousarray(arr.reshape(1, -1), dtype=np.float64), True
    if arr.ndim == 2 and 0 not in arr.shape:
        if not np.isfinite(arr).all():
            raise FloatingPointError("frames contain non-finite values.")
        return np.ascontiguousarray(arr, dtype=np.float64), False
    raise ValueError("frames must be a non-empty 1D or 2D array.")


def _initial_batch(initial: Any | None, *, n_frames: int) -> np.ndarray | None:
    if initial is None:
        return None
    arr = np.asarray(initial, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise FloatingPointError("initial contains non-finite values.")
    if arr.ndim == 1:
        return np.broadcast_to(arr.reshape(1, -1), (int(n_frames), arr.size)).copy()
    if arr.ndim == 2 and arr.shape[0] == int(n_frames):
        return np.ascontiguousarray(arr, dtype=np.float64)
    raise ValueError("initial must be a 1D state or a frame-aligned 2D state batch.")


def _digest_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    payload = (
        str(arr.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in arr.shape]).encode()
        + b"|"
        + arr.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _is_nonincreasing(values: list[float] | tuple[float, ...]) -> bool:
    return all(
        float(right) <= float(left) + 1.0e-10
        for left, right in zip(values, values[1:], strict=False)
    )


__all__ = [
    "TVIRLSResult",
    "solve_tv_irls_batch",
    "solve_tv_irls_frame",
    "tv_irls_objective",
    "tv_irls_prior_from_state",
]
