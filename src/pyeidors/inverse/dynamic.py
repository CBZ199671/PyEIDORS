"""Batch spatiotemporal inverse solvers for dynamic EIT windows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
)
from pyeidors.data.dynamic_sequence import DynamicMeasurementSequence
from pyeidors.inverse.prior import RtRPrior, as_rtr_prior
from pyeidors.inverse.reconstruction_matrix import (
    OneStepRMResult,
    build_one_step_rm,
    reconstruct_difference,
)


SPATIOTEMPORAL_GN_SCHEMA = "pyeidors-spatiotemporal-gn-v1"


@dataclass(frozen=True)
class SpatiotemporalGNResult:
    """Windowed 4D GN reconstruction plus baseline/solver metadata."""

    values: np.ndarray
    rowwise_baseline: np.ndarray | None
    metadata: MappingProxyType
    normal_operator: sparse.csr_matrix | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.values.shape)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def temporal_difference_operator(n_frames: int, *, order: int = 1) -> sparse.csr_matrix:
    """Build first/second-order temporal finite-difference operator ``Dt``."""

    frames = int(n_frames)
    if frames <= 0:
        raise ValueError("n_frames must be positive.")
    resolved_order = int(order)
    if resolved_order not in {1, 2}:
        raise ValueError("temporal order must be 1 or 2.")
    n_rows = max(frames - resolved_order, 0)
    if n_rows == 0:
        return sparse.csr_matrix((0, frames), dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    stencil = (-1.0, 1.0) if resolved_order == 1 else (1.0, -2.0, 1.0)
    for row in range(n_rows):
        for offset, value in enumerate(stencil):
            rows.append(row)
            cols.append(row + offset)
            data.append(value)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_rows, frames),
        dtype=np.float64,
    )


def solve_batch_spatiotemporal_gn(
    jacobian: Any,
    residuals: Any,
    *,
    spatial_prior: Any = None,
    temporal_order: int = 1,
    lambda_s: float = 1.0e-2,
    lambda_t: float = 0.0,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    rowwise_rm_baseline: bool = True,
    rowwise_rm_mode: str = "tikhonov",
    solver: str = "spsolve",
    return_normal_operator: bool = False,
    max_dense_prior_n: int | None = 4096,
    metadata: Mapping[str, Any] | None = None,
) -> SpatiotemporalGNResult:
    """Solve a windowed spatiotemporal GN/4D-prior normal equation.

    The objective is
    ``sum_t ||sqrt(W_t)(J_t x_t - y_t)||²
    + lambda_s² sum_t x_t.T Rs x_t
    + lambda_t² ||(Dt ⊗ I) vec(X)||²``.

    ``jacobian`` may be one shared ``(M, N)`` matrix/operator or a per-frame
    ``(T, M, N)`` stack. ``residuals`` may be a plain ``(T, M)`` array or a
    :class:`DynamicMeasurementSequence`; sequence masks/weights are used when
    explicit arguments are omitted.
    """

    residual_batch, sequence_meta, resolved_mask, resolved_weights = _resolve_residuals(
        residuals,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    n_frames, n_measurements = residual_batch.shape
    jac_stack, jacobian_kind = _as_jacobian_stack(
        jacobian,
        n_frames=n_frames,
        n_measurements=n_measurements,
    )
    n_parameters = int(jac_stack[0].shape[1])
    lam_s = _nonnegative_finite(lambda_s, name="lambda_s")
    lam_t = _nonnegative_finite(lambda_t, name="lambda_t")

    spatial = _resolve_spatial_prior(
        spatial_prior,
        n_parameters=n_parameters,
        max_dense_prior_n=max_dense_prior_n,
    )
    dt = temporal_difference_operator(n_frames, order=temporal_order)

    frame_normals: list[sparse.csr_matrix] = []
    rhs_blocks: list[np.ndarray] = []
    weight_kinds: list[str] = []
    bad_counts: list[int] = []
    for frame_idx in range(n_frames):
        mask_t = _frame_channel_mask(
            resolved_mask,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        weights_t = _frame_measurement_weights(
            resolved_weights,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        weighted_j, contract = apply_measurement_contract_to_jacobian(
            jac_stack[frame_idx],
            channel_mask=mask_t,
            measurement_weights=weights_t,
        )
        weighted_y, _ = apply_measurement_contract_to_vector(
            residual_batch[frame_idx],
            channel_mask=mask_t,
            measurement_weights=weights_t,
        )
        frame_normals.append(sparse.csr_matrix(weighted_j.T @ weighted_j))
        rhs_blocks.append(np.asarray(weighted_j.T @ weighted_y, dtype=np.float64))
        weight_kinds.append(contract.weight_kind)
        bad_counts.append(contract.bad_channel_count)

    normal = sparse.block_diag(frame_normals, format="csr")
    if lam_s:
        normal = normal + (lam_s * lam_s) * sparse.kron(
            sparse.eye(n_frames, format="csr", dtype=np.float64),
            spatial.matrix,
            format="csr",
        )
    if lam_t and dt.shape[0]:
        temporal_rtr = (dt.T @ dt).tocsr()
        normal = normal + (lam_t * lam_t) * sparse.kron(
            temporal_rtr,
            sparse.eye(n_parameters, format="csr", dtype=np.float64),
            format="csr",
        )
    normal = normal.tocsr()
    rhs = np.concatenate(rhs_blocks).astype(np.float64, copy=False)
    solution, solver_used = _solve_block_system(normal, rhs, solver=solver)
    values = np.ascontiguousarray(solution.reshape(n_frames, n_parameters))

    baseline = None
    comparison: dict[str, Any]
    baseline_metadata: dict[str, Any]
    if rowwise_rm_baseline:
        baseline, baseline_metadata = _rowwise_rm_baseline(
            jac_stack,
            residual_batch,
            spatial_prior=spatial.prior,
            lambda_s=lam_s,
            rowwise_rm_mode=rowwise_rm_mode,
            channel_mask=resolved_mask,
            measurement_weights=resolved_weights,
        )
        comparison = _compare_baseline(values, baseline)
    else:
        baseline_metadata = {"enabled": False}
        comparison = {"enabled": False}

    meta = {
        "schema": SPATIOTEMPORAL_GN_SCHEMA,
        "algorithm": "batch-spatiotemporal-gn",
        "solver_family": "gauss-newton",
        "prior_family": "4d_l2_spatiotemporal",
        "windowed_solve": True,
        "n_frames": int(n_frames),
        "n_measurements": int(n_measurements),
        "n_parameters": int(n_parameters),
        "jacobian_kind": jacobian_kind,
        "dual_mesh_semantics": "J columns are inverse-grid parameters; dual-mesh Jv/JTr operators may be materialized for small batch windows.",
        "lambda_s": float(lam_s),
        "lambda_t": float(lam_t),
        "lambda_s_squared": float(lam_s * lam_s),
        "lambda_t_squared": float(lam_t * lam_t),
        "spatial_prior_kind": spatial.prior.kind,
        "spatial_prior_signature_hash": spatial.prior.signature_hash,
        "spatial_prior_nnz": int(spatial.matrix.nnz),
        "spatial_prior_metadata": dict(spatial.prior.metadata),
        "temporal_order": int(temporal_order),
        "temporal_operator_shape": tuple(int(v) for v in dt.shape),
        "temporal_operator_nnz": int(dt.nnz),
        "block_normal_operator": True,
        "normal_operator_shape": tuple(int(v) for v in normal.shape),
        "normal_operator_nnz": int(normal.nnz),
        "normal_equation_formula": "block_JtWJ_plus_lambda_s2_I_kron_Rs_plus_lambda_t2_DtTDt_kron_I",
        "measurement_weight_kinds": tuple(weight_kinds),
        "bad_channel_counts": tuple(int(v) for v in bad_counts),
        "measurement_contract_applied": True,
        "rowwise_rm_baseline": baseline_metadata,
        "rowwise_rm_comparison": comparison,
        "solver": solver_used,
        "online_hot_path_replaced": False,
        "intended_tier": "dynamic_foundation_batch_cold_path",
        "sequence_metadata": sequence_meta,
    }
    if metadata:
        meta["user_metadata"] = dict(metadata)

    return SpatiotemporalGNResult(
        values=values,
        rowwise_baseline=baseline,
        metadata=MappingProxyType(meta),
        normal_operator=normal if return_normal_operator else None,
    )


@dataclass(frozen=True)
class _SpatialPrior:
    prior: RtRPrior
    matrix: sparse.csr_matrix


def _resolve_residuals(
    residuals: Any,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> tuple[np.ndarray, MappingProxyType, Any | None, Any | None]:
    if isinstance(residuals, DynamicMeasurementSequence):
        sequence = residuals
        frames = _frame_batch(sequence.frames, name="residuals")
        meta = MappingProxyType(sequence.summary())
        resolved_mask = (
            sequence.bad_channel_mask if channel_mask is None else channel_mask
        )
        resolved_weights = (
            sequence.measurement_weights
            if measurement_weights is None
            else measurement_weights
        )
        return frames, meta, resolved_mask, resolved_weights
    return (
        _frame_batch(residuals, name="residuals"),
        MappingProxyType({}),
        channel_mask,
        measurement_weights,
    )


def _as_jacobian_stack(
    jacobian: Any,
    *,
    n_frames: int,
    n_measurements: int,
) -> tuple[list[np.ndarray], str]:
    if isinstance(jacobian, (list, tuple)):
        if len(jacobian) != n_frames:
            raise ValueError(
                f"per-frame jacobian count {len(jacobian)} does not match {n_frames}."
            )
        stack = [_jacobian_matrix(item) for item in jacobian]
        kind = "per_frame"
    else:
        matrix = _jacobian_matrix(jacobian)
        if matrix.ndim == 3:
            if matrix.shape[0] != n_frames:
                raise ValueError(
                    f"jacobian frame count {matrix.shape[0]} does not match {n_frames}."
                )
            stack = [
                np.ascontiguousarray(matrix[idx], dtype=np.float64)
                for idx in range(n_frames)
            ]
            kind = "per_frame"
        else:
            stack = [matrix for _ in range(n_frames)]
            kind = "shared"
    for matrix in stack:
        if matrix.ndim != 2:
            raise ValueError("jacobian entries must be 2D matrices.")
        if matrix.shape[0] != n_measurements:
            raise ValueError(
                f"jacobian measurement count {matrix.shape[0]} does not match {n_measurements}."
            )
        if matrix.shape[1] != stack[0].shape[1]:
            raise ValueError("all jacobian frames must have the same parameter count.")
    return stack, kind


def _jacobian_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "to_dense") and callable(value.to_dense):
        raw = value.to_dense()
    elif sparse.issparse(value):
        raw = value.toarray()
    else:
        raw = value
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim not in {2, 3} or 0 in array.shape:
        raise ValueError("jacobian must be a non-empty 2D matrix or 3D stack.")
    if not np.isfinite(array).all():
        raise FloatingPointError("jacobian contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _resolve_spatial_prior(
    spatial_prior: Any,
    *,
    n_parameters: int,
    max_dense_prior_n: int | None,
) -> _SpatialPrior:
    prior = as_rtr_prior(
        spatial_prior,
        n_parameters=n_parameters,
        name="spatiotemporal_spatial",
        metadata={
            "prior_role": "spatial_Rs",
            "signature_hint": "spatiotemporal_spatial_Rs",
        },
    )
    explicit = prior.as_RtR(dense=False)
    if sparse.issparse(explicit):
        matrix = explicit.tocsr()
    elif isinstance(explicit, np.ndarray):
        matrix = sparse.csr_matrix(explicit)
    else:
        matrix = sparse.csr_matrix(
            prior.as_RtR(dense=True, max_dense_n=max_dense_prior_n)
        )
    if matrix.shape != (n_parameters, n_parameters):
        raise ValueError(
            f"spatial_prior shape {matrix.shape} does not match {(n_parameters, n_parameters)}."
        )
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError("spatial_prior contains non-finite values.")
    return _SpatialPrior(prior=prior, matrix=matrix)


def _solve_block_system(
    normal: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    solver: str,
) -> tuple[np.ndarray, str]:
    resolved = str(solver).strip().lower()
    if resolved not in {"spsolve", "lsmr"}:
        raise ValueError("solver must be one of: spsolve, lsmr.")
    if resolved == "spsolve":
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                solution = spla.spsolve(normal, rhs)
                out = np.asarray(solution, dtype=np.float64).reshape(-1)
                if np.isfinite(out).all():
                    return np.ascontiguousarray(out), "spsolve"
            except (spla.MatrixRankWarning, RuntimeError, ValueError):
                pass
    lsmr = spla.lsmr(normal, rhs)
    out = np.asarray(lsmr[0], dtype=np.float64).reshape(-1)
    if not np.isfinite(out).all():
        raise FloatingPointError("spatiotemporal GN solve produced non-finite values.")
    return np.ascontiguousarray(out), "lsmr"


def _rowwise_rm_baseline(
    jac_stack: list[np.ndarray],
    residual_batch: np.ndarray,
    *,
    spatial_prior: RtRPrior,
    lambda_s: float,
    rowwise_rm_mode: str,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    solvers: list[str] = []
    signatures: list[str] = []
    n_frames, n_measurements = residual_batch.shape
    for frame_idx, matrix in enumerate(jac_stack):
        mask_t = _frame_channel_mask(
            channel_mask,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        weights_t = _frame_measurement_weights(
            measurement_weights,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        rm = build_one_step_rm(
            matrix,
            regularization=spatial_prior,
            lambda_=lambda_s,
            mode=rowwise_rm_mode,
            channel_mask=mask_t,
            measurement_weights=weights_t,
            return_metadata=True,
        )
        assert isinstance(rm, OneStepRMResult)
        rows.append(
            reconstruct_difference(
                rm.rm,
                residual_batch[frame_idx],
                normalize=False,
                channel_mask=mask_t,
                measurement_weights=weights_t,
                device="cpu",
            )
        )
        solvers.append(str(rm.metadata.get("solver", "")))
        signatures.append(str(rm.metadata.get("RtR_signature_hash", "")))
    baseline = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
    return baseline, {
        "enabled": True,
        "mode": str(rowwise_rm_mode),
        "lambda_s": float(lambda_s),
        "n_frames": int(n_frames),
        "baseline_shape": tuple(int(v) for v in baseline.shape),
        "solvers": tuple(solvers),
        "RtR_signature_hashes": tuple(signatures),
    }


def _compare_baseline(values: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    baseline_norm = max(float(np.linalg.norm(baseline)), np.finfo(np.float64).eps)
    return {
        "enabled": True,
        "l2_delta": float(np.linalg.norm(diff)),
        "relative_l2_delta": float(np.linalg.norm(diff) / baseline_norm),
        "rmse_delta": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_delta": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def _frame_batch(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError(f"{name} must be a non-empty 1D/2D frame array.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _frame_channel_mask(
    channel_mask: Any | None,
    *,
    frame_idx: int,
    n_frames: int,
    n_measurements: int,
) -> Any | None:
    if channel_mask is None:
        return None
    arr = np.asarray(channel_mask)
    if arr.ndim == 2:
        if arr.shape != (n_frames, n_measurements):
            raise ValueError(
                "per-frame channel_mask must have shape "
                f"{(n_frames, n_measurements)}, got {arr.shape}."
            )
        return arr[int(frame_idx)]
    return channel_mask


def _frame_measurement_weights(
    measurement_weights: Any | None,
    *,
    frame_idx: int,
    n_frames: int,
    n_measurements: int,
) -> Any | None:
    if measurement_weights is None or sparse.issparse(measurement_weights):
        return measurement_weights
    arr = np.asarray(measurement_weights, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (n_frames, n_measurements):
        return arr[int(frame_idx)]
    if arr.ndim == 3:
        if arr.shape != (n_frames, n_measurements, n_measurements):
            raise ValueError(
                "per-frame full measurement_weights must have shape "
                f"{(n_frames, n_measurements, n_measurements)}, got {arr.shape}."
            )
        return arr[int(frame_idx)]
    return measurement_weights


def _nonnegative_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


__all__ = [
    "SPATIOTEMPORAL_GN_SCHEMA",
    "SpatiotemporalGNResult",
    "solve_batch_spatiotemporal_gn",
    "temporal_difference_operator",
]
