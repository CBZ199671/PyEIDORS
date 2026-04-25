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
from pyeidors.inverse.prior import RtRPrior, as_rtr_prior, graph_difference_operator
from pyeidors.inverse.reconstruction_matrix import (
    OneStepRMResult,
    build_one_step_rm,
    reconstruct_difference,
)


SPATIOTEMPORAL_GN_SCHEMA = "pyeidors-spatiotemporal-gn-v1"
SPATIOTEMPORAL_TV_HUBER_SCHEMA = "pyeidors-spatiotemporal-tv-huber-v1"


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


@dataclass(frozen=True)
class SpatiotemporalTVHuberResult:
    """IRLS spatiotemporal TV/Huber reconstruction plus T65 comparison."""

    values: np.ndarray
    l2_baseline: np.ndarray
    metadata: MappingProxyType
    normal_operator: sparse.csr_matrix | None = None
    objective_history: tuple[float, ...] = ()

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


def solve_spatiotemporal_tv_huber(
    jacobian: Any,
    residuals: Any,
    *,
    spatial_graph: Any | None = None,
    spatial_difference: Any | None = None,
    temporal_order: int = 1,
    lambda_s: float = 1.0e-2,
    lambda_t: float = 1.0e-2,
    huber_delta: float = 5.0e-2,
    epsilon: float = 1.0e-8,
    penalty: str = "huber",
    roi_mask: Any | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    initial: Any | None = None,
    max_outer_iterations: int = 6,
    tolerance: float = 1.0e-6,
    solver: str = "spsolve",
    return_normal_operator: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SpatiotemporalTVHuberResult:
    """Solve a windowed spatiotemporal TV/Huber IRLS problem.

    The robust penalty is separable: graph spatial differences ``Ls @ x_t`` and
    temporal differences ``Dt @ X[:, p]`` get independent IRLS weights. Large
    jumps receive smaller Huber weights, so abrupt onsets/wavefronts are not
    smoothed as strongly as the T65 L2 temporal prior.
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
    delta = _positive_finite(huber_delta, name="huber_delta")
    eps = _positive_finite(epsilon, name="epsilon")
    max_outer = _positive_int(max_outer_iterations, name="max_outer_iterations")
    tol = _nonnegative_finite(tolerance, name="tolerance")
    penalty_kind = _robust_penalty_kind(penalty)
    roi = _roi_mask(roi_mask, n_parameters=n_parameters)

    spatial_difference_matrix = _resolve_spatial_difference_operator(
        spatial_graph=spatial_graph,
        spatial_difference=spatial_difference,
        n_parameters=n_parameters,
        roi_mask=roi,
    )
    spatial_rtr = (spatial_difference_matrix.T @ spatial_difference_matrix).tocsr()
    dt = temporal_difference_operator(n_frames, order=temporal_order)

    frame_normals, rhs_blocks, weight_kinds, bad_counts = _normal_blocks_and_rhs(
        jac_stack,
        residual_batch,
        channel_mask=resolved_mask,
        measurement_weights=resolved_weights,
    )
    data_normal = sparse.block_diag(frame_normals, format="csr")
    rhs = np.concatenate(rhs_blocks).astype(np.float64, copy=False)

    l2 = solve_batch_spatiotemporal_gn(
        jac_stack,
        residual_batch,
        spatial_prior=spatial_rtr,
        temporal_order=temporal_order,
        lambda_s=lam_s,
        lambda_t=lam_t,
        channel_mask=resolved_mask,
        measurement_weights=resolved_weights,
        rowwise_rm_baseline=False,
        solver=solver,
    )
    baseline = np.asarray(l2.values, dtype=np.float64)
    current = _initial_dynamic_state(
        initial,
        fallback=baseline,
        n_frames=n_frames,
        n_parameters=n_parameters,
    )

    objective_history: list[float] = []
    update_history: list[float] = []
    final_normal = data_normal
    solver_used = str(l2.metadata.get("solver", solver))
    for _ in range(max_outer):
        spatial_normal, spatial_weight_range = _spatial_robust_normal(
            spatial_difference_matrix,
            current,
            penalty=penalty_kind,
            huber_delta=delta,
            epsilon=eps,
        )
        temporal_normal, temporal_weight_range = _temporal_robust_normal(
            dt,
            current,
            roi_mask=roi,
            penalty=penalty_kind,
            huber_delta=delta,
            epsilon=eps,
        )
        normal = data_normal
        if lam_s and spatial_normal.nnz:
            normal = normal + (lam_s * lam_s) * spatial_normal
        if lam_t and temporal_normal.nnz:
            normal = normal + (lam_t * lam_t) * temporal_normal
        final_normal = normal.tocsr()
        solution, solver_used = _solve_block_system(final_normal, rhs, solver=solver)
        updated = np.ascontiguousarray(solution.reshape(n_frames, n_parameters))
        objective_history.append(
            _robust_spatiotemporal_objective(
                frame_normals,
                rhs_blocks,
                updated,
                spatial_difference_matrix=spatial_difference_matrix,
                dt=dt,
                roi_mask=roi,
                lambda_s=lam_s,
                lambda_t=lam_t,
                penalty=penalty_kind,
                huber_delta=delta,
                epsilon=eps,
            )
        )
        relative_update = float(
            np.linalg.norm(updated - current)
            / max(float(np.linalg.norm(current)), np.finfo(np.float64).eps)
        )
        update_history.append(relative_update)
        current = updated
        if relative_update <= tol:
            break

    l2_comparison = _compare_baseline(current, baseline)
    meta = {
        "schema": SPATIOTEMPORAL_TV_HUBER_SCHEMA,
        "algorithm": "spatiotemporal-tv-huber-irls",
        "solver_family": "gauss-newton-irls",
        "prior_family": "4d_tv_huber_spatiotemporal",
        "windowed_solve": True,
        "n_frames": int(n_frames),
        "n_measurements": int(n_measurements),
        "n_parameters": int(n_parameters),
        "jacobian_kind": jacobian_kind,
        "dual_mesh_semantics": "J columns are inverse-grid parameters; robust penalties operate on inverse-grid graph/time differences.",
        "lambda_s": float(lam_s),
        "lambda_t": float(lam_t),
        "lambda_s_squared": float(lam_s * lam_s),
        "lambda_t_squared": float(lam_t * lam_t),
        "penalty": penalty_kind,
        "huber_delta": float(delta),
        "epsilon": float(eps),
        "max_outer_iterations": int(max_outer),
        "outer_iterations": int(len(objective_history)),
        "tolerance": float(tol),
        "update_history": tuple(update_history),
        "objective_history": tuple(objective_history),
        "spatial_difference_shape": tuple(
            int(v) for v in spatial_difference_matrix.shape
        ),
        "spatial_difference_nnz": int(spatial_difference_matrix.nnz),
        "spatial_rtr_nnz": int(spatial_rtr.nnz),
        "temporal_order": int(temporal_order),
        "temporal_operator_shape": tuple(int(v) for v in dt.shape),
        "temporal_operator_nnz": int(dt.nnz),
        "roi_enabled": roi_mask is not None,
        "roi_parameter_count": int(np.count_nonzero(roi)),
        "roi_policy": "robust penalties restricted to ROI parameters; data term remains full-window",
        "normal_operator_shape": tuple(int(v) for v in final_normal.shape),
        "normal_operator_nnz": int(final_normal.nnz),
        "normal_equation_formula": "block_JtWJ_plus_weighted_LsTWsLs_plus_weighted_DtTWtDt",
        "measurement_weight_kinds": tuple(weight_kinds),
        "bad_channel_counts": tuple(int(v) for v in bad_counts),
        "measurement_contract_applied": True,
        "t65_l2_baseline": {
            "enabled": True,
            "schema": l2.metadata["schema"],
            "algorithm": l2.metadata["algorithm"],
            "lambda_s": float(l2.metadata["lambda_s"]),
            "lambda_t": float(l2.metadata["lambda_t"]),
            "temporal_order": int(l2.metadata["temporal_order"]),
            "normal_operator_shape": tuple(
                int(v) for v in l2.metadata["normal_operator_shape"]
            ),
        },
        "t65_l2_comparison": l2_comparison,
        "spatial_weight_range": spatial_weight_range,
        "temporal_weight_range": temporal_weight_range,
        "solver": solver_used,
        "online_hot_path_replaced": False,
        "intended_tier": "dynamic_quality_batch_cold_path",
        "sequence_metadata": sequence_meta,
    }
    if metadata:
        meta["user_metadata"] = dict(metadata)
    return SpatiotemporalTVHuberResult(
        values=current,
        l2_baseline=baseline,
        metadata=MappingProxyType(meta),
        normal_operator=final_normal if return_normal_operator else None,
        objective_history=tuple(objective_history),
    )


@dataclass(frozen=True)
class _SpatialPrior:
    prior: RtRPrior
    matrix: sparse.csr_matrix


def _normal_blocks_and_rhs(
    jac_stack: list[np.ndarray],
    residual_batch: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> tuple[list[sparse.csr_matrix], list[np.ndarray], list[str], list[int]]:
    n_frames, n_measurements = residual_batch.shape
    frame_normals: list[sparse.csr_matrix] = []
    rhs_blocks: list[np.ndarray] = []
    weight_kinds: list[str] = []
    bad_counts: list[int] = []
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
        weighted_j, contract = apply_measurement_contract_to_jacobian(
            matrix,
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
    return frame_normals, rhs_blocks, weight_kinds, bad_counts


def _resolve_spatial_difference_operator(
    *,
    spatial_graph: Any | None,
    spatial_difference: Any | None,
    n_parameters: int,
    roi_mask: np.ndarray,
) -> sparse.csr_matrix:
    if spatial_difference is not None:
        matrix = _as_sparse_difference(
            spatial_difference,
            n_parameters=n_parameters,
            name="spatial_difference",
        )
    elif spatial_graph is not None:
        if _looks_like_difference_matrix(spatial_graph, n_parameters=n_parameters):
            matrix = _as_sparse_difference(
                spatial_graph,
                n_parameters=n_parameters,
                name="spatial_graph",
            )
        else:
            matrix = graph_difference_operator(spatial_graph).tocsr()
    else:
        matrix = _line_difference_operator(n_parameters)
    if matrix.shape[1] != n_parameters:
        raise ValueError(
            f"spatial difference column count {matrix.shape[1]} does not match {n_parameters}."
        )
    return _restrict_difference_rows_to_roi(matrix, roi_mask)


def _as_sparse_difference(
    value: Any,
    *,
    n_parameters: int,
    name: str,
) -> sparse.csr_matrix:
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value, dtype=np.float64)
    else:
        matrix = sparse.csr_matrix(np.asarray(value, dtype=np.float64))
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2D difference operator.")
    if matrix.shape[1] != int(n_parameters):
        raise ValueError(
            f"{name} column count {matrix.shape[1]} does not match {n_parameters}."
        )
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return matrix.tocsr()


def _looks_like_difference_matrix(value: Any, *, n_parameters: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None:
        return sparse.issparse(value) or isinstance(value, np.ndarray)
    raw = tuple(int(v) for v in shape)
    return len(raw) == 2 and raw[1] == int(n_parameters)


def _line_difference_operator(n_parameters: int) -> sparse.csr_matrix:
    n = int(n_parameters)
    if n <= 0:
        raise ValueError("n_parameters must be positive.")
    if n == 1:
        return sparse.csr_matrix((0, 1), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in range(n - 1):
        rows.extend([row, row])
        cols.extend([row, row + 1])
        data.extend([-1.0, 1.0])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n - 1, n), dtype=np.float64)


def _restrict_difference_rows_to_roi(
    matrix: sparse.csr_matrix,
    roi_mask: np.ndarray,
) -> sparse.csr_matrix:
    if bool(np.all(roi_mask)):
        return matrix.tocsr()
    csr = matrix.tocsr()
    keep: list[int] = []
    for row in range(csr.shape[0]):
        cols = csr.indices[csr.indptr[row] : csr.indptr[row + 1]]
        if cols.size and bool(np.all(roi_mask[cols])):
            keep.append(row)
    if not keep:
        return sparse.csr_matrix((0, csr.shape[1]), dtype=np.float64)
    return csr[keep, :].tocsr()


def _spatial_robust_normal(
    difference: sparse.csr_matrix,
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> tuple[sparse.csr_matrix, tuple[float, float]]:
    n_frames, n_parameters = values.shape
    if difference.shape[0] == 0:
        return (
            sparse.csr_matrix((n_frames * n_parameters, n_frames * n_parameters)),
            (0.0, 0.0),
        )
    diffs = np.asarray(values @ difference.T, dtype=np.float64)
    weights = _robust_irls_weights(
        diffs,
        penalty=penalty,
        huber_delta=huber_delta,
        epsilon=epsilon,
    )
    blocks = [
        (
            difference.T @ sparse.diags(weights[idx], 0, format="csr") @ difference
        ).tocsr()
        for idx in range(n_frames)
    ]
    return sparse.block_diag(blocks, format="csr"), _weight_range(weights)


def _temporal_robust_normal(
    dt: sparse.csr_matrix,
    values: np.ndarray,
    *,
    roi_mask: np.ndarray,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> tuple[sparse.csr_matrix, tuple[float, float]]:
    n_frames, n_parameters = values.shape
    if dt.shape[0] == 0:
        return (
            sparse.csr_matrix((n_frames * n_parameters, n_frames * n_parameters)),
            (0.0, 0.0),
        )
    diffs = np.asarray(dt @ values, dtype=np.float64)
    weights = _robust_irls_weights(
        diffs,
        penalty=penalty,
        huber_delta=huber_delta,
        epsilon=epsilon,
    )
    weights[:, ~roi_mask] = 0.0
    return _temporal_weighted_normal(dt, weights), _weight_range(weights[:, roi_mask])


def _temporal_weighted_normal(
    dt: sparse.csr_matrix,
    weights: np.ndarray,
) -> sparse.csr_matrix:
    n_temporal_rows, n_parameters = weights.shape
    n_frames = int(dt.shape[1])
    if int(dt.shape[0]) != n_temporal_rows:
        raise ValueError("temporal weights row count must match Dt row count.")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for param_idx in range(n_parameters):
        column_weights = weights[:, param_idx]
        if not np.any(column_weights > 0.0):
            continue
        block = (dt.T @ sparse.diags(column_weights, 0, format="csr") @ dt).tocoo()
        rows.extend((block.row * n_parameters + param_idx).astype(int).tolist())
        cols.extend((block.col * n_parameters + param_idx).astype(int).tolist())
        data.extend(block.data.astype(float).tolist())
    shape = (n_frames * n_parameters, n_frames * n_parameters)
    return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float64)


def _robust_irls_weights(
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> np.ndarray:
    abs_values = np.sqrt(np.asarray(values, dtype=np.float64) ** 2 + epsilon * epsilon)
    if penalty == "tv":
        return 1.0 / abs_values
    return np.where(abs_values <= huber_delta, 1.0, huber_delta / abs_values)


def _robust_penalty_values(
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    abs_values = np.sqrt(arr * arr + epsilon * epsilon)
    if penalty == "tv":
        return abs_values
    return np.where(
        abs_values <= huber_delta,
        0.5 * abs_values * abs_values,
        huber_delta * (abs_values - 0.5 * huber_delta),
    )


def _robust_spatiotemporal_objective(
    frame_normals: list[sparse.csr_matrix],
    rhs_blocks: list[np.ndarray],
    values: np.ndarray,
    *,
    spatial_difference_matrix: sparse.csr_matrix,
    dt: sparse.csr_matrix,
    roi_mask: np.ndarray,
    lambda_s: float,
    lambda_t: float,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> float:
    data = 0.0
    for frame_idx, frame in enumerate(values):
        data += 0.5 * float(frame @ (frame_normals[frame_idx] @ frame))
        data -= float(rhs_blocks[frame_idx] @ frame)
    spatial = 0.0
    if spatial_difference_matrix.shape[0]:
        spatial_diffs = np.asarray(values @ spatial_difference_matrix.T)
        spatial = float(
            np.sum(
                _robust_penalty_values(
                    spatial_diffs,
                    penalty=penalty,
                    huber_delta=huber_delta,
                    epsilon=epsilon,
                )
            )
        )
    temporal = 0.0
    if dt.shape[0]:
        temporal_diffs = np.asarray(dt @ values)
        temporal_diffs[:, ~roi_mask] = 0.0
        temporal = float(
            np.sum(
                _robust_penalty_values(
                    temporal_diffs[:, roi_mask],
                    penalty=penalty,
                    huber_delta=huber_delta,
                    epsilon=epsilon,
                )
            )
        )
    return float(data + lambda_s * lambda_s * spatial + lambda_t * lambda_t * temporal)


def _initial_dynamic_state(
    initial: Any | None,
    *,
    fallback: np.ndarray,
    n_frames: int,
    n_parameters: int,
) -> np.ndarray:
    if initial is None:
        return np.ascontiguousarray(fallback, dtype=np.float64)
    arr = np.asarray(initial, dtype=np.float64)
    if arr.shape != (n_frames, n_parameters):
        raise ValueError(
            f"initial shape {arr.shape} does not match {(n_frames, n_parameters)}."
        )
    if not np.isfinite(arr).all():
        raise FloatingPointError("initial contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _roi_mask(value: Any | None, *, n_parameters: int) -> np.ndarray:
    if value is None:
        return np.ones(int(n_parameters), dtype=bool)
    arr = np.asarray(value)
    if arr.dtype == bool:
        mask = arr.reshape(-1)
        if mask.size != int(n_parameters):
            raise ValueError(
                f"roi_mask length {mask.size} does not match {n_parameters}."
            )
        return np.ascontiguousarray(mask, dtype=bool)
    indices = arr.astype(np.int64, copy=False).reshape(-1)
    if np.any((indices < 0) | (indices >= int(n_parameters))):
        raise ValueError("roi_mask indices are out of range.")
    mask = np.zeros(int(n_parameters), dtype=bool)
    mask[indices] = True
    return mask


def _robust_penalty_kind(value: str) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    if resolved not in {"huber", "tv"}:
        raise ValueError("penalty must be one of: huber, tv.")
    return resolved


def _weight_range(weights: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return (0.0, 0.0)
    return (float(np.min(positive)), float(np.max(positive)))


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


def _positive_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return out


def _positive_int(value: int, *, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


__all__ = [
    "SPATIOTEMPORAL_GN_SCHEMA",
    "SPATIOTEMPORAL_TV_HUBER_SCHEMA",
    "SpatiotemporalGNResult",
    "SpatiotemporalTVHuberResult",
    "solve_batch_spatiotemporal_gn",
    "solve_spatiotemporal_tv_huber",
    "temporal_difference_operator",
]
