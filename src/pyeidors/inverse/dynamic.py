"""Batch spatiotemporal inverse solvers for dynamic EIT windows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence
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
DYNAMIC_KALMAN_SCHEMA = "pyeidors-dynamic-kalman-fixed-lag-v1"


def _all_finite_values(values: np.ndarray, *, chunk_size: int = 65536) -> bool:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return True
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    work = np.empty(block_size, dtype=bool)
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        count = stop - start
        work_view = work[:count]
        np.isfinite(arr[start:stop], out=work_view)
        if not bool(work_view.all()):
            return False
    return True


def _dense_identity(n: int, *, scale: float = 1.0) -> np.ndarray:
    """Build a dense scaled identity without an intermediate identity product."""

    size = int(n)
    matrix = np.zeros((size, size), dtype=np.float64)
    if size > 0 and float(scale) != 0.0:
        matrix.reshape(-1)[:: size + 1] = float(scale)
    return matrix


def _dense_diagonal(values: Any) -> np.ndarray:
    diagonal = np.asarray(values, dtype=np.float64).reshape(-1)
    size = int(diagonal.size)
    matrix = np.zeros((size, size), dtype=np.float64)
    if size > 0:
        matrix.reshape(-1)[:: size + 1] = diagonal
    return matrix


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


@dataclass(frozen=True)
class DynamicKalmanResult:
    """Online Kalman / fixed-lag smoother reconstruction result."""

    filtered: np.ndarray
    smoothed: np.ndarray
    predicted: np.ndarray
    covariance_trace: np.ndarray
    metadata: MappingProxyType

    @property
    def values(self) -> np.ndarray:
        return self.smoothed

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.smoothed.shape)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.smoothed, dtype=dtype)


def _stack_row_vectors_direct(
    rows: Sequence[Any],
    *,
    dtype: Any = np.float64,
    name: str = "rows",
) -> np.ndarray:
    """Build a 2D frame-by-state matrix without retaining a vstack temporary."""

    if not rows:
        raise ValueError(f"{name} must contain at least one row.")
    resolved_dtype = np.dtype(dtype)
    first = np.asarray(rows[0], dtype=resolved_dtype).reshape(-1)
    out = np.empty((len(rows), first.size), dtype=resolved_dtype)
    out[0, :] = first
    for row_idx in range(1, len(rows)):
        row = np.asarray(rows[row_idx], dtype=resolved_dtype).reshape(-1)
        if row.size != first.size:
            raise ValueError(
                f"{name} row {row_idx} length {row.size} does not match {first.size}."
            )
        out[row_idx, :] = row
    return np.ascontiguousarray(out, dtype=resolved_dtype)


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
    rhs = np.empty(n_frames * n_parameters, dtype=np.float64)
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
        rhs_start = frame_idx * n_parameters
        rhs[rhs_start : rhs_start + n_parameters] = np.asarray(
            weighted_j.T @ weighted_y,
            dtype=np.float64,
        ).reshape(-1)
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
    rhs = np.empty(n_frames * n_parameters, dtype=np.float64)
    for frame_idx, rhs_block in enumerate(rhs_blocks):
        rhs_start = frame_idx * n_parameters
        rhs[rhs_start : rhs_start + n_parameters] = np.asarray(
            rhs_block,
            dtype=np.float64,
        ).reshape(-1)

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


def run_dynamic_kalman_filter(
    observation_model: Any,
    observations: Any,
    *,
    observation_mode: str = "jacobian",
    transition: Any | None = None,
    process_noise: Any | None = None,
    measurement_noise: Any | None = None,
    initial_state: Any | None = None,
    initial_covariance: Any | None = None,
    fixed_lag: int = 0,
    process_noise_hook: Any | None = None,
    measurement_noise_hook: Any | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    innovation_gate: str = "none",
    innovation_gate_candidates: Any | None = None,
    innovation_nis_threshold: float | None = None,
    innovation_max_variance_inflation: float = 1.0e6,
    timestamps: Any | None = None,
    sampling_rate_hz: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DynamicKalmanResult:
    """Run linear online Kalman filtering with optional fixed-lag smoothing.

    ``observation_mode="jacobian"`` treats ``observation_model`` as ``J`` and
    observations as measurement frames ``y_t``. ``"rm_observation"`` treats
    ``observation_model`` as a reconstruction matrix and first projects
    measurements into state observations ``z_t = RM @ y_t``; the Kalman
    observation matrix is then identity. Robust innovation gating is deliberately
    candidate-constrained: a large innovation alone cannot distinguish a physical
    step from an isolated outlier. This stays a prototype dynamic layer and does
    not replace the cached RM hot path.
    """

    mode = _kalman_observation_mode(observation_mode)
    obs_batch = _frame_batch(observations, name="observations")
    n_frames, n_observations_raw = obs_batch.shape
    projected, observation_matrices, contract_kinds, bad_counts = _kalman_observations(
        observation_model,
        obs_batch,
        mode=mode,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    n_state = int(observation_matrices[0].shape[1])
    transition_matrix = _transition_matrix(transition, n_state=n_state)
    initial = _kalman_initial_state(initial_state, n_state=n_state)
    initial_cov = _kalman_covariance(
        initial_covariance,
        n=n_state,
        name="initial_covariance",
        default_scale=1.0,
    )
    fixed_lag_frames = _nonnegative_int(fixed_lag, name="fixed_lag")
    gate_policy = _innovation_gate_policy(innovation_gate)
    gate_candidates = _innovation_gate_candidate_frames(
        innovation_gate_candidates,
        n_frames=n_frames,
    )
    nis_threshold = _innovation_nis_threshold(
        innovation_nis_threshold,
        gate_policy=gate_policy,
    )
    max_variance_inflation = _positive_finite_float(
        innovation_max_variance_inflation,
        name="innovation_max_variance_inflation",
        minimum=1.0,
    )
    times = _optional_timestamps(timestamps, n_frames=n_frames)
    latency_seconds = _latency_seconds(
        fixed_lag_frames,
        timestamps=times,
        sampling_rate_hz=sampling_rate_hz,
    )

    predicted_states: list[np.ndarray] = []
    predicted_covs: list[np.ndarray] = []
    filtered_states: list[np.ndarray] = []
    filtered_covs: list[np.ndarray] = []
    innovation_norms: list[float] = []
    innovation_nis_values: list[float] = []
    innovation_gate_triggered: list[bool] = []
    innovation_gate_actions: list[str] = []
    innovation_variance_inflations: list[float] = []
    kalman_gain_norms: list[float] = []
    process_noise_sources: list[str] = []
    measurement_noise_sources: list[str] = []
    x_prev = initial
    p_prev = initial_cov
    identity_state = _dense_identity(n_state)
    for frame_idx in range(n_frames):
        q_t, q_source = _resolve_kalman_noise(
            process_noise,
            hook=process_noise_hook,
            n=n_state,
            frame_idx=frame_idx,
            state=x_prev,
            observation=projected[frame_idx],
            default_scale=1.0e-4,
            name="process_noise",
        )
        x_pred = transition_matrix @ x_prev
        p_pred = transition_matrix @ p_prev @ transition_matrix.T + q_t
        h_t = observation_matrices[frame_idx]
        r_t, r_source = _resolve_measurement_noise(
            measurement_noise,
            hook=measurement_noise_hook,
            n_observations=h_t.shape[0],
            frame_idx=frame_idx,
            state=x_pred,
            observation=projected[frame_idx],
            default_scale=1.0e-2,
            mode=mode,
        )
        innovation = projected[frame_idx] - h_t @ x_pred
        innovation_cov = h_t @ p_pred @ h_t.T + r_t
        innovation_nis = _normalized_innovation_squared(
            innovation,
            innovation_cov,
        )
        gate_triggered = bool(
            gate_policy != "none"
            and gate_candidates[frame_idx]
            and innovation_nis > nis_threshold
        )
        gate_action = "none"
        variance_inflation = 1.0
        if gate_triggered and gate_policy == "reject":
            kalman_gain = np.zeros(
                (n_state, h_t.shape[0]),
                dtype=np.float64,
            )
            x_filt = x_pred
            p_filt = p_pred
            gate_action = "reject"
        else:
            if gate_triggered and gate_policy == "inflate":
                variance_inflation = min(
                    max_variance_inflation,
                    max(1.0, innovation_nis / nis_threshold),
                )
                r_t = _symmetrize(r_t * variance_inflation)
                innovation_cov = h_t @ p_pred @ h_t.T + r_t
                gate_action = "inflate"
            kalman_gain = _kalman_gain(p_pred, h_t, innovation_cov)
            x_filt = x_pred + kalman_gain @ innovation
            kh = kalman_gain @ h_t
            i_minus_kh = identity_state.copy()
            np.subtract(i_minus_kh, kh, out=i_minus_kh)
            p_filt = (
                i_minus_kh @ p_pred @ i_minus_kh.T + kalman_gain @ r_t @ kalman_gain.T
            )
        predicted_states.append(np.ascontiguousarray(x_pred, dtype=np.float64))
        predicted_covs.append(_symmetrize(p_pred))
        filtered_states.append(np.ascontiguousarray(x_filt, dtype=np.float64))
        filtered_covs.append(_symmetrize(p_filt))
        innovation_norms.append(float(np.linalg.norm(innovation)))
        innovation_nis_values.append(float(innovation_nis))
        innovation_gate_triggered.append(gate_triggered)
        innovation_gate_actions.append(gate_action)
        innovation_variance_inflations.append(float(variance_inflation))
        kalman_gain_norms.append(float(np.linalg.norm(kalman_gain)))
        process_noise_sources.append(q_source)
        measurement_noise_sources.append(r_source)
        x_prev = x_filt
        p_prev = _symmetrize(p_filt)

    predicted = _stack_row_vectors_direct(
        predicted_states,
        dtype=np.float64,
        name="predicted_states",
    )
    filtered = _stack_row_vectors_direct(
        filtered_states,
        dtype=np.float64,
        name="filtered_states",
    )
    smoothed, smoother_meta = _fixed_lag_smoother(
        filtered_states,
        filtered_covs,
        predicted_covs,
        transition_matrix=transition_matrix,
        fixed_lag=fixed_lag_frames,
    )
    cov_trace = np.asarray(
        [float(np.trace(covariance)) for covariance in filtered_covs],
        dtype=np.float64,
    )
    meta = {
        "schema": DYNAMIC_KALMAN_SCHEMA,
        "algorithm": "online-kalman-fixed-lag-smoother",
        "observation_mode": mode,
        "state_model": "x_t=A_x_prev_plus_q",
        "n_frames": int(n_frames),
        "n_state": int(n_state),
        "n_observations_raw": int(n_observations_raw),
        "n_observations_effective": int(projected.shape[1]),
        "transition_shape": tuple(int(v) for v in transition_matrix.shape),
        "process_noise_hook_used": process_noise_hook is not None,
        "measurement_noise_hook_used": measurement_noise_hook is not None,
        "process_noise_sources": tuple(process_noise_sources),
        "measurement_noise_sources": tuple(measurement_noise_sources),
        "fixed_lag": int(fixed_lag_frames),
        "latency_frames": int(fixed_lag_frames),
        "latency_seconds": float(latency_seconds),
        "latency_policy": "fixed_lag_frames",
        "smoother": smoother_meta,
        "measurement_contract_applied": mode == "jacobian",
        "measurement_weight_kinds": tuple(contract_kinds),
        "bad_channel_counts": tuple(int(v) for v in bad_counts),
        "innovation_norms": tuple(innovation_norms),
        "innovation_nis": tuple(innovation_nis_values),
        "innovation_gate_policy": gate_policy,
        "innovation_gate_candidates": tuple(bool(v) for v in gate_candidates),
        "innovation_nis_threshold": None
        if gate_policy == "none"
        else float(nis_threshold),
        "innovation_gate_triggered": tuple(innovation_gate_triggered),
        "innovation_gate_actions": tuple(innovation_gate_actions),
        "innovation_variance_inflations": tuple(innovation_variance_inflations),
        "innovation_gate_count": int(sum(innovation_gate_triggered)),
        "innovation_reject_count": int(
            sum(action == "reject" for action in innovation_gate_actions)
        ),
        "innovation_inflate_count": int(
            sum(action == "inflate" for action in innovation_gate_actions)
        ),
        "kalman_gain_norms": tuple(kalman_gain_norms),
        "covariance_trace_min": float(np.min(cov_trace)) if cov_trace.size else 0.0,
        "covariance_trace_max": float(np.max(cov_trace)) if cov_trace.size else 0.0,
        "online_hot_path": "rm_observation_plus_kalman"
        if mode == "rm_observation"
        else "jacobian_observation_kalman",
        "online_hot_path_replaced": False,
        "default_enabled": False,
        "intended_tier": "dynamic_quality_realtime_prototype",
        "forward_solve_count": 0,
        "adjoint_solve_count": 0,
        "ksp_solve_count": 0,
        "jacobian_rebuild_count": 0,
        "requires_t69_gate_before_default": True,
    }
    if metadata:
        meta["user_metadata"] = dict(metadata)
    return DynamicKalmanResult(
        filtered=filtered,
        smoothed=smoothed,
        predicted=predicted,
        covariance_trace=cov_trace,
        metadata=MappingProxyType(meta),
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
    if matrix.nnz and not _all_finite_values(matrix.data):
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
    roi = np.asarray(roi_mask, dtype=bool).reshape(-1)
    if bool(np.all(roi)):
        return matrix.tocsr()
    csr = matrix.tocsr()
    keep: list[int] = []
    for row in range(csr.shape[0]):
        cols = csr.indices[csr.indptr[row] : csr.indptr[row + 1]]
        if cols.size and _all_mask_indices_enabled(roi, cols):
            keep.append(row)
    if not keep:
        return sparse.csr_matrix((0, csr.shape[1]), dtype=np.float64)
    return csr[keep, :].tocsr()


def _all_mask_indices_enabled(mask: np.ndarray, indices: np.ndarray) -> bool:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    for raw_idx in np.asarray(indices).reshape(-1):
        if not bool(values[int(raw_idx)]):
            return False
    return True


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
    roi_all = bool(np.all(roi_mask))
    if not roi_all:
        _zero_non_roi_columns_in_place(weights, roi_mask)
    return _temporal_weighted_normal(dt, weights), _weight_range(
        weights,
        column_mask=None if roi_all else roi_mask,
    )


def _zero_non_roi_columns_in_place(values: np.ndarray, roi_mask: np.ndarray) -> None:
    for col_idx, enabled in enumerate(np.asarray(roi_mask, dtype=bool).reshape(-1)):
        if not bool(enabled):
            values[:, col_idx] = 0.0


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
        if not _has_positive_value(column_weights):
            continue
        block = (dt.T @ sparse.diags(column_weights, 0, format="csr") @ dt).tocoo()
        rows.extend((block.row * n_parameters + param_idx).astype(int).tolist())
        cols.extend((block.col * n_parameters + param_idx).astype(int).tolist())
        data.extend(block.data.astype(float).tolist())
    shape = (n_frames * n_parameters, n_frames * n_parameters)
    return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float64)


def _has_positive_value(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return bool(arr.size) and float(np.max(arr)) > 0.0


def _robust_irls_weights(
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> np.ndarray:
    abs_values = np.array(values, dtype=np.float64, copy=True)
    np.square(abs_values, out=abs_values)
    abs_values += epsilon * epsilon
    np.sqrt(abs_values, out=abs_values)
    if penalty == "tv":
        np.reciprocal(abs_values, out=abs_values)
        return abs_values
    weights = np.empty_like(abs_values)
    np.divide(huber_delta, abs_values, out=weights)
    np.minimum(weights, 1.0, out=weights)
    return weights


def _robust_penalty_values(
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
) -> np.ndarray:
    abs_values = np.array(values, dtype=np.float64, copy=True)
    np.square(abs_values, out=abs_values)
    abs_values += epsilon * epsilon
    np.sqrt(abs_values, out=abs_values)
    if penalty == "tv":
        return abs_values
    quadratic_mask = abs_values <= huber_delta
    penalty_values = np.empty_like(abs_values)
    np.multiply(abs_values, huber_delta, out=penalty_values)
    penalty_values -= 0.5 * huber_delta * huber_delta
    np.square(abs_values, out=penalty_values, where=quadratic_mask)
    np.multiply(penalty_values, 0.5, out=penalty_values, where=quadratic_mask)
    return penalty_values


def _robust_penalty_sum(
    values: np.ndarray,
    *,
    penalty: str,
    huber_delta: float,
    epsilon: float,
    column_mask: np.ndarray | None = None,
) -> float:
    arr = np.asarray(values)
    if column_mask is None or arr.ndim != 2:
        return float(
            np.sum(
                _robust_penalty_values(
                    arr,
                    penalty=penalty,
                    huber_delta=huber_delta,
                    epsilon=epsilon,
                )
            )
        )
    mask = np.asarray(column_mask, dtype=bool).reshape(-1)
    if mask.size != arr.shape[1]:
        raise ValueError("column_mask length must match values columns.")
    if bool(np.all(mask)):
        return float(
            np.sum(
                _robust_penalty_values(
                    arr,
                    penalty=penalty,
                    huber_delta=huber_delta,
                    epsilon=epsilon,
                )
            )
        )
    total = 0.0
    for column_idx, enabled in enumerate(mask):
        if not bool(enabled):
            continue
        total += float(
            np.sum(
                _robust_penalty_values(
                    arr[:, column_idx],
                    penalty=penalty,
                    huber_delta=huber_delta,
                    epsilon=epsilon,
                )
            )
        )
    return total


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
        temporal = _robust_penalty_sum(
            temporal_diffs,
            penalty=penalty,
            huber_delta=huber_delta,
            epsilon=epsilon,
            column_mask=roi_mask,
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
    if not _all_finite_values(arr):
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
    if not _indices_within_range(indices, upper=int(n_parameters)):
        raise ValueError("roi_mask indices are out of range.")
    mask = np.zeros(int(n_parameters), dtype=bool)
    mask[indices] = True
    return mask


def _indices_within_range(indices: np.ndarray, *, upper: int) -> bool:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return True
    return int(np.min(values)) >= 0 and int(np.max(values)) < int(upper)


def _robust_penalty_kind(value: str) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    if resolved not in {"huber", "tv"}:
        raise ValueError("penalty must be one of: huber, tv.")
    return resolved


def _weight_range(
    weights: np.ndarray,
    *,
    column_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    if column_mask is not None:
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("column_mask requires 2D weights.")
        mask = np.asarray(column_mask, dtype=bool).reshape(-1)
        if mask.size != matrix.shape[1]:
            raise ValueError("column_mask length must match weights columns.")
        if not bool(np.all(mask)):
            return _positive_range_masked_columns(matrix, mask)
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    min_val = np.inf
    max_val = 0.0
    count = 0
    for raw_value in np.nditer(arr, flags=["refs_ok"], op_flags=["readonly"]):
        value = float(raw_value)
        if value <= 0.0:
            continue
        count += 1
        if value < min_val:
            min_val = value
        if value > max_val:
            max_val = value
    if count == 0:
        return (0.0, 0.0)
    return (float(min_val), float(max_val))


def _positive_range_masked_columns(
    weights: np.ndarray,
    column_mask: np.ndarray,
) -> tuple[float, float]:
    min_val = np.inf
    max_val = 0.0
    count = 0
    for column_idx, enabled in enumerate(column_mask):
        if not bool(enabled):
            continue
        for raw_value in np.nditer(
            weights[:, column_idx],
            flags=["refs_ok"],
            op_flags=["readonly"],
        ):
            value = float(raw_value)
            if value <= 0.0:
                continue
            count += 1
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value
    if count == 0:
        return (0.0, 0.0)
    return (float(min_val), float(max_val))


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


def _kalman_observation_mode(value: str) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    aliases = {
        "j": "jacobian",
        "measurement": "jacobian",
        "measurement_jacobian": "jacobian",
        "rm": "rm_observation",
        "rm_observation_shortcut": "rm_observation",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"jacobian", "rm_observation"}:
        raise ValueError("observation_mode must be 'jacobian' or 'rm_observation'.")
    return resolved


def _kalman_observations(
    observation_model: Any,
    observations: np.ndarray,
    *,
    mode: str,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> tuple[np.ndarray, list[np.ndarray], list[str], list[int]]:
    n_frames, n_raw = observations.shape
    if mode == "rm_observation":
        rm = _rm_matrix(observation_model, n_measurements=n_raw)
        projected_rows = [
            reconstruct_difference(
                rm,
                observations[idx],
                normalize=False,
                channel_mask=_frame_channel_mask(
                    channel_mask,
                    frame_idx=idx,
                    n_frames=n_frames,
                    n_measurements=n_raw,
                ),
                measurement_weights=_frame_measurement_weights(
                    measurement_weights,
                    frame_idx=idx,
                    n_frames=n_frames,
                    n_measurements=n_raw,
                ),
                device="cpu",
            )
            for idx in range(n_frames)
        ]
        projected = _stack_row_vectors_direct(
            projected_rows,
            dtype=np.float64,
            name="projected_rows",
        )
        identity_observation = _dense_identity(rm.shape[0])
        h_stack = [identity_observation] * n_frames
        return projected, h_stack, ["state_observation"] * n_frames, [0] * n_frames

    h_raw_stack, _ = _as_jacobian_stack(
        observation_model,
        n_frames=n_frames,
        n_measurements=n_raw,
    )
    projected_rows: list[np.ndarray] = []
    h_stack: list[np.ndarray] = []
    contract_kinds: list[str] = []
    bad_counts: list[int] = []
    for frame_idx, matrix in enumerate(h_raw_stack):
        mask_t = _frame_channel_mask(
            channel_mask,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_raw,
        )
        weights_t = _frame_measurement_weights(
            measurement_weights,
            frame_idx=frame_idx,
            n_frames=n_frames,
            n_measurements=n_raw,
        )
        h_t, contract = apply_measurement_contract_to_jacobian(
            matrix,
            channel_mask=mask_t,
            measurement_weights=weights_t,
        )
        y_t, _ = apply_measurement_contract_to_vector(
            observations[frame_idx],
            channel_mask=mask_t,
            measurement_weights=weights_t,
        )
        h_stack.append(h_t)
        projected_rows.append(y_t)
        contract_kinds.append(contract.weight_kind)
        bad_counts.append(contract.bad_channel_count)
    return (
        _stack_row_vectors_direct(
            projected_rows,
            dtype=np.float64,
            name="projected_rows",
        ),
        h_stack,
        contract_kinds,
        bad_counts,
    )


def _rm_matrix(value: Any, *, n_measurements: int) -> np.ndarray:
    if sparse.issparse(value):
        matrix = np.asarray(value.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("RM observation model must be a non-empty 2D matrix.")
    if matrix.shape[1] != int(n_measurements):
        raise ValueError(
            f"RM column count {matrix.shape[1]} does not match observation length {n_measurements}."
        )
    if not _all_finite_values(matrix):
        raise FloatingPointError("RM observation model contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _transition_matrix(value: Any | None, *, n_state: int) -> np.ndarray:
    if value is None:
        matrix = _dense_identity(n_state)
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (n_state, n_state):
        raise ValueError(
            f"transition shape {matrix.shape} does not match {(n_state, n_state)}."
        )
    if not _all_finite_values(matrix):
        raise FloatingPointError("transition contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _kalman_initial_state(value: Any | None, *, n_state: int) -> np.ndarray:
    if value is None:
        return np.zeros(n_state, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != int(n_state):
        raise ValueError(f"initial_state length {arr.size} does not match {n_state}.")
    if not _all_finite_values(arr):
        raise FloatingPointError("initial_state contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _kalman_covariance(
    value: Any | None,
    *,
    n: int,
    name: str,
    default_scale: float,
) -> np.ndarray:
    if value is None:
        matrix = _dense_identity(n, scale=float(default_scale))
    else:
        matrix = _covariance_matrix(value, n=n, name=name)
    return matrix


def _resolve_kalman_noise(
    value: Any | None,
    *,
    hook: Any | None,
    n: int,
    frame_idx: int,
    state: np.ndarray,
    observation: np.ndarray,
    default_scale: float,
    name: str,
) -> tuple[np.ndarray, str]:
    if hook is not None:
        raw = hook(
            MappingProxyType(
                {
                    "frame_index": int(frame_idx),
                    "state": state.copy(),
                    "observation": observation.copy(),
                    "noise_name": name,
                }
            )
        )
        return _covariance_matrix(raw, n=n, name=name), "hook"
    if value is None:
        return _dense_identity(n, scale=float(default_scale)), "default"
    return _frame_covariance(value, frame_idx=frame_idx, n=n, name=name), "provided"


def _resolve_measurement_noise(
    value: Any | None,
    *,
    hook: Any | None,
    n_observations: int,
    frame_idx: int,
    state: np.ndarray,
    observation: np.ndarray,
    default_scale: float,
    mode: str,
) -> tuple[np.ndarray, str]:
    source_name = "measurement_noise"
    if hook is not None:
        raw = hook(
            MappingProxyType(
                {
                    "frame_index": int(frame_idx),
                    "state": state.copy(),
                    "observation": observation.copy(),
                    "noise_name": source_name,
                    "observation_mode": mode,
                }
            )
        )
        return (
            _covariance_matrix(raw, n=n_observations, name=source_name),
            "hook",
        )
    if value is None:
        return _dense_identity(n_observations, scale=float(default_scale)), "default"
    return (
        _frame_covariance(
            value,
            frame_idx=frame_idx,
            n=n_observations,
            name=source_name,
        ),
        "provided",
    )


def _frame_covariance(
    value: Any,
    *,
    frame_idx: int,
    n: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[1:] != (n, n):
            raise ValueError(
                f"{name} per-frame shape {arr.shape} does not match (*,{n},{n})."
            )
        return _covariance_matrix(arr[int(frame_idx)], n=n, name=name)
    if arr.ndim == 2 and arr.shape[0] != arr.shape[1] and arr.shape[1] == n:
        if int(frame_idx) >= arr.shape[0]:
            raise ValueError(f"{name} frame index {frame_idx} out of range.")
        return _covariance_matrix(arr[int(frame_idx)], n=n, name=name)
    return _covariance_matrix(arr, n=n, name=name)


def _covariance_matrix(value: Any, *, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        matrix = _dense_identity(n, scale=float(arr))
    elif arr.ndim == 1:
        if arr.size != int(n):
            raise ValueError(f"{name} diagonal length {arr.size} does not match {n}.")
        matrix = _dense_diagonal(arr)
    elif arr.ndim == 2:
        if arr.shape != (n, n):
            raise ValueError(
                f"{name} matrix shape {arr.shape} does not match {(n, n)}."
            )
        matrix = arr
    else:
        raise ValueError(f"{name} must be scalar, diagonal, or covariance matrix.")
    if not _all_finite_values(matrix):
        raise FloatingPointError(f"{name} contains non-finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} covariance matrix must be symmetric.")
    eig_min = float(np.min(np.linalg.eigvalsh(matrix)))
    if eig_min < -1.0e-10:
        raise ValueError(f"{name} covariance matrix must be positive semidefinite.")
    return _symmetrize(matrix)


def _kalman_gain(
    p_pred: np.ndarray,
    h_t: np.ndarray,
    innovation_covariance: np.ndarray,
) -> np.ndarray:
    rhs = h_t @ p_pred
    try:
        gain_t = np.linalg.solve(innovation_covariance, rhs)
    except np.linalg.LinAlgError:
        gain_t = np.linalg.pinv(innovation_covariance) @ rhs
    return np.asarray(gain_t.T, dtype=np.float64)


def _innovation_gate_policy(value: Any) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "hard": "reject",
        "hard_reject": "reject",
        "variance": "inflate",
        "variance_inflation": "inflate",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"none", "reject", "inflate"}:
        raise ValueError("innovation_gate must be 'none', 'reject', or 'inflate'.")
    return resolved


def _innovation_gate_candidate_frames(
    value: Any | None,
    *,
    n_frames: int,
) -> np.ndarray:
    if value is None:
        return np.zeros(int(n_frames), dtype=bool)
    candidates = np.asarray(value, dtype=bool).reshape(-1)
    if candidates.size != int(n_frames):
        raise ValueError(
            "innovation_gate_candidates length must match observation frames."
        )
    return np.ascontiguousarray(candidates, dtype=bool)


def _innovation_nis_threshold(
    value: float | None,
    *,
    gate_policy: str,
) -> float:
    if gate_policy == "none":
        return float("inf")
    if value is None:
        raise ValueError(
            "innovation_nis_threshold is required when innovation_gate is enabled."
        )
    return _positive_finite_float(value, name="innovation_nis_threshold")


def _positive_finite_float(
    value: Any,
    *,
    name: str,
    minimum: float = 0.0,
) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0 or resolved < float(minimum):
        comparator = f">= {minimum:g}" if minimum > 0.0 else "> 0"
        raise ValueError(f"{name} must be finite and {comparator}.")
    return resolved


def _normalized_innovation_squared(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    try:
        whitened = np.linalg.solve(innovation_covariance, innovation)
    except np.linalg.LinAlgError:
        whitened = np.linalg.pinv(innovation_covariance) @ innovation
    value = float(innovation @ whitened)
    if not np.isfinite(value):
        raise FloatingPointError("innovation NIS is non-finite.")
    return max(0.0, value)


def _fixed_lag_smoother(
    filtered_states: list[np.ndarray],
    filtered_covs: list[np.ndarray],
    predicted_covs: list[np.ndarray],
    *,
    transition_matrix: np.ndarray,
    fixed_lag: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_frames = len(filtered_states)
    if fixed_lag <= 0 or n_frames == 0:
        return _stack_row_vectors_direct(
            filtered_states,
            dtype=np.float64,
            name="filtered_states",
        ), {
            "enabled": False,
            "fixed_lag": int(fixed_lag),
        }
    smoothed = [state.copy() for state in filtered_states]
    window_count = 0
    for end in range(n_frames):
        start = max(0, end - int(fixed_lag))
        x_window = [state.copy() for state in filtered_states[start : end + 1]]
        p_window = [cov.copy() for cov in filtered_covs[start : end + 1]]
        for local in range(len(x_window) - 2, -1, -1):
            global_next = start + local + 1
            p_pred_next = predicted_covs[global_next]
            gain = _smoother_gain(
                p_window[local],
                transition_matrix,
                p_pred_next,
            )
            x_window[local] = x_window[local] + gain @ (
                x_window[local + 1] - transition_matrix @ x_window[local]
            )
            p_window[local] = _symmetrize(
                p_window[local] + gain @ (p_window[local + 1] - p_pred_next) @ gain.T
            )
        for offset, state in enumerate(x_window):
            smoothed[start + offset] = np.ascontiguousarray(state, dtype=np.float64)
        window_count += 1
    return _stack_row_vectors_direct(
        smoothed,
        dtype=np.float64,
        name="smoothed_states",
    ), {
        "enabled": True,
        "fixed_lag": int(fixed_lag),
        "window_count": int(window_count),
        "policy": "online_fixed_lag_rts_windows",
    }


def _smoother_gain(
    p_filt: np.ndarray,
    transition_matrix: np.ndarray,
    p_pred_next: np.ndarray,
) -> np.ndarray:
    rhs = transition_matrix @ p_filt
    try:
        solved = np.linalg.solve(p_pred_next, rhs)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(p_pred_next) @ rhs
    return np.asarray(solved.T, dtype=np.float64)


def _optional_timestamps(value: Any | None, *, n_frames: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != int(n_frames):
        raise ValueError(f"timestamps length {arr.size} does not match {n_frames}.")
    if not _all_finite_values(arr):
        raise FloatingPointError("timestamps contain non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _latency_seconds(
    fixed_lag: int,
    *,
    timestamps: np.ndarray | None,
    sampling_rate_hz: float | None,
) -> float:
    lag = int(fixed_lag)
    if lag <= 0:
        return 0.0
    if timestamps is not None and timestamps.size > 1:
        dt = float(np.median(np.diff(timestamps)))
        return float(max(dt, 0.0) * lag)
    if sampling_rate_hz is not None and float(sampling_rate_hz) > 0.0:
        return float(lag / float(sampling_rate_hz))
    return 0.0


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    out = 0.5 * (
        np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
    )
    return np.ascontiguousarray(out, dtype=np.float64)


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
    if not _all_finite_values(array):
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
    if matrix.nnz and not _all_finite_values(matrix.data):
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
                if _all_finite_values(out):
                    return np.ascontiguousarray(out), "spsolve"
            except (spla.MatrixRankWarning, RuntimeError, ValueError):
                pass
    lsmr = spla.lsmr(normal, rhs)
    out = np.asarray(lsmr[0], dtype=np.float64).reshape(-1)
    if not _all_finite_values(out):
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
    baseline = _stack_row_vectors_direct(
        rows,
        dtype=np.float64,
        name="rowwise_rm_baseline_rows",
    )
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
    if not _all_finite_values(arr):
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


def _nonnegative_int(value: int, *, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative.")
    return out


__all__ = [
    "DYNAMIC_KALMAN_SCHEMA",
    "SPATIOTEMPORAL_GN_SCHEMA",
    "SPATIOTEMPORAL_TV_HUBER_SCHEMA",
    "DynamicKalmanResult",
    "SpatiotemporalGNResult",
    "SpatiotemporalTVHuberResult",
    "run_dynamic_kalman_filter",
    "solve_batch_spatiotemporal_gn",
    "solve_spatiotemporal_tv_huber",
    "temporal_difference_operator",
]
