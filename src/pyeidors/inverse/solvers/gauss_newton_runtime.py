"""Runtime iteration helpers for the Gauss-Newton solver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np
import torch
from dolfinx import fem
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

try:  # pragma: no cover - optional dependency
    import pyamg
except Exception:  # pragma: no cover
    pyamg = None

try:  # pragma: no cover - optional dependency
    from sksparse.cholmod import cholesky as cholmod_cholesky
except Exception:  # pragma: no cover
    cholmod_cholesky = None

from ...data.difference import (
    normalize_difference_mode,
    normalize_difference_orientation,
    project_measurement_jacobian,
    project_measurement_vector,
)
from ...data.structures import EITImage
from ...femx import function_get_array, function_set_array
from ...perf.capabilities import (
    detect_performance_capabilities,
    select_fast_linear_path,
    select_fused_strategy,
    select_preconditioner,
)
from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
    rom_signature,
)
from ...utils.numeric_ops import safe_dot
from ..reduced.inexact_controller import InexactController
from ..reduced.lowrank_subspace import build_lowrank_subspace
from ..reduced.pod_basis import compute_pod_basis, merge_orthonormal_bases
from ..reduced.reduced_gn_step import build_reduced_operator, solve_reduced_step
from ..reduced.snapshot_bank import SnapshotBank, select_snapshot_matrix
from ..contracts import SolverOutput
from .gauss_newton_weights import build_weight_reference


@dataclass(slots=True)
class _IterationLog:
    iteration: int
    residual: float
    residual_weighted: float
    relative_residual: float
    relative_residual_weighted: float | None
    residual_max: float
    meas_norm: float
    pred_norm: float
    meas_max: float
    pred_max: float
    jtr_norm: float
    delta_norm: float
    step: float
    lambda_eff: float
    relative_change: float
    res_drop: float | None
    meas_misfit: float
    prior_misfit: float
    total_objective: float

    def to_payload(self) -> dict[str, float | int | None]:
        return {
            "iteration": self.iteration,
            "residual": self.residual,
            "residual_weighted": self.residual_weighted,
            "relative_residual": self.relative_residual,
            "relative_residual_weighted": self.relative_residual_weighted,
            "residual_max": self.residual_max,
            "meas_norm": self.meas_norm,
            "pred_norm": self.pred_norm,
            "meas_max": self.meas_max,
            "pred_max": self.pred_max,
            "JTr_norm": self.jtr_norm,
            "delta_norm": self.delta_norm,
            "step": self.step,
            "lambda_eff": self.lambda_eff,
            "relative_change": self.relative_change,
            "res_drop": self.res_drop,
            "meas_misfit": self.meas_misfit,
            "prior_misfit": self.prior_misfit,
            "total_objective": self.total_objective,
        }


def _to_runtime_tensor(reconstructor, values) -> torch.Tensor:
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(
        values,
        device=reconstructor.device,
        dtype=reconstructor._torch_dtype,
    )


def _to_runtime_tensor_cached(
    reconstructor,
    name: str,
    values,
) -> torch.Tensor:
    """Reuse per-iteration tensor buffers to reduce repeated allocations."""
    source = _to_runtime_tensor(reconstructor, values)
    cache = getattr(reconstructor, "_runtime_tensor_cache", None)
    if cache is None:
        cache = {}
        reconstructor._runtime_tensor_cache = cache
    target = cache.get(name)
    if target is None or tuple(target.shape) != tuple(source.shape):
        target = source.clone()
        cache[name] = target
        return target
    target.copy_(source)
    return target


def ensure_measurement_weights(reconstructor, sigma_function: fem.Function) -> None:
    """Compute measurement weights based on baseline forward solution."""
    strategy = reconstructor.measurement_weight_strategy
    if not reconstructor.use_measurement_weights or strategy == "none":
        reconstructor._meas_weight_sqrt = None
        reconstructor._baseline_measurement = None
        return

    img = EITImage(elem_data=function_get_array(sigma_function), fwd_model=reconstructor.fwd_model)
    baseline_data, _ = reconstructor.fwd_model.fwd_solve(img)
    baseline_vector = project_measurement_vector(
        baseline_data.meas, **_measurement_space_kwargs(reconstructor)
    )
    reconstructor._baseline_measurement = baseline_vector.copy()

    reference_vector = build_weight_reference(
        strategy=strategy,
        baseline_vector=baseline_vector,
        measured_vector=reconstructor._measured_vector,
        floor=reconstructor.weight_floor,
    )

    weights = reference_vector ** 2
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, reconstructor.weight_floor)
    median = np.median(weights)
    if median > 0:
        weights = weights / median

    reconstructor._meas_weight_sqrt = _to_runtime_tensor(
        reconstructor,
        np.sqrt(weights),
    )
    if reconstructor.verbose:
        finite_weights = weights[np.isfinite(weights)]
        w_min = finite_weights.min() if finite_weights.size else float("nan")
        w_max = finite_weights.max() if finite_weights.size else float("nan")
        w_med = np.median(finite_weights) if finite_weights.size else float("nan")
        print(
            f"[INFO] measurement weights ({strategy}): min={w_min:.3e}, med={w_med:.3e}, max={w_max:.3e}"
        )


def _finite_summary(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "finite_count=0"
    return (
        f"finite_count={finite.size} "
        f"min={float(finite.min()):.6e} "
        f"max={float(finite.max()):.6e} "
        f"l2={float(np.linalg.norm(finite)):.6e}"
    )


def _require_finite(name: str, values, iteration: int | None = None) -> None:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if np.isfinite(arr).all():
        return
    iter_tag = "init" if iteration is None else str(iteration)
    raise FloatingPointError(
        f"Non-finite values detected in {name} at iteration={iter_tag}. "
        f"{_finite_summary(arr.astype(np.float64, copy=False))}"
    )


def _require_scalar_finite(name: str, value: float, iteration: int | None = None) -> None:
    if np.isfinite(float(value)):
        return
    iter_tag = "init" if iteration is None else str(iteration)
    raise FloatingPointError(
        f"Non-finite scalar detected in {name} at iteration={iter_tag}: {value!r}"
    )


def _apply_regularization_np(reconstructor, vector: np.ndarray) -> np.ndarray:
    matrix = getattr(reconstructor, "R_matrix", None)
    if matrix is None:
        raise RuntimeError("Regularization matrix is not initialized.")

    vec = np.asarray(vector, dtype=np.float64)
    if isspmatrix(matrix):
        return np.asarray(matrix.dot(vec), dtype=np.float64)
    if isinstance(matrix, LinearOperator):
        return np.asarray(matrix.matvec(vec), dtype=np.float64)
    dense = np.asarray(matrix, dtype=np.float64)
    return np.asarray(dense @ vec, dtype=np.float64)


def _diag_preconditioner(reconstructor, J_weighted_np: np.ndarray, lambda_eff: float) -> np.ndarray:
    diag_h = np.sum(J_weighted_np * J_weighted_np, axis=0).astype(np.float64)
    reg_diag = getattr(reconstructor, "R_diag", None)
    if reg_diag is not None and reg_diag.shape[0] == diag_h.shape[0]:
        diag_h = diag_h + float(lambda_eff) * reg_diag
    else:
        diag_h = diag_h + float(lambda_eff)
    diag_h = np.maximum(diag_h, 1e-12)
    return np.asarray(diag_h, dtype=np.float64)


def _solve_linear_system_fast(
    reconstructor,
    *,
    J_weighted_np: np.ndarray,
    weighted_residual_np: np.ndarray,
    de_current_np: np.ndarray,
    lambda_eff: float,
    iteration: int,
) -> tuple[np.ndarray, float, float]:
    """Matrix-free fast solve with Woodbury/PCG fallback chain.

    Solve ``(J'J + lambda*R) delta = -g`` without constructing a dense
    ``n x n`` Hessian in automatic fast mode.
    """
    J_weighted_np = np.asarray(J_weighted_np, dtype=np.float64)
    weighted_residual_np = np.asarray(weighted_residual_np, dtype=np.float64)
    de_current_np = np.asarray(de_current_np, dtype=np.float64)

    jtr = np.asarray(
        safe_dot(J_weighted_np.T, weighted_residual_np, "gauss_newton.fast.jtr"),
        dtype=np.float64,
    )
    rhs = -jtr
    if reconstructor.use_prior_term:
        rhs = rhs - float(lambda_eff) * _apply_regularization_np(reconstructor, de_current_np)
    _require_finite("rhs_fast", rhs, iteration)

    n_param = int(J_weighted_np.shape[1])
    n_meas = int(J_weighted_np.shape[0])

    def _matvec(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64)
        projected = np.asarray(
            safe_dot(J_weighted_np, vv, "gauss_newton.fast.hv.project"),
            dtype=np.float64,
        )
        back_projected = np.asarray(
            safe_dot(J_weighted_np.T, projected, "gauss_newton.fast.hv.back_project"),
            dtype=np.float64,
        )
        return np.asarray(
            back_projected + float(lambda_eff) * _apply_regularization_np(reconstructor, vv),
            dtype=np.float64,
        )

    h_op = LinearOperator(
        (n_param, n_param),
        matvec=_matvec,
        rmatvec=_matvec,
        dtype=np.float64,
    )

    diag_precond = _diag_preconditioner(reconstructor, J_weighted_np, lambda_eff)
    diag_inv_op = LinearOperator(
        (n_param, n_param),
        matvec=lambda x: np.asarray(x, dtype=np.float64) / diag_precond,
        dtype=np.float64,
    )

    solver_mode = str(getattr(reconstructor, "linear_solver", "auto")).strip().lower()
    preconditioner_mode = str(getattr(reconstructor, "preconditioner", "auto")).strip().lower()
    capabilities = detect_performance_capabilities()
    resolved_preconditioner = select_preconditioner(
        preconditioner_mode,
        capabilities=capabilities,
    )
    fast_linear_mode = str(getattr(reconstructor, "fast_linear_path", "auto")).strip().lower()

    cg_rtol = 1e-8 if reconstructor.performance_mode == "aggressive" else 1e-10
    cg_maxiter = max(200, min(2500, n_param))

    fallback_reasons: list[str] = []
    fast_linear_path_reason = "explicit"

    def _add_fallback(reason: str | None) -> None:
        token = "" if reason is None else str(reason).strip()
        if token and token not in fallback_reasons:
            fallback_reasons.append(token)

    def _set_fast_meta(
        *,
        path: str,
        resolved_precond: str,
        reason: str | None = None,
        selected_path: str,
        path_reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        meta_payload: dict[str, object] = {
            "path": path,
            "resolved_preconditioner": resolved_precond,
            "fallback_reason": reason,
            "fast_linear_path_selected": selected_path,
            "fast_linear_path_reason": path_reason,
        }
        if isinstance(extra, dict):
            meta_payload.update(extra)
        reconstructor._last_fast_linear_meta = meta_payload

    def _regularization_signature() -> str:
        reg = getattr(reconstructor, "R_matrix", None)
        if reg is None:
            return "none"
        if isspmatrix(reg):
            mat = reg.tocsr()
            payload = {
                "shape": list(mat.shape),
                "indptr_hash": hashlib.sha256(
                    np.ascontiguousarray(mat.indptr, dtype=np.int64).tobytes()
                ).hexdigest(),
                "indices_hash": hashlib.sha256(
                    np.ascontiguousarray(mat.indices, dtype=np.int64).tobytes()
                ).hexdigest(),
                "data_hash": hashlib.sha256(
                    np.ascontiguousarray(mat.data, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        if isinstance(reg, LinearOperator):
            return f"linear_operator:{reg.shape[0]}x{reg.shape[1]}"
        dense = np.ascontiguousarray(np.asarray(reg, dtype=np.float64))
        return hashlib.sha256(dense.tobytes()).hexdigest()

    def _regularization_meta() -> dict[str, object]:
        cache = getattr(reconstructor, "_regularization_meta_cache", None)
        reg = getattr(reconstructor, "R_matrix", None)
        cache_token = (id(reg), n_param)
        if isinstance(cache, dict) and cache.get("token") == cache_token:
            return dict(cache["meta"])

        meta = {
            "is_diagonal": False,
            "is_sparse_spd": False,
            "diag_vector": None,
        }
        if reg is None:
            pass
        elif isspmatrix(reg):
            reg_csr = reg.tocsr()
            diag_vec = np.asarray(reg_csr.diagonal(), dtype=np.float64)
            coo = reg_csr.tocoo(copy=False)
            is_diag = bool(coo.nnz <= n_param and np.all(coo.row == coo.col))
            symmetric = False
            if reg_csr.shape[0] == reg_csr.shape[1]:
                diff = reg_csr - reg_csr.T
                symmetric = diff.nnz == 0 or np.max(np.abs(diff.data)) <= 1e-9
            positive_diag = diag_vec.size == n_param and np.all(diag_vec > 0)
            meta.update(
                {
                    "is_diagonal": is_diag,
                    "is_sparse_spd": bool(symmetric and positive_diag),
                    "diag_vector": diag_vec if diag_vec.size == n_param else None,
                }
            )
        elif isinstance(reg, LinearOperator):
            meta.update({"is_diagonal": False, "is_sparse_spd": False, "diag_vector": None})
        else:
            dense = np.asarray(reg, dtype=np.float64)
            diag_vec = np.asarray(np.diag(dense), dtype=np.float64)
            is_diag = bool(dense.ndim == 2 and dense.shape == (n_param, n_param))
            if is_diag:
                off_diag = dense - np.diag(diag_vec)
                is_diag = bool(np.all(np.abs(off_diag) <= 1e-12))
            symmetric = bool(dense.ndim == 2 and np.allclose(dense, dense.T, rtol=1e-8, atol=1e-12))
            positive_diag = diag_vec.size == n_param and np.all(diag_vec > 0)
            meta.update(
                {
                    "is_diagonal": is_diag,
                    "is_sparse_spd": bool(symmetric and positive_diag),
                    "diag_vector": diag_vec if diag_vec.size == n_param else None,
                }
            )

        reconstructor._regularization_meta_cache = {"token": cache_token, "meta": meta}
        return dict(meta)

    def _solve_linear_system_fast_woodbury_diag(diag_vector: np.ndarray) -> np.ndarray:
        diag_scaled = float(lambda_eff) * np.asarray(diag_vector, dtype=np.float64)
        diag_scaled = np.maximum(diag_scaled, 1e-12)
        inv_diag = 1.0 / diag_scaled

        u = inv_diag * rhs
        ja_inv = J_weighted_np * inv_diag[None, :]
        small_rhs = np.asarray(
            safe_dot(J_weighted_np, u, "gauss_newton.fast.woodbury.small_rhs"),
            dtype=np.float64,
        )
        s_matrix = np.eye(n_meas, dtype=np.float64) + np.asarray(
            safe_dot(ja_inv, J_weighted_np.T, "gauss_newton.fast.woodbury.small_system"),
            dtype=np.float64,
        )
        s_matrix = 0.5 * (s_matrix + s_matrix.T)
        jitter = 1e-12
        factor, lower = cho_factor(
            s_matrix + (jitter * np.eye(n_meas, dtype=np.float64)),
            overwrite_a=False,
            check_finite=False,
        )
        y = cho_solve((factor, lower), small_rhs, check_finite=False)
        correction = inv_diag * np.asarray(
            safe_dot(J_weighted_np.T, y, "gauss_newton.fast.woodbury.correction"),
            dtype=np.float64,
        )
        return np.asarray(u - correction, dtype=np.float64)

    def _build_cholmod_preconditioner_from_R() -> tuple[LinearOperator | None, str | None]:
        if cholmod_cholesky is None:
            return None, "cholmod_unavailable"

        reg = getattr(reconstructor, "R_matrix", None)
        if not isspmatrix(reg):
            return None, "regularization_not_sparse"

        max_n = int(getattr(reconstructor, "cholmod_max_n", 50000))
        if max_n > 0 and n_param > max_n:
            return None, "cholmod_n_limit"

        reg_csc = reg.tocsc().astype(np.float64)
        max_memory_gib = float(getattr(reconstructor, "cholmod_max_memory_gib", 4.0))
        estimated_bytes = float(reg_csc.nnz) * 24.0 + float(n_param) * 64.0
        if estimated_bytes > max_memory_gib * (1024.0**3):
            return None, "cholmod_memory_limit"

        shift = max(1e-12, abs(float(lambda_eff)) * 1e-12)
        precond_matrix = (float(lambda_eff) * reg_csc) + sparse.identity(
            n_param,
            format="csc",
            dtype=np.float64,
        ) * shift

        cache = getattr(reconstructor, "_cholmod_precond_factor_cache", None)
        if cache is None:
            cache = {}
            reconstructor._cholmod_precond_factor_cache = cache
        key_payload = {
            "reg_signature": _regularization_signature(),
            "lambda_eff": float(lambda_eff),
            "n_param": int(n_param),
            "max_n": int(max_n),
            "max_memory_gib": float(max_memory_gib),
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        factor = cache.get(key)
        if factor is None:
            try:
                factor = cholmod_cholesky(precond_matrix)
            except Exception as exc:  # pragma: no cover - guarded runtime fallback
                return None, f"cholmod_precond_failed:{type(exc).__name__}"
            cache[key] = factor

        operator = LinearOperator(
            (n_param, n_param),
            matvec=lambda x: np.asarray(factor.solve_A(np.asarray(x, dtype=np.float64)), dtype=np.float64),
            dtype=np.float64,
        )
        return operator, None

    def _solve_pcg(precond_choice: str) -> tuple[np.ndarray | None, str, str, str | None]:
        choice = str(precond_choice)
        m_op: LinearOperator | None = None
        path = "pcg-diag-precond"
        fallback: str | None = None

        if choice == "cholmod":
            m_op, reason = _build_cholmod_preconditioner_from_R()
            if m_op is None:
                fallback = reason or "cholmod_precond_unavailable"
                choice = "diag"
            else:
                path = "pcg-cholmod-precond"

        if choice == "pyamg":
            if pyamg is None:
                fallback = "pyamg_unavailable"
                choice = "diag"
            else:
                reg = getattr(reconstructor, "R_matrix", None)
                if isspmatrix(reg):
                    try:
                        amg_mat = reg.tocsr().astype(np.float64)
                        amg_mat = amg_mat + sparse.identity(amg_mat.shape[0], format="csr") * 1e-12
                        ml = pyamg.smoothed_aggregation_solver(amg_mat)
                        m_op = ml.aspreconditioner(cycle="V")
                        path = "pcg-pyamg-precond"
                    except Exception as exc:  # pragma: no cover - optional path
                        fallback = f"pyamg_precond_failed:{type(exc).__name__}"
                        choice = "diag"
                else:
                    fallback = "pyamg_requires_sparse_regularization"
                    choice = "diag"

        if choice in {"diag", "petsc-gamg"}:
            m_op = diag_inv_op
            path = "pcg-diag-precond"
            if choice == "petsc-gamg":
                fallback = "petsc_gamg_not_supported_in_matrix_free"

        delta, info = cg(h_op, rhs, M=m_op, rtol=cg_rtol, maxiter=cg_maxiter)
        if info != 0:
            return None, path, choice, "pcg_not_converged"
        return np.asarray(delta, dtype=np.float64), path, choice, fallback

    def _solve_cholmod_direct_debug() -> tuple[np.ndarray | None, str | None]:
        if cholmod_cholesky is None:
            return None, "cholmod_unavailable"
        reg = getattr(reconstructor, "R_matrix", None)
        if not isspmatrix(reg):
            return None, "regularization_not_sparse"
        max_n = int(getattr(reconstructor, "cholmod_max_n", 50000))
        if max_n > 0 and n_param > max_n:
            return None, "cholmod_n_limit"
        max_memory_gib = float(getattr(reconstructor, "cholmod_max_memory_gib", 4.0))
        estimated_bytes = 2.5 * float(n_param) * float(n_param) * 8.0
        if estimated_bytes > max_memory_gib * (1024.0**3):
            return None, "cholmod_memory_limit"
        try:
            h_sparse = sparse.csc_matrix(
                safe_dot(J_weighted_np.T, J_weighted_np, "gauss_newton.fast.cholmod_direct.jtj")
            )
            h_sparse = h_sparse + float(lambda_eff) * reg.tocsc()
            factor = cholmod_cholesky(h_sparse)
            return np.asarray(factor.solve_A(rhs), dtype=np.float64), None
        except Exception as exc:  # pragma: no cover - guarded fallback
            return None, f"cholmod_direct_failed:{type(exc).__name__}"

    reg_meta = _regularization_meta()
    regularization_is_diagonal = bool(reg_meta.get("is_diagonal", False))
    regularization_is_sparse_spd = bool(reg_meta.get("is_sparse_spd", False))
    mesh_dim = 2
    if hasattr(reconstructor, "fwd_model"):
        try:
            mesh_dim = int(getattr(reconstructor.fwd_model.mesh.geometry, "dim", 2))
        except Exception:
            mesh_dim = 2
    fused_strategy = select_fused_strategy(
        solver_mode=str(getattr(reconstructor, "solver_mode", "strict")),
        mesh_dim=mesh_dim,
        n_param=n_param,
        n_meas=n_meas,
        rom_mode=str(getattr(reconstructor, "rom_mode", "auto")),
        inexact_mode=str(getattr(reconstructor, "inexact_mode", "auto")),
        lowrank_mode=str(getattr(reconstructor, "lowrank_mode", "auto")),
        regularization_is_diagonal=regularization_is_diagonal,
        capabilities=capabilities,
    )

    def _effective_feature_mode(mode_value: str) -> str:
        mode_norm = str(mode_value).strip().lower()
        if mode_norm in {"off", "on", "auto"}:
            return mode_norm
        return "off"

    rom_mode_effective = _effective_feature_mode(str(getattr(reconstructor, "rom_mode", "auto")))
    inexact_mode_effective = _effective_feature_mode(str(getattr(reconstructor, "inexact_mode", "auto")))
    lowrank_mode_effective = _effective_feature_mode(str(getattr(reconstructor, "lowrank_mode", "auto")))

    def _snapshot_bank() -> SnapshotBank:
        bank = getattr(reconstructor, "_rom_snapshot_bank", None)
        if not isinstance(bank, SnapshotBank):
            bank = SnapshotBank(max_snapshots=32, normalize=True)
            reconstructor._rom_snapshot_bank = bank
        return bank

    def _inexact_controller() -> InexactController:
        controller = getattr(reconstructor, "_inexact_controller", None)
        mode = str(getattr(reconstructor, "inexact_forcing", "eisenstat-walker"))
        eta0 = float(getattr(reconstructor, "inexact_eta0", 0.2))
        eta_min = float(getattr(reconstructor, "inexact_eta_min", 1e-3))
        eta_max = float(getattr(reconstructor, "inexact_eta_max", 0.5))
        token = (mode, eta0, eta_min, eta_max)
        cached_token = getattr(reconstructor, "_inexact_controller_token", None)
        if not isinstance(controller, InexactController) or cached_token != token:
            controller = InexactController(
                mode=mode,
                eta0=eta0,
                eta_min=eta_min,
                eta_max=eta_max,
            )
            reconstructor._inexact_controller = controller
            reconstructor._inexact_controller_token = token
        return controller

    def _synthetic_snapshots(diag_vector: np.ndarray | None) -> np.ndarray:
        snapshots: list[np.ndarray] = []
        snapshots.append(np.asarray(rhs, dtype=np.float64))
        snapshots.append(np.asarray(de_current_np, dtype=np.float64))
        grad_proxy = np.asarray(
            safe_dot(J_weighted_np.T, weighted_residual_np, "gauss_newton.fused.grad_proxy"),
            dtype=np.float64,
        )
        snapshots.append(grad_proxy)
        if isinstance(diag_vector, np.ndarray) and diag_vector.shape[0] == n_param:
            inv_diag = 1.0 / np.maximum(float(lambda_eff) * diag_vector, 1e-12)
            snapshots.append(inv_diag * rhs)
        for idx in range(min(6, n_meas)):
            snapshots.append(np.asarray(J_weighted_np[idx, :], dtype=np.float64))
        return np.ascontiguousarray(np.column_stack(snapshots), dtype=np.float64)

    def _build_global_basis(diag_vector: np.ndarray | None) -> tuple[np.ndarray, str]:
        source = str(getattr(reconstructor, "rom_snapshot_source", "hybrid")).strip().lower()
        rank_global = int(max(1, getattr(reconstructor, "rom_rank_global", 32)))
        synthetic = _synthetic_snapshots(diag_vector)
        bank = _snapshot_bank()
        bank.add(rhs)
        bank.add(de_current_np)
        cached_matrix = np.zeros((n_param, 0), dtype=np.float64)
        cache_manager = getattr(reconstructor, "cache_manager", None)
        lookup_layer = "compute"
        if cache_manager is not None and bool(getattr(cache_manager, "enabled", False)):
            snapshot_payload = {
                "solver": "gn_absolute",
                "artifact": "rom_snapshot_bank",
                "model_signature": model_signature_from_forward_model(reconstructor.fwd_model),
                "pattern_signature": pattern_signature_from_forward_model(reconstructor.fwd_model),
                "backend_signature": backend_signature_from_forward_model(reconstructor.fwd_model),
                "n_param": int(n_param),
                "source": source,
                "rank_global": int(rank_global),
            }
            cached_matrix, lookup = cache_manager.get_or_compute_semantic(
                artifact="rom_snapshot_bank",
                name="absolute_rom_snapshot_bank",
                namespace="absolute",
                cache_obj=snapshot_payload,
                payload=snapshot_payload,
                compute_fn=lambda: synthetic,
                persist=True,
                cost=2.0,
                effort_seconds=0.5,
            )
            lookup_layer = str(getattr(lookup, "layer", "compute"))
        snapshot_matrix = select_snapshot_matrix(
            source,
            n_param=n_param,
            bank_matrix=bank.matrix(),
            synthetic_matrix=synthetic,
            cached_matrix=np.asarray(cached_matrix, dtype=np.float64),
        )
        if snapshot_matrix.shape[1] == 0:
            return np.zeros((n_param, 0), dtype=np.float64), lookup_layer

        basis_payload = {
            "solver": "gn_absolute",
            "artifact": "rom_global_basis",
            "model_signature": model_signature_from_forward_model(reconstructor.fwd_model),
            "pattern_signature": pattern_signature_from_forward_model(reconstructor.fwd_model),
            "backend_signature": backend_signature_from_forward_model(reconstructor.fwd_model),
            "snapshot_hash": hashlib.sha256(
                np.ascontiguousarray(snapshot_matrix, dtype=np.float64).tobytes()
            ).hexdigest(),
            "rank_global": int(rank_global),
            "source": source,
        }
        basis_payload["rom_signature"] = rom_signature(
            rank_global=int(rank_global),
            rank_adaptive=int(getattr(reconstructor, "rom_rank_adaptive", 16)),
            lowrank_rank=int(getattr(reconstructor, "lowrank_rank", 16)),
            lowrank_energy=float(getattr(reconstructor, "lowrank_energy", 0.995)),
            lowrank_method=str(getattr(reconstructor, "lowrank_method", "tsvd")),
            snapshot_source=source,
            snapshot_hash=str(basis_payload["snapshot_hash"]),
            refresh_every=int(getattr(reconstructor, "rom_refresh_every", 2)),
        )
        cache_manager = getattr(reconstructor, "cache_manager", None)
        if cache_manager is not None and bool(getattr(cache_manager, "enabled", False)):
            basis, lookup = cache_manager.get_or_compute_semantic(
                artifact="rom_global_basis",
                name="absolute_rom_global_basis",
                namespace="absolute",
                cache_obj=basis_payload,
                payload=basis_payload,
                compute_fn=lambda: compute_pod_basis(
                    snapshot_matrix,
                    rank=rank_global,
                    energy=float(getattr(reconstructor, "lowrank_energy", 0.995)),
                ),
                persist=True,
                cost=4.0,
                effort_seconds=1.0,
            )
            lookup_layer = str(getattr(lookup, "layer", lookup_layer))
            return np.asarray(basis, dtype=np.float64), lookup_layer

        return (
            compute_pod_basis(
                snapshot_matrix,
                rank=rank_global,
                energy=float(getattr(reconstructor, "lowrank_energy", 0.995)),
            ),
            lookup_layer,
        )

    def _adaptive_basis() -> tuple[np.ndarray, str]:
        refresh_every = int(max(1, getattr(reconstructor, "rom_refresh_every", 2)))
        rank_adaptive = int(max(0, getattr(reconstructor, "rom_rank_adaptive", 16)))
        lowrank_rank = int(max(1, getattr(reconstructor, "lowrank_rank", 16)))
        lowrank_method = str(getattr(reconstructor, "lowrank_method", "tsvd")).strip().lower()
        lowrank_energy = float(getattr(reconstructor, "lowrank_energy", 0.995))
        if rank_adaptive <= 0:
            return np.zeros((n_param, 0), dtype=np.float64), "disabled"

        if (iteration % refresh_every) != 0:
            cached_basis = getattr(reconstructor, "_rom_last_adaptive_basis", None)
            if isinstance(cached_basis, np.ndarray) and cached_basis.shape[0] == n_param:
                return np.asarray(cached_basis, dtype=np.float64), "reuse"

        payload = {
            "solver": "gn_absolute",
            "artifact": "rom_adaptive_basis",
            "jacobian_hash": hashlib.sha256(
                np.ascontiguousarray(J_weighted_np, dtype=np.float64).tobytes()
            ).hexdigest(),
            "rank_adaptive": int(rank_adaptive),
            "lowrank_rank": int(lowrank_rank),
            "lowrank_method": str(lowrank_method),
            "lowrank_energy": float(lowrank_energy),
        }
        cache_manager = getattr(reconstructor, "cache_manager", None)
        if cache_manager is not None and bool(getattr(cache_manager, "enabled", False)):
            basis, lookup = cache_manager.get_or_compute_semantic(
                artifact="rom_adaptive_basis",
                name="absolute_rom_adaptive_basis",
                namespace="absolute",
                cache_obj=payload,
                payload=payload,
                compute_fn=lambda: build_lowrank_subspace(
                    J_weighted_np,
                    rank=max(lowrank_rank, rank_adaptive),
                    energy=lowrank_energy,
                    method=lowrank_method,
                )[0][:, :rank_adaptive],
                persist=True,
                cost=4.0,
                effort_seconds=1.0,
            )
            basis_arr = np.asarray(basis, dtype=np.float64)
            reconstructor._rom_last_adaptive_basis = basis_arr
            return basis_arr, str(getattr(lookup, "layer", "compute"))

        basis_arr = build_lowrank_subspace(
            J_weighted_np,
            rank=max(lowrank_rank, rank_adaptive),
            energy=lowrank_energy,
            method=lowrank_method,
        )[0][:, :rank_adaptive]
        basis_arr = np.asarray(basis_arr, dtype=np.float64)
        reconstructor._rom_last_adaptive_basis = basis_arr
        return basis_arr, "compute"

    def _solve_linear_system_fused(diag_vector: np.ndarray | None) -> tuple[np.ndarray | None, dict[str, object]]:
        if rom_mode_effective == "off":
            return None, {"reason": "rom_off"}
        rom_mode_raw = str(getattr(reconstructor, "rom_mode", "auto")).strip().lower()
        if n_param < 5000 and rom_mode_raw != "on":
            return None, {"reason": "problem_too_small"}
        residual_limit = float(max(0.05, getattr(reconstructor, "rom_residual_limit", 0.6)))

        global_basis, global_source = _build_global_basis(diag_vector)
        adaptive_basis = np.zeros((n_param, 0), dtype=np.float64)
        adaptive_source = "disabled"
        if lowrank_mode_effective != "off" and bool(fused_strategy.get("lowrank", False)):
            adaptive_basis, adaptive_source = _adaptive_basis()

        combined_basis = merge_orthonormal_bases(
            global_basis,
            adaptive_basis,
            rank_cap=int(max(1, int(getattr(reconstructor, "rom_rank_global", 32)) + int(max(0, getattr(reconstructor, "rom_rank_adaptive", 16))))),
        )
        if combined_basis.size == 0 or combined_basis.shape[1] == 0:
            return None, {"reason": "empty_basis"}

        stage_attempts = []
        stage_attempts.append(
            {
                "name": "rom+inexact+lowrank",
                "use_lowrank": adaptive_basis.shape[1] > 0,
                "use_inexact": inexact_mode_effective != "off" and bool(fused_strategy.get("inexact", False)),
            }
        )
        stage_attempts.append(
            {
                "name": "rom+inexact",
                "use_lowrank": False,
                "use_inexact": inexact_mode_effective != "off" and bool(fused_strategy.get("inexact", False)),
            }
        )
        stage_attempts.append({"name": "rom", "use_lowrank": False, "use_inexact": False})

        errors: list[str] = []
        for stage in stage_attempts:
            if stage["name"] == "rom+inexact+lowrank" and adaptive_basis.shape[1] == 0:
                continue
            if stage["name"] == "rom+inexact" and not stage["use_inexact"]:
                continue
            stage_basis = merge_orthonormal_bases(
                global_basis,
                adaptive_basis if stage["use_lowrank"] else None,
                rank_cap=int(max(1, combined_basis.shape[1])),
            )
            if stage_basis.shape[1] == 0:
                errors.append(f"{stage['name']}:empty_basis")
                continue
            try:
                op_payload = {
                    "solver": "gn_absolute",
                    "artifact": "rom_reduced_operator_absolute",
                    "basis_hash": hashlib.sha256(
                        np.ascontiguousarray(stage_basis, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "jacobian_hash": hashlib.sha256(
                        np.ascontiguousarray(J_weighted_np, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "lambda_eff": float(lambda_eff),
                    "reg_signature": _regularization_signature(),
                    "stage": stage["name"],
                }
                cache_manager = getattr(reconstructor, "cache_manager", None)
                if cache_manager is not None and bool(getattr(cache_manager, "enabled", False)):
                    reduced_op, op_lookup = cache_manager.get_or_compute_semantic(
                        artifact="rom_reduced_operator_absolute",
                        name="absolute_rom_reduced_operator",
                        namespace="absolute",
                        cache_obj=op_payload,
                        payload=op_payload,
                        compute_fn=lambda: build_reduced_operator(
                            jacobian=J_weighted_np,
                            basis=stage_basis,
                            regularization_apply=lambda vec: _apply_regularization_np(reconstructor, vec),
                            lambda_eff=float(lambda_eff),
                        ),
                        persist=True,
                        cost=6.0,
                        effort_seconds=2.0,
                    )
                    op_source = str(getattr(op_lookup, "layer", "compute"))
                else:
                    reduced_op = build_reduced_operator(
                        jacobian=J_weighted_np,
                        basis=stage_basis,
                        regularization_apply=lambda vec: _apply_regularization_np(reconstructor, vec),
                        lambda_eff=float(lambda_eff),
                    )
                    op_source = "compute"

                controller = _inexact_controller()
                inexact_tol = None
                if stage["use_inexact"]:
                    inexact_tol = float(controller.suggest_eta())
                delta_candidate, solve_info = solve_reduced_step(
                    reduced_operator=reduced_op,
                    rhs=rhs,
                    inexact_tol=inexact_tol,
                    maxiter=max(50, stage_basis.shape[1] * 4),
                )
                if not np.isfinite(delta_candidate).all():
                    raise FloatingPointError("non_finite_delta")
                full_residual = np.asarray(_matvec(delta_candidate) - rhs, dtype=np.float64)
                full_residual_ratio = float(
                    np.linalg.norm(full_residual) / max(np.linalg.norm(rhs), 1e-12)
                )
                if full_residual_ratio > residual_limit:
                    raise RuntimeError("fused_residual_high")

                linear_residual_ratio = float(solve_info.get("linear_residual_ratio", 0.0))
                outer_prev_raw = getattr(reconstructor, "_outer_prev_residual", None)
                outer_prev = float(outer_prev_raw) if isinstance(outer_prev_raw, (int, float)) else None
                outer_curr = float(np.linalg.norm(weighted_residual_np))
                if stage["use_inexact"]:
                    controller.update(
                        outer_prev=outer_prev,
                        outer_curr=outer_curr,
                        linear_residual_ratio=linear_residual_ratio,
                        step_rejected=False,
                        stalled=False,
                    )
                    if linear_residual_ratio > max(controller.eta * 1.5, 5e-2):
                        reconstructor._force_jacobian_refresh = True

                _snapshot_bank().add(delta_candidate)
                meta = {
                    "stage": stage["name"],
                    "source": op_source,
                    "global_source": global_source,
                    "adaptive_source": adaptive_source,
                    "global_rank": int(global_basis.shape[1]),
                    "adaptive_rank": int(adaptive_basis.shape[1]),
                    "rank_effective": int(stage_basis.shape[1]),
                    "inexact_eta": float(_inexact_controller().eta),
                    "inexact_eta_history": list(_inexact_controller().history[-12:]),
                    "linear_residual_ratio": linear_residual_ratio,
                    "full_linear_residual_ratio": full_residual_ratio,
                    "degrade_reason": ";".join(errors) if errors else "",
                    "fast_solver_path": f"fused-{stage['name']}",
                }
                return np.asarray(delta_candidate, dtype=np.float64), meta
            except Exception as exc:
                detail = str(exc).strip().replace(";", ",")
                if detail:
                    errors.append(f"{stage['name']}:{type(exc).__name__}:{detail}")
                else:
                    errors.append(f"{stage['name']}:{type(exc).__name__}")
                continue

        return None, {"reason": ";".join(errors) if errors else "fused_failed"}

    selected_fast_path = select_fast_linear_path(
        fast_linear_mode,
        regularization_is_diagonal=regularization_is_diagonal,
        regularization_is_sparse_spd=regularization_is_sparse_spd,
        capabilities=capabilities,
    )

    if solver_mode == "cholmod" and fast_linear_mode == "auto":
        selected_fast_path = "cholmod-direct"
        fast_linear_path_reason = "auto:linear_solver_cholmod_direct"
    elif solver_mode == "pyamg-cg":
        selected_fast_path = "pcg"
        resolved_preconditioner = "pyamg"
        if fast_linear_mode == "auto":
            fast_linear_path_reason = "auto:linear_solver_pyamg"

    if fast_linear_mode == "auto":
        if selected_fast_path == "woodbury":
            fast_linear_path_reason = "auto:diagonal_regularization"
        elif selected_fast_path == "pcg" and regularization_is_sparse_spd and capabilities.get("cholmod", False):
            fast_linear_path_reason = "auto:sparse_spd_with_cholmod"
        elif selected_fast_path == "pcg":
            fast_linear_path_reason = "auto:matrix_free_pcg"
        else:
            fast_linear_path_reason = f"auto:{selected_fast_path}"

    delta_np: np.ndarray | None = None
    fast_solver_path = ""
    fused_diag = reg_meta.get("diag_vector")
    if bool(fused_strategy.get("enabled", False)):
        if (
            selected_fast_path == "woodbury"
            and regularization_is_diagonal
            and rom_mode_effective != "on"
        ):
            _add_fallback("fused_skipped:woodbury_optimal")
        else:
            fused_delta, fused_meta = _solve_linear_system_fused(
                fused_diag if isinstance(fused_diag, np.ndarray) else None
            )
            if fused_delta is not None:
                delta_np = np.asarray(fused_delta, dtype=np.float64)
                fast_solver_path = str(fused_meta.get("fast_solver_path", "fused-rom"))
                resolved_preconditioner = "reduced"
                selected_fast_path = "fused"
                fast_linear_path_reason = str(fused_strategy.get("reason", "fused_enabled"))
                fused_reason = str(fused_meta.get("degrade_reason", "")).strip()
                _set_fast_meta(
                    path=fast_solver_path,
                    resolved_precond=resolved_preconditioner,
                    reason=fused_reason if fused_reason else None,
                    selected_path=selected_fast_path,
                    path_reason=fast_linear_path_reason,
                    extra={
                        "rom_enabled_effective": True,
                        "rom_rank_effective": int(fused_meta.get("rank_effective", 0)),
                        "lowrank_rank_effective": int(fused_meta.get("adaptive_rank", 0)),
                        "inexact_eta": float(fused_meta.get("inexact_eta", 0.0)),
                        "inexact_eta_history": fused_meta.get("inexact_eta_history", []),
                        "degrade_stage": str(fused_meta.get("stage", "rom")),
                        "degrade_reason": str(fused_meta.get("degrade_reason", "")),
                        "effective_solver_path": fast_solver_path,
                        "fused_meta": fused_meta,
                    },
                )
            else:
                _add_fallback(f"fused_failed:{fused_meta.get('reason', 'unknown')}")

    if selected_fast_path == "strict" or solver_mode == "petsc-ksp":
        _set_fast_meta(
            path="strict-fallback",
            resolved_precond=resolved_preconditioner,
            reason="strict_requested",
            selected_path=selected_fast_path,
            path_reason=fast_linear_path_reason,
        )
        raise RuntimeError("fast_linear_path_requested_strict")

    if solver_mode == "scipy-lsmr":
        if delta_np is None:
            delta_np = np.asarray(
                lsmr(h_op, rhs, atol=cg_rtol, btol=cg_rtol, maxiter=cg_maxiter)[0],
                dtype=np.float64,
            )
            fast_solver_path = "lsmr-direct"
    else:
        if delta_np is None and selected_fast_path == "woodbury":
            diag_vector = reg_meta.get("diag_vector")
            if isinstance(diag_vector, np.ndarray) and diag_vector.shape[0] == n_param:
                try:
                    delta_np = _solve_linear_system_fast_woodbury_diag(diag_vector)
                    fast_solver_path = "woodbury-diag"
                    resolved_preconditioner = "woodbury"
                except Exception as exc:
                    _add_fallback(f"woodbury_failed:{type(exc).__name__}")
            else:
                _add_fallback("woodbury_requires_diagonal_regularization")

        if delta_np is None and selected_fast_path == "cholmod-direct":
            delta_np, reason = _solve_cholmod_direct_debug()
            if delta_np is not None:
                fast_solver_path = "cholmod-direct"
                resolved_preconditioner = "cholmod"
            else:
                _add_fallback(reason)

        if delta_np is None:
            pcg_delta, pcg_path, pcg_choice, pcg_reason = _solve_pcg(resolved_preconditioner)
            resolved_preconditioner = pcg_choice
            if pcg_reason:
                _add_fallback(pcg_reason)
            if pcg_delta is not None:
                delta_np = pcg_delta
                fast_solver_path = pcg_path

        if delta_np is None:
            try:
                delta_np = np.asarray(
                    lsmr(h_op, rhs, atol=1e-7, btol=1e-7, maxiter=cg_maxiter * 2)[0],
                    dtype=np.float64,
                )
                fast_solver_path = "lsmr-fallback"
            except Exception as exc:
                _add_fallback(f"lsmr_failed:{type(exc).__name__}")
                _set_fast_meta(
                    path="strict-fallback",
                    resolved_precond=resolved_preconditioner,
                    reason=";".join(fallback_reasons) if fallback_reasons else "fast_linear_failed",
                    selected_path=selected_fast_path,
                    path_reason=fast_linear_path_reason,
                )
                raise RuntimeError("fast_linear_solver_failed") from exc

    _require_finite("delta_sigma_fast", delta_np, iteration)
    delta_norm = float(np.linalg.norm(delta_np))
    _require_scalar_finite("delta_norm_fast", delta_norm, iteration)
    jtr_norm = float(np.linalg.norm(jtr))
    _require_scalar_finite("jtr_norm_fast", jtr_norm, iteration)

    existing_meta = getattr(reconstructor, "_last_fast_linear_meta", {})
    passthrough_extra = {}
    if isinstance(existing_meta, dict):
        for key, value in existing_meta.items():
            if key not in {
                "path",
                "resolved_preconditioner",
                "fallback_reason",
                "fast_linear_path_selected",
                "fast_linear_path_reason",
            }:
                passthrough_extra[key] = value

    _set_fast_meta(
        path=fast_solver_path,
        resolved_precond=resolved_preconditioner,
        reason=";".join(fallback_reasons) if fallback_reasons else None,
        selected_path=selected_fast_path,
        path_reason=fast_linear_path_reason,
        extra=passthrough_extra,
    )
    return delta_np, delta_norm, jtr_norm


def _extract_measured_vector(measured_data) -> np.ndarray:
    if hasattr(measured_data, "meas"):
        return np.asarray(measured_data.meas, dtype=np.float64).reshape(-1)
    return np.asarray(measured_data, dtype=np.float64).reshape(-1)


def _configure_measurement_space(reconstructor, measured_data) -> None:
    measurement_type = str(getattr(measured_data, "type", "real")).strip().lower()
    reference_meas_raw = getattr(measured_data, "reference_meas", None)
    target_meas_raw = getattr(measured_data, "target_meas", None)
    reference_meas = (
        np.asarray(reference_meas_raw, dtype=np.float64).reshape(-1)
        if reference_meas_raw is not None
        else None
    )
    target_meas = (
        np.asarray(target_meas_raw, dtype=np.float64).reshape(-1)
        if target_meas_raw is not None
        else None
    )

    if measurement_type == "difference" and reference_meas is not None:
        reconstructor._measurement_space_type = "difference"
        reconstructor._difference_reference_meas = reference_meas.copy()
        reconstructor._difference_target_meas = target_meas.copy() if target_meas is not None else None
        reconstructor._difference_mode_effective = normalize_difference_mode(
            getattr(measured_data, "difference_mode", reconstructor.difference_mode),
            default=reconstructor.difference_mode,
        )
        reconstructor._difference_orientation_effective = normalize_difference_orientation(
            getattr(measured_data, "difference_orientation", reconstructor.difference_orientation),
            default=reconstructor.difference_orientation,
        )
        return

    reconstructor._measurement_space_type = "real"
    reconstructor._difference_reference_meas = None
    reconstructor._difference_target_meas = None
    reconstructor._difference_mode_effective = reconstructor.difference_mode
    reconstructor._difference_orientation_effective = reconstructor.difference_orientation


def _measurement_space_kwargs(reconstructor) -> dict[str, object]:
    """Common keyword arguments for measurement projection functions."""
    return {
        "measurement_type": getattr(reconstructor, "_measurement_space_type", "real"),
        "reference_meas": getattr(reconstructor, "_difference_reference_meas", None),
        "difference_mode": getattr(
            reconstructor, "_difference_mode_effective", reconstructor.difference_mode
        ),
        "difference_orientation": getattr(
            reconstructor, "_difference_orientation_effective", reconstructor.difference_orientation
        ),
    }


def _project_simulated_measurements(reconstructor, simulated_meas: np.ndarray) -> np.ndarray:
    return project_measurement_vector(simulated_meas, **_measurement_space_kwargs(reconstructor))


def _project_measurement_jacobian(reconstructor, jacobian: np.ndarray) -> np.ndarray:
    return project_measurement_jacobian(jacobian, **_measurement_space_kwargs(reconstructor))


def _startup_cache_payload(reconstructor, sigma_array: np.ndarray, jacobian_method: str) -> dict[str, object]:
    sigma_hash = hashlib.sha256(
        np.ascontiguousarray(sigma_array, dtype=np.float64).tobytes()
    ).hexdigest()
    return {
        "solver": "gn_absolute",
        "mode": str(getattr(reconstructor, "solver_mode", "strict")),
        "jacobian_method": str(jacobian_method),
        "sigma_hash": sigma_hash,
        "model_signature": model_signature_from_forward_model(reconstructor.fwd_model),
        "pattern_signature": pattern_signature_from_forward_model(reconstructor.fwd_model),
        "backend_signature": backend_signature_from_forward_model(reconstructor.fwd_model),
        "solver_config": {
            "linear_solver": str(getattr(reconstructor, "linear_solver", "auto")),
            "preconditioner": str(getattr(reconstructor, "preconditioner", "auto")),
            "line_search_mode": str(getattr(reconstructor, "line_search_mode", "full")),
            "jacobian_update_every": int(getattr(reconstructor, "jacobian_update_every", 1)),
            "jacobian_reuse_tol": float(getattr(reconstructor, "jacobian_reuse_tol", 0.0)),
            "rom_mode": str(getattr(reconstructor, "rom_mode", "off")),
            "rom_rank_global": int(getattr(reconstructor, "rom_rank_global", 32)),
            "rom_rank_adaptive": int(getattr(reconstructor, "rom_rank_adaptive", 16)),
            "rom_refresh_every": int(getattr(reconstructor, "rom_refresh_every", 2)),
            "rom_snapshot_source": str(getattr(reconstructor, "rom_snapshot_source", "hybrid")),
            "inexact_mode": str(getattr(reconstructor, "inexact_mode", "off")),
            "inexact_forcing": str(getattr(reconstructor, "inexact_forcing", "eisenstat-walker")),
            "inexact_eta0": float(getattr(reconstructor, "inexact_eta0", 0.2)),
            "inexact_eta_min": float(getattr(reconstructor, "inexact_eta_min", 1e-3)),
            "inexact_eta_max": float(getattr(reconstructor, "inexact_eta_max", 0.5)),
            "lowrank_mode": str(getattr(reconstructor, "lowrank_mode", "off")),
            "lowrank_rank": int(getattr(reconstructor, "lowrank_rank", 16)),
            "lowrank_method": str(getattr(reconstructor, "lowrank_method", "tsvd")),
            "lowrank_energy": float(getattr(reconstructor, "lowrank_energy", 0.995)),
        },
    }


def _startup_cache_lookup(
    reconstructor,
    sigma_current: fem.Function,
    jacobian_method: str,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if (
        reconstructor.solver_mode != "fast"
        or not bool(getattr(reconstructor, "absolute_startup_cache", True))
        or getattr(reconstructor, "cache_manager", None) is None
    ):
        return None, {"hit": False, "layer": "disabled", "artifact": "absolute_startup_jacobian"}

    sigma_array = function_get_array(sigma_current)
    payload = _startup_cache_payload(reconstructor, sigma_array, jacobian_method)
    jacobian, lookup = reconstructor.cache_manager.get_or_compute_semantic(
        artifact="absolute_startup_jacobian",
        name="gn_absolute_startup_jacobian",
        namespace="absolute",
        cache_obj=payload,
        payload=payload,
        compute_fn=lambda: reconstructor.jacobian_calculator.calculate(
            sigma_current,
            method=jacobian_method,
        ),
        persist=True,
        cost=10.0,
        effort_seconds=6.0,
    )
    cached = np.asarray(jacobian, dtype=np.float64)
    if reconstructor.negate_jacobian:
        cached = -cached
    return cached, {
        "hit": bool(lookup.hit),
        "layer": str(lookup.layer),
        "artifact": str(lookup.artifact),
        "key": str(lookup.key),
    }


def _init_sigma_function(
    reconstructor,
    initial_conductivity,
) -> tuple[fem.Function, float | np.ndarray]:
    if initial_conductivity is None:
        initial_conductivity = 1.0
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    if np.isscalar(initial_conductivity):
        function_set_array(
            sigma_current,
            np.full(reconstructor.n_elements, float(initial_conductivity), dtype=float),
        )
    else:
        function_set_array(
            sigma_current, np.asarray(initial_conductivity).flatten()
        )
    return sigma_current, initial_conductivity


def _prepare_prior(
    reconstructor,
    prior_data: np.ndarray | None,
    initial_conductivity: float | np.ndarray,
) -> torch.Tensor:
    if prior_data is not None:
        reconstructor._prior_data = np.asarray(prior_data).flatten()
    elif np.isscalar(initial_conductivity):
        reconstructor._prior_data = np.full(reconstructor.n_elements, initial_conductivity)
    else:
        reconstructor._prior_data = np.asarray(initial_conductivity).flatten()
    return _to_runtime_tensor(reconstructor, reconstructor._prior_data)


def _best_homog_bounds(
    reconstructor,
    initial_conductivity: float | np.ndarray,
) -> tuple[float, float]:
    if np.isscalar(initial_conductivity):
        center = max(float(initial_conductivity), 1e-6)
    else:
        arr = np.asarray(initial_conductivity, dtype=np.float64).reshape(-1)
        center = max(float(np.mean(arr)), 1e-6)
    clip_values = getattr(reconstructor, "clip_values", None)
    if clip_values is not None:
        lower = max(float(clip_values[0]), 1e-6)
        upper = max(float(clip_values[1]), lower * 1.01)
        return (lower, upper)
    return (max(center * 0.2, 1e-6), max(center * 5.0, center + 1e-6))


def _estimate_best_homogeneous_conductivity(
    reconstructor,
    *,
    measured_vector: np.ndarray,
    initial_conductivity: float | np.ndarray,
) -> dict[str, object]:
    mode = str(getattr(reconstructor, "best_homog_mode", "off")).strip().lower()
    info: dict[str, object] = {
        "mode": mode,
        "applied": False,
        "value": None,
        "objective": None,
    }
    if mode != "optimize":
        info["reason"] = "disabled"
        return info
    if getattr(reconstructor, "_measurement_space_type", "real") != "real":
        info["reason"] = "difference_measurement_space"
        return info

    bounds = _best_homog_bounds(reconstructor, initial_conductivity)
    info["bounds"] = [float(bounds[0]), float(bounds[1])]
    calls = {"count": 0}

    def _objective(conductivity_value: float) -> float:
        calls["count"] += 1
        image = EITImage(
            elem_data=np.full(
                reconstructor.n_elements,
                float(conductivity_value),
                dtype=np.float64,
            ),
            fwd_model=reconstructor.fwd_model,
        )
        simulated, _ = reconstructor.fwd_model.fwd_solve(image)
        simulated_vector = _project_simulated_measurements(reconstructor, simulated.meas)
        residual = np.asarray(simulated_vector, dtype=np.float64) - measured_vector
        return float(np.dot(residual, residual))

    try:
        result = minimize_scalar(
            _objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 32},
        )
        value = float(result.x)
        objective = float(result.fun)
        info["success"] = bool(result.success)
        info["message"] = str(result.message)
    except Exception as exc:
        info["reason"] = f"optimization_failed:{type(exc).__name__}"
        info["eval_count"] = int(calls["count"])
        return info

    info["eval_count"] = int(calls["count"])
    info["applied"] = True
    info["value"] = value
    info["objective"] = objective
    return info


def _difference_step_size_objective(
    reconstructor,
    *,
    prior_sigma: np.ndarray,
    delta_sigma: np.ndarray,
    measured_vector: np.ndarray,
    alpha: float,
) -> float:
    sigma = np.asarray(prior_sigma + float(alpha) * delta_sigma, dtype=np.float64)
    clip_values = getattr(reconstructor, "clip_values", None)
    if clip_values is not None:
        sigma = np.clip(sigma, clip_values[0], clip_values[1])
    image = EITImage(elem_data=sigma, fwd_model=reconstructor.fwd_model)
    simulated, _ = reconstructor.fwd_model.fwd_solve(image)
    simulated_vector = _project_simulated_measurements(reconstructor, simulated.meas)
    residual = np.asarray(simulated_vector, dtype=np.float64) - measured_vector
    return float(np.dot(residual, residual))


def _apply_difference_step_size(
    reconstructor,
    *,
    sigma_final: np.ndarray,
    measured_vector: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    mode = str(getattr(reconstructor, "difference_step_size_mode", "off")).strip().lower()
    info: dict[str, object] = {
        "mode": mode,
        "applied": False,
        "value": 1.0,
        "objective": None,
    }
    if getattr(reconstructor, "_measurement_space_type", "real") != "difference":
        info["reason"] = "real_measurement_space"
        return np.asarray(sigma_final, dtype=np.float64), info
    if mode == "off":
        info["reason"] = "disabled"
        return np.asarray(sigma_final, dtype=np.float64), info
    if str(getattr(reconstructor, "active_preset_name", "")).strip().lower() not in {
        "eidors_one_step_noser",
        "eidors_demo3d_tv",
    }:
        info["reason"] = "preset_not_one_step"
        return np.asarray(sigma_final, dtype=np.float64), info

    prior_raw = getattr(reconstructor, "_prior_data", None)
    if prior_raw is None:
        info["reason"] = "missing_prior"
        return np.asarray(sigma_final, dtype=np.float64), info
    prior_sigma = np.asarray(prior_raw, dtype=np.float64).reshape(-1)
    if prior_sigma.shape[0] != sigma_final.shape[0]:
        info["reason"] = "missing_prior"
        return np.asarray(sigma_final, dtype=np.float64), info

    delta_sigma = np.asarray(sigma_final - prior_sigma, dtype=np.float64)
    if np.linalg.norm(delta_sigma) <= 1e-18:
        info["reason"] = "zero_delta"
        return np.asarray(sigma_final, dtype=np.float64), info

    if mode == "fixed":
        alpha = float(
            1.0 if reconstructor.difference_step_size_value is None else reconstructor.difference_step_size_value
        )
        objective = _difference_step_size_objective(
            reconstructor,
            prior_sigma=prior_sigma,
            delta_sigma=delta_sigma,
            measured_vector=measured_vector,
            alpha=alpha,
        )
        info["applied"] = True
        info["value"] = alpha
        info["objective"] = objective
        return np.asarray(prior_sigma + alpha * delta_sigma, dtype=np.float64), info

    bounds = tuple(float(v) for v in getattr(reconstructor, "difference_step_size_bounds", (0.0, 4.0)))
    options = {"xatol": 1e-3, "maxiter": 32}
    options.update(dict(getattr(reconstructor, "difference_step_size_fmin_options", {}) or {}))
    info["bounds"] = [float(bounds[0]), float(bounds[1])]
    calls = {"count": 0}

    def _objective(alpha: float) -> float:
        calls["count"] += 1
        return _difference_step_size_objective(
            reconstructor,
            prior_sigma=prior_sigma,
            delta_sigma=delta_sigma,
            measured_vector=measured_vector,
            alpha=float(alpha),
        )

    try:
        result = minimize_scalar(
            _objective,
            bounds=bounds,
            method="bounded",
            options=options,
        )
        alpha = float(result.x)
        objective = float(result.fun)
        info["success"] = bool(result.success)
        info["message"] = str(result.message)
    except Exception as exc:
        info["reason"] = f"optimization_failed:{type(exc).__name__}"
        info["eval_count"] = int(calls["count"])
        return np.asarray(sigma_final, dtype=np.float64), info

    info["eval_count"] = int(calls["count"])
    info["applied"] = True
    info["value"] = alpha
    info["objective"] = objective
    return np.asarray(prior_sigma + alpha * delta_sigma, dtype=np.float64), info


def _compute_residuals(
    reconstructor,
    simulated_meas: np.ndarray,
    meas_torch: torch.Tensor,
    iteration: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
]:
    data_sim_torch = _to_runtime_tensor_cached(
        reconstructor, "simulated_meas", simulated_meas
    )
    residual_torch = data_sim_torch - meas_torch
    if reconstructor._meas_weight_sqrt is not None:
        weighted_residual_torch = residual_torch * reconstructor._meas_weight_sqrt
    else:
        weighted_residual_torch = residual_torch

    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
    residual_norm = torch.norm(residual_torch).item()
    residual_max = torch.max(torch.abs(residual_torch)).item()
    _require_scalar_finite("residual_norm_weighted", residual_norm_weighted, iteration)
    _require_scalar_finite("residual_norm", residual_norm, iteration)
    _require_scalar_finite("residual_max", residual_max, iteration)
    return (
        data_sim_torch,
        residual_torch,
        weighted_residual_torch,
        residual_norm_weighted,
        residual_norm,
        residual_max,
    )


def _compute_objective(
    reconstructor,
    weighted_residual_torch: torch.Tensor,
    de_current: torch.Tensor,
    lambda_eff: float,
    iteration: int,
) -> tuple[float, float, float, torch.Tensor]:
    meas_misfit = 0.5 * torch.dot(weighted_residual_torch, weighted_residual_torch).item()
    if reconstructor.R_torch is not None:
        RtR_de = torch.mv(reconstructor.R_torch, de_current)
        prior_misfit = 0.5 * lambda_eff * torch.dot(de_current, RtR_de).item()
    else:
        de_np = de_current.detach().cpu().numpy()
        rde_np = _apply_regularization_np(reconstructor, de_np)
        prior_misfit = 0.5 * lambda_eff * float(np.dot(de_np, rde_np))
        RtR_de = _to_runtime_tensor_cached(reconstructor, "RtR_de_fast", rde_np)
    total_objective = meas_misfit + prior_misfit
    _require_scalar_finite("meas_misfit", meas_misfit, iteration)
    _require_scalar_finite("prior_misfit", prior_misfit, iteration)
    _require_scalar_finite("total_objective", total_objective, iteration)
    return meas_misfit, prior_misfit, total_objective, RtR_de


def _build_linear_system(
    reconstructor,
    JTJ: torch.Tensor,
    JTr: torch.Tensor,
    de_torch: torch.Tensor,
    lambda_eff: float,
    iteration: int,
    RtR_de: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = JTJ + lambda_eff * reconstructor.R_torch
    if reconstructor.use_prior_term:
        if RtR_de is None:
            RtR_de = torch.mv(reconstructor.R_torch, de_torch)
        b = -(JTr + lambda_eff * RtR_de)
    else:
        b = -JTr
    _require_finite("A", A, iteration)
    _require_finite("b", b, iteration)
    return A, b


def _solve_linear_system_torch_cg(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int | None = None,
) -> torch.Tensor:
    if max_iter is None:
        max_iter = max(512, min(int(A.shape[0]) * 2, 4096))
    x = torch.zeros_like(b)
    r = b.clone()
    diag = torch.diagonal(A)
    safe_diag = torch.where(torch.abs(diag) > 1e-18, diag, torch.ones_like(diag))
    z = r / safe_diag
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = float(torch.linalg.vector_norm(b).item())
    tol = max(float(atol), float(rtol) * max(b_norm, 1e-18))
    for _ in range(int(max_iter)):
        Ap = torch.mv(A, p)
        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom) or torch.abs(denom) <= 1e-30:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if float(torch.linalg.vector_norm(r).item()) <= tol:
            return x
        z = r / safe_diag
        rz_new = torch.dot(r, z)
        if not torch.isfinite(rz_new):
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    raise RuntimeError("torch CG fallback did not converge")


def _solve_linear_system(
    reconstructor,
    A: torch.Tensor,
    b: torch.Tensor,
    JTJ: torch.Tensor,
    iteration: int,
) -> tuple[torch.Tensor, float]:
    try:
        delta_sigma_torch = torch.linalg.solve(A, b)
    except RuntimeError as exc:
        message = str(exc).lower()
        runtime_linalg_unavailable = any(
            token in message for token in ("libtorch_cuda_linalg", "cusolver", "undefined symbol", "dlopen")
        )
        if runtime_linalg_unavailable and A.device.type == "cuda":
            delta_sigma_torch = _solve_linear_system_torch_cg(
                A,
                b,
                rtol=1e-12,
                atol=1e-14,
                max_iter=max(2048, min(int(A.shape[0]) * 4, 8192)),
            )
        else:
            A_regularized = JTJ + (reconstructor.regularization_param * 10) * reconstructor.R_torch
            _require_finite("A_regularized", A_regularized, iteration)
            try:
                delta_sigma_torch = torch.linalg.solve(A_regularized, b)
            except RuntimeError:
                delta_sigma_torch = _solve_linear_system_torch_cg(
                    A_regularized if A_regularized.device.type == "cuda" else A,
                    b,
                    rtol=1e-10,
                    atol=1e-12,
                )
    _require_finite("delta_sigma_torch", delta_sigma_torch, iteration)
    delta_norm = torch.norm(delta_sigma_torch).item()
    _require_scalar_finite("delta_norm", delta_norm, iteration)
    return delta_sigma_torch, delta_norm


def _select_step_size(
    reconstructor,
    iteration: int,
    sigma_current: fem.Function,
    delta_sigma_torch: torch.Tensor,
    meas_torch: torch.Tensor,
    residual_norm_weighted: float,
    prior_torch: torch.Tensor,
    lambda_eff: float,
) -> float:
    if (
        getattr(reconstructor, "_measurement_space_type", "real") == "difference"
        and str(getattr(reconstructor, "active_preset_name", "")).strip().lower()
        in {"eidors_one_step_noser", "eidors_demo3d_tv"}
        and int(getattr(reconstructor, "max_iterations", 1)) <= 1
    ):
        return 1.0

    if reconstructor.solver_mode == "fast" and reconstructor.line_search_mode == "fast":
        quick_step = min(float(reconstructor.max_step), 1.0)
        if reconstructor.min_step is not None:
            quick_step = max(float(reconstructor.min_step), quick_step)
        _require_scalar_finite("optimal_step_size", quick_step, iteration)
        return quick_step

    if reconstructor.step_schedule is not None and iteration < len(reconstructor.step_schedule):
        optimal_step_size = float(reconstructor.step_schedule[iteration])
    else:
        optimal_step_size = reconstructor._line_search_torch(
            sigma_current,
            delta_sigma_torch,
            meas_torch,
            residual_norm_weighted,
            reconstructor._meas_weight_sqrt,
            prior_torch=prior_torch,
            lambda_eff=lambda_eff,
        )
        if reconstructor.min_step is not None and optimal_step_size < reconstructor.min_step:
            optimal_step_size = reconstructor.min_step
    _require_scalar_finite("optimal_step_size", optimal_step_size, iteration)
    return optimal_step_size


def _maybe_rollback(
    reconstructor,
    sigma_current: fem.Function,
    sigma_old_values: np.ndarray,
    residual_norm: float,
    prev_residual: float | None,
    residual_history: list[float],
    sigma_change_history: list[float],
    consecutive_rollbacks: int,
    max_consecutive_rollbacks: int,
) -> tuple[bool, bool, int]:
    if prev_residual is None or residual_norm <= prev_residual:
        return False, False, consecutive_rollbacks

    consecutive_rollbacks += 1
    if reconstructor.verbose:
        print(
            f"[WARN] residual increased ({residual_norm:.3e} > {prev_residual:.3e}), "
            f"rolling back step ({consecutive_rollbacks}/{max_consecutive_rollbacks})"
        )
    function_set_array(sigma_current, sigma_old_values)
    residual_history[-1] = prev_residual
    sigma_change_history[-1] = 0.0

    if consecutive_rollbacks >= max_consecutive_rollbacks:
        if reconstructor.verbose:
            print(f"[STOP] {max_consecutive_rollbacks} consecutive rollbacks, terminating early")
        return True, True, consecutive_rollbacks

    return True, False, consecutive_rollbacks


def _record_iteration_log(
    iteration_logs: list[_IterationLog],
    *,
    iteration: int,
    residual_norm: float,
    residual_norm_weighted: float,
    rel_residual: float,
    rel_residual_weighted: float | None,
    residual_max: float,
    meas_norm: float,
    pred_norm: float,
    meas_max: float,
    pred_max: float,
    jtr_norm: float,
    delta_norm: float,
    optimal_step_size: float,
    lambda_eff: float,
    relative_change: float,
    res_drop: float | None,
    meas_misfit: float,
    prior_misfit: float,
    total_objective: float,
) -> None:
    iteration_logs.append(
        _IterationLog(
            iteration=iteration,
            residual=residual_norm,
            residual_weighted=residual_norm_weighted,
            relative_residual=rel_residual,
            relative_residual_weighted=rel_residual_weighted,
            residual_max=residual_max,
            meas_norm=meas_norm,
            pred_norm=pred_norm,
            meas_max=meas_max,
            pred_max=pred_max,
            jtr_norm=jtr_norm,
            delta_norm=delta_norm,
            step=optimal_step_size,
            lambda_eff=lambda_eff,
            relative_change=relative_change,
            res_drop=res_drop,
            meas_misfit=meas_misfit,
            prior_misfit=prior_misfit,
            total_objective=total_objective,
        )
    )


def run_reconstruction(
    reconstructor,
    measured_data: object | np.ndarray,
    initial_conductivity: float = 1.0,
    jacobian_method: str = "efficient",
    prior_data: np.ndarray | None = None,
    record_conductivity_history: bool = False,
    conductivity_history_stride: int = 1,
) -> SolverOutput:
    """Execute Gauss-Newton iterations and return typed solver output."""
    reconstructor._meas_weight_sqrt = None
    reconstructor._baseline_measurement = None

    meas_vector = _extract_measured_vector(measured_data)
    _configure_measurement_space(reconstructor, measured_data)
    reconstructor._difference_step_size_info = {
        "mode": str(getattr(reconstructor, "difference_step_size_mode", "off")),
        "applied": False,
        "value": 1.0,
    }
    reconstructor._best_homog_info = {
        "mode": str(getattr(reconstructor, "best_homog_mode", "off")),
        "applied": False,
        "value": None,
    }

    if len(meas_vector) != reconstructor.n_measurements:
        raise ValueError(
            f"Measurement data length mismatch: {len(meas_vector)} vs {reconstructor.n_measurements}"
        )
    _require_finite("measured_data", meas_vector)

    meas_torch = _to_runtime_tensor(reconstructor, meas_vector)
    reconstructor._measured_vector = meas_vector.copy()
    reconstructor.ensure_regularization_ready()

    meas_norm = torch.norm(meas_torch).item()
    meas_max = torch.max(torch.abs(meas_torch)).item()
    _require_scalar_finite("meas_norm", meas_norm)
    _require_scalar_finite("meas_max", meas_max)
    meas_weighted_norm = None

    if prior_data is None:
        best_homog_info = _estimate_best_homogeneous_conductivity(
            reconstructor,
            measured_vector=meas_vector,
            initial_conductivity=initial_conductivity,
        )
        reconstructor._best_homog_info = best_homog_info
        if best_homog_info.get("applied") and best_homog_info.get("value") is not None:
            initial_conductivity = float(best_homog_info["value"])

    sigma_current, initial_conductivity = _init_sigma_function(reconstructor, initial_conductivity)
    reconstructor._ensure_measurement_weights(sigma_current)

    if reconstructor._meas_weight_sqrt is not None:
        meas_weighted_norm = torch.norm(meas_torch * reconstructor._meas_weight_sqrt).item()
        _require_scalar_finite("meas_weighted_norm", meas_weighted_norm)

    prior_torch = _prepare_prior(reconstructor, prior_data, initial_conductivity)

    residual_history = []
    sigma_change_history = []
    iteration_logs: list[_IterationLog] = []
    conductivity_history = []
    history_stride = max(1, int(conductivity_history_stride))
    if record_conductivity_history:
        conductivity_history.append(function_get_array(sigma_current).copy())

    consecutive_rollbacks = 0
    max_consecutive_rollbacks = 5

    if reconstructor.verbose:
        print(
            f"[INFO] lambda={reconstructor.regularization_param:.3e}, "
            f"hp={reconstructor.hyperparameter:.3e}"
        )
        print("\nStarting modular Gauss-Newton reconstruction...")
        print(f"Using Jacobian method: {jacobian_method}")

    prev_residual = None
    relative_change = float("inf")
    reconstructor._runtime_tensor_cache = {}
    reconstructor._force_jacobian_refresh = False
    prev_jacobian_np: np.ndarray | None = None
    prev_jacobian_iter = -1
    fast_fallback_reason: str | None = None
    resolved_preconditioner: str | None = None
    fast_solver_path: str | None = None
    fast_linear_path_selected: str | None = None
    fast_linear_path_reason: str | None = None
    degrade_stage_counts: dict[str, int] = {}
    effective_solver_path_counts: dict[str, int] = {}
    inexact_eta_history: list[float] = []
    rom_rank_effective: int = 0
    lowrank_rank_effective: int = 0
    rom_enabled_effective = False
    timing_totals = {
        "forward": 0.0,
        "jacobian": 0.0,
        "linear_solve": 0.0,
        "line_search": 0.0,
    }
    startup_jacobian_np, startup_cache_lookup = _startup_cache_lookup(
        reconstructor,
        sigma_current,
        jacobian_method,
    )
    if startup_jacobian_np is not None:
        startup_jacobian_np = _project_measurement_jacobian(reconstructor, startup_jacobian_np)
    final_simulated_measurement: np.ndarray | None = None

    lambda_eff = reconstructor.regularization_param
    with reconstructor._progress(total=reconstructor.max_iterations) as pbar:
        for iteration in range(reconstructor.max_iterations):
            sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array", sigma_array, iteration)
            img_current = EITImage(elem_data=sigma_array, fwd_model=reconstructor.fwd_model)
            forward_start = perf_counter()
            data_simulated, _ = reconstructor.fwd_model.fwd_solve(img_current)
            timing_totals["forward"] += perf_counter() - forward_start
            _require_finite("data_simulated.meas", data_simulated.meas, iteration)
            simulated_measurement = _project_simulated_measurements(
                reconstructor,
                data_simulated.meas,
            )
            _require_finite("simulated_measurement", simulated_measurement, iteration)

            lambda_eff = reconstructor.regularization_param
            (
                data_sim_torch,
                residual_torch,
                weighted_residual_torch,
                residual_norm_weighted,
                residual_norm,
                residual_max,
            ) = _compute_residuals(
                reconstructor,
                simulated_measurement,
                meas_torch,
                iteration,
            )

            sigma_vec_torch = _to_runtime_tensor_cached(
                reconstructor,
                "sigma_vector",
                sigma_array,
            )
            de_current = sigma_vec_torch - prior_torch
            meas_misfit, prior_misfit, total_objective, RtR_de = _compute_objective(
                reconstructor,
                weighted_residual_torch,
                de_current,
                lambda_eff,
                iteration,
            )

            residual_history.append(residual_norm)
            res_drop = None if prev_residual is None else prev_residual - residual_norm
            reconstructor._outer_prev_residual = prev_residual

            jacobian_reused = False
            reuse_tol = float(reconstructor.jacobian_reuse_tol)
            large_fast_problem = reconstructor.solver_mode == "fast" and reconstructor.n_elements >= 5000
            reuse_change_ok = float(relative_change) <= reuse_tol
            if large_fast_problem:
                # For large 3D runs, update cadence + residual trend already guard stability.
                # Skipping per-iteration Jacobian rebuilds is the dominant speed lever.
                reuse_change_ok = True
            force_refresh = bool(getattr(reconstructor, "_force_jacobian_refresh", False))
            if force_refresh:
                reconstructor._force_jacobian_refresh = False
            can_reuse = (
                reconstructor.solver_mode == "fast"
                and prev_jacobian_np is not None
                and (iteration - prev_jacobian_iter) < max(1, reconstructor.jacobian_update_every)
                and reuse_change_ok
                and not force_refresh
                and (
                    prev_residual is None
                    or residual_norm <= (float(prev_residual) * 1.05)
                )
            )
            if can_reuse:
                measurement_jacobian_np = prev_jacobian_np
                jacobian_reused = True
            elif iteration == 0 and startup_jacobian_np is not None:
                measurement_jacobian_np = np.asarray(startup_jacobian_np, dtype=np.float64)
                prev_jacobian_np = measurement_jacobian_np
                prev_jacobian_iter = iteration
            else:
                jacobian_start = perf_counter()
                measurement_jacobian_np = reconstructor.jacobian_calculator.calculate(
                    sigma_current,
                    method=jacobian_method,
                )
                timing_totals["jacobian"] += perf_counter() - jacobian_start
                if reconstructor.negate_jacobian:
                    measurement_jacobian_np = -measurement_jacobian_np
                measurement_jacobian_np = _project_measurement_jacobian(
                    reconstructor,
                    measurement_jacobian_np,
                )
                _require_finite("measurement_jacobian_np", measurement_jacobian_np, iteration)
                prev_jacobian_np = measurement_jacobian_np
                prev_jacobian_iter = iteration

            if reconstructor._meas_weight_sqrt is not None:
                meas_weight_np = reconstructor._meas_weight_sqrt.detach().cpu().numpy()
                weighted_residual_np = residual_torch.detach().cpu().numpy() * meas_weight_np
            else:
                meas_weight_np = None
                weighted_residual_np = residual_torch.detach().cpu().numpy()

            def _solve_strict_path() -> tuple[torch.Tensor, float, float]:
                J_torch_local = _to_runtime_tensor_cached(
                    reconstructor,
                    "measurement_jacobian",
                    measurement_jacobian_np,
                )
                if reconstructor._meas_weight_sqrt is not None:
                    J_weighted_local = J_torch_local * reconstructor._meas_weight_sqrt.unsqueeze(1)
                else:
                    J_weighted_local = J_torch_local

                JTJ_local = torch.mm(J_weighted_local.t(), J_weighted_local)
                JTr_local = torch.mv(J_weighted_local.t(), weighted_residual_torch)

                de_torch_local = de_current
                A_local, b_local = _build_linear_system(
                    reconstructor,
                    JTJ_local,
                    JTr_local,
                    de_torch_local,
                    lambda_eff,
                    iteration,
                    RtR_de=RtR_de,
                )
                jtr_norm_local = torch.norm(JTr_local).item()
                delta_sigma_torch_local, delta_norm_local = _solve_linear_system(
                    reconstructor,
                    A_local,
                    b_local,
                    JTJ_local,
                    iteration,
                )
                return delta_sigma_torch_local, delta_norm_local, float(jtr_norm_local)

            if reconstructor.solver_mode == "fast":
                J_weighted_np = (
                    measurement_jacobian_np * meas_weight_np[:, None]
                    if meas_weight_np is not None
                    else measurement_jacobian_np
                )
                de_current_np = de_current.detach().cpu().numpy()
                linear_start = perf_counter()
                try:
                    delta_sigma_np, delta_norm, jtr_norm = _solve_linear_system_fast(
                        reconstructor,
                        J_weighted_np=J_weighted_np,
                        weighted_residual_np=weighted_residual_np,
                        de_current_np=de_current_np,
                        lambda_eff=lambda_eff,
                        iteration=iteration,
                    )
                    delta_sigma_torch = _to_runtime_tensor_cached(
                        reconstructor,
                        "delta_sigma_fast",
                        delta_sigma_np,
                    )
                    fast_meta = getattr(reconstructor, "_last_fast_linear_meta", {})
                    if isinstance(fast_meta, dict):
                        path = fast_meta.get("path")
                        if isinstance(path, str):
                            fast_solver_path = path
                        resolved = fast_meta.get("resolved_preconditioner")
                        if isinstance(resolved, str):
                            resolved_preconditioner = resolved
                        reason = fast_meta.get("fallback_reason")
                        if isinstance(reason, str) and reason:
                            fast_fallback_reason = reason
                        selected_path = fast_meta.get("fast_linear_path_selected")
                        if isinstance(selected_path, str):
                            fast_linear_path_selected = selected_path
                        selected_reason = fast_meta.get("fast_linear_path_reason")
                        if isinstance(selected_reason, str):
                            fast_linear_path_reason = selected_reason
                        degrade_stage = fast_meta.get("degrade_stage")
                        if isinstance(degrade_stage, str) and degrade_stage:
                            degrade_stage_counts[degrade_stage] = degrade_stage_counts.get(degrade_stage, 0) + 1
                        effective_path = fast_meta.get("effective_solver_path")
                        if isinstance(effective_path, str) and effective_path:
                            effective_solver_path_counts[effective_path] = (
                                effective_solver_path_counts.get(effective_path, 0) + 1
                            )
                        else:
                            if isinstance(fast_solver_path, str) and fast_solver_path:
                                effective_solver_path_counts[fast_solver_path] = (
                                    effective_solver_path_counts.get(fast_solver_path, 0) + 1
                                )
                        eta_value = fast_meta.get("inexact_eta")
                        if isinstance(eta_value, (int, float)):
                            inexact_eta_history.append(float(eta_value))
                        rank_effective = fast_meta.get("rom_rank_effective")
                        if isinstance(rank_effective, int):
                            rom_rank_effective = max(rom_rank_effective, int(rank_effective))
                        lowrank_effective = fast_meta.get("lowrank_rank_effective")
                        if isinstance(lowrank_effective, int):
                            lowrank_rank_effective = max(lowrank_rank_effective, int(lowrank_effective))
                        if bool(fast_meta.get("rom_enabled_effective", False)):
                            rom_enabled_effective = True
                except Exception as exc:
                    fast_fallback_reason = f"fast_linear_solver_failed:{type(exc).__name__}"
                    fast_solver_path = "strict-fallback"
                    fast_linear_path_selected = "strict"
                    fast_linear_path_reason = "fast_solver_exception"
                    delta_sigma_torch, delta_norm, jtr_norm = _solve_strict_path()
                timing_totals["linear_solve"] += perf_counter() - linear_start
            else:
                linear_start = perf_counter()
                delta_sigma_torch, delta_norm, jtr_norm = _solve_strict_path()
                timing_totals["linear_solve"] += perf_counter() - linear_start

            pred_norm = torch.norm(data_sim_torch).item()
            pred_max = torch.max(torch.abs(data_sim_torch)).item()
            rel_residual = residual_norm / (meas_norm + 1e-12)
            rel_residual_weighted = (
                residual_norm_weighted / (meas_weighted_norm + 1e-12)
                if meas_weighted_norm
                else None
            )
            _require_scalar_finite("pred_norm", pred_norm, iteration)
            _require_scalar_finite("pred_max", pred_max, iteration)
            _require_scalar_finite("jtr_norm", jtr_norm, iteration)
            _require_scalar_finite("rel_residual", rel_residual, iteration)
            if rel_residual_weighted is not None:
                _require_scalar_finite("rel_residual_weighted", rel_residual_weighted, iteration)

            if reconstructor.verbose and jacobian_reused:
                print(f"[INFO] iteration={iteration}: reused Jacobian (fast mode)")

            line_search_start = perf_counter()
            optimal_step_size = _select_step_size(
                reconstructor,
                iteration,
                sigma_current,
                delta_sigma_torch,
                meas_torch,
                residual_norm_weighted,
                prior_torch,
                lambda_eff,
            )
            timing_totals["line_search"] += perf_counter() - line_search_start

            needs_snapshot = prev_residual is not None or reconstructor.clip_values is not None
            sigma_old_values = sigma_array.copy() if needs_snapshot else None
            delta_sigma_np = delta_sigma_torch.detach().cpu().numpy()
            sigma_array[:] += optimal_step_size * delta_sigma_np
            _require_finite("sigma_array_updated", sigma_array, iteration)

            if reconstructor.clip_values is not None:
                function_set_array(
                    sigma_current,
                    np.clip(sigma_array, reconstructor.clip_values[0], reconstructor.clip_values[1]),
                )
                sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array_clipped", sigma_array, iteration)

            sigma_new_norm = float(np.linalg.norm(sigma_array))
            if sigma_old_values is None:
                sigma_change = abs(optimal_step_size) * delta_norm
            else:
                sigma_change = float(np.linalg.norm(sigma_array - sigma_old_values))
            relative_change = sigma_change / (sigma_new_norm + 1e-12)
            _require_scalar_finite("sigma_change", sigma_change, iteration)
            _require_scalar_finite("relative_change", relative_change, iteration)

            rolled_back, should_stop, consecutive_rollbacks = _maybe_rollback(
                reconstructor,
                sigma_current,
                sigma_old_values if sigma_old_values is not None else sigma_array,
                residual_norm,
                prev_residual,
                residual_history,
                sigma_change_history,
                consecutive_rollbacks,
                max_consecutive_rollbacks,
            )
            if rolled_back:
                if should_stop:
                    break
                continue

            consecutive_rollbacks = 0
            sigma_change_history.append(relative_change)
            if record_conductivity_history and (iteration + 1) % history_stride == 0:
                conductivity_history.append(function_get_array(sigma_current).copy())
            final_simulated_measurement = simulated_measurement.copy()

            _record_iteration_log(
                iteration_logs,
                iteration=iteration,
                residual_norm=residual_norm,
                residual_norm_weighted=residual_norm_weighted,
                rel_residual=rel_residual,
                rel_residual_weighted=rel_residual_weighted,
                residual_max=residual_max,
                meas_norm=meas_norm,
                pred_norm=pred_norm,
                meas_max=meas_max,
                pred_max=pred_max,
                jtr_norm=jtr_norm,
                delta_norm=delta_norm,
                optimal_step_size=optimal_step_size,
                lambda_eff=lambda_eff,
                relative_change=relative_change,
                res_drop=res_drop,
                meas_misfit=meas_misfit,
                prior_misfit=prior_misfit,
                total_objective=total_objective,
            )
            prev_residual = residual_norm

            if relative_change < reconstructor.convergence_tol and iteration + 1 >= reconstructor.min_iterations:
                if reconstructor.verbose:
                    print(
                        f"\nConverged! Iteration {iteration}, relative change: {relative_change:.2e}"
                    )
                break

            if reconstructor.verbose:
                pbar.set_postfix_str(
                    f"residual={residual_norm:.2e}, step={optimal_step_size:.3f}, Δσ={relative_change:.2e}"
                )
                pbar.update(1)

    cache_stats = {}
    if getattr(reconstructor, "cache_manager", None) is not None:
        cache_stats = reconstructor.cache_manager.stats()
    sigma_final_array = function_get_array(sigma_current).copy()
    sigma_final_array, difference_step_size_info = _apply_difference_step_size(
        reconstructor,
        sigma_final=sigma_final_array,
        measured_vector=meas_vector,
    )
    reconstructor._difference_step_size_info = difference_step_size_info
    if reconstructor.clip_values is not None:
        sigma_final_array = np.clip(
            sigma_final_array,
            reconstructor.clip_values[0],
            reconstructor.clip_values[1],
        )
    function_set_array(sigma_current, sigma_final_array)
    final_img = EITImage(
        elem_data=sigma_final_array.copy(),
        fwd_model=reconstructor.fwd_model,
    )
    final_data_simulated, _ = reconstructor.fwd_model.fwd_solve(final_img)
    final_simulated_measurement = _project_simulated_measurements(
        reconstructor,
        final_data_simulated.meas,
    )
    jacobian_block_tune_info = (
        reconstructor.jacobian_calculator.block_tuning_info()
        if hasattr(reconstructor.jacobian_calculator, "block_tuning_info")
        else {}
    )
    petsc_backend_info = getattr(reconstructor.fwd_model, "_petsc_backend_info", {}) or {}
    inverse_device_requested = str(getattr(reconstructor, "device_requested", "cpu"))
    inverse_device_effective = str(getattr(reconstructor, "device_effective", str(getattr(reconstructor, "device", "cpu"))))
    petsc_effective = str(petsc_backend_info.get("petsc_device_effective", "cpu"))
    jacobian_block_backend = (
        jacobian_block_tune_info.get("jacobian_block_backend")
        if isinstance(jacobian_block_tune_info, dict)
        else None
    )
    if petsc_effective == "cuda" and inverse_device_effective == "cuda" and jacobian_block_backend == "torch-cuda":
        execution_profile = "cuda"
    elif petsc_effective == "cuda" or inverse_device_effective == "cuda" or jacobian_block_backend == "torch-cuda":
        execution_profile = "mixed"
    else:
        execution_profile = "cpu"

    backend_info = {
        "linear_backend": getattr(reconstructor.fwd_model, "linear_backend", "unknown"),
        "performance_mode": getattr(reconstructor, "performance_mode", "aggressive"),
        "solver_mode": getattr(reconstructor, "solver_mode", "strict"),
        "linear_solver": getattr(reconstructor, "linear_solver", "auto"),
        "line_search_mode": getattr(reconstructor, "line_search_mode", "full"),
        "preconditioner": getattr(reconstructor, "preconditioner", "auto"),
        "resolved_preconditioner": resolved_preconditioner,
        "rom_mode": getattr(reconstructor, "rom_mode", "off"),
        "rom_snapshot_source": getattr(reconstructor, "rom_snapshot_source", "hybrid"),
        "inexact_mode": getattr(reconstructor, "inexact_mode", "off"),
        "inexact_forcing": getattr(reconstructor, "inexact_forcing", "eisenstat-walker"),
        "lowrank_mode": getattr(reconstructor, "lowrank_mode", "off"),
        "lowrank_method": getattr(reconstructor, "lowrank_method", "tsvd"),
        "fast_solver_path": fast_solver_path,
        "fast_linear_path_selected": fast_linear_path_selected,
        "fast_linear_path_reason": fast_linear_path_reason,
        "fallback_reason": fast_fallback_reason,
        "fast_fallback_reason": fast_fallback_reason,
        "rom_enabled_effective": bool(rom_enabled_effective),
        "rom_rank_effective": int(rom_rank_effective),
        "lowrank_rank_effective": int(lowrank_rank_effective),
        "inexact_eta_history": inexact_eta_history[-32:],
        "degrade_stage_counts": degrade_stage_counts,
        "effective_solver_path_counts": effective_solver_path_counts,
        "startup_cache_lookup": startup_cache_lookup,
        "forward_cache_lookup": getattr(reconstructor.fwd_model, "_last_cache_lookup", {}),
        "jacobian_cache_lookup": getattr(reconstructor.jacobian_calculator, "_last_cache_lookup", {}),
        "petsc_device_requested": petsc_backend_info.get("petsc_device_requested", "auto"),
        "petsc_device_effective": petsc_effective,
        "petsc_mat_type": petsc_backend_info.get("petsc_mat_type"),
        "petsc_vec_type": petsc_backend_info.get("petsc_vec_type"),
        "forward_mat_solve_effective": petsc_backend_info.get("forward_mat_solve_effective"),
        "gpu_fallback_reason": petsc_backend_info.get("gpu_fallback_reason"),
        "forward_factor_backend": petsc_backend_info.get("forward_factor_backend"),
        "inverse_device_requested": inverse_device_requested,
        "inverse_device_effective": inverse_device_effective,
        "inverse_device_fallback_reason": getattr(reconstructor, "device_fallback_reason", None),
        "execution_profile": execution_profile,
        "jacobian_backend_requested": jacobian_block_tune_info.get("jacobian_backend_requested")
        if isinstance(jacobian_block_tune_info, dict)
        else None,
        "jacobian_backend_effective": jacobian_block_tune_info.get("jacobian_backend_effective")
        if isinstance(jacobian_block_tune_info, dict)
        else None,
        "jacobian_block_backend": jacobian_block_backend,
        "jacobian_transfer_estimate": jacobian_block_tune_info.get("jacobian_transfer_estimate")
        if isinstance(jacobian_block_tune_info, dict)
        else None,
        "jacobian_cuda_threshold_hit": jacobian_block_tune_info.get("jacobian_cuda_threshold_hit")
        if isinstance(jacobian_block_tune_info, dict)
        else None,
        "jacobian_block_tune": jacobian_block_tune_info,
        "jacobian_assembly_elapsed_only": (
            float(jacobian_block_tune_info.get("assembly_elapsed_only", 0.0))
            if isinstance(jacobian_block_tune_info, dict)
            else 0.0
        ),
        "forward_static_setup_lookup": (
            dict(reconstructor.fwd_model.get_backend_diagnostics().get("static_setup_lookup", {}))
            if hasattr(reconstructor.fwd_model, "get_backend_diagnostics")
            else {}
        ),
        "timing_totals": timing_totals,
    }

    results = SolverOutput(
        conductivity=sigma_current,
        residual_history=residual_history,
        sigma_change_history=sigma_change_history,
        iterations=len(residual_history),
        converged=relative_change < reconstructor.convergence_tol,
        final_residual=residual_history[-1],
        final_relative_change=relative_change,
        jacobian_method=jacobian_method,
        regularization_type=type(reconstructor.regularization).__name__,
        iteration_logs=[item.to_payload() for item in iteration_logs],
        conductivity_history=conductivity_history if record_conductivity_history else None,
        baseline_measurement=(
            reconstructor._baseline_measurement.copy()
            if reconstructor._baseline_measurement is not None
            else None
        ),
        measurement_weight=(
            reconstructor._meas_weight_sqrt.detach().cpu().numpy() ** 2
            if reconstructor._meas_weight_sqrt is not None
            else None
        ),
        simulated_measurement=final_simulated_measurement,
        diagnostics={
            "cache_hits": cache_stats.get("total_hits", 0),
            "cache_misses": cache_stats.get("total_misses", 0),
            "cache_stats": cache_stats,
            "backend_info": backend_info,
            "timing": timing_totals,
            "hyperparameter": float(reconstructor.hyperparameter),
            "lambda_eff": float(reconstructor.regularization_param),
            "difference_step_size": reconstructor._difference_step_size_info,
            "best_homog": reconstructor._best_homog_info,
            "preset_name": str(getattr(reconstructor, "active_preset_name", "")),
            "jacobian_background_conductivity": float(
                getattr(reconstructor, "jacobian_background_conductivity", 1.0)
            ),
            "measurement_space": {
                "type": getattr(reconstructor, "_measurement_space_type", "real"),
                "difference_mode": getattr(
                    reconstructor,
                    "_difference_mode_effective",
                    reconstructor.difference_mode,
                ),
                "difference_orientation": getattr(
                    reconstructor,
                    "_difference_orientation_effective",
                    reconstructor.difference_orientation,
                ),
            },
        },
    )

    if reconstructor.verbose:
        print("\nReconstruction complete:")
        print(f"  Iterations: {results.iterations}")
        print(f"  Final residual: {results.final_residual:.2e}")
        print(f"  Jacobian method: {jacobian_method}")
        print(f"  Regularization type: {results.regularization_type}")

    return results
