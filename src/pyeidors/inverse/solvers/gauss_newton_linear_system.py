"""Linear-system helpers for the Gauss-Newton runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Callable

import numpy as np
import torch
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator, cg, lsmr

try:  # pragma: no cover - optional dependency
    import pyamg
except Exception:  # pragma: no cover
    pyamg = None

try:  # pragma: no cover - optional dependency
    from sksparse.cholmod import cholesky as cholmod_cholesky
except Exception:  # pragma: no cover
    cholmod_cholesky = None

try:  # pragma: no cover - optional in trimmed environments
    from petsc4py import PETSc as _PETSc
except Exception:  # pragma: no cover
    _PETSc = None

from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
    rom_signature,
)
from ...perf.capabilities import (
    detect_performance_capabilities,
    select_fast_linear_path,
    select_fused_strategy,
    select_preconditioner,
)
from ...utils.numeric_ops import safe_dot
from ..jacobian.linearized import JacobianLinearization
from ..matrix_free.dual_mesh import DualMeshJacobianOperator
from ..reduced.inexact_controller import InexactController
from ..reduced.lowrank_subspace import build_lowrank_subspace
from ..reduced.pod_basis import compute_pod_basis, merge_orthonormal_bases
from ..reduced.reduced_gn_step import build_reduced_operator, solve_reduced_step
from ..reduced.snapshot_bank import SnapshotBank, select_snapshot_matrix


def _petsc_vec_to_numpy(vec) -> np.ndarray:
    """Safely extract a dense numpy array from a PETSc Vec wrapper."""
    if hasattr(vec, "array_r"):
        try:
            return np.asarray(vec.array_r, dtype=np.float64)
        except Exception:
            pass
    if hasattr(vec, "getArray"):
        try:
            return np.asarray(vec.getArray(readonly=True), dtype=np.float64)
        except Exception:
            pass
    if hasattr(vec, "array"):
        return np.asarray(vec.array, dtype=np.float64)
    raise TypeError("Unsupported PETSc Vec wrapper in matrix-free CG helper.")


class _PETScMatrixFreeHessianContext:
    """PCSHELL/MATSHELL context applying ``H v`` via a SciPy LinearOperator."""

    __slots__ = ("_op",)

    def __init__(self, op: LinearOperator) -> None:
        self._op = op

    def mult(self, _mat, x, y) -> None:
        result = np.asarray(self._op.matvec(_petsc_vec_to_numpy(x)), dtype=np.float64)
        y.getArray(readonly=False)[:] = result


class _PETScMatrixFreePCContext:
    """PCSHELL context applying ``M^{-1} r`` via a SciPy LinearOperator."""

    __slots__ = ("_op",)

    def __init__(self, op: LinearOperator) -> None:
        self._op = op

    def apply(self, _pc, x, y) -> None:
        result = np.asarray(self._op.matvec(_petsc_vec_to_numpy(x)), dtype=np.float64)
        y.getArray(readonly=False)[:] = result


def _solve_matrix_free_hessian_via_petsc(
    h_op: LinearOperator,
    rhs: np.ndarray,
    m_op: LinearOperator | None,
    *,
    rtol: float,
    maxiter: int,
    petsc_module=None,
) -> tuple[np.ndarray, int, bool, str | None]:
    """Solve ``H delta = rhs`` using PETSc KSP(CG) + MATSHELL + PCSHELL."""
    petsc = petsc_module if petsc_module is not None else _PETSc
    if petsc is None:
        raise RuntimeError("petsc4py_unavailable")

    n = int(np.asarray(rhs).size)
    comm = getattr(petsc, "COMM_SELF", None)

    h_mat = petsc.Mat().createPython(
        (n, n),
        context=_PETScMatrixFreeHessianContext(h_op),
        comm=comm,
    )
    if hasattr(h_mat, "setUp"):
        h_mat.setUp()

    ksp = petsc.KSP().create(comm=comm)
    ksp.setOperators(h_mat)
    ksp.setType("cg")
    pc = ksp.getPC()
    if m_op is None:
        if hasattr(pc, "setType"):
            pc.setType("none")
    else:
        pc.setType("python")
        pc.setPythonContext(_PETScMatrixFreePCContext(m_op))
    if hasattr(ksp, "setTolerances"):
        ksp.setTolerances(rtol=float(rtol), max_it=int(max(1, maxiter)))
    if hasattr(ksp, "setUp"):
        ksp.setUp()

    b = h_mat.createVecRight()
    b.getArray(readonly=False)[:] = np.asarray(rhs, dtype=np.float64)
    x = h_mat.createVecRight()
    try:
        ksp.solve(b, x)
        result = np.asarray(_petsc_vec_to_numpy(x), dtype=np.float64).reshape(-1).copy()
        iterations = (
            int(ksp.getIterationNumber()) if hasattr(ksp, "getIterationNumber") else 0
        )
        reason = (
            int(ksp.getConvergedReason()) if hasattr(ksp, "getConvergedReason") else 0
        )
        converged = bool(reason > 0)
        fallback = None if converged else f"petsc_ksp_reason_{reason}"
    finally:
        for obj in (b, x, ksp, h_mat):
            destroy = getattr(obj, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass

    return result, iterations, converged, fallback


@dataclass(slots=True)
class _JacobianActionBundle:
    shape: tuple[int, int]
    representation: str
    dense: np.ndarray | None
    matvec: Callable[[np.ndarray], np.ndarray]
    rmatvec: Callable[[np.ndarray], np.ndarray]
    linearization: "JacobianLinearization | None" = None


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


def _require_scalar_finite(
    name: str, value: float, iteration: int | None = None
) -> None:
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
    apply = getattr(matrix, "apply", None)
    if callable(apply):
        return np.asarray(apply(vec), dtype=np.float64)
    if isspmatrix(matrix):
        return np.asarray(matrix.dot(vec), dtype=np.float64)
    if isinstance(matrix, LinearOperator):
        return np.asarray(matrix.matvec(vec), dtype=np.float64)
    if callable(matrix):
        return np.asarray(matrix(vec), dtype=np.float64)
    dense = np.asarray(matrix, dtype=np.float64)
    return np.asarray(dense @ vec, dtype=np.float64)


def _as_sparse_regularization_matrix(matrix) -> sparse.spmatrix | None:
    if isspmatrix(matrix):
        return matrix
    as_rtr = getattr(matrix, "as_RtR", None)
    if not callable(as_rtr):
        as_rtr = getattr(matrix, "as_rtr", None)
    if not callable(as_rtr):
        return None
    explicit = as_rtr(dense=False)
    return explicit if isspmatrix(explicit) else None


def _diag_preconditioner(
    reconstructor, J_weighted_np: np.ndarray, lambda_eff: float
) -> np.ndarray:
    diag_h = np.sum(J_weighted_np * J_weighted_np, axis=0).astype(np.float64)
    reg_diag = getattr(reconstructor, "R_diag", None)
    if reg_diag is not None and reg_diag.shape[0] == diag_h.shape[0]:
        diag_h = diag_h + float(lambda_eff) * reg_diag
    else:
        diag_h = diag_h + float(lambda_eff)
    diag_h = np.maximum(diag_h, 1e-12)
    return np.asarray(diag_h, dtype=np.float64)


def _coerce_preconditioner_diag(values, n_param: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        diag = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if diag.shape[0] != int(n_param):
        return None
    return diag


def _matrix_free_pc_floor(reconstructor) -> float:
    for attr in ("matrix_free_pc_floor", "noser_floor"):
        raw = getattr(reconstructor, attr, None)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return max(value, 1e-12)
    return 1e-12


def _regularization_looks_like_noser(reconstructor) -> bool:
    regularization = getattr(reconstructor, "regularization", None)
    reg_name = (
        type(regularization).__name__.lower() if regularization is not None else ""
    )
    type_name = str(getattr(reconstructor, "regularization_type", "")).strip().lower()
    preset_name = str(getattr(reconstructor, "active_preset_name", "")).strip().lower()
    return "noser" in reg_name or type_name == "noser" or preset_name.endswith("noser")


def _sanitize_preconditioner_diag(
    diag: np.ndarray,
    *,
    n_param: int,
    floor: float,
    source: str,
) -> tuple[np.ndarray, str | None]:
    arr = np.asarray(diag, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_param):
        raise ValueError(
            f"Preconditioner diagonal length mismatch: expected {n_param}, got {arr.shape[0]}."
        )
    bad_mask = (~np.isfinite(arr)) | (arr <= float(floor))
    reason = f"{source}_diag_clamped" if bool(np.any(bad_mask)) else None
    arr = np.where(np.isfinite(arr), arr, float(floor))
    return np.maximum(arr, float(floor)).astype(np.float64), reason


def _operator_diag_preconditioner(
    reconstructor,
    n_param: int,
    lambda_eff: float,
    *,
    preferred: str = "diag",
    auto_hessian_diag_fn: Callable[[], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Matrix-free Hessian PC contract based on explicit diag/NOSER/prior data.

    ``auto_hessian_diag_fn`` is a last-resort callback that returns the full
    ``diag(J^T W J [+ alpha R_diag])``. It is used when no explicit diag attrs
    are set on the reconstructor so the PC is not silently forced to identity.
    """
    floor = _matrix_free_pc_floor(reconstructor)
    mode = str(preferred).strip().lower()
    source = "identity"
    reason: str | None = None

    hessian_diag = None
    for attr in ("matrix_free_hessian_diag", "hessian_diag"):
        hessian_diag = _coerce_preconditioner_diag(
            getattr(reconstructor, attr, None), n_param
        )
        if hessian_diag is not None:
            source = attr
            break
    if hessian_diag is not None:
        diag_h = np.asarray(hessian_diag, dtype=np.float64)
    else:
        explicit_diag = _coerce_preconditioner_diag(
            getattr(reconstructor, "matrix_free_preconditioner_diag", None),
            n_param,
        )
        noser_diag = _coerce_preconditioner_diag(
            getattr(reconstructor, "noser_diag", None), n_param
        )
        prior_diag = _coerce_preconditioner_diag(
            getattr(reconstructor, "prior_diag", None), n_param
        )
        reg_diag = _coerce_preconditioner_diag(
            getattr(reconstructor, "R_diag", None), n_param
        )

        base_diag = None
        if explicit_diag is not None:
            base_diag = explicit_diag
            source = "explicit"
        elif mode == "noser" and noser_diag is not None:
            base_diag = noser_diag
            source = "noser"
        elif mode == "prior" and prior_diag is not None:
            base_diag = prior_diag
            source = "prior"
        elif mode == "noser" and reg_diag is not None:
            base_diag = reg_diag
            source = (
                "noser" if _regularization_looks_like_noser(reconstructor) else "prior"
            )
        elif mode == "prior" and reg_diag is not None:
            base_diag = reg_diag
            source = "prior"
        elif noser_diag is not None:
            base_diag = noser_diag
            source = "noser"
        elif reg_diag is not None:
            base_diag = reg_diag
            source = (
                "noser" if _regularization_looks_like_noser(reconstructor) else "prior"
            )

        if base_diag is not None:
            diag_h = float(lambda_eff) * np.asarray(base_diag, dtype=np.float64)
        elif auto_hessian_diag_fn is not None:
            diag_h = None
            try:
                computed = auto_hessian_diag_fn()
            except Exception as exc:
                diag_h = None
                reason = f"auto_hessian_diag_failed:{type(exc).__name__}"
            else:
                arr = np.asarray(computed, dtype=np.float64).reshape(-1)
                if arr.shape[0] == int(n_param) and np.isfinite(arr).all():
                    diag_h = arr
                    source = "auto_linearization_diag"
                else:
                    reason = "auto_hessian_diag_invalid_output"
            if diag_h is None:
                diag_h = np.full(
                    int(n_param), max(float(lambda_eff), 1.0), dtype=np.float64
                )
        else:
            diag_h = np.full(
                int(n_param), max(float(lambda_eff), 1.0), dtype=np.float64
            )
            reason = "matrix_free_pc_missing_diag"

    diag_h, clamp_reason = _sanitize_preconditioner_diag(
        diag_h,
        n_param=n_param,
        floor=floor,
        source=source,
    )
    if clamp_reason is not None:
        reason = ";".join(part for part in (reason, clamp_reason) if part)

    meta = {
        "matrix_free_pc_source": source,
        "matrix_free_pc_mode": mode,
        "matrix_free_pc_floor": float(floor),
        "matrix_free_pc_min": float(np.min(diag_h)) if diag_h.size else float(floor),
        "matrix_free_pc_max": float(np.max(diag_h)) if diag_h.size else float(floor),
        "matrix_free_pc_reason": reason,
        "matrix_free_pmat_available": bool(
            getattr(reconstructor, "matrix_free_pmat", None) is not None
        ),
    }
    return np.asarray(diag_h, dtype=np.float64), meta


def _matrix_free_pmat_candidates(
    reconstructor, preferred: str
) -> list[tuple[str, object]]:
    mode = str(preferred).strip().lower()
    coarse_attrs = ("matrix_free_coarse_pmat", "coarse_hessian_pmat", "coarse_pmat")
    pmat_attrs = ("matrix_free_pmat", "pmat")
    attrs = coarse_attrs + pmat_attrs if mode == "coarse" else pmat_attrs + coarse_attrs
    candidates: list[tuple[str, object]] = []
    for attr in attrs:
        value = getattr(reconstructor, attr, None)
        if value is not None:
            candidates.append((attr, value))
    return candidates


def _build_matrix_free_custom_pc_operator(
    reconstructor,
    n_param: int,
) -> tuple[LinearOperator | None, dict[str, object], str | None]:
    for attr in (
        "matrix_free_pc_action",
        "matrix_free_pcshell_action",
        "custom_pc_action",
    ):
        action = getattr(reconstructor, attr, None)
        if not callable(action):
            continue

        def _apply(x, *, _action=action):
            return np.asarray(
                _action(np.asarray(x, dtype=np.float64)), dtype=np.float64
            )

        try:
            probe = _apply(np.ones(int(n_param), dtype=np.float64))
        except Exception as exc:
            return None, {}, f"{attr}_failed:{type(exc).__name__}"
        if probe.shape != (int(n_param),) or not np.isfinite(probe).all():
            return None, {}, f"{attr}_invalid_output"

        return (
            LinearOperator(
                (int(n_param), int(n_param)), matvec=_apply, dtype=np.float64
            ),
            {
                "matrix_free_pc_source": "custom-pcshell",
                "matrix_free_pmat_kind": "callable-inverse-action",
                "matrix_free_pmat_attr": attr,
                "matrix_free_pmat_available": True,
            },
            None,
        )
    return None, {}, "custom_pc_action_missing"


def _build_matrix_free_pmat_inverse_operator(
    pmat,
    *,
    n_param: int,
    attr: str,
    reconstructor,
) -> tuple[LinearOperator | None, dict[str, object], str | None]:
    n = int(n_param)
    max_n = int(getattr(reconstructor, "matrix_free_pmat_max_n", 20000))
    if max_n > 0 and n > max_n:
        return None, {}, "pmat_n_limit"

    source = "coarse-pmat" if "coarse" in attr else "pmat"
    shift = float(getattr(reconstructor, "matrix_free_pmat_shift", 1e-12))
    if not np.isfinite(shift) or shift < 0.0:
        shift = 1e-12

    try:
        if isinstance(pmat, LinearOperator):
            solve = getattr(pmat, "solve", None)
            if not callable(solve):
                return None, {}, "pmat_linear_operator_requires_solve_or_pc_action"

            def _apply_linear_operator(x):
                return np.asarray(
                    solve(np.asarray(x, dtype=np.float64)), dtype=np.float64
                )

            apply = _apply_linear_operator
            kind = "linear-operator-solve"
        elif isspmatrix(pmat):
            mat = pmat.tocsc().astype(np.float64)
            if mat.shape != (n, n):
                return None, {}, f"pmat_shape_mismatch:{mat.shape[0]}x{mat.shape[1]}"
            if mat.nnz == 0 or not np.isfinite(mat.data).all():
                return None, {}, "pmat_invalid_sparse_data"
            coo = mat.tocoo(copy=False)
            if coo.nnz <= n and np.all(coo.row == coo.col):
                diag = np.asarray(mat.diagonal(), dtype=np.float64)
                diag, _ = _sanitize_preconditioner_diag(
                    diag,
                    n_param=n,
                    floor=_matrix_free_pc_floor(reconstructor),
                    source=source,
                )

                def _apply_sparse_diag(x):
                    return np.asarray(x, dtype=np.float64) / diag

                apply = _apply_sparse_diag
                kind = "sparse-diagonal"
            else:
                factor = sparse.linalg.splu(
                    mat + sparse.identity(n, format="csc") * shift
                )

                def _apply_sparse_lu(x):
                    return np.asarray(
                        factor.solve(np.asarray(x, dtype=np.float64)), dtype=np.float64
                    )

                apply = _apply_sparse_lu
                kind = "sparse-lu"
        else:
            dense = np.asarray(pmat, dtype=np.float64)
            if dense.ndim == 1:
                diag, _ = _sanitize_preconditioner_diag(
                    dense,
                    n_param=n,
                    floor=_matrix_free_pc_floor(reconstructor),
                    source=source,
                )

                def _apply_dense_diag(x):
                    return np.asarray(x, dtype=np.float64) / diag

                apply = _apply_dense_diag
                kind = "dense-diagonal"
            else:
                if dense.ndim != 2:
                    return None, {}, f"pmat_shape_mismatch:{dense.ndim}d"
                if dense.shape != (n, n):
                    return (
                        None,
                        {},
                        f"pmat_shape_mismatch:{dense.shape[0]}x{dense.shape[1]}",
                    )
                if not np.isfinite(dense).all():
                    return None, {}, "pmat_invalid_dense_data"
                dense = dense + np.eye(n, dtype=np.float64) * shift
                if np.allclose(dense, dense.T, rtol=1e-8, atol=1e-12):
                    factor, lower = cho_factor(
                        dense, overwrite_a=False, check_finite=False
                    )

                    def _apply_dense_cholesky(x):
                        return np.asarray(
                            cho_solve(
                                (factor, lower),
                                np.asarray(x, dtype=np.float64),
                                check_finite=False,
                            ),
                            dtype=np.float64,
                        )

                    apply = _apply_dense_cholesky
                    kind = "dense-cholesky"
                else:

                    def _apply_dense_solve(x):
                        return np.asarray(
                            np.linalg.solve(dense, np.asarray(x, dtype=np.float64)),
                            dtype=np.float64,
                        )

                    apply = _apply_dense_solve
                    kind = "dense-solve"

        probe = apply(np.ones(n, dtype=np.float64))
    except Exception as exc:
        return None, {}, f"pmat_factor_failed:{type(exc).__name__}"

    if probe.shape != (n,) or not np.isfinite(probe).all():
        return None, {}, "pmat_invalid_inverse_action"
    meta = {
        "matrix_free_pc_source": source,
        "matrix_free_pmat_kind": kind,
        "matrix_free_pmat_attr": attr,
        "matrix_free_pmat_available": True,
    }
    return LinearOperator((n, n), matvec=apply, dtype=np.float64), meta, None


def _build_matrix_free_explicit_pc_operator(
    reconstructor,
    n_param: int,
    *,
    preferred: str,
) -> tuple[LinearOperator | None, dict[str, object], str | None]:
    mode = str(preferred).strip().lower()
    if mode == "custom":
        return _build_matrix_free_custom_pc_operator(reconstructor, n_param)

    for attr, pmat in _matrix_free_pmat_candidates(reconstructor, mode):
        op, meta, reason = _build_matrix_free_pmat_inverse_operator(
            pmat,
            n_param=n_param,
            attr=attr,
            reconstructor=reconstructor,
        )
        if op is not None:
            return op, meta, None
        if reason:
            return None, meta, reason

    if mode in {"pmat", "coarse", "petsc-gamg"}:
        return None, {}, "matrix_free_pmat_missing"
    return None, {}, None


def _as_jacobian_action_bundle(
    jacobian,
    *,
    measurement_weight_np: np.ndarray | None = None,
) -> _JacobianActionBundle:
    """Normalize dense and matrix-free Jacobian inputs into Jv/J^T r actions."""
    weight = None
    if measurement_weight_np is not None:
        weight = np.asarray(measurement_weight_np, dtype=np.float64).reshape(-1)

    def _apply_weight(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if weight is None:
            return arr
        if arr.size != weight.size:
            raise ValueError(
                f"Jacobian measurement dimension mismatch: expected {weight.size}, got {arr.size}."
            )
        return np.asarray(weight * arr, dtype=np.float64)

    def _apply_weight_to_residual(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if weight is None:
            return arr
        if arr.size != weight.size:
            raise ValueError(
                f"Residual dimension mismatch: expected {weight.size}, got {arr.size}."
            )
        return np.asarray(weight * arr, dtype=np.float64)

    if isinstance(jacobian, JacobianLinearization):
        n_meas, n_param = jacobian.shape
        if weight is not None and weight.size != n_meas:
            raise ValueError(
                f"Expected {n_meas} measurement weights, got {weight.size}."
            )

        def _matvec(v: np.ndarray) -> np.ndarray:
            return _apply_weight(jacobian.matvec(v))

        def _rmatvec(r: np.ndarray) -> np.ndarray:
            return np.asarray(
                jacobian.rmatvec(_apply_weight_to_residual(r)), dtype=np.float64
            )

        return _JacobianActionBundle(
            shape=(int(n_meas), int(n_param)),
            representation="jacobian_linearization",
            dense=None,
            matvec=_matvec,
            rmatvec=_rmatvec,
            linearization=jacobian,
        )

    if isinstance(jacobian, LinearOperator):
        n_meas, n_param = (int(jacobian.shape[0]), int(jacobian.shape[1]))
        if weight is not None and weight.size != n_meas:
            raise ValueError(
                f"Expected {n_meas} measurement weights, got {weight.size}."
            )

        def _matvec(v: np.ndarray) -> np.ndarray:
            return _apply_weight(jacobian.matvec(np.asarray(v, dtype=np.float64)))

        def _rmatvec(r: np.ndarray) -> np.ndarray:
            return np.asarray(
                jacobian.rmatvec(_apply_weight_to_residual(r)),
                dtype=np.float64,
            )

        return _JacobianActionBundle(
            shape=(n_meas, n_param),
            representation="linear_operator",
            dense=None,
            matvec=_matvec,
            rmatvec=_rmatvec,
        )

    if _is_jv_jtr_action(jacobian):
        n_meas, n_param = _jv_jtr_action_shape(jacobian)
        if weight is not None and weight.size != n_meas:
            raise ValueError(
                f"Expected {n_meas} measurement weights, got {weight.size}."
            )

        def _matvec(v: np.ndarray) -> np.ndarray:
            return _apply_weight(jacobian.Jv(np.asarray(v, dtype=np.float64)))

        def _rmatvec(r: np.ndarray) -> np.ndarray:
            return np.asarray(
                jacobian.JTr(_apply_weight_to_residual(r)),
                dtype=np.float64,
            )

        return _JacobianActionBundle(
            shape=(n_meas, n_param),
            representation=_jv_jtr_action_representation(jacobian),
            dense=None,
            matvec=_matvec,
            rmatvec=_rmatvec,
        )

    dense = np.asarray(jacobian, dtype=np.float64)
    if dense.ndim != 2:
        raise ValueError(
            "Jacobian input must be a 2D array, LinearOperator, "
            "JacobianLinearization, or Jv/JTr action object."
        )
    if weight is not None:
        if weight.size != dense.shape[0]:
            raise ValueError(
                f"Expected {dense.shape[0]} measurement weights, got {weight.size}."
            )
        dense = dense * weight[:, None]
    dense = np.asarray(dense, dtype=np.float64)

    def _matvec(v: np.ndarray) -> np.ndarray:
        return np.asarray(
            safe_dot(dense, np.asarray(v, dtype=np.float64), "gauss_newton.fast.jv"),
            dtype=np.float64,
        )

    def _rmatvec(r: np.ndarray) -> np.ndarray:
        return np.asarray(
            safe_dot(dense.T, np.asarray(r, dtype=np.float64), "gauss_newton.fast.jtr"),
            dtype=np.float64,
        )

    return _JacobianActionBundle(
        shape=(int(dense.shape[0]), int(dense.shape[1])),
        representation="dense",
        dense=dense,
        matvec=_matvec,
        rmatvec=_rmatvec,
    )


def _solve_linear_system_fast(
    reconstructor,
    *,
    J_weighted_np,
    weighted_residual_np: np.ndarray,
    de_current_np: np.ndarray,
    lambda_eff: float,
    iteration: int,
    measurement_weight_np: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Matrix-free fast solve with Woodbury/PCG fallback chain.

    Solve ``(J'J + lambda*R) delta = -g`` without constructing a dense
    ``n x n`` Hessian in automatic fast mode.
    """
    jacobian_actions = _as_jacobian_action_bundle(
        J_weighted_np,
        measurement_weight_np=measurement_weight_np,
    )
    J_weighted_dense_np = jacobian_actions.dense
    weighted_residual_np = np.asarray(weighted_residual_np, dtype=np.float64).reshape(
        -1
    )
    de_current_np = np.asarray(de_current_np, dtype=np.float64).reshape(-1)

    n_param = int(jacobian_actions.shape[1])
    n_meas = int(jacobian_actions.shape[0])
    if weighted_residual_np.shape[0] != n_meas:
        raise ValueError(
            f"Weighted residual length mismatch: expected {n_meas}, got {weighted_residual_np.shape[0]}."
        )
    if de_current_np.shape[0] != n_param:
        raise ValueError(
            f"Current conductivity vector length mismatch: expected {n_param}, got {de_current_np.shape[0]}."
        )

    jtr = np.asarray(jacobian_actions.rmatvec(weighted_residual_np), dtype=np.float64)
    rhs = -jtr
    if reconstructor.use_prior_term:
        rhs = rhs - float(lambda_eff) * _apply_regularization_np(
            reconstructor, de_current_np
        )
    _require_finite("rhs_fast", rhs, iteration)

    def _matvec(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64)
        projected = np.asarray(jacobian_actions.matvec(vv), dtype=np.float64)
        back_projected = np.asarray(
            jacobian_actions.rmatvec(projected), dtype=np.float64
        )
        return np.asarray(
            back_projected
            + float(lambda_eff) * _apply_regularization_np(reconstructor, vv),
            dtype=np.float64,
        )

    h_op = LinearOperator(
        (n_param, n_param),
        matvec=_matvec,
        rmatvec=_matvec,
        dtype=np.float64,
    )

    solver_mode = str(getattr(reconstructor, "linear_solver", "auto")).strip().lower()
    preconditioner_mode = (
        str(getattr(reconstructor, "preconditioner", "auto")).strip().lower()
    )
    capabilities = detect_performance_capabilities()
    resolved_preconditioner = select_preconditioner(
        preconditioner_mode,
        capabilities=capabilities,
    )
    fast_linear_mode = (
        str(getattr(reconstructor, "fast_linear_path", "auto")).strip().lower()
    )

    if J_weighted_dense_np is not None:
        diag_precond = _diag_preconditioner(
            reconstructor, J_weighted_dense_np, lambda_eff
        )
        pc_diag_meta = {
            "matrix_free_pc_source": "dense-sensitivity",
            "matrix_free_pc_mode": resolved_preconditioner,
            "matrix_free_pc_floor": 1e-12,
            "matrix_free_pc_min": (
                float(np.min(diag_precond)) if diag_precond.size else 1e-12
            ),
            "matrix_free_pc_max": (
                float(np.max(diag_precond)) if diag_precond.size else 1e-12
            ),
            "matrix_free_pc_reason": None,
            "matrix_free_pmat_available": True,
        }
    else:
        auto_hessian_diag_fn: Callable[[], np.ndarray] | None = None
        linearization = jacobian_actions.linearization
        if linearization is not None:
            weights_sqrt = (
                np.asarray(measurement_weight_np, dtype=np.float64).reshape(-1)
                if measurement_weight_np is not None
                else None
            )
            weights_for_diag = (
                weights_sqrt * weights_sqrt if weights_sqrt is not None else None
            )

            def auto_hessian_diag_fn() -> np.ndarray:
                return linearization.hessian_diag(
                    measurement_weights=weights_for_diag,
                )

        diag_precond, pc_diag_meta = _operator_diag_preconditioner(
            reconstructor,
            n_param,
            lambda_eff,
            preferred=resolved_preconditioner,
            auto_hessian_diag_fn=auto_hessian_diag_fn,
        )
    diag_inv_op = LinearOperator(
        (n_param, n_param),
        matvec=lambda x: np.asarray(x, dtype=np.float64) / diag_precond,
        dtype=np.float64,
    )
    explicit_pc_op: LinearOperator | None = None
    explicit_pc_meta: dict[str, object] = {}
    explicit_pc_reason: str | None = None
    if J_weighted_dense_np is None:
        explicit_pc_op, explicit_pc_meta, explicit_pc_reason = (
            _build_matrix_free_explicit_pc_operator(
                reconstructor,
                n_param,
                preferred=resolved_preconditioner,
            )
        )
    pc_meta = dict(pc_diag_meta)
    if explicit_pc_meta:
        pc_meta.update(explicit_pc_meta)

    cg_rtol = 1e-8 if reconstructor.performance_mode == "aggressive" else 1e-10
    cg_maxiter = max(200, min(2500, n_param))

    fallback_reasons: list[str] = []
    fast_linear_path_reason = "explicit"
    linear_iterations: int | None = None

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
            "jacobian_representation": jacobian_actions.representation,
            "jacobian_shape": [int(n_meas), int(n_param)],
            "dense_jacobian_materialized": J_weighted_dense_np is not None,
        }
        meta_payload.update(pc_meta)
        if isinstance(extra, dict):
            meta_payload.update(extra)
        if linear_iterations is not None:
            meta_payload["linear_iterations"] = int(linear_iterations)
        reconstructor._last_fast_linear_meta = meta_payload

    def _regularization_signature() -> str:
        reg = getattr(reconstructor, "R_matrix", None)
        if reg is None:
            return "none"
        signature_hash = getattr(reg, "signature_hash", None)
        if signature_hash:
            return str(signature_hash)
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
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        if isinstance(reg, LinearOperator):
            return f"linear_operator:{reg.shape[0]}x{reg.shape[1]}"
        if callable(reg):
            return f"callable:{type(reg).__module__}.{type(reg).__qualname__}:{n_param}"
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
        reg_sparse = _as_sparse_regularization_matrix(reg) if reg is not None else None
        if reg is None:
            pass
        elif reg_sparse is not None:
            reg_csr = reg_sparse.tocsr()
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
            meta.update(
                {"is_diagonal": False, "is_sparse_spd": False, "diag_vector": None}
            )
        elif callable(reg):
            meta.update(
                {"is_diagonal": False, "is_sparse_spd": False, "diag_vector": None}
            )
        else:
            as_rtr = getattr(reg, "as_RtR", None)
            explicit = as_rtr(dense=False) if callable(as_rtr) else reg
            if isinstance(explicit, LinearOperator):
                diag_fn = getattr(reg, "diag", None)
                raw_diag = diag_fn() if callable(diag_fn) else None
                diag_vec = (
                    None
                    if raw_diag is None
                    else np.asarray(raw_diag, dtype=np.float64).reshape(-1)
                )
                meta.update(
                    {
                        "is_diagonal": False,
                        "is_sparse_spd": False,
                        "diag_vector": diag_vec
                        if diag_vec is not None and diag_vec.size == n_param
                        else None,
                    }
                )
                reconstructor._regularization_meta_cache = {
                    "token": cache_token,
                    "meta": meta,
                }
                return dict(meta)
            dense = np.asarray(explicit, dtype=np.float64)
            diag_vec = np.asarray(np.diag(dense), dtype=np.float64)
            is_diag = bool(dense.ndim == 2 and dense.shape == (n_param, n_param))
            if is_diag:
                off_diag = dense - np.diag(diag_vec)
                is_diag = bool(np.all(np.abs(off_diag) <= 1e-12))
            symmetric = bool(
                dense.ndim == 2 and np.allclose(dense, dense.T, rtol=1e-8, atol=1e-12)
            )
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
        if J_weighted_dense_np is None:
            raise RuntimeError("woodbury_requires_dense_jacobian")
        diag_scaled = float(lambda_eff) * np.asarray(diag_vector, dtype=np.float64)
        diag_scaled = np.maximum(diag_scaled, 1e-12)
        inv_diag = 1.0 / diag_scaled

        u = inv_diag * rhs
        ja_inv = J_weighted_dense_np * inv_diag[None, :]
        small_rhs = np.asarray(
            safe_dot(J_weighted_dense_np, u, "gauss_newton.fast.woodbury.small_rhs"),
            dtype=np.float64,
        )
        s_matrix = np.eye(n_meas, dtype=np.float64) + np.asarray(
            safe_dot(
                ja_inv, J_weighted_dense_np.T, "gauss_newton.fast.woodbury.small_system"
            ),
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
            safe_dot(J_weighted_dense_np.T, y, "gauss_newton.fast.woodbury.correction"),
            dtype=np.float64,
        )
        return np.asarray(u - correction, dtype=np.float64)

    def _build_cholmod_preconditioner_from_R() -> tuple[
        LinearOperator | None, str | None
    ]:
        if cholmod_cholesky is None:
            return None, "cholmod_unavailable"

        reg = getattr(reconstructor, "R_matrix", None)
        reg_sparse = _as_sparse_regularization_matrix(reg)
        if reg_sparse is None:
            return None, "regularization_not_sparse"

        max_n = int(getattr(reconstructor, "cholmod_max_n", 50000))
        if max_n > 0 and n_param > max_n:
            return None, "cholmod_n_limit"

        reg_csc = reg_sparse.tocsc().astype(np.float64)
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
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
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
            matvec=lambda x: np.asarray(
                factor.solve_A(np.asarray(x, dtype=np.float64)), dtype=np.float64
            ),
            dtype=np.float64,
        )
        return operator, None

    def _solve_pcg(
        precond_choice: str,
    ) -> tuple[np.ndarray | None, str, str, str | None]:
        nonlocal linear_iterations, pc_meta
        choice = str(precond_choice)
        m_op: LinearOperator | None = None
        path = "pcg-diag-precond"
        fallback: str | None = None

        explicit_modes = {"pmat", "coarse", "custom"}
        use_explicit_pc = explicit_pc_op is not None and (
            choice in explicit_modes
            or preconditioner_mode == "auto"
            or choice == "petsc-gamg"
        )
        if use_explicit_pc:
            m_op = explicit_pc_op
            pc_meta.update(explicit_pc_meta)
            source = str(explicit_pc_meta.get("matrix_free_pc_source", "pmat"))
            if source == "custom-pcshell":
                choice = "custom"
                path = "pcg-custom-pcshell-precond"
            elif source == "coarse-pmat":
                choice = "coarse"
                path = "pcg-coarse-pmat-precond"
            else:
                choice = "pmat"
                path = "pcg-pmat-precond"
            pc_meta["matrix_free_pmat_requested_preconditioner"] = precond_choice
        elif choice in explicit_modes:
            fallback = explicit_pc_reason or f"{choice}_preconditioner_unavailable"
            choice = "diag"

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
                reg_sparse = _as_sparse_regularization_matrix(reg)
                if reg_sparse is not None:
                    try:
                        amg_mat = reg_sparse.tocsr().astype(np.float64)
                        amg_mat = (
                            amg_mat
                            + sparse.identity(amg_mat.shape[0], format="csr") * 1e-12
                        )
                        ml = pyamg.smoothed_aggregation_solver(amg_mat)
                        m_op = ml.aspreconditioner(cycle="V")
                        path = "pcg-pyamg-precond"
                    except Exception as exc:  # pragma: no cover - optional path
                        fallback = f"pyamg_precond_failed:{type(exc).__name__}"
                        choice = "diag"
                else:
                    fallback = "pyamg_requires_sparse_regularization"
                    choice = "diag"

        if choice in {"diag", "noser", "prior", "petsc-gamg"}:
            m_op = diag_inv_op
            path = (
                f"pcg-{choice}-precond"
                if choice in {"noser", "prior"}
                else "pcg-diag-precond"
            )
            if choice == "petsc-gamg":
                choice = "diag"
                fallback = "petsc_gamg_not_supported_in_matrix_free"
                if (
                    explicit_pc_reason
                    and explicit_pc_reason != "matrix_free_pmat_missing"
                ):
                    fallback = f"{fallback};{explicit_pc_reason}"

        iteration_counter = {"count": 0}

        def _count_iteration(_xk) -> None:
            iteration_counter["count"] += 1

        backend_requested = (
            str(getattr(reconstructor, "matrix_free_ksp_backend", "scipy"))
            .strip()
            .lower()
            or "scipy"
        )
        if backend_requested == "auto":
            backend_effective = "petsc" if _PETSc is not None else "scipy"
        elif backend_requested == "petsc":
            backend_effective = "petsc" if _PETSc is not None else "scipy"
        else:
            backend_effective = "scipy"
        pc_meta["matrix_free_ksp_backend_requested"] = backend_requested
        pc_meta["matrix_free_ksp_backend_effective"] = backend_effective

        if backend_requested == "petsc" and _PETSc is None:
            backend_reason = "petsc_backend_unavailable"
            pc_meta["matrix_free_ksp_backend_fallback_reason"] = backend_reason
            if fallback is None:
                fallback = backend_reason
            else:
                fallback = f"{fallback};{backend_reason}"

        if backend_effective == "petsc":
            try:
                delta_arr, petsc_iters, petsc_converged, petsc_fallback = (
                    _solve_matrix_free_hessian_via_petsc(
                        h_op,
                        rhs,
                        m_op,
                        rtol=cg_rtol,
                        maxiter=cg_maxiter,
                    )
                )
            except Exception as exc:
                pc_meta["matrix_free_ksp_backend_effective"] = "scipy"
                pc_meta["matrix_free_ksp_backend_fallback_reason"] = (
                    f"petsc_backend_failed:{type(exc).__name__}"
                )
            else:
                linear_iterations = int(petsc_iters)
                if not petsc_converged:
                    reason = petsc_fallback or "pcg_not_converged"
                    return None, path, choice, reason
                return np.asarray(delta_arr, dtype=np.float64), path, choice, fallback

        delta, info = cg(
            h_op,
            rhs,
            M=m_op,
            rtol=cg_rtol,
            maxiter=cg_maxiter,
            callback=_count_iteration,
        )
        linear_iterations = int(iteration_counter["count"])
        if info != 0:
            return None, path, choice, "pcg_not_converged"
        return np.asarray(delta, dtype=np.float64), path, choice, fallback

    def _solve_cholmod_direct_debug() -> tuple[np.ndarray | None, str | None]:
        if J_weighted_dense_np is None:
            return None, "cholmod_direct_requires_dense_jacobian"
        if cholmod_cholesky is None:
            return None, "cholmod_unavailable"
        reg = getattr(reconstructor, "R_matrix", None)
        reg_sparse = _as_sparse_regularization_matrix(reg)
        if reg_sparse is None:
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
                safe_dot(
                    J_weighted_dense_np.T,
                    J_weighted_dense_np,
                    "gauss_newton.fast.cholmod_direct.jtj",
                )
            )
            h_sparse = h_sparse + float(lambda_eff) * reg_sparse.tocsc()
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

    rom_mode_effective = _effective_feature_mode(
        str(getattr(reconstructor, "rom_mode", "auto"))
    )
    inexact_mode_effective = _effective_feature_mode(
        str(getattr(reconstructor, "inexact_mode", "auto"))
    )
    lowrank_mode_effective = _effective_feature_mode(
        str(getattr(reconstructor, "lowrank_mode", "auto"))
    )

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
        if J_weighted_dense_np is None:
            return np.ascontiguousarray(
                np.column_stack([rhs, de_current_np]),
                dtype=np.float64,
            )
        snapshots: list[np.ndarray] = []
        snapshots.append(np.asarray(rhs, dtype=np.float64))
        snapshots.append(np.asarray(de_current_np, dtype=np.float64))
        grad_proxy = np.asarray(
            safe_dot(
                J_weighted_dense_np.T,
                weighted_residual_np,
                "gauss_newton.fused.grad_proxy",
            ),
            dtype=np.float64,
        )
        snapshots.append(grad_proxy)
        if isinstance(diag_vector, np.ndarray) and diag_vector.shape[0] == n_param:
            inv_diag = 1.0 / np.maximum(float(lambda_eff) * diag_vector, 1e-12)
            snapshots.append(inv_diag * rhs)
        for idx in range(min(6, n_meas)):
            snapshots.append(np.asarray(J_weighted_dense_np[idx, :], dtype=np.float64))
        return np.ascontiguousarray(np.column_stack(snapshots), dtype=np.float64)

    def _build_global_basis(diag_vector: np.ndarray | None) -> tuple[np.ndarray, str]:
        source = (
            str(getattr(reconstructor, "rom_snapshot_source", "hybrid")).strip().lower()
        )
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
                "model_signature": model_signature_from_forward_model(
                    reconstructor.fwd_model
                ),
                "pattern_signature": pattern_signature_from_forward_model(
                    reconstructor.fwd_model
                ),
                "backend_signature": backend_signature_from_forward_model(
                    reconstructor.fwd_model
                ),
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
            "model_signature": model_signature_from_forward_model(
                reconstructor.fwd_model
            ),
            "pattern_signature": pattern_signature_from_forward_model(
                reconstructor.fwd_model
            ),
            "backend_signature": backend_signature_from_forward_model(
                reconstructor.fwd_model
            ),
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
        lowrank_method = (
            str(getattr(reconstructor, "lowrank_method", "tsvd")).strip().lower()
        )
        lowrank_energy = float(getattr(reconstructor, "lowrank_energy", 0.995))
        if rank_adaptive <= 0:
            return np.zeros((n_param, 0), dtype=np.float64), "disabled"

        if (iteration % refresh_every) != 0:
            cached_basis = getattr(reconstructor, "_rom_last_adaptive_basis", None)
            if (
                isinstance(cached_basis, np.ndarray)
                and cached_basis.shape[0] == n_param
            ):
                return np.asarray(cached_basis, dtype=np.float64), "reuse"

        payload = {
            "solver": "gn_absolute",
            "artifact": "rom_adaptive_basis",
            "jacobian_hash": hashlib.sha256(
                np.ascontiguousarray(J_weighted_dense_np, dtype=np.float64).tobytes()
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
                    J_weighted_dense_np,
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
            J_weighted_dense_np,
            rank=max(lowrank_rank, rank_adaptive),
            energy=lowrank_energy,
            method=lowrank_method,
        )[0][:, :rank_adaptive]
        basis_arr = np.asarray(basis_arr, dtype=np.float64)
        reconstructor._rom_last_adaptive_basis = basis_arr
        return basis_arr, "compute"

    def _solve_linear_system_fused(
        diag_vector: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        if J_weighted_dense_np is None:
            return None, {"reason": "fused_requires_dense_jacobian"}
        if rom_mode_effective == "off":
            return None, {"reason": "rom_off"}
        rom_mode_raw = str(getattr(reconstructor, "rom_mode", "auto")).strip().lower()
        if n_param < 5000 and rom_mode_raw != "on":
            return None, {"reason": "problem_too_small"}
        residual_limit = float(
            max(0.05, getattr(reconstructor, "rom_residual_limit", 0.6))
        )

        global_basis, global_source = _build_global_basis(diag_vector)
        adaptive_basis = np.zeros((n_param, 0), dtype=np.float64)
        adaptive_source = "disabled"
        if lowrank_mode_effective != "off" and bool(
            fused_strategy.get("lowrank", False)
        ):
            adaptive_basis, adaptive_source = _adaptive_basis()

        combined_basis = merge_orthonormal_bases(
            global_basis,
            adaptive_basis,
            rank_cap=int(
                max(
                    1,
                    int(getattr(reconstructor, "rom_rank_global", 32))
                    + int(max(0, getattr(reconstructor, "rom_rank_adaptive", 16))),
                )
            ),
        )
        if combined_basis.size == 0 or combined_basis.shape[1] == 0:
            return None, {"reason": "empty_basis"}

        stage_attempts = []
        stage_attempts.append(
            {
                "name": "rom+inexact+lowrank",
                "use_lowrank": adaptive_basis.shape[1] > 0,
                "use_inexact": inexact_mode_effective != "off"
                and bool(fused_strategy.get("inexact", False)),
            }
        )
        stage_attempts.append(
            {
                "name": "rom+inexact",
                "use_lowrank": False,
                "use_inexact": inexact_mode_effective != "off"
                and bool(fused_strategy.get("inexact", False)),
            }
        )
        stage_attempts.append(
            {"name": "rom", "use_lowrank": False, "use_inexact": False}
        )

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
                        np.ascontiguousarray(
                            J_weighted_dense_np, dtype=np.float64
                        ).tobytes()
                    ).hexdigest(),
                    "lambda_eff": float(lambda_eff),
                    "reg_signature": _regularization_signature(),
                    "stage": stage["name"],
                }
                cache_manager = getattr(reconstructor, "cache_manager", None)
                if cache_manager is not None and bool(
                    getattr(cache_manager, "enabled", False)
                ):
                    reduced_op, op_lookup = cache_manager.get_or_compute_semantic(
                        artifact="rom_reduced_operator_absolute",
                        name="absolute_rom_reduced_operator",
                        namespace="absolute",
                        cache_obj=op_payload,
                        payload=op_payload,
                        compute_fn=lambda: build_reduced_operator(
                            jacobian=J_weighted_dense_np,
                            basis=stage_basis,
                            regularization_apply=lambda vec: _apply_regularization_np(
                                reconstructor, vec
                            ),
                            lambda_eff=float(lambda_eff),
                        ),
                        persist=True,
                        cost=6.0,
                        effort_seconds=2.0,
                    )
                    op_source = str(getattr(op_lookup, "layer", "compute"))
                else:
                    reduced_op = build_reduced_operator(
                        jacobian=J_weighted_dense_np,
                        basis=stage_basis,
                        regularization_apply=lambda vec: _apply_regularization_np(
                            reconstructor, vec
                        ),
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
                full_residual = np.asarray(
                    _matvec(delta_candidate) - rhs, dtype=np.float64
                )
                full_residual_ratio = float(
                    np.linalg.norm(full_residual) / max(np.linalg.norm(rhs), 1e-12)
                )
                if full_residual_ratio > residual_limit:
                    raise RuntimeError("fused_residual_high")

                linear_residual_ratio = float(
                    solve_info.get("linear_residual_ratio", 0.0)
                )
                outer_prev_raw = getattr(reconstructor, "_outer_prev_residual", None)
                outer_prev = (
                    float(outer_prev_raw)
                    if isinstance(outer_prev_raw, (int, float))
                    else None
                )
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
        elif (
            selected_fast_path == "pcg"
            and regularization_is_sparse_spd
            and capabilities.get("cholmod", False)
        ):
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
                fast_linear_path_reason = str(
                    fused_strategy.get("reason", "fused_enabled")
                )
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
                        "lowrank_rank_effective": int(
                            fused_meta.get("adaptive_rank", 0)
                        ),
                        "inexact_eta": float(fused_meta.get("inexact_eta", 0.0)),
                        "inexact_eta_history": fused_meta.get(
                            "inexact_eta_history", []
                        ),
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
            lsmr_result = lsmr(
                h_op, rhs, atol=cg_rtol, btol=cg_rtol, maxiter=cg_maxiter
            )
            delta_np = np.asarray(lsmr_result[0], dtype=np.float64)
            if len(lsmr_result) > 2:
                linear_iterations = int(lsmr_result[2])
            fast_solver_path = "lsmr-direct"
    else:
        if delta_np is None and selected_fast_path == "woodbury":
            diag_vector = reg_meta.get("diag_vector")
            if J_weighted_dense_np is None:
                _add_fallback("woodbury_requires_dense_jacobian")
            elif (
                isinstance(diag_vector, np.ndarray) and diag_vector.shape[0] == n_param
            ):
                try:
                    delta_np = _solve_linear_system_fast_woodbury_diag(diag_vector)
                    linear_iterations = 0
                    fast_solver_path = "woodbury-diag"
                    resolved_preconditioner = "woodbury"
                except Exception as exc:
                    _add_fallback(f"woodbury_failed:{type(exc).__name__}")
            else:
                _add_fallback("woodbury_requires_diagonal_regularization")

        if delta_np is None and selected_fast_path == "cholmod-direct":
            delta_np, reason = _solve_cholmod_direct_debug()
            if delta_np is not None:
                linear_iterations = 0
                fast_solver_path = "cholmod-direct"
                resolved_preconditioner = "cholmod"
            else:
                _add_fallback(reason)

        if delta_np is None:
            pcg_delta, pcg_path, pcg_choice, pcg_reason = _solve_pcg(
                resolved_preconditioner
            )
            resolved_preconditioner = pcg_choice
            if pcg_reason:
                _add_fallback(pcg_reason)
            if pcg_delta is not None:
                delta_np = pcg_delta
                fast_solver_path = pcg_path

        if delta_np is None:
            try:
                lsmr_result = lsmr(
                    h_op, rhs, atol=1e-7, btol=1e-7, maxiter=cg_maxiter * 2
                )
                delta_np = np.asarray(lsmr_result[0], dtype=np.float64)
                if len(lsmr_result) > 2:
                    linear_iterations = int(lsmr_result[2])
                fast_solver_path = "lsmr-fallback"
            except Exception as exc:
                _add_fallback(f"lsmr_failed:{type(exc).__name__}")
                _set_fast_meta(
                    path="strict-fallback",
                    resolved_precond=resolved_preconditioner,
                    reason=(
                        ";".join(fallback_reasons)
                        if fallback_reasons
                        else "fast_linear_failed"
                    ),
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


def _is_jv_jtr_action(jacobian) -> bool:
    if isinstance(jacobian, (JacobianLinearization, LinearOperator)):
        return False
    return callable(getattr(jacobian, "Jv", None)) and callable(
        getattr(jacobian, "JTr", None)
    )


def _jv_jtr_action_shape(jacobian) -> tuple[int, int]:
    shape = getattr(jacobian, "shape", None)
    if shape is None:
        n_meas = getattr(jacobian, "n_measurements", None)
        n_param = getattr(jacobian, "n_coarse_cells", None)
        shape = (n_meas, n_param)
    if len(shape) != 2:
        raise ValueError("Jv/JTr action shape must have two dimensions.")
    n_meas, n_param = int(shape[0]), int(shape[1])
    if n_meas <= 0 or n_param <= 0:
        raise ValueError("Jv/JTr action shape must be positive.")
    return n_meas, n_param


def _jv_jtr_action_representation(jacobian) -> str:
    if isinstance(jacobian, DualMeshJacobianOperator):
        return "dual_mesh_jacobian_operator"
    return "jv_jtr_action"
